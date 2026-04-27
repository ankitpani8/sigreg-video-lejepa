from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_FPS_WARN_LOW = 20.0
_FPS_WARN_HIGH = 40.0
_FPS_SCAN_SAMPLE = 20


class BaseVideoDataset(Dataset, ABC):
    """Abstract base for video datasets using decord for I/O.

    Subclasses call super().__init__() after building self.samples.
    Handles: decord VideoReader lifecycle, frame sampling, short-video looping,
    spatial transform application, and an FPS sanity scan at init.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        num_frames: int,
        frame_stride: int,
        transform: Callable,
        split: str,
    ) -> None:
        """
        Args:
            samples:      List of (video_path, label) pairs.
            num_frames:   Number of frames to sample per clip.
            frame_stride: Temporal stride between sampled frames.
            transform:    Callable (T,H,W,C) uint8 → (C,T,H,W) float32.
            split:        'train' | 'val' | 'test'
        """
        self.samples = samples
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform
        self.split = split

        self._scan_fps()

    @abstractmethod
    def _load_samples(self) -> list[tuple[Path, int]]:
        """Build and return the (video_path, label) list. Called by subclass."""
        ...

    # ------------------------------------------------------------------
    # FPS scan
    # ------------------------------------------------------------------

    def _scan_fps(self) -> None:
        """Open a sample of videos and warn on unusual FPS values."""
        n = min(_FPS_SCAN_SAMPLE, len(self.samples))
        indices = random.sample(range(len(self.samples)), n)
        for idx in indices:
            path, _ = self.samples[idx]
            try:
                vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
                fps = vr.get_avg_fps()
                del vr
                if fps < _FPS_WARN_LOW or fps > _FPS_WARN_HIGH:
                    logger.warning(
                        "Unusual FPS %.1f for %s; expected 24-30 fps. "
                        "Effective sample rate may differ from configured stride.",
                        fps,
                        path.name,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read FPS for %s: %s", path.name, exc)

    # ------------------------------------------------------------------
    # Frame index sampling
    # ------------------------------------------------------------------

    def _sample_frame_indices_train(self, total_frames: int) -> list[int]:
        """Random start, stride=frame_stride, looping for short videos.

        If total_frames < num_frames * frame_stride, indices wrap via modulo so
        short clips are repeated rather than padded with black frames.
        """
        start = random.randint(0, max(0, total_frames - 1))
        return [(start + t * self.frame_stride) % total_frames for t in range(self.num_frames)]

    def _sample_frame_indices_eval(self, total_frames: int) -> list[list[int]]:
        """4 evenly-spaced clips for multi-clip evaluation. Phase 3 implements this."""
        raise NotImplementedError("multi-clip eval: Phase 3")

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        # VideoReader must be created inside __getitem__ — fork-based multiprocessing
        # (Linux default) corrupts handles created in the parent process.
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        total_frames = len(vr)

        if self.split == "train":
            frame_indices = self._sample_frame_indices_train(total_frames)
        else:
            frame_indices = self._sample_frame_indices_eval(total_frames)[0]

        frames: np.ndarray = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C) uint8
        del vr

        clip: torch.Tensor = self.transform(frames)  # (C, T, H, W) float32
        return clip, label

    def __len__(self) -> int:
        return len(self.samples)
