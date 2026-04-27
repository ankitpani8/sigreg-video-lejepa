from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from sigreg_video_lejepa.data.base import BaseVideoDataset
from sigreg_video_lejepa.data.cache import ensure_cached

logger = logging.getLogger(__name__)

_REQUIRED_SPLIT_FILES = ("classInd.txt", "trainlist01.txt", "testlist01.txt")
_MIN_CLASS_DIRS = 100


class UCF101Dataset(BaseVideoDataset):
    """UCF101 video dataset.

    Parses trainlist01.txt / testlist01.txt split files, builds a (path, label)
    sample list, and delegates frame decoding and sampling to BaseVideoDataset.

    Directory layout expected at data_root::

        data_root/
          ApplyEyeMakeup/
            v_ApplyEyeMakeup_g01_c01.avi
            ...
          Archery/
            ...

    Split files expected at split_root::

        split_root/
          classInd.txt          (label_1indexed ClassName)
          trainlist01.txt       (ClassName/video.avi label_1indexed)
          testlist01.txt        (ClassName/video.avi)       ← no label column

    Args:
        data_root:    Path to UCF-101/ directory (101 class subdirs).
        split_root:   Path to ucfTrainTestlist/ directory.
        split:        'train' or 'test'.
        num_frames:   Frames to sample per clip.
        frame_stride: Temporal stride between sampled frames.
        transform:    Callable (T,H,W,C) uint8 → (C,T,H,W) float32.
        local_cache:  If non-null, copy data_root here on first use for faster
                      SSD reads. Null = read directly from data_root (e.g. Drive).
    """

    def __init__(
        self,
        data_root: str | Path,
        split_root: str | Path,
        split: str,
        num_frames: int,
        frame_stride: int,
        transform: Callable,
        local_cache: str | Path | None = None,
    ) -> None:
        data_root = Path(data_root)
        split_root = Path(split_root)

        self._validate_paths(data_root, split_root)

        # Cache copy happens in __init__ (main process) — before DataLoader forks.
        if local_cache is not None:
            data_root = ensure_cached(data_root, Path(local_cache))

        class_to_idx = self._load_class_index(split_root / "classInd.txt")
        samples = self._load_samples_from_split(data_root, split_root, split, class_to_idx)

        super().__init__(
            samples=samples,
            num_frames=num_frames,
            frame_stride=frame_stride,
            transform=transform,
            split=split,
        )

        logger.info(
            "UCF101Dataset: split=%s, %d clips, %d classes.",
            split,
            len(samples),
            len(class_to_idx),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_paths(data_root: Path, split_root: Path) -> None:
        if not data_root.exists():
            raise FileNotFoundError(
                f"UCF101 data_root not found: {data_root}\n"
                "Expected structure: data_root/ClassName/v_ClassName_gXX_cXX.avi\n"
                "See README.md → Setup → Dataset for download instructions."
            )
        class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
        if len(class_dirs) < _MIN_CLASS_DIRS:
            raise FileNotFoundError(
                f"UCF101 data_root has only {len(class_dirs)} subdirectories "
                f"(expected ≥ {_MIN_CLASS_DIRS}): {data_root}\n"
                "See README.md → Setup → Dataset for download instructions."
            )

        if not split_root.exists():
            raise FileNotFoundError(
                f"UCF101 split_root not found: {split_root}\n"
                "Expected files: classInd.txt, trainlist01.txt, testlist01.txt\n"
                "See README.md → Setup → Dataset for download instructions."
            )
        for fname in _REQUIRED_SPLIT_FILES:
            if not (split_root / fname).exists():
                raise FileNotFoundError(
                    f"Required split file missing: {split_root / fname}\n"
                    "See README.md → Setup → Dataset for download instructions."
                )

    # ------------------------------------------------------------------
    # Class index
    # ------------------------------------------------------------------

    @staticmethod
    def _load_class_index(classind_path: Path) -> dict[str, int]:
        """Parse classInd.txt → {ClassName: 0-indexed label}."""
        class_to_idx: dict[str, int] = {}
        for line in classind_path.read_text().strip().splitlines():
            parts = line.strip().split()
            label_1indexed = int(parts[0])
            class_name = parts[1]
            class_to_idx[class_name] = label_1indexed - 1
        return class_to_idx

    # ------------------------------------------------------------------
    # Split parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_split_file(
        path: Path, has_labels: bool, class_to_idx: dict[str, int]
    ) -> list[tuple[str, int]]:
        """Parse a UCF101 split file into (relative_path, 0-indexed label) pairs.

        trainlist01.txt format: ClassName/video.avi LABEL_1INDEXED
        testlist01.txt format:  ClassName/video.avi   (no label column)
        """
        entries: list[tuple[str, int]] = []
        for line in path.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            rel_path = parts[0]  # ClassName/v_ClassName_gXX_cXX.avi
            if has_labels:
                label = int(parts[1]) - 1  # 1-indexed → 0-indexed
            else:
                class_name = Path(rel_path).parent.name
                label = class_to_idx[class_name]
            entries.append((rel_path, label))
        return entries

    @staticmethod
    def _load_samples_from_split(
        data_root: Path,
        split_root: Path,
        split: str,
        class_to_idx: dict[str, int],
    ) -> list[tuple[Path, int]]:
        if split == "train":
            split_file = split_root / "trainlist01.txt"
            has_labels = True
        elif split in ("test", "val"):
            split_file = split_root / "testlist01.txt"
            has_labels = False
        else:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        entries = UCF101Dataset._parse_split_file(split_file, has_labels, class_to_idx)
        samples = [(data_root / rel_path, label) for rel_path, label in entries]
        return samples

    def _load_samples(self) -> list[tuple[Path, int]]:
        # Required by ABC; actual loading is done in __init__ above.
        raise NotImplementedError
