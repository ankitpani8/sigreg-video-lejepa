from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticVideoDataset(Dataset):
    """In-memory random video clips for pipeline smoke tests. No disk I/O."""

    def __init__(
        self,
        num_clips: int = 10,
        num_frames: int = 4,
        height: int = 32,
        width: int = 32,
        num_classes: int = 5,
        seed: int = 42,
    ) -> None:
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        g = torch.Generator()
        g.manual_seed(self.seed * 1000 + idx)
        clip = torch.rand(3, self.num_frames, self.height, self.width, generator=g)
        label = idx % self.num_classes
        return clip, label
