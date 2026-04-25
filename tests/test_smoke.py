"""Phase 0 smoke tests — pipeline plumbing on synthetic data."""
from __future__ import annotations

import subprocess
import sys

import torch

from sigreg_video_lejepa.data.synthetic import SyntheticVideoDataset


def test_synthetic_dataset_shape() -> None:
    ds = SyntheticVideoDataset(num_clips=10, num_frames=4, height=32, width=32, num_classes=5)
    assert len(ds) == 10
    clip, label = ds[0]
    assert clip.shape == (3, 4, 32, 32)
    assert clip.dtype == torch.float32
    assert 0 <= label < 5


def test_synthetic_dataset_determinism() -> None:
    ds = SyntheticVideoDataset()
    clip_a, _ = ds[3]
    clip_b, _ = ds[3]
    assert torch.equal(clip_a, clip_b)


def test_synthetic_dataset_label_roundrobin() -> None:
    ds = SyntheticVideoDataset(num_clips=10, num_classes=5)
    labels = [ds[i][1] for i in range(10)]
    assert labels == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
