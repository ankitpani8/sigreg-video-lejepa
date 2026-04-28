"""Tests for Phase 2: UCF101 data pipeline."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sigreg_video_lejepa.data.transforms import UCF101Transform
from sigreg_video_lejepa.data.ucf101 import UCF101Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRANSFORM = UCF101Transform(crop_size=64)
_NUM_FRAMES = 16
_FRAME_STRIDE = 6


def _make_dataset(ucf101_root: Path, split: str = "train") -> UCF101Dataset:
    return UCF101Dataset(
        data_root=ucf101_root / "videos",
        split_root=ucf101_root / "splits",
        split=split,
        num_frames=_NUM_FRAMES,
        frame_stride=_FRAME_STRIDE,
        transform=_TRANSFORM,
        local_cache=None,
    )


# ---------------------------------------------------------------------------
# Unit tests (always run, no real data needed)
# ---------------------------------------------------------------------------


def test_instantiation(ucf101_root: Path) -> None:
    ds = _make_dataset(ucf101_root)
    # 3 classes × 2 train clips = 6
    assert len(ds) == 6


def test_getitem_shape(ucf101_root: Path) -> None:
    ds = _make_dataset(ucf101_root)
    clip, label = ds[0]
    # Must be (C, T, H, W) — (T, C, H, W) is wrong
    assert clip.shape == (3, _NUM_FRAMES, 64, 64), f"Got {clip.shape}"
    assert clip.dtype == torch.float32


def test_normalized_range(ucf101_root: Path) -> None:
    ds = _make_dataset(ucf101_root)
    clip, _ = ds[0]
    # ImageNet normalization maps [0,1] roughly to [-3, 3]
    assert clip.min().item() > -4.0
    assert clip.max().item() < 4.0


def test_short_video_looping(ucf101_root: Path) -> None:
    """32-frame clips trigger the looping path.

    16 frames × stride 6 = 96 source frames needed; 32 < 96 so every clip
    in the fixture exercises the modulo-looping code.
    """
    ds = _make_dataset(ucf101_root)
    clip, _ = ds[0]
    assert clip.shape == (3, _NUM_FRAMES, 64, 64)


def test_label_mapping(ucf101_root: Path) -> None:
    ds = _make_dataset(ucf101_root)
    labels = {ds[i][1] for i in range(len(ds))}
    # 3 classes → labels 0, 1, 2
    assert labels == {0, 1, 2}


def test_test_split_labels(ucf101_root: Path) -> None:
    """testlist01.txt has no label column; labels must be inferred from directory."""
    ds = _make_dataset(ucf101_root, split="test")
    assert len(ds) == 3  # 1 clip per class
    labels = {label for _, label in ds.samples}
    assert labels == {0, 1, 2}


def test_eval_getitem_multicliip_shape(ucf101_root: Path) -> None:
    """Eval split __getitem__ returns (4, C, T, H, W) — 4 evenly-spaced clips."""
    ds = _make_dataset(ucf101_root, split="test")
    clip, label = ds[0]
    assert clip.shape == (4, 3, _NUM_FRAMES, 64, 64), f"Got {clip.shape}"
    assert clip.dtype == torch.float32
    assert 0 <= label < 3


def test_invalid_data_root() -> None:
    with pytest.raises(FileNotFoundError, match="data_root"):
        UCF101Dataset(
            data_root="/nonexistent/UCF-101",
            split_root="/nonexistent/splits",
            split="train",
            num_frames=16,
            frame_stride=6,
            transform=_TRANSFORM,
        )


def test_invalid_split_root(ucf101_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        UCF101Dataset(
            data_root=ucf101_root / "videos",
            split_root="/nonexistent/splits",
            split="train",
            num_frames=16,
            frame_stride=6,
            transform=_TRANSFORM,
        )


# ---------------------------------------------------------------------------
# Integration test — realistic token count (slow, CPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lightning_with_ucf101_small_masker(ucf101_root: Path) -> None:
    """End-to-end: encoder + masker + predictor at ucf101_small scale (512 tubelets).

    First time the masking + encoder + predictor stack is exercised at realistic
    token counts (not the Phase 0/1 smoke scale of 8 tubes). Also verifies no
    silent NaN propagation through the loss.
    """
    import lightning as L
    from torch.utils.data import DataLoader

    from sigreg_video_lejepa.data.masking import TubeMasker
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder
    from sigreg_video_lejepa.models.predictor import VideoJEPAPredictor
    from sigreg_video_lejepa.models.projector import SIGRegProjector
    from sigreg_video_lejepa.models.target_encoder import SharedTargetEncoder
    from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule
    from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss

    # ucf101_small config: 64×64, 16 frames, 2×8×8 tubelet → 512 tubes
    enc = VideoViTEncoder(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        embed_dim=192,
        img_size=64,
        num_frames=16,
        t_patch=2,
        h_patch=8,
        w_patch=8,
    )
    assert enc.num_tubelets == 512, f"Expected 512, got {enc.num_tubelets}"

    module = VideoJEPAModule(
        encoder=enc,
        target_encoder=SharedTargetEncoder(),
        predictor=VideoJEPAPredictor(embed_dim=192, depth=6, num_heads=6),
        projector=SIGRegProjector(embed_dim=192),
        sigreg_loss=SIGRegLoss(num_projections=256),
        masker=TubeMasker(mask_ratio=0.75),  # 384 tgt, 128 ctx
        lam=0.0,
    )

    ds = _make_dataset(ucf101_root)
    loader = DataLoader(ds, batch_size=2, num_workers=0)

    logged: dict = {}

    class _LossCapture(L.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            logged.update({k: v.item() for k, v in trainer.callback_metrics.items()})

    trainer = L.Trainer(
        max_steps=2,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
        callbacks=[_LossCapture()],
    )
    trainer.fit(module, loader)

    assert "train/loss" in logged, "train/loss not logged"
    assert "train/l_pred" in logged, "train/l_pred not logged"
    assert "train/l_sigreg" in logged, "train/l_sigreg not logged"

    assert torch.isfinite(torch.tensor(logged["train/loss"])), "train/loss is not finite"
    assert torch.isfinite(torch.tensor(logged["train/l_pred"])), "train/l_pred is not finite"
    assert torch.isfinite(torch.tensor(logged["train/l_sigreg"])), "train/l_sigreg is not finite"


# ---------------------------------------------------------------------------
# Dry-run integration test (Colab only — skipped locally)
# ---------------------------------------------------------------------------

_UCF101_DRIVE_PATH = Path("/content/drive/MyDrive/datasets/ucf101/UCF-101")


@pytest.mark.skipif(
    not _UCF101_DRIVE_PATH.exists(),
    reason=(
        "UCF101 data not found at expected path; "
        "run on Colab after mounting Drive (expected: /content/drive/MyDrive/datasets/ucf101/)"
    ),
)
def test_ucf101_dryrun() -> None:
    """10-step end-to-end dry run on real UCF101 data via Hydra + Lightning."""
    import os

    result = subprocess.run(
        [sys.executable, "scripts/pretrain.py", "+experiment=ucf101_dryrun"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, (
        f"pretrain.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
