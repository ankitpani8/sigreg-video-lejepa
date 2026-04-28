"""Tests for Phase 3: linear probe evaluation pipeline."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from sigreg_video_lejepa.evaluation.feature_extractor import FeatureExtractor
from sigreg_video_lejepa.evaluation.linear_probe import LinearProbe
from sigreg_video_lejepa.models.encoder import VideoViTEncoder

# Tiny encoder: 8 tubelets (4/2 * 32/16 * 32/16), D=192 — matches smoke test scale
_ENC_KWARGS = dict(
    model_name="vit_tiny_patch16_224",
    pretrained=False,
    embed_dim=192,
    img_size=32,
    num_frames=4,
    t_patch=2,
    h_patch=16,
    w_patch=16,
)

_EMBED_DIM = 192
_NUM_CLASSES = 5
_N_TRAIN = 20
_N_TEST = 10


def _make_encoder() -> VideoViTEncoder:
    return VideoViTEncoder(**_ENC_KWARGS)


# ---------------------------------------------------------------------------
# FeatureExtractor tests
# ---------------------------------------------------------------------------


def test_feature_extractor_single_clip_shape() -> None:
    """5D batch (B, C, T, H, W) → (N, D) features, single-clip path."""
    enc = _make_encoder()
    extractor = FeatureExtractor(enc, torch.device("cpu"))
    clips = torch.randn(4, 3, 4, 32, 32)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=4)
    features, out_labels = extractor.extract(loader)
    assert features.shape == (4, _EMBED_DIM)
    assert out_labels.shape == (4,)


def test_feature_extractor_multi_clip_shape() -> None:
    """6D batch (B, 4, C, T, H, W) → (N, D) averaged features, multi-clip path."""
    enc = _make_encoder()
    extractor = FeatureExtractor(enc, torch.device("cpu"))
    clips = torch.randn(3, 4, 3, 4, 32, 32)  # (B=3, n_clips=4, C, T, H, W)
    labels = torch.zeros(3, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=3)
    features, out_labels = extractor.extract(loader)
    assert features.shape == (3, _EMBED_DIM)
    assert out_labels.shape == (3,)


def test_multi_clip_averaging_correctness() -> None:
    """4 identical clips → same feature vector as encoding the clip once."""
    enc = _make_encoder()
    extractor = FeatureExtractor(enc, torch.device("cpu"))

    single_clip = torch.randn(1, 3, 4, 32, 32)
    # Expand to (1, 4, 3, 4, 32, 32) — 4 identical copies
    multi_clips = single_clip.unsqueeze(1).expand(1, 4, -1, -1, -1, -1).contiguous()

    loader_single = DataLoader(
        TensorDataset(single_clip, torch.zeros(1, dtype=torch.long)), batch_size=1
    )
    loader_multi = DataLoader(
        TensorDataset(multi_clips, torch.zeros(1, dtype=torch.long)), batch_size=1
    )
    feat_single, _ = extractor.extract(loader_single)
    feat_multi, _ = extractor.extract(loader_multi)
    assert torch.allclose(feat_single, feat_multi, atol=1e-5)


def test_feature_extractor_no_grad() -> None:
    """FeatureExtractor must not accumulate gradients."""
    enc = _make_encoder()
    extractor = FeatureExtractor(enc, torch.device("cpu"))
    clips = torch.randn(2, 3, 4, 32, 32)
    labels = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=2)
    features, _ = extractor.extract(loader)
    assert not features.requires_grad


# ---------------------------------------------------------------------------
# LinearProbe tests
# ---------------------------------------------------------------------------


def test_linear_probe_trains_two_steps() -> None:
    """LinearProbe forward + backward runs without error."""
    features = torch.randn(_N_TRAIN, _EMBED_DIM)
    labels = torch.randint(0, _NUM_CLASSES, (_N_TRAIN,))
    loader = DataLoader(TensorDataset(features, labels), batch_size=8)

    probe = LinearProbe(embed_dim=_EMBED_DIM, num_classes=_NUM_CLASSES, num_epochs=2)
    trainer = L.Trainer(max_epochs=2, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(probe, loader)


def test_top1_top5_metrics_computed() -> None:
    """Top-1 and top-5 metrics are logged and finite after a validation epoch."""
    features = torch.randn(_N_TEST, _EMBED_DIM)
    labels = torch.randint(0, _NUM_CLASSES, (_N_TEST,))
    train_loader = DataLoader(TensorDataset(features, labels), batch_size=8)
    val_loader = DataLoader(TensorDataset(features, labels), batch_size=8)

    probe = LinearProbe(embed_dim=_EMBED_DIM, num_classes=_NUM_CLASSES, num_epochs=2)
    trainer = L.Trainer(max_epochs=2, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(probe, train_loader, val_loader)

    metrics = trainer.callback_metrics
    assert "val/top1" in metrics, f"val/top1 missing from metrics: {list(metrics)}"
    assert "val/top5" in metrics, f"val/top5 missing from metrics: {list(metrics)}"
    assert torch.isfinite(torch.tensor(metrics["val/top1"].item()))
    assert torch.isfinite(torch.tensor(metrics["val/top5"].item()))
    # top5 ≥ top1 always
    assert metrics["val/top5"].item() >= metrics["val/top1"].item()


def test_linear_probe_output_shape() -> None:
    """LinearProbe forward returns (B, num_classes) logits."""
    probe = LinearProbe(embed_dim=_EMBED_DIM, num_classes=_NUM_CLASSES)
    x = torch.randn(8, _EMBED_DIM)
    logits = probe(x)
    assert logits.shape == (8, _NUM_CLASSES)


# ---------------------------------------------------------------------------
# End-to-end: synthetic features → linear_probe.py subprocess
# ---------------------------------------------------------------------------


def test_end_to_end_synthetic_features(tmp_path: Path) -> None:
    """Create synthetic .pt features, run linear_probe.py, assert exit 0."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    torch.save(torch.randn(_N_TRAIN, _EMBED_DIM), features_dir / "train_features.pt")
    torch.save(torch.randint(0, _NUM_CLASSES, (_N_TRAIN,)), features_dir / "train_labels.pt")
    torch.save(torch.randn(_N_TEST, _EMBED_DIM), features_dir / "test_features.pt")
    torch.save(torch.randint(0, _NUM_CLASSES, (_N_TEST,)), features_dir / "test_labels.pt")

    project_root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "scripts/linear_probe.py",
            "+experiment=ucf101_linprobe",
            f"evaluation.features_dir={features_dir}",
            f"evaluation.num_classes={_NUM_CLASSES}",
            "evaluation.batch_size=8",
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
    )
    assert result.returncode == 0, (
        f"linear_probe.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "top-1" in result.stdout, "Expected top-1 accuracy in output"
