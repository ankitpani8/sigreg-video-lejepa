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


def test_encoder_forward() -> None:
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder

    enc = VideoViTEncoder(model_name="vit_tiny_patch16_224", pretrained=False, embed_dim=192, img_size=32)
    x = torch.randn(2, 3, 4, 32, 32)   # B=2, C=3, T=4, H=32, W=32
    out = enc(x)
    assert out.shape == (2, 16, 192)    # T*N=4*4=16 patches, D=192
    assert out.dtype == torch.float32


def test_shared_target_encoder() -> None:
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder
    from sigreg_video_lejepa.models.target_encoder import SharedTargetEncoder

    enc = VideoViTEncoder(model_name="vit_tiny_patch16_224", pretrained=False, embed_dim=192, img_size=32)
    target = SharedTargetEncoder()
    x = torch.randn(2, 3, 4, 32, 32)
    out = target.encode(enc, x)
    assert out.shape == (2, 16, 192)
    assert not out.requires_grad
    target.update(enc, decay=0.996)  # must not raise


def test_ema_target_encoder() -> None:
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder
    from sigreg_video_lejepa.models.target_encoder import EMATargetEncoder

    enc = VideoViTEncoder(model_name="vit_tiny_patch16_224", pretrained=False, embed_dim=192, img_size=32)
    target = EMATargetEncoder(enc)

    # perturb encoder weights so they differ from shadow
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(torch.ones_like(p))

    target.update(enc, decay=0.9)

    # shadow should now differ from both original and current encoder
    x = torch.randn(2, 3, 4, 32, 32)
    out = target.encode(enc, x)
    assert out.shape == (2, 16, 192)
    assert not out.requires_grad


def test_predictor_forward() -> None:
    from sigreg_video_lejepa.models.predictor import VideoJEPAPredictor

    pred = VideoJEPAPredictor(embed_dim=192, depth=2, num_heads=4)
    context = torch.randn(2, 16, 192)
    mask_tokens = torch.randn(2, 4, 192)
    out = pred(context, mask_tokens)
    assert out.shape == (2, 4, 192)
    assert out.dtype == torch.float32


def test_projector_forward() -> None:
    from sigreg_video_lejepa.models.projector import SIGRegProjector

    proj = SIGRegProjector(embed_dim=192, hidden_dim=2048, proj_dim=128)
    x = torch.randn(2, 16, 192)
    out = proj(x)
    assert out.shape == (2, 16, 128)
    assert out.dtype == torch.float32


def test_sigreg_loss_forward() -> None:
    from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss, sigreg_video_loss

    loss_fn = SIGRegLoss(num_projections=32, knots=17, t_max=3.0)

    proj = torch.randn(2, 16, 128, requires_grad=True)
    stat = loss_fn(proj)
    assert stat.shape == ()                 # scalar
    assert torch.isfinite(stat)
    stat.backward()
    assert proj.grad is not None

    # lam=0.0 → total loss equals L_pred exactly
    pred = torch.randn(2, 4, 192)
    target = torch.randn(2, 4, 192)
    proj2 = torch.randn(2, 16, 128)
    total, l_pred, l_sigreg = sigreg_video_loss(pred, target, proj2, loss_fn, lam=0.0)
    import torch.nn.functional as F
    assert torch.allclose(total, l_pred)


def test_synthetic_dataset_label_roundrobin() -> None:
    ds = SyntheticVideoDataset(num_clips=10, num_classes=5)
    labels = [ds[i][1] for i in range(10)]
    assert labels == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
