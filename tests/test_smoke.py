"""Smoke tests — Phase 0 pipeline plumbing and Phase 1 tubelet embedding + tube masking."""
from __future__ import annotations

import subprocess
import sys

import torch

from sigreg_video_lejepa.data.synthetic import SyntheticVideoDataset

# Shared encoder params used across tests: 8 tubelets (4/2 * 32/16 * 32/16)
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


def test_tubelet_embed_shape() -> None:
    from sigreg_video_lejepa.models.encoder import TubeletEmbed

    embed = TubeletEmbed(t_patch=2, h_patch=16, w_patch=16, embed_dim=192)
    x = torch.randn(2, 3, 4, 32, 32)    # B=2, C=3, T=4, H=32, W=32
    out = embed(x)
    # N = (4/2)*(32/16)*(32/16) = 8 tubelets
    assert out.shape == (2, 8, 192)
    assert out.dtype == torch.float32


def test_encoder_forward() -> None:
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder

    enc = VideoViTEncoder(**_ENC_KWARGS)
    x = torch.randn(2, 3, 4, 32, 32)
    out = enc(x)
    assert out.shape == (2, 8, 192)      # 8 tubelets, D=192
    assert out.dtype == torch.float32
    assert enc.num_tubelets == 8


def test_encoder_token_selection() -> None:
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder

    enc = VideoViTEncoder(**_ENC_KWARGS)
    x = torch.randn(2, 3, 4, 32, 32)
    ctx_idx = torch.tensor([0, 3])       # select 2 of 8 tubes
    out = enc(x, token_indices=ctx_idx)
    assert out.shape == (2, 2, 192)


def test_shared_target_encoder() -> None:
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder
    from sigreg_video_lejepa.models.target_encoder import SharedTargetEncoder

    enc = VideoViTEncoder(**_ENC_KWARGS)
    target = SharedTargetEncoder()
    x = torch.randn(2, 3, 4, 32, 32)
    out = target.encode(enc, x)
    assert out.shape == (2, 8, 192)
    assert not out.requires_grad
    target.update(enc, decay=0.996)      # must not raise


def test_ema_target_encoder() -> None:
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder
    from sigreg_video_lejepa.models.target_encoder import EMATargetEncoder

    enc = VideoViTEncoder(**_ENC_KWARGS)
    target = EMATargetEncoder()
    target.initialize_from(enc)

    with torch.no_grad():
        for p in enc.parameters():
            p.add_(torch.ones_like(p))

    target.update(enc, decay=0.9)

    x = torch.randn(2, 3, 4, 32, 32)
    out = target.encode(enc, x)
    assert out.shape == (2, 8, 192)
    assert not out.requires_grad


def test_predictor_forward() -> None:
    from sigreg_video_lejepa.models.predictor import VideoJEPAPredictor

    pred = VideoJEPAPredictor(embed_dim=192, depth=2, num_heads=4)
    context = torch.randn(2, 2, 192)     # 2 context tubes
    mask_tokens = torch.randn(2, 6, 192) # 6 target tubes
    out = pred(context, mask_tokens)
    assert out.shape == (2, 6, 192)
    assert out.dtype == torch.float32


def test_projector_forward() -> None:
    from sigreg_video_lejepa.models.projector import SIGRegProjector

    proj = SIGRegProjector(embed_dim=192, hidden_dim=2048, proj_dim=128)
    x = torch.randn(2, 8, 192)          # 8 tubelets
    out = proj(x)
    assert out.shape == (2, 8, 128)
    assert out.dtype == torch.float32


def test_sigreg_loss_forward() -> None:
    from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss, sigreg_video_loss

    loss_fn = SIGRegLoss(num_projections=32, knots=17, t_max=3.0)

    proj = torch.randn(2, 8, 128, requires_grad=True)
    stat = loss_fn(proj)
    assert stat.shape == ()
    assert torch.isfinite(stat)
    stat.backward()
    assert proj.grad is not None

    # lam=0.0 → total loss equals L_pred exactly
    pred = torch.randn(2, 6, 192)
    target = torch.randn(2, 6, 192)
    proj2 = torch.randn(2, 2, 128)
    total, l_pred, l_sigreg = sigreg_video_loss(pred, target, proj2, loss_fn, lam=0.0)
    assert torch.allclose(total, l_pred)


def test_tube_masker() -> None:
    from sigreg_video_lejepa.data.masking import TubeMasker

    masker = TubeMasker(mask_ratio=0.75)
    ctx_idx, tgt_idx = masker(8, device=torch.device("cpu"))

    assert ctx_idx.shape == (2,)         # 25% of 8
    assert tgt_idx.shape == (6,)         # 75% of 8
    # indices are sorted
    assert (ctx_idx == ctx_idx.sort().values).all()
    assert (tgt_idx == tgt_idx.sort().values).all()
    # union is all 8 tubes
    all_idx = torch.cat([ctx_idx, tgt_idx]).sort().values
    assert torch.equal(all_idx, torch.arange(8))


def test_lightning_two_steps() -> None:
    """Phase 0 compat path: masker=None, 4-token proxy target."""
    import lightning as L
    from torch.utils.data import DataLoader

    from sigreg_video_lejepa.models.encoder import VideoViTEncoder
    from sigreg_video_lejepa.models.predictor import VideoJEPAPredictor
    from sigreg_video_lejepa.models.projector import SIGRegProjector
    from sigreg_video_lejepa.models.target_encoder import SharedTargetEncoder
    from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule
    from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss

    enc = VideoViTEncoder(**_ENC_KWARGS)
    module = VideoJEPAModule(
        encoder=enc,
        target_encoder=SharedTargetEncoder(),
        predictor=VideoJEPAPredictor(embed_dim=192),
        projector=SIGRegProjector(embed_dim=192),
        sigreg_loss=SIGRegLoss(num_projections=32),
        masker=None,
        lam=0.0,
    )
    ds = SyntheticVideoDataset(num_clips=10, num_frames=4, height=32, width=32)
    loader = DataLoader(ds, batch_size=2)
    trainer = L.Trainer(max_steps=2, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(module, loader)


def test_lightning_with_masking() -> None:
    """Phase 1 path: TubeMasker active, context encoder sees only unmasked tubes."""
    import lightning as L
    from torch.utils.data import DataLoader

    from sigreg_video_lejepa.data.masking import TubeMasker
    from sigreg_video_lejepa.models.encoder import VideoViTEncoder
    from sigreg_video_lejepa.models.predictor import VideoJEPAPredictor
    from sigreg_video_lejepa.models.projector import SIGRegProjector
    from sigreg_video_lejepa.models.target_encoder import SharedTargetEncoder
    from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule
    from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss

    enc = VideoViTEncoder(**_ENC_KWARGS)
    module = VideoJEPAModule(
        encoder=enc,
        target_encoder=SharedTargetEncoder(),
        predictor=VideoJEPAPredictor(embed_dim=192),
        projector=SIGRegProjector(embed_dim=192),
        sigreg_loss=SIGRegLoss(num_projections=32),
        masker=TubeMasker(mask_ratio=0.75),
        lam=0.0,
    )
    ds = SyntheticVideoDataset(num_clips=10, num_frames=4, height=32, width=32)
    loader = DataLoader(ds, batch_size=2)
    trainer = L.Trainer(max_steps=2, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(module, loader)


def test_synthetic_dataset_label_roundrobin() -> None:
    ds = SyntheticVideoDataset(num_clips=10, num_classes=5)
    labels = [ds[i][1] for i in range(10)]
    assert labels == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_smoke_end_to_end_phase0() -> None:
    """Integration test: Phase 0 Hydra + Lightning wiring (no masking)."""
    import os
    result = subprocess.run(
        [sys.executable, "scripts/pretrain.py", "+experiment=smoke_test_phase0"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, (
        f"pretrain.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_smoke_end_to_end_phase1() -> None:
    """Integration test: Phase 1 Hydra + Lightning wiring (tube masking active)."""
    import os
    result = subprocess.run(
        [sys.executable, "scripts/pretrain.py", "+experiment=smoke_test_phase1"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, (
        f"pretrain.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
