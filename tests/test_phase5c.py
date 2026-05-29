"""Phase 5c tests: SPMD harness components (CPU-runnable; no XLA required).

What's tested here:
  - compute_loss / _forward_representations callable outside Lightning Trainer
  - build_optimizer_and_scheduler param isolation and scheduler-resume correctness
  - SIGReg global-batch mathematical property (what the XLA all-gather must guarantee)
  - Config composition: existing phase5_*_tpu configs read correctly by pretrain_tpu.py

What cannot be tested here (requires real XLA device):
  - The xs.mark_sharding all-gather collective
  - EMA shadow update on actual TPU
  - End-to-end pretrain_tpu.py on 8-chip v5e-8
"""
from __future__ import annotations

import math

import torch

# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_tiny_module():
    """Instantiate VideoJEPAModule from the smoke_test_tpu model config."""
    from hydra import compose, initialize
    from hydra.utils import instantiate

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase5_tpu_smoke"])

    from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule

    encoder = instantiate(cfg.model.encoder)
    target_encoder = instantiate(cfg.model.target_encoder)
    predictor = instantiate(cfg.model.predictor)
    projector = instantiate(cfg.model.projector)
    sigreg_loss = instantiate(cfg.model.sigreg_loss)
    masker_cfg = cfg.model.get("masker")
    masker = instantiate(masker_cfg) if masker_cfg is not None else None

    module = VideoJEPAModule(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        projector=projector,
        sigreg_loss=sigreg_loss,
        masker=masker,
        lam=0.02,
        ema_decay=None,
        lr=3e-4,
        weight_decay=0.05,
        warmup_steps=10,
        total_steps=100,
    )
    return module, cfg


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_compute_loss_cpu() -> None:
    """compute_loss and _forward_representations work on CPU without a Trainer."""
    module, cfg = _build_tiny_module()
    module.eval()

    # smoke_test_tpu: img_size=64, num_frames=8, channels=3
    x = torch.randn(4, 3, 8, 64, 64)

    pred, tgt_slice, proj = module._forward_representations(x)
    assert pred.shape[0] == 4
    assert pred.ndim == 3  # (B, N_tgt, D)
    assert proj.ndim == 3  # (B, N_ctx, proj_dim)

    total, l_pred, l_sigreg = module.compute_loss(x)
    assert total.ndim == 0 and torch.isfinite(total)
    assert l_pred.ndim == 0 and torch.isfinite(l_pred)
    assert l_sigreg.ndim == 0 and torch.isfinite(l_sigreg)


def test_build_optimizer_and_scheduler() -> None:
    """build_optimizer_and_scheduler isolates correct param groups and resumes cleanly."""
    from sigreg_video_lejepa.training.lightning_module import build_optimizer_and_scheduler

    module, _ = _build_tiny_module()

    # ── with scheduler ────────────────────────────────────────────────────────
    opt, sched = build_optimizer_and_scheduler(
        module, lr=1e-3, weight_decay=0.05, warmup_steps=10, total_steps=100
    )
    assert isinstance(opt, torch.optim.AdamW)
    assert sched is not None

    # Optimizer must contain ONLY encoder + predictor + projector params.
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    for p in list(module.encoder.parameters()) + list(module.predictor.parameters()) + list(module.projector.parameters()):
        assert id(p) in opt_ids, "encoder/predictor/projector param missing from optimizer"

    # target_encoder is SharedTargetEncoder (plain class, no nn.Module params).
    # sigreg_loss buffers must NOT be in the optimizer.
    for buf in module.sigreg_loss.buffers():
        assert id(buf) not in opt_ids, "sigreg_loss buffer leaked into optimizer"

    # ── without scheduler (total_steps=0) ────────────────────────────────────
    opt2, sched2 = build_optimizer_and_scheduler(
        module, lr=1e-3, weight_decay=0.05, warmup_steps=0, total_steps=0
    )
    assert sched2 is None

    # ── scheduler resume via last_epoch ───────────────────────────────────────
    start_step = 20
    opt3, sched3 = build_optimizer_and_scheduler(
        module, lr=1e-3, weight_decay=0.05, warmup_steps=10, total_steps=100,
        last_epoch=start_step - 1,
    )
    assert sched3 is not None
    # After construction with last_epoch=start_step-1, get_last_lr() reflects step 20.
    # At step 20 (post-warmup): progress=(20-10)/(100-10)=1/9; lr=lr*0.5*(1+cos(pi/9))
    progress = (start_step - 10) / (100 - 10)
    expected_factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    actual_factor = sched3.get_last_lr()[0] / 1e-3
    assert abs(actual_factor - expected_factor) < 1e-6, (
        f"Scheduler resume wrong: expected factor {expected_factor:.6f}, got {actual_factor:.6f}"
    )


def test_sigreg_global_batch_semantics() -> None:
    """SIGReg on full batch equals SIGReg on all-gathered shards (same tensor, same seed).

    This is the mathematical property that the XLA xs.mark_sharding all-gather must
    satisfy on real hardware. On CPU we verify it directly: sharding and re-gathering
    must be a no-op at the tensor level.

    Concretely: the EP test statistic is determined by the set of samples passed in.
    cat(shards) == full_batch → sigreg(cat(shards)) == sigreg(full_batch) (with same seed).
    """
    from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss

    sigreg = SIGRegLoss(num_projections=64, knots=9, t_max=3.0)
    sigreg.eval()

    # Global batch: 64 samples, 32 context tokens, proj_dim=16
    B_global, N_ctx, proj_dim = 64, 32, 16
    torch.manual_seed(7)
    proj_full = torch.randn(B_global, N_ctx, proj_dim)

    # Simulate 8-core SPMD: split into 8 shards of 8
    num_shards = 8
    shards = torch.split(proj_full, B_global // num_shards, dim=0)
    assert len(shards) == num_shards
    assert shards[0].shape == (8, N_ctx, proj_dim)

    # Simulate all-gather: concatenate shards → must exactly reconstruct the full batch
    proj_gathered = torch.cat(shards, dim=0)
    assert torch.equal(proj_gathered, proj_full), "all-gather must be a no-op on identical shards"

    # With fixed seed, both calls use identical random projections A → results must match
    torch.manual_seed(42)
    result_full = sigreg(proj_full)

    torch.manual_seed(42)
    result_gathered = sigreg(proj_gathered)

    assert torch.allclose(result_full, result_gathered, atol=1e-5), (
        f"SIGReg mismatch: full={result_full.item():.6f}, gathered={result_gathered.item():.6f}"
    )


def test_phase5c_config_composition() -> None:
    """Existing phase5_*_tpu configs compose with fields pretrain_tpu.py reads."""
    from hydra import compose, initialize

    # Smoke config: max_steps=50, batch_size=8 (1/core minimum), synthetic data
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase5_tpu_smoke"])

    assert cfg.trainer.max_steps == 50
    assert cfg.data.batch_size == 8, (
        f"phase5_tpu_smoke must use batch_size=8 (1/core for 8-core SPMD), got {cfg.data.batch_size}"
    )

    # Real run configs: batch_size=64 (global; 8/core × 8 cores = GPU DDP effective batch)
    real_tpu_exps = [
        "phase5_sigreg_seed0_tpu",
        "phase5_sigreg_seed1_tpu",
        "phase5_ema_seed0_tpu",
        "phase5_ema_seed1_tpu",
    ]
    with initialize(config_path="../configs", version_base="1.3"):
        for exp in real_tpu_exps:
            cfg = compose("config", overrides=[f"+experiment={exp}"])
            assert cfg.trainer.max_steps == 75000, f"{exp}: expected max_steps=75000"
            assert cfg.data.batch_size == 64, (
                f"{exp}: expected batch_size=64 (SPMD global batch), got {cfg.data.batch_size}"
            )

def test_smoke_data_model_shape_alignment():
    """Synthetic-data smoke configs must match their model's tubelet count."""
    from hydra import compose, initialize
    for exp in ['phase5_tpu_smoke', 'phase5_gpu_smoke']:
        with initialize(config_path='../configs', version_base='1.3'):
            cfg = compose('config', overrides=[f'+experiment={exp}'])
        enc = cfg.model.encoder
        ds = cfg.data.dataset
        model_tubelets = (
            (enc.num_frames // enc.t_patch)
            * (enc.img_size // enc.h_patch)
            * (enc.img_size // enc.w_patch)
        )
        data_tubelets = (
            (ds.num_frames // enc.t_patch)
            * (ds.height // enc.h_patch)
            * (ds.width // enc.w_patch)
        )
        assert model_tubelets == data_tubelets, (
            f"{exp}: model expects {model_tubelets} tubelets but data produces {data_tubelets}"
        )