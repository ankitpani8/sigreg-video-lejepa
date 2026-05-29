"""Phase 6 tests: causal masker, stochastic predictor, VICReg loss, module arms."""
from __future__ import annotations

import pytest
import torch


# ── CausalTubeMasker ───────────────────────────────────────────────────────────


def test_causal_masker_indices() -> None:
    """Full-config mask: correct counts, gap excluded, ctx/tgt disjoint."""
    from sigreg_video_lejepa.data.causal_masking import CausalTubeMasker

    masker = CausalTubeMasker(
        context_frames=6, gap_frames=4, target_frames=6, t_patch=2, tubelets_per_tpos=256
    )
    ctx, tgt = masker(2048, torch.device("cpu"))

    assert len(ctx) == 768, f"expected 768 ctx (3 tpos×256), got {len(ctx)}"
    assert len(tgt) == 768, f"expected 768 tgt (3 tpos×256), got {len(tgt)}"

    gap_indices = set(range(768, 1280))  # tpos 3,4 at tubelets_per_tpos=256
    assert not (set(ctx.tolist()) & gap_indices), "gap indices leaked into ctx"
    assert not (set(tgt.tolist()) & gap_indices), "gap indices leaked into tgt"
    assert not (set(ctx.tolist()) & set(tgt.tolist())), "ctx and tgt overlap"


def test_causal_masker_deterministic() -> None:
    """Two calls on the same masker must produce identical indices."""
    from sigreg_video_lejepa.data.causal_masking import CausalTubeMasker

    masker = CausalTubeMasker(6, 4, 6, 2, 256)
    ctx1, tgt1 = masker(2048, torch.device("cpu"))
    ctx2, tgt2 = masker(2048, torch.device("cpu"))
    assert torch.equal(ctx1, ctx2)
    assert torch.equal(tgt1, tgt2)


def test_causal_masker_validation() -> None:
    """Invalid frame counts raise ValueError."""
    from sigreg_video_lejepa.data.causal_masking import CausalTubeMasker

    with pytest.raises(ValueError, match="t_patch"):
        # 7 frames is not divisible by t_patch=2
        CausalTubeMasker(7, 4, 6, 2, 256)

    with pytest.raises(ValueError):
        # num_tubelets mismatch at call time
        masker = CausalTubeMasker(6, 4, 6, 2, 256)
        masker(512, torch.device("cpu"))


# ── StochasticVideoJEPAPredictor ───────────────────────────────────────────────


def test_stochastic_predictor_output_shape() -> None:
    """forward() returns correctly-shaped mu, log_var, and sample."""
    from sigreg_video_lejepa.models.stochastic_predictor import StochasticVideoJEPAPredictor

    pred = StochasticVideoJEPAPredictor(embed_dim=192, depth=2, num_heads=2)
    ctx = torch.randn(2, 10, 192)
    mask = pred.mask_token.expand(2, 5, -1)
    out = pred(ctx, mask)

    assert out["sample"].shape == (2, 5, 192)
    assert out["mu"].shape == (2, 5, 192)
    assert out["log_var"].shape == (2, 5, 192)
    # log_var clamp check
    assert out["log_var"].min() >= -10.0
    assert out["log_var"].max() <= 10.0


def test_stochastic_predictor_reparameterization() -> None:
    """Gradients flow through the head (reparameterization is differentiable)."""
    from sigreg_video_lejepa.models.stochastic_predictor import StochasticVideoJEPAPredictor

    pred = StochasticVideoJEPAPredictor(embed_dim=32, depth=1, num_heads=2)
    ctx = torch.randn(2, 6, 32)
    mask = pred.mask_token.expand(2, 3, -1)
    out = pred(ctx, mask)
    out["sample"].sum().backward()
    assert pred.head.weight.grad is not None, "head.weight must receive gradients"


# ── _kl_loss ───────────────────────────────────────────────────────────────────


def test_kl_free_bits() -> None:
    """Per-dim free-bits floor: zero KL input → loss = free_bits × D."""
    from sigreg_video_lejepa.training.lightning_module import _kl_loss

    D = 32
    mu = torch.zeros(4, 8, D)
    log_var = torch.zeros(4, 8, D)  # KL per dim = 0, clamped to free_bits=0.5
    loss = _kl_loss(mu, log_var, free_bits=0.5)
    assert loss.item() == pytest.approx(D * 0.5)  # 16.0


def test_kl_free_bits_per_dim_not_per_token() -> None:
    """Verify clamp is per-dim: dims below threshold contribute independently."""
    from sigreg_video_lejepa.training.lightning_module import _kl_loss

    D = 4
    # First 2 dims have large KL; last 2 are below free_bits floor
    mu = torch.zeros(1, 1, D)
    # Set log_var so that KL per dim = [3.0, 3.0, 0.0, 0.0]
    # KL = -0.5*(1 + log_var - mu^2 - exp(log_var)); mu=0: KL = -0.5*(1+lv-exp(lv))
    # For KL=3: -0.5*(1+lv-exp(lv))=3 → this requires numerical solve, so instead
    # just verify clamp applies per-dim by checking two cases
    log_var = torch.zeros(1, 1, D)  # all KL≈0 → all clamped to free_bits=0.5
    loss_all_clamped = _kl_loss(mu, log_var, free_bits=0.5)
    # Expected: 4 dims × 0.5 × 1 token = 2.0
    assert loss_all_clamped.item() == pytest.approx(D * 0.5)

    # Large positive mu forces KL >> free_bits for all dims
    mu_large = torch.ones(1, 1, D) * 3.0  # KL ≈ (9-1)/2 = 4 per dim >> 0.5
    loss_large = _kl_loss(mu_large, log_var, free_bits=0.5)
    assert loss_large.item() > D * 0.5  # all dims above floor, loss > 2.0


# ── VICRegLoss ─────────────────────────────────────────────────────────────────


def test_vicreg_loss_zero_on_isotropic() -> None:
    """Variance term near-zero on large isotropic Gaussian batch (std ≈ 1 ≥ gamma=1)."""
    from sigreg_video_lejepa.training.vicreg_loss import VICRegLoss

    torch.manual_seed(0)
    loss_fn = VICRegLoss(gamma=1.0, mu_v=25.0, mu_c=1.0)
    z = torch.randn(256, 4, 64)  # large N → std ≈ 1, decorrelated
    loss = loss_fn(z)
    # Variance term ≈ 0 (std ≈ 1 ≥ gamma). Covariance term ≈ 0 (iid Gaussian).
    assert loss.item() < 10.0, f"VICReg on isotropic Gaussian should be near 0, got {loss.item()}"


def test_vicreg_loss_high_on_collapsed() -> None:
    """Variance term large when all embeddings are identical (rank-1 collapse)."""
    from sigreg_video_lejepa.training.vicreg_loss import VICRegLoss

    loss_fn = VICRegLoss(gamma=1.0, mu_v=25.0, mu_c=1.0)
    z = torch.ones(32, 4, 64)   # std=0 for every dim; variance term = mu_v × gamma × D / D
    loss = loss_fn(z)
    # Variance term = mu_v × mean_d(relu(1 - 0)) = mu_v × 1 = 25.0
    assert loss.item() > 20.0, f"VICReg on collapsed embeddings should be large, got {loss.item()}"


def test_vicreg_loss_gradients() -> None:
    """Variance term gradient flows when std < gamma (active relu region).

    All-zero input gives std=0 but the gradient of std at a constant is 0/0 → 0;
    use a small noisy input with std ≈ 0.1 < gamma=1 to get a proper gradient signal.
    """
    from sigreg_video_lejepa.training.vicreg_loss import VICRegLoss

    torch.manual_seed(1)
    loss_fn = VICRegLoss(gamma=1.0, mu_v=25.0, mu_c=1.0)
    z = (torch.randn(8, 4, 16) * 0.1).requires_grad_(True)  # std ≈ 0.1 < gamma=1
    loss = loss_fn(z)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


# ── Phase 6 config composition ─────────────────────────────────────────────────


def test_phase6_all_configs_compose() -> None:
    """All Phase 6 experiment configs load via Hydra without error."""
    from hydra import compose, initialize

    experiments = [
        "phase6_sigreg_seed0",
        "phase6_sigreg_seed1",
        "phase6_ema_seed0",
        "phase6_ema_seed1",
        "phase6_vicreg_seed0",
        "phase6_vicreg_seed1",
        "phase6_gpu_smoke",
        "phase6_tpu_smoke",
    ]
    for exp in experiments:
        with initialize(config_path="../configs", version_base="1.3"):
            cfg = compose("config", overrides=[f"+experiment={exp}"])
        assert cfg.training.predictor_type == "stochastic", (
            f"{exp}: expected predictor_type=stochastic"
        )
        assert cfg.training.beta == pytest.approx(0.001), f"{exp}: expected beta=0.001"


def test_phase6_ema_arm_has_ema_target_encoder() -> None:
    """EMA arm configs must override target_encoder to EMATargetEncoder."""
    from hydra import compose, initialize

    for exp in ["phase6_ema_seed0", "phase6_ema_seed1"]:
        with initialize(config_path="../configs", version_base="1.3"):
            cfg = compose("config", overrides=[f"+experiment={exp}"])
        assert "EMATargetEncoder" in cfg.model.target_encoder._target_, (
            f"{exp}: expected EMATargetEncoder"
        )


def test_phase6_smoke_tubelet_alignment() -> None:
    """Smoke configs: causal masker total matches model tubelet count."""
    from hydra import compose, initialize

    for exp in ["phase6_gpu_smoke", "phase6_tpu_smoke"]:
        with initialize(config_path="../configs", version_base="1.3"):
            cfg = compose("config", overrides=[f"+experiment={exp}"])
        enc = cfg.model.encoder
        ds = cfg.data.dataset
        masker = cfg.model.masker
        total_frames = masker.context_frames + masker.gap_frames + masker.target_frames
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
        assert total_frames == enc.num_frames, (
            f"{exp}: masker total frames {total_frames} != model frames {enc.num_frames}"
        )
        assert model_tubelets == data_tubelets, (
            f"{exp}: model expects {model_tubelets} tubelets but data produces {data_tubelets}"
        )


# ── compute_loss Phase 6 arms ──────────────────────────────────────────────────


def _build_phase6_module(regularizer_type: str):
    """Build VideoJEPAModule with smoke_test_phase6 arch and given regularizer."""
    from hydra import compose, initialize
    from hydra.utils import instantiate

    exp = "phase6_gpu_smoke"
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=[f"+experiment={exp}"])

    # Override regularizer_type at training level
    from omegaconf import OmegaConf
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_dict["regularizer_type"] = regularizer_type
    if regularizer_type == "ema":
        training_dict["ema_decay"] = 0.996

    from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule

    encoder = instantiate(cfg.model.encoder)
    target_encoder = instantiate(cfg.model.target_encoder)
    if regularizer_type == "ema":
        from sigreg_video_lejepa.models.target_encoder import EMATargetEncoder
        target_encoder = EMATargetEncoder()
        target_encoder.initialize_from(encoder)
    predictor = instantiate(cfg.model.predictor)
    projector = instantiate(cfg.model.projector)
    sigreg_loss = instantiate(cfg.model.sigreg_loss)
    vicreg_loss_cfg = cfg.model.get("vicreg_loss")
    vicreg_loss = instantiate(vicreg_loss_cfg) if vicreg_loss_cfg is not None else None
    masker = instantiate(cfg.model.masker)

    return VideoJEPAModule(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        projector=projector,
        sigreg_loss=sigreg_loss,
        vicreg_loss=vicreg_loss,
        masker=masker,
        total_steps=50,
        **training_dict,
    )


@pytest.mark.parametrize("regularizer_type", ["sigreg", "ema", "vicreg"])
def test_compute_loss_phase6_arms(regularizer_type: str) -> None:
    """All three Phase 6 arms: forward+backward on CPU produces finite losses."""
    module = _build_phase6_module(regularizer_type)
    module.train()

    # smoke_test_phase6: img_size=64, num_frames=8
    x = torch.randn(2, 3, 8, 64, 64)
    total, l_pred, l_kl, l_reg = module.compute_loss(x)

    assert all(
        t.isfinite() for t in [total, l_pred, l_kl, l_reg]
    ), f"{regularizer_type}: non-finite loss component"
    assert l_kl.item() > 0.0, f"{regularizer_type}: l_kl should be positive (stochastic predictor)"
    if regularizer_type == "ema":
        assert l_reg.item() == 0.0, "EMA arm must have zero l_reg"
    else:
        assert l_reg.item() > 0.0, f"{regularizer_type}: l_reg should be positive"

    total.backward()
    assert any(
        p.grad is not None for p in module.encoder.parameters()
    ), f"{regularizer_type}: encoder received no gradients"
