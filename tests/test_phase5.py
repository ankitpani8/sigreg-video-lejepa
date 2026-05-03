"""Phase 5 tests: config composition for the 4 controlled-comparison experiments."""
from __future__ import annotations

import pytest


def test_pretrain_sigreg_p5_config_loads() -> None:
    """Hydra compose for phase5_sigreg_seed0 has correct SIGReg hyperparams."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase5_sigreg_seed0"])

    assert cfg.training.lam == pytest.approx(0.02)
    assert cfg.training.warmup_steps == 3750
    assert cfg.training.ema_decay is None
    assert cfg.trainer.max_steps == 75000
    assert cfg.trainer.precision == "16-mixed"
    assert cfg.seed == 0


def test_pretrain_ema_p5_config_loads() -> None:
    """Hydra compose for phase5_ema_seed0 has correct EMA hyperparams and target_encoder."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase5_ema_seed0"])

    assert cfg.training.lam == pytest.approx(0.0)
    assert cfg.training.ema_decay == pytest.approx(0.996)
    assert cfg.training.warmup_steps == 3750
    assert cfg.model.target_encoder._target_ == (
        "sigreg_video_lejepa.models.target_encoder.EMATargetEncoder"
    )
    assert cfg.seed == 0


def test_phase5_all_seeds_distinct() -> None:
    """All 4 Phase 5 experiment configs compose cleanly and have correct seed values."""
    from hydra import compose, initialize

    expected_seeds = {
        "phase5_sigreg_seed0": 0,
        "phase5_sigreg_seed1": 1,
        "phase5_ema_seed0": 0,
        "phase5_ema_seed1": 1,
    }

    for exp, seed in expected_seeds.items():
        with initialize(config_path="../configs", version_base="1.3"):
            cfg = compose("config", overrides=[f"+experiment={exp}"])
        assert cfg.seed == seed, f"{exp}: expected seed={seed}, got {cfg.seed}"
        assert cfg.trainer.max_steps == 75000


def test_phase5_data_and_model_are_medium() -> None:
    """Phase 5 configs use 128×128 crop and img_size."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase5_sigreg_seed0"])

    assert cfg.data.dataset.transform.crop_size == 128
    assert cfg.model.encoder.img_size == 128
    assert cfg.model.predictor.depth == 12
