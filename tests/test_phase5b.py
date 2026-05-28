"""Phase 5b tests: TPU v5e-8 config composition and GPU config regression."""
from __future__ import annotations

from pathlib import Path


def test_gpu_smoke_config_loads() -> None:
    """phase5_gpu_smoke composes with fp16 precision and single GPU."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase5_gpu_smoke"])

    assert cfg.trainer.precision == "16-mixed"
    assert cfg.trainer.accelerator == "gpu"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.max_steps == 50


def test_tpu_smoke_config_loads() -> None:
    """phase5_tpu_smoke composes with bf16 precision, 8 devices, xla strategy."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase5_tpu_smoke"])

    assert cfg.trainer.precision == "bf16-true"
    assert cfg.trainer.accelerator == "tpu"
    assert cfg.trainer.devices == 8
    assert cfg.trainer.strategy == "xla"
    assert cfg.trainer.max_steps == 50
    assert cfg.data.persistent_workers is False


def test_tpu_experiment_configs_load() -> None:
    """All 4 phase5_*_tpu configs compose with tpu/8/bf16/xla and correct seeds."""
    from hydra import compose, initialize

    expected = {
        "phase5_sigreg_seed0_tpu": 0,
        "phase5_sigreg_seed1_tpu": 1,
        "phase5_ema_seed0_tpu": 0,
        "phase5_ema_seed1_tpu": 1,
    }
    ema_exps = {"phase5_ema_seed0_tpu", "phase5_ema_seed1_tpu"}

    with initialize(config_path="../configs", version_base="1.3"):
        for exp, seed in expected.items():
            cfg = compose("config", overrides=[f"+experiment={exp}"])

            assert cfg.trainer.precision == "bf16-true", f"{exp}: expected bf16-true"
            assert cfg.trainer.accelerator == "tpu", f"{exp}: expected tpu"
            assert cfg.trainer.devices == 8, f"{exp}: expected 8 devices"
            assert cfg.trainer.strategy == "xla", f"{exp}: expected xla strategy"
            assert cfg.trainer.max_steps == 75000, f"{exp}: expected 75000 steps"
            assert cfg.data.persistent_workers is False, f"{exp}: expected persistent_workers=false"
            assert cfg.seed == seed, f"{exp}: expected seed={seed}, got {cfg.seed}"

            if exp in ema_exps:
                assert "EMATargetEncoder" in cfg.model.target_encoder._target_, f"{exp}: expected EMA target"


def test_gpu_experiment_configs_unchanged() -> None:
    """All phase4/phase5 GPU configs still use 16-mixed precision (regression guard).

    Excludes _tpu variants, *_smoke configs, and *_benchmark to avoid false positives.
    Uses glob so new GPU configs added later are automatically covered.
    """
    from hydra import compose, initialize

    config_dir = Path(__file__).parent.parent / "configs" / "experiment"
    gpu_configs = [
        p.stem
        for p in sorted(config_dir.glob("phase[45]_*.yaml"))
        if not any(p.stem.endswith(suffix) for suffix in ("_tpu", "_smoke", "_benchmark"))
    ]

    assert gpu_configs, "No GPU experiment configs found — check glob pattern"

    with initialize(config_path="../configs", version_base="1.3"):
        for exp in gpu_configs:
            cfg = compose("config", overrides=[f"+experiment={exp}"])
            assert cfg.trainer.precision == "16-mixed", (
                f"{exp}: expected 16-mixed but got {cfg.trainer.precision} — "
                "GPU configs must not be changed to bf16"
            )
