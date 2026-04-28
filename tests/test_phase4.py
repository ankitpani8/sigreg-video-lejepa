"""Phase 4 tests: config composition, checkpoint utilities, and resume logic."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------

def test_pretrain_sigreg_config_loads() -> None:
    """Hydra compose for phase4_sigreg_seed0 succeeds and has expected values."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase4_sigreg_seed0"])

    assert cfg.training.lam == pytest.approx(0.02)
    assert cfg.training.warmup_steps == 1250
    assert cfg.training.ema_decay is None
    assert cfg.trainer.max_steps == 25000
    assert cfg.trainer.precision == "16-mixed"
    assert cfg.seed == 0


def test_pretrain_ema_config_loads() -> None:
    """Hydra compose for phase4_ema_seed0 succeeds and target_encoder is EMATargetEncoder."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("config", overrides=["+experiment=phase4_ema_seed0"])

    assert cfg.training.lam == pytest.approx(0.0)
    assert cfg.training.ema_decay == pytest.approx(0.996)
    assert cfg.training.warmup_steps == 1250
    assert cfg.model.target_encoder._target_ == (
        "sigreg_video_lejepa.models.target_encoder.EMATargetEncoder"
    )
    assert cfg.seed == 0


# ---------------------------------------------------------------------------
# HF Hub checkpoint utilities (mocked — no real token needed)
# ---------------------------------------------------------------------------

def test_checkpoint_to_hf_roundtrip(tmp_path: Path) -> None:
    """save_checkpoint_to_hf calls HfApi.upload_file with the correct path_in_repo."""
    from sigreg_video_lejepa.utils.checkpointing import save_checkpoint_to_hf

    ckpt = tmp_path / "step_0001000.ckpt"
    ckpt.write_bytes(b"dummy checkpoint content")

    with patch("sigreg_video_lejepa.utils.checkpointing.HfApi") as MockApi:
        instance = MockApi.return_value
        instance.create_repo.return_value = None
        instance.upload_file.return_value = None

        save_checkpoint_to_hf(ckpt, "user/repo", "phase4_sigreg_seed0", 1000, "fake_token")

        instance.create_repo.assert_called_once_with(
            "user/repo", repo_type="model", exist_ok=True, token="fake_token"
        )
        instance.upload_file.assert_called_once()
        call_kwargs = instance.upload_file.call_args.kwargs
        assert call_kwargs["path_in_repo"] == "checkpoints/phase4_sigreg_seed0/step_0001000.ckpt"
        assert call_kwargs["repo_id"] == "user/repo"


def test_resume_from_hf_logic(tmp_path: Path) -> None:
    """load_latest_checkpoint_from_hf picks the highest-step checkpoint."""
    from sigreg_video_lejepa.utils.checkpointing import (
        list_checkpoints_on_hf,
        load_latest_checkpoint_from_hf,
    )

    fake_files = [
        "checkpoints/phase4_sigreg_seed0/step_0001000.ckpt",
        "checkpoints/phase4_sigreg_seed0/step_0002000.ckpt",
        "checkpoints/phase4_sigreg_seed0/step_0003000.ckpt",
        "results/phase4_sigreg_seed0/linprobe.json",  # non-checkpoint file, must be ignored
        "checkpoints/phase4_ema_seed0/step_0001000.ckpt",  # different experiment, must be ignored
    ]
    fake_local = tmp_path / "checkpoints" / "phase4_sigreg_seed0" / "step_0003000.ckpt"
    fake_local.parent.mkdir(parents=True)
    fake_local.write_bytes(b"ckpt")

    with patch("sigreg_video_lejepa.utils.checkpointing.HfApi") as MockApi, \
         patch("sigreg_video_lejepa.utils.checkpointing.hf_hub_download") as mock_dl:

        MockApi.return_value.list_repo_files.return_value = fake_files
        mock_dl.return_value = str(fake_local)

        steps = list_checkpoints_on_hf("user/repo", "phase4_sigreg_seed0", "token")
        assert steps == [1000, 2000, 3000]

        result = load_latest_checkpoint_from_hf(
            "user/repo", "phase4_sigreg_seed0", "token", tmp_path
        )
        assert result is not None
        call_kwargs = mock_dl.call_args.kwargs
        assert "step_0003000.ckpt" in call_kwargs["filename"]


def test_list_checkpoints_returns_empty_on_missing_repo(tmp_path: Path) -> None:
    """list_checkpoints_on_hf returns [] when the repo doesn't exist (no crash)."""
    from sigreg_video_lejepa.utils.checkpointing import list_checkpoints_on_hf

    with patch("sigreg_video_lejepa.utils.checkpointing.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.side_effect = Exception("404 repo not found")

        steps = list_checkpoints_on_hf("user/nonexistent", "exp", "token")
        assert steps == []
