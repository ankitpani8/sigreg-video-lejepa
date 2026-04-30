"""Pre-training entry point for Video JEPA + SIGReg.

Usage (local, synthetic data):
    python scripts/pretrain.py                                       # 100 steps, cpu
    python scripts/pretrain.py +experiment=smoke_test_phase0         # 2 steps, no masking
    python scripts/pretrain.py +experiment=smoke_test_phase1         # 2 steps, tube masking

Usage (Kaggle, UCF101):
    python scripts/pretrain.py \\
        +experiment=phase4_sigreg_seed0 \\
        +run_name=phase4_sigreg_seed0 \\
        +hf_repo_id=ankitpani/sigreg-video-lejepa-checkpoints \\
        +hf_token=$HF_TOKEN \\
        +resume_from_hf=true \\
        +checkpoint_every_n_steps=1000 \\
        data.dataset.data_root=<path> \\
        data.dataset.split_root=<path> \\
        data.dataset.local_cache=null

Benchmark (measure sec/step before committing to full runs):
    python scripts/pretrain.py +experiment=phase4_benchmark \\
        data.dataset.data_root=<path> data.dataset.split_root=<path> data.dataset.local_cache=null
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule
from sigreg_video_lejepa.utils.checkpointing import (
    load_latest_checkpoint_from_hf,
    save_checkpoint_to_hf,
)
from sigreg_video_lejepa.utils.wandb_setup import get_or_create_run_id

log = logging.getLogger(__name__)


class HFPushCallback(L.Callback):
    """Pushes each new ModelCheckpoint file to HF Hub immediately after it is written."""

    def __init__(self, repo_id: str, experiment_name: str, token: str) -> None:
        self.repo_id = repo_id
        self.experiment_name = experiment_name
        self.token = token
        self._last_pushed: str = ""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                path = cb.last_model_path
                if path and path != self._last_pushed and Path(path).exists():
                    step = trainer.global_step
                    try:
                        save_checkpoint_to_hf(
                            Path(path), self.repo_id, self.experiment_name, step, self.token
                        )
                        log.info("Pushed checkpoint step %d to HF Hub.", step)
                        self._last_pushed = path
                    except Exception as exc:
                        log.warning("HF push failed at step %d: %s", step, exc)
                break


class BenchmarkTimingCallback(L.Callback):
    """Measures average sec/step over steps 101+ and prints a summary at training end.

    The first `warmup_steps` steps are skipped because dataloader worker
    initialisation inflates their wall-clock time.
    """

    def __init__(self, warmup_steps: int = 100) -> None:
        self.warmup_steps = warmup_steps
        self._start: float = 0.0
        self._times: list[float] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        elapsed = time.perf_counter() - self._start
        if trainer.global_step > self.warmup_steps:
            self._times.append(elapsed)

    def on_train_end(self, trainer, pl_module) -> None:
        if not self._times:
            print("\n=== BENCHMARK: no steps measured past warmup ===\n")
            return
        avg = sum(self._times) / len(self._times)
        print(
            f"\n=== BENCHMARK RESULT ===\n"
            f"Steps measured : {len(self._times)}\n"
            f"Avg sec/step   : {avg:.3f}\n"
            f"Est. hrs/25k   : {avg * 25_000 / 3600:.1f}\n"
            f"========================\n"
        )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.get("seed", 42), workers=True)

    dataset = instantiate(cfg.data.dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        persistent_workers=cfg.data.get("persistent_workers", False),
    )

    encoder = instantiate(cfg.model.encoder)
    target_encoder = instantiate(cfg.model.target_encoder)
    if hasattr(target_encoder, "initialize_from"):
        target_encoder.initialize_from(encoder)
    predictor = instantiate(cfg.model.predictor)
    projector = instantiate(cfg.model.projector)
    sigreg_loss = instantiate(cfg.model.sigreg_loss)
    masker_cfg = cfg.model.get("masker")
    masker = instantiate(masker_cfg) if masker_cfg is not None else None

    total_steps = cfg.trainer.get("max_steps") or 0
    module = VideoJEPAModule(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        projector=projector,
        sigreg_loss=sigreg_loss,
        masker=masker,
        total_steps=total_steps,
        **OmegaConf.to_container(cfg.training, resolve=True),
    )

    # Optional HF Hub + WandB settings (all off by default for local/smoke runs)
    run_name: str | None = cfg.get("run_name", None)
    hf_repo_id: str | None = cfg.get("hf_repo_id", None)
    hf_token: str | None = (
        cfg.get("hf_token", None)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    wandb_token: str | None = cfg.get("wandb_token", None) or os.environ.get("WANDB_API_KEY")
    wandb_project: str = cfg.get("wandb_project", "sigreg-video-lejepa")
    ckpt_every: int = cfg.get("checkpoint_every_n_steps", 0)

    # Build callbacks
    callbacks: list[L.Callback] = []

    if ckpt_every > 0:
        ckpt_cb = ModelCheckpoint(
            dirpath="checkpoints",
            every_n_train_steps=ckpt_every,
            save_top_k=3,
            monitor="step",
            mode="max",
            save_last=True,
            filename="{step:07d}",
        )
        callbacks.append(ckpt_cb)
        if hf_repo_id and hf_token:
            experiment = run_name or "unnamed"
            callbacks.append(HFPushCallback(hf_repo_id, experiment, hf_token))
    
    # Always add ModelCheckpoint so HFPushCallback has files to push.
    # Without this, Lightning disables its default checkpointing when callbacks list is non-empty.
    if not cfg.get("benchmark", False):
        ckpt_dir = Path("results/checkpoints") / (run_name or "unnamed")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="step_{step:07d}",
                every_n_train_steps=cfg.get("checkpoint_every_n_steps", 1000),
                save_top_k=-1, # Save all step-wise checkpoints; HF push handles persistence
                save_last=True,
                save_on_train_epoch_end=False,
            )
        )
    
    if cfg.get("benchmark", False):
        callbacks.append(BenchmarkTimingCallback(warmup_steps=100))

    # Build trainer kwargs — handle logger separately so WandbLogger can be injected
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    raw_logger = trainer_kwargs.pop("logger", True)

    if run_name and raw_logger:
        from lightning.pytorch.loggers import WandbLogger

        run_id = get_or_create_run_id(wandb_project, "ankitpani", run_name, wandb_token)
        trainer_kwargs["logger"] = WandbLogger(
            project=wandb_project,
            name=run_name,
            id=run_id,
            resume="allow" if run_id else None,
        )
    else:
        trainer_kwargs["logger"] = raw_logger

    trainer = L.Trainer(callbacks=callbacks or None, **trainer_kwargs)

    # Optionally resume from the latest HF Hub checkpoint
    resume_ckpt: str | None = None
    if cfg.get("resume_from_hf", False) and hf_repo_id and hf_token:
        local_dir = Path("/tmp/hf_resume")
        local_dir.mkdir(parents=True, exist_ok=True)
        experiment = run_name or "unnamed"
        found = load_latest_checkpoint_from_hf(hf_repo_id, experiment, hf_token, local_dir)
        if found:
            log.info("Resuming from HF checkpoint: %s", found)
            resume_ckpt = str(found)
        else:
            log.info("No HF checkpoint found — starting from scratch.")

    trainer.fit(module, loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
