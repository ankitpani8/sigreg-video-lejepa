"""SPMD training entry point for Kaggle TPU v5e-8.

Single-process, 8-core SPMD via torch_xla. Drop-in replacement for pretrain.py on TPU.
The GPU Lightning path (pretrain.py) is completely unchanged.

Usage (Kaggle TPU session):
    python scripts/pretrain_tpu.py +experiment=phase5_tpu_smoke
    python scripts/pretrain_tpu.py \\
        +experiment=phase5_sigreg_seed0_tpu \\
        +run_name=phase5_sigreg_seed0_tpu \\
        +hf_repo_id=ankitpani/sigreg-video-lejepa-checkpoints \\
        +hf_token=$HF_TOKEN \\
        +resume_from_hf=true \\
        +checkpoint_every_n_steps=1000 \\
        data.dataset.data_root=<path> \\
        data.dataset.split_root=<path> \\
        data.dataset.local_cache=null

SPMD must be enabled before any XLA tensors are created — xr.use_spmd() is called
at module import time, before the Hydra decorator initialises anything.
"""
from __future__ import annotations

import torch_xla.runtime as xr  # SPMD init: must be first torch_xla import

xr.use_spmd()  # must precede xla_device() and all tensor creation

import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_xla  # noqa: E402
import torch_xla.core.xla_model as xm  # noqa: E402
import torch_xla.distributed.spmd as xs  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sigreg_video_lejepa.training.lightning_module import (  # noqa: E402
    VideoJEPAModule,
    _kl_loss,
    build_optimizer_and_scheduler,
)
from sigreg_video_lejepa.utils.checkpointing import (  # noqa: E402
    load_latest_checkpoint_from_hf,
    save_checkpoint_to_hf,
)
from sigreg_video_lejepa.utils.wandb_setup import get_or_create_run_id  # noqa: E402

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_mesh(num_devices: int) -> xs.Mesh:
    """Data-parallel mesh: shard batch across 'data' axis, replicate model on 'model'."""
    return xs.Mesh(np.arange(num_devices), (num_devices, 1), ("data", "model"))


def _all_gather_proj(proj: torch.Tensor, mesh: xs.Mesh, num_devices: int) -> torch.Tensor:
    """All-gather proj to give SIGRegLoss the full global batch.

    In SPMD data-parallel mode the input batch is sharded on the 'data' axis, so
    proj (B_local, N_ctx, proj_dim) holds 1/num_devices of the global batch per core.
    Re-annotating as fully replicated inserts an XLA all-gather collective, making
    SIGRegLoss operate on the complete global batch — necessary for a valid
    Epps-Pulley distributional test.

    When num_devices == 1 (single-chip or CPU fallback) this is a no-op.
    """
    if num_devices <= 1:
        return proj
    xs.mark_sharding(proj, mesh, (None, None, None))
    return proj


def _save_checkpoint(
    module: VideoJEPAModule,
    step: int,
    ckpt_dir: Path,
    run_name: str | None,
    hf_repo_id: str | None,
    hf_token: str | None,
) -> None:
    """Save step checkpoint via xm.save, copy to last.ckpt, and optionally push to HF."""
    step_path = ckpt_dir / f"step_{step:07d}.ckpt"
    last_path = ckpt_dir / "last.ckpt"

    # xm.save transfers XLA tensors to CPU; master_only=True is always satisfied
    # in single-process SPMD (ordinal is always 0).
    xm.save({"state_dict": module.state_dict(), "global_step": step}, str(step_path))
    shutil.copy2(step_path, last_path)
    log.info("Saved checkpoint: %s", step_path.name)

    if hf_repo_id and hf_token and run_name:
        try:
            save_checkpoint_to_hf(step_path, hf_repo_id, run_name, step, hf_token)
            log.info("Pushed step %d to HF Hub.", step)
        except Exception as exc:
            log.warning("HF push failed at step %d: %s", step, exc)


# ── Main ───────────────────────────────────────────────────────────────────────


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.get("seed", 42))

    # ── Device / mesh ──────────────────────────────────────────────────────────
    num_devices = xr.global_runtime_device_count()
    mesh = _build_mesh(num_devices)
    device = xm.xla_device()
    log.info("SPMD: %d devices, mesh %s", num_devices, mesh)

    # ── Config extraction ─────────────────────────────────────────────────────
    max_steps: int = cfg.trainer.max_steps
    ckpt_every: int = cfg.get("checkpoint_every_n_steps", 1000)
    log_every: int = cfg.get("log_every_n_steps", 50)
    run_name: str | None = cfg.get("run_name", None)
    hf_repo_id: str | None = cfg.get("hf_repo_id", None)
    hf_token: str | None = (
        cfg.get("hf_token", None)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    wandb_token: str | None = cfg.get("wandb_token", None) or os.environ.get("WANDB_API_KEY")
    wandb_project: str = cfg.get("wandb_project", "sigreg-video-lejepa")
    wandb_entity: str = cfg.get("wandb_entity", os.environ.get("WANDB_ENTITY", "ankitpani"))

    # ── Build model components ─────────────────────────────────────────────────
    encoder = instantiate(cfg.model.encoder)
    target_encoder = instantiate(cfg.model.target_encoder)
    if hasattr(target_encoder, "initialize_from"):
        target_encoder.initialize_from(encoder)
    predictor = instantiate(cfg.model.predictor)
    projector = instantiate(cfg.model.projector)
    sigreg_loss = instantiate(cfg.model.sigreg_loss)
    masker_cfg = cfg.model.get("masker")
    masker = instantiate(masker_cfg) if masker_cfg is not None else None
    vicreg_loss_cfg = cfg.model.get("vicreg_loss")
    vicreg_loss = instantiate(vicreg_loss_cfg) if vicreg_loss_cfg is not None else None

    module = VideoJEPAModule(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        projector=projector,
        sigreg_loss=sigreg_loss,
        vicreg_loss=vicreg_loss,
        masker=masker,
        total_steps=max_steps,
        **OmegaConf.to_container(cfg.training, resolve=True),
    )

    # ── Resume from HF (load on CPU before moving to device) ──────────────────
    start_step = 0
    if cfg.get("resume_from_hf", False) and hf_repo_id and hf_token:
        local_dir = Path("/tmp/hf_resume")
        local_dir.mkdir(parents=True, exist_ok=True)
        experiment = run_name or "unnamed"
        found = load_latest_checkpoint_from_hf(hf_repo_id, experiment, hf_token, local_dir)
        if found:
            sd = torch.load(found, map_location="cpu", weights_only=True)
            module.load_state_dict(sd["state_dict"])
            start_step = int(sd.get("global_step", 0))
            log.info("Resumed from HF checkpoint at step %d: %s", start_step, found)
        else:
            log.info("No HF checkpoint found — starting from scratch.")

    # Move to XLA device after CPU state-dict load (avoids device-mismatch on load_state_dict)
    module = module.to(device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(
        module,
        lr=module.hparams.lr,
        weight_decay=module.hparams.weight_decay,
        warmup_steps=module.hparams.warmup_steps,
        total_steps=max_steps,
        last_epoch=start_step - 1,  # clean resume: no scheduler.step() loop
    )

    # ── DataLoader (standard; SPMD shards via mark_sharding, not DistributedSampler) ──
    dataset = instantiate(cfg.data.dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        persistent_workers=False,  # TPU host-device handoff prefers fresh workers
    )

    # ── Checkpoint directory ───────────────────────────────────────────────────
    ckpt_dir = Path("results/checkpoints") / (run_name or "unnamed")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB init ─────────────────────────────────────────────────────────────
    wandb_run = None
    if run_name and wandb_token:
        try:
            import wandb

            wandb.login(key=wandb_token, relogin=True)
            run_id = get_or_create_run_id(wandb_project, wandb_entity, run_name, wandb_token)
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                id=run_id,
                resume="allow" if run_id else None,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:
            log.warning("WandB init failed (continuing without logging): %s", exc)

    # ── Training loop ──────────────────────────────────────────────────────────
    log.info("Starting SPMD training: steps %d → %d", start_step, max_steps)
    step = start_step

    while step < max_steps:
        for x, _ in loader:
            if step >= max_steps:
                break

            # Shard the global batch across the data axis.
            # Each core processes batch_size / num_devices samples.
            x = x.to(device)
            xs.mark_sharding(x, mesh, ("data", None, None, None, None))  # (B, C, T, H, W)

            optimizer.zero_grad()

            # CRITICAL 2: encoder/predictor forward in bf16; regularizer stats in fp32.
            # The Epps-Pulley CF test and VICReg variance/covariance are numerically
            # unstable in bf16. Smooth-L1 / MSE prediction loss is bf16-stable.
            with torch.autocast("xla", dtype=torch.bfloat16):
                pred, tgt_slice, proj, pred_aux = module._forward_representations(x)
                if module.hparams.predictor_type == "stochastic":
                    l_pred = torch.nn.functional.smooth_l1_loss(pred, tgt_slice)
                else:
                    l_pred = torch.nn.functional.mse_loss(pred, tgt_slice)

            # KL term (zero for deterministic; pred_aux has fp32 mu/log_var)
            if module.hparams.predictor_type == "stochastic":
                l_kl = _kl_loss(pred_aux["mu"].float(), pred_aux["log_var"].float())
            else:
                l_kl = torch.zeros((), device=device)

            # CRITICAL 1: all-gather proj so SIGReg / VICReg sees the full global batch.
            # Both are full-batch distributional statistics; per-shard computation
            # produces biased estimates that diverge from the GPU baseline.
            regularizer_type = module.hparams.regularizer_type
            lam = module.hparams.lam
            if regularizer_type in ("sigreg", "vicreg"):
                proj_fp32 = proj.to(torch.float32)
                proj_global = _all_gather_proj(proj_fp32, mesh, num_devices)
                if regularizer_type == "sigreg":
                    l_reg = module.sigreg_loss(proj_global)
                else:
                    l_reg = module.vicreg_loss(proj_global)
            else:  # "ema"
                l_reg = torch.zeros((), device=device)

            if module.hparams.predictor_type == "stochastic":
                total_loss = l_pred + module.hparams.beta * l_kl + lam * l_reg
            else:
                total_loss = (1.0 - lam) * l_pred + lam * l_reg

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # EMA update before sync so it is traced in the same XLA graph.
            if module.hparams.ema_decay is not None:
                module.target_encoder.update(module.encoder, module.hparams.ema_decay)

            torch_xla.sync()
            step += 1

            # ── Logging (post-sync; .item() is a host read, not a new sync) ───
            if step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(
                    "step=%d  loss=%.4f  l_pred=%.4f  l_kl=%.4f  l_reg=%.4f  lr=%.2e",
                    step,
                    total_loss.item(),
                    l_pred.item(),
                    l_kl.item(),
                    l_reg.item(),
                    lr_now,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": total_loss.item(),
                            "train/l_pred": l_pred.item(),
                            "train/l_kl": l_kl.item(),
                            "train/l_reg": l_reg.item(),
                            "train/lr": lr_now,
                            "global_step": step,
                        },
                        step=step,
                    )

            # ── Checkpointing ─────────────────────────────────────────────────
            if ckpt_every > 0 and step % ckpt_every == 0:
                _save_checkpoint(module, step, ckpt_dir, run_name, hf_repo_id, hf_token)

    # ── Final checkpoint ───────────────────────────────────────────────────────
    _save_checkpoint(module, step, ckpt_dir, run_name, hf_repo_id, hf_token)
    log.info("Training complete at step %d.", step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
