from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss


def _kl_loss(mu: torch.Tensor, log_var: torch.Tensor, free_bits: float = 0.5) -> torch.Tensor:
    """KL divergence KL(q(z|x) || N(0,I)) with per-dimension free-bits floor.

    Per-dimension clamping (Kingma et al. IAF; Variational JEPA / VJ-VCR) allows
    individual dimensions to rise above the floor independently. The floor prevents
    stochastic collapse to deterministic at init without blocking gradient on all dims.

    At init (mu≈0, log_var≈0 → kl_per_dim≈0): each dim is clamped to free_bits,
    so kl_per_token = free_bits × D, kl_loss = free_bits × D (unchanged by mean).

    Returns: scalar mean over B × N_tgt of (sum-over-D of clamped per-dim KL).
    """
    kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())  # (B, N, D)
    kl_per_dim = kl_per_dim.clamp(min=free_bits)                      # per-dim floor
    kl_per_token = kl_per_dim.sum(-1)                                 # (B, N)
    return kl_per_token.mean()


def build_optimizer_and_scheduler(
    module: VideoJEPAModule,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
    last_epoch: int = -1,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR | None]:
    """Build AdamW + cosine-warmup LR scheduler for encoder, predictor, and projector.

    Single source of truth for both GPU (Lightning) and TPU (SPMD) training paths.
    target_encoder, sigreg_loss, and vicreg_loss buffers/params are excluded from optimizer.

    Args:
        last_epoch: pass start_step - 1 when resuming to skip the scheduler.step() loop hack.
    """
    params = (
        list(module.encoder.parameters())
        + list(module.predictor.parameters())
        + list(module.projector.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if total_steps > 0:
        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # PyTorch >= 2.4 requires 'initial_lr' in param_groups when last_epoch >= 0;
        # it is set automatically only when last_epoch == -1. Set it explicitly here
        # so resume (last_epoch = start_step - 1) does not raise KeyError.
        if last_epoch >= 0:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)
        return optimizer, scheduler

    return optimizer, None


class VideoJEPAModule(L.LightningModule):
    """Lightning module wiring encoder, target encoder, predictor, projector, and losses.

    Phase 5 and earlier: deterministic predictor, sigreg or ema regularizer.
    Phase 6: stochastic predictor (Gaussian reparameterization), three-way regularizer
             comparison (sigreg / ema / vicreg). Controlled via predictor_type and
             regularizer_type hparams; defaults preserve Phase 5 behavior exactly.

    Two duck-typed interfaces (unchanged from Phase 5):
      - target_encoder: SharedTargetEncoder or EMATargetEncoder — both expose .encode()
      - masker: TubeMasker, CausalTubeMasker, or None — all expose __call__(N, device)
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_encoder: Any,            # SharedTargetEncoder or EMATargetEncoder (duck-typed)
        predictor: nn.Module,
        projector: nn.Module,
        sigreg_loss: SIGRegLoss,
        masker: Any = None,             # TubeMasker/CausalTubeMasker or None
        vicreg_loss: nn.Module | None = None,
        lam: float = 0.0,
        ema_decay: float | None = 0.996,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 0,
        total_steps: int = 0,
        regularizer_type: str = "sigreg",    # "sigreg" | "vicreg" | "ema"
        predictor_type: str = "deterministic",  # "deterministic" | "stochastic"
        beta: float = 0.0,                   # KL weight; 0 means no KL contribution
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "target_encoder", "predictor",
                                          "projector", "sigreg_loss", "masker", "vicreg_loss"])
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.projector = projector
        self.sigreg_loss = sigreg_loss
        self.vicreg_loss = vicreg_loss
        self.masker = masker

    def _forward_representations(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
        """Encode x and produce (pred, tgt_slice, proj, pred_aux).

        pred_aux is None for deterministic predictors; for stochastic predictors it is
        {"mu": ..., "log_var": ...} needed to compute the KL term.

        Separated from compute_loss so pretrain_tpu.py can all-gather proj across
        SPMD cores before calling SIGRegLoss/VICRegLoss on the full global batch.
        """
        B = x.size(0)

        if self.masker is not None:
            ctx_idx, tgt_idx = self.masker(self.encoder.num_tubelets, device=x.device)
            ctx = self.encoder(x, token_indices=ctx_idx)        # (B, N_ctx, D)
            tgt = self.target_encoder.encode(self.encoder, x)  # (B, N, D) — all tubes, detached
            tgt_slice = tgt[:, tgt_idx, :]                     # (B, N_tgt, D)
        else:
            # Phase 0 compat: no masking, first 4 tokens as proxy target
            ctx = self.encoder(x)
            tgt = self.target_encoder.encode(self.encoder, x)
            tgt_slice = tgt[:, :4, :]

        N_tgt = tgt_slice.size(1)
        mask_tokens = self.predictor.mask_token.expand(B, N_tgt, -1)
        raw = self.predictor(ctx, mask_tokens)

        if isinstance(raw, dict):  # stochastic predictor returns dict
            pred = raw["sample"]
            pred_aux = raw
        else:
            pred = raw
            pred_aux = None

        proj = self.projector(ctx)                              # (B, N_ctx, proj_dim)
        return pred, tgt_slice, proj, pred_aux

    def compute_loss(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward + loss for the GPU Lightning path.

        Returns (total, l_pred, l_kl, l_reg).
        l_kl is zero for deterministic predictor_type (Phase 5 backward-compat).
        l_reg is zero for regularizer_type="ema".

        Loss formula:
          deterministic (Phase 5): (1 - lam) * l_pred + lam * l_reg
          stochastic    (Phase 6): l_pred + beta * l_kl + lam * l_reg
        """
        pred, tgt_slice, proj, pred_aux = self._forward_representations(x)

        if self.hparams.predictor_type == "stochastic":
            l_pred = F.smooth_l1_loss(pred, tgt_slice)
            l_kl = _kl_loss(pred_aux["mu"], pred_aux["log_var"])
        else:
            l_pred = F.mse_loss(pred, tgt_slice)
            l_kl = torch.zeros((), device=pred.device)

        if self.hparams.regularizer_type == "sigreg":
            l_reg = self.sigreg_loss(proj)
        elif self.hparams.regularizer_type == "vicreg":
            l_reg = self.vicreg_loss(proj)
        else:  # "ema"
            l_reg = torch.zeros((), device=pred.device)

        if self.hparams.predictor_type == "stochastic":
            total = l_pred + self.hparams.beta * l_kl + self.hparams.lam * l_reg
        else:
            total = (1.0 - self.hparams.lam) * l_pred + self.hparams.lam * l_reg
        return total, l_pred, l_kl, l_reg

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        total, l_pred, l_kl, l_reg = self.compute_loss(x)
        self.log_dict(
            {
                "train/loss": total,
                "train/l_pred": l_pred,
                "train/l_kl": l_kl,
                "train/l_reg": l_reg,
            },
            on_step=True,
            prog_bar=True,
        )
        return total

    def on_after_backward(self) -> None:
        if self.hparams.ema_decay is not None:
            self.target_encoder.update(self.encoder, self.hparams.ema_decay)

    def configure_optimizers(self):
        optimizer, scheduler = build_optimizer_and_scheduler(
            self,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            warmup_steps=self.hparams.warmup_steps,
            total_steps=self.hparams.total_steps,
        )
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return optimizer
