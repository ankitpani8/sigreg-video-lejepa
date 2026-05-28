from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn as nn

from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss, sigreg_video_loss


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
    target_encoder and sigreg_loss buffers are intentionally excluded from optimizer.

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

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)
        return optimizer, scheduler

    return optimizer, None


class VideoJEPAModule(L.LightningModule):
    """Lightning module wiring encoder, target encoder, predictor, projector, and losses.

    Two heads share the context encoder:
      - predictor head → L_pred (masked MSE vs stop-gradient target)
      - projector head → L_SIGReg (Epps-Pulley Gaussianity penalty)

    When masker is provided (Phase 1+), the context encoder sees only unmasked tubes and
    the predictor reconstructs target-encoder output at masked positions.
    When masker is None (Phase 0 compat), a 4-token proxy target is used instead.
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_encoder: Any,            # SharedTargetEncoder or EMATargetEncoder (duck-typed)
        predictor: nn.Module,
        projector: nn.Module,
        sigreg_loss: SIGRegLoss,
        masker: Any = None,             # TubeMasker or None (duck-typed: __call__(N, device))
        lam: float = 0.0,
        ema_decay: float | None = 0.996,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 0,
        total_steps: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "target_encoder", "predictor",
                                          "projector", "sigreg_loss", "masker"])
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.projector = projector
        self.sigreg_loss = sigreg_loss
        self.masker = masker

    def _forward_representations(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode x and produce (pred, tgt_slice, proj) without computing the loss.

        Separated from compute_loss so pretrain_tpu.py can all-gather proj across
        SPMD cores before calling SIGRegLoss on the full global batch.
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
        pred = self.predictor(ctx, mask_tokens)             # (B, N_tgt, D)
        proj = self.projector(ctx)                          # (B, N_ctx, proj_dim)
        return pred, tgt_slice, proj

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward + loss for the GPU Lightning path. Returns (total, l_pred, l_sigreg)."""
        pred, tgt_slice, proj = self._forward_representations(x)
        return sigreg_video_loss(pred, tgt_slice, proj, self.sigreg_loss, self.hparams.lam)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        total, l_pred, l_sigreg = self.compute_loss(x)
        self.log_dict(
            {"train/loss": total, "train/l_pred": l_pred, "train/l_sigreg": l_sigreg},
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
