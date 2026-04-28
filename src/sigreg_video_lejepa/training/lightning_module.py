from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn as nn

from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss, sigreg_video_loss


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

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, _ = batch                                        # (B, C, T, H, W)
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

        total, l_pred, l_sigreg = sigreg_video_loss(
            pred, tgt_slice, proj, self.sigreg_loss, self.hparams.lam
        )
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
        params = (
            list(self.encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.projector.parameters())
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.total_steps > 0:
            warmup = self.hparams.warmup_steps
            total = self.hparams.total_steps

            def _lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step) / max(1, warmup)
                progress = float(step - warmup) / max(1, total - warmup)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return optimizer
