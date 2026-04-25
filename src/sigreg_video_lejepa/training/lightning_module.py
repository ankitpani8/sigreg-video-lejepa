from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import lightning as L

from sigreg_video_lejepa.training.sigreg_loss import SIGRegLoss, sigreg_video_loss


class VideoJEPAModule(L.LightningModule):
    """Lightning module wiring encoder, target encoder, predictor, projector, and losses.

    Two heads share the context encoder:
      - predictor head → L_pred (masked MSE vs stop-gradient target)
      - projector head → L_SIGReg (Epps-Pulley Gaussianity penalty)

    Phase 0: lam=0.0, no actual masking (all tokens are context, 4-token proxy target).
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_encoder: Any,            # SharedTargetEncoder or EMATargetEncoder (duck-typed)
        predictor: nn.Module,
        projector: nn.Module,
        sigreg_loss: SIGRegLoss,
        lam: float = 0.0,
        ema_decay: float = 0.996,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "target_encoder", "predictor",
                                          "projector", "sigreg_loss"])
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.projector = projector
        self.sigreg_loss = sigreg_loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, _ = batch                                        # (B, C, T, H, W)
        B = x.size(0)

        ctx = self.encoder(x)                               # (B, T*N, D)
        tgt = self.target_encoder.encode(self.encoder, x)  # (B, T*N, D) — detached

        # Phase 0 proxy: use first 4 tokens as the "masked" target positions
        N_mask = 4
        mask_tokens = self.predictor.mask_token.expand(B, N_mask, -1)
        pred = self.predictor(ctx, mask_tokens)             # (B, N_mask, D)
        tgt_slice = tgt[:, :N_mask, :]                     # (B, N_mask, D)

        proj = self.projector(ctx)                          # (B, T*N, proj_dim)

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
        self.target_encoder.update(self.encoder, self.hparams.ema_decay)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = (
            list(self.encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.projector.parameters())
        )
        return torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
