from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LinearProbe(L.LightningModule):
    """Single linear layer trained on frozen encoder features.

    No hidden layers or dropout — the point is to measure representation quality,
    not classifier quality. SGD + cosine LR following V-JEPA 2 linear-probe protocol.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        num_epochs: int = 20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(embed_dim, num_classes)

        metric_kwargs = {"task": "multiclass", "num_classes": num_classes}
        self.train_top1 = torchmetrics.Accuracy(top_k=1, **metric_kwargs)
        self.train_top5 = torchmetrics.Accuracy(top_k=5, **metric_kwargs)
        self.val_top1 = torchmetrics.Accuracy(top_k=1, **metric_kwargs)
        self.val_top5 = torchmetrics.Accuracy(top_k=5, **metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_top1(logits, y)
        self.train_top5(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/top1", self.train_top1, on_step=False, on_epoch=True)
        self.log("train/top5", self.train_top5, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        self.val_top1(logits, y)
        self.val_top5(logits, y)
        self.log("val/top1", self.val_top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/top5", self.val_top5, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> tuple:
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.num_epochs
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
