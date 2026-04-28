"""Linear probe training on pre-extracted encoder features.

Loads feature tensors written by extract_features.py, trains a single linear
layer, and reports top-1 / top-5 accuracy on the test split.

Run extract_features.py first, then this script:

    python scripts/extract_features.py +experiment=ucf101_linprobe
    python scripts/linear_probe.py     +experiment=ucf101_linprobe
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from sigreg_video_lejepa.evaluation.linear_probe import LinearProbe

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="linprobe_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    features_path = Path(cfg.evaluation.features_dir)
    if not features_path.is_absolute():
        features_path = Path(get_original_cwd()) / features_path

    train_features = torch.load(features_path / "train_features.pt", weights_only=True)
    train_labels = torch.load(features_path / "train_labels.pt", weights_only=True)
    test_features = torch.load(features_path / "test_features.pt", weights_only=True)
    test_labels = torch.load(features_path / "test_labels.pt", weights_only=True)

    embed_dim = train_features.shape[1]
    logger.info(
        "Features: train=%s  test=%s  embed_dim=%d",
        tuple(train_features.shape),
        tuple(test_features.shape),
        embed_dim,
    )

    eval_cfg = cfg.evaluation
    num_epochs = cfg.trainer.max_epochs

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=eval_cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    probe = LinearProbe(
        embed_dim=embed_dim,
        num_classes=eval_cfg.num_classes,
        lr=eval_cfg.lr,
        momentum=eval_cfg.momentum,
        weight_decay=eval_cfg.weight_decay,
        num_epochs=num_epochs,
    )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(probe, train_loader, test_loader)
    results = trainer.test(probe, test_loader, verbose=False)

    top1 = results[0].get("val/top1", float("nan"))
    top5 = results[0].get("val/top5", float("nan"))
    print(f"\nLinear probe results — top-1: {top1:.4f}  top-5: {top5:.4f}")


if __name__ == "__main__":
    main()
