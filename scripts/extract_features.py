"""Feature extraction for linear probe evaluation.

Runs a frozen encoder over UCF101 train and test splits, mean-pools tokens,
and saves feature tensors to disk. Resumable: skips splits whose .pt files
already exist.

Usage:
    python scripts/extract_features.py +experiment=ucf101_linprobe
    python scripts/extract_features.py +experiment=ucf101_linprobe checkpoint_path=/path/to.ckpt
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from sigreg_video_lejepa.evaluation.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    """Return the best available device: CUDA > XLA > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415

        return xm.xla_device()
    except ImportError:
        return torch.device("cpu")


@hydra.main(config_path="../configs", config_name="linprobe_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    features_path = Path(cfg.evaluation.features_dir)
    if not features_path.is_absolute():
        features_path = Path(get_original_cwd()) / features_path
    features_path.mkdir(parents=True, exist_ok=True)

    encoder = instantiate(cfg.model.encoder)

    if cfg.checkpoint_path is not None:
        ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
        encoder_sd = {
            k.removeprefix("encoder."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("encoder.")
        }
        encoder.load_state_dict(encoder_sd)
        logger.info("Loaded encoder weights from %s.", cfg.checkpoint_path)
    else:
        logger.info("Using randomly-initialized encoder (no checkpoint_path set).")

    device = _select_device()
    logger.info("Using device: %s", device)
    extractor = FeatureExtractor(encoder, device)

    # train split: single-clip, center-crop (no augmentation).
    # test split:  multi-clip (4 clips), center-crop — handled by dataset eval path.
    for split in ("train", "test"):
        feat_path = features_path / f"{split}_features.pt"
        label_path = features_path / f"{split}_labels.pt"

        if feat_path.exists() and label_path.exists():
            logger.info("Skipping %s split — features already exist at %s.", split, feat_path)
            continue

        # Instantiate dataset with split override; both splits use UCF101EvalTransform
        # (no augmentation) so features are deterministic.
        dataset = instantiate(cfg.data.dataset, split=split)
        loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        )

        logger.info("Extracting %s features (%d videos)...", split, len(dataset))
        features, labels = extractor.extract(loader)
        torch.save(features, feat_path)
        torch.save(labels, label_path)
        logger.info(
            "Saved: %s shape=%s, %s shape=%s",
            feat_path.name, tuple(features.shape),
            label_path.name, tuple(labels.shape),
        )

    logger.info("Feature extraction complete. Output dir: %s", features_path)


if __name__ == "__main__":
    main()
