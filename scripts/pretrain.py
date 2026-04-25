"""Pre-training entry point for Video JEPA + SIGReg.

Usage:
    python scripts/pretrain.py                                   # default: 100 steps, cpu
    python scripts/pretrain.py +experiment=smoke_test_phase0     # Phase 0: 2 steps, no masking
    python scripts/pretrain.py +experiment=smoke_test_phase1     # Phase 1: 2 steps, tube masking
"""
from __future__ import annotations

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.data.dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )

    encoder = instantiate(cfg.model.encoder)
    target_encoder = instantiate(cfg.model.target_encoder)
    predictor = instantiate(cfg.model.predictor)
    projector = instantiate(cfg.model.projector)
    sigreg_loss = instantiate(cfg.model.sigreg_loss)
    masker_cfg = cfg.model.get("masker")
    masker = instantiate(masker_cfg) if masker_cfg is not None else None

    module = VideoJEPAModule(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        projector=projector,
        sigreg_loss=sigreg_loss,
        masker=masker,
        **OmegaConf.to_container(cfg.training, resolve=True),
    )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(module, loader)


if __name__ == "__main__":
    main()
