import os
os.chdir('/home/ankitpani/projects/sigreg-video-lejepa')

import sys
sys.path.insert(0, 'src')

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from sigreg_video_lejepa.training.lightning_module import VideoJEPAModule

with initialize(config_path='', version_base='1.3'):
    cfg = compose('config', overrides=['+experiment=phase5_tpu_smoke'])

torch.manual_seed(20260529)

module = VideoJEPAModule(
    encoder=instantiate(cfg.model.encoder),
    predictor=instantiate(cfg.model.predictor),
    projector=instantiate(cfg.model.projector),
    sigreg_loss=instantiate(cfg.model.sigreg_loss),
    masker=instantiate(cfg.model.masker),
    target_encoder=instantiate(cfg.model.target_encoder),
    lr=cfg.training.lr,
    weight_decay=cfg.training.weight_decay,
    warmup_steps=cfg.training.get('warmup_steps'),
    lam=cfg.training.lam,
    ema_decay=cfg.training.get('ema_decay'),
)
module.eval()

torch.manual_seed(20260529)
x = torch.randn(8, 3, 8, 64, 64)

with torch.no_grad():
    total, l_pred, l_sigreg = module.compute_loss(x)

print(f'CPU SIGReg on fixed batch: {l_sigreg.item():.6f}')
print(f'CPU l_pred on fixed batch: {l_pred.item():.6f}')
print(f'CPU total on fixed batch: {total.item():.6f}')