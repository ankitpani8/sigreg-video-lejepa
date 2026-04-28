# sigreg-video-lejepa

Applying LeJEPA's SIGReg training recipe to a V-JEPA-style video architecture, evaluated on UCF101 and Something-Something v2.

## Status

🚧 Under active development. v1.0 (UCF101) target: in progress.

## Motivation

[LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, Nov 2025) introduced SIGReg — a clean, principled regularizer that replaces the EMA target encoder used in earlier JEPAs. It was validated on images.

[V-JEPA](https://arxiv.org/abs/2404.08471) and [V-JEPA 2](https://arxiv.org/abs/2506.09985) extended JEPA to video, but still rely on the EMA recipe.

This project bridges the two: **can SIGReg training extend cleanly from images to video?**

## Roadmap

- [x] **Phase 0 — End-to-end pipeline on synthetic data** ✅
  - [x] Synthetic video dataset
  - [x] V-JEPA-style ViT encoder (frame-as-batch patchification)
  - [x] Target encoder with Shared and EMA modes
  - [x] Predictor (shallow Transformer)
  - [x] SIGReg projector head
  - [x] SIGReg loss (Epps-Pulley + Cramér-Wold projections)
  - [x] PyTorch Lightning training module
  - [x] Hydra config system
  - [x] pytest suite (11 tests, including subprocess integration test)
- [x] **Phase 1 — Tubelet embedding + masking** ✅
  - [x] 3D Conv tubelet embedding (replaces frame-as-batch patchification)
  - [x] Random tube masking (shared per-step, V-JEPA style)
  - [x] Masked-encoding code path with token_indices
  - [x] Phase 0 + Phase 1 smoke configs both passing
- [x] **Phase 2 — UCF101 data pipeline** ✅
  - [x] BaseVideoDataset (decord, frame sampling, short-video looping, FPS scan)
  - [x] UCF101Dataset (split parsing, strict validation, Drive→SSD cache)
  - [x] UCF101Transform (V-JEPA 2 augmentations: scale=(0.3,1.0), no color jitter)
  - [x] ensure_cached utility (idempotent Drive→SSD copy)
  - [x] ucf101_small config (64×64, 512 tubelets, ViT-Tiny, predictor depth=6)
  - [x] ucf101_vjepa2_match config (256×256, 2048 tubelets — for scale-up)
  - [x] ucf101_dryrun experiment (10 steps, accelerator=auto)
  - [x] pytest suite: 8 unit tests + 1 @slow integration test at 512-tube scale
- [x] **Phase 3 — Linear probe evaluation** ✅
  - [x] UCF101EvalTransform (center-crop, no augmentation)
  - [x] Multi-clip eval path in BaseVideoDataset (4 evenly-spaced clips → (4,C,T,H,W))
  - [x] FeatureExtractor (mean-pool tokens, multi-clip averaging)
  - [x] LinearProbe (SGD + cosine LR, top-1/top-5 via torchmetrics)
  - [x] extract_features.py + linear_probe.py (Hydra, resumable extraction)
  - [x] ucf101_linprobe experiment config (random-init encoder for Phase 3 testing)
  - [x] pytest suite: 8 tests including subprocess end-to-end
  - [x] Kaggle notebook: notebooks/02_kaggle_linprobe.ipynb
- [ ] **Phase 4 — UCF101 pretraining run** (first time λ > 0)
  - [x] Phase 4 harness ready
- [ ] **v1.0 — UCF101 results released** (preliminary)
- [ ] **Phase 5–7 — SSv2 scaling**
- [ ] **v2.0 — SSv2 results released** (LinkedIn launch)

## Setup

```bash
git clone https://github.com/ankitpani8/sigreg-video-lejepa.git
cd sigreg-video-lejepa
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Dataset

Download UCF101 once from [CRCV](https://www.crcv.ucf.edu/data/UCF101.php) or the HuggingFace mirror (`quchenyuan/UCF101-ZIP`). Extract and place at:

- `MyDrive/datasets/ucf101/UCF-101/` — 101 class folders, 13,320 `.avi` files
- `MyDrive/datasets/ucf101/ucfTrainTestlist/` — split files (`classInd.txt`, `trainlist01.txt`, `testlist01.txt`)

One-time setup. When mounted in Colab at `/content/drive/MyDrive/`, the default configs in `configs/data/ucf101_small.yaml` will find the data automatically. The first training run copies data to `/content/ucf101_local/` (SSD) for fast I/O; subsequent runs reuse the cache.

## License

@ankitpani8

