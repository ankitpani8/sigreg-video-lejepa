# sigreg-video-lejepa

Applying LeJEPA's SIGReg training recipe to a V-JEPA-style video architecture, evaluated on UCF101 and Something-Something v2.

## Status

🚧 Under active development. v1.0 (UCF101) target: in progress.

## Motivation

[LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, Nov 2025) introduced SIGReg — a clean, principled regularizer that replaces the EMA target encoder used in earlier JEPAs. It was validated on images.

[V-JEPA](https://arxiv.org/abs/2404.08471) and [V-JEPA 2](https://arxiv.org/abs/2506.09985) extended JEPA to video, but still rely on the EMA recipe.

This project bridges the two: **can SIGReg training extend cleanly from images to video?**

## Roadmap

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
- [ ] **Phase 2 — UCF101 data pipeline**
- [ ] **Phase 3 — Linear probe evaluation**
- [ ] **Phase 4 — UCF101 pretraining run** (first time λ > 0)
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

## License

MIT
