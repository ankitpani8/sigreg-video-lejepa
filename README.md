# sigreg-video-lejepa

Applying LeJEPA's SIGReg training recipe to a V-JEPA-style video architecture, evaluated on UCF101 and Something-Something v2.

## Status

Phase 5 (SIGReg vs EMA controlled comparison, 128×128 / 75k steps) is in progress.
Phase 4 completed with weak results: SIGReg 3.44% / EMA 4.23% top-1 — both undertrained
at 64×64 / 25k steps, establishing pipeline correctness but not a meaningful signal.
Next milestone: v1.1 UCF101 results at meaningful compute scale.

### Phase 4 Results

UCF101 linear probe after 25k steps at 64×64, ViT-Tiny, split 1 (multi-clip):

| Mode   | Top-1  | Top-5  |
|--------|--------|--------|
| SIGReg | 3.44%  | 14.03% |
| EMA    | 4.23%  | 15.25% |

Both modes trained stably with no divergence. The ~0.8% gap is within run-to-run
variance; neither result meaningfully exceeds the random-features baseline (~5-8%
for untrained ViT-Tiny on UCF101). Diagnosis: compute scale insufficient.
Full analysis in [`docs/phase4_results.md`](docs/phase4_results.md).

### Phase 5 (in progress)

128×128 spatial resolution, 75k steps, predictor depth=12 (V-JEPA 2 reference),
2 seeds per mode (4 runs total). Designed to generate a clean, well-resourced
comparison between SIGReg and EMA at compute sufficient for useful representation
learning. See [`docs/design_decisions.md`](docs/design_decisions.md) for the full
rationale behind every design choice.

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
- [x] **Phase 4 — UCF101 pretraining baseline** ✅ (weak result — pipeline validated, representations undertrained at 64×64 / 25k steps)
  - [x] Phase 4 harness: HF Hub checkpointing, resume, linear probe pipeline
  - [x] 4 experiment configs: SIGReg/EMA × seed 0/1
  - [x] Kaggle training notebook (notebooks/03_kaggle_pretrain.ipynb)
- [x] **v1.0 — UCF101 pipeline complete** ✅ (end-to-end, both modes, honest result reported)
- [ ] **Phase 5 — SIGReg vs EMA controlled comparison (128×128 / 75k steps)** 🚧 in progress
  - [x] ucf101_medium configs (128×128, 2,048 tubelets, predictor depth=12)
  - [x] 4 experiment configs: phase5_{sigreg,ema}_seed{0,1}
  - [x] docs/design_decisions.md — rationale for all Phase 5 choices
  - [ ] 4 training runs on Kaggle
  - [ ] aggregate_results.py --phase 5
- [ ] **Phase 5b — TPU v5e-8 support** 🚧 in progress
  - [x] 4 TPU experiment configs: phase5_{sigreg,ema}_seed{0,1}_tpu (bf16-mixed, 8 chips)
  - [x] smoke_test_tpu model config + phase5_{gpu,tpu}_smoke experiment configs
  - [x] pretrain.py: XLA strategy guard + is_global_zero log gating
  - [x] torch_xla optional dep: `uv pip install -e ".[tpu]"`
  - [x] Notebook: ACCELERATOR variable, conditional torch_xla install, dynamic EXPERIMENT_NAME
  - [ ] TPU smoke test verified on Kaggle TPU session
  - [ ] 4 TPU training runs (GPU or TPU — whichever finishes first is the result)
- [ ] **v1.1 — UCF101 results at meaningful scale** (Phase 5 conclusion; GPU or TPU path)
- [ ] **Phase 6 — Block masking ablation on UCF101** (V-JEPA 2 improved masking; GPU or TPU)
- [ ] **v2.0 — Block masking results** (UCF101 ablation complete)
- [ ] **Phase 7+ — SSv2 scaling** (deferred until UCF101 results conclusive)
- [ ] **v3.0 — SSv2 results** (Industrial Benchmark)

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

One-time setup. When mounted in Colab at `/content/drive/MyDrive/`, the default configs in `configs/data/ucf101_small.yaml` will find the data automatically. The first training run copies data to `/content/ucf101_local/` (SSD) for fast I/O; subsequent runs reuse the cache. Phase 5 configs (`ucf101_medium`) use `/content/ucf101_local_128` — a distinct path so both caches can coexist.

## Open Research Questions

Design choices considered for this project and deferred, with brief rationale.
See [`docs/design_decisions.md`](docs/design_decisions.md) for the choices that
were made for Phase 5 and why.

- **Causal-temporal masking** — mask future frames, predict from past. Theoretically
  motivates temporal extrapolation rather than holistic reconstruction. Deferred because
  it confounds the SIGReg-vs-EMA comparison and aligns with forecasting tasks rather
  than UCF101 classification.

- **Stochastic prediction** — Gaussian-output predictor instead of deterministic point
  estimates. Theoretically captures multi-modal uncertainty in masked tubelet content.
  Deferred because it changes both the architecture and the loss formulation simultaneously,
  requiring careful disentanglement from the anti-collapse comparison.

- **Frozen-teacher / SALT-style pretraining** — replace EMA with a pre-trained frozen
  target encoder (Apple, 2025). Theoretically improves compute efficiency and removes
  co-evolution dynamics. Deferred because it requires a separate pixel-reconstruction
  pretraining stage; out of scope for v1.x.

- **VICReg or hybrid anti-collapse mechanisms** — variance-invariance-covariance
  regularization or combinations with EMA. Deferred because the SIGReg-vs-EMA comparison
  is the project's core question; mixing methods loses interpretability.

- **Larger encoder (ViT-Small, ViT-Base)** — theoretical fit improves with more data.
  Deferred because UCF101's 9,500 videos don't justify it; needs Kinetics-400 or larger
  pretraining data first.

- **Higher spatial resolution (256×256)** — approaches V-JEPA 2 reference. Deferred
  due to Kaggle T4 VRAM constraints; future TPU work.

- **Action anticipation / world-model evaluation** — evaluate on benchmarks where
  multi-future imagination matters. Deferred because UCF101 linear probe doesn't test
  this; would require different evaluation infrastructure.

## License

MIT
