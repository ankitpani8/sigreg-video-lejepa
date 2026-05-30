# sigreg-video-lejepa

Applying LeJEPA's SIGReg training recipe to a V-JEPA-style video architecture, evaluated on UCF101 and Something-Something v2.

## Status

**Phase 5 complete** SIGReg vs EMA controlled comparison at
128×128 / 75k steps on UCF101 produced the project's central finding: SIGReg
maintains ~10× higher effective embedding rank than EMA (42.7 vs 4.2 out of 192),
demonstrating SIGReg's collapse-prevention claim extends from images to video at
matched compute. Both methods achieve modest absolute linear-probe accuracy (~4-6%),
attributed to temporal shortcut learning under random tube masking.

**Phase 5c complete (TPU SPMD path validated).** A standalone SPMD training harness
runs on Kaggle TPU v5e-8 (`scripts/pretrain_tpu.py`), since Lightning's `XLAStrategy`
is incompatible with Kaggle's single-process TPU topology. Numerical equivalence
verified: SIGReg on a fixed batch matches CPU reference within 0.3%. The project now
has a working dual-accelerator pipeline (Lightning for GPU, SPMD for TPU).

**Phase 6 in progress (design locked).** Three-way regularizer comparison (SIGReg vs EMA
vs VICReg-VC) on a fixed causal+stochastic-prediction testbed. CausalTubeMasker,
StochasticVideoJEPAPredictor, and VICRegLoss implemented; all tests pass (16 new Phase 6
tests + all existing Phase 4/5 tests); smoke configs verified on CPU. Training infrastructure
ready for Kaggle GPU/TPU sessions.

### Phase 5 Results

UCF101 linear probe after 75k steps at 128×128, ViT-Tiny + predictor depth=12,
split 1 (multi-clip):

| Mode   | Steps  | Top-1  | Top-5  | Effective rank | Var in top-10 dims |
|--------|--------|--------|--------|----------------|--------------------|
| SIGReg | 75,000 | 5.87%  | 17.57% | 42.7 / 192     | 57.3%              |
| EMA    | 75,000 | 3.38%  | 11.42% | 4.2 / 192      | 92.9%              |

The **rank gap** is evident. SIGReg preserves a high-dimensional embedding
distribution (43 of 192 dims carry meaningful variance); EMA undergoes near-total
dimensional collapse despite its mechanism being designed to prevent collapse (4
effective dims; the predictor adapts to the collapsed target, avoiding training
failure while losing representational capacity).

Additionally, EMA's top singular value carries 72.7% of total variance, indicating the encoder has collapsed to nearly a single dominant direction. Re-running the linear probe at step 75,000 (vs the earlier 69,750 partial) showed top-1 accuracy decreasing from 4.04% to 3.38% — late-stage EMA training actively degraded downstream quality.

Both methods produce only modestly useful representations in absolute terms.
The interpretation is that low `l_pred` + high rank + low classification accuracy
indicates **temporal shortcut learning**: the model predicts masked tubelets by
exploiting nearby visible tubelets (adjacent video frames are near-identical) rather
than learning semantic content. Random tube masking does not enforce semantic
prediction on video as strongly as it does on images.

Full analysis: [`docs/phase5_results.md`](docs/phase5_results.md).

### Phase 4 Results (early baseline, undertrained)

UCF101 linear probe after 25k steps at 64×64, ViT-Tiny + predictor depth=6:

| Mode   | Top-1  | Top-5  |
|--------|--------|--------|
| SIGReg | 3.44%  | 14.03% |
| EMA    | 3.38%  | 11.42% |

Established pipeline correctness but did not produce a meaningful comparison —
both modes near random-features baseline due to undertraining. Retained as
historical baseline. Full analysis: [`docs/phase4_results.md`](docs/phase4_results.md).

## Motivation

[LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, Nov 2025) introduced SIGReg — a clean, principled regularizer that replaces the EMA target encoder used in earlier JEPAs. It was validated on images.

[V-JEPA](https://arxiv.org/abs/2404.08471) and [V-JEPA 2](https://arxiv.org/abs/2506.09985) extended JEPA to video, but still rely on the EMA recipe.

This project bridges the two: **can SIGReg training extend cleanly from images to video?**

## Roadmap

- [x] **Phase 0** — End-to-end pipeline on synthetic data ✅
- [x] **Phase 1** — Tubelet embedding + random tube masking ✅
- [x] **Phase 2** — UCF101 data pipeline (decord, augmentation, splits) ✅
- [x] **Phase 3** — Linear probe evaluation pipeline (extract → probe, multi-clip) ✅
- [x] **Phase 4** — UCF101 baseline at 64×64 / 25k ✅ (weak result, pipeline validated)
- [x] **v1.0** — UCF101 pipeline complete ✅
- [x] **Phase 5** — SIGReg vs EMA at 128×128 / 75k ✅ (key result: effective rank 42.7 vs 4.2)
  - [x] ucf101_medium configs (128×128, 2,048 tubelets, predictor depth=12)
  - [x] 4 experiment configs: phase5_{sigreg,ema}_seed{0,1}
  - [x] phase5_sigreg_seed0 to 75k → linear probe + rank diagnostic complete
  - [x] phase5_ema_seed0 to 69,750 → linear probe + rank diagnostic complete
  - [ ] phase5_ema_seed0 final 5,250 steps (next GPU session)
  - [ ] phase5_{sigreg,ema}_seed1 (variance estimate, time permitting)
- [x] **Phase 5b** — TPU v5e-8 support via Lightning XLAStrategy ❌ DEPRECATED
  - Multi-process `xmp.spawn` incompatible with Kaggle TPU single-process topology
  - Configs and tests retained as historical record; do not use
- [x] **Phase 5c** — TPU SPMD training path ✅ validated end-to-end
  - [x] scripts/pretrain_tpu.py with single-process SPMD harness
  - [x] bf16 forward / fp32 SIGReg with explicit all-gather for global-batch EP test
  - [x] Checkpoint format interoperable with GPU Lightning path
  - [x] Smoke test passes on Kaggle TPU v5e-8 (50 steps, finite losses, EMA updates)
  - [x] SIGReg numerical equivalence verified (TPU within 0.3% of CPU reference)
- [ ] **v1.1** — UCF101 results at meaningful scale (Phase 5 completion + seed1 replication)
- [ ] **Phase 6** — Regularizer comparison on causal/stochastic-prediction testbed 🚧
  - [x] CausalTubeMasker: C=6/G=4/T=6 deterministic mask, TPU-safe (int32 precomputed)
  - [x] StochasticVideoJEPAPredictor: Gaussian (mu, log_var), per-dim free-bits KL
  - [x] VICRegLoss: variance + covariance (VC-only; JEPA pred loss serves as invariance)
  - [x] VideoJEPAModule extended: predictor_type / regularizer_type / beta hparams
  - [x] pretrain_tpu.py: stochastic branch + VICReg all-gather (same pattern as SIGReg)
  - [x] 14 experiment configs (GPU + TPU × 3 regularizers × 2 seeds + 2 smoke)
  - [x] 16 new tests; all Phase 4/5 paths verified unchanged
  - [ ] phase6_{sigreg,ema,vicreg}_seed{0,1} training runs on Kaggle GPU (next sessions)
  - [ ] Linear probe + effective-rank diagnostic on Phase 6 representations
- [ ] **v2.0** — Phase 6 results (regularizer paper, workshop / arXiv preprint target)
- [ ] **Phase 7+** — SSv2 scaling (deferred until UCF101 results conclusive)
- [ ] **v3.0** — SSv2 results


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

Design choices considered for this project. Some are now active in Phase 6
(see `docs/design_decisions.md` for Phase 6 details when planning completes).

- **Causal-temporal masking** — used as a Phase 6 testbed component (mask future
  frames, predict from past) to break temporal shortcut learning. Not claimed as
  contribution since VJ-VCR (Dec 2024) and others use related variants on
  smaller-scale or toy datasets.

- **Stochastic prediction** — used as a Phase 6 testbed component (Gaussian-output
  predictor with bounded KL) to provide richer training signal under causal masking.
  Not claimed as contribution since Variational JEPA (Jan 2026) and Var-JEPA (Mar 2026)
  already use stochastic JEPA prediction in non-video domains.

- **VICReg as third regularizer arm** — Phase 6 variable. Tests the hypothesis that
  isotropic-Gaussian regularization (SIGReg) may suppress task-relevant cluster
  structure on video, whereas VICReg's softer variance-covariance constraint may
  preserve it.

- **Frozen-teacher / SALT-style pretraining** — replace EMA with a pre-trained
  frozen target encoder (Apple, 2025). Deferred because it requires a separate
  pixel-reconstruction pretraining stage; out of scope for v2.x.

- **Larger encoder (ViT-Small, ViT-Base)** — theoretical fit improves with more data.
  Deferred because UCF101's 9,500 videos don't justify it; needs Kinetics-400 or larger
  pretraining data first.

- **Higher spatial resolution (256×256)** — approaches V-JEPA 2 reference. Deferred
  due to T4 VRAM constraints; possible on TPU SPMD path now but compute cost is high.

- **Action anticipation / world-model evaluation** — evaluate on benchmarks where
  multi-future imagination matters. Deferred because UCF101 linear probe doesn't test
  this; would require different evaluation infrastructure.


## License

MIT
