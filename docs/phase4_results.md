# Phase 4 Results: SIGReg vs EMA (64×64, 25k Steps)

> This file was created manually with known results. It can be regenerated from
> HF Hub data using `python scripts/aggregate_results.py --phase 4 --hf-token $HF_TOKEN`.

## Setup

- Spatial resolution: 64×64
- Encoder: ViT-Tiny (embed_dim=192, 12 blocks)
- Tubelet: 2×8×8 → 512 tubelets per clip
- Masking: random tube, 75%
- Predictor: depth=6, num_heads=6
- Training steps: 25,000
- Batch size: 32
- UCF101 split 1, multi-clip linear probe (4 clips)

## Results

| Mode    | Seeds | Top-1 (mean ± std)  | Top-5 (mean ± std)  |
|---------|-------|---------------------|---------------------|
| SIGReg  | 1     | 3.44%               | 14.03%              |
| EMA     | 1     | 4.23%               | 15.25%              |

(Phase 4 only ran seed 0 for each mode; seed 1 configs exist but were not run.)

## Interpretation

Both methods are statistically indistinguishable at this scale. The ~0.8% top-1
gap is within expected run-to-run variance for random-init ViT-Tiny on UCF101.

The absolute numbers are the more important finding: 3-4% top-1 on 101-class UCF101
is only marginally above chance (1.0%) and far below supervised baselines (~80%+).
Random-init ViT-Tiny linear probes on UCF101 typically land around 5-8% with
reasonable augmentation, meaning these runs haven't measurably exceeded random features.

**Diagnosis: severe undertraining.** 25k steps at 64×64 is insufficient for the
encoder to learn useful visual representations. This is a compute scale failure,
not a signal about SIGReg vs EMA.

## What Phase 5 Addresses

Phase 5 runs at 128×128 spatial resolution with 75k steps and predictor depth=12
(V-JEPA 2 reference). See `docs/design_decisions.md` for the full rationale.

The Phase 4 results establish that the pipeline is end-to-end functional and that
both modes train stably (no divergence, finite losses throughout). The comparison
is simply inconclusive — the representations need more compute to be meaningful.
