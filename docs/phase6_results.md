# Phase 6 Results: Regularizer Comparison on Causal-Stochastic Video JEPA

## Setup

Fixed testbed (same for all arms)

- **Encoder**: ViT-Tiny (192-dim, depth 12), 3D tubelet 2×8×8, 128×128, 16 frames, 2,048 tubelets
- **Masking**: causal with gap — context frames 0-5, gap frames 6-9 (hidden), target frames 10-15.
  768 context tubelets, 768 target tubelets, 512 gap tubelets excluded.
- **Predictor**: stochastic, single-Gaussian output head (μ, log_var), reparameterized sample,
  KL toward N(0,I) with per-dimension free-bits floor at 0.5 nats/dim, β=1e-3
- **Training**: 75,000 steps, warmup 3,750, batch 64 global (TPU SPMD v5e-8), bf16 forward / fp32 regularizer
- **Eval**: linear probe on UCF101 split 1 (multi-clip), frozen mean-pooled tokens

The regularizer is the only variable across arms:
- **SIGReg**: λ=0.02, SharedTargetEncoder, no EMA
- **EMA**: λ=0, EMATargetEncoder decay=0.996 (V-JEPA default, reference baseline)
- **VICReg-VC**: λ=1.0 (μ_v=25, μ_c=1), SharedTargetEncoder, no EMA

## Results

| Arm    | Seed | Steps  | Top-1  | Top-5   | Effective rank | Var in top-10 dims |
|--------|------|--------|--------|---------|----------------|--------------------|
| SIGReg | 0    | 75,000 | 8.75%  | 24.47%  | 110.4 / 192    | 29.5%              |
| EMA    | 0    | —      | pending| pending | pending        | pending            |
| VICReg | 0    | —      | pending| pending | pending        | pending            |

## SIGReg seed0 — observations

### Causal masking broke the temporal shortcut
Phase 5 SIGReg (random tube masking) drove train l_pred to 0.045 — the predictor copied
nearby visible tubelets, learning easy-but-uninformative features. Phase 6 SIGReg
(causal masking + 4-frame gap) held l_pred at ~0.36 through training: the hidden gap
removes the nearest-neighbor to copy, forcing genuine temporal extrapolation. The premise
behind the causal-masking testbed is empirically confirmed.

### Rank exploded; accuracy improved partially
Effective rank rose from 42.7 (Phase 5 SIGReg) to 110.4 — over half the embedding
dimensions now carry meaningful variance, variance-in-top-10 dropped from 57.3% to 29.5%.
The representation is the highest-rank, least-collapsed of any run in the project.
Top-1 accuracy improved 5.87% → 8.75%. So the harder testbed produced both a higher-rank
AND a more useful representation than Phase 5 SIGReg (clean within-regularizer comparison).

### Stochasticity barely engaged
Train l_kl settled near the free-bits floor (~96-100 = D × 0.5), indicating the predictor
used little predictive variance beyond the minimum the free-bits clamp enforces. For this
testbed (6 context frames, 6 target frames, 4-frame gap), the future is determined enough
that the unimodal Gaussian rarely needs to spread. This is early evidence that the
single-Gaussian choice was adequate at this scale, and a data point for the deferred
MoG question (Phase 7): if futures aren't multimodal here, MoG's extra capacity wouldn't help.

## Open question: rank and accuracy appear decoupled

Cross-referencing the Phase 5 decay sweep (a separate sub-study):

| Config (cross-phase, NOT a clean comparison) | Effective rank | Top-1 |
|----------------------------------------------|----------------|-------|
| Phase 6 SIGReg (causal+stochastic)           | 110.4          | 8.75% |
| Phase 5 EMA decay=0.999 (random+determ., 25k)| 19.6           | 10.89%|

The highest-rank representation is NOT the highest-accuracy one. SIGReg maximizes effective
rank but a moderate-rank tuned EMA achieves higher linear-probe accuracy. This suggests
isotropic-Gaussian regularization prevents collapse but does not, by itself, produce the
most discriminative representation — consistent with the multimodal-averaging concern raised
by Huang (2026) and the isotropy-vs-task-structure tension noted in Var-JEPA's ablations.

**Critical caveat**: this cross-phase comparison confounds TWO variables (regularizer AND
testbed). It cannot attribute the accuracy difference to the regularizer. The clean test is
the WITHIN-Phase-6 three-way comparison (SIGReg vs EMA vs VICReg, all on causal+stochastic),
pending the EMA and VICReg arms. The rank-accuracy decoupling hypothesis stands or falls on
that clean comparison, not on this confounded cross-phase one.

## Caveats

- Single seed. The rank difference (110 vs 20) far exceeds plausible seed variance, but the
  ~2-point accuracy gaps may not. Seed-1 runs deferred pending the within-Phase-6 comparison.
- EMA arm uses the V-JEPA default decay (0.996) as a reference baseline at standard settings,
  not a tuned competitor. The decay sweep is a separate Phase 5 sub-study. SIGReg and VICReg
  use published defaults. No method is selectively tuned within the Phase 6 comparison.
- ViT-Tiny + UCF101 ceiling remains ~30-40% top-1; all absolute numbers are low and should be
  read only for internal (cross-arm) comparison, not against full-scale V-JEPA benchmarks.

## Next steps

1. Phase 6 EMA arm (decay=0.996) → linear probe + rank
2. Phase 6 VICReg arm → linear probe + rank
3. The clean three-way comparison resolves whether SIGReg's rank advantage translates to
   accuracy, or whether the rank-accuracy decoupling holds within a controlled testbed.
4. Seed-1 replication of whichever finding emerges, for variance.