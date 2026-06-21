# Phase 6 Results: Regularizer Comparison on Causal-Stochastic Video JEPA

## Setup (identical across all three arms)

- **Encoder**: ViT-Tiny (192-dim, depth 12), 3D tubelet 2×8×8, 128×128, 16 frames, 2,048 tubelets
- **Masking**: causal with gap — context frames 0-5, gap frames 6-9 (hidden), target frames 10-15
  (768 context tubelets, 768 target tubelets, 512 gap tubelets excluded)
- **Predictor**: stochastic single-Gaussian head (μ, log_var), reparameterized sample,
  KL to N(0,I) with per-dimension free-bits floor (0.5 nats/dim), β=1e-3
- **Training**: 75,000 steps, warmup 3,750, global batch 64
- **Eval**: linear probe on UCF101 split 1 (multi-clip), frozen mean-pooled tokens

The regularizer is the ONLY variable:
- **EMA**: λ=0, EMATargetEncoder, decay=0.996 (V-JEPA default, untuned)
- **SIGReg**: λ=0.02, SharedTargetEncoder, no EMA
- **VICReg-VC**: λ=1.0 (μ_v=25, μ_c=1), SharedTargetEncoder, no EMA

SIGReg and EMA arms trained on TPU v5e-8 (SPMD); VICReg on TPU after the SPMD
covariance fix (see design_decisions.md). All probed identically on CPU.

## Results (seed 0)

| Arm    | Top-1   | Top-5   | Effective rank | Var in top-10 dims |
|--------|---------|---------|----------------|--------------------|
| **EMA**    | **20.75%** | **44.62%** | 115.2 / 192    | 18.8%              |
| SIGReg | 8.75%   | 24.47%  | 110.4 / 192    | 29.5%              |
| VICReg | 7.61%   | 21.97%  | 71.5 / 192     | 44.3%              |

## Primary finding: EMA substantially outperforms both distributional regularizers

On this causal-stochastic video testbed, EMA's co-evolving target encoder beats both
SIGReg and VICReg by a wide margin: 20.75% top-1 vs 8.75% (SIGReg) and 7.61% (VICReg) —
a 2.4–2.7× gap. The separation holds on top-5 (44.6% vs 24.5% / 22.0%). This is the
largest and clearest effect measured in the project, and it inverts the original
hypothesis that SIGReg (per LeJEPA, Balestriero & LeCun 2025) would serve as a clean
replacement for EMA on video.

Notably, EMA wins at its **untuned default** decay (0.996) — it was given no
hyperparameter advantage over the regularizers (which also use published defaults).

## Secondary finding: effective rank is a poor proxy for video representation quality

The rank–accuracy relationship is not merely weak — it is actively misleading:

- **EMA and SIGReg have near-identical effective rank** (115.2 vs 110.4) yet EMA
  classifies **2.4× better** (20.75% vs 8.75%). Same rank, vastly different quality.
- **VICReg has the lowest rank** (71.5) and the lowest accuracy (7.61%), but the
  rank gap to SIGReg (110.4) does not predict the small accuracy gap (8.75 vs 7.61).

Two representations with the same effective rank can differ 2.4× in downstream task
performance. Effective rank measures how many dimensions carry variance; it says nothing
about whether that variance is task-relevant. This is a direct caution against
rank-maximization (RankMe-style objectives) as a target for video SSL.

## Interpretation

All three methods achieve high effective rank (72–115 of 192) — i.e., all three prevent
representational collapse. But preventing collapse is not the same as producing
discriminative structure:

- **EMA** prevents collapse via a slowly co-evolving prediction target. The target is a
  momentum average of the network's own representations, so the model predicts
  semantically coherent, drifting targets — collapse-prevention that carries a
  task-meaningful learning signal.
- **SIGReg / VICReg** prevent collapse by constraining the embedding *distribution*
  (toward isotropic Gaussian / toward decorrelated high variance). This guarantees high
  rank but is agnostic to task semantics — variance is spread into dimensions that need
  not be discriminative.

The variance distribution supports this: EMA spreads variance most evenly (top-10 dims
18.8%) and is most useful; VICReg concentrates most (44.3%) and is least useful. The
issue for the regularizers is not insufficient spread but *spread into the wrong
subspace*. This is consistent with the isotropy-cost concern noted by Huang (2026) and
the isotropy degradation in Var-JEPA's ablations: isotropic-Gaussian regularization buys
rank at the cost of discriminative structure that EMA's co-evolution preserves.

## Caveats

- **Single seed per arm.** EMA's 12-point lead over the regularizers far exceeds the
  ~3-point seed-to-seed variance observed in Phase 5, so the EMA-wins conclusion is very
  likely robust to seed noise. The exact numbers require seed-1 replication before
  publication. The SIGReg-vs-VICReg gap (8.75 vs 7.61, 1.1 points) is within plausible
  seed variance and should be read as "comparable," not "SIGReg beats VICReg."
- **ViT-Tiny + UCF101** ceiling is ~30-40% top-1; absolute numbers are low and meaningful
  only for internal (cross-arm) comparison, not against full-scale V-JEPA benchmarks.
  EMA's 20.75% is the closest any run has come to a useful representation.
- The finding is specific to this testbed (causal masking + stochastic single-Gaussian
  predictor, UCF101, ViT-Tiny). Whether EMA's advantage holds at larger scale or under
  different masking is open.

## Revised project narrative

The project began testing whether SIGReg cleanly replaces EMA for collapse prevention on
video. The honest result is the inverse: on real video with a transformer and an
action-classification probe, **EMA's co-evolving target substantially outperforms both
SIGReg and VICReg**, and **effective rank fails to predict this** — the methods are
rank-comparable but accuracy-divergent. The contribution is therefore (1) a negative
result for distributional regularizers as EMA replacements on video JEPA, and (2) a
positive methodological caution that effective rank is not a reliable quality proxy for
video representations.

## Next steps

1. Seed-1 replication of all three Phase 6 arms — confirm the EMA ≫ SIGReg ≈ VICReg
   ordering holds. Given the gap size this is confirmation, not discovery, but it is
   required for publication-grade claims.
2. (Optional) Probe-quality controls: confirm the gap is not a probe-optimization
   artifact (e.g., longer probe training, probe LR sweep) — though all arms used the
   identical probe protocol, so relative comparison is fair.
3. Write-up targeting workshop / arXiv: "EMA vs distributional regularizers on video
   JEPA, and the failure of effective rank as a quality proxy."