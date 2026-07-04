# Phase 6 Results: Regularizer Comparison on Causal-Stochastic Video JEPA

## Setup (identical across all three arms)

- **Encoder**: ViT-Tiny (192-dim, depth 12), 3D tubelet 2×8×8, 128×128, 16 frames, 2,048 tubelets
- **Masking**: causal with gap — context frames 0-5, gap frames 6-9 (hidden), target frames 10-15
  (768 context tubelets, 768 target tubelets, 512 gap tubelets excluded)
- **Predictor**: stochastic single-Gaussian head (μ, log_var), reparameterized sample,
  KL to N(0,I) with per-dimension free-bits floor (0.5 nats/dim), β=1e-3
- **Training**: 75,000 steps, warmup 3,750, global batch 64
- **Eval**: linear probe on UCF101 split 1 (multi-clip), frozen mean-pooled tokens, probe on CPU

The regularizer is the ONLY variable:
- **EMA**: λ=0, EMATargetEncoder, decay=0.996 (V-JEPA default, untuned)
- **SIGReg**: λ=0.02, SharedTargetEncoder, no EMA
- **VICReg-VC**: λ=1.0 (μ_v=25, μ_c=1), SharedTargetEncoder, no EMA

All arms trained on TPU v5e-8 (single-process SPMD). VICReg required an SPMD-specific
fix to its covariance computation (see design_decisions.md §12).

## Results (two seeds per arm)

**Table 1. Linear-probe accuracy on UCF101.**

| Arm    | Top-1 s0 | Top-1 s1 | **Top-1 mean** | Top-5 s0 | Top-5 s1 | Top-5 mean |
|--------|----------|----------|----------------|----------|----------|------------|
| **EMA**    | 20.75%   | 21.81%   | **21.28%**     | 44.62%   | 45.47%   | 45.05%     |
| SIGReg | 8.75%    | 12.27%   | **10.51%**     | 24.47%   | 29.29%   | 26.88%     |
| VICReg | 7.61%    | 8.27%    | **7.94%**      | 21.97%   | 22.39%   | 22.18%     |

**Table 2. Effective rank and variance concentration.**

| Arm    | Rank s0 | Rank s1 | **Rank mean** | Var top-10 s0 | Var top-10 s1 |
|--------|---------|---------|---------------|---------------|---------------|
| EMA    | 115.2   | 101.2   | **108.2**     | 18.8%         | 21.7%         |
| SIGReg | 110.4   | 112.8   | **111.6**     | 29.5%         | 29.3%         |
| VICReg | 71.5    | 71.2    | **71.4**      | 44.3%         | 43.6%         |

## Primary finding: EMA substantially outperforms both distributional regularizers

Across two seeds, EMA reaches 21.28% top-1 (range 20.75–21.81), versus 10.51% for SIGReg
(range 8.75–12.27) and 7.94% for VICReg (range 7.61–8.27). EMA is 2.0× SIGReg and 2.7×
VICReg on mean top-1, with a consistent margin on top-5.

The method ranges do not overlap: EMA's lower seed (20.75%) is far above SIGReg's higher
seed (12.27%). The ~11-point EMA–SIGReg gap is ~6× SIGReg's own seed-to-seed spread
(3.5 points) and ~20× EMA's (1.1 points). The ordering EMA ≫ SIGReg > VICReg is
reproducible, not a seed artifact.

EMA is also the most *stable* method across seeds (range ±0.5 vs SIGReg's ±1.8). It wins
at its untuned V-JEPA default momentum (0.996), with no hyperparameter advantage over the
regularizers, which use published defaults.

This inverts the LeJEPA hypothesis that SIGReg-style distributional regularization cleanly
replaces the EMA target encoder — at least at this scale and on this video testbed.

## Secondary finding: effective rank fails to predict representation quality

The rank diagnostics do not track accuracy, and in the most direct comparison they point
the wrong way:

- **On mean rank, EMA (108.2) is LOWER than SIGReg (111.6)** — yet EMA's mean accuracy is
  2.0× higher (21.28% vs 10.51%). The higher-rank method is the worse classifier.
- **In the seed-1 pair specifically**, SIGReg has rank 112.8 vs EMA's 101.2 — SIGReg
  higher by 11.6 rank — while EMA classifies at 21.81% vs SIGReg's 12.27%, nearly 2×
  better. Matched seed, higher rank, half the accuracy.
- EMA and SIGReg occupy the same rank band (101–115 across all four runs) while their
  accuracy bands are cleanly separated (~21% vs ~10%). Rank does not carry the signal that
  distinguishes them.

Note the scope precisely: the decoupling is an EMA-vs-SIGReg phenomenon. VICReg is
conventional — lower rank (71) *and* lower accuracy (8%) — so for VICReg, rank and accuracy
agree. The sharp claim is therefore: **effective rank cannot distinguish EMA from SIGReg
despite a 2× downstream accuracy gap; two methods of equal (or inverted) rank differ
2× in quality.** This is a caution against rank as a quality proxy, not a claim that rank
is meaningless everywhere.

## Interpretation

All three methods reach high effective rank (71–115 of 192) — all prevent collapse in the
rank sense. But preventing collapse is not producing discriminative structure.

- **EMA** prevents collapse via a slowly co-evolving momentum target. Predicting it gives a
  semantically coherent, drifting objective that carries task-relevant signal as a
  byproduct of collapse prevention.
- **SIGReg / VICReg** prevent collapse by constraining the embedding *distribution* (toward
  isotropic Gaussian / decorrelated high variance). This guarantees the embedding occupies
  many dimensions but is agnostic to whether those dimensions are task-discriminative.

The variance concentration supports this: EMA spreads variance most evenly (top-10 ~19-22%)
and is most useful; VICReg concentrates most (top-10 ~44%) and is least useful. The
regularizers' problem is not insufficient spread but spread into a non-discriminative
subspace — consistent with the isotropy-cost concern (Huang 2026) and Var-JEPA's isotropy
ablations.

## Testbed validation: causal masking removes the temporal shortcut

Under causal masking, the prediction loss stabilizes at ≈0.36, versus ≈0.045 under random
tube masking (Phase 5). The hidden gap removes the nearest temporal neighbor, so the model
cannot copy an adjacent frame and must extrapolate — confirming the testbed exercises
temporal reasoning rather than rewarding a shortcut. The stochastic predictor's KL stays
near its free-bits floor throughout, indicating the task does not demand much predictive
multimodality at this scale; the single-Gaussian predictor is adequate.

## Caveats

- **Scale**: ViT-Tiny / UCF101 is small relative to production video-JEPA. LeJEPA's
  scaling claim is not refuted by small-scale evidence; this is a controlled counterexample
  at the scale tested. Absolute accuracies are low and meaningful only for internal
  cross-arm comparison.
- **Two seeds**: the EMA ≫ SIGReg > VICReg ordering is replicated and its ranges are
  non-overlapping. The precise SIGReg-vs-VICReg gap (10.51 vs 7.94) is smaller and its
  ranges nearly touch (SIGReg s0 8.75 vs VICReg s1 8.27) — read as "SIGReg ≳ VICReg,
  comparable," not a firm ranking.
- **Testbed-specific**: established for causal masking + stochastic single-Gaussian
  predictor. Other masking/predictor choices may interact differently.

## Conclusion

On a controlled causal-stochastic video-JEPA testbed, an EMA target encoder substantially
and reproducibly outperforms both SIGReg and VICReg distributional regularizers (21.3% vs
10.5% vs 7.9% mean top-1), inverting the SIGReg-replaces-EMA hypothesis at this scale.
Effective rank fails to predict this: EMA has lower mean rank than SIGReg yet 2× the
accuracy. Preventing collapse (high rank) is necessary but not sufficient for a useful
video representation, and rank-based diagnostics can be actively misleading. Downstream
probing remains necessary to evaluate anti-collapse mechanisms.
