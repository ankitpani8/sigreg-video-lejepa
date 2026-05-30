# Phase 5 Results: SIGReg vs EMA on UCF101 at 128×128

## Setup

- **Encoder**: ViT-Tiny (embed_dim=192, depth=12), 3D tubelet (2×8×8)
- **Spatial / temporal**: 128×128, 16 frames, 2,048 tubelets, 75% random tube masking
- **Predictor**: V-JEPA-style transformer, depth=12 (V-JEPA 2 reference)
- **Training**: 75,000 steps, warmup 3,750, batch_size=32 (T4×2 DDP, effective global 64), fp16-mixed
- **Compute**: Kaggle T4×2, ~42 hours per run, multi-session HF Hub resume
- **Evaluation**: linear probe on UCF101 split 1 (multi-clip, 4 evenly-spaced clips), frozen mean-pooled tokens

## Results (seed 0)

| Mode   | Steps  | Top-1  | Top-5  | Effective rank | Var in top-10 dims | Top-1 singular value |
|--------|--------|--------|--------|----------------|--------------------|----------------------|
| SIGReg | 75,000 | 5.87%  | 17.57% | 42.7 / 192     | 57.3%              | ~0.26                |
| EMA    | 75,000 | 3.38%  | 11.42% | 4.2 / 192      | 92.9%              | **0.727**            |

## Key Findings

### 1. SIGReg maintains ~10× higher effective embedding rank than EMA

The headline result. SIGReg's embeddings occupy a 43-dimensional subspace of the 192-dim
embedding space; EMA's collapse to a 4-dimensional subspace. This demonstrates SIGReg's
collapse-prevention claim extends from images to video for the first time in a controlled
comparison on a transformer-based video JEPA with action-classification linear probe.

### 2. EMA's collapse is severe, concentrated, and stable

The top singular value carries **72.7%** of total variance in EMA's embeddings — meaning
nearly three-quarters of the encoder's expressive power lives in a *single* direction.
This is collapse beyond the rank metric: it's near-1-dimensional structure with low-energy
noise around it.

The effective rank of 4.2 was unchanged from step 69,750 to step 75,000 — collapse is
the local optimum the model has converged to, not a transient state.

### 3. EMA's downstream accuracy *decreases* with more training

EMA at step 69,750 achieved 4.04% top-1; the same encoder at step 75,000 dropped to
3.38%. Additional training under the EMA objective actively degraded representation
quality. This is consistent with the model deepening its collapse toward the single
dominant direction.

### 4. SIGReg's representations don't translate to high accuracy

SIGReg's 5.87% top-1 is only modestly above the EMA 3.38% and well below useful
absolute accuracy (~30% would be reasonable for ViT-Tiny on UCF101 with V-JEPA-scale
training). The combination of low `train/l_pred` (0.045) + high rank + low downstream
accuracy indicates **temporal shortcut learning**: under random tube masking, adjacent
video frames are near-identical, so the model can predict masked tubelet embeddings by
copying nearby visible tubelets without learning semantic structure.

## Interpretation

SIGReg keeps the embedding space high-dimensional and well-distributed; EMA does not.
But both methods, under random tube masking on video, fall into representations
sufficient for the SSL objective without capturing action semantics. The masking strategy
— not the regularizer — appears to be the dominant bottleneck for representation quality.

This finding motivates Phase 6: replace random tube masking with **causal masking + a
gap** (predict frames 10-15 from frames 0-5, with frames 6-9 hidden) to remove the
temporal-copy shortcut, then compare regularizers under this harder, fairer testbed.

## Caveats

- **Single seed.** Variance estimate via seed 1 of both modes is the canonical defense
  against seed-luck objections. The 10× rank gap is far beyond plausible seed variance,
  but for paper-quality claims, ≥2 seeds per mode is the floor.
- **EMA decay sensitivity untested.** Phase 5 used `ema_decay=0.996` (standard V-JEPA
  value). It's reasonable to ask whether collapse is robust to decay choice or specific
  to this value. A 25k-step run at `ema_decay=0.999` is in progress to address this; if
  collapse persists, the finding is "EMA collapses on video robustly across decay values."
  If decay=0.999 prevents collapse, the claim narrows to "EMA at standard V-JEPA decay
  collapses on video; collapse is decay-sensitive."
- **ViT-Tiny + UCF101 ceiling is ~30-40% top-1 even with V-JEPA-quality training.**
  Absolute numbers in this work should not be compared against V-JEPA's full-scale
  benchmarks; the comparison is *internal* (which method/configuration is better at
  this scale).

## Next steps (Phase 6)

Three-way regularizer comparison (SIGReg vs EMA vs VICReg-VC) on a fixed causal-stochastic
prediction testbed. Causal masking with gap removes temporal-copy shortcut; stochastic
predictor (Gaussian output head with bounded KL via free-bits) provides richer training
signal. Tests whether (a) the testbed change rescues absolute accuracy and (b) SIGReg's
rank advantage holds, narrows, or reverses against VICReg's variance-covariance penalty
on this harder task.

See `docs/design_decisions.md` Section 13 for Phase 6 design rationale.