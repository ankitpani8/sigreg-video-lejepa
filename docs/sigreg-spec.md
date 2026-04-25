# SIGReg Loss Specification

**Source paper:** Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics," arXiv 2511.08544, Nov 2025.
**Reference implementation:** https://github.com/rbalestr-lab/lejepa

This document is the authoritative specification for `training/sigreg_loss.py`. Any implementation
deviation from this spec is a bug — open an issue and update this doc rather than patching silently.

---

## 1. The Problem SIGReg Solves

Standard JEPA objectives can collapse: the encoder learns to output constant or low-rank
representations that trivially minimize prediction loss. Prior work prevented this via
stop-gradients, teacher-student momentum encoders, or explicit whitening. SIGReg prevents
collapse without any of these heuristics by adding a differentiable distributional penalty
that forces encoder outputs to be isotropic Gaussian. Theorem 4 of the paper proves bounded
gradients; Theorem 6 proves the minibatch bias vanishes with N.

---

## 2. The Cramér-Wold Reduction (Why 1D Projections Suffice)

Testing whether a D-dimensional distribution is multivariate Gaussian is expensive and
statistically unreliable in high dimensions. SIGReg exploits **Lemma 3 (Hyperspherical
Cramér-Wold)** from the paper: a distribution on ℝᴰ is standard Gaussian if and only if
its pushforward under every unit-vector projection **a** ∈ Sᴰ⁻¹ is standard Gaussian on ℝ.

This reduces the multivariate test to many independent 1D tests. Each 1D test is:

```
H₀(a): P_θ^(a) = 𝒩(0,1)    vs.    H₁(a): P_θ^(a) ≠ 𝒩(0,1)
```

where P_θ^(a) = pushforward of the embedding distribution along direction **a**.

In practice we sample M random unit-vector directions A = {a₁, ..., aₘ} and average the
test statistics. This is the "sliced" or "random projection" approximation:

```
SIGReg(z) = (1/M) Σ_{m=1}^{M} T({aₘᵀ zₙ}_{n=1}^{N})
```

where T is a 1D Gaussianity test (see §3) and z ∈ ℝᴺˣᴰ are the encoder embeddings.

---

## 3. The 1D Test: Epps-Pulley via Characteristic Function

The paper recommends the **Epps-Pulley** statistic (characteristic function distance) over
KS, Anderson-Darling, or moment tests. Rationale: bounded gradients and curvature regardless
of input distribution; O(N) complexity; numerically stable in float32.

### 3.1 Characteristic Function Basics

The **empirical characteristic function** (ECF) of samples {x₁, ..., xₙ} ⊂ ℝ at frequency t:

```
φ̂(t) = (1/N) Σⱼ exp(it·xⱼ) = (1/N) Σⱼ [cos(t·xⱼ) + i·sin(t·xⱼ)]
```

The target is the **standard Gaussian** characteristic function:

```
φ(t) = exp(−t²/2)     (real-valued; imaginary part is zero by symmetry of 𝒩(0,1))
```

### 3.2 The Epps-Pulley Statistic

```
EP(z) = N · ∫ |φ̂(t) − φ(t)|² · w(t) dt
```

where w(t) = exp(−t²/2) is a Gaussian integration window (same function as φ for 𝒩(0,1)).

**Symmetry trick (used in reference code):** Since 𝒩(0,1) is symmetric, the imaginary
part of the ECF integrand is an odd function and integrates to zero. We only need to
integrate over t ≥ 0 and double the weights:

```
EP(z) = N · ∫₀^{t_max} [(cos_mean(t) − φ(t))² + sin_mean(t)²] · w(t) · 2 dt
```

where cos_mean(t) = (1/N) Σⱼ cos(t·xⱼ) and sin_mean(t) = (1/N) Σⱼ sin(t·xⱼ).

Half-weight at t=0 (trapezoidal endpoint correction), full 2·dt elsewhere.

### 3.3 Numerical Integration

Integration method: **trapezoidal rule** on [0, t_max].

| Parameter    | Default | Notes                                              |
|--------------|---------|----------------------------------------------------|
| t_max        | 3.0     | Paper tests [−1,1], [−3,3], [−5,5]; all stable    |
| knots        | 17      | Integration points (must be odd for trapezoidal)   |
| dt           | t_max / (knots − 1) = 3/16 ≈ 0.1875             |

Weights tensor (shape: knots):
- w[0] = dt · φ(0) = dt · 1.0
- w[1:-1] = 2·dt · φ(tᵢ)
- w[-1] = dt · φ(t_max)

These are precomputed once at construction and registered as buffers.

### 3.4 Reference Implementation (from MINIMAL.md)

```python
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)   # w(t)·φ(t) precomputed

    def forward(self, proj):
        # proj: (V, N, D) — V views, N batch samples, D embedding dim
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))                     # unit-norm columns
        x_t = (proj @ A).unsqueeze(-1) * self.t            # (V, N, M, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() \
            + x_t.sin().mean(-3).square()                   # (V, M, knots)
        statistic = (err @ self.weights) * proj.size(-2)    # N scaling; (V, M)
        return statistic.mean()                             # scalar
```

Note: `.mean(-3)` averages over the N samples dimension. The N factor in `* proj.size(-2)`
matches the EP formula's leading N.

---

## 4. Total Training Loss

```
L = (1 − λ) · L_pred + λ · L_SIGReg
```

### 4.1 Prediction Loss (Masked MSE)

We use **V-JEPA-style masked prediction**, not LeJEPA's multi-view invariance.
The context encoder sees only unmasked patches; the predictor is given context tokens plus
positional mask tokens and must predict the target encoder's output at masked positions.
The target encoder is always detached — gradients must not flow into it regardless of mode
(SharedTargetEncoder or EMATargetEncoder).

```
L_pred = (1 / |masked_tokens|) Σ_masked ‖predictor(context_tokens, mask_pos) − sg(target_encoder(masked_patches))‖²
```

where `sg(·)` denotes stop-gradient (`.detach()` in PyTorch).

**For reference — LeJEPA's image invariance loss (not used here):**
In the original LeJEPA code (no masking, image SSL), the prediction term is a multi-view
invariance loss: `(proj.mean(0) - proj).square().mean()`, i.e., each view's projection is
penalized for deviating from the mean projection across views. We record this for completeness;
our implementation does not use it.

### 4.2 SIGReg Applied to: Which Embeddings (Resolved)

**Decision:** SIGReg is applied to a dedicated **MLP projector head** on top of the context
encoder. The target encoder is detached, so gradients from L_SIGReg would have nowhere to go
if applied to target encoder outputs. The predictor head and projector head are separate;
they share only the context encoder backbone.

Architecture:

```
video frames
    │
    ▼
[Context Encoder]  ──────────────────────────────┐
    │                                             │
    ▼                                             ▼
[Predictor Head]                          [Projector Head]
(masked → target positions)               (MLP: D→2048→2048→proj_dim)
    │                                             │
    ▼                                             ▼
L_pred = MSE vs sg(target_encoder)         L_SIGReg = EP test
```

**Projector architecture** (matching LeJEPA reference):
- Input dim: context encoder output dim D (e.g. 384 for ViT-Small)
- Hidden layers: two layers of width 2048, each followed by BatchNorm1d + GELU
- Output dim: `proj_dim` (default 128; configurable)
- No normalization on output layer

**Which tokens:** SIGReg is computed on the projector output for **all context (unmasked)
tokens**, pooled across the batch. For a batch of N clips each with T_ctx unmasked tokens,
the EP statistic sees N × T_ctx samples. This keeps the Epps-Pulley statistic well-populated
(the O(1/N) bias shrinks as more tokens are used). Masked positions are excluded — the context
encoder never sees them.

---

## 5. Hyperparameters

| Parameter          | Symbol | Default | Stable range | Notes                                           |
|--------------------|--------|---------|-------------|--------------------------------------------------|
| Loss weight        | λ      | 0.02    | 0.01–0.2    | Paper tests 0.01, 0.02, 0.05, 0.1               |
| Num projections    | M      | 256     | 64–1024+    | 256 in MINIMAL; 1000 in ablation scripts         |
| Integration points | K      | 17      | 5–41        | Table 1a: all values work; 17 is sweet spot      |
| Integration max    | t_max  | 3.0     | 1–5         | Paper tests [−1,1], [−3,3], [−5,5]; all stable  |
| Batch size         | N      | ≥128    | ≥32 usable  | Bias is O(1/N); EP stat less reliable at N<32    |
| Num views          | V      | 4–8     | 2+          | More views = more samples for the EP test        |

SIGReg has **one primary hyperparameter** (λ). All others have stable defaults.

---

## 6. Implementation Details and Gotchas

### 6.1 Projections are resampled every step

`A` is sampled fresh at each `forward()` call using the current RNG state. The reference
production code seeds with `global_step` (deterministic per-step, synchronized across DDP
ranks). The MINIMAL code just calls `torch.randn(...)` without seeding. For Phase 0 (no
DDP), unseeded is fine. For multi-GPU training, synchronize seeds so all ranks use the
same projection matrix.

### 6.2 Gradient flows through embeddings, not through A

`A` is sampled inside `torch.no_grad()` in the production `SlicingUnivariateTest` class
(but not in MINIMAL). In MINIMAL it's sampled normally — gradient will technically flow
through it, but it's functionally a `no_grad` operation since A is freshly sampled and
not in any parameter group. Either approach is correct; wrapping in `no_grad` is cleaner.

### 6.3 No explicit standardization of inputs

SIGReg pushes embeddings toward 𝒩(0,1) implicitly — you do not need to pre-normalize.
If embeddings have mean µ ≠ 0 or variance σ² ≠ 1, the EP statistic will be large and
gradients will push toward 𝒩(0,1). This is the intended behavior.

### 6.4 Numerical stability

Bounded gradient guarantee (Theorem 4): |∂EP/∂zᵢ| ≤ 4σ²/N. No gradient clipping needed.
The exp(−t²/2) window decays the integrand at large t, preventing blowup. float32 is safe.

### 6.5 DDP synchronization

In multi-GPU training, cos_mean and sin_mean must be all-reduced across ranks so that the
EP statistic uses N = global_batch_size, not per-rank batch size. Phase 0 is single-GPU,
so skip this. Add when implementing distributed training.

### 6.6 Phase 0 smoke test caveats

Phase 0 sets λ=0.0. The SIGReg term computes (verifying the forward pass is correct and
the statistic is a finite scalar) but contributes nothing to the gradient. This is intentional:
Phase 0 tests pipeline plumbing, not learning. λ becomes nonzero starting Phase 4
(UCF101 pretraining), at which point it should be tuned from the default of 0.02.

---

## 7. Open Questions

- [ ] **λ schedule:** Should λ be constant or warmed up from 0? The reference code uses
      constant λ throughout. A warmup from 0 might help if L_pred dominates early training
      and the representations aren't yet spread enough for the EP test to be meaningful.
      Defer to Phase 4 (UCF101 pretraining); start with constant λ=0.02 and ablate if
      training is unstable.

---

## 8. Verification Checklist

When reviewing `training/sigreg_loss.py` against this spec:

- [ ] `t` is linspace(0, t_max, knots) — starts at 0, not negative
- [ ] weights are 2·dt everywhere except dt at endpoints (trapezoidal symmetry trick)
- [ ] `weights` buffer = weights · φ(t) (precomputed product)
- [ ] Final statistic is multiplied by N (batch size dimension)
- [ ] A is L2-normalized per column (unit direction vectors)
- [ ] Return value is a scalar (`.mean()` over all slices and views)
- [ ] λ=0.0 produces zero SIGReg contribution to total loss
- [ ] Statistic is finite (not NaN/Inf) for random inputs
