# Design Decisions: Phase 5 Controlled Comparison

This document records the design choices made for Phase 5 of sigreg-video-lejepa:
the controlled SIGReg vs EMA comparison at 128×128 / 75k steps on UCF101.
It is intended to be readable as a standalone reference for paper writing and
future reproduction.

For research questions that were considered and deferred, see the
[Open Research Questions](../README.md#open-research-questions) section of the README.

---

## 1. Spatial Resolution: 128×128

**Chosen over 64×64 (Phase 4) and 256×256 (V-JEPA 2 reference).**

Phase 4 at 64×64 produced representations too weak to distinguish from random noise
(linear probe ~4% top-1 on 101-class UCF101). The spatial resolution was identified as
a primary contributor: 64×64 clips lose most discriminative texture and motion detail.

256×256 is V-JEPA 2's reference resolution but requires ~A100-class VRAM for ViT-Tiny
at batch 32 with 2,048 tubelets (the T4 has 16GB; 256×256 ViT-Tiny at batch 32 is
borderline or OOM). It also multiplies training wall-time 4×.

128×128 is the midpoint: sufficient for UCF101 action categories (motion, rough shape),
fits comfortably in T4 16GB at batch 32, and gives 4× more spatial tokens than Phase 4
without the 16× token blowup of 256×256.

---

## 2. Training Duration: 75,000 Steps

**Chosen over Phase 4's 25,000 steps and V-JEPA 2's ~252,000 iteration reference.**

V-JEPA-style SSL on video converges slowly. Phase 4's 25k steps were selected as a
rapid iteration budget — not as a convergence estimate. The resulting representations
were far below useful quality (see `docs/phase4_results.md`), consistent with
undertraining rather than a method failure.

75k steps = 3× Phase 4. Based on V-JEPA-style scaling observations, useful feature
learning (above random baseline by a comfortable margin) typically requires at least
O(50k) steps at ViT-Tiny scale on a medium-size dataset. 75k is a conservative estimate
that fits within two Kaggle GPU sessions (~30-40 GPU hours per run on T4).

V-JEPA 2's 252k iterations would be the gold standard, but with 4 runs (2 methods ×
2 seeds) this would require ~300-400 GPU hours — exceeding the project's compute budget
for a v1.x prototype. 75k is the compute-budget compromise.

---

## 3. Encoder Architecture: ViT-Tiny

**Chosen over ViT-Small and larger variants.**

UCF101's training split has ~9,500 videos. SSL encoders typically need ≥O(100k) images
(or their video equivalent) to benefit from larger architectures without overfitting in
the linear probe. ViT-Small at 384-dim would have ~3× more parameters with no capacity
benefit at this dataset scale.

ViT-Tiny (embed_dim=192, 12 transformer blocks) is the standard SSL choice for
medium-small datasets. It fits comfortably in T4 16GB at 128×128 / batch 32 / 2,048
tubelets. Upgrading to ViT-Small would require either halving batch size (destabilizing
the SIGReg EP statistic, which is sensitive to effective batch size) or moving to a
larger GPU.

---

## 4. Tubelet Size: 2×8×8 (Unchanged from Phase 4)

**Same as Phase 4 and consistent with V-JEPA 2's non-overlapping tubelet design.**

2×8×8 at 128×128 / 16 frames gives N = (16/2)×(128/8)×(128/8) = 2,048 tubelets —
4× more than Phase 4's 512. The temporal stride (t_patch=2) ensures each tubelet spans
multiple frames, capturing local motion. The spatial patch size (8×8) at 128×128 gives
16 patches per axis, matching V-JEPA 2's spatial granularity at their 16×16 patches
over 256×256.

Changing tubelet size between Phase 4 and Phase 5 would conflate the resolution change
with a token-budget change. Keeping it fixed isolates the effect of resolution.

---

## 5. Masking Strategy: Random Tube Masking at 75%

**Chosen over causal-temporal masking, block masking (Phase 6), and per-sample masks.**

Random tube masking at 75% is V-JEPA's standard recipe. It masks entire temporal tubes
(same spatial position across all frames in a tube) rather than individual patches,
forcing the predictor to reason about motion and temporal context.

**Why not causal-temporal masking?** Causal masking (mask future frames, predict from
past) would shift the learning objective toward temporal forecasting rather than
holistic action representation. UCF101 linear probe measures holistic action recognition;
causal masking would confound the SIGReg-vs-EMA comparison by changing the downstream
task alignment. It is documented as Phase 6 future work.

**Why not per-sample masks?** Phase 1 uses shared-per-step masks (same mask applied to
all examples in a batch). Per-sample masks may improve gradient variance but add
implementation complexity. Deferred as a post-v1.0 ablation.

---

## 6. Anti-Collapse Mechanism: The Comparison

This is the project's core question. The two conditions are:

| Mode   | Target encoder          | SIGReg (λ) | EMA         |
|--------|------------------------|------------|-------------|
| SIGReg | SharedTargetEncoder    | 0.02       | None        |
| EMA    | EMATargetEncoder       | 0.0        | decay=0.996 |

**SIGReg mode** uses a stop-gradient target (the encoder itself, detached) and relies
entirely on SIGReg's Epps-Pulley distributional penalty to prevent collapse. No momentum
averaging — the target encoder is always the current encoder frozen for a single step.

**EMA mode** uses a momentum-averaged shadow encoder as the target. No SIGReg —
λ=0.0 means the EP loss computes but contributes zero gradient. Collapse prevention
comes entirely from the slowly-moving EMA target.

The comparison is controlled: same encoder architecture, same masking, same data,
same LR/WD schedule, same batch size, same training duration. The only structural
difference is the anti-collapse mechanism.

---

## 7. Seeds and Replication: 2 Seeds per Mode

Two seeds provide a minimal variance estimate to distinguish signal from run-to-run
noise. With 4 runs (2 modes × 2 seeds), we can compute per-mode mean ± std and
report whether the confidence intervals overlap. Two seeds is the bare minimum for
this — it's honest about the project's compute budget rather than inflating the N.

---

## 8. Predictor Depth: 12 (V-JEPA 2 Reference)

Phase 4 used predictor depth=6 as a Colab-T4 compute compromise. With ViT-Tiny at
128×128 fitting comfortably in T4 memory at batch 32, Phase 5 returns to V-JEPA 2's
reference predictor depth=12. The predictor must be expressive enough to map context
tokens across the 1,536-token masked region; depth=6 was likely a bottleneck in Phase 4.

---

## 9. Evaluation: Linear Probe on UCF101 Split 1

Linear probe with multi-clip eval (4 evenly-spaced clips, averaged) is the standard
SSL evaluation protocol. UCF101 split 1 is the default; all Phase 4 and Phase 5 runs
use the same split for comparability.

The linear probe is intentionally simple (a single linear layer, no finetuning) to
measure representation quality directly without confounding from fine-tuning tricks.

---

## 11. Phase 5b: TPU v5e-8 Support

Phase 5b adds a parallel training path for Kaggle TPU v5e-8 (8 chips, 16GB HBM each).
The same four Phase 5 experiments can run on either GPU or TPU by changing only config values.

**Why fp16 stays on GPU.**  
T4 has dedicated fp16 tensor cores; fp16 is measurably faster than bf16 on T4.
The Phase 4/5 GPU configs keep `precision: 16-mixed` and are not touched.

**Why bf16 is forced on TPU.**  
Kaggle's v5e-8 chips do not support fp16 at the hardware level. Attempting fp16 raises
an XLA compilation error. All `*_tpu` configs use `precision: bf16-true` (XLA requires -true variants; bf16-mixed is CUDA-only).

**`persistent_workers: false` on TPU.**  
TPU's host-device handoff pattern prefers fresh DataLoader workers between epochs/batches.
`persistent_workers: true` can cause hangs or stale tensor errors on XLA. GPU configs
keep `persistent_workers: true` (fast worker reuse) unchanged.

**Strategy.**  
`strategy: xla` activates Lightning's `XLAStrategy`, which handles DDP-style distribution
across the 8 TPU chips transparently. `devices: 8` sets the per-node device count.
`pretrain.py` also injects `strategy: xla` automatically if the user sets `accelerator: tpu`
without specifying a strategy (safety net for ad-hoc overrides).

**Smoke test strategy.**  
Two 50-step smoke configs (`phase5_gpu_smoke`, `phase5_tpu_smoke`) use the `smoke_test_tpu`
minimal model (img_size=64, 256 tubelets, predictor depth=2) to verify each code path
compiles and produces finite losses before committing to 75k-step runs. Both use synthetic
data so no UCF101 files are required. Local CPU testing: override `trainer.accelerator=cpu`.

**Expected throughput.**  
For ViT-Tiny transformer workloads, v5e-8 is expected to deliver ~3–5× the throughput of
T4×2. The exact speedup depends on XLA compilation and data pipeline efficiency.

---

## 12. Phase 5c: SPMD Training Path

Phase 5c replaces the Lightning `XLAStrategy` (Phase 5b) with a standalone SPMD harness
(`scripts/pretrain_tpu.py`) for Kaggle TPU v5e-8.

**Why SPMD, not Lightning XLAStrategy.**  
Lightning's `XLAStrategy` uses `xmp.spawn` to launch 8 separate processes. Kaggle TPU v5e-8
exposes all 8 cores in a **single process** — multi-process SPMD is explicitly unsupported
("Multi processing is not compatible with SPMD; under SPMD there is always a single process
per host"). Attempting `xmp.spawn` on Kaggle's TPU runtime fails at process launch.

The architecturally correct fix is `torch_xla` SPMD: one process, all 8 cores, with batch
sharding declared via `xs.mark_sharding`. The XLA compiler handles the necessary collectives
automatically.

**Two-harness tradeoff.**  
GPU keeps Lightning (`pretrain.py`): mature, well-tested, handles checkpointing/logging
callbacks. TPU gets a thin custom loop (`pretrain_tpu.py`, ~200 lines): necessary because
Lightning cannot express single-process SPMD. The GPU path is completely unchanged; all
existing tests still pass. The two scripts share model code, loss functions, checkpointing
utilities, and Hydra configs.

**SPMD topology and batch sharding.**  
Device mesh: `(8, 1)` with axes `('data', 'model')`. Input batch `(B, C, T, H, W)` is
sharded on the `'data'` axis — each of the 8 cores processes `B/8` samples. Model
parameters are replicated across all cores (default SPMD behavior when weights are not
explicitly sharded). Gradients are all-reduced by XLA during backward automatically.

**Precision: bf16 forward, fp32 SIGReg.**  
Encoder and predictor forward passes run in `torch.autocast('xla', dtype=torch.bfloat16)`.
The SIGReg Epps-Pulley CF test runs in fp32: `proj` embeddings are cast to fp32 before
the loss call. The CF test evaluates `cos(x·t)` and `sin(x·t)` at scaled values; in bf16
the low-precision trig accumulates enough error to NaN the statistic. MSE prediction loss
stays in bf16 (numerically stable). The `XLA_USE_BF16` environment variable, deprecated in
torch_xla 2.8, is NOT used.

**SIGReg global-batch all-gather (CRITICAL).**  
Under SPMD data sharding, each core holds `batch_size / 8` samples of the proj embeddings.
The Epps-Pulley test is a distributional statistic over the batch — running it per-shard
would compute 8 independent EP tests on 1/8-scale batches, producing a weaker and biased
statistic that diverges from the GPU baseline. Before calling `SIGRegLoss.forward()`, the
SPMD harness re-annotates `proj_fp32` as fully replicated via
`xs.mark_sharding(proj_fp32, mesh, (None, None, None))`. This inserts an XLA all-gather
collective that assembles the global batch on every core before the EP test. The correct
output of SIGReg is computed once on the full global batch.

**SIGReg batch-size comparability between GPU and TPU runs.**  
GPU DDP (T4 × 2) uses `batch_size=32` per rank → effective global batch = 64. However,
each rank computes SIGReg on its **local** batch of 32 samples, then both GPUs average
the SIGReg gradient (as with all other gradients). TPU SPMD with `batch_size=64` and
all-gather computes SIGReg on the **global** batch of 64 samples before any gradient
averaging. Both are valid Epps-Pulley tests (more samples = stronger signal). The TPU
sees a stronger SIGReg signal (global-64 vs. local-32 per GPU). Phase 6 internal TPU
comparisons are unaffected (all arms use the same statistic). The Phase 5 GPU baseline is
complementary to the TPU runs, not a strict numerical replicate — the gradient magnitude
for SIGReg will differ by a factor related to sample size, which is expected and documented.

**Checkpoint format interoperability.**  
Both `pretrain.py` (Lightning) and `pretrain_tpu.py` (SPMD) save checkpoints as
`{"state_dict": module.state_dict(), "global_step": N}`. The `encoder.*` key prefix is
identical. `extract_features.py` loads either format without modification. The same
`step_{step:07d}.ckpt` filename convention and HF Hub path structure are used.

**EMA update placement.**  
In the SPMD loop, `target_encoder.update()` is called **before** `torch_xla.sync()` so
the EMA weight update is traced in the same XLA computation graph as the optimizer step.
Placing it after `sync()` would defer it to the next step's graph and potentially create
a one-step lag in the shadow weights.

**Scheduler resume.**  
The LR scheduler is constructed with `last_epoch=start_step-1` on resume, which
correctly initialises `get_last_lr()` without requiring a `scheduler.step()` loop.
PyTorch >= 2.4 requires `initial_lr` to be present in optimizer param groups when
`last_epoch >= 0`; `build_optimizer_and_scheduler` sets this explicitly before
constructing the scheduler.

---

## 10. Compute Budget Assumptions

- **GPU**: Kaggle T4 (16GB VRAM), free tier. ~30-40 GPU hours per 75k-step run.
- **Sessions**: Kaggle provides ~30 GPU hours/week on the free tier. With 4 runs,
  this is a ~4-week training commitment assuming no pre-emptions.
- **Checkpointing**: HF Hub checkpoint persistence across sessions. Automatic resume.
- **Batch size fallback**: If batch_size=32 causes OOM at 128×128 (possible with
  longer sequences), drop to 16. The SIGReg EP statistic is less reliable at smaller
  batch sizes but the paper shows it is usable down to N≈32.
