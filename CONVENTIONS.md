# CONVENTIONS — tracking, checkpointing, secrets

*Standing engineering conventions. Obeyed across every phase. Claude Code follows these without being re-told.*

## 0. The rule that overrides convenience
`experiments/<phase>/provenance.json` is the **canonical record** and is committed to git.
W&B and the HF Hub are mirrors and durable stores — convenient, but not the record.
**A claim without a committed `provenance.json` does not count** (plan.md §0).

## 1. Secrets
Set once; never commit. Names are fixed so kernels/scripts can rely on them.

| Secret | Where to set | Used for |
|---|---|---|
| `WANDB_API_KEY` | Kaggle Secrets + local (`wandb login`) | experiment tracking |
| `HF_TOKEN` (write scope) | Kaggle Secrets + local (`huggingface-cli login`) | durable checkpoint/feature storage |
| `KAGGLE` (kaggle.json) | `~/.kaggle/kaggle.json` (already set) | CLI orchestration |

In a Kaggle kernel, read secrets via `from kaggle_secrets import UserSecretsClient`.
`.gitignore` already blocks `kaggle.json` and `.env`. **Never echo a secret into a log or provenance file.**

## 2. Weights & Biases
- **Project:** `gswm`.  **Group:** the phase (`phase0`, `phase1`, …).  **Job type:** the task (`extract`, `probe`, `train`).
- **Run name:** `<phase>-<task>-<gitsha7>-<seed>` so every run traces to a commit.
- **Log:** the full config (dataset id+revision, model id, pooling, N, split method, thresholds), metrics, and at the
  end dump the same dict that goes to `provenance.json` into `wandb.summary`.
- **Artifacts:** register the feature cache / checkpoint as a W&B artifact for lineage (the bytes live on HF, see §3).
- **Graceful degradation:** wrap init in a helper (`gswm/tracking.py`) that no-ops if `WANDB_API_KEY` is absent or
  `WANDB_MODE=offline` — a missing key must never block or crash a run.

## 3. Checkpointing & durable storage (don't lose progress)
Two stores, always: **Kaggle local** (`/kaggle/working`, persists as kernel output, ~20 GB) **and** the **HF Hub**
(cross-platform, durable). The HF Hub is the source of truth for resuming.

**HF Hub repos** (namespace `<HF_USER>` — confirm your HF username; default `ankitpani`):
| Artifact | Repo type | Name |
|---|---|---|
| V-JEPA feature caches | `dataset` | `<HF_USER>/gswm-seamless-vjepa-feats` |
| Trained module checkpoints (Phase 1+) | `model` | `<HF_USER>/gswm-<phase>` |

**Shard-and-resume pattern (required for any long Kaggle run):**
1. Work in chunks of `K` items (start `K=100`). After each chunk, write a shard
   (`features_part{idx:04d}.npy` + labels + group_ids) to `/kaggle/working` **and** `upload_file` it to the HF repo.
2. On startup, list existing shards on the HF repo and **resume from the next index** — never recompute.
3. On completion, consolidate shards → single `features.npy` (+ `labels.npy`, `group_ids.npy`, `meta.json`),
   push the consolidated cache, and version a private Kaggle dataset `gswm-seamless-vjepa-feats`.
4. Log chunk progress + throughput to W&B so a stalled run is visible.

Phase 0 has **no model to checkpoint** — "progress" is the feature cache, so the above applies to features only.
Model checkpointing (same dual-store pattern, `model` repo, save every N steps + best-by-val) starts in Phase 1.

## 4. Traceability
Every `provenance.json` and W&B run records: git commit SHA, dataset id + revision, model id, config hash, seeds,
and the pre-registered thresholds alongside the result. Reproducibility is part of the deliverable.

## 5. `meta.json` (ships with every feature cache)
`{dataset_id, revision, config (improvised/naturalistic), split, model_id, pooling_spec, n_clips,
class_distribution, frame_sampling, extracted_at, git_sha}` — enough to regenerate the cache from scratch.
