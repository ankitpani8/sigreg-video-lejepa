# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is read at the start of every Claude Code session. Keep it current.

## Project Goal

Apply LeJEPA's SIGReg training recipe (Balestriero & LeCun, Nov 2025) to a V-JEPA-style video architecture, evaluated on UCF101 (v1.0) and Something-Something v2 (v2.0).

This is a research project. The objective is honest investigation, not a predetermined result. Negative results are valuable.

## Owner

Ankit Pani (ankitpani8 on GitHub). 6+ years data science / analytics consulting. Learning JEPA architectures hands-on.

## Working Style Preferences

- **Be direct.** Honest feedback over diplomatic hedging. Push back when I'm wrong.
- **Plan before acting on substantial changes.** Use plan mode (Shift+Tab) for anything beyond small edits.
- **Explain decisions briefly.** When you make a non-obvious architectural choice, leave a one-line comment explaining why.
- **Prefer industry-standard patterns** over clever custom code. PyTorch Lightning, Hydra, type hints, ruff-formatted.
- **No silent fallbacks.** If something fails, fail loudly with a clear error.
- **Write tests for non-trivial code.** Skip tests for one-off scripts.

## Stack

- **Language**: Python 3.11
- **Package manager**: uv
- **DL framework**: PyTorch 2.4+
- **Training**: PyTorch Lightning 2.4+
- **Configs**: Hydra 1.3+
- **Experiment tracking**: Weights & Biases (wandb)
- **Linting**: ruff (line length 100; E501/N806/N812 ignored)
- **Testing**: pytest

## Commands

```bash
# Install (editable, with dev deps)
uv pip install -e ".[dev]"

# Lint
ruff check src/

# All tests (fast only)
pytest -m 'not slow'

# All tests including slow
pytest

# Single test file
pytest tests/test_smoke.py

# Single test by name
pytest tests/test_smoke.py::test_encoder_forward

# Pretrain (default: synthetic data, 100 steps, CPU)
python scripts/pretrain.py

# Pretrain with an experiment override
python scripts/pretrain.py +experiment=smoke_test_phase0   # 2 steps, synthetic
python scripts/pretrain.py +experiment=smoke_test_phase1   # 2 steps, masking active
python scripts/pretrain.py +experiment=ucf101_dryrun       # 10 steps, real UCF101 (Colab)

# Ad-hoc Hydra overrides
python scripts/pretrain.py trainer.max_steps=50 data.batch_size=8
```

## Architecture

### Training Loop Data Flow

```
Input video (B, C, T, H, W)
  → TubeletEmbed (Conv3d) → N tubelets of shape (B, N, D)
  → TubeMasker → ctx_idx (25%), tgt_idx (75%)

Context path:
  encoder(x, token_indices=ctx_idx) → (B, N_ctx, D)
  predictor(ctx_tokens, tgt_idx)    → (B, N_tgt, D)  [mask tokens fill gaps]
  projector(ctx_tokens)             → (B, N_ctx, proj_dim)  [for SIGReg]

Target path:
  target_encoder(x)[..., tgt_idx, :] → (B, N_tgt, D)  [detached, no grad]

Losses:
  L_pred   = MSE(predictor output, target encoder output)
  L_SIGReg = Epps-Pulley characteristic function test on projector output
  L_total  = (1 − λ) * L_pred + λ * L_SIGReg
```

**Phase 0** (masker=None): uses a 4-token proxy target instead of real masked prediction. This was the initial scaffolding phase and is retained for smoke tests.

### Key Abstractions

Two duck-typed interfaces avoid isinstance checks in `VideoJEPAModule`:

- **`target_encoder`**: either `SharedTargetEncoder` (weights shared with encoder, update is a no-op) or `EMATargetEncoder` (shadow copy updated with momentum in `on_after_backward`). Both expose `.encode()`.
- **`masker`**: either `TubeMasker` (returns `ctx_idx, tgt_idx`) or `None` (Phase 0 bypass path).

### SIGReg Loss (`training/sigreg_loss.py`)

Penalizes non-Gaussianity of the projector's embeddings via the Epps-Pulley test:
1. Project embeddings onto `num_projections=256` random unit vectors → scalar projections.
2. For each direction, evaluate the empirical characteristic function at `knots=17` points in `[0, t_max=3.0]`.
3. Integrate the squared deviation from the Gaussian CF using the trapezoidal rule (integration buffers precomputed at `__init__`).

Full mathematical spec: `docs/sigreg-spec.md`.

### Config System (Hydra)

Configs compose from four groups: `data/`, `model/`, `training/`, `experiment/`. Root defaults are in `configs/config.yaml`. Experiment configs use `@package _global_` to override any group simultaneously — this is the main entry point for named runs:

```
configs/experiment/ucf101_dryrun.yaml
  → overrides /data=ucf101_small, /model=ucf101_small, /training=ucf101_pretrain
  → sets trainer.max_steps, trainer.accelerator, data.batch_size
```

`scripts/pretrain.py` uses `OmegaConf.to_container(cfg, resolve=True)` to convert the composed config to plain dicts before passing to module constructors.

### Data Pipeline

`BaseVideoDataset` handles decord frame sampling and short-clip looping (modulo, no black-frame padding). `UCF101Dataset` adds split-file parsing and a ≥100-class validation gate. `VideoReader` is opened inside `__getitem__` (not `__init__`) for multiprocessing fork safety.

`UCF101Transform` follows V-JEPA 2 augmentations: `RandomResizedCrop(scale=(0.3, 1.0))` + `RandomHFlip`; no color jitter or Gaussian blur. Input: `(T, H, W, C)` uint8; output: `(C, T, H, W)` float32, ImageNet-normalized.

## Compute

- **Local laptop**: Dell Vostro 3525, AMD Ryzen 5 5625U, 8GB RAM, integrated graphics. **Editing/git only — never train here.**
- **Training**: Google Colab Pro. Free tier T4 for prototyping; A100/V100 when available.
- **Storage**: Google Drive Premium (5TB). Mounts in Colab as `/content/drive/MyDrive/`.

## Key References

- LeJEPA paper: https://arxiv.org/abs/2511.08544
- LeJEPA code: https://github.com/rbalestr-lab/lejepa
- V-JEPA: https://arxiv.org/abs/2404.08471
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- I-JEPA (original): https://arxiv.org/abs/2301.08243
- UCF101: https://www.crcv.ucf.edu/data/UCF101.php
- SSv2: https://developer.qualcomm.com/software/ai-datasets/something-something

## Data Strategy

- **v1.0 (UCF101)**: ~7GB, downloads cleanly, fast iteration. Training data in `/content/drive/MyDrive/datasets/ucf101/`.
- **v2.0 (SSv2)**: ~50GB, requires academic registration. Training data in `/content/drive/MyDrive/datasets/ssv2/`.
- Code is **modular**: a `BaseVideoDataset` with `UCF101Dataset` and `SSv2Dataset` subclasses. Same model, trainer, eval pipeline for both. Switch via Hydra config.

## Methodology Commitments

- **Honest evaluation.** Hold out test split from day one. Never tune on test.
- **Reproducibility.** Seed everything. Log full config + git SHA per run.
- **Document failures.** Every failed approach gets a note in `docs/failed-approaches.md` with diagnosis.
- **No cherry-picking.** Report all runs, including the embarrassing ones.

## Versioning Plan

- v0.x: development on UCF101
- v1.0: UCF101 results released, marked as "preliminary, prototype"
- v2.0: SSv2 results — this is the LinkedIn launch version
- If SIGReg fails on video, v2.0 is a negative-results post. That's still a valid release.

## Things To Always Do

- Run `ruff check src/` before commits
- Run `pytest` before commits
- Update README roadmap checkboxes when phases complete
- Log to wandb for any training run > 5 minutes
- Commit small, commit often, write meaningful commit messages

## Things To Never Do

- Train on the laptop GPU
- Commit data files (>1MB) to the repo
- Commit secrets (.env should be gitignored)
- Tune hyperparameters on the test set
- Claim alpha or capability we haven't honestly demonstrated
- Force-push to main

## Phase Status

- Phase 0: ✅ complete (synthetic data, end-to-end pipeline)
- Phase 1: ✅ complete (tubelet embedding, random tube masking)
- Phase 2: ✅ complete (UCF101 data pipeline — real data, configs, integration test)
- Phase 3: next (linear probe evaluation)

## Open Questions / Active Decisions

(Update this section as the project evolves.)

- [x] Backbone size for UCF101 prototype: **ViT-Tiny** (embed_dim=192, resolved in Phase 2)
- [x] Frame sampling rate for UCF101: **16 frames at ~4 fps** (stride=6, resolved in Phase 2)
- [ ] SIGReg lambda schedule: constant or warmup? (defer to Phase 4)

## Key Design Decisions (Phase 2)

- **ucf101_small uses 64×64 spatial** (vs V-JEPA 2's 224×224) — deliberate Colab T4 constraint.
- **Predictor depth=6** (vs V-JEPA 2's depth=12) — same constraint; revisit at scale-up.
- **V-JEPA 2 augmentations**: scale=(0.3, 1.0), no color jitter, no Gaussian blur.
- **decord VideoReader opened inside `__getitem__`** (fork safety with DataLoader workers).
- **Frame looping via modulo** for short clips (no black-frame padding, no skipping).
- **multi-clip eval** stubbed as `raise NotImplementedError("multi-clip eval: Phase 3")`.

## Deferred Ablations

- **Per-sample masks vs shared-per-step masks.** Phase 1 uses shared. Per-sample may help; defer test to post-v1.0.
## Git Commit Guidelines
- Never include "Co-Authored-By: Claude" or any AI attribution trailer in commit messages.
- Never include "🤖 Generated with Claude Code" or similar promotional footers.
- Commit messages should be plain, focused, and human-style.
