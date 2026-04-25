# CLAUDE.md — Project Memory for sigreg-video-lejepa

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
- **Linting**: ruff
- **Testing**: pytest

## Repo Structure


src/sigreg_video_lejepa/
models/        # ViT encoder, predictor, full JEPA
data/          # UCF101, SSv2 datasets, transforms
training/      # Lightning modules, SIGReg loss
evaluation/    # Linear probe, k-NN eval
utils/         # Logging, checkpointing helpers
configs/         # Hydra configs (model/, data/, training/, experiment/)
notebooks/       # EDA, result analysis, demos
scripts/         # CLI entry points (pretrain.py, eval.py, etc.)
tests/           # pytest unit tests
results/         # Checkpoints, logs, figures (gitignored except .gitkeep)
docs/            # Extended notes, design decisions
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

## Open Questions / Active Decisions

(Update this section as the project evolves.)

- [ ] Backbone size for UCF101 prototype: ViT-Tiny vs ViT-Small?
- [ ] SIGReg lambda schedule: constant or warmup?
- [ ] Frame sampling rate for UCF101: 8, 16, or 32 frames per clip?

