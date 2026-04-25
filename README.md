# sigreg-video-lejepa

Applying LeJEPA's SIGReg training recipe to a V-JEPA-style video architecture, evaluated on UCF101 and Something-Something v2.

## Status

🚧 Under active development. v1.0 (UCF101) target: in progress.

## Motivation

[LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, Nov 2025) introduced SIGReg — a clean, principled regularizer that replaces the EMA target encoder used in earlier JEPAs. It was validated on images.

[V-JEPA](https://arxiv.org/abs/2404.08471) and [V-JEPA 2](https://arxiv.org/abs/2506.09985) extended JEPA to video, but still rely on the EMA recipe.

This project bridges the two: **can SIGReg training extend cleanly from images to video?**

## Roadmap

- [ ] **v0.x — Development on UCF101**
  - [ ] Data pipeline
  - [ ] V-JEPA-style architecture
  - [ ] SIGReg loss
  - [ ] Pretraining on UCF101
  - [ ] Linear-probe evaluation
- [ ] **v1.0 — UCF101 release** (preliminary)
- [ ] **v2.0 — SSv2 release** (Launch)

## Setup

```bash
git clone https://github.com/ankitpani8/sigreg-video-lejepa.git
cd sigreg-video-lejepa
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## License

MIT
