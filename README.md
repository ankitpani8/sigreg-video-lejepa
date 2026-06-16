# Grounded Social World Model (GSWM)

> The first world model whose model of **other minds** is real enough to survive its own falsification test.
> Not a better physics predictor — a grounded *social* one.

GSWM fuses a **frozen physical world model** (V-JEPA 2) with a trained **Bayesian social-latent module**
that infers each agent's belief / desire / intention — and holds that latent to a strict standard: it must be
*functional and manipulable*, not decorative. Everything is built **falsification-first**: the validation
instrument is built alongside the model, and the documented record of what survives it **is the deliverable.**

`plan.md` is the project constitution. Read it before touching code.

---

## The one principle
You can **falsify "no theory of mind"** but you can **never prove "genuine theory of mind"** from behavior
alone. So we falsify the negative, never claim the positive, and ship a **provenance record** for every claim
that the model understands intent. The science is the provenance.

## Design (one paragraph)
A two-stage architecture, **backbone frozen, module trained**. V-JEPA 2 supplies the "what is physically
happening" substrate (it never trains). On top, a variational module infers `m_i = (belief, desire, intention)`
per agent, with an agent's *belief* modeled by re-running the same backbone from that agent's vantage point
(weight-sharing makes recursion tractable). A two-timescale memory separates a slow, person-specific prior
(people you know) from a fast episode latent (strangers). Belief updates are amortized variational inference.
A game-theoretic head plans by best-response over inferred goals. The central engineering risk is **posterior
collapse** — a model that predicts behavior while its belief variable does no work — which is exactly the
"no real theory of mind" failure. The **grounding battery (T0–T4)** is the instrument that detects and prevents it.

## The grounding battery
| Tier | Test | Establishes |
|---|---|---|
| T0 | Behavioral accuracy | Nothing on its own — pattern-matching passes it |
| T1 | Divergence under surface perturbation (false-belief, worst-case) | Falsifies shallow/surface models |
| T2 | Counterfactual mental-state intervention (freeze observable, vary the belief) | A belief variable exists |
| T3 | Linear probe decodes the belief latent (with controls) | The belief is *explicitly represented* |
| T4 | Causal patching — overwrite belief activations, prediction must flip | The variable is *used*, not just present |

Label "grounded" only if a model survives T1, responds to T2, is T3-probeable, **and** T4-patchable — each
survivor shipping a provenance record. Failing T3/T4 is *weak* evidence; passing is *strong*. Respect the asymmetry.

---

## Status
| Phase | What | Kill-criterion | Status | Record |
|---|---|---|---|---|
| **0** | Kill-test: can a frozen V-JEPA 2 + linear probe read a social signal off video? | probe beats strongest baseline ≥10 pts (non-overlapping CIs) **and** ≥1.5× majority | 🟡 in progress — T0 ✓ | `experiments/phase0/` |
| 1 | Minimal social latent + grounding instrument (T3+T4 from day one) | latent probeable **and** patchable on held-out; improves prediction | ⛔ blocked on P0 | — |
| 2 | False-belief reasoning (T1 perturbation + T2 intervention) | survives worst-case perturbation + correct interventions | ⛔ blocked | — |
| 3 | Two-timescale person-prior (stranger vs. known) | concentrated prior updates slower than diffuse | ⛔ blocked | — |
| 4 | Recursion + game-theoretic head | partial expected; deception is the ceiling | ⛔ blocked | — |

**Gate rule:** do not build Phase N+1 until Phase N's kill-criterion passes. Each phase graduates or dies.

---

## Repo structure
```
gswm/                 # importable library (pip install -e .)
  backbone/           #   frozen V-JEPA 2 loading + feature extraction
  social_module/      #   variational BDI encoder, belief reuse, memory, predictive head
  grounding_tests/    #   T0–T4: probe, patch, perturbation, intervention
  scenarios/          #   procedural false-belief generator (Phase 2)
  data/               #   dataset download + preprocessing
  tpu/                #   torch_xla / SPMD training code (Kaggle TPU)
  tracking.py         #   W&B + provenance helpers (see CONVENTIONS.md)
scripts/              # phase entry points (run locally or on Kaggle)
kaggle/               # kernel metadata + push.sh orchestration + per-kernel code
experiments/          # one dir per phase; provenance.json is the deliverable
briefs/               # per-phase work orders (briefs/phaseN.md)
plan.md               # the constitution
CONVENTIONS.md        # tracking / checkpointing / secrets — obeyed across all phases
```

## Install & use
```bash
pip install -e .                      # importable library
python scripts/<phase_entry>.py       # run a phase step locally (CPU-smoke first)
./kaggle/push.sh kaggle/kernels/<k>   # push a kernel to Kaggle, poll, pull results
```

## Infrastructure
- **Experiment tracking:** Weights & Biases, project `gswm`, run group per phase. See `CONVENTIONS.md`.
- **Durable storage:** feature caches and model checkpoints are versioned on the HF Hub (and Kaggle datasets),
  so progress survives session resets. See `CONVENTIONS.md`.
- **Provenance:** `experiments/<phase>/provenance.json` is the canonical record, committed to git.

## Documents
- **`plan.md`** — the thesis, architecture, validation philosophy, phased plan with kill-criteria, and the known walls.
- **`briefs/`** — precise per-phase work orders.
- **`CONVENTIONS.md`** — the standing engineering conventions.
