# sigreg-video-lejepa

A controlled comparison of anti-collapse mechanisms (EMA, SIGReg, VICReg) in video JEPA, on UCF101. This repository accompanies the paper *"EMA, SIGReg, and VICReg on Video JEPA: A Controlled Comparison, and the Limits of Effective Rank as a Quality Signal."*

## Summary

Video JEPAs prevent representational collapse with one of several mechanisms: an EMA target encoder, VICReg's variance–covariance regularization, or SIGReg's distribution-matching toward an isotropic Gaussian ([LeJEPA](https://arxiv.org/abs/2511.08544), Balestriero & LeCun 2025). LeJEPA argues SIGReg can replace the EMA machinery; whether this holds for video representation learning is untested under controlled conditions. We fix a video-JEPA testbed (ViT-Tiny, causal masking with a gap, a stochastic predictor) and vary **only** the anti-collapse regularizer across three arms, with two seeds each, evaluating with a UCF101 action-classification linear probe.

**Two findings:**

1. **In our controlled setting, EMA outperforms both distributional regularizers** — 21.2% top-1 vs 10.8% (SIGReg) and 7.8% (VICReg), non-overlapping seed ranges, at EMA's untuned default momentum.
2. **Effective rank fails to predict quality** — EMA and SIGReg reach nearly identical mean effective rank (104 vs 100 of 192) yet EMA is ~2× more accurate. Rank tracks collapse, not representation quality.

These add controlled video evidence to an active 2026 debate on whether isotropic distributional regularization suffices, and are consistent with concurrent image-domain findings and theory that isotropy is costly under structured downstream geometry.

## Phase 6 Results (causal masking + stochastic predictor, UCF101, two seeds)

| Arm    | Top-1 (s0 / s1) | Top-1 mean | Top-5 mean | Eff. rank (s0 / s1) | Rank mean |
|--------|-----------------|------------|------------|---------------------|-----------|
| **EMA**    | 20.70 / 21.70   | **21.20%** | 44.52%     | 110.1 / 97.1        | **103.6** |
| SIGReg | 10.15 / 11.39   | **10.77%** | 30.16%     | 98.2 / 100.9        | **99.5**  |
| VICReg | 7.98 / 7.56     | **7.77%**  | 22.15%     | 64.7 / 64.5         | **64.6**  |

EMA outperforms both distributional regularizers (2.0× SIGReg, 2.7× VICReg on mean top-1) with non-overlapping seed ranges. EMA and SIGReg have nearly identical effective rank (~100–104) yet a ~2× accuracy gap — so rank does not separate them, though it correctly flags VICReg's lower-rank, lower-accuracy regime. Full analysis: [`docs/phase6_results.md`](docs/phase6_results.md).

Effective rank is the Roy–Vetterli measure (exponential of the entropy of the normalized, unsquared singular-value spectrum), computed on the test-set feature matrix. All numbers come from a single consistent re-evaluation pipeline over the six final checkpoints.

## Checkpoints

Final (75,000-step) checkpoints for all six runs: [HuggingFace](https://huggingface.co/ankitpani/sigreg-video-lejepa-checkpoints/tree/main/checkpoints/phase6_final).

## Testbed

- **Encoder**: ViT-Tiny (192-dim, depth 12), 3D tubelets 2×8×8, 16 frames at 128×128 (2,048 tubelets).
- **Masking**: deterministic causal mask with a temporal gap — context positions 0–2, hidden gap 3–4, target positions 5–7. Removes the temporal-copy shortcut that random tube masking permits on video.
- **Predictor**: stochastic single-Gaussian head (μ, log σ²), reparameterized sample, smooth-L1 prediction loss, per-dimension free-bits KL floor (0.5 nats/dim), β=1e-3.
- **Regularizer (the only variable)**: EMA (momentum 0.996, V-JEPA default) / SIGReg (λ=0.02) / VICReg-VC (μ_v=25, μ_c=1, λ=1.0).

Testbed components (causal masking, stochastic prediction) are not claimed as contributions; they define a fixed, non-trivial prediction task on which the regularizers compete.

## Compute

Pretraining runs on Kaggle TPU v5e-8 under a single-process SPMD harness (`scripts/pretrain_tpu.py`), since Lightning's `XLAStrategy` is incompatible with Kaggle's single-process TPU topology. VICReg's covariance computation required an SPMD-specific fix to run within TPU memory. See [`docs/design_decisions.md`](docs/design_decisions.md) for the engineering details. A GPU path (`scripts/pretrain.py`, PyTorch Lightning) is also provided.

## Setup

```bash
git clone https://github.com/ankitpani8/sigreg-video-lejepa.git
cd sigreg-video-lejepa
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Dataset

Download UCF101 from [CRCV](https://www.crcv.ucf.edu/data/UCF101.php) or the HuggingFace mirror (`quchenyuan/UCF101-ZIP`), then place:

- `ucf101/UCF-101/` — 101 class folders, 13,320 `.avi` files
- `ucf101/ucfTrainTestlist/` — split files (`classInd.txt`, `trainlist01.txt`, `testlist01.txt`)

The default configs locate the data automatically; the first training run caches it to local SSD for fast I/O.

## Limitations and scope

Small scale (ViT-Tiny, UCF101) relative to production video-JEPA; absolute accuracies are low and meaningful only for internal cross-arm comparison. LeJEPA's scaling claim is not refuted by small-scale evidence — this is a controlled counterexample at the scale tested. Extending to Something-Something-v2 with a larger encoder (encoder capacity and data jointly) is the natural next step and is left as future work.

## Roadmap

- [x] **Phase 0–3** — Pipeline, tubelet embedding, UCF101 data, linear-probe evaluation
- [x] **Phase 4** — UCF101 baseline at 64×64 / 25k (pipeline validation)
- [x] **Phase 5** — SIGReg vs EMA at 128×128 / 75k, including an EMA momentum-sensitivity sub-study (appendix material)
- [x] **Phase 5c** — TPU SPMD training path validated (numerical equivalence to CPU reference)
- [x] **Phase 6** — Three-way regularizer comparison (EMA / SIGReg / VICReg), two seeds, causal-stochastic testbed — **complete**
- [ ] **Future** — SSv2 / larger-encoder scaling; disentangling isotropy-cost vs. gradient-vanishing explanations for SIGReg's underperformance

## Citation

If you use this code or the checkpoints, please cite the accompanying paper (preprint link to be added).

## License

MIT
