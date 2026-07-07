# Chempy multi-star hierarchical HMC — PyTorch emulator + NumPyro

Replication of the ChempyMulti hierarchical multi-star inference (Philcox & Rybizki
2019, [arXiv:1909.00812](https://arxiv.org/abs/1909.00812)), modernized:

* **Emulator**: the trained PyTorch NN from `sbi_chemical_abundances/01_train_torch_chempy.py`
  (6→100→40→9, tanh/tanh/linear) replaces the old 2-layer Theano net.
* **Sampler**: NumPyro NUTS on GPU (JAX autodiff), replacing PyMC3/Theano.
* **GPU selection**: `autocvd` picks a free GPU per process; the three N-runs go in
  parallel across three GPUs, 4 chains vectorized per run.

## Key modeling points (Theano → PyTorch)

* **No standardization.** The old Theano model standardized inputs/outputs
  (`in_mean/std`, `out_mean/std`); the PyTorch emulator is trained on **raw**
  params → raw abundances (verified by grepping the whole sbi repo and reading the
  dataset-creation script). The likelihood therefore lives directly in abundance space.
* **Element handling.** The emulator outputs 9 elements incl. H; H (index 2) is
  normalization-only and dropped → 8 elements matching `Mock_Data_TNG.npz` order exactly.
* **Input.** 6-dim `[Lambda(2), Theta(3), time]` — no engineered `time^2` feature.
* **Gradients for NUTS.** The NN forward pass is rebuilt in JAX from the extracted
  weights (`emulator.py`); NUTS gets analytic gradients for free. The JAX port was
  verified bit-faithful to torch (max abs diff 1e-6) and reproduces the emulator APE
  (median ~2.1%).

Model structure (`model.py`): global `Lambda = [alpha_IMF, log10 N_Ia]` shared across
stars; per-star local `Theta` (3) + `time`; HalfCauchy per-element model error added in
quadrature to the TNG observational errors (paper-faithful `elem_err=True`).

## Results

Single hierarchical fit for N = 1, 10, 200 stars (first N of the 1000 TNG mock stars).
Recovered global parameters (median ± 1σ):

| N | alpha_IMF | log10 N_Ia | MCMC health |
|---|---|---|---|
| 1 | −2.06 ± 0.13 | −2.86 ± 0.17 | healthy (r̂=1.00, ESS≈2500) |
| 10 | −2.25 ± 0.04 | −2.88 ± 0.04 | healthy (r̂=1.00, ESS≈4000) |
| 200 | −2.26 ± 0.007 | −2.875 ± 0.007 | see note below |

The `n_stars` plot (`plots/n_stars_hmc_TNG.png`) reproduces the paper's figure: the
global-parameter uncertainty tightens with N; log10 N_Ia converges cleanly onto the
nominal truth.

### The alpha_IMF offset is physical, not a bug

alpha_IMF converges to ≈ −2.26, above the nominal −2.30. This offset is already present
in the fully-healthy N=10 fit and is unchanged by every sampler variant, so it is **not**
a sampling artifact: `Mock_Data_TNG.npz` is real IllustrisTNG data, **not** Chempy output
generated at alpha=−2.30, so −2.30/−2.89 are prior/nominal values, not true generators.
The truth sits outside 2σ in both HMC and ADVI — a statement about model/yield–data
mismatch, consistent with Philcox & Rybizki's TNG results.

### N=200 convergence study

The 802-dim N=200 posterior has a rough, weakly-degenerate geometry that makes NUTS
metastable (chains crawl through slightly different sub-regions). Four configurations were
tried (all in `posteriors/` + `plots/` subdirs):

| approach | r̂ (α/logN) | ESS (α/logN) | median α / logN |
|---|---|---|---|
| centered | 1.77 / 1.82 | 6 / 6 | −2.263 / −2.872 |
| **non-centered** (best) | **1.10 / 1.32** | **40 / 11** | −2.262 / −2.874 |
| non-centered + 3k draws + tree-depth 12 (`200_3k`) | 1.19 / 1.41 | 23 / 8 | −2.264 / −2.880 |
| non-centered + ADVI-init + dense mass (`200_advi_dense`) | 1.30 / 1.43 | 11 / 9 | −2.260 / −2.874 |

More draws, deeper trees, ADVI-init, and a dense mass matrix all **plateau** at r̂≈1.3 —
the difficulty is intrinsic, not a tuning knob. Crucially, **every method (and an
independent full-covariance ADVI fit) agrees on Λ to ~0.01**, and the between-chain spread
is at the level of the reported σ. So the Λ posterior location and width are robust; the
poor r̂ reflects the geometry, not a wrong answer.

* `corner_gauss.py` — Gaussianized posterior (mean, covariance, ρ=0.705) with 1/2σ ellipses.
* `advi_corner.py` — full-covariance ADVI (`AutoMultivariateNormal`) posterior vs HMC;
  the two overlap (means to ~0.002, σ to ~0.001, ρ≈0.7–0.8), independently confirming the
  HMC result. The ADVI Gaussian is a clean, MCMC-convergence-free alternative deliverable
  (and is essentially what the paper's analytic Gaussian-product method computes).

## Layout

```
config.py          constants (priors, bounds, element map) — no runtime Chempy dep
extract_weights.py torch state_dict -> weights.npz  (run once)
emulator.py        JAX forward pass (H dropped)
model.py           NumPyro hierarchical model (centered + non-centered) + mock loader
run_hmc.py         one fit on one autocvd GPU; --tag, --advi_init, --dense_mass, ...
run_all.py         launch N=1,10,200 in parallel (one GPU each)
eval_ape.py        emulator APE sanity check vs torch reference
diagnostics.py     r̂/ESS/BFMI, trace/rank/energy/pair (per N or --tag)
corner_gauss.py    corner + Gaussian approximation of Lambda
advi_corner.py     ADVI Gaussian posterior vs HMC
plot_n_stars.py    the paper-style N_stars figure
posteriors/        outputs: posterior_N{n}.nc (full chain), .npz (Lambda), logs
plots/             figures + per-run summary.txt
```

Run: `uv sync`, then `uv run python run_all.py` (or `run_hmc.py --n_stars N ...`).
The full-dimensional N=200 `.nc` chains are gitignored (100s of MB, regenerable); the
Lambda posteriors (`.npz`) and all diagnostics/plots are committed.
