# CLAUDE.md — chempy_hmc

Project guide for future sessions. This repo ports the ChempyMulti hierarchical multi-star
inference to **JAX + NumPyro** (GPU NUTS). There are **two inference methods** — `parent`
(raw PyTorch emulator, sampled raw) and `paper_net` (standardized sklearn net, sampled
standardized) — run against **three observation datasets**, one directory per dataset.
Backward compatibility is a hard rule: no dataset/method directory touches another's files.

## Repository layout

Everything lives under `hmc_numpyro/` (which holds the shared **uv** env: `.venv/`,
`pyproject.toml`, `uv.lock`). Each dataset directory contains a `parent/` and a `paper_net/`
subfolder, each fully self-contained (own `config`, `model`, `emulator`, run/diagnostic
scripts, committed emulator weights, and `posteriors/` + `plots/` outputs):

```
hmc_numpyro/
├─ .venv/ pyproject.toml uv.lock          # shared environment (do not duplicate)
├─ TNG_set/        # dataset 1: Mock_Data_TNG.npz (the original, validated results)
│   ├─ parent/     └─ paper_net/
├─ mock_chempyTNG/ # dataset 2: model_comp_data/mock_abundances.npy (Chempy mock ~ TNG)
│   ├─ parent/     └─ paper_net/
└─ alt_yields/     # dataset 3: model_comp_data/chempy_alternative_yields_obs.npz["arr_0"]
    ├─ parent/     └─ paper_net/           #   (alternative-yields mock = misspecification test)
```

The per-method file inventories in **Experiment 1** (parent) and **Experiment 2** (paper_net)
below describe the files inside each `parent/` and `paper_net/` subfolder respectively. Run a
script from inside its own subfolder (e.g. `cd mock_chempyTNG/parent && python run_hmc.py …`)
so `import config`/`import model` resolve locally. Each of the three datasets runs N=1,10,200.

## GPU / running experiments — ALWAYS use autocvd

**Every script that runs on the GPU must select its GPU with `autocvd`, and autocvd must be
called BEFORE JAX (or anything that imports JAX/XLA) is imported.** This is non-negotiable —
the cluster is shared and jobs must claim a free GPU instead of colliding on one. The pattern
(see any `<dataset>/parent/run_hmc.py` / `<dataset>/paper_net/run_hmc_paper.py`):

```python
def main():
    from autocvd import autocvd
    autocvd(num_gpus=1)                                    # picks a free GPU, waits if none
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax                                             # only now import JAX
    ...
```

- `autocvd(num_gpus=1)` sets `CUDA_VISIBLE_DEVICES` to a free GPU and blocks until one frees up.
- Parallel launchers (`run_all.py`, `run_all_paper.py`) spawn one child per N; each child calls
  `autocvd` itself so it grabs a distinct GPU. Launches are staggered (~20 s) so the utilization
  snapshots don't all fire at once.
- NUTS runs 4 chains via `chain_method="vectorized"` (vmap on one GPU).
- Env is managed with **uv** (`hmc_numpyro/pyproject.toml`, `uv.lock`). Run scripts with the
  project venv. When launching a long fit, prefer a harness-tracked background job so completion
  is reported.

## Environment / setup facts

- The **paper-net emulator is unnormalized-free**: it standardizes inputs and outputs. The
  **parent (raw) emulator is unnormalized** (raw params → raw abundances). Do not mix them up.
- 8 TNG elements everywhere: `['C','Fe','He','Mg','N','Ne','O','Si']` (H is dropped).
- `scikit-learn` (1.9.0) is a dependency (paper-net training only). torch is used *only* by
  `extract_weights.py`; the samplers never import torch.

---

## Experiment 1 — `parent/` method (raw PyTorch emulator)

Files below live in each dataset's `parent/` subfolder (`TNG_set/parent/`,
`mock_chempyTNG/parent/`, `alt_yields/parent/`).

Modern raw-space emulator from `references/sbi_chemical_abundances` (PyTorch 6→100→40→9, no
standardization), sampled raw. Historically **metastable at N=200** (r̂~1.3, ESS~10 even at 10⁴
draws) — this is what motivated Experiment 2.

| file | what it does |
|---|---|
| `config.py` | Constants for the raw experiment. Paths to the PyTorch state dict, TNG val data, Mock_Data_TNG. Element bookkeeping (9 NN outputs, drop H → 8 obs elements). Priors, bounds, `N_STARS_LIST=[1,10,200]`, output dirs (`weights.npz`, `posteriors/`, `plots/`). |
| `extract_weights.py` | **Run once.** Loads the PyTorch `state_dict` (l1/l2/l3), dumps weights to `weights.npz` for the JAX forward pass. Only file that imports torch. |
| `emulator.py` | JAX forward pass: `y = tanh(tanh(tanh(x@W1'+b1)@W2'+b2))@W3'+b3` (no normalization). `predict_obs` drops H → 8 abundances in TNG order. Differentiable for NUTS. |
| `eval_ape.py` | Sanity check: recompute Absolute Percentage Error of the JAX port on the TNG val set, confirm it matches torch. Raw net ≈ a few % (~0.01–0.05 dex). |
| `model.py` | NumPyro hierarchical model (port of `run_pymc3.py`) in **raw abundance space**. `load_mock` aligns TNG mock to `OBS_ELEMENTS`. Global `Lambda`, per-star `Theta`/`time` in a plate, emulator likelihood, element error. Centered and non-centered parameterizations. |
| `run_hmc.py` | One fit for N stars on one autocvd GPU. Args: `--n_stars --draws --tune --chains --target_accept --max_tree_depth --seed --no_elem_err --centered --tag --advi_init --advi_steps --advi_lr --dense_mass`. Default here is **non-centered**. Saves `posteriors/<tag>/posterior_N{n}.nc` + `.npz`. |
| `run_all.py` | Launch N=1,10,200 in parallel (one GPU each), forwarding extra args to each `run_hmc.py`; logs to `posteriors/run_N{n}.log`. |
| `diagnostics.py` | r̂/ESS/BFMI + trace/rank/energy/autocorr/pair plots and `summary.txt` for a saved `.nc`, focused on `Lambda`. Outputs `plots/N{n}/`. |
| `plot_n_stars.py` | Global-parameter posterior (median ±1σ/2σ) vs N — the paper's n_stars figure — for the raw HMC posteriors. |
| `corner_gauss.py` | Corner plot of `Lambda` samples with a Gaussianized (mean+cov) overlay: 1/2σ ellipses + Gaussian marginals. `--n_stars --tag`. |
| `advi_corner.py` | Fit full-covariance ADVI (`AutoMultivariateNormal`) and overlay its Gaussian `Lambda` posterior on the HMC samples. Diagnostic for how Gaussian the posterior is. |

---

## Experiment 2 — `paper_net/` method (original ChempyMulti net)

Files below live in each dataset's `paper_net/` subfolder (`TNG_set/paper_net/`,
`mock_chempyTNG/paper_net/`, `alt_yields/paper_net/`).

Reproduces the *original* net + inference from
`references/ChempyMulti/Multi-Star Inference with Chempy - Tutorial.ipynb` and `run_pymc3.py`,
ported to JAX + NumPyro. **Standardized inputs & outputs, sampled in standardized space.** This
is what made N=200 mix cleanly (r̂=1.00, ESS~4700, 0 div, ~61 s) where the raw model stalled.

| file | what it does |
|---|---|
| `config_paper.py` | Constants: paths (train/test/mock data, reference weights, `weights_paper.npz`, `posteriors/`, `plots/`). `ELS` (8 TNG els), `NEURONS=40`, `EPOCHS=3000`, `N_POLY=2` (T² feature). Prior means/widths (Λ `[-2.30,-2.75]`, Θ `[-0.30,0.55,0.50]`), standardized local bounds incl. `LOG_SFR_CRIT≈0.294`, `MIN/MAX_TIME`, `ELEM_ERR_BETA_PHYS=0.01`, `TRUE_LAMBDA=[-2.30,-2.89]`, `N_STARS_LIST=[1,10,200]`. |
| `train_net.py` | **Run once.** Replicates notebook cells 22–40 on `TNG_Training_Data.npz` (500k): filter failed/inf/birth-time≤0.99, standardize (birth-time→[0,1]), add T², train 8 per-element `MLPRegressor((40,), tanh, adam, early_stopping)`, stack into block-sparse `(w0,b0,w1,b1)`. Verifies standardization is **bit-identical** to the authors' weights (max|diff|=0.0); test L1 median **0.0051 dex**. Saves `weights_paper.npz`. |
| `emulator_paper.py` | JAX standardized forward pass: `tanh(xin@w0+b0)@w1+b1` with T² appended. `load_weights` casts to float32. `predict_phys` de-standardizes for APE checks. |
| `model_paper.py` | NumPyro model, port of `run_pymc3.py`, **all sampling in standardized space**. `Lambda_std ~ Normal` (unbounded); per-star `Theta_std`/`time_std` truncated to run_pymc3 interval bounds (Θ±5σ, log10_sfr_scale ≥ SFR-crit edge, time∈std[1,13.8]); `elem_err ~ HalfCauchy(0.01/out_std)`; Gaussian likelihood on standardized abundances. Physical `Lambda`/`Theta`/`time`/`pred` recorded as deterministics. Centered (default) + `--noncentered`. |
| `run_hmc_paper.py` | One fit on one autocvd GPU. Args like the parent plus `--noncentered` (default **centered**), `--advi_init` (`advi+adapt_diag` warm-start, default in practice), `--advi_steps --advi_lr --dense_mass --chain_method`. Saves `posteriors/<tag>/posterior_N{n}.nc` + `.npz`. |
| `run_all_paper.py` | Launch N=1,10,200 in parallel, one autocvd GPU each. |
| `diagnostics_paper.py` | Same diagnostics as parent (r̂/ESS/BFMI, trace/rank/energy/pair, summary.txt) for paper-net posteriors → `plots/N{n}/`. |
| `plot_n_stars_paper.py` | Paper-style n_stars figure (Λ median ±1/2σ vs N) for the paper-net posteriors → `plots/n_stars_paper.png`. Also prints the numbers. |
| `plot_compare.py` | n_stars figure, **paper-net only** (the raw PyTorch series was removed per request) → `plots/compare_n_stars.png`. |
| `corner_gauss_paper.py` | Corner plot of paper-net `Lambda` with Gaussian (mean+cov) overlay + printed ρ/σ. `--n_stars --tag` → `plots/N{n}/corner_gauss_N{n}.png`. |
| `weights_paper.npz` | Retrained network + standardization constants (44 KB, **committed to git**). |
| `README.md` | Full write-up of the paper-faithful experiment (tables, validation, results). |

### Key result / finding

Both experiments recover the **same** global Λ. N=200 paper-net: α_IMF −2.281±0.007,
log₁₀N_Ia −2.888±0.008 (ρ≈0.805). The α_IMF offset above nominal −2.30 is **physical** (the
mock is real IllustrisTNG data, not Chempy output at −2.30), not a sampling artifact.

**What fixed the MCMC (parameter-space changes):** standardizing every latent to ~O(1) scale
(inputs `(x−in_mean)/in_std`, birth-time→[0,1]) makes the posterior near-isotropic so NUTS's
diagonal mass matrix + single step size fits all ~800 dims; standardizing outputs balances the
8-element likelihood residuals; the more accurate net (0.005 dex vs raw ~2%) + T² give a smooth,
non-degenerate likelihood; ADVI warm-start starts near the mode. Together: raw r̂~1.3/ESS~10 →
paper-net r̂=1.00/ESS~4700.

## Model-comparison datasets (`mock_chempyTNG/`, `alt_yields/`)

These two datasets reuse the **exact same emulators** as `TNG_set` (parent `weights.npz` and
paper_net `weights_paper.npz` are copied in, **not retrained** — the emulator is held fixed;
only the observations change). `alt_yields` is thus a deliberate **model-misspecification test**
(observations generated with alternative nucleosynthesis yields, fit with the standard-yields
emulator). Both `parent/config.py`/`paper_net/config_paper.py` only differ from `TNG_set` by:

- **`MOCK_DATA`** → `references/GCE_compass/data/MA_data/SBI_Chempy/model_comp_data/`
  `mock_abundances.npy` (bare `(200,8)` float32) / `chempy_alternative_yields_obs.npz["arr_0"]`.
- The observation files are **bare `(200,8)` abundance arrays already in `OBS_ELEMENTS`/`ELS`
  order** (H already dropped) — no `elements`/`errs`/`time` keys. The adapted `load_mock` reads
  them directly (auto-detects `.npy` vs `.npz`), no element alignment.
- **`OBS_ERR = 0.05`** — fixed 5% observational error (the notebook builds these mocks with
  `pc_ab=5`), used flat for every element in place of the missing per-star `abundance_errs`.
- **Uniform birth-time prior** `Uniform[2.0, 12.8]` Gyr (`TIME_LOW/HIGH`, and
  `MIN/MAX_TIME` in std space) instead of the TNG per-star age-centred TruncatedNormal, because
  these mocks carry no ages. This is the HMC analog of the notebook's SBI marginalizing over the
  birth-time nuisance. `model()`/`model_paper()` sample `time`/`time_std` from Uniform in both
  the centered and non-centered paths; `load_mock` still returns unused `time_mu`/`time_sd`
  placeholders so the run scripts are unchanged.
- Reference notebook: `references/GCE_compass/MA_scripts/inference_mock.ipynb` (how the mocks are
  loaded/fed to the SBI emulator — confirms the shared 8-column ordering).

Status: **scaffolded and dry-checked (imports + `load_mock` shapes + a CPU numpyro trace of both
parameterizations pass); no fits have been run yet.**

## Data / reference locations (read-only)

- `references/ChempyMulti/` — original tutorial notebook, `run_pymc3.py` (inference model),
  `tutorial_data/Mock_Data_TNG.npz` (1000-star TNG mock), `TNG_Training_Data.npz`,
  `TNG_Network_Weights.npz` (authors' weights, for validation).
- `references/sbi_chemical_abundances/` — modern PyTorch emulator + its training data/val data.
- `references/GCE_compass/data/MA_data/SBI_Chempy/model_comp_data/` — the model-comparison mocks
  (`mock_abundances.npy`, `chempy_alternative_yields_obs.npz`) + a copy of the standard-yields
  `pytorch_state_dict_5sigma_uni_prior.pt` (byte-identical to the parent emulator's state dict).
- `.gitignore` excludes the large `posterior_N200.nc` (regenerable); `weights_paper.npz` is committed.

## Git

- Branch `main`. Latest: `35c4aa1` adds the paper-faithful experiment.
- Note: pushing over HTTPS has failed in-session ("No anonymous write access"). Pushes need the
  user's own credentials — have them run `! git push origin main` after auth.
