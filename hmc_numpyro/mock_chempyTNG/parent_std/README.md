# `parent_std/` — raw emulator, standardized sampling (mock_chempyTNG)

**Question this experiment answers:** the `parent/` method (raw PyTorch emulator, sampled in
physical units) mixes badly at N=200 (r̂ ~ 1.3, ESS ~ 10). `paper_net/` mixes cleanly
(r̂ = 1.00, ESS ~ 4700) — but it changes *two* things at once: a different, more accurate
retrained network **and** standardized sampling/likelihood. This experiment isolates the
**standardization** by keeping the **exact same raw net** and only moving the sampling +
likelihood into standardized space.

## What is (and isn't) changed vs `parent/`

| | `parent/` | `parent_std/` (this) |
|---|---|---|
| Network | raw PyTorch net (`weights.npz`) | **same** raw net (`weights.npz`, byte-copied) |
| Emulator forward | raw params → raw abundances | **same** raw forward |
| `Lambda` sampled in | physical units, `TruncatedNormal` | **standardized**, unbounded `Normal` |
| `Theta`/`time` sampled in | physical units | **standardized** (bounds standardized too) |
| Likelihood evaluated in | raw abundance space | **standardized** abundance space |
| Priors / bounds / mock / 5% obs err / Uniform time prior | — | **identical** |

The network weights are **not** retrained. Standardization lives only in the sampling and
likelihood space: the model samples standardized latents, **de-standardizes** them to physical
units, feeds the **raw** net, then **re-standardizes** the raw output for a well-conditioned
Gaussian likelihood — the paper_net recipe (`paper_net/model_paper.py`) applied to the raw net.

## Standardization constants

Computed once by `build_standardization.py` from `chempy_train_uniform_prior_5sigma.npz` — the
exact training distribution the raw net (`pytorch_state_dict_5sigma_uni_prior.pt`) was fit on —
and saved to `standardization.npz`:

- `in_mean/in_std` (6,) over the 6 raw inputs `[α_IMF, log₁₀N₀, log₁₀SFE, log₁₀sfr_scale, outflow, time]`
- `out_mean/out_std` (8,) over the 8 obs elements (H dropped, `OBS_ELEMENTS` order)

`in_std` for Λ is ≈ 0.866 = 1.5/√3, i.e. the std of the ±5σ uniform training prior — a good
consistency check.

## Files

| file | what it does |
|---|---|
| `config_std.py` | Constants. Physical priors/bounds identical to `parent/config.py`; adds `TRAIN_DATA` and `STANDARDIZATION_NPZ` paths. |
| `build_standardization.py` | **Run once.** Computes `in_mean/in_std/out_mean/out_std` from the 5σ training data (drops NaN/inf rows) → `standardization.npz`. Only file that reads the big training file. |
| `emulator_std.py` | Raw net forward (identical to `parent/emulator.py`); `load_weights` bundles raw weights + standardization constants into one dict. |
| `model_std.py` | The experiment: standardized-space hierarchical model on the raw net (de-standardize → raw net → re-standardize). Centered (default arg) + non-centered. |
| `run_hmc_std.py` | One fit for N stars on one autocvd GPU. Same CLI as `parent/run_hmc.py` (`--advi_init`, `--dense_mass`, `--centered`, …). Saves `posteriors/<tag>/posterior_N{n}.nc` + `.npz`. |
| `run_all_std.py` | Launch N=1,10,200 in parallel, one autocvd GPU each. |
| `weights.npz` | Raw net (byte-copied from `parent/weights.npz`). |
| `standardization.npz` | Input/output mean+std (produced by `build_standardization.py`). |

## How to run

```bash
cd mock_chempyTNG/parent_std
python build_standardization.py          # once → standardization.npz
python run_all_std.py --advi_init --tag advi   # N=1,10,200, one GPU each
```

## Results

_(to be filled in once the fits complete — compare N=200 r̂/ESS/divergences against
`parent/` (raw, physical) and `paper_net/` (retrained, standardized).)_
