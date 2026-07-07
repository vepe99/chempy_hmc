# Paper-faithful experiment — original ChempyMulti net, in JAX + NumPyro

This subpackage is a **second, independent experiment** that reproduces the *original*
ChempyMulti network and inference exactly as in
`references/ChempyMulti/Multi-Star Inference with Chempy - Tutorial.ipynb` and
`references/ChempyMulti/run_pymc3.py`, but ported to **JAX + NumPyro** (GPU NUTS).

It lives beside — and does not touch — the parent experiment (which uses the modern
raw-space PyTorch emulator from `sbi_chemical_abundances`). Full backward compatibility:
the parent `config.py` / `model.py` / `emulator.py` / `run_hmc.py` are unchanged.

## What "paper-faithful" means here

Everything that differs from the parent experiment is deliberately matched to the paper:

| aspect | parent (raw) | **this (paper-faithful)** |
|---|---|---|
| network | PyTorch 6→100→40→9, raw in/out | sklearn, **8 × (7→40→1)** tanh nets, block-sparse |
| standardization | none | **inputs & outputs standardized**; birth-time → [0,1] |
| input features | `[Λ, Θ, t]` (6) | `[Λ, Θ, t, t²]` (**7**, T² augmentation) |
| elements | 9 incl. H, drop H → 8 | **8** (`C,Fe,He,Mg,N,Ne,O,Si`), H never present |
| inference space | raw abundance | **standardized abundance** |
| Λ prior mean | `[-2.30, -2.89]` | `a.p0 = [-2.30, -2.75]` (run_pymc3) |
| local bounds | ±5σ truncation | interval bounds incl. **SFR-crit edge** on `log10_sfr_scale` |
| element error | `HalfCauchy(0.05)` raw | `HalfCauchy(0.01/out_std)` std |
| init | `init_to_median` | **ADVI warm-start** (`advi+adapt_diag`) |

The **only** modernizations are: the sampler (NumPyro NUTS w/ JAX autodiff instead of
PyMC3/Theano) and the training library (sklearn `MLPRegressor` on modern Python).

## Network training (`train_net.py`)

Reproduces notebook cells 22–40 on the same `TNG_Training_Data.npz` (500k samples):
select the 8 TNG elements (drop H), filter failed runs / infinities / birth-times ≤ 0.99,
standardize (birth-time mapped to [0,1]), add the T² feature, and train one
`MLPRegressor(hidden=(40,), activation='tanh', solver='adam', early_stopping)` per element,
then stack into the block-sparse `(w0,b0,w1,b1)` network.

**Validation:** the standardization constants (`in_mean/std`, `out_mean/std`) come out
**bit-identical** to the authors' `TNG_Network_Weights.npz` (max abs diff `0.0`), confirming
the data filtering + standardization match the paper exactly. Test-set network accuracy:
**median 0.0051 dex, mean 0.0084 dex** L1 error — matching the paper's few-millidex network.

## Model (`model_paper.py`) — direct port of `run_pymc3.py`

All sampling is in standardized space:

* `Lambda_std ~ Normal(std prior mean, std prior width)` — unbounded (as run_pymc3).
* per star: `Theta_std`, `time_std ~ Normal` restricted to the run_pymc3 interval bounds
  (Θ within ±5σ, `log10_sfr_scale` ≥ critical-SFR edge, time ∈ standardized [1, 13.8] Gyr).
* network run standardized + T²-augmented; `elem_err ~ HalfCauchy(0.01/out_std)`; Gaussian
  likelihood on standardized abundances.

Physical `Lambda`, `Theta`, `time` are recorded as deterministics for reporting.
A `--noncentered` reparameterization is available but centered (paper-faithful) is default.

## Results

Single hierarchical fit for N = 1, 10, 200 stars (first N of the 1000 TNG mock stars),
centered parameterization + ADVI init, 4 chains × 1000 draws (1000 warmup):

| N | α_IMF | log₁₀ N_Ia | r̂ | ESS (bulk) | div | BFMI |
|---|---|---|---|---|---|---|
| 1  | −2.000 ± 0.105 | −2.805 ± 0.108 | 1.00 | ~2600 | 0 | >0.8 |
| 10 | −2.249 ± 0.036 | −2.890 ± 0.037 | 1.00 | ~2600 | 0 | >0.8 |
| 200 | _(pending — see plots/N200/summary.txt)_ | | | | | |

These agree closely with the parent raw-emulator run (N=10: −2.25±0.04 / −2.88±0.04),
i.e. **the ~2% raw emulator and the standardized paper net recover the same global Λ.**
The α_IMF offset above the nominal −2.30 persists here too (physical, not a sampling
artifact — the mock is real IllustrisTNG data, not Chempy output at −2.30).

## Layout

```
config_paper.py       constants (paths, els, priors, std bounds)
train_net.py          replicate the notebook's sklearn training -> weights_paper.npz
emulator_paper.py     JAX standardized + T² forward pass
model_paper.py        NumPyro model (std space), port of run_pymc3.py
run_hmc_paper.py      one fit on one autocvd GPU  (--advi_init, --noncentered, --tag)
run_all_paper.py      launch N=1,10,200 in parallel
diagnostics_paper.py  r̂/ESS/BFMI, trace/rank/energy/pair per N
plot_compare.py       paper-net vs raw emulator, n_stars figure
weights_paper.npz     retrained network + standardization constants
posteriors/, plots/   outputs (posterior_N200.nc gitignored — large, regenerable)
```

Run: `python train_net.py` (once), then `python run_all_paper.py` (or per-N).
