"""Constants for the *standardized-parent* experiment (mock_chempyTNG).

This experiment applies the paper_net recipe to the RAW PyTorch emulator:
  * the network stays exactly the raw net from `parent/` (unnormalized raw params
    -> raw abundances, weights.npz is byte-copied, NOT retrained);
  * but every latent is SAMPLED IN STANDARDIZED SPACE and the 8-element likelihood
    is evaluated in STANDARDIZED SPACE, exactly like paper_net/model_paper.py.

The standardization constants (in_mean/in_std over the 6 inputs, out_mean/out_std
over the 8 obs elements) are computed once by build_standardization.py from the
same training distribution the raw net was trained on
(`chempy_train_uniform_prior_5sigma.npz`, matching pytorch_state_dict_5sigma_uni_prior.pt)
and saved to standardization.npz. The model de-standardizes latents -> feeds the raw
net -> standardizes outputs for the Gaussian likelihood.

Motivation: the raw `parent/` model samples in physical units and stalls at N=200
(r-hat ~ 1.3, ESS ~ 10). paper_net mixes cleanly BECAUSE it standardizes I/O so the
posterior geometry is near-isotropic for NUTS' diagonal mass matrix. This experiment
isolates whether that same reparameterization fixes the RAW net (same emulator, only
the sampling/likelihood space changes).

Everything else (priors, bounds, mock, 5% obs error, Uniform birth-time prior) is
identical to mock_chempyTNG/parent/config.py so the two are directly comparable.
"""
import os

# --- Paths (absolute, to the reference repos) ---
_REF = "/export/data/vgiusepp/chempy_hmc/references"
STATE_DICT = f"{_REF}/sbi_chemical_abundances/data/pytorch_state_dict_5sigma_uni_prior.pt"
VAL_DATA = f"{_REF}/sbi_chemical_abundances/data/chempy_data/chempy_TNG_val_data.npz"
# Training set the raw net was fit on (5-sigma uniform prior). Used ONLY by
# build_standardization.py to compute the input/output standardization constants.
TRAIN_DATA = f"{_REF}/sbi_chemical_abundances/data/chempy_data/chempy_train_uniform_prior_5sigma.npz"
# model-comparison mock: bare (200, 8) abundances (Chempy mock mimicking TNG),
# already in OBS_ELEMENTS order. Fit with the SAME (standard-yields) emulator.
_MC = f"{_REF}/GCE_compass/data/MA_data/SBI_Chempy/model_comp_data"
MOCK_DATA = f"{_MC}/mock_abundances.npy"

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NPZ = os.path.join(_HERE, "weights.npz")               # raw net (copied from parent/)
STANDARDIZATION_NPZ = os.path.join(_HERE, "standardization.npz")  # in/out mean+std
POSTERIOR_DIR = os.path.join(_HERE, "posteriors")
PLOT_DIR = os.path.join(_HERE, "plots")

# --- Element bookkeeping ---
NN_ELEMENTS = ["C", "Fe", "H", "He", "Mg", "N", "Ne", "O", "Si"]  # 9 emulator outputs
H_INDEX = NN_ELEMENTS.index("H")                                   # = 2, dropped
OBS_ELEMENTS = [e for e in NN_ELEMENTS if e != "H"]               # 8, == TNG order

# --- Priors (physical). Identical to parent/config.py. ---
# Global Lambda: high_mass_slope (alpha_IMF), log10_N_0 (log10 N_Ia)
LAMBDA_MEAN = [-2.30, -2.89]
LAMBDA_STD = [0.30, 0.30]
# Local Theta: log10_starformation_efficiency, log10_sfr_scale, outflow_feedback_fraction
THETA_MEAN = [-0.30, 0.55, 0.50]
THETA_STD = [0.30, 0.10, 0.10]

# --- Truncation bounds = emulator training ranges (uniform +/-5 sigma; time in [1,13.8]) ---
LAMBDA_LOW = [-3.80, -4.39]
LAMBDA_HIGH = [-0.80, -1.39]
THETA_LOW = [-1.80, 0.05, 0.00]
THETA_HIGH = [1.20, 1.05, 1.00]
# Birth-time prior bounds. These mocks carry no per-star ages, so time is given a
# Uniform[TIME_LOW, TIME_HIGH] prior in model_std() (HMC analog of the notebook's
# SBI marginalizing over the birth-time nuisance).
TIME_LOW = 2.0
TIME_HIGH = 12.8

# Fixed observational error (dex): the notebook builds these mocks with pc_ab=5%.
OBS_ERR = 0.05

# --- Model-error prior (HalfCauchy per element, absorbs emulator + model error) ---
# Physical (raw-abundance) scale; standardized by out_std inside model_std().
MODEL_ERR_BETA = 0.05

# --- "Ground truth" global params used as the reference line in the n_stars plot ---
TRUE_LAMBDA = [-2.30, -2.89]

N_STARS_LIST = [1, 10, 200]
