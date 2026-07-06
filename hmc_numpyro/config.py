"""Shared constants for the Chempy multi-star HMC.

All values are derived from `Chempy.parameter.ModelParameters` in the
`sbi_chemical_abundances` repo (the pipeline that trained the emulator) and are
hardcoded here so the runtime does not depend on the old Chempy package.

Emulator recap (sbi_chemical_abundances/01_train_torch_chempy.py):
  * 6 inputs : [high_mass_slope, log10_N_0,        <- global Lambda (2)
                log10_SFE, log10_sfr_scale, outflow_fraction,  <- local Theta (3)
                time]                                          <- local (1)
  * 9 outputs: ['C','Fe','H','He','Mg','N','Ne','O','Si']  ('H' at idx 2 is
                normalization-only and is dropped -> 8 elements matching the
                Mock_Data_TNG ordering exactly).
  * NO input/output standardization (unlike the old Theano run_pymc3.py).
"""
import os

# --- Paths (absolute, to the reference repos) ---
_REF = "/export/data/vgiusepp/chempy_hmc/references"
STATE_DICT = f"{_REF}/sbi_chemical_abundances/data/pytorch_state_dict_5sigma_uni_prior.pt"
VAL_DATA = f"{_REF}/sbi_chemical_abundances/data/chempy_data/chempy_TNG_val_data.npz"
MOCK_DATA = f"{_REF}/ChempyMulti/tutorial_data/Mock_Data_TNG.npz"

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NPZ = os.path.join(_HERE, "weights.npz")
POSTERIOR_DIR = os.path.join(_HERE, "posteriors")
PLOT_DIR = os.path.join(_HERE, "plots")

# --- Element bookkeeping ---
NN_ELEMENTS = ["C", "Fe", "H", "He", "Mg", "N", "Ne", "O", "Si"]  # 9 emulator outputs
H_INDEX = NN_ELEMENTS.index("H")                                   # = 2, dropped
OBS_ELEMENTS = [e for e in NN_ELEMENTS if e != "H"]               # 8, == TNG order

# --- Priors (a.priors means/widths from ModelParameters; modern pipeline) ---
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
TIME_LOW = 1.0
TIME_HIGH = 13.8

# --- Model-error prior (HalfCauchy per element, absorbs emulator + model error) ---
# Raw abundance space; weakly-informative at the few-percent emulator error level.
MODEL_ERR_BETA = 0.05

# --- "Ground truth" global params used as the reference line in the n_stars plot
#     (matches 04_plot_multistar_inference.ipynb global_params). ---
TRUE_LAMBDA = [-2.30, -2.89]

N_STARS_LIST = [1, 10, 200]
