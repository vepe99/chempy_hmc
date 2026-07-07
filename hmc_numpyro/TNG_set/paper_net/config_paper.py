"""Constants for the *paper-faithful* experiment.

This variant reproduces the ORIGINAL ChempyMulti network and inference exactly as
in `ChempyMulti/Multi-Star Inference with Chempy - Tutorial.ipynb` and
`ChempyMulti/run_pymc3.py`, but ported to JAX + NumPyro:

  * Network trained with sklearn (one MLPRegressor per element, tanh, 40 neurons),
    on the SAME `TNG_Training_Data.npz`, with the SAME standardization:
      - inputs  standardized  x_hat = (x - in_mean)/in_std, with the birth-time
        component mapped to [0,1]  (in_mean[-1]=min T, in_std[-1]=max T - min T);
      - the birth-time is augmented with a T^2 feature -> 7-dim network input;
      - outputs standardized  y_hat = (y - out_mean)/out_std;
      - 8 TNG elements ['C','Fe','He','Mg','N','Ne','O','Si']  (NO H at all).
  * Inference performed entirely in STANDARDIZED space, exactly like run_pymc3.py.

This lives in its own subpackage so the raw-emulator experiment (parent dir) is
untouched: full backward compatibility.
"""
import os

# --- Paths ---
_REF = "/export/data/vgiusepp/chempy_hmc/references"
TRAIN_DATA = f"{_REF}/ChempyMulti/tutorial_data/TNG_Training_Data.npz"
TEST_DATA = f"{_REF}/ChempyMulti/tutorial_data/TNG_Test_Data.npz"
MOCK_DATA = f"{_REF}/ChempyMulti/tutorial_data/Mock_Data_TNG.npz"
REF_WEIGHTS = f"{_REF}/ChempyMulti/tutorial_data/TNG_Network_Weights.npz"  # authors' net

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NPZ = os.path.join(_HERE, "weights_paper.npz")   # our retrained net
POSTERIOR_DIR = os.path.join(_HERE, "posteriors")
PLOT_DIR = os.path.join(_HERE, "plots")

# --- Elements (TNG order; H dropped, used only for normalization upstream) ---
ELS = ["C", "Fe", "He", "Mg", "N", "Ne", "O", "Si"]  # 8 outputs, matches Mock_Data_TNG

# --- Network / training hyper-parameters (from the notebook) ---
NEURONS = 40
EPOCHS = 3000
N_POLY = 2  # augment birth-time with T^2

# --- Priors (physical). run_pymc3.py uses a.p0 as the prior MEAN. ---
#   a.p0 = [high_mass_slope, log10_N_0, log10_SFE, log10_sfr_scale, outflow]
#        = [-2.3, -2.75, -0.3, 0.55, 0.5]      (parameter.py: SSP+ISM parameters)
LAMBDA_MEAN = [-2.30, -2.75]        # a.p0[:2]
LAMBDA_STD = [0.30, 0.30]           # run_pymc3 Lambda_prior_width
THETA_MEAN = [-0.30, 0.55, 0.50]    # a.p0[2:5]
THETA_STD = [0.30, 0.10, 0.10]      # run_pymc3 Theta_prior_width

# --- Interval bounds on the local params (run_pymc3 lowBound/upBound, std space) ---
#   Theta = [log10_SFE, log10_sfr_scale, outflow]; log10_sfr_scale >= log_SFR_crit.
THETA_STD_LOW = [-5.0, None, -5.0]   # None -> filled from LOG_SFR_CRIT at runtime
THETA_STD_HIGH = [5.0, 5.0, 5.0]
LOG_SFR_CRIT = 0.29402               # physical lower edge on log10_sfr_scale
MIN_TIME, MAX_TIME = 1.0, 13.8       # physical birth-time bounds

# --- Element-error prior: HalfCauchy(beta = 0.01/out_std) in std space ---
ELEM_ERR_BETA_PHYS = 0.01

# --- Reference "truth" line for plots (notebook cell 69: true=[-2.3,-2.89]) ---
TRUE_LAMBDA = [-2.30, -2.89]

N_STARS_LIST = [1, 10, 200]
