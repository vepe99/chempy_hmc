"""NumPyro hierarchical multi-star model — raw abundance space, PyTorch emulator.

Adapted for the model-comparison mocks (mock_chempyTNG / alt_yields). The
observation file is a bare (N, 8) abundance array already in config.OBS_ELEMENTS
order, with NO per-star ages or errors. We therefore use a fixed 5% observational
error (config.OBS_ERR) and a *Uniform* birth-time prior over
[config.TIME_LOW, config.TIME_HIGH] Gyr (no age likelihood) — the HMC analog of the
notebook's SBI marginalizing over the birth-time nuisance. Everything else (the
emulator, the Lambda/Theta priors) is identical to the TNG_set parent experiment.
"""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import config
from emulator import predict_obs


def load_mock(n_stars, mock_path=config.MOCK_DATA):
    """First `n_stars` of the mock: a bare (N, 8) abundance array already in
    config.OBS_ELEMENTS order (no element alignment needed).

    These mocks ship no ages/errors, so obs_err is the fixed 5% error and the
    returned time_mu/time_sd are unused placeholders (the birth-time prior is
    Uniform in model()); they are kept only so run_hmc.py needs no change.
    """
    arr = np.load(mock_path, allow_pickle=True)
    if hasattr(arr, "files"):                          # .npz -> first stored array
        arr = arr[arr.files[0]]
    obs = np.asarray(arr, dtype=np.float64)[:n_stars]  # (n, 8)
    n = obs.shape[0]
    mid = 0.5 * (config.TIME_LOW + config.TIME_HIGH)
    return {
        "obs": obs,
        "obs_err": np.full_like(obs, config.OBS_ERR),  # 5% flat (notebook pc_ab=5)
        "time_mu": np.full(n, mid),                    # unused (Uniform time prior)
        "time_sd": np.full(n, config.TIME_HIGH - config.TIME_LOW),  # unused
    }


def model(w, obs, obs_err, time_mu, time_sd, elem_err=True, centered=True):
    # time_mu / time_sd are accepted for run_hmc.py signature compatibility but are
    # unused here: the birth-time prior is Uniform[config.TIME_LOW, config.TIME_HIGH].
    n_stars = obs.shape[0]

    lam_mean = jnp.asarray(config.LAMBDA_MEAN)
    lam_std = jnp.asarray(config.LAMBDA_STD)
    th_mean = jnp.asarray(config.THETA_MEAN)
    th_std = jnp.asarray(config.THETA_STD)

    # --- Global parameters Lambda (shared across all stars) ---
    Lambda = numpyro.sample(
        "Lambda",
        dist.TruncatedNormal(lam_mean, lam_std,
                             low=jnp.asarray(config.LAMBDA_LOW),
                             high=jnp.asarray(config.LAMBDA_HIGH)).to_event(1),
    )  # (2,)

    # --- Local parameters per star: Theta (3) and time (1) ---
    if centered:
        # Centered parameterization: sample the physical locals directly.
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta = numpyro.sample(
                "Theta",
                dist.TruncatedNormal(th_mean, th_std,
                                     low=jnp.asarray(config.THETA_LOW),
                                     high=jnp.asarray(config.THETA_HIGH)).to_event(1),
            )  # (n_stars, 3)
            time = numpyro.sample(
                "time", dist.Uniform(config.TIME_LOW, config.TIME_HIGH))  # (n_stars,)
    else:
        # Non-centered: decouple Theta from the shared Lambda (unsticks high-N chains).
        # The birth-time prior is flat, so "time_raw" is a standard Uniform mapped
        # affinely onto [TIME_LOW, TIME_HIGH] (no centering issue for a flat prior).
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta_raw = numpyro.sample(
                "Theta_raw", dist.Normal(0.0, 1.0).expand([3]).to_event(1))  # (n,3)
            time_u = numpyro.sample("time_raw", dist.Uniform(0.0, 1.0))      # (n,)
        Theta = numpyro.deterministic("Theta", th_mean + th_std * Theta_raw)
        time = numpyro.deterministic(
            "time", config.TIME_LOW + (config.TIME_HIGH - config.TIME_LOW) * time_u)

    # --- Emulator forward pass ---
    Lam_b = jnp.broadcast_to(Lambda, (n_stars, 2))
    x = jnp.concatenate([Lam_b, Theta, time[:, None]], axis=1)  # (n_stars, 6)
    pred = predict_obs(x, w)  # (n_stars, 8)
    numpyro.deterministic("pred", pred)

    # --- Error model ---
    obs_err = jnp.asarray(obs_err)
    if elem_err:
        model_err = numpyro.sample(
            "model_err",
            dist.HalfCauchy(config.MODEL_ERR_BETA * jnp.ones(obs.shape[1])).to_event(1),
        )  # (8,)
        tot_err = jnp.sqrt(obs_err ** 2 + model_err[None, :] ** 2)
    else:
        tot_err = obs_err

    numpyro.sample("obs", dist.Normal(pred, tot_err).to_event(2), obs=jnp.asarray(obs))
