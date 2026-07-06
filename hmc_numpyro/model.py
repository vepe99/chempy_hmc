"""NumPyro hierarchical multi-star model (faithful to ChempyMulti/run_pymc3.py,
but in raw abundance space with the PyTorch emulator)."""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import config
from emulator import predict_obs


def load_mock(n_stars, mock_path=config.MOCK_DATA):
    """Return the first `n_stars` of the TNG mock, aligned to config.OBS_ELEMENTS."""
    d = np.load(mock_path, allow_pickle=True)
    mock_els = [str(e) for e in d["elements"]]
    idx = np.array([mock_els.index(e) for e in config.OBS_ELEMENTS])  # align order

    sl = slice(0, n_stars)
    data = {
        "obs": np.asarray(d["abundances"])[sl][:, idx],       # (n, 8)
        "obs_err": np.asarray(d["abundance_errs"])[sl][:, idx],  # (n, 8)
        "time_mu": np.asarray(d["obs_time"])[sl],             # (n,)
        "time_sd": np.asarray(d["obs_time_err"])[sl],         # (n,)
        "true_time": np.asarray(d["true_time"])[sl],
    }
    return data


def model(w, obs, obs_err, time_mu, time_sd, elem_err=True, centered=True):
    n_stars = obs.shape[0]

    lam_mean = jnp.asarray(config.LAMBDA_MEAN)
    lam_std = jnp.asarray(config.LAMBDA_STD)
    th_mean = jnp.asarray(config.THETA_MEAN)
    th_std = jnp.asarray(config.THETA_STD)
    time_mu = jnp.asarray(time_mu)
    time_sd = jnp.asarray(time_sd)

    # --- Global parameters Lambda (shared across all stars) ---
    Lambda = numpyro.sample(
        "Lambda",
        dist.TruncatedNormal(lam_mean, lam_std,
                             low=jnp.asarray(config.LAMBDA_LOW),
                             high=jnp.asarray(config.LAMBDA_HIGH)).to_event(1),
    )  # (2,)

    # --- Local parameters per star: Theta (3) and time (1) ---
    if centered:
        # Centered parameterization (original): sample the physical locals directly.
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta = numpyro.sample(
                "Theta",
                dist.TruncatedNormal(th_mean, th_std,
                                     low=jnp.asarray(config.THETA_LOW),
                                     high=jnp.asarray(config.THETA_HIGH)).to_event(1),
            )  # (n_stars, 3)
            time = numpyro.sample(
                "time",
                dist.TruncatedNormal(time_mu, time_sd,
                                     low=config.TIME_LOW, high=config.TIME_HIGH),
            )  # (n_stars,)
    else:
        # Non-centered parameterization: sample standardized z ~ N(0,1) and map to
        # physical units. Decouples the local geometry from the shared Lambda, which
        # is what unsticks the high-N chains. (The +/-5 sigma truncation of the
        # original priors is numerically negligible, so we use plain normals here;
        # outflow's [0,1] box == mean +/- 5 sigma, so N(0,1) essentially respects it.)
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta_raw = numpyro.sample(
                "Theta_raw", dist.Normal(0.0, 1.0).expand([3]).to_event(1))  # (n_stars,3)
            time_raw = numpyro.sample("time_raw", dist.Normal(0.0, 1.0))     # (n_stars,)
        Theta = numpyro.deterministic("Theta", th_mean + th_std * Theta_raw)
        time = numpyro.deterministic("time", time_mu + time_sd * time_raw)

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
