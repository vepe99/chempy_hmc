"""NumPyro hierarchical multi-star model — RAW net, STANDARDIZED sampling space.

This is the experiment: take the raw PyTorch emulator (parent/) unchanged, but apply
the paper_net recipe so NUTS sees a near-isotropic geometry. Every latent is sampled
standardized and the 8-element Gaussian likelihood is evaluated standardized:

    Lambda_std ~ Normal(std prior)            (global; unbounded, like run_pymc3)
    Theta_std  ~ TruncatedNormal(std prior, std bounds)   (per star)
    time_std   ~ Uniform(std bounds)          (per star; no ages in these mocks)
      -> de-standardize to physical (x = x_std*in_std + in_mean)
      -> raw net  predict_obs(x_phys)  -> raw abundances
      -> standardize output  out_std = (y_raw - out_mean)/out_std
      -> Normal likelihood on standardized data

vs. parent/model.py (physical sampling, raw likelihood). Same raw net, same priors,
same mock, same 5% obs error, same Uniform birth-time prior — ONLY the sampling and
likelihood space differ, isolating whether standardization alone unsticks the raw net
at N=200 (parent: r-hat ~ 1.3, ESS ~ 10).

The observation file is a bare (N, 8) abundance array already in config_std.OBS_ELEMENTS
order, with no per-star ages or errors (see mock_chempyTNG parent/paper_net models).
"""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import config_std as C
from emulator_std import predict_obs


def load_mock(n_stars, mock_path=C.MOCK_DATA):
    """First `n_stars` of the mock: a bare (N, 8) abundance array already in
    config_std.OBS_ELEMENTS order (no element alignment needed).

    These mocks ship no ages/errors, so obs_err is the fixed 5% error and the returned
    time_mu/time_sd are unused placeholders (the birth-time prior is Uniform in
    model_std()); they are kept only so run_hmc_std.py needs no change.
    """
    arr = np.load(mock_path, allow_pickle=True)
    if hasattr(arr, "files"):                          # .npz -> first stored array
        arr = arr[arr.files[0]]
    obs = np.asarray(arr, dtype=np.float64)[:n_stars]  # (n, 8)
    n = obs.shape[0]
    mid = 0.5 * (C.TIME_LOW + C.TIME_HIGH)
    return {
        "obs": obs,
        "obs_err": np.full_like(obs, C.OBS_ERR),       # 5% flat (notebook pc_ab=5)
        "time_mu": np.full(n, mid),                    # unused (Uniform time prior)
        "time_sd": np.full(n, C.TIME_HIGH - C.TIME_LOW),  # unused
    }


def model(w, obs, obs_err, time_mu, time_sd, elem_err=True, centered=True):
    # time_mu / time_sd accepted for run_hmc_std.py signature compatibility but unused:
    # the birth-time prior is Uniform[C.TIME_LOW, C.TIME_HIGH] (in standardized space).
    n_stars = obs.shape[0]
    in_mean, in_std = w["in_mean"], w["in_std"]      # (6,)
    out_mean, out_std = w["out_mean"], w["out_std"]  # (8,)

    # --- Standardize the physical priors & bounds (like paper_net/model_paper.py) ---
    lam_mean = (jnp.asarray(C.LAMBDA_MEAN) - in_mean[:2]) / in_std[:2]
    lam_width = jnp.asarray(C.LAMBDA_STD) / in_std[:2]

    th_mean = (jnp.asarray(C.THETA_MEAN) - in_mean[2:5]) / in_std[2:5]
    th_width = jnp.asarray(C.THETA_STD) / in_std[2:5]
    th_low = (jnp.asarray(C.THETA_LOW) - in_mean[2:5]) / in_std[2:5]
    th_high = (jnp.asarray(C.THETA_HIGH) - in_mean[2:5]) / in_std[2:5]

    # Uniform birth-time prior bounds, standardized like the network input.
    t_min = (C.TIME_LOW - in_mean[5]) / in_std[5]
    t_max = (C.TIME_HIGH - in_mean[5]) / in_std[5]

    # --- Standardize the data (output space) ---
    obs_s = (jnp.asarray(obs) - out_mean) / out_std
    err_s = jnp.asarray(obs_err) / out_std

    # --- Global Lambda (standardized; unbounded Normal, exactly as run_pymc3) ---
    Lambda_s = numpyro.sample("Lambda_std",
                              dist.Normal(lam_mean, lam_width).to_event(1))  # (2,)
    numpyro.deterministic("Lambda", Lambda_s * in_std[:2] + in_mean[:2])

    # --- Local Theta (3) + time (1) per star ---
    if centered:
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta_s = numpyro.sample(
                "Theta_std",
                dist.TruncatedNormal(th_mean, th_width, low=th_low, high=th_high)
                .to_event(1))                                    # (n, 3)
            time_s = numpyro.sample(
                "time_std", dist.Uniform(t_min, t_max))          # (n,) Uniform prior
    else:
        # Non-centered: sample z ~ N(0,1) for Theta and affine-map (clip to std bounds);
        # the birth-time prior is flat, so "time_raw" is a standard Uniform mapped on.
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta_raw = numpyro.sample("Theta_raw",
                                       dist.Normal(0.0, 1.0).expand([3]).to_event(1))
            time_u = numpyro.sample("time_raw", dist.Uniform(0.0, 1.0))
        Theta_s = jnp.clip(th_mean + th_width * Theta_raw, th_low, th_high)
        time_s = t_min + (t_max - t_min) * time_u

    numpyro.deterministic("Theta", Theta_s * in_std[2:5] + in_mean[2:5])
    numpyro.deterministic("time", time_s * in_std[5] + in_mean[5])

    # --- Emulator forward: de-standardize latents -> RAW net -> standardize output ---
    Lam_b = jnp.broadcast_to(Lambda_s, (n_stars, 2))
    x6_s = jnp.concatenate([Lam_b, Theta_s, time_s[:, None]], axis=1)  # (n, 6) std
    x6_phys = x6_s * in_std + in_mean                                  # (n, 6) raw
    pred_phys = predict_obs(x6_phys, w)                               # (n, 8) raw
    out_s = (pred_phys - out_mean) / out_std                          # (n, 8) std
    numpyro.deterministic("pred", pred_phys)

    # --- Error model (standardized) ---
    if elem_err:
        beta = C.MODEL_ERR_BETA / out_std                            # (8,)
        elem_err_s = numpyro.sample("elem_err_std",
                                    dist.HalfCauchy(beta).to_event(1))  # (8,)
        numpyro.deterministic("model_err", elem_err_s * out_std)
        tot_s = jnp.sqrt(err_s ** 2 + elem_err_s[None, :] ** 2)
    else:
        tot_s = err_s

    numpyro.sample("obs", dist.Normal(out_s, tot_s).to_event(2), obs=obs_s)
