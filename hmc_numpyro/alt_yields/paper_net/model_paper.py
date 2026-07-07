"""NumPyro hierarchical multi-star model — paper-faithful (standardized space).

Adapted for the model-comparison mocks (mock_chempyTNG / alt_yields). Identical to
the TNG_set paper_net model EXCEPT the observations are a bare (N, 8) abundance
array already in C.ELS order with no ages/errors, so:
  * obs_err is the fixed 5% observational error (C.OBS_ERR);
  * the birth-time latent gets a *Uniform* prior over [C.MIN_TIME, C.MAX_TIME] Gyr
    (standardized), instead of a per-star age-centred TruncatedNormal.
The (standard-yields) retrained net and all standardization constants are unchanged.
"""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import config_paper as C


def load_mock(n_stars, mock_path=C.MOCK_DATA):
    """First `n_stars`: a bare (N, 8) abundance array already in C.ELS order.

    No ages/errors ship with these mocks, so obs_err is the fixed 5% error and the
    returned time_mu/time_sd are unused placeholders (the birth-time prior is
    Uniform in model()); they are kept only so run_hmc_paper.py needs no change.
    """
    arr = np.load(mock_path, allow_pickle=True)
    if hasattr(arr, "files"):                          # .npz -> first stored array
        arr = arr[arr.files[0]]
    obs = np.asarray(arr, dtype=np.float64)[:n_stars]  # (n, 8)
    n = obs.shape[0]
    mid = 0.5 * (C.MIN_TIME + C.MAX_TIME)
    return {
        "obs": obs,
        "obs_err": np.full_like(obs, C.OBS_ERR),       # 5% flat (notebook pc_ab=5)
        "time_mu": np.full(n, mid),                    # unused (Uniform time prior)
        "time_sd": np.full(n, C.MAX_TIME - C.MIN_TIME),  # unused
    }


def model(w, obs, obs_err, time_mu, time_sd, elem_err=True, centered=True):
    # time_mu / time_sd are accepted for run_hmc_paper.py signature compatibility but
    # unused: the birth-time prior is Uniform[C.MIN_TIME, C.MAX_TIME] (standardized).
    from emulator_paper import net_std

    n_stars = obs.shape[0]
    in_mean, in_std = w["in_mean"], w["in_std"]
    out_mean, out_std = w["out_mean"], w["out_std"]

    # --- Standardize priors (run_pymc3 lines 67-78) ---
    lam_mean = (jnp.asarray(C.LAMBDA_MEAN) - in_mean[:2]) / in_std[:2]
    lam_width = jnp.asarray(C.LAMBDA_STD) / in_std[:2]
    th_mean = (jnp.asarray(C.THETA_MEAN) - in_mean[2:5]) / in_std[2:5]
    th_width = jnp.asarray(C.THETA_STD) / in_std[2:5]

    std_sfr_crit = (C.LOG_SFR_CRIT - in_mean[3]) / in_std[3]
    th_low = jnp.array([C.THETA_STD_LOW[0], std_sfr_crit, C.THETA_STD_LOW[2]])
    th_high = jnp.asarray(C.THETA_STD_HIGH)

    # Uniform birth-time prior bounds, standardized like the network input.
    t_min = (C.MIN_TIME - in_mean[5]) / in_std[5]
    t_max = (C.MAX_TIME - in_mean[5]) / in_std[5]

    # --- Standardize data ---
    obs_s = (jnp.asarray(obs) - out_mean) / out_std
    err_s = jnp.asarray(obs_err) / out_std

    # --- Global Lambda (standardized; unbounded, exactly as run_pymc3) ---
    Lambda_s = numpyro.sample("Lambda_std",
                              dist.Normal(lam_mean, lam_width).to_event(1))  # (2,)
    numpyro.deterministic("Lambda", Lambda_s * in_std[:2] + in_mean[:2])

    # --- Local Theta (3) + time (1) ---
    if centered:
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta_s = numpyro.sample(
                "Theta_std",
                dist.TruncatedNormal(th_mean, th_width, low=th_low, high=th_high)
                .to_event(1))                                    # (n, 3)
            time_s = numpyro.sample(
                "time_std", dist.Uniform(t_min, t_max))          # (n,) Uniform prior
    else:
        # Non-centered: sample z ~ N(0,1) for Theta and affine-map; the birth-time
        # prior is flat, so "time_raw" is a standard Uniform mapped onto [t_min,t_max].
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta_raw = numpyro.sample("Theta_raw",
                                       dist.Normal(0.0, 1.0).expand([3]).to_event(1))
            time_u = numpyro.sample("time_raw", dist.Uniform(0.0, 1.0))
        Theta_s = jnp.clip(th_mean + th_width * Theta_raw, th_low, th_high)
        time_s = t_min + (t_max - t_min) * time_u

    numpyro.deterministic("Theta", Theta_s * in_std[2:5] + in_mean[2:5])
    numpyro.deterministic("time", time_s * in_std[5] + in_mean[5])

    # --- Network forward (standardized) ---
    Lam_b = jnp.broadcast_to(Lambda_s, (n_stars, 2))
    x6 = jnp.concatenate([Lam_b, Theta_s, time_s[:, None]], axis=1)  # (n, 6) std
    out_s = net_std(x6, w)                                          # (n, 8) std
    numpyro.deterministic("pred", out_s * out_std + out_mean)

    # --- Error model (standardized) ---
    if elem_err:
        beta = C.ELEM_ERR_BETA_PHYS / out_std                      # (8,)
        elem_err_s = numpyro.sample("elem_err_std",
                                    dist.HalfCauchy(beta).to_event(1))  # (8,)
        numpyro.deterministic("elem_err", elem_err_s * out_std)
        tot_s = jnp.sqrt(err_s ** 2 + elem_err_s[None, :] ** 2)
    else:
        tot_s = err_s

    numpyro.sample("obs", dist.Normal(out_s, tot_s).to_event(2), obs=obs_s)
