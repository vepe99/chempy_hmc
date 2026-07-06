"""NumPyro hierarchical multi-star model — paper-faithful (standardized space).

Direct port of `ChempyMulti/run_pymc3.py`:
  * global  Lambda_std ~ Normal(std prior mean, std prior width)          (no bounds)
  * local   Theta_std, time_std ~ Normal, restricted to interval bounds
            (Theta[1]=log10_sfr_scale bounded below by the critical-SFR edge)
  * network run in standardized space, T^2-augmented (emulator_paper.net_std)
  * element error ~ HalfCauchy(beta=0.01/out_std), added in quadrature (std space)
  * Gaussian likelihood on standardized abundances.

All priors/data are standardized on the fly with the constants stored in the weights.
Physical Lambda / Theta / time are recorded as deterministics for reporting.
"""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import config_paper as C


def load_mock(n_stars, mock_path=C.MOCK_DATA):
    """First `n_stars` of the TNG mock, aligned to C.ELS (already the mock's order)."""
    d = np.load(mock_path, allow_pickle=True)
    mock_els = [str(e) for e in d["elements"]]
    idx = np.array([mock_els.index(e) for e in C.ELS])
    sl = slice(0, n_stars)
    return {
        "obs": np.asarray(d["abundances"])[sl][:, idx],       # (n, 8) physical
        "obs_err": np.asarray(d["abundance_errs"])[sl][:, idx],
        "time_mu": np.asarray(d["obs_time"])[sl],             # (n,) physical Gyr
        "time_sd": np.asarray(d["obs_time_err"])[sl],
        "true_time": np.asarray(d["true_time"])[sl],
    }


def model(w, obs, obs_err, time_mu, time_sd, elem_err=True, centered=True):
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

    t_min = (C.MIN_TIME - in_mean[5]) / in_std[5]
    t_max = (C.MAX_TIME - in_mean[5]) / in_std[5]
    time_mu_s = (jnp.asarray(time_mu) - in_mean[5]) / in_std[5]
    time_sd_s = jnp.asarray(time_sd) / in_std[5]

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
                "time_std",
                dist.TruncatedNormal(time_mu_s, time_sd_s, low=t_min, high=t_max))  # (n,)
    else:
        # Non-centered: sample z ~ N(0,1) and affine-map. The wide +/-5 sigma Theta
        # bounds are numerically negligible; the SFR-crit lower edge is kept as a soft
        # clip so predictions stay in the trained region.
        with numpyro.plate("stars", n_stars, dim=-1):
            Theta_raw = numpyro.sample("Theta_raw",
                                       dist.Normal(0.0, 1.0).expand([3]).to_event(1))
            time_raw = numpyro.sample("time_raw", dist.Normal(0.0, 1.0))
        Theta_s = jnp.clip(th_mean + th_width * Theta_raw, th_low, th_high)
        time_s = jnp.clip(time_mu_s + time_sd_s * time_raw, t_min, t_max)

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
