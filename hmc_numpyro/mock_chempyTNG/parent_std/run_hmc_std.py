"""Run one hierarchical HMC (NUTS) fit for N stars on a single free GPU.

Standardized-parent experiment: the RAW emulator, sampled/likelihood'd in STANDARDIZED
space (see model_std.py). Usage:
    python run_hmc_std.py --n_stars 200 [--draws 1000 --tune 2000 --advi_init --dense_mass]
autocvd picks a free GPU (waiting if necessary) BEFORE JAX is imported.
"""
import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_stars", type=int, required=True)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--max_tree_depth", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_elem_err", action="store_true")
    p.add_argument("--centered", action="store_true",
                   help="use the centered parameterization (default: non-centered reparam)")
    p.add_argument("--tag", default="",
                   help="subdir under posteriors/ to save into (e.g. 200_3k); keeps experiments separate")
    p.add_argument("--advi_init", action="store_true",
                   help="warm-start NUTS from a mean-field ADVI (AutoDiagonalNormal) fit")
    p.add_argument("--advi_steps", type=int, default=20000)
    p.add_argument("--advi_lr", type=float, default=1e-3)
    p.add_argument("--dense_mass", action="store_true",
                   help="use a dense NUTS mass matrix (preconditions the correlated ridge)")
    p.add_argument("--chain_method", default="vectorized",
                   choices=["vectorized", "parallel", "sequential"],
                   help="vectorized = all chains in parallel on one GPU via vmap")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Grab a free GPU BEFORE importing JAX ---
    from autocvd import autocvd
    autocvd(num_gpus=1)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import numpy as np
    import arviz as az
    import numpyro
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDiagonalNormal
    from numpyro.infer.initialization import init_to_value, init_to_median

    import config_std as config
    from emulator_std import load_weights
    from model_std import model, load_mock

    numpyro.set_host_device_count(1)
    print(f"[N={args.n_stars}] devices: {jax.devices()} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    w = load_weights()
    data = load_mock(args.n_stars)
    elem_err = not args.no_elem_err
    centered = args.centered
    param = "centered" if centered else "non-centered"
    model_kwargs = dict(w=w, obs=data["obs"], obs_err=data["obs_err"],
                        time_mu=data["time_mu"], time_sd=data["time_sd"],
                        elem_err=elem_err, centered=centered)
    print(f"[N={args.n_stars}] parameterization: {param}, chain_method={args.chain_method}, "
          f"advi_init={args.advi_init}, dense_mass={args.dense_mass}")

    # --- Optional ADVI (mean-field) warm-start: fit an AutoDiagonalNormal guide and
    #     use its median to initialize NUTS (paper-faithful 'advi+...' initialization). ---
    init_strategy = init_to_median
    if args.advi_init:
        guide = AutoDiagonalNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(args.advi_lr), Trace_ELBO())
        t_advi = time.time()
        svi_res = svi.run(jax.random.PRNGKey(args.seed + 1), args.advi_steps, **model_kwargs)
        init_vals = guide.median(svi_res.params)  # dict of latent-site medians
        init_strategy = init_to_value(values=init_vals)
        print(f"[N={args.n_stars}] ADVI init: {args.advi_steps} steps in "
              f"{time.time()-t_advi:.1f}s, final ELBO loss={float(svi_res.losses[-1]):.1f}")

    kernel = NUTS(model, target_accept_prob=args.target_accept,
                  max_tree_depth=args.max_tree_depth,
                  dense_mass=args.dense_mass,
                  init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=args.tune, num_samples=args.draws,
                num_chains=args.chains, chain_method=args.chain_method,
                progress_bar=True)

    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(args.seed), **model_kwargs,
             extra_fields=("diverging", "energy", "num_steps"))
    runtime = time.time() - t0
    print(f"[N={args.n_stars}] sampling finished in {runtime:.1f}s")

    idata = az.from_numpyro(mcmc)
    idata.attrs.update(dict(n_stars=args.n_stars, runtime_s=runtime,
                            elem_err=int(elem_err), draws=args.draws,
                            tune=args.tune, chains=args.chains,
                            parameterization=param,
                            advi_init=int(args.advi_init),
                            dense_mass=int(args.dense_mass),
                            max_tree_depth=args.max_tree_depth,
                            target_accept=args.target_accept))

    # Optional --tag routes outputs into posteriors/<tag>/ so experiments stay separate.
    outdir = os.path.join(config.POSTERIOR_DIR, args.tag) if args.tag else config.POSTERIOR_DIR
    os.makedirs(outdir, exist_ok=True)
    nc = os.path.join(outdir, f"posterior_N{args.n_stars}.nc")
    idata.to_netcdf(nc)

    # Plain npz of the (physical) global Lambda samples (flattened over chains).
    lam = np.asarray(idata.posterior["Lambda"]).reshape(-1, 2)
    npz = os.path.join(outdir, f"posterior_N{args.n_stars}.npz")
    np.savez(npz, Lambda=lam, n_stars=args.n_stars, runtime_s=runtime)

    # Divergences + a quick Lambda summary
    ndiv = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    print(f"[N={args.n_stars}] divergences: {ndiv}")
    print(az.summary(idata, var_names=["Lambda"]))
    print(f"[N={args.n_stars}] saved -> {nc} , {npz}")


if __name__ == "__main__":
    main()
