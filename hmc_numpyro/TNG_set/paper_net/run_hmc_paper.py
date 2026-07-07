"""Run one paper-faithful hierarchical NUTS fit for N stars on a single free GPU.

Standardized-space model (model_paper.py) + retrained standardized net (weights_paper.npz).
Defaults mirror run_pymc3.py: centered parameterization, target_accept=0.9,
optional ADVI warm-start (the paper's 'advi+adapt_diag' init).

Usage: python run_hmc_paper.py --n_stars 10 [--draws 1000 --tune 1000 --advi_init]
"""
import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_stars", type=int, required=True)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--max_tree_depth", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_elem_err", action="store_true")
    p.add_argument("--noncentered", action="store_true",
                   help="non-centered reparam (default: centered, faithful to run_pymc3)")
    p.add_argument("--tag", default="")
    p.add_argument("--advi_init", action="store_true",
                   help="warm-start NUTS from a mean-field ADVI fit ('advi+adapt_diag')")
    p.add_argument("--advi_steps", type=int, default=20000)
    p.add_argument("--advi_lr", type=float, default=1e-3)
    p.add_argument("--dense_mass", action="store_true")
    p.add_argument("--chain_method", default="vectorized",
                   choices=["vectorized", "parallel", "sequential"])
    return p.parse_args()


def main():
    args = parse_args()

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

    import config_paper as C
    from emulator_paper import load_weights
    from model_paper import model, load_mock

    numpyro.set_host_device_count(1)
    print(f"[N={args.n_stars}] devices: {jax.devices()} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    w = load_weights()
    data = load_mock(args.n_stars)
    elem_err = not args.no_elem_err
    centered = not args.noncentered
    param = "centered" if centered else "non-centered"
    mk = dict(w=w, obs=data["obs"], obs_err=data["obs_err"],
              time_mu=data["time_mu"], time_sd=data["time_sd"],
              elem_err=elem_err, centered=centered)
    print(f"[N={args.n_stars}] paper-faithful (standardized net), {param}, "
          f"chain_method={args.chain_method}, advi_init={args.advi_init}, "
          f"dense_mass={args.dense_mass}")

    init_strategy = init_to_median
    if args.advi_init:
        guide = AutoDiagonalNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(args.advi_lr), Trace_ELBO())
        t_advi = time.time()
        svi_res = svi.run(jax.random.PRNGKey(args.seed + 1), args.advi_steps, **mk)
        init_strategy = init_to_value(values=guide.median(svi_res.params))
        print(f"[N={args.n_stars}] ADVI init: {args.advi_steps} steps in "
              f"{time.time()-t_advi:.1f}s, ELBO loss={float(svi_res.losses[-1]):.1f}")

    kernel = NUTS(model, target_accept_prob=args.target_accept,
                  max_tree_depth=args.max_tree_depth, dense_mass=args.dense_mass,
                  init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=args.tune, num_samples=args.draws,
                num_chains=args.chains, chain_method=args.chain_method,
                progress_bar=True)

    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(args.seed), **mk,
             extra_fields=("diverging", "energy", "num_steps"))
    runtime = time.time() - t0
    print(f"[N={args.n_stars}] sampling finished in {runtime:.1f}s")

    idata = az.from_numpyro(mcmc)
    idata.attrs.update(dict(n_stars=args.n_stars, runtime_s=runtime,
                            elem_err=int(elem_err), draws=args.draws, tune=args.tune,
                            chains=args.chains, parameterization=param,
                            advi_init=int(args.advi_init),
                            dense_mass=int(args.dense_mass),
                            max_tree_depth=args.max_tree_depth,
                            target_accept=args.target_accept, variant="paper_net"))

    outdir = os.path.join(C.POSTERIOR_DIR, args.tag) if args.tag else C.POSTERIOR_DIR
    os.makedirs(outdir, exist_ok=True)
    nc = os.path.join(outdir, f"posterior_N{args.n_stars}.nc")
    idata.to_netcdf(nc)

    lam = np.asarray(idata.posterior["Lambda"]).reshape(-1, 2)
    npz = os.path.join(outdir, f"posterior_N{args.n_stars}.npz")
    np.savez(npz, Lambda=lam, n_stars=args.n_stars, runtime_s=runtime)

    ndiv = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    print(f"[N={args.n_stars}] divergences: {ndiv}")
    print(az.summary(idata, var_names=["Lambda"]))
    print(f"[N={args.n_stars}] saved -> {nc} , {npz}")


if __name__ == "__main__":
    main()
