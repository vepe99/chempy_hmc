"""MCMC health diagnostics for a paper-net posterior (.nc InferenceData).

Usage: python diagnostics_paper.py --n_stars 10 [--tag ...]
Outputs -> plots/N{n}/ (or plots/<tag>/): summary.txt, trace/rank/autocorr/energy/pair/posterior.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

import config_paper as C

LAM_LABELS = [r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_stars", type=int, required=True)
    p.add_argument("--tag", default="")
    return p.parse_args()


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.relpath(path, C.PLOT_DIR))


def main():
    args = parse_args()
    n = args.n_stars
    base = os.path.join(C.POSTERIOR_DIR, args.tag) if args.tag else C.POSTERIOR_DIR
    idata = az.from_netcdf(os.path.join(base, f"posterior_N{n}.nc"))

    outdir = os.path.join(C.PLOT_DIR, args.tag if args.tag else f"N{n}")
    os.makedirs(outdir, exist_ok=True)
    print(f"=== N={n} (paper_net) diagnostics -> {outdir} ===")

    summ = az.summary(idata, var_names=["Lambda"], stat_focus="mean", hdi_prob=0.95)
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_samples = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    try:
        bfmi = az.bfmi(idata)
    except Exception:
        bfmi = np.array([np.nan])

    lam = np.asarray(idata.posterior["Lambda"]).reshape(-1, 2)
    pcts = np.percentile(lam, [2.5, 16, 50, 84, 97.5], axis=0)

    txt = [f"N_stars = {n}   (paper_net, standardized emulator)",
           f"total draws = {n_samples} "
           f"({idata.posterior.sizes['chain']} chains x {idata.posterior.sizes['draw']})",
           f"divergences = {div}  ({100*div/max(n_samples,1):.2f}%)",
           f"BFMI per chain = {np.array2string(np.asarray(bfmi), precision=3)}  (healthy > 0.3)",
           "", "Lambda summary (r_hat < 1.01; ESS > 400 healthy):", summ.to_string(), "",
           "Lambda percentiles [2.5, 16, 50, 84, 97.5]:",
           f"  alpha_IMF : {np.array2string(pcts[:,0], precision=4)}",
           f"  log10N_Ia : {np.array2string(pcts[:,1], precision=4)}"]
    report = "\n".join(txt)
    print(report)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(report + "\n")

    axes = az.plot_trace(idata, var_names=["Lambda"], compact=False)
    savefig(axes.ravel()[0].figure, os.path.join(outdir, "trace_Lambda.png"))
    axes = az.plot_rank(idata, var_names=["Lambda"])
    savefig(np.atleast_1d(axes).ravel()[0].figure, os.path.join(outdir, "rank_Lambda.png"))
    axes = az.plot_autocorr(idata, var_names=["Lambda"], combined=True)
    savefig(np.atleast_1d(axes).ravel()[0].figure, os.path.join(outdir, "autocorr_Lambda.png"))
    try:
        ax = az.plot_energy(idata)
        savefig(ax.figure, os.path.join(outdir, "energy.png"))
    except Exception as e:
        print("  energy plot skipped:", e)
    try:
        axes = az.plot_pair(idata, var_names=["Lambda"], kind="kde",
                            divergences=True, marginals=True)
        savefig(np.atleast_1d(axes).ravel()[0].figure, os.path.join(outdir, "pair_Lambda.png"))
    except Exception as e:
        print("  pair plot skipped:", e)
    ax = az.plot_posterior(idata, var_names=["Lambda"],
                           ref_val=list(C.TRUE_LAMBDA), hdi_prob=0.95)
    savefig(np.atleast_1d(ax).ravel()[0].figure, os.path.join(outdir, "posterior_Lambda.png"))
    print(f"=== N={n} done ===\n")


if __name__ == "__main__":
    main()
