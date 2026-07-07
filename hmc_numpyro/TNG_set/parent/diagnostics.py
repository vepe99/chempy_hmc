"""MCMC health diagnostics for a saved posterior (.nc InferenceData).

Focus on the global parameters Lambda = [alpha_IMF, log10 N_Ia] (what we report),
plus the sampler-level energy/divergence checks that reflect the full 802-dim geometry.

Usage: python diagnostics.py --n_stars 10
Outputs -> plots/N{n}/  (trace, rank, energy, autocorr, pair, summary.txt)
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

import config

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
    print("  saved", os.path.relpath(path, config.PLOT_DIR))


def main():
    args = parse_args()
    n = args.n_stars
    base = os.path.join(config.POSTERIOR_DIR, args.tag) if args.tag else config.POSTERIOR_DIR
    nc = os.path.join(base, f"posterior_N{n}.nc")
    idata = az.from_netcdf(nc)

    outdir = os.path.join(config.PLOT_DIR, args.tag if args.tag else f"N{n}")
    os.makedirs(outdir, exist_ok=True)
    print(f"=== N={n} diagnostics -> {outdir} ===")

    # ---------- Numerical summary (r_hat, ESS, MCSE) ----------
    summ = az.summary(idata, var_names=["Lambda"],
                      stat_focus="mean",
                      hdi_prob=0.95)
    # Also include model error if present
    var_names = ["Lambda"]
    if "model_err" in idata.posterior:
        var_names.append("model_err")

    # Divergences + energy (BFMI)
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_samples = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    try:
        bfmi = az.bfmi(idata)
    except Exception:
        bfmi = np.array([np.nan])

    lam = np.asarray(idata.posterior["Lambda"]).reshape(-1, 2)
    pcts = np.percentile(lam, [2.5, 16, 50, 84, 97.5], axis=0)

    txt = [f"N_stars = {n}", f"total draws = {n_samples} "
           f"({idata.posterior.sizes['chain']} chains x {idata.posterior.sizes['draw']})",
           f"divergences = {div}  ({100*div/max(n_samples,1):.2f}%)",
           f"BFMI per chain = {np.array2string(np.asarray(bfmi), precision=3)}  "
           f"(healthy if > 0.3)", "",
           "Lambda summary (r_hat healthy < 1.01; ESS healthy > 400):",
           summ.to_string(), "",
           "Lambda percentiles [2.5, 16, 50, 84, 97.5]:",
           f"  alpha_IMF : {np.array2string(pcts[:,0], precision=4)}",
           f"  log10N_Ia : {np.array2string(pcts[:,1], precision=4)}"]
    report = "\n".join(txt)
    print(report)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(report + "\n")

    # ---------- Trace plot ----------
    axes = az.plot_trace(idata, var_names=["Lambda"], compact=False)
    savefig(axes.ravel()[0].figure, os.path.join(outdir, "trace_Lambda.png"))

    # ---------- Rank plot (better than trace for mixing) ----------
    axes = az.plot_rank(idata, var_names=["Lambda"])
    fig = np.atleast_1d(axes).ravel()[0].figure
    savefig(fig, os.path.join(outdir, "rank_Lambda.png"))

    # ---------- Autocorrelation ----------
    axes = az.plot_autocorr(idata, var_names=["Lambda"], combined=True)
    fig = np.atleast_1d(axes).ravel()[0].figure
    savefig(fig, os.path.join(outdir, "autocorr_Lambda.png"))

    # ---------- Energy (funnel / BFMI visual) ----------
    try:
        ax = az.plot_energy(idata)
        savefig(ax.figure, os.path.join(outdir, "energy.png"))
    except Exception as e:
        print("  energy plot skipped:", e)

    # ---------- Pair plot with divergences ----------
    try:
        axes = az.plot_pair(idata, var_names=["Lambda"], kind="kde",
                            divergences=True, marginals=True)
        fig = np.atleast_1d(axes).ravel()[0].figure
        # ground-truth crosshair
        savefig(fig, os.path.join(outdir, "pair_Lambda.png"))
    except Exception as e:
        print("  pair plot skipped:", e)

    # ---------- Posterior with ground truth ----------
    ax = az.plot_posterior(idata, var_names=["Lambda"],
                           ref_val=list(config.TRUE_LAMBDA), hdi_prob=0.95)
    fig = np.atleast_1d(ax).ravel()[0].figure
    savefig(fig, os.path.join(outdir, "posterior_Lambda.png"))

    print(f"=== N={n} done ===\n")


if __name__ == "__main__":
    main()
