"""Plain corner plot of the global Lambda = [alpha_IMF, log10 N_Ia] posterior
samples -- histograms + 2D density, NO Gaussian approximation overlay.

Usage: python corner_plain.py --n_stars 100 [--tag 200_3k]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

LABELS = [r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_stars", type=int, default=100)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    import arviz as az
    args = parse_args()
    base = os.path.join(config.POSTERIOR_DIR, args.tag) if args.tag else config.POSTERIOR_DIR
    nc = os.path.join(base, f"posterior_N{args.n_stars}.nc")
    idata = az.from_netcdf(nc)
    lam = np.asarray(idata.posterior["Lambda"]).reshape(-1, 2)  # (nsamp, 2)

    med = np.median(lam, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    # Diagonal marginals: histogram only
    for i in range(2):
        ax = axes[i, i]
        ax.hist(lam[:, i], bins=60, density=True, color="0.6", edgecolor="none")
        ax.axvline(med[i], color="C0", ls="-", lw=1.5)
        ax.axvline(config.TRUE_LAMBDA[i], color="k", ls=":", lw=2)
        ax.set_yticks([])

    # Lower-left: joint 2D histogram (no ellipses)
    ax = axes[1, 0]
    ax.hist2d(lam[:, 0], lam[:, 1], bins=80, cmap="Greys",
              range=[[lam[:, 0].min(), lam[:, 0].max()],
                     [lam[:, 1].min(), lam[:, 1].max()]])
    ax.plot(*med, "C0+", ms=12, mew=2, label="median")
    ax.plot(config.TRUE_LAMBDA[0], config.TRUE_LAMBDA[1], "kx", ms=10, mew=2,
            label="truth (nominal)")
    ax.set_xlabel(LABELS[0], fontsize=14)
    ax.set_ylabel(LABELS[1], fontsize=14)
    ax.legend(fontsize=9, loc="upper right")

    axes[0, 1].axis("off")
    axes[0, 0].set_title(LABELS[0], fontsize=13)
    axes[1, 1].set_xlabel(LABELS[1], fontsize=14)

    fig.suptitle(f"N={args.n_stars} $\\Lambda$ posterior (samples)", fontsize=15)
    fig.tight_layout()

    outdir = os.path.join(config.PLOT_DIR, args.tag if args.tag else f"N{args.n_stars}")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"corner_N{args.n_stars}.png")
    fig.savefig(out, dpi=120)
    print("saved ->", out)


if __name__ == "__main__":
    main()
