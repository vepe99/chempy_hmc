"""Corner plot of the paper-net global Lambda = [alpha_IMF, log10 N_Ia] posterior,
with a Gaussianized approximation (mean + covariance) overlaid as 1/2-sigma ellipses.

Usage: python corner_gauss_paper.py --n_stars 200 [--tag ...]
Output: plots/N{n}/ (or plots/<tag>/) corner_gauss_N{n}.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import arviz as az

import config_paper as C

LABELS = [r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_stars", type=int, default=200)
    p.add_argument("--tag", default="")
    return p.parse_args()


def nsigma_ellipse(ax, mu, cov, n, **kw):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n * np.sqrt(vals)
    e = Ellipse(xy=mu, width=width, height=height, angle=angle, fill=False, **kw)
    ax.add_patch(e)
    return e


def main():
    args = parse_args()
    base = os.path.join(C.POSTERIOR_DIR, args.tag) if args.tag else C.POSTERIOR_DIR
    idata = az.from_netcdf(os.path.join(base, f"posterior_N{args.n_stars}.nc"))
    lam = np.asarray(idata.posterior["Lambda"]).reshape(-1, 2)

    mu = lam.mean(axis=0)
    cov = np.cov(lam.T)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)

    print(f"N={args.n_stars} (paper_net)  n_samples={len(lam)}")
    print("mean        :", np.array2string(mu, precision=4))
    print("std (1sigma):", np.array2string(sd, precision=4))
    print("covariance  :\n", np.array2string(cov, precision=6))
    print("correlation :\n", np.array2string(corr, precision=4))

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    xs = [np.linspace(lam[:, i].min(), lam[:, i].max(), 200) for i in range(2)]

    def gauss(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    for i in range(2):
        ax = axes[i, i]
        ax.hist(lam[:, i], bins=60, density=True, color="0.8", edgecolor="none")
        ax.plot(xs[i], gauss(xs[i], mu[i], sd[i]), "r-", lw=2, label="Gaussian approx")
        ax.axvline(mu[i], color="r", ls="-", lw=1)
        ax.axvline(mu[i] - sd[i], color="r", ls="--", lw=1)
        ax.axvline(mu[i] + sd[i], color="r", ls="--", lw=1)
        ax.axvline(C.TRUE_LAMBDA[i], color="k", ls=":", lw=2)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.hist2d(lam[:, 0], lam[:, 1], bins=80, cmap="Greys",
              range=[[lam[:, 0].min(), lam[:, 0].max()],
                     [lam[:, 1].min(), lam[:, 1].max()]])
    nsigma_ellipse(ax, mu, cov, 1, color="r", lw=2, label=r"$1\sigma$")
    nsigma_ellipse(ax, mu, cov, 2, color="r", lw=2, ls="--", label=r"$2\sigma$")
    ax.plot(*mu, "r+", ms=12, mew=2)
    ax.plot(C.TRUE_LAMBDA[0], C.TRUE_LAMBDA[1], "kx", ms=10, mew=2, label="ref (nominal)")
    ax.set_xlabel(LABELS[0], fontsize=14)
    ax.set_ylabel(LABELS[1], fontsize=14)
    ax.legend(fontsize=9, loc="upper right")

    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5,
                    f"Gaussianized N={args.n_stars}\n(paper-faithful net)\n\n"
                    fr"$\rho$ = {corr[0,1]:.3f}" + "\n\n"
                    fr"$\sigma_\alpha$ = {sd[0]:.4f}" + "\n"
                    fr"$\sigma_{{\log N}}$ = {sd[1]:.4f}",
                    ha="center", va="center", fontsize=13, transform=axes[0, 1].transAxes)
    axes[0, 0].set_title(LABELS[0], fontsize=13)
    axes[1, 1].set_xlabel(LABELS[1], fontsize=14)
    fig.suptitle(f"N={args.n_stars} $\\Lambda$ posterior: samples + Gaussian approximation",
                 fontsize=15)
    fig.tight_layout()

    outdir = os.path.join(C.PLOT_DIR, args.tag if args.tag else f"N{args.n_stars}")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"corner_gauss_N{args.n_stars}.png")
    fig.savefig(out, dpi=120)
    print("saved ->", out)


if __name__ == "__main__":
    main()
