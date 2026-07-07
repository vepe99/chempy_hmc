"""Fit a full-covariance ADVI (AutoMultivariateNormal) to the N-star model and show
the ADVI Gaussian posterior for Lambda as a corner plot, overlaid on the HMC samples.

Usage: python advi_corner.py --n_stars 200 [--hmc_tag ""] [--steps 40000]
"""
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_stars", type=int, default=200)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--hmc_tag", default="", help="tag of the HMC posterior to overlay")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def nsigma_ellipse(ax, mu, cov, n, **kw):
    import numpy as np
    from matplotlib.patches import Ellipse
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

    from autocvd import autocvd
    autocvd(num_gpus=1)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import arviz as az
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoMultivariateNormal

    import config
    from emulator import load_weights
    from model import model, load_mock

    LABELS = [r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]

    w = load_weights()
    data = load_mock(args.n_stars)
    mk = dict(w=w, obs=data["obs"], obs_err=data["obs_err"],
              time_mu=data["time_mu"], time_sd=data["time_sd"],
              elem_err=True, centered=False)

    # --- Fit full-covariance ADVI ---
    guide = AutoMultivariateNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(args.lr), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(args.seed), args.steps, **mk)
    print(f"ADVI (full-cov) {args.steps} steps, final ELBO loss = {float(res.losses[-1]):.1f}")

    post = guide.sample_posterior(jax.random.PRNGKey(args.seed + 1), res.params,
                                  sample_shape=(args.draws,))
    lam_advi = np.asarray(post["Lambda"]).reshape(-1, 2)
    mu_a, cov_a = lam_advi.mean(0), np.cov(lam_advi.T)
    sd_a = np.sqrt(np.diag(cov_a))
    corr_a = cov_a / np.outer(sd_a, sd_a)
    print("ADVI mean :", np.array2string(mu_a, precision=4))
    print("ADVI std  :", np.array2string(sd_a, precision=4))
    print("ADVI corr :\n", np.array2string(corr_a, precision=4))

    # --- HMC overlay ---
    base = os.path.join(config.POSTERIOR_DIR, args.hmc_tag) if args.hmc_tag else config.POSTERIOR_DIR
    nc = os.path.join(base, f"posterior_N{args.n_stars}.nc")
    lam_hmc = np.asarray(az.from_netcdf(nc).posterior["Lambda"]).reshape(-1, 2)
    mu_h, cov_h = lam_hmc.mean(0), np.cov(lam_hmc.T)
    sd_h = np.sqrt(np.diag(cov_h))
    print("HMC  mean :", np.array2string(mu_h, precision=4))
    print("HMC  std  :", np.array2string(sd_h, precision=4))
    print("HMC  corr :", f"{cov_h[0,1]/(sd_h[0]*sd_h[1]):.4f}")

    # --- Corner plot: HMC samples (grey) + ADVI Gaussian (blue) + HMC Gaussian (red) ---
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    def gauss(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    for i in range(2):
        ax = axes[i, i]
        lo = min(lam_hmc[:, i].min(), lam_advi[:, i].min())
        hi = max(lam_hmc[:, i].max(), lam_advi[:, i].max())
        xs = np.linspace(lo, hi, 200)
        ax.hist(lam_hmc[:, i], bins=60, density=True, color="0.8", edgecolor="none",
                label="HMC samples")
        ax.plot(xs, gauss(xs, mu_a[i], sd_a[i]), "b-", lw=2, label="ADVI")
        ax.plot(xs, gauss(xs, mu_h[i], sd_h[i]), "r--", lw=2, label="HMC Gaussian")
        ax.axvline(config.TRUE_LAMBDA[i], color="k", ls=":", lw=2)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(lam_hmc[:, 0], lam_hmc[:, 1], s=3, color="0.7", alpha=0.3, label="HMC samples")
    nsigma_ellipse(ax, mu_a, cov_a, 1, color="b", lw=2)
    nsigma_ellipse(ax, mu_a, cov_a, 2, color="b", lw=2, ls="--")
    nsigma_ellipse(ax, mu_h, cov_h, 1, color="r", lw=2)
    nsigma_ellipse(ax, mu_h, cov_h, 2, color="r", lw=2, ls="--")
    ax.plot(*mu_a, "b+", ms=12, mew=2, label="ADVI 1/2$\\sigma$")
    ax.plot(*mu_h, "rx", ms=10, mew=2, label="HMC 1/2$\\sigma$")
    ax.plot(config.TRUE_LAMBDA[0], config.TRUE_LAMBDA[1], "k*", ms=14, label="truth (nominal)")
    ax.set_xlabel(LABELS[0], fontsize=14)
    ax.set_ylabel(LABELS[1], fontsize=14)
    ax.legend(fontsize=8, loc="upper left")

    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5,
                    "ADVI (full-cov) vs HMC\n\n"
                    f"ADVI:  $\\rho$={corr_a[0,1]:.3f}\n"
                    f"   $\\sigma_\\alpha$={sd_a[0]:.4f}, $\\sigma_{{\\log N}}$={sd_a[1]:.4f}\n\n"
                    f"HMC:   $\\rho$={cov_h[0,1]/(sd_h[0]*sd_h[1]):.3f}\n"
                    f"   $\\sigma_\\alpha$={sd_h[0]:.4f}, $\\sigma_{{\\log N}}$={sd_h[1]:.4f}",
                    ha="center", va="center", fontsize=11, transform=axes[0, 1].transAxes)
    axes[0, 0].set_title(LABELS[0], fontsize=13)
    axes[1, 1].set_xlabel(LABELS[1], fontsize=14)
    fig.suptitle(f"N={args.n_stars}: ADVI Gaussian posterior vs HMC", fontsize=15)
    fig.tight_layout()

    outdir = os.path.join(config.PLOT_DIR, "advi")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"advi_corner_N{args.n_stars}.png")
    fig.savefig(out, dpi=120)
    print("saved ->", out)


if __name__ == "__main__":
    main()
