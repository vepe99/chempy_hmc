"""Reproduce the n_stars_plot from 04_plot_multistar_inference.ipynb, but for the
HMC posteriors: global params (alpha_IMF, log10 N_Ia) median +/- 1sigma/2sigma
as a function of N_stars, with the ground-truth reference line.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

PARAM_NAMES = [r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]


def load_lambda(n):
    npz = os.path.join(config.POSTERIOR_DIR, f"posterior_N{n}.npz")
    return np.load(npz)["Lambda"]  # (nsamples, 2)


def main():
    ns = config.N_STARS_LIST
    pcts = {}
    for n in ns:
        lam = load_lambda(n)
        pcts[n] = np.percentile(lam, [2.275, 15.865, 50.0, 84.135, 97.725], axis=0)
    lo2 = np.array([pcts[n][0] for n in ns])
    lo1 = np.array([pcts[n][1] for n in ns])
    med = np.array([pcts[n][2] for n in ns])
    up1 = np.array([pcts[n][3] for n in ns])
    up2 = np.array([pcts[n][4] for n in ns])

    x = np.array(ns)
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    for i, name in enumerate(PARAM_NAMES):
        a = ax[i]
        a.plot(x, med[:, i], "o-", color="r", label="HMC median")
        a.fill_between(x, lo1[:, i], up1[:, i], alpha=0.25, color="r", label=r"1 & 2 $\sigma$")
        a.fill_between(x, lo2[:, i], up2[:, i], alpha=0.12, color="r")
        true = config.TRUE_LAMBDA[i]
        a.axhline(true, color="k", linestyle=":", linewidth=2, label="Ground truth")
        a.set_xscale("log")
        a.set_xlim([1, x[-1]])
        a.set_xlabel(r"$N_{\rm stars}$", fontsize=30)
        a.set_ylabel(name, fontsize=30)
        a.tick_params(labelsize=18)
    ax[0].legend(fontsize=15, fancybox=True, shadow=True)
    plt.tight_layout()

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    out = os.path.join(config.PLOT_DIR, "n_stars_hmc_TNG.png")
    plt.savefig(out, dpi=120)
    print(f"saved -> {out}")

    # Print a small table (compare to Philcox & Rybizki 2019 Table 3)
    print(f"\n{'N':>5} {'alpha_med':>10} {'alpha_1sig':>18} {'logN_med':>10} {'logN_1sig':>18}")
    for j, n in enumerate(ns):
        print(f"{n:>5} {med[j,0]:>10.3f} [{lo1[j,0]:.3f},{up1[j,0]:.3f}]   "
              f"{med[j,1]:>10.3f} [{lo1[j,1]:.3f},{up1[j,1]:.3f}]")


if __name__ == "__main__":
    main()
