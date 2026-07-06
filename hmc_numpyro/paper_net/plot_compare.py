"""Compare the paper-faithful (standardized) net vs the raw PyTorch emulator.

Left/right panels: alpha_IMF and log10 N_Ia posterior median +/- 1/2 sigma vs N
(paper's n_stars figure), overlaying both experiments. Reads the Lambda .npz files.

Usage: python plot_compare.py
Output: plots/compare_n_stars.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_paper as C

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "posteriors"))
LABELS = [r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]


def load(dir_, n):
    f = os.path.join(dir_, f"posterior_N{n}.npz")
    return np.load(f)["Lambda"] if os.path.exists(f) else None


def stats(lam):
    p = np.percentile(lam, [2.275, 15.865, 50, 84.135, 97.725], axis=0)
    return p  # rows: -2s,-1s,med,+1s,+2s ; cols: alpha, logN


def main():
    Ns = C.N_STARS_LIST
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for dir_, name, color in [(C.POSTERIOR_DIR, "paper-net (standardized)", "C0")]:
        xs, P = [], []
        for n in Ns:
            lam = load(dir_, n)
            if lam is None:
                continue
            xs.append(n)
            P.append(stats(lam))
        if not xs:
            continue
        P = np.array(P)  # (n_N, 5, 2)
        xs = np.array(xs, float)
        for ind in [0, 1]:
            ax[ind].plot(xs, P[:, 2, ind], "o-", color=color, label=name + " median", zorder=3)
            ax[ind].fill_between(xs, P[:, 1, ind], P[:, 3, ind], alpha=0.30, color=color,
                                 label=r"$1\sigma$")
            ax[ind].fill_between(xs, P[:, 0, ind], P[:, 4, ind], alpha=0.15, color=color,
                                 label=r"$2\sigma$")

    for ind in [0, 1]:
        ax[ind].set_xscale("log")
        ax[ind].axhline(C.TRUE_LAMBDA[ind], color="k", ls=":", lw=1.5,
                        label="reference (-2.3,-2.89)")
        ax[ind].set_xlabel("number of stars $N$", fontsize=13)
        ax[ind].set_ylabel(LABELS[ind], fontsize=14)
        ax[ind].set_xticks(Ns)
        ax[ind].set_xticklabels(Ns)
        ax[ind].legend(fontsize=10)
    fig.suptitle("Global-parameter recovery vs N (paper-faithful standardized net)",
                 fontsize=14)
    fig.tight_layout()
    os.makedirs(C.PLOT_DIR, exist_ok=True)
    out = os.path.join(C.PLOT_DIR, "compare_n_stars.png")
    fig.savefig(out, dpi=130)
    print("saved ->", out)


if __name__ == "__main__":
    main()
