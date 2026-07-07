"""Paper-style n_stars figure for the paper-faithful experiment.

Shows the global-parameter posterior (median + 1/2 sigma bands) tightening as the
number of stars N grows, exactly like notebook cell 69 / the ChempyMulti paper figure.

Usage: python plot_n_stars_paper.py
Output: plots/n_stars_paper.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_paper as C

LABELS = [r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]


def main():
    Ns, P = [], []
    for n in C.N_STARS_LIST:
        f = os.path.join(C.POSTERIOR_DIR, f"posterior_N{n}.npz")
        if not os.path.exists(f):
            print("missing", f)
            continue
        lam = np.load(f)["Lambda"]
        Ns.append(n)
        P.append(np.percentile(lam, [2.275, 15.865, 50, 84.135, 97.725], axis=0))
    P = np.array(P)              # (n_N, 5, 2)
    Ns = np.array(Ns, float)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for ind in [0, 1]:
        ax[ind].set_xscale("log")
        ax[ind].plot(Ns, P[:, 2, ind], "x-", color="b", zorder=3, label="median")
        ax[ind].fill_between(Ns, P[:, 1, ind], P[:, 3, ind], alpha=0.30, color="b",
                             label=r"$1\sigma$")
        ax[ind].fill_between(Ns, P[:, 0, ind], P[:, 4, ind], alpha=0.18, color="b",
                             label=r"$2\sigma$")
        ax[ind].axhline(C.TRUE_LAMBDA[ind], color="k", ls=":", lw=1.5,
                        label=r"reference $(-2.3,-2.89)$")
        ax[ind].set_xlabel("number of stars $N$", fontsize=14)
        ax[ind].set_ylabel(LABELS[ind], fontsize=15)
        ax[ind].set_xticks(Ns)
        ax[ind].set_xticklabels([int(n) for n in Ns])
        ax[ind].legend(fontsize=11)
    fig.suptitle("Paper-faithful net: global-parameter posterior vs number of stars",
                 fontsize=15)
    fig.tight_layout()
    os.makedirs(C.PLOT_DIR, exist_ok=True)
    out = os.path.join(C.PLOT_DIR, "n_stars_paper.png")
    fig.savefig(out, dpi=130)
    print("saved ->", out)

    # print the numbers behind the figure
    print("\nN     alpha_IMF (med [-1s,+1s])          log10N_Ia (med [-1s,+1s])")
    for i, n in enumerate(Ns):
        print(f"{int(n):>4}  {P[i,2,0]:+.3f} [{P[i,1,0]:+.3f},{P[i,3,0]:+.3f}]   "
              f"{P[i,2,1]:+.3f} [{P[i,1,1]:+.3f},{P[i,3,1]:+.3f}]")


if __name__ == "__main__":
    main()
