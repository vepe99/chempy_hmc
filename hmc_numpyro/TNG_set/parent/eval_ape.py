"""Sanity-check the JAX emulator port: recompute the Absolute Percentage Error
on the TNG validation set and confirm it matches the torch reference.

APE = 100 * |y_true - y_pred| / |y_true|  over all 9 emulator outputs.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from emulator import load_weights, nn_forward


def clean(x, y):
    idx = np.where((y == 0).all(axis=1))[0]
    x, y = np.delete(x, idx, axis=0), np.delete(y, idx, axis=0)
    idx = np.where(np.isfinite(y).all(axis=1))[0]
    return x[idx], y[idx]


def main():
    d = np.load(config.VAL_DATA, mmap_mode="r")
    val_x, val_y = clean(np.asarray(d["params"]), np.asarray(d["abundances"]))

    w = load_weights()
    pred = np.asarray(nn_forward(val_x, w))

    ape = 100.0 * np.abs((val_y - pred) / val_y)
    p25, p50, p75 = np.percentile(ape, [25, 50, 75])
    print(f"APE median: {p50:.3f}%  (25th={p25:.3f}%, 75th={p75:.3f}%)")
    print(f"APE mean:   {ape.mean():.3f}%   max: {ape.max():.3f}%")
    print(f"per-element median APE ({config.NN_ELEMENTS}):")
    print("  " + np.array2string(np.median(ape, axis=0), precision=3))

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    fig, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.2, 0.8)})
    ax_hist.hist(ape.flatten(), bins=100, density=True, range=(0, 30), color="tomato")
    ax_hist.set_xlabel("Error (%)", fontsize=15)
    ax_hist.set_ylabel("Density", fontsize=15)
    ax_hist.axvline(p50, color="black", linestyle="--")
    ax_hist.axvline(p25, color="black", linestyle="dotted")
    ax_hist.axvline(p75, color="black", linestyle="dotted")
    ax_hist.text(p50, 0.2, fr"${p50:.1f}^{{+{p75-p50:.1f}}}_{{-{p50-p25:.1f}}}\%$", fontsize=12)
    ax_box.boxplot(ape.flatten(), vert=False, widths=0.5, patch_artist=True,
                   showfliers=False, boxprops=dict(facecolor="tomato"),
                   medianprops=dict(color="black"))
    ax_box.set(yticks=[])
    fig.suptitle("APE of the JAX emulator (val set)", fontsize=18)
    plt.xlim(0, 30)
    fig.tight_layout()
    out = os.path.join(config.PLOT_DIR, "ape_jax_emulator.png")
    plt.savefig(out, dpi=120)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
