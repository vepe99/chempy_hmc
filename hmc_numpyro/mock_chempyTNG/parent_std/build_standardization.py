"""Compute the input/output standardization constants for the raw emulator.

Run ONCE. This is the only file that reads the large training file. It derives the
per-dimension mean/std used to standardize the sampling and likelihood spaces:

  * in_mean, in_std  (6,) over the 6 raw network INPUTS
      [alpha_IMF, log10_N_0, log10_SFE, log10_sfr_scale, outflow, time]
  * out_mean, out_std (8,) over the 8 OBS abundances (H dropped, OBS_ELEMENTS order)

computed from `chempy_train_uniform_prior_5sigma.npz` — the exact training
distribution the raw net (pytorch_state_dict_5sigma_uni_prior.pt) was fit on. This
mirrors what paper_net's train_net.py bakes into weights_paper.npz, except here the
NETWORK IS UNCHANGED; only the standardization constants are new. NaN/inf training
rows are dropped before computing the statistics.

Saves standardization.npz {in_mean,in_std,out_mean,out_std}.
"""
import numpy as np

import config_std as C


def main():
    d = np.load(C.TRAIN_DATA, allow_pickle=True)
    params = np.asarray(d["params"], dtype=np.float64)       # (M, 6)
    abund = np.asarray(d["abundances"], dtype=np.float64)     # (M, 9), NN order
    els = [str(e) for e in d["elements"]]
    assert els == C.NN_ELEMENTS, f"element order mismatch: {els} != {C.NN_ELEMENTS}"

    # Drop failed / non-finite Chempy rows before computing statistics.
    finite = np.isfinite(params).all(1) & np.isfinite(abund).all(1)
    params, abund = params[finite], abund[finite]
    print(f"training rows: {finite.sum()} / {finite.size} finite "
          f"({100*finite.mean():.3f}%)")

    # Inputs: all 6 dims.
    in_mean = params.mean(0)
    in_std = params.std(0)

    # Outputs: drop H (index 2) -> 8 obs elements in OBS_ELEMENTS order.
    abund_obs = np.delete(abund, C.H_INDEX, axis=1)           # (M, 8)
    out_mean = abund_obs.mean(0)
    out_std = abund_obs.std(0)

    # Guard against any zero-variance dimension.
    assert (in_std > 0).all() and (out_std > 0).all(), "zero-variance dimension"

    np.savez(C.STANDARDIZATION_NPZ,
             in_mean=in_mean.astype(np.float32), in_std=in_std.astype(np.float32),
             out_mean=out_mean.astype(np.float32), out_std=out_std.astype(np.float32))

    np.set_printoptions(precision=5, suppress=True)
    print("in_mean  :", in_mean)
    print("in_std   :", in_std)
    print("out_mean :", out_mean, f"({C.OBS_ELEMENTS})")
    print("out_std  :", out_std)
    print("saved ->", C.STANDARDIZATION_NPZ)


if __name__ == "__main__":
    main()
