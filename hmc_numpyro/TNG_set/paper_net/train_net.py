"""Train the ChempyMulti emulator EXACTLY as in the tutorial notebook.

Replicates `Multi-Star Inference with Chempy - Tutorial.ipynb` cells 22-40:
  * load TNG_Training_Data.npz, select the 8 TNG elements (drop H), filter bad runs,
  * standardize inputs/outputs (birth-time mapped to [0,1]) and add a T^2 feature,
  * train one sklearn MLPRegressor per element (tanh, 40 neurons, Adam),
  * stack into the block-sparse (w0,b0,w1,b1) network and save with the
    standardization constants.

Run:  python train_net.py [--epochs 3000] [--seed 0]
Output: weights_paper.npz  (w0,b0,w1,b1,in_mean,in_std,out_mean,out_std,els)
"""
import argparse
import time

import numpy as np

import config_paper as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--neurons", type=int, default=C.NEURONS)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_and_filter(path, el_indices):
    """Notebook cells 22/24/30: select els, drop failed runs / infs / bad birth times."""
    d = np.load(path, mmap_mode="r")
    params = np.asarray(d["params"])
    abun = np.asarray(d["abundances"])[:, el_indices]
    # remove failed Chempy runs (first abundance exactly 0) and non-finite rows
    keep = abun[:, 0] != 0
    params, abun = params[keep], abun[keep]
    good = np.isfinite(abun).all(axis=1)
    params, abun = params[good], abun[good]
    # remove bad birth times (T <= 0.99 Gyr)
    good2 = params[:, -1] > 0.99
    return params[good2], abun[good2]


def add_tsq(X, n_poly):
    """Augment the (standardized) birth-time with T^2, ... , T^n_poly (notebook cell 28)."""
    out = np.zeros([X.shape[0], X.shape[1] + n_poly - 1])
    out[:, : X.shape[1]] = X
    for i in range(n_poly - 1):
        out[:, X.shape[1] + i] = X[:, -1] ** (i + 2)
    return out


def main():
    args = parse_args()
    from sklearn.neural_network import MLPRegressor

    np.random.seed(args.seed)
    els = C.ELS

    # Element indices within the 9-element training arrays
    ref = np.load(C.TRAIN_DATA, mmap_mode="r")
    all_els = [str(e) for e in ref["elements"]]
    el_indices = np.array([all_els.index(e) for e in els], dtype=int)
    print(f"elements {els} -> indices {el_indices.tolist()} (of {all_els})")

    # --- Load + filter training and test data ---
    cut_params3, cut_abuns3 = load_and_filter(C.TRAIN_DATA, el_indices)
    cut_params3T, cut_abuns3T = load_and_filter(C.TEST_DATA, el_indices)
    print(f"train points: {len(cut_params3)}   test points: {len(cut_params3T)}")

    # --- Standardization (notebook cell 26): birth-time mapped to [0,1] ---
    par_mean = cut_params3.mean(axis=0)
    par_std = cut_params3.std(axis=0)
    ab_mean = cut_abuns3.mean(axis=0)
    ab_std = cut_abuns3.std(axis=0)
    par_mean[-1] = cut_params3[:, -1].min()
    par_std[-1] = cut_params3[:, -1].max() - cut_params3[:, -1].min()

    # randomize training order
    perm = np.random.choice(len(cut_abuns3), replace=False, size=len(cut_abuns3))
    trainX = (cut_params3[perm] - par_mean) / par_std
    trainY = (cut_abuns3[perm] - ab_mean) / ab_std
    testX = (cut_params3T - par_mean) / par_std
    testY = (cut_abuns3T - ab_mean) / ab_std

    sq_trainX = add_tsq(trainX, C.N_POLY)
    sq_testX = add_tsq(testX, C.N_POLY)
    dim_in = sq_trainX.shape[1]
    print(f"network shape: {dim_in} -> {args.neurons} -> {len(els)} (per element)")

    # --- Train one MLPRegressor per element (notebook cells 32/34) ---
    coeffs, scores = [], []
    t0 = time.time()
    for el_i, el in enumerate(els):
        model = MLPRegressor(
            solver="adam", alpha=0.001, max_iter=args.epochs,
            learning_rate="adaptive", tol=1e-13, hidden_layer_sizes=(args.neurons,),
            activation="tanh", shuffle=True, early_stopping=True,
            random_state=args.seed + el_i,
        )
        model.fit(sq_trainX, trainY[:, el_i])
        pred = model.predict(sq_testX)
        score = np.mean((pred - testY[:, el_i]) ** 2)
        w0_i, w1_i = model.coefs_       # (dim_in,neurons), (neurons,1)
        b0_i, b1_i = model.intercepts_  # (neurons,), (1,)
        coeffs.append([w0_i, w1_i, b0_i, b1_i])
        scores.append(score)
        print(f"  [{el_i+1}/{len(els)}] {el:>3}: iters={model.n_iter_:4d}  "
              f"test MSE(std)={score:.4e}")
    print(f"trained {len(els)} nets in {time.time()-t0:.1f}s")

    # --- Combine into block-sparse stacked net (notebook cells 36/38) ---
    w0 = np.hstack([co[0] for co in coeffs])              # (dim_in, neurons*n_els)
    b0 = np.hstack([co[2] for co in coeffs])              # (neurons*n_els,)
    b1 = np.hstack([co[3] for co in coeffs])              # (n_els,)
    w1 = np.zeros([w0.shape[1], b1.shape[0]])             # (neurons*n_els, n_els)
    for i in range(len(coeffs)):
        w1[args.neurons * i: args.neurons * (i + 1), i] = coeffs[i][1][:, 0]

    def stacked(x):  # x: (...,dim_in) standardized -> (...,n_els) standardized
        return np.tanh(x @ w0 + b0) @ w1 + b1

    # --- Diagnostics: L1 error in physical (dex) space on the test set ---
    pred_std = stacked(sq_testX)
    l1_dex = np.abs((pred_std - testY) * ab_std)          # de-standardize
    print("\n--- test-set L1 error [dex] per element (median / mean / 99pct) ---")
    for i, el in enumerate(els):
        c = l1_dex[:, i]
        print(f"  {el:>3}: median={np.median(c):.4f}  mean={c.mean():.4f}  "
              f"99pct={np.percentile(c,99):.4f}")
    print(f"  ALL: median={np.median(l1_dex):.4f}  mean={l1_dex.mean():.4f}")

    # Absolute Percentage Error in physical abundance space
    true_phys = cut_abuns3T
    pred_phys = pred_std * ab_std + ab_mean
    ape = np.abs(pred_phys - true_phys) / (np.abs(true_phys) + 1e-8) * 100.0
    print(f"  APE physical: median={np.median(ape):.3f}%  mean={ape.mean():.3f}%")

    # --- Sanity check vs the authors' reference standardization ---
    try:
        ref_w = np.load(C.REF_WEIGHTS)
        print("\n--- standardization vs authors' TNG_Network_Weights.npz ---")
        for name, ours, theirs in [
            ("in_mean", par_mean, ref_w["in_mean"]),
            ("in_std", par_std, ref_w["in_std"]),
            ("out_mean", ab_mean, ref_w["out_mean"]),
            ("out_std", ab_std, ref_w["out_std"]),
        ]:
            print(f"  {name:>8}: max|ours-theirs| = {np.abs(ours-theirs).max():.2e}")
    except Exception as e:
        print("reference comparison skipped:", e)

    # --- Save ---
    np.savez(C.WEIGHTS_NPZ, w0=w0, b0=b0, w1=w1, b1=b1,
             in_mean=par_mean, in_std=par_std, out_mean=ab_mean, out_std=ab_std,
             els=np.array(els), neurons=args.neurons, n_poly=C.N_POLY,
             test_mse_std=np.array(scores))
    print(f"\nsaved -> {C.WEIGHTS_NPZ}")


if __name__ == "__main__":
    main()
