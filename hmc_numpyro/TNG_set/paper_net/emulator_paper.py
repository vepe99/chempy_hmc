"""JAX reimplementation of the standardized ChempyMulti emulator (paper-faithful).

Network forward pass (all in STANDARDIZED space), matching run_pymc3.py:
    x7   = [Lambda_std(2), Theta_std(3), time_std(1), time_std^2(1)]     (T^2 feature)
    out  = tanh(x7 @ w0 + b0) @ w1 + b1        # (..., 8) standardized abundances

NUTS differentiates through this via JAX autodiff. Standardization constants
(in_mean/std, out_mean/std) travel with the weights so the model can move between
physical and standardized space.
"""
import numpy as np
import jax.numpy as jnp

import config_paper as C


def load_weights(path=C.WEIGHTS_NPZ):
    d = np.load(path, allow_pickle=True)
    w = {k: jnp.asarray(np.asarray(d[k], dtype=np.float32))
         for k in ["w0", "b0", "w1", "b1", "in_mean", "in_std", "out_mean", "out_std"]}
    w["n_poly"] = int(d["n_poly"]) if "n_poly" in d.files else C.N_POLY
    return w


def net_std(x6_std, w):
    """Standardized 6-dim input [Lam(2),Theta(3),time(1)] -> standardized 8-dim output.

    The birth-time (last input column) is augmented with T^2 to form the 7-dim
    network input, then run through the block-sparse tanh net.
    """
    feats = [x6_std]
    t = x6_std[..., -1:]
    for i in range(w["n_poly"] - 1):
        feats.append(t ** (i + 2))
    xin = jnp.concatenate(feats, axis=-1)              # (..., 7)
    return jnp.tanh(xin @ w["w0"] + w["b0"]) @ w["w1"] + w["b1"]   # (..., 8)


def predict_phys(x6_phys, w):
    """Convenience: physical 6-dim input -> physical 8-dim abundances (for APE checks)."""
    x6_std = (x6_phys - w["in_mean"]) / w["in_std"]
    return net_std(x6_std, w) * w["out_std"] + w["out_mean"]
