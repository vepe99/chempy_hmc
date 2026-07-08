"""Raw PyTorch Chempy emulator (JAX) + standardization constants.

The forward pass is IDENTICAL to parent/emulator.py — the raw net maps raw params
-> raw abundances with no normalization:

    y = tanh(tanh(tanh(x @ W1' + b1) @ W2' + b2)) @ W3' + b3

`load_weights` bundles the raw weights (weights.npz) together with the standardization
constants (standardization.npz) into one dict, so model_std.py can de-standardize its
latents into raw space, call `predict_obs` (raw), and re-standardize the raw output for
a well-conditioned Gaussian likelihood. The NETWORK ITSELF IS UNTOUCHED; standardization
lives only in the sampling/likelihood space.
"""
import numpy as np
import jax.numpy as jnp

import config_std as C


def load_weights(weights_path=C.WEIGHTS_NPZ, std_path=C.STANDARDIZATION_NPZ):
    d = np.load(weights_path)
    w = {k: jnp.asarray(d[k], dtype=jnp.float32) for k in
         ["W1", "b1", "W2", "b2", "W3", "b3"]}
    s = np.load(std_path)
    for k in ["in_mean", "in_std", "out_mean", "out_std"]:
        w[k] = jnp.asarray(s[k], dtype=jnp.float32)
    return w


def nn_forward(x, w):
    """x: (..., 6) raw inputs -> (..., 9) raw abundances (all 9 emulator outputs)."""
    x = jnp.tanh(x @ w["W1"].T + w["b1"])
    x = jnp.tanh(x @ w["W2"].T + w["b2"])
    x = x @ w["W3"].T + w["b3"]
    return x


def predict_obs(x, w):
    """(..., 6) raw inputs -> (..., 8) raw abundances, H (index 2) dropped."""
    y = nn_forward(x, w)
    return jnp.delete(y, C.H_INDEX, axis=-1)
