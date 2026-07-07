"""JAX reimplementation of the PyTorch Chempy emulator.

Forward pass:  y = tanh(tanh(x @ W1' + b1) @ W2' + b2) @ W3' + b3
No normalization (the PyTorch model was trained on raw params -> raw abundances).
NUTS gets analytic gradients through this via JAX autodiff.
"""
import numpy as np
import jax.numpy as jnp

import config


def load_weights(path=config.WEIGHTS_NPZ):
    d = np.load(path)
    return {k: jnp.asarray(d[k]) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}


def nn_forward(x, w):
    """x: (..., 6) -> (..., 9) raw abundances (all 9 emulator outputs)."""
    x = jnp.tanh(x @ w["W1"].T + w["b1"])
    x = jnp.tanh(x @ w["W2"].T + w["b2"])
    x = x @ w["W3"].T + w["b3"]
    return x


def predict_obs(x, w):
    """(..., 6) -> (..., 8) abundances with H (index 2) dropped, TNG element order."""
    y = nn_forward(x, w)
    return jnp.delete(y, config.H_INDEX, axis=-1)
