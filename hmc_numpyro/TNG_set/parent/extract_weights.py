"""Extract the PyTorch emulator weights to a plain .npz for the JAX forward pass.

Run once. Uses torch only here; the sampler never imports torch.
"""
import numpy as np
import torch

import config


def main():
    sd = torch.load(config.STATE_DICT, map_location="cpu")
    out = {}
    for i, layer in enumerate(["l1", "l2", "l3"], start=1):
        out[f"W{i}"] = sd[f"{layer}.weight"].numpy().astype(np.float64)  # (out, in)
        out[f"b{i}"] = sd[f"{layer}.bias"].numpy().astype(np.float64)    # (out,)
    np.savez(config.WEIGHTS_NPZ, **out)
    for k, v in out.items():
        print(f"{k}: {v.shape}")
    print(f"Saved -> {config.WEIGHTS_NPZ}")


if __name__ == "__main__":
    main()
