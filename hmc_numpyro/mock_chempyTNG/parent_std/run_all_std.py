"""Launch the N=1, 10, 200 standardized-parent HMC fits in parallel, one GPU each.

Each child process calls autocvd(num_gpus=1) itself, so it grabs a distinct free GPU
and blocks until one is available. Launches are staggered so the autocvd utilization
snapshots don't all fire on the same instant. Extra args are forwarded to run_hmc_std.py
(e.g. --advi_init --dense_mass --tune 3000).
"""
import subprocess
import sys
import time
import os

import config_std as config

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def main():
    extra = sys.argv[1:]  # forwarded to each run_hmc_std.py
    os.makedirs(config.POSTERIOR_DIR, exist_ok=True)
    procs = []
    for n in config.N_STARS_LIST:
        cmd = [PY, os.path.join(HERE, "run_hmc_std.py"), "--n_stars", str(n)] + extra
        log = open(os.path.join(config.POSTERIOR_DIR, f"run_N{n}.log"), "w")
        print("launching:", " ".join(cmd))
        p = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        procs.append((n, p, log))
        time.sleep(20)  # let each claim its GPU before the next snapshot

    rc = 0
    for n, p, log in procs:
        ret = p.wait()
        log.close()
        status = "OK" if ret == 0 else f"FAILED (rc={ret})"
        print(f"N={n}: {status}  (log: posteriors/run_N{n}.log)")
        rc |= ret
    sys.exit(rc)


if __name__ == "__main__":
    main()
