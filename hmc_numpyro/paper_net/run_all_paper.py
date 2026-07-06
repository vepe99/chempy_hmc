"""Launch the paper-faithful N=1,10,200 fits in parallel (one free GPU each).

Each subprocess calls autocvd itself to grab a distinct free GPU (waiting if none
free). Mirrors the parent run_all.py.
"""
import subprocess
import sys
import time

import config_paper as C

COMMON = ["--draws", "1000", "--tune", "1000", "--chains", "4",
          "--target_accept", "0.9", "--advi_init"]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    procs = []
    for n in C.N_STARS_LIST:
        cmd = [sys.executable, "run_hmc_paper.py", "--n_stars", str(n)] + COMMON
        if tag:
            cmd += ["--tag", tag]
        log = open(f"run_N{n}.log", "w")
        print("launching:", " ".join(cmd))
        procs.append((n, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(20)  # stagger so autocvd sees distinct free GPUs

    fail = 0
    for n, p in procs:
        rc = p.wait()
        print(f"N={n} exited with code {rc}")
        fail += rc != 0
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
