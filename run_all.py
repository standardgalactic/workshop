#!/usr/bin/env python3
"""Run every analysis script in this folder, collecting outputs in ./outputs.

Put this file in the same folder as the .py scripts (modul.py, nma.py, ...), then:

    python run_all.py              run everything
    python run_all.py modul nma    run only the named scripts

Each script is copied into its own subfolder of ./outputs, with any
hard-coded /tmp/workN/ paths rewritten to that subfolder, so nothing
depends on the original sandbox layout. Figures (.pdf) and
*_results.json files end up in outputs/<script name>/.

Requires: numpy scipy matplotlib statsmodels
"""
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# sim.py is the slowest (200,000 samples). seeds.py execs sim.py from its own
# folder. shap.py is a helper from earlier work and is run last.
ORDER = ["modul", "nma", "pooled", "teach", "nomorb", "attrib", "paired",
         "wireless", "calib", "sim", "seeds", "shap"]
TIMEOUT = 3600  # seconds allowed per script


def rewrite(text, wd):
    return re.sub(r"/tmp/work\d+/?", str(wd) + "/", text)


def main():
    want = sys.argv[1:] or ORDER
    OUT.mkdir(exist_ok=True)
    summary = []
    for name in want:
        src = HERE / f"{name}.py"
        if not src.exists():
            print(f"\n=== {name} === not found, skipping", flush=True)
            summary.append((name, "MISSING", 0.0))
            continue
        wd = OUT / name
        wd.mkdir(exist_ok=True)
        (wd / f"{name}.py").write_text(rewrite(src.read_text(), wd))
        if name == "seeds" and (HERE / "sim.py").exists():
            (wd / "sim.py").write_text(rewrite((HERE / "sim.py").read_text(), wd))
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, f"{name}.py"], cwd=wd, timeout=TIMEOUT)
            status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        summary.append((name, status, time.time() - t0))

    print("\n=== summary ===")
    for n, s, t in summary:
        print(f"{n:10s} {s:18s} {t:7.1f}s")
    print(f"\nOutputs are in {OUT}/<script name>/")


if __name__ == "__main__":
    main()
