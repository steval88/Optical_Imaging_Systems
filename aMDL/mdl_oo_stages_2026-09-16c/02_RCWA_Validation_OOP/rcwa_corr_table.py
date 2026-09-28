"""
rcwa_corr_table.py -- the design's efficiency_corr.npz from a zone sweep,
with the noise of weak cells and irregular zones taken out (rcwaval.table).

    python 02_RCWA_Validation_OOP\\rcwa_corr_table.py runs\\<run> [--sweep <folder>] [options]

Options (every value echoed and stored in run_info.json):
    --sweep DIR      the rcwa\\<stamp>_zones folder (default: the run's newest)
    --eta-min E      keep a cell only where the scalar efficiency >= E (0.10)
    --clip LO,HI     clip the kept ratios to [LO, HI] (0.3,1.3)
    --smooth N       running median over N solved zones along the radius (5)
    --balance-tol T  drop cells whose R + T misses 1 by more than T (1e-3)

Outputs in <run>\\rcwa\\<stamp>_table\\: efficiency_corr.npz (the design's
contract), corr_table.npz (raw / kept / smoothed maps and the thresholds),
fig_corr_table.png, run_info.json. Seconds; no OpticStudio.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rcwaval import SCRIPT_VERSION                       # noqa: E402
from rcwaval.table import CorrTable, latest_sweep_dir    # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--sweep")
    ap.add_argument("--eta-min", type=float, default=0.10)
    ap.add_argument("--clip", default="0.3,1.3")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--balance-tol", type=float, default=1e-3)
    args = ap.parse_args(argv)
    run_dir = args.run.rstrip("/\\")
    sweep = args.sweep or latest_sweep_dir(run_dir)
    if not sweep:
        raise SystemExit("no rcwa\\*_zones folder with zone_sweep_results.json under %s" % run_dir)
    lo, hi = (float(v) for v in args.clip.split(","))
    out_dir = os.path.join(run_dir, "rcwa", time.strftime("%Y%m%d_%H%M%S") + "_table")
    os.makedirs(out_dir, exist_ok=True)
    lines: List[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    settings: Dict[str, Any] = {"sweep": sweep, "eta_min": args.eta_min, "clip": [lo, hi], "smooth": args.smooth,
                                "balance_tol": args.balance_tol}
    log("rcwaval %s | %s" % (SCRIPT_VERSION, " ".join(sys.argv)))
    log("settings: %s" % settings)
    ct = CorrTable(run_dir, sweep, args.eta_min, (lo, hi), args.smooth, args.balance_tol, log)
    t = ct.build()
    ct.write(out_dir, t)
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump({"script_version": SCRIPT_VERSION, "command": " ".join(sys.argv), "run": run_dir,
                   "settings": settings, "w_mean": t["w_mean"].tolist(), "lam_um": t["lam_um"].tolist()},
                  fh, indent=1)
    with open(os.path.join(out_dir, "corr_table.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("done -> %s" % out_dir)


if __name__ == "__main__":
    main()
