"""
rcwa_zone_sweep.py -- STAGE 2c: rigorous (RCWA) efficiency of the design's
own zone profiles, no OpticStudio.

    python 02_RCWA_Validation_OOP\\rcwa_zone_sweep.py runs\\<run> [options]

Options (every value echoed and stored in run_info.json):
    --p-max P        zones with a period above P um are not solved (corr = 1);
                     default 60 (the wide paraxial zones, P/lam > 100, are where
                     the thin-element model is exact and the solver slowest)
    --margin M       harmonics N = ceil(M n P / lam), default 1.2
    --lam-stride K   every K-th design line (the rest interpolated), default 1
    --zones a,b,c    only these zone ids
    --workers W      parallel processes, default = CPU count - 1
    --plan           print the cost table and stop
    --resume DIR     continue an interrupted sweep from its output folder

Outputs in <run>\\rcwa\\<YYYYMMDD_HHMMSS>_zones\\: efficiency_corr.npz (the
design's contract: lam_um, r_um, eta), zone_sweep.npz (eta_rcwa, eta_scalar,
ratio per line x zone), zone_sweep_results.json (every solve, TE/TM, order
window, energy balance -- the checkpoint), fig_zone_sweep.png, run_info.json.
"""
from __future__ import annotations

import argparse
import os
import sys

# one BLAS thread per process: the sweep parallelizes over (zone, line) tasks
# and oversubscribed BLAS threads made a 4-worker run SLOWER than one worker
# (2026-09-21). Set before numpy is imported -- also in the spawned workers,
# which re-import this module on Windows.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json                                              # noqa: E402
import time                                              # noqa: E402
from typing import Any, Dict, List, Optional             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rcwaval import SCRIPT_VERSION                       # noqa: E402
from rcwaval.zonesweep import ZoneSweep                  # noqa: E402


class Log:
    def __init__(self, path: Optional[str] = None) -> None:
        self.t0 = time.time()
        self.fh = open(path, "a") if path else None

    def __call__(self, msg: str) -> None:
        line = "[%7.1fs] %s" % (time.time() - self.t0, msg)
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--p-max", type=float, default=60.0)
    ap.add_argument("--margin", type=float, default=1.2)
    ap.add_argument("--lam-stride", type=int, default=1)
    ap.add_argument("--zones")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--resume")
    args = ap.parse_args(argv)
    run_dir = args.run.rstrip("/\\")
    if not os.path.exists(os.path.join(run_dir, "zone_table.npz")):
        raise SystemExit("no zone_table.npz in %s (a run folder of run_MDL_design.py)" % run_dir)
    out_dir = args.resume or os.path.join(run_dir, "rcwa", time.strftime("%Y%m%d_%H%M%S") + "_zones")
    os.makedirs(out_dir, exist_ok=True)
    log = Log(os.path.join(out_dir, "zone_sweep.log"))
    log("rcwaval %s | %s | output %s" % (SCRIPT_VERSION, " ".join(sys.argv), out_dir))
    settings: Dict[str, Any] = {"p_max_um": args.p_max, "margin": args.margin, "lam_stride": args.lam_stride,
                                "zones": args.zones, "workers": args.workers}
    log("settings: %s" % settings)
    sweep = ZoneSweep(run_dir, out_dir, args.p_max, args.margin, args.lam_stride, args.workers,
                      [int(v) for v in args.zones.split(",")] if args.zones else None, log)
    if args.resume:
        log("resuming: %d results in the checkpoint" % sweep.load_checkpoint())
    est = sweep.plan()
    info = {"script_version": SCRIPT_VERSION, "command": " ".join(sys.argv), "run": run_dir,
            "settings": settings, "estimated_solver_s": est}
    if args.plan:
        with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        log("plan only")
        return
    t0 = time.time()
    try:
        sweep.run()
    except KeyboardInterrupt:
        sweep.save_checkpoint()
        log("interrupted: %d results checkpointed; continue with --resume %s" % (len(sweep.results), out_dir))
    sweep.write()
    info["elapsed_s"] = round(time.time() - t0, 1)
    info["solved"] = len(sweep.results)
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    log("done (%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
