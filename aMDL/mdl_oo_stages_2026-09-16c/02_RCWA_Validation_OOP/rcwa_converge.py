r"""
rcwa_converge.py -- harmonic convergence of one zone cell of the sweep: the
same profile solved at several harmonic counts, eta at the focusing order
(TE, TM, unpolarized), the +-3 window and the energy balance per count.

    python 02_RCWA_Validation_OOP\rcwa_converge.py runs\<run> --zone 34 --lam 0.400 [--margins 1.2,1.5,2,3]

The sweep used margin 1.2 (N = ceil(1.2 n P / lam)); a cell is converged
when eta stops moving between margins to the tolerance one cares about.
Cost grows as (2N+1)^3 per layer: at margin 3 on a 22-um zone at 400 nm
(N 330) expect a few minutes on a desktop CPU. Every setting is echoed and
stored with the results in <run>\rcwa\<stamp>_converge\rcwa_converge.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rcwaval import SCRIPT_VERSION                                       # noqa: E402
from rcwaval.profiles import staircase                                   # noqa: E402
from rcwaval.rcwa1d import harmonics_for, solve                          # noqa: E402
from rcwaval.zonesweep import focusing_order, load_zone_table            # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--zone", type=int, required=True)
    ap.add_argument("--lam", type=float, required=True, help="line [um]; the nearest table line is used")
    ap.add_argument("--margins", default="1.2,1.5,2,3", help="N = ceil(margin n P / lam) for each")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--dn", default="",
                    help="index offsets, e.g. 0.003,0.006: the cell re-solved at n + dn (densest margin) -- the "
                         "sensitivity to a small phase error, such as an FDTD grid's numerical dispersion")
    args = ap.parse_args(argv)
    run_dir = args.run.rstrip("/\\")
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    F = float(cfg.get("focal_um") or cfg["derived"]["focal_um"])
    zones, step_um, lams_all, n_real = load_zone_table(os.path.join(run_dir, "zone_table.npz"))
    z = next(zz for zz in zones if zz.zone_id == args.zone)
    il = int(np.argmin(np.abs(lams_all - args.lam)))
    lam, n = float(lams_all[il]), float(n_real[il])
    m, sin_c, eta_s, ok = focusing_order(z, lam, n, F)
    margins = [float(v) for v in args.margins.split(",")]
    out_dir = os.path.join(run_dir, "rcwa", time.strftime("%Y%m%d_%H%M%S") + "_converge")
    os.makedirs(out_dir, exist_ok=True)
    lines: List[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    log("rcwaval %s | rcwa_converge | %s" % (SCRIPT_VERSION, " ".join(sys.argv)))
    log("zone %d: P %.3f um, %d rings of %.3f um, h_max %.3f um, %d distinct levels | lam %.3f n %.4f | "
        "m_solver %+d (P sin_c / lam = %.3f)%s | scalar x T %.4f"
        % (z.zone_id, z.period_um, z.n_rings, step_um, float(z.h_um.max()), z.n_levels, lam, n, m,
           z.period_um * sin_c / lam, "" if ok else " (not well defined)", eta_s))
    g = staircase(z.h_um, step_um, n, n, 1.0, period_um=z.period_um)
    rows: List[Dict[str, Any]] = []
    w = args.window
    for mg in margins:
        N = harmonics_for(z.period_um, lam, n, mg)
        t0 = time.time()
        e = solve(g, lam, N)
        dt = time.time() - t0
        window = {int(k): float(e.order(int(k))) for k in range(m - w, m + w + 1)}
        eta_te, eta_tm, eta = float(e.order(m, "te")), float(e.order(m, "tm")), float(e.order(m))
        sum_T, sum_R = float(e.T.sum()), float(e.R.sum())
        rows.append({"margin": mg, "n_harm": N, "eta_te": eta_te, "eta_tm": eta_tm, "eta": eta,
                     "sum_T": sum_T, "sum_R": sum_R, "window": window, "seconds": dt})
        log("  margin %.2f N %4d: eta TE %.4f TM %.4f unpol %.4f | T %.4f R %.4f | %.0f s"
            % (mg, N, eta_te, eta_tm, eta, sum_T, sum_R, dt))
        log("      window: " + "  ".join("%+d: %.4f" % (k, v) for k, v in window.items()))
    if len(rows) > 1:
        ref = float(rows[-1]["eta"])
        log("  drift against the densest count (%.4f): " % ref
            + "  ".join("%.2f: %+.1f %%" % (float(r["margin"]), 100.0 * (float(r["eta"]) / ref - 1.0))
                        for r in rows[:-1]))
    sens: List[Dict[str, Any]] = []
    for dn in ([float(v) for v in args.dn.split(",")] if args.dn else []):
        N = harmonics_for(z.period_um, lam, n + dn, margins[-1])
        e = solve(staircase(z.h_um, step_um, n + dn, n + dn, 1.0, period_um=z.period_um), lam, N)
        phi_top = 2.0 * np.pi * dn * float(z.h_um.max()) / lam
        sens.append({"dn": dn, "n": n + dn, "n_harm": N, "eta_te": float(e.order(m, "te")),
                     "eta_tm": float(e.order(m, "tm")), "eta": float(e.order(m)), "phase_top_rad": phi_top,
                     "window": {int(k): float(e.order(int(k))) for k in range(m - w, m + w + 1)}})
        log("  n %+.4f -> %.4f (phase of the tallest ring %+.2f rad): eta TE %.4f TM %.4f unpol %.4f (%+.1f %%)"
            % (dn, n + dn, phi_top, sens[-1]["eta_te"], sens[-1]["eta_tm"], sens[-1]["eta"],
               100.0 * (sens[-1]["eta"] / float(rows[-1]["eta"]) - 1.0)))
        log("      window: " + "  ".join("%+d: %.4f" % (k, v) for k, v in sens[-1]["window"].items()))
    with open(os.path.join(out_dir, "rcwa_converge.json"), "w") as fh:
        json.dump({"script_version": SCRIPT_VERSION, "command": " ".join(sys.argv),
                   "settings": {"zone": args.zone, "lam_um": lam, "n": n, "margins": margins, "window": w,
                                "dn": args.dn},
                   "m_solver": m, "eta_scalar": eta_s, "rows": rows, "sensitivity": sens}, fh, indent=1)
    with open(os.path.join(out_dir, "rcwa_converge.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("done -> %s" % out_dir)


if __name__ == "__main__":
    main()
