"""dll_direct -- the srg RCWA DLL called directly from Python (ctypes), no OpticStudio.

    python 02_NonSequential_RT_Zemax_Validation_OOP\\dll_direct.py
    python ...\\dll_direct.py --period 5 --max-order 20 --lam 0.6
    python ...\\dll_direct.py --dll srg_step_RCWA.dll --depth 0.95 --n-steps 8

What it prints, in order (every setting echoed, nothing hidden):
  1. the DLL's exports and the parameter names it reports (start/stop -3..+3)
  2. Diff2DSample.dll with T(1,0) = 1: the harness itself works (rc 0, energy 1)
     -- if THIS returns -1 the calling convention is wrong, not the physics
  3. srg blaze, null-test parameters: rc / energy / flag / dphase-dx per
     order, transmit and reflect, TE, TM and their mean, with sinc^2 beside
     -- the sign of dphase/dx at m = +1 IS the DLL's order convention
  4. the same with the exit index = n_grate (does the DLL read data[13]?)
  5. the old angle pair (alpha from the surface, beta 90) -> expected rc -1
  6. P = 50.0 exactly (cut-off coincidence)       -> expected rc -1
  7. Max Order 5 .. 50 at m = +1: energy and milliseconds (convergence, cost)
Results also go to runs\\_standalone_nsc\\<stamp>_direct\\direct.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nscval import SCRIPT_VERSION                                          # noqa: E402
from nscval.base import STANDALONE_RUNS_DIR                                # noqa: E402
from nscval.direct import DiffractionDll, dll_dir, sinc2                   # noqa: E402
from nscval.dlls import LABELS, slots                                      # noqa: E402
from nscval.settings import (NULL_SETTINGS, blaze_alpha_deg, blaze_depth_of_angles,  # noqa: E402
                             blaze_depth_um, cutoff_orders)


def values_from_slots(sl: Dict[int, float], n: int) -> List[float]:
    """{0-based index: value} -> a dense list of n parameter values (0 elsewhere)."""
    out = [0.0] * n
    for k, v in sl.items():
        if 0 <= k < n:
            out[k] = float(v)
    return out


def table(log: Any, ans: Dict[int, Dict[str, Any]], p_waves: float, label: str) -> None:
    log("  %s" % label)
    log("    m   rc   energy(mean)   TE       TM       flag  dP/dx        sinc^2(m-p)   ms")
    for m in sorted(ans):
        a = ans[m]
        log("   %+d   %2d   %8.4f     %7.4f  %7.4f   %d    %+.5f    %7.4f    %6.1f"
            % (m, a["rc"], a["energy"], a.get("energy_te", a["energy"]), a.get("energy_tm", a["energy"]),
               a["flag"], a["dpdx"], sinc2(m, p_waves), a["ms"]))
    tot = sum(a["energy"] for a in ans.values() if a["rc"] == 0)
    log("    sum over the listed orders: %.4f" % tot)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dll", default=NULL_SETTINGS["dll"])
    ap.add_argument("--dll-dir", default=dll_dir())
    ap.add_argument("--period", type=float, default=float(NULL_SETTINGS["period_um"]))
    ap.add_argument("--lam", type=float, default=float(NULL_SETTINGS["lams_um"][0]))
    ap.add_argument("--lam0", type=float, default=float(NULL_SETTINGS["lam0_um"]))
    ap.add_argument("--n-grate", type=float, default=float(NULL_SETTINGS["n_grate"]))
    ap.add_argument("--n-env", type=float, default=float(NULL_SETTINGS["n_env"]))
    ap.add_argument("--fill", type=float, default=float(NULL_SETTINGS["fill"]))
    ap.add_argument("--beta", type=float, default=float(NULL_SETTINGS["beta_deg"]))
    ap.add_argument("--max-order", type=int, default=int(NULL_SETTINGS["max_order"]))
    ap.add_argument("--orders", default="-3,-2,-1,0,1,2,3")
    ap.add_argument("--depth", type=float, default=None, help="staircase depth [um] (srg_step)")
    ap.add_argument("--n-steps", type=int, default=8, help="steps per period (srg_step)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the Max Order sweep")
    a = ap.parse_args(list(argv))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(STANDALONE_RUNS_DIR, "_standalone_nsc", "%s_direct" % stamp)
    os.makedirs(out_dir, exist_ok=True)
    lines: List[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    orders = [int(v) for v in a.orders.split(",")]
    depth = blaze_depth_um(a.lam0, a.n_grate, a.n_env)
    alpha = blaze_alpha_deg(a.period, depth, a.fill)
    p_waves = depth * (a.n_grate - a.n_env) / a.lam
    log("dll_direct | nscval %s | %s" % (SCRIPT_VERSION, stamp))
    log("output -> %s" % out_dir)
    log("settings: dll %s | P %.4f um | lam %.4f um | lam0 %.3f | n_grate %.4f | n_env %.3f | fill %.2f | "
        "depth %.4f um | alpha %.4f deg (from the vertical) | beta %.2f | Max Order %d | orders %s"
        % (a.dll, a.period, a.lam, a.lam0, a.n_grate, a.n_env, a.fill, depth, alpha, a.beta, a.max_order, orders))
    log("  the DLL rebuilds depth = fill P / (tan a + tan b) = %.4f um; p = depth (n - 1) / lam = %.4f waves"
        % (blaze_depth_of_angles(a.period, alpha, a.beta, a.fill), p_waves))
    hits = cutoff_orders(a.lam, a.period, [a.n_env, a.n_grate])
    log("  cut-off check: m = n P / lam = %s -> %s"
        % (", ".join("%.3f" % (n * a.period / a.lam) for n in (1.0, a.n_env, a.n_grate)),
           "COINCIDENCE %s" % hits if hits else "ok"))
    results: Dict[str, Any] = {"settings": vars(a), "depth_um": depth, "alpha_deg": alpha}

    # 1 load, exports, names
    log("")
    log("1. %s" % os.path.join(a.dll_dir, a.dll))
    dll = DiffractionDll(os.path.join(a.dll_dir, a.dll))
    log("   exports: %s" % ", ".join("%s=%s" % kv for kv in dll.exports.items()))
    names = dll.param_names(min(orders), max(orders))
    log("   parameter names (%d, 1-based as the DLL numbers them): %s"
        % (len(names), " | ".join("[%d]%s" % (i + 1, n) for i, n in enumerate(names)) or "none reported"))
    labels = names if names else LABELS.get(a.dll, [])
    results["names"] = names

    # 2 the harness itself, on Zemax's sample DLL
    log("")
    log("2. Diff2DSample.dll, X/Y period %.2f um, T(1,0) = 1 (harness self-test)" % a.period)
    try:
        smp = DiffractionDll(os.path.join(a.dll_dir, "Diff2DSample.dll"))
        smp_names = smp.param_names(1, 1)
        log("   exports: %s; names: %s" % (", ".join(k for k, v in smp.exports.items() if v), " | ".join(smp_names)))
        vals = [a.period, a.period, 0.0, 1.0]        # X period, Y period, slant, T(1,0)
        r = smp.call(a.lam, 1, 1, 1, vals, 1.0, 1.0, 0)
        log("   order +1 transmit: rc %d energy %.4f flag %d dP/dx %+.5f (expected: 0, 1.0, 1, %+.5f)"
            % (r["rc"], r["energy"], r["flag"], r["dpdx"], a.lam / a.period))
        results["self_test"] = r
    except Exception as exc:
        log("   self-test failed: %s: %s" % (type(exc).__name__, exc))

    # 3 the srg DLL with the null-test parameters
    def blaze_values(**over: float) -> List[float]:
        kw: Dict[str, float] = dict(period_um=a.period, max_order=a.max_order, fill=a.fill, alpha_deg=alpha,
                                    beta_deg=a.beta, index_grate_r=a.n_grate, index_grate_i=0.0,
                                    index_env_r=a.n_env, index_env_i=0.0, n_layer=1, interpolation=0,
                                    stochastic=0, only_orders=0)
        kw.update(over)
        return values_from_slots(slots(a.dll, labels, **kw), max(len(labels), 23))

    def step_values(**over: float) -> List[float]:
        d = a.depth if a.depth is not None else depth
        kw: Dict[str, float] = dict(period_um=a.period, max_order=a.max_order, depth_um=d, n_steps=a.n_steps,
                                    layers_per_step=1, alpha_deg=0.0, index_grate_r=a.n_grate,
                                    index_grate_i=0.0, index_env_r=a.n_env, index_env_i=0.0,
                                    interpolation=0, stochastic=0, only_orders=0)
        kw.update(over)
        return values_from_slots(slots(a.dll, labels, **kw), max(len(labels), 22))

    is_blaze = "blaze" in a.dll.lower()
    vals = blaze_values() if is_blaze else step_values()
    log("")
    log("3. %s, transmit, air both sides (n_in = n_out = 1)" % a.dll)
    log("   values: %s" % " | ".join("[%d]%s=%g" % (i, labels[i] if i < len(labels) else "?", v)
                                     for i, v in enumerate(vals) if v != 0.0))
    ans_t = dll.efficiencies(a.lam, orders, vals, 1.0, 1.0, 0)
    table(log, ans_t, p_waves, "transmit (data[11] = 0)")
    ans_r = dll.efficiencies(a.lam, orders, vals, 1.0, 1.0, 1)
    table(log, ans_r, p_waves, "reflect (data[11] = 1)")
    results["srg_transmit"] = ans_t
    results["srg_reflect"] = ans_r
    n_refused = sum(1 for x in ans_t.values() if x["rc"] != 0)
    if n_refused == len(orders):
        log("   EVERY call refused (rc -1). If the self-test above passed, the DLL refuses these "
            "parameters or wants a running OpticStudio (license check).")

    # 4 exit index = n_grate
    log("")
    log("4. the same with the exit-side index data[13] = %.4f (a glass object)" % a.n_grate)
    ans_g = dll.efficiencies(a.lam, orders, vals, 1.0, a.n_grate, 0)
    table(log, ans_g, p_waves, "transmit into n = %.4f" % a.n_grate)
    results["srg_glass_exit"] = ans_g

    if is_blaze:
        # 5 the old convention
        import math
        a_old = math.degrees(math.atan2(depth, a.period))
        log("")
        log("5. old pair alpha %.4f (from the surface), beta 90 -> the DLL's depth %.3g um" % (
            a_old, blaze_depth_of_angles(a.period, a_old, 90.0, a.fill)))
        r5 = dll.call(a.lam, min(orders), max(orders), 1, blaze_values(alpha_deg=a_old, beta_deg=90.0))
        log("   order +1: rc %d energy %.4f flag %d" % (r5["rc"], r5["energy"], r5["flag"]))
        results["old_convention"] = r5

        # 6 the cut-off coincidence
        P50 = 50.0
        log("")
        log("6. P = %.1f um exactly: m = n P / lam = %.3f (%s)"
            % (P50, a.n_grate * P50 / a.lam, cutoff_orders(a.lam, P50, [a.n_env, a.n_grate]) or "no coincidence"))
        r6 = dll.call(a.lam, min(orders), max(orders), 1,
                      blaze_values(period_um=P50, alpha_deg=blaze_alpha_deg(P50, depth, a.fill)))
        log("   order +1: rc %d energy %.4f flag %d" % (r6["rc"], r6["energy"], r6["flag"]))
        results["cutoff_50"] = r6

    # 7 convergence and cost
    if not a.no_sweep:
        log("")
        log("7. Max Order sweep at m = +1 (transmit, mean of TE/TM): energy and cost")
        sweep: List[Dict[str, Any]] = []
        for N in (5, 10, 20, 30, 40, 50):
            v = blaze_values(max_order=N) if is_blaze else step_values(max_order=N)
            r = dll.efficiencies(a.lam, [1], v, 1.0, 1.0, 0)[1]
            log("   Max Order %2d: rc %2d  energy %.4f  (TE %.4f TM %.4f)  %7.1f ms per call"
                % (N, r["rc"], r["energy"], r.get("energy_te", 0.0), r.get("energy_tm", 0.0), r["ms"]))
            sweep.append(dict(r, max_order=N))
        results["sweep"] = sweep

    with open(os.path.join(out_dir, "direct.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    with open(os.path.join(out_dir, "direct.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log("")
    log("paste this whole log back")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
