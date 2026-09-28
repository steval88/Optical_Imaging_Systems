r"""
lumerical_zone_check.py -- a few zones of the sweep through Lumerical FDTD (2-D,
periodic), as the independent full-wave witness of rcwaval's RCWA.

    python 02_RCWA_Validation_OOP\lumerical_zone_check.py runs\<run> --lumapi "C:\Program Files\Lumerical\v241\api\python"
           [--sweep <folder>] [--zones 34,80,158] [--lams 0.45,0.75,1.05 | --auto 3] [--mesh 4] [--dx 0.012]
           [--hide] [--dry-run] [--check-api]

--lumapi takes the api\python folder or lumapi.py itself. --auto K picks,
per zone, K lines spread over the band among those the sweep solved (a
comparison needs an RCWA value on the other side); --lams overrides it.

--dry-run prints the plan and does NOT import lumapi (so it proves nothing
about the API). --check-api imports lumapi, opens an FDTD session, prints
the module file and the Lumerical version, closes it and stops: the test
that the API is reachable from this environment, without a simulation. A
real run prints the same two lines before the first cell.

First run 2026-09-21 13:08 (zone 34, 0.400 um): the API, the build and the
monitors work (T + R = 0.9998); three things were wrong and are fixed in
.08: (1) `run` refuses an unsaved project and pops a save dialog -- every
cell is now saved first as <out>\fsp\zoneNNN_lamL.LLL_pol.fsp, which also
keeps the model for inspection; (2) the wavelength was set on the source
object, which follows the GLOBAL source settings unless told otherwise --
the line is now set through setglobalsource / setglobalmonitor and read
back after the run (source and monitor wavelength, mesh cells, accuracy,
polarization angle, substrate index are logged per cell); (3) the order
sign: Lumerical numbers orders physically (kx_n = kx0 + n 2 pi / P), the
solver's index is the mirror (rcwa1d docstring, test 7), so eta is read at
n = -m_solver; the mirror side and the whole +-3 window (FDTD against the
sweep's) are logged so a sign slip shows up as a swapped column, not as a
0.02 ratio.

Per selected zone and line, both polarizations:
    * 2-D FDTD cell: x = one zone period (periodic / Bloch at normal
      incidence), y = substrate below the relief, air above, PML top and
      bottom; the relief as one rectangle per ring (n from the zone table),
      substrate of the same index (a thick slab through the bottom PML);
    * plane-wave source in the substrate, injected upward, narrow band at the
      line (the RCWA is monochromatic and its n is per line; a dispersive
      resist fit would compare something else);
    * power monitors across the whole period above the relief (T) and in
      the substrate below the source (R); `transmission` = the fraction of the
      source power, `grating` / `gratingn` = the share per diffraction order
      in the exit half space;
    * eta_m = transmission x grating share of order m, m = the sweep's
      focusing order (the same convention: a height descending with r feeds
      m > 0 in rcwaval, which is the +x direction of the cell).
Lumerical 2-D polarization: angle 90 = E out of plane (along the grooves,
the RCWA's TE); angle 0 = E in plane (the RCWA's TM).

Outputs: <run>\rcwa\<stamp>_lumerical\lumerical_zone_check.json (every
run: T, R, per-order shares, eta at the focusing order) and a comparison
table against zone_sweep_results.json in the log.
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
from rcwaval.table import latest_sweep_dir                               # noqa: E402
from rcwaval.zonesweep import focusing_order, load_zone_table            # noqa: E402

UM = 1e-6


def load_lumapi(lumapi_arg: str, log: Any) -> Any:
    r"""Import lumapi from the folder given (or from the folder of the lumapi.py
    given), log where it was loaded from and return the module.

    Parameters
    ----------
    lumapi_arg : str
        `...\api\python` or `...\api\python\lumapi.py`.
    log : callable
        The run's logger.
    """
    api_dir = lumapi_arg[:-len("lumapi.py")] if lumapi_arg.lower().endswith("lumapi.py") else lumapi_arg
    api_dir = api_dir.rstrip("/\\")
    if not os.path.isfile(os.path.join(api_dir, "lumapi.py")):
        raise SystemExit("no lumapi.py in %s" % api_dir)
    sys.path.append(api_dir)
    import lumapi                                                        # type: ignore  # noqa: E402
    log("lumapi loaded from %s" % getattr(lumapi, "__file__", api_dir))
    return lumapi


def open_fdtd(lumapi: Any, hide: bool, log: Any) -> Any:
    """Start an FDTD session and log the Lumerical version it reports."""
    t0 = time.time()
    fdtd = lumapi.FDTD(hide=bool(hide))
    try:
        ver = str(fdtd.version())
    except Exception as exc:                                             # version() missing in some releases
        ver = "unknown (%s)" % exc
    log("Lumerical FDTD session open: version %s (%.1f s to start)" % (ver, time.time() - t0))
    return fdtd


def yee_phase_error(dx_um: float, lam_um: float, n: float, courant: float = 0.99) -> float:
    """Relative numerical wavenumber error k_num / k - 1 of the 2-D Yee scheme
    for on-axis propagation in a medium of index n on a uniform mesh dx (dt at
    the given fraction of the 2-D Courant limit, Lumerical's default 0.99)."""
    lam_n = lam_um / n
    k = 2.0 * np.pi / lam_n
    c_dt = courant * dx_um / np.sqrt(2.0)                # c dt
    w_dt_2 = np.pi * (c_dt / n) / lam_n                  # omega dt / 2
    s = np.sin(w_dt_2) * dx_um / (c_dt / n)              # sin(k_num dx / 2)
    return float(2.0 * np.arcsin(min(s, 1.0)) / dx_um / k - 1.0)


def dispersion_equivalent_dn(dx_um: float, lam_um: float, n: float, h_max_um: float) -> float:
    """The index offset dn that gives the tallest ring the same excess phase
    (resist path minus air path) as the mesh's numerical dispersion: the FDTD
    at this mesh behaves like an exact solver at n + dn (2026-09-21, zone 34
    at 400 nm: predicted 0.0049, the RCWA at n + 0.005 reproduced the FDTD
    to 1 %)."""
    excess = (yee_phase_error(dx_um, lam_um, n) * n - yee_phase_error(dx_um, lam_um, 1.0)) \
        * 2.0 * np.pi * h_max_um / lam_um
    return float(excess * lam_um / (2.0 * np.pi * h_max_um))


def build_and_run(fdtd: Any, period_um: float, step_um: float, heights_um: np.ndarray, n: float,
                  lam_um: float, pol_angle: float, mesh_accuracy: int, dx_um: float,
                  sim_time_fs: float, fsp_path: str) -> Dict[str, Any]:
    """One FDTD run: builds the cell, saves it as fsp_path (the solver refuses
    to run an unsaved project -- that was the dialog), runs, and reads back
    what was simulated (source wavelength, monitor wavelength, mesh cells,
    mesh accuracy, Lumerical's own TE/TM label) next to the results.

    Returns transmission T, reflection R, the order numbers of `gratingn`
    (Lumerical's physical numbering, kx_n = kx0 + n 2 pi / P) and the per-order
    shares of the transmitted power, plus the read-back under "check".
    """
    P = period_um * UM
    h_max = float(heights_um.max()) * UM
    y_sub = -4.0 * UM                       # substrate extent below the relief base (y = 0)
    y_air = h_max + 4.0 * UM
    fdtd.newproject()
    fdtd.addfdtd(dimension="2D", x=0.5 * P, x_span=P, y=0.5 * (y_sub + y_air), y_span=(y_air - y_sub),
                 x_min_bc="Periodic", x_max_bc="Periodic", y_min_bc="PML", y_max_bc="PML",
                 mesh_accuracy=int(mesh_accuracy), simulation_time=sim_time_fs * 1e-15)
    # the line: through the GLOBAL source / monitor settings, which every
    # source and monitor follows unless told otherwise (a wavelength set on
    # the source object alone is ignored while "override global source
    # settings" is off)
    fdtd.setglobalsource("wavelength start", lam_um * UM)
    fdtd.setglobalsource("wavelength stop", lam_um * UM)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("frequency points", 1)
    # substrate: a slab from far below (through the PML) to the relief base
    fdtd.addrect(name="substrate", x=0.5 * P, x_span=2.0 * P, y_min=y_sub - 2.0 * UM, y_max=0.0,
                 material="<Object defined dielectric>", index=float(n))
    for i, h in enumerate(heights_um):
        if h <= 0.0:
            continue
        fdtd.addrect(name="ring_%d" % i, x_min=i * step_um * UM, x_max=(i + 1) * step_um * UM,
                     y_min=0.0, y_max=float(h) * UM, material="<Object defined dielectric>", index=float(n))
    # fine mesh over the relief
    fdtd.addmesh(name="relief_mesh", x=0.5 * P, x_span=P, y_min=-0.2 * UM, y_max=h_max + 0.2 * UM,
                 override_x_mesh=1, override_y_mesh=1, dx=dx_um * UM, dy=dx_um * UM)
    # plane wave from the substrate, upward
    fdtd.addplane(name="source", injection_axis="y", direction="forward", x=0.5 * P, x_span=P,
                  y=y_sub + 1.5 * UM, polarization_angle=float(pol_angle))
    fdtd.addpower(name="T", monitor_type="Linear X", x=0.5 * P, x_span=P, y=h_max + 2.5 * UM)
    fdtd.addpower(name="R", monitor_type="Linear X", x=0.5 * P, x_span=P, y=y_sub + 0.8 * UM)
    fdtd.save(fsp_path)
    fdtd.run()
    tr = np.real(np.asarray(fdtd.transmission("T"))).ravel()
    rf = np.real(np.asarray(fdtd.transmission("R"))).ravel()
    T, R = float(tr[0]), float(-rf[0])                   # backward flux is negative through R
    shares = np.asarray(fdtd.grating("T")).ravel()
    orders = np.asarray(fdtd.gratingn("T")).ravel().astype(int)
    # read-back of what was simulated
    check: Dict[str, Any] = {"n_freq": int(tr.size), "h_max_um": h_max / UM, "y_T_um": (h_max + 2.5 * UM) / UM,
                             "y_R_um": (y_sub + 0.8 * UM) / UM, "y_source_um": (y_sub + 1.5 * UM) / UM,
                             "y_min_um": y_sub / UM, "y_max_um": y_air / UM}
    try:
        check["lam_source_um"] = float(fdtd.getglobalsource("wavelength start")) / UM
        f = np.asarray(fdtd.getresult("T", "f")).ravel()
        check["lam_monitor_um"] = float(299792458.0 / f[0]) / UM
        xg = np.asarray(fdtd.getresult("T", "x")).ravel()
        check["mesh_cells_x"] = int(xg.size)
        check["dx_min_um"] = float(np.min(np.diff(xg))) / UM
        check["dx_max_um"] = float(np.max(np.diff(xg))) / UM
        check["mesh_accuracy"] = int(fdtd.getnamed("FDTD", "mesh accuracy"))
        check["sim_time_fs"] = float(fdtd.getnamed("FDTD", "simulation time")) * 1e15
        check["pol_angle"] = float(fdtd.getnamed("source", "polarization angle"))
        check["index_substrate"] = float(fdtd.getnamed("substrate", "index"))
    except Exception as exc:                                             # a name off in some release
        check["error"] = str(exc)
    return {"T": T, "R": R, "orders": orders.tolist(), "shares": shares.tolist(), "check": check,
            "fsp": fsp_path}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--lumapi", required=True, help="folder that holds lumapi.py")
    ap.add_argument("--sweep")
    ap.add_argument("--zones", default="34,80,158")
    ap.add_argument("--lams", default=None, help="explicit lines [um]; default: --auto")
    ap.add_argument("--auto", type=int, default=3, help="lines per zone chosen among the sweep's solved cells")
    ap.add_argument("--mesh", type=int, default=4)
    ap.add_argument("--dx", type=float, default=0.012, help="relief mesh [um]")
    ap.add_argument("--sim-time", type=float, default=1000.0,
                    help="fs; the solver stops earlier at its auto-shutoff (1e-5 of the peak energy)")
    ap.add_argument("--hide", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, do not import lumapi")
    ap.add_argument("--check-api", action="store_true",
                    help="import lumapi, open and close an FDTD session, report the version, stop")
    args = ap.parse_args(argv)
    if args.check_api:
        lumapi = load_lumapi(args.lumapi, print)
        fdtd = open_fdtd(lumapi, args.hide, print)
        fdtd.close()
        print("API check done")
        return
    run_dir = args.run.rstrip("/\\")
    sweep = args.sweep or latest_sweep_dir(run_dir)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    F = float(cfg.get("focal_um") or cfg["derived"]["focal_um"])
    zones, step_um, lams_all, n_real = load_zone_table(os.path.join(run_dir, "zone_table.npz"))
    results_rcwa: Dict[str, Dict[str, Any]] = {}
    if sweep and os.path.exists(os.path.join(sweep, "zone_sweep_results.json")):
        results_rcwa = json.load(open(os.path.join(sweep, "zone_sweep_results.json")))
    want_z = [int(v) for v in args.zones.split(",")]
    want_l = [float(v) for v in args.lams.split(",")] if args.lams else []
    out_dir = os.path.join(run_dir, "rcwa", time.strftime("%Y%m%d_%H%M%S") + "_lumerical")
    os.makedirs(out_dir, exist_ok=True)
    lines: List[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    log("rcwaval %s | lumerical_zone_check | %s" % (SCRIPT_VERSION, " ".join(sys.argv)))
    plan = []
    for zid in want_z:
        z = next(zz for zz in zones if zz.zone_id == zid)
        lines_z = want_l
        if not lines_z:                                   # --auto: solved lines, spread over the band
            solved = sorted(int(k.split(":")[1]) for k in results_rcwa if int(k.split(":")[0]) == zid)
            if not solved:
                log("  zone %3d: no solved line in the sweep -- skipped" % zid)
                continue
            pick = np.unique(np.round(np.linspace(0, len(solved) - 1, max(1, min(args.auto, len(solved)))))).astype(int)
            lines_z = [float(lams_all[solved[i]]) for i in pick]
        for lam in lines_z:
            il = int(np.argmin(np.abs(lams_all - lam)))
            lam_t, n = float(lams_all[il]), float(n_real[il])
            m, sin_c, eta_s, ok = focusing_order(z, lam_t, n, F)
            key = "%d:%d" % (zid, il)
            ref = results_rcwa.get(key)
            plan.append((z, il, lam_t, n, m, eta_s, ok, ref))
            log("  zone %3d P %5.1f um %2d rings | lam %.3f n %.4f | m %+d scalar x T %.4f%s | RCWA %s"
                % (zid, z.period_um, z.n_rings, lam_t, n, m, eta_s, "" if ok else " (order not well defined)",
                   ("%.4f (TE %.4f TM %.4f, T %.4f R %.4f)" % (ref["eta_rcwa"], ref["eta_te"], ref["eta_tm"],
                                                              ref["sum_T"], ref["sum_R"])) if ref else "not in sweep"))
    if args.dry_run:
        log("dry run")
        return
    lumapi = load_lumapi(args.lumapi, log)
    fdtd = open_fdtd(lumapi, args.hide, log)
    settings = {"lumapi": args.lumapi, "sweep": sweep, "zones": want_z, "lams": want_l or "auto %d" % args.auto,
                "mesh_accuracy": args.mesh, "dx_um": args.dx, "sim_time_fs": args.sim_time}
    log("settings: %s" % settings)
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump({"script_version": SCRIPT_VERSION, "command": " ".join(sys.argv), "settings": settings}, fh, indent=1)
    out: Dict[str, Any] = {}
    fsp_dir = os.path.join(out_dir, "fsp")
    os.makedirs(fsp_dir, exist_ok=True)
    W = 3
    try:
        for z, il, lam_t, n, m, eta_s, ok, ref in plan:
            # Lumerical numbers the orders physically (kx_n = kx0 + n 2 pi / P);
            # the solver's index is the mirror (rcwa1d docstring, test 7)
            n_lum = -int(m)
            rec: Dict[str, Any] = {"zone": z.zone_id, "lam_um": lam_t, "n": n, "m_solver": m, "n_lumerical": n_lum,
                                   "eta_scalar": eta_s}
            for pol_name, angle in (("te", 90.0), ("tm", 0.0)):
                t0 = time.time()
                fsp = os.path.abspath(os.path.join(fsp_dir, "zone%03d_lam%.3f_%s.fsp" % (z.zone_id, lam_t, pol_name)))
                r = build_and_run(fdtd, z.period_um, step_um, z.h_um, n, lam_t, angle, args.mesh, args.dx,
                                  args.sim_time, fsp)
                sh = dict(zip(r["orders"], r["shares"]))
                share, share_mirror = sh.get(n_lum, 0.0), sh.get(-n_lum, 0.0)
                win_fdtd = {int(k): r["T"] * sh.get(-k, 0.0) for k in range(m - W, m + W + 1)}   # solver index
                rec[pol_name] = dict(r, eta_focus=r["T"] * share, eta_mirror=r["T"] * share_mirror,
                                     window_solver_index=win_fdtd, seconds=time.time() - t0)
                c = r["check"]
                log("  zone %3d lam %.3f %s: T %.4f R %.4f T+R %.4f | Lumerical order %+d share %.4f -> eta %.4f "
                    "(mirror order %+d: %.4f) | RCWA %s | %.0f s"
                    % (z.zone_id, lam_t, pol_name.upper(), r["T"], r["R"], r["T"] + r["R"], n_lum, share,
                       r["T"] * share, -n_lum, r["T"] * share_mirror,
                       ("%.4f" % ref["eta_" + pol_name]) if ref else "--", time.time() - t0))
                log("      simulated: lam source %s um, monitor %s um (%d pt) | mesh %s cells in x, dx %s..%s um, "
                    "accuracy %s | pol angle %s | n_sub %s | cell y %.1f..%.1f um, h_max %.3f, source y %.1f, "
                    "R y %.1f, T y %.3f | %s"
                    % (("%.4f" % c["lam_source_um"]) if "lam_source_um" in c else "?",
                       ("%.4f" % c["lam_monitor_um"]) if "lam_monitor_um" in c else "?", c["n_freq"],
                       c.get("mesh_cells_x", "?"),
                       ("%.4f" % c["dx_min_um"]) if "dx_min_um" in c else "?",
                       ("%.4f" % c["dx_max_um"]) if "dx_max_um" in c else "?",
                       c.get("mesh_accuracy", "?"), c.get("pol_angle", "?"), c.get("index_substrate", "?"),
                       c["y_min_um"], c["y_max_um"], c["h_max_um"], c["y_source_um"], c["y_R_um"], c["y_T_um"],
                       os.path.basename(fsp)))
                if "error" in c:
                    log("      read-back error: %s" % c["error"])
                dn_eq = dispersion_equivalent_dn(args.dx, lam_t, n, float(z.h_um.max()))
                log("      mesh dispersion: this FDTD ~ an exact solve at n %+.4f (tallest ring %+.2f rad); "
                    "dn 0.0005 would need dx %.4f um -- rcwa_converge.py --dn %.4f reproduces it"
                    % (dn_eq, 2.0 * np.pi * dn_eq * float(z.h_um.max()) / lam_t,
                       args.dx * np.sqrt(0.0005 / max(dn_eq, 1e-9)), dn_eq))
                log("      window (solver index m: FDTD this pol / RCWA unpolarized): "
                    + "  ".join("%+d: %.4f / %s" % (k, win_fdtd[k], ("%.4f" % ref["window"][str(k)])
                                                     if ref and str(k) in ref["window"] else "--")
                                for k in range(m - W, m + W + 1)))
            rec["eta_focus"] = 0.5 * (rec["te"]["eta_focus"] + rec["tm"]["eta_focus"])
            rec["eta_mirror"] = 0.5 * (rec["te"]["eta_mirror"] + rec["tm"]["eta_mirror"])
            log("  zone %3d lam %.3f unpolarized: FDTD %.4f (mirror side %.4f) | RCWA %s | ratio %s"
                % (z.zone_id, lam_t, rec["eta_focus"], rec["eta_mirror"],
                   ("%.4f" % ref["eta_rcwa"]) if ref else "--",
                   ("%.3f" % (rec["eta_focus"] / max(ref["eta_rcwa"], 1e-9))) if ref else "--"))
            out["%d:%d" % (z.zone_id, il)] = rec
            with open(os.path.join(out_dir, "lumerical_zone_check.json"), "w") as fh:
                json.dump(out, fh, indent=1)
    finally:
        try:
            fdtd.close()
        except Exception:
            pass
    with open(os.path.join(out_dir, "lumerical_zone_check.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("done -> %s" % out_dir)


if __name__ == "__main__":
    main()
