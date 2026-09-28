"""zonesweep -- the design's zone_table.npz through the rigorous solver:
eta_RCWA of every zone into its focusing order at every design line,
the scalar efficiency of the same ring profile, their ratio as the
efficiency_corr.npz the design consumes (MDLProblem.apply_efficiency).

Per zone z (period P_z = zone width, heights h_i on rings of width DELTA,
centre radius r_c) and line lam (resist index n(lam) from the table):

    structure   substrate (n) -> resist relief h_i (n, air above) -> air,
                light from the substrate, normal incidence
    focusing order  |m| = round(P_z sin(theta_c) / lam),
                    sin(theta_c) = r_c / sqrt(r_c^2 + F^2), the sign taken
                    from the scalar spectrum (the side the blaze feeds)
    eta_scalar  |c_m|^2 of the piecewise-constant transmission
                exp(i k (n - 1) h_i) -- nscval.tea.profile_orders, the
                exact scalar integral of the staircase (it carries the
                flat-tread sinc factor the design's S table also carries)
                times the Fresnel transmission 4 n / (n + 1)^2 of the flat
                interface
    corr        eta_RCWA / eta_scalar (mdl.zones.relative_correction_table:
                1 where eta_scalar < 0.02, clipped to [0, 2])

The RCWA cost is (2N+1)^3 per layer per polarization, N = margin n P / lam,
layers = distinct heights in the zone: the wide inner zones (P of hundreds
of um, paraxial, P/lam > 100) are where the thin-element model is exact
and the solver is slowest, so zones with P > p_max_um are not solved and
get corr = 1 (listed in the log). The sweep runs the (zone, line) tasks
in parallel, largest first, and checkpoints every result, so an
interrupted run resumes.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .profiles import staircase
from .rcwa1d import harmonics_for, solve


@dataclass
class ZoneRecord:
    """One zone of the table.

    Attributes
    ----------
    zone_id, r_in, r_out, r_center, period_um   as written by mdl.zones
    h_um        heights of the zone's rings [um], inner to outer
    n_rings     len(h_um)
    n_levels    distinct non-zero heights = RCWA layers
    """
    zone_id: int
    r_in: float
    r_out: float
    r_center: float
    period_um: float
    h_um: np.ndarray

    @property
    def n_rings(self) -> int:
        return int(self.h_um.size)

    @property
    def n_levels(self) -> int:
        return len(set(float(v) for v in np.round(self.h_um, 9) if v > 0.0))


def load_zone_table(path: str) -> Tuple[List[ZoneRecord], float, np.ndarray, np.ndarray]:
    """(zones, ring_width_um, lams_um, n_real) from zone_table.npz."""
    z = np.load(path, allow_pickle=True)
    zones = [ZoneRecord(int(z["zone_id"][i]), float(z["r_in"][i]), float(z["r_out"][i]),
                        float(z["r_center"][i]), float(z["period_um"][i]),
                        np.asarray(z["h_profile"][i], float))
             for i in range(z["zone_id"].size)]
    return zones, float(z["ring_width_um"]), np.asarray(z["lams_um"], float), np.asarray(z["n_real"], float)


def scalar_orders(h_um: np.ndarray, lam_um: float, n: float, orders: Sequence[int]) -> np.ndarray:
    """|c_m|^2 of exp(i k (n - 1) h_i) on equal-width rings (nscval.tea formula)."""
    phi = 2.0 * np.pi * (n - 1.0) * h_um / lam_um
    N = phi.size
    k = np.arange(N) + 0.5
    out = np.empty(len(orders))
    for j, m in enumerate(orders):
        amp = np.mean(np.exp(1j * phi) * np.exp(-2j * np.pi * m * k / N))
        out[j] = abs(amp) ** 2 * np.sinc(m / N) ** 2
    return out


def fresnel(n: float) -> float:
    return 4.0 * n / (n + 1.0) ** 2


#: a zone is a local grating for the design order only when P sin(theta_c) / lam
#: sits within this distance of an integer and the zone has at least these rings
ORDER_TOL = 0.3
MIN_RINGS = 4


def focusing_order(zone: ZoneRecord, lam_um: float, n: float, focal_um: float
                   ) -> Tuple[int, float, float, bool]:
    """(m_focus in the solver's convention, sin theta_c, eta_scalar(m_focus)
    incl. Fresnel, well_defined). The focusing direction sin theta_c =
    r_c / sqrt(r_c^2 + F^2) is an integer order of the zone's own period
    only when P sin theta_c / lam is near an integer -- a zone of a few
    rings, or one the optimizer left irregular, has no single order there
    and is reported as not well defined (excluded from the table)."""
    sin_c = zone.r_center / math.sqrt(zone.r_center ** 2 + focal_um ** 2)
    m_frac = zone.period_um * sin_c / lam_um
    m_abs = int(round(m_frac))
    ok = abs(m_frac - m_abs) <= ORDER_TOL and zone.n_rings >= MIN_RINGS and m_abs > 0
    if m_abs == 0:
        return 0, sin_c, float(scalar_orders(zone.h_um, lam_um, n, [0])[0]) * fresnel(n), ok
    # scalar_orders follows tea's sign (a phase decreasing with x feeds m < 0);
    # the solver's order index is the opposite (rcwa1d docstring, tests)
    eta = scalar_orders(zone.h_um, lam_um, n, [-m_abs, m_abs])
    m_tea = -m_abs if eta[0] >= eta[1] else m_abs
    return -m_tea, sin_c, float(max(eta)) * fresnel(n), ok


def cost_seconds(period_um: float, lam_um: float, n: float, n_levels: int, margin: float) -> float:
    """Wall-time model of one solve, single BLAS thread (fitted 2026-09-21 on a
    2-core sandbox: 0.02 + 0.7 (M/371)^3 s per eigen-decomposition, two
    polarizations per layer, M = 2N + 1)."""
    N = harmonics_for(period_um, lam_um, n, margin)
    M = 2 * N + 1
    return 2.0 * n_levels * (0.02 + 0.7 * (M / 371.0) ** 3)


def solve_task(args: Tuple[int, float, float, float, float, float, np.ndarray, float, int, float]
               ) -> Dict[str, Any]:
    """One (zone, line): returns the efficiencies. Module-level for Pool."""
    zi, lam, n, P, step, r_center, h, focal, m_focus, margin = args
    N = harmonics_for(P, lam, n, margin)
    t0 = time.time()
    g = staircase(h, step, n, n, 1.0, period_um=P)
    e = solve(g, lam, N)
    dt = time.time() - t0
    w = 3
    window = {int(m): e.order(int(m)) for m in range(m_focus - w, m_focus + w + 1)}
    return {"zone": zi, "lam_um": lam, "n_harm": N, "seconds": dt, "eta_rcwa": e.order(m_focus),
            "eta_te": e.order(m_focus, "te"), "eta_tm": e.order(m_focus, "tm"),
            "sum_T": float(e.T.sum()), "sum_R": float(e.R.sum()), "window": window}


class ZoneSweep:
    """The sweep over a run folder's zone table.

    Attributes
    ----------
    run_dir, out_dir     the run folder and <run>/rcwa/<stamp>_zones
    zones, step_um, lams, n_real   the table
    focal_um             from config.json
    p_max_um             zones with a longer period are not solved (corr = 1)
    margin               harmonics = ceil(margin n P / lam)
    lam_idx              indices of the lines to solve (lam_stride)
    workers              processes
    """

    def __init__(self, run_dir: str, out_dir: str, p_max_um: float = 60.0, margin: float = 1.2,
                 lam_stride: int = 1, workers: int = 1, zone_ids: Optional[Sequence[int]] = None,
                 log: Any = print) -> None:
        self.run_dir, self.out_dir, self.log = run_dir, out_dir, log
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        self.focal_um = float(cfg.get("focal_um") or cfg["derived"]["focal_um"])
        self.zones, self.step_um, self.lams, self.n_real = load_zone_table(os.path.join(run_dir, "zone_table.npz"))
        self.p_max_um, self.margin, self.workers = float(p_max_um), float(margin), int(workers)
        stride = max(1, int(lam_stride))
        self.lam_idx = list(range(0, self.lams.size, stride))
        if stride > 1 and (self.lams.size - 1) not in self.lam_idx:
            self.lam_idx.append(self.lams.size - 1)
        self.zone_ids = [int(v) for v in zone_ids] if zone_ids else [z.zone_id for z in self.zones]
        self.results: Dict[str, Dict[str, Any]] = {}
        self.ckpt = os.path.join(out_dir, "zone_sweep_results.json")

    # -- planning ----------------------------------------------------------------------
    def tasks(self) -> List[Tuple[Any, ...]]:
        out = []
        for z in self.zones:
            if z.zone_id not in self.zone_ids or z.period_um > self.p_max_um or z.n_levels == 0:
                continue
            for il in self.lam_idx:
                lam, n = float(self.lams[il]), float(self.n_real[il])
                m_focus, sin_c, eta_s, ok = focusing_order(z, lam, n, self.focal_um)
                key = "%d:%d" % (z.zone_id, il)
                if key in self.results or not ok:
                    continue
                out.append((z.zone_id, lam, n, z.period_um, self.step_um, z.r_center, z.h_um, self.focal_um,
                            m_focus, self.margin))
        # largest first: the long ones start early and the small ones fill the gaps
        out.sort(key=lambda t: -cost_seconds(t[3], t[1], t[2], self.zone_by_id(t[0]).n_levels, self.margin))
        return out

    def zone_by_id(self, zid: int) -> ZoneRecord:
        return next(z for z in self.zones if z.zone_id == zid)

    def plan(self) -> float:
        """Log the table and the cost; return the estimated seconds."""
        log = self.log
        log("zone table: %d zones, ring width %.3f um, %d lines (%.2f-%.2f um), F = %.1f mm"
            % (len(self.zones), self.step_um, self.lams.size, self.lams.min(), self.lams.max(),
               self.focal_um / 1000.0))
        log("lines solved: %s" % ", ".join("%.3f" % self.lams[i] for i in self.lam_idx))
        log("  zone  r_c(um)  P(um)  rings  levels  |m| at %.2f/%.2f um  lines ok  harmonics   est. s"
            % (self.lams[self.lam_idx[0]], self.lams[self.lam_idx[-1]]))
        total = 0.0
        skipped = []
        undefined = []
        for z in self.zones:
            if z.zone_id not in self.zone_ids:
                continue
            if z.period_um > self.p_max_um or z.n_levels == 0:
                skipped.append(z)
                continue
            sec = 0.0
            ms = []
            hs = []
            n_ok = 0
            for il in self.lam_idx:
                lam, n = float(self.lams[il]), float(self.n_real[il])
                m_focus, sin_c, eta_s, ok = focusing_order(z, lam, n, self.focal_um)
                ms.append(abs(m_focus))
                if ok:
                    n_ok += 1
                    sec += cost_seconds(z.period_um, lam, n, z.n_levels, self.margin)
                    hs.append(harmonics_for(z.period_um, lam, n, self.margin))
            if n_ok == 0:
                undefined.append(z)
                continue
            total += sec
            log("  %4d  %7.0f  %5.1f  %5d  %6d   %3d / %3d         %2d / %2d   %3d..%3d    %7.0f"
                % (z.zone_id, z.r_center, z.period_um, z.n_rings, z.n_levels, ms[0], ms[-1],
                   n_ok, len(self.lam_idx), min(hs), max(hs), sec))
        if skipped:
            log("paraxial / flat, not solved, corr = 1 (P > %.0f um: P/lam > %.0f, the thin-element model "
                "holds): zones %s" % (self.p_max_um, self.p_max_um / self.lams.max(),
                                       ", ".join("%d (P %.0f um)" % (z.zone_id, z.period_um) for z in skipped)))
        if undefined:
            log("no well-defined focusing order at any line (fewer than %d rings, or P sin(theta_c)/lam "
                "farther than %.1f from an integer), left out of the table: zones %s"
                % (MIN_RINGS, ORDER_TOL, ", ".join("%d (P %.0f um, %d rings)" % (z.zone_id, z.period_um, z.n_rings)
                                                    for z in undefined)))
        log("estimated %.0f min of solver time, %d workers -> ~%.0f min wall"
            % (total / 60.0, self.workers, total / 60.0 / max(1, self.workers)))
        return total

    # -- running -----------------------------------------------------------------------
    def load_checkpoint(self) -> int:
        if os.path.exists(self.ckpt):
            with open(self.ckpt) as fh:
                self.results = json.load(fh)
        return len(self.results)

    def save_checkpoint(self) -> None:
        tmp = self.ckpt + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.results, fh)
        os.replace(tmp, self.ckpt)

    def run(self) -> None:
        log = self.log
        tasks = self.tasks()
        log("%d (zone, line) tasks to solve (%d already in the checkpoint)" % (len(tasks), len(self.results)))
        t0 = time.time()
        done = 0

        def take(res: Dict[str, Any]) -> None:
            nonlocal done
            il = int(np.argmin(np.abs(self.lams - res["lam_um"])))
            key = "%d:%d" % (res["zone"], il)
            self.results[key] = res
            done += 1
            if done % 5 == 0 or done == len(tasks):
                self.save_checkpoint()
            z = self.zone_by_id(res["zone"])
            lam, n = float(self.lams[il]), float(self.n_real[il])
            m_focus, sin_c, eta_s, ok = focusing_order(z, lam, n, self.focal_um)
            log("  [%4d/%4d %6.0f s] zone %3d P %5.1f um  lam %.3f  N %3d  m %+3d: eta_RCWA %.4f (TE %.4f TM %.4f) "
                "scalar x T %.4f  ratio %.3f | sum T %.4f R %.4f | %.1f s"
                % (done, len(tasks), time.time() - t0, res["zone"], z.period_um, lam, res["n_harm"], m_focus,
                   res["eta_rcwa"], res["eta_te"], res["eta_tm"], eta_s,
                   res["eta_rcwa"] / max(eta_s, 1e-9), res["sum_T"], res["sum_R"], res["seconds"]))

        if self.workers > 1 and len(tasks) > 1:
            with Pool(self.workers) as pool:
                for res in pool.imap_unordered(solve_task, tasks):
                    take(res)
        else:
            for t in tasks:
                take(solve_task(t))
        self.save_checkpoint()

    # -- the table -----------------------------------------------------------------------
    def table(self) -> Dict[str, Any]:
        """eta_rcwa / eta_scalar per (line, zone) (nan where not solved) and the
        design's corr table over the FULL line list: solved zones with their
        lines interpolated over wavelength; paraxial zones (P > p_max) at 1.0
        (the model holds there); zones without a well-defined order left out
        (the design interpolates across them in r); an r = 0 anchor at 1.0."""
        K, Z = self.lams.size, len(self.zones)
        eta_r = np.full((K, Z), np.nan)
        eta_s = np.full((K, Z), np.nan)
        for key, res in self.results.items():
            zid, il = (int(v) for v in key.split(":"))
            zi = next(i for i, z in enumerate(self.zones) if z.zone_id == zid)
            z = self.zones[zi]
            lam, n = float(self.lams[il]), float(self.n_real[il])
            eta_r[il, zi] = res["eta_rcwa"]
            eta_s[il, zi] = focusing_order(z, lam, n, self.focal_um)[2]
        ratio = np.where(np.isfinite(eta_r) & (eta_s > 0.02), eta_r / np.maximum(eta_s, 1e-12), np.nan)
        cols: List[np.ndarray] = [np.ones(K)]
        radii: List[float] = [0.0]
        kinds: List[str] = ["anchor"]
        for zi, z in enumerate(self.zones):
            col = ratio[:, zi]
            ok = np.isfinite(col)
            if ok.sum() > 0:
                cols.append(np.interp(self.lams, self.lams[ok], col[ok]) if ok.sum() < K else col.copy())
                radii.append(z.r_center)
                kinds.append("solved")
            elif z.period_um > self.p_max_um and z.zone_id in self.zone_ids:
                cols.append(np.ones(K))
                radii.append(z.r_center)
                kinds.append("paraxial")
        order = np.argsort(radii)
        corr = np.clip(np.stack(cols, axis=1)[:, order], 0.0, 2.0)
        r_um = np.asarray(radii)[order]
        return {"lam_um": self.lams, "r_um": r_um, "eta": corr, "kinds": [kinds[i] for i in order],
                "eta_rcwa": eta_r, "eta_scalar": eta_s, "ratio": ratio,
                "zone_r_um": np.array([z.r_center for z in self.zones]),
                "periods_um": np.array([z.period_um for z in self.zones])}

    def write(self) -> str:
        t = self.table()
        path = os.path.join(self.out_dir, "efficiency_corr.npz")
        np.savez(path, lam_um=t["lam_um"], r_um=t["r_um"], eta=t["eta"])
        np.savez(os.path.join(self.out_dir, "zone_sweep.npz"), lam_um=t["lam_um"], r_um=t["r_um"], eta=t["eta"],
                 kinds=np.array(t["kinds"]), eta_rcwa=t["eta_rcwa"], eta_scalar=t["eta_scalar"], ratio=t["ratio"],
                 zone_r_um=t["zone_r_um"], periods_um=t["periods_um"],
                 p_max_um=self.p_max_um, margin=self.margin, focal_um=self.focal_um)
        solved = np.isfinite(t["ratio"])
        self.log("corr table: %d lines x %d radii (%d solved zones, %d paraxial at 1.0, r = 0 anchor), "
                 "%d (line, zone) cells solved, ratio %.3f-%.3f (median %.3f)"
                 % (t["lam_um"].size, t["r_um"].size, t["kinds"].count("solved"), t["kinds"].count("paraxial"),
                    int(solved.sum()),
                    np.nanmin(t["ratio"]) if solved.any() else float("nan"),
                    np.nanmax(t["ratio"]) if solved.any() else float("nan"),
                    np.nanmedian(t["ratio"]) if solved.any() else float("nan")))
        self.log("wrote %s (config efficiency_corr_npz)" % path)
        self.figure(t)
        return path

    def figure(self, t: Dict[str, Any]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:                        # pragma: no cover
            self.log("figure skipped (%s)" % exc)
            return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        im = ax.imshow(t["ratio"], aspect="auto", origin="lower", vmin=0.5, vmax=1.2, cmap="RdBu_r",
                       extent=[t["zone_r_um"].min(), t["zone_r_um"].max(), 1000 * t["lam_um"].min(),
                               1000 * t["lam_um"].max()])
        ax.set_xlabel("zone centre radius (um)")
        ax.set_ylabel("wavelength (nm)")
        ax.set_title("eta_RCWA / eta_scalar per zone (white = not solved)", fontsize=9)
        fig.colorbar(im, ax=ax)
        ax = axes[1]
        for il in range(0, t["lam_um"].size, max(1, t["lam_um"].size // 5)):
            ax.plot(t["r_um"], t["eta"][il], marker=".", ms=3, label="%.0f nm" % (1000 * t["lam_um"][il]))
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("zone centre radius (um)")
        ax.set_ylabel("corr for the design")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "fig_zone_sweep.png"), dpi=130)
        plt.close(fig)
        self.log("saved fig_zone_sweep.png")
