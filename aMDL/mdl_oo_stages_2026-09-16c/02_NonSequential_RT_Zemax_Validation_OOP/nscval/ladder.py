"""ladder -- where does the thin-element model break for DELTA-wide treads?

For every case (N' treads, slope fraction) of LADDER_SETTINGS and every
design line: an equal-step staircase of N' treads of width DELTA (the
ring width) and riser h_s = DELTA sin(theta) / (n - 1) -- the lens's local
staircase at that slope, shortened to a period P' = N' DELTA the RCWA
converges on -- is put on the Diffraction Grating object through
srg_step_RCWA (Depth = (N' - 1) h_s, the SPAN of the N' levels; Number of
Steps = N'; Index Grate = n(lam) of the design's resist model, Index Env 1).
The orders m = p0 - w .. p0 + w around the staircase's design order
p0 = round((n - 1) N' h_s / lam) are traced one at a time (the srg_step
label is the physical order) and eta_RCWA(m) is compared with

    eta_ref(m) = T_fresnel * eta_TEA(m)      (nscval.tea.staircase_orders)

of the same staircase. The tables give, per case and line, the ratio at
p0 and the ratio of the window sums: 1.0 where the thin-element model
holds, less where the treads are too narrow or the risers too tall for
it -- the correction the design's efficiency_corr_npz mechanism consumes.
The harmonic count follows n P'/lam per line (max_order "auto"); a line
the DLL refuses (energy-balance check, design order reads 0) is retried
at Max Order + delta over max_order_retries; Test Mode is on and the
DLL's log is read after every line. The run prints its cost estimate
before tracing.

Outputs: ladder.npz, ladder.json, fig_ladder.png, nsc_ladder.zos.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import tea
from .base import PKG_ROOT, NscAnalysis
from .dlls import slot_of, slots
from .nsc import DiffractionTab, DllLog, NscSystem, NscTrace, order_geometry
from .corr import correction_from_ladder, write_correction
from .settings import (cutoff_orders, fresnel_transmission, LADDER_SETTINGS, TRACE_SETTINGS,
                       rcwa_seconds_per_ray_order)


def design_index_model() -> Callable[[float], float]:
    """n(lam) of the design (01_design_oo/mdl/material.py), else the same
    Cauchy fit."""
    oo = os.path.join(PKG_ROOT, "01_design_oo")
    if os.path.isdir(os.path.join(oo, "mdl")):
        if oo not in sys.path:
            sys.path.insert(0, oo)
        try:
            from mdl.material import n_az4562
            return lambda lam: float(n_az4562(lam))
        except Exception:
            pass
    return lambda lam: 1.594 + 0.01152 / lam ** 2


class LadderCase:
    """One short staircase.

    Attributes
    ----------
    n_treads    N' treads per period
    slope       sin(theta) of the local blaze (a fraction of the rim NA)
    period_um   P' = N' DELTA
    riser_um    h_s = DELTA sin(theta) / (n_mid - 1)
    depth_um    d' = N' h_s, the depth the scalar reference is written for
                (N' levels of riser h_s)
    dll_depth_um  what srg_step's "Depth (um)" receives: (N' - 1) h_s, the
                span of the levels ("span" convention), or N' h_s ("levels")
    step_um     the tread actually used (DELTA stretched by the nudge)
    nudge       the stretch factor that cleared the cut-off coincidences
    """

    def __init__(self, n_treads: int, slope: float, step_um: float, riser_um: float,
                 depth_convention: str = "span") -> None:
        self.n_treads = int(n_treads)
        self.slope = float(slope)
        self.step_um = float(step_um)
        self.period_um = self.n_treads * float(step_um)
        self.riser_um = float(riser_um)
        self.depth_um = self.n_treads * self.riser_um
        self.dll_depth_um = ((self.n_treads - 1) if depth_convention == "span" else self.n_treads) * self.riser_um
        self.nudge: float = 0.0

    def label(self) -> str:
        return "N%d_s%.3f" % (self.n_treads, self.slope)


class TeaLadder(NscAnalysis):
    MODE = "ladder"
    DESCRIPTION = ("short equal-step staircases with the design's tread and riser, srg_step_RCWA "
                   "vs the scalar staircase x Fresnel: the TEA-validity map")

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.s: Dict[str, Any] = dict(LADDER_SETTINGS)
        self.t: Dict[str, Any] = dict(TRACE_SETTINGS)
        for key, v in self.overrides.items():
            (self.s if key in self.s else self.t)[key] = v
        s, ctx = self.s, self.ctx
        if s["step_um"] == "ring":
            s["step_um"] = float(ctx.cfg.get("ring_width_um", 2.0))
        if s["lams_um"] == "design":
            lams = ctx.design_lams()
            stride = max(1, int(s.get("lam_stride", 1)))
            s["lams_um"] = lams[::stride] if stride > 1 else lams
            if stride > 1 and lams and lams[-1] not in s["lams_um"]:
                s["lams_um"] = list(s["lams_um"]) + [lams[-1]]
        if s["na"] == "design":
            D, F = ctx.cfg.get("diameter_um"), ctx.cfg.get("focal_um")
            if not D or not F:
                raise SystemExit("the run's config.json has no diameter_um / focal_um; pass --na <value>")
            r = 0.5 * float(D)
            s["na"] = r / math.sqrt(r * r + float(F) ** 2)
        s["na"] = float(s["na"])
        self.n_of = design_index_model() if s["n_grate"] == "design" \
            else (lambda lam, v=float(s["n_grate"]): v)
        lam_list = [float(v) for v in s["lams_um"]]
        s["lam_mid_um"] = 0.5 * (min(lam_list) + max(lam_list)) if lam_list else 0.6
        s["n_mid"] = float(self.n_of(s["lam_mid_um"]))
        # a period that puts an order EXACTLY at cut-off at any line makes the srg
        # DLL refuse every ray (cutoff_orders); ring widths like 2.0 um on a 50-nm
        # comb hit that constantly (6 / 0.4 = 15). The tread is stretched by the
        # smallest factor in period_nudges that clears every line: the physics
        # (tread / lam, riser / lam) moves by that percent, nothing else.
        nudges = [float(v) for v in s.get("period_nudges", [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04])]
        self.cases: List[LadderCase] = []
        for frac in s["slopes"]:
            sin_t = float(frac) * s["na"]
            riser = (s["step_um"] * sin_t / (s["n_mid"] - float(s["n_env"]))
                     if s["riser_um"] == "design" else float(s["riser_um"]))
            for n_tr in s["treads"]:
                eps = self.clear_cutoffs(int(n_tr) * s["step_um"], lam_list, nudges)
                self.cases.append(LadderCase(int(n_tr), sin_t, s["step_um"] * (1.0 + eps), riser,
                                             str(s.get("depth_convention", "span"))))
                self.cases[-1].nudge = eps
        # every ray is identical: LADDER_SETTINGS["rays"] replaces the statistical count
        self.t["analysis_rays"] = int(self.overrides.get("analysis_rays", s.get("rays", 4)))
        self.sign: Optional[int] = None if s["order_sign"] == "auto" else int(s["order_sign"])

    def clear_cutoffs(self, period_um: float, lams: List[float], nudges: List[float]) -> float:
        """The first stretch factor eps in `nudges` such that P (1 + eps) has no
        order exactly at cut-off at any line, in air or in the resist."""
        for eps in nudges:
            P = period_um * (1.0 + eps)
            if not any(cutoff_orders(lam, P, [float(self.s["n_env"]), self.n_of(lam)]) for lam in lams):
                return eps
        return nudges[-1] if nudges else 0.0

    def settings_record(self) -> Dict[str, Any]:
        return {"ladder": self.s, "trace": self.t,
                "cases": [dict(n_treads=c.n_treads, slope=c.slope, period_um=c.period_um,
                               riser_um=c.riser_um, depth_um=c.depth_um, dll_depth_um=c.dll_depth_um,
                               nudge=c.nudge) for c in self.cases]}

    def harmonics(self, period_um: float, lam: float, n: float) -> Tuple[int, bool]:
        """(Max Order for this line, converged?) from n P'/lam and the margin."""
        s = self.s
        need = n * period_um / lam
        cap = int(s["max_order_cap"])
        if s["max_order"] == "auto":
            mo = int(min(cap, max(20, math.ceil(float(s["harmonic_margin"]) * need))))
        else:
            mo = int(s["max_order"])
        return mo, float(s["harmonic_margin"]) * need <= mo + 1e-9

    def run(self) -> None:
        log, s, t = self.log, self.s, self.t
        lams: List[float] = [float(v) for v in s["lams_um"]]
        w = int(s["order_window"])
        n_env = float(s["n_env"])
        S = NscSystem(self.session, log)
        log.section("build the NSC system", "source -> short staircase grating -> detector")
        S.add_source_ellipse(-10.0, s["beam_half_mm"], s["beam_half_mm"],
                             t["analysis_rays"], t["layout_rays"], t["source_power_w"])
        gr = S.add_diffraction_grating(0.0, s["grating_clear_mm"], 1.0 / self.cases[0].period_um, 1,
                                       thickness_mm=s["grating_thickness_mm"],
                                       material=s["grating_material"])
        S.add_detector_rect(s["detector_z_mm"], s["detector_half_mm"], s["detector_half_mm"],
                            s["detector_pixels"])
        tab = DiffractionTab(S, gr)
        tab.use_dll(s["dll"], 0, 0)
        names = tab.names()
        trace = NscTrace(S, t["split_rays"], t["scatter_rays"], t["use_polarization"],
                         t["ignore_errors"])
        det = S.objects["detector"]
        log("rim NA %.4f, tread DELTA %.3f um, n(%.3f um) = %.4f, %d rays per trace, window +/-%d orders"
            % (s["na"], s["step_um"], s["lam_mid_um"], s["n_mid"], t["analysis_rays"], w))
        log("  case          N'  slope    P'(um)  riser(um)  depth(um)   harmonics n P'/lam at %.2f / %.2f um"
            "   (P' stretched where the tread is marked *)" % (min(lams), max(lams)))
        est_total = 0.0
        for c in self.cases:
            hs = [self.harmonics(c.period_um, lam, self.n_of(lam)) for lam in lams]
            for lam, (mo, okc) in zip(lams, hs):
                est_total += rcwa_seconds_per_ray_order(mo, c.n_treads * int(s["layers_per_step"])) \
                    * t["analysis_rays"] * (2 * w + 1)
            log("  %-12s %3d  %.4f  %7.3f%s %8.4f  %9.4f   %6.1f / %5.1f -> Max Order %d..%d%s"
                % (c.label(), c.n_treads, c.slope, c.period_um, "*" if c.nudge else " ", c.riser_um, c.depth_um,
                   self.n_of(min(lams)) * c.period_um / min(lams), self.n_of(max(lams)) * c.period_um / max(lams),
                   min(m for m, _ in hs), max(m for m, _ in hs),
                   "" if all(k for _, k in hs) else "  (UNCONVERGED at the short lines)"))
        log("estimated trace time %.0f min (cost model 0.83 s x ((2N+1)/61)^3 x layers/5 per ray per order)"
            % (est_total / 60.0))

        n_case, n_lam, n_win = len(self.cases), len(lams), 2 * w + 1
        eta_r = np.full((n_case, n_lam, n_win), np.nan)
        eta_t = np.full_like(eta_r, np.nan)
        p_tab = np.zeros((n_case, n_lam))
        p0_tab = np.zeros((n_case, n_lam), dtype=int)
        mo_tab = np.zeros((n_case, n_lam), dtype=int)
        conv_tab = np.ones((n_case, n_lam), dtype=bool)
        refused = np.zeros((n_case, n_lam), dtype=bool)
        conv: Dict[str, Any] = {}
        dll_log = DllLog(str(s["dll"]))
        retries = [int(v) for v in s.get("max_order_retries", [])]
        log("DLL log (Test Mode %s): %s" % (s.get("test_mode", 0), dll_log.path))
        for ic, c in enumerate(self.cases):
            P, d, N = c.period_um, c.depth_um, c.n_treads
            S.par(gr, 10, 1.0 / P)
            log.section("case %d/%d: %s -- P' = %.3f um, N' = %d treads of %.4f um%s, riser %.4f um, "
                        "depth %.4f um (DLL Depth %.4f um)"
                        % (ic + 1, n_case, c.label(), P, N, c.step_um,
                           (" (DELTA stretched %.1f %% to clear cut-off coincidences)" % (100 * c.nudge)) if c.nudge else "",
                           c.riser_um, d, c.dll_depth_um))
            for il, lam in enumerate(lams):
                n = self.n_of(lam)
                t_fres = fresnel_transmission(n, n_env)
                p = tea.waves_of_depth(d, lam, n, n_env)
                p0 = tea.design_order(p)
                p_tab[ic, il], p0_tab[ic, il] = p, p0
                mo, converged = self.harmonics(P, lam, n)
                conv_tab[ic, il] = converged
                hits = cutoff_orders(lam, P, [n_env, n])
                if hits:
                    log("  lam %.3f: order %d exactly at cut-off in n = %.4f (m = n P / lam = %.4f) -- "
                        "the srg DLL would refuse every ray; line skipped"
                        % (lam, hits[0][2], hits[0][0], hits[0][1]))
                    continue
                S.set_wavelength(lam)

                def set_line(max_order: int) -> None:
                    tab.set_slots(slots(s["dll"], names, period_um=P, max_order=max_order,
                                        depth_um=c.dll_depth_um, n_steps=N,
                                        layers_per_step=s["layers_per_step"],
                                        alpha_deg=s["alpha_deg"], index_grate_r=n,
                                        index_grate_i=0.0, index_env_r=n_env,
                                        index_env_i=0.0, interpolation=0, stochastic=0,
                                        only_orders=0, test_mode=int(s.get("test_mode", 0))))

                set_line(mo)
                if ic == 0 and il == 0:
                    log("  object Par 10 (Lines/um)          = %g   (period %.2f um)" % (1.0 / P, P))
                    for k, v in sorted(slots(s["dll"], names, period_um=P, max_order=mo,
                                             depth_um=c.dll_depth_um, n_steps=N, index_grate_r=n).items()):
                        log("  param [%2d] %-22s = %g" % (k, names[k] if k < len(names) else "?", v))
                phys = [p0 + k for k in range(-w, w + 1)]
                ref = t_fres * tea.staircase_orders(N, p, phys)
                geo = order_geometry(lam, P, phys, s["detector_z_mm"])
                j0 = w
                if self.sign is None:                      # measure the label sign on a line with power
                    got = self.measure_sign(tab, trace, det, p0 if p0 != 0 else 1)
                    if got is not None:
                        self.sign = got
                sign = self.sign if self.sign is not None else 1
                # the design order first: a 0 where the reference expects power is a refused
                # line (energy-balance check) -> retry the harmonic count
                dll_log.mark()
                tab.set_orders(sign * p0, sign * p0)
                trace.run()
                e0 = trace.detector_total(det) / t["source_power_w"]
                tried = [(mo, e0)]
                if e0 <= 0.0 and ref[j0] > 0.02:
                    for delta in retries:
                        mo_try = int(min(int(s["max_order_cap"]), max(5, mo + delta)))
                        if mo_try == mo or any(mo_try == m_ for m_, _ in tried):
                            continue
                        set_line(mo_try)
                        trace.run()
                        e_try = trace.detector_total(det) / t["source_power_w"]
                        tried.append((mo_try, e_try))
                        if e_try > 0.0:
                            mo, e0 = mo_try, e_try
                            break
                    log("      refused at Max Order %d; retries %s -> %s"
                        % (tried[0][0], ", ".join("%d: %.4f" % (m_, e_) for m_, e_ in tried[1:]),
                           ("Max Order %d kept" % mo) if e0 > 0.0 else "the line stays refused"))
                    if e0 <= 0.0:
                        refused[ic, il] = True
                mo_tab[ic, il] = mo
                row: List[float] = [float("nan")] * n_win
                row[j0] = e0
                if e0 > 0.0:
                    for j, m in enumerate(phys):
                        if j == j0:
                            continue
                        if abs(m * lam / P) >= 1.0:                # evanescent
                            continue
                        tab.set_orders(sign * m, sign * m)
                        trace.run()
                        row[j] = trace.detector_total(det) / t["source_power_w"]
                summary, _lines = dll_log.since()
                eta_r[ic, il], eta_t[ic, il] = row, ref
                fin = [k for k in range(n_win) if np.isfinite(row[k])]
                sum_r = float(np.nansum(row))
                sum_t = float(sum(ref[k] for k in fin))
                log("  lam %.3f  n %.4f  MO %2d%s  p %.3f -> p0 %+d (x_det %.2f mm): eta_RCWA(p0) %.4f "
                    "ref %.4f ratio %.3f | window sums %.4f / %.4f ratio %.3f | DLL log: %s"
                    % (lam, n, mo, "" if converged else "!", p, p0, geo[j0][2], row[j0], ref[j0],
                       row[j0] / max(ref[j0], 1e-9), sum_r, sum_t, sum_r / max(sum_t, 1e-9), summary))
                log("      window m %s" % "  ".join("%+d: %s/%.4f" % (m, ("%.4f" % row[k]) if np.isfinite(row[k]) else "  --  ", ref[k])
                                                   for k, m in enumerate(phys)))
                if not conv and e0 > 0.0 and s.get("convergence_max_order"):
                    mo2 = int(mo + int(s["convergence_max_order"])) if int(s["convergence_max_order"]) < 0 \
                        else int(s["convergence_max_order"])
                    mo2 = max(5, mo2)
                    slot_mo = slot_of(s["dll"], "max_order", names)
                    tab.set_slot(slot_mo, float(mo2))
                    tab.set_orders(sign * p0, sign * p0)
                    trace.run()
                    e2 = trace.detector_total(det) / t["source_power_w"]
                    conv = {"case": c.label(), "lam_um": lam, "order": p0,
                            "eta_max_order_%d" % mo: row[j0], "eta_max_order_%d" % mo2: e2}
                    log("  convergence at p0: Max Order %d -> %.4f, %d -> %.4f (diff %.4f)"
                        % (mo, row[j0], mo2, e2, row[j0] - e2))
                    tab.set_slot(slot_mo, float(mo))
        S.save(os.path.join(self.out_dir, "nsc_ladder.zos"))

        log.section("summary: eta_RCWA / eta_ref at the design order ('!' = harmonics short of the margin, "
                    "'x' = refused by the DLL at every Max Order tried)")
        log("  case          N'  slope   | " + "  ".join("%5.0f" % (1000 * l) for l in lams) + "  nm")
        ratio0 = eta_r[:, :, w] / np.maximum(eta_t[:, :, w], 1e-9)
        ref_sum = np.nansum(np.where(np.isfinite(eta_r), eta_t, 0.0), axis=2)
        ratio_sum = np.where(ref_sum > 0, np.nansum(eta_r, axis=2) / np.maximum(ref_sum, 1e-9), np.nan)
        for ic, c in enumerate(self.cases):
            log("  %-12s %3d  %.4f | " % (c.label(), c.n_treads, c.slope)
                + "  ".join(("%5.3f" % v if np.isfinite(v) else "  -- ")
                            + ("x" if refused[ic, il] else ("!" if not conv_tab[ic, il] else " "))
                            for il, v in enumerate(ratio0[ic])))
        log("  window-sum ratios:")
        for ic, c in enumerate(self.cases):
            log("  %-12s %3d  %.4f | " % (c.label(), c.n_treads, c.slope)
                + "  ".join(("%5.3f" % v if np.isfinite(v) else "  -- ") + " " for v in ratio_sum[ic]))
        log("order-label sign used: %+d" % (self.sign if self.sign is not None else 0))
        np.savez(os.path.join(self.out_dir, "ladder.npz"), lams_um=np.array(lams),
                 n_treads=np.array([c.n_treads for c in self.cases]),
                 slopes=np.array([c.slope for c in self.cases]),
                 periods_um=np.array([c.period_um for c in self.cases]),
                 depths_um=np.array([c.depth_um for c in self.cases]),
                 step_um=s["step_um"], p=p_tab, p0=p0_tab, max_order=mo_tab, converged=conv_tab, refused=refused,
                 eta_rcwa=eta_r, eta_ref=eta_t, ratio_p0=ratio0, ratio_sum=ratio_sum,
                 order_window=w, order_sign=self.sign if self.sign is not None else 0)
        with open(os.path.join(self.out_dir, "ladder.json"), "w") as fh:
            json.dump({"lams_um": lams, "cases": self.settings_record()["cases"],
                       "step_um": s["step_um"], "ratio_p0": ratio0.tolist(),
                       "ratio_sum": ratio_sum.tolist(), "max_order": mo_tab.tolist(),
                       "converged": conv_tab.tolist(), "refused": refused.tolist(), "order_sign": self.sign,
                       "convergence": conv}, fh, indent=1, default=float)
        self.figure(lams, ratio0, conv_tab)
        F = float(self.ctx.cfg.get("focal_um") or 0.0)
        if F > 0.0:
            log.section("efficiency correction table for the design")
            try:
                corr = correction_from_ladder(os.path.join(self.out_dir, "ladder.npz"), F, log=log)
                write_correction(self.out_dir, corr, log=log)
            except SystemExit as exc:
                log("  not written: %s" % exc)
        self.members_found = dict(S.members.found)

    def measure_sign(self, tab: DiffractionTab, trace: NscTrace, det: int, p0: int) -> Optional[int]:
        """Which srg label carries the design order: trace +p0 and -p0 once.
        None when neither returns power (a refused line: try the next)."""
        got: Dict[int, float] = {}
        for sg in (-1, +1):
            tab.set_orders(sg * p0, sg * p0)
            trace.run()
            got[sg] = trace.detector_total(det) / self.t["source_power_w"]
        if max(got.values()) <= 0.02:
            self.log("  order-label sign: label %+d -> %.4f, label %+d -> %.4f -- no power, undecided on this line"
                     % (-p0, got[-1], p0, got[+1]))
            return None
        sign = -1 if got[-1] > got[+1] else +1
        self.log("  order-label sign: label %+d -> %.4f, label %+d -> %.4f => the srg label is %+d x the "
                 "physical order" % (-p0, got[-1], p0, got[+1], sign))
        return sign

    def figure(self, lams: List[float], ratio0: np.ndarray, conv: np.ndarray) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                     # pragma: no cover
            self.log("figure skipped (%s)" % e)
            return
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        x = 1000 * np.array(lams)
        for ic, c in enumerate(self.cases):
            ln, = ax.plot(x, ratio0[ic], marker="o", ms=3,
                          label="N' %d, sin(theta) %.3f (P' %.0f um, riser %.2f um)"
                          % (c.n_treads, c.slope, c.period_um, c.riser_um))
            bad = ~conv[ic]
            if bad.any():
                ax.plot(x[bad], ratio0[ic][bad], marker="x", ms=7, ls="none", color=ln.get_color())
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("eta_RCWA / (T_fresnel eta_TEA) at the design order")
        ax.set_title("TEA validity ladder, DELTA = %.2f um treads (x = harmonics short of the margin)"
                     % self.s["step_um"], fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "fig_ladder.png"), dpi=130)
        plt.close(fig)
        self.log("saved fig_ladder.png")
