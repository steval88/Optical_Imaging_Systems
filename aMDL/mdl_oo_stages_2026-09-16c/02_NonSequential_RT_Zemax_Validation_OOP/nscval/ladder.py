"""ladder -- where does the thin-element model break for DELTA-wide
treads at fold depth?

For every case (period P, depth d = depth_frac * H_fold) of
LADDER_SETTINGS and every design wavelength: an equal-step staircase of
N = P / DELTA steps is put on the Diffraction Grating object through
srg_step_RCWA (Depth, Number of Steps, Index Grate = n(lam) of the
design's resist model, Index Env 1), the orders m = p0 - w .. p0 + w
around the design order p0 = round((n - 1) d / lam) are traced one at a
time, and eta_RCWA(m) is compared with eta_TEA(m) of the same staircase
(nscval.tea.staircase_orders). The output tables give, per case and
line, the ratio at p0 and the ratio of the window sums -- the numbers
the design's efficiency_corr_npz mechanism consumes once the real zone
profiles follow (srg_user_defined_RCWA, next rung).

Outputs: ladder.npz, ladder.json, fig_ladder.png, nsc_ladder.zos.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from . import tea
from .base import PKG_ROOT, NscAnalysis
from .dlls import slot_of, slots
from .nsc import DiffractionTab, NscSystem, NscTrace, order_geometry
from .settings import LADDER_SETTINGS, TRACE_SETTINGS


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


class TeaLadder(NscAnalysis):
    MODE = "ladder"
    DESCRIPTION = ("equal-step staircases at fold depth, srg_step_RCWA vs the scalar "
                   "staircase efficiency: the TEA-validity map")

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.s: Dict[str, Any] = dict(LADDER_SETTINGS)
        self.t: Dict[str, Any] = dict(TRACE_SETTINGS)
        for key, v in self.overrides.items():
            (self.s if key in self.s else self.t)[key] = v
        s, ctx = self.s, self.ctx
        if s["depth_um"] == "fold":
            hf = ctx.fold_um()
            if hf is None:
                raise SystemExit("the run has no ladder/echelle seed record (h_fold_um); "
                                 "pass --depth <um>")
            s["depth_um"] = hf
        if s["step_um"] == "ring":
            s["step_um"] = float(ctx.cfg.get("ring_width_um", 2.0))
        if s["lams_um"] == "design":
            s["lams_um"] = ctx.design_lams()
        self.n_of = design_index_model() if s["n_grate"] == "design" \
            else (lambda lam, v=float(s["n_grate"]): v)

    def settings_record(self) -> Dict[str, Any]:
        return {"ladder": self.s, "trace": self.t}

    def run(self) -> None:
        log, s, t = self.log, self.s, self.t
        lams: List[float] = [float(v) for v in s["lams_um"]]
        cases: List[Tuple[float, float]] = [(float(P), float(f)) for P, f in s["cases"]]
        w, sign = int(s["order_window"]), int(s["order_sign"])
        S = NscSystem(self.session, log)
        log.section("build the NSC system", "source -> staircase grating -> detector")
        S.add_source_ellipse(-10.0, s["beam_half_mm"], s["beam_half_mm"],
                             t["analysis_rays"], t["layout_rays"], t["source_power_w"])
        gr = S.add_diffraction_grating(0.0, s["grating_clear_mm"], 1.0 / cases[0][0], 1)
        S.add_detector_rect(s["detector_z_mm"], s["detector_half_mm"], s["detector_half_mm"],
                            s["detector_pixels"])
        tab = DiffractionTab(S, gr)
        tab.use_dll(s["dll"], 0, 0)
        names = tab.names()
        trace = NscTrace(S, t["split_rays"], t["scatter_rays"], t["use_polarization"],
                         t["ignore_errors"])
        det = S.objects["detector"]
        log("fold depth H_f = %.4f um, tread DELTA = %.3f um, %d cases x %d lines, "
            "window +/-%d orders, order sign %+d"
            % (s["depth_um"], s["step_um"], len(cases), len(lams), w, sign))

        n_case, n_lam, n_win = len(cases), len(lams), 2 * w + 1
        eta_r = np.full((n_case, n_lam, n_win), np.nan)
        eta_t = np.full_like(eta_r, np.nan)
        p_tab = np.zeros((n_case, n_lam))
        p0_tab = np.zeros((n_case, n_lam), dtype=int)
        conv: Dict[str, Any] = {}
        for ic, (P, frac) in enumerate(cases):
            d = frac * float(s["depth_um"])
            N = int(round(P / float(s["step_um"])))
            S.par(gr, 10, 1.0 / P)
            log.section("case %d/%d: P = %.1f um, depth %.3f um (%.2f H_f), N = %d steps"
                        % (ic + 1, n_case, P, d, frac, N))
            for il, lam in enumerate(lams):
                n = self.n_of(lam)
                p = tea.waves_of_depth(d, lam, n, s["n_env"])
                p0 = tea.design_order(p)
                orders = [sign * (p0 + k) for k in range(-w, w + 1)]
                p_tab[ic, il], p0_tab[ic, il] = p, p0
                if p0 * lam >= P:
                    log("  lam %.3f: design order %d is evanescent at P = %.0f um -- skipped"
                        % (lam, p0, P))
                    continue
                S.set_wavelength(lam)
                tab.set_slots(slots(s["dll"], names, max_order=s["max_order"],
                                    depth_um=d, n_steps=N,
                                    layers_per_step=s["layers_per_step"],
                                    alpha_deg=s["alpha_deg"], index_grate_r=n,
                                    index_grate_i=0.0, index_env_r=s["n_env"],
                                    index_env_i=0.0, interpolation=0, stochastic=0,
                                    only_orders=0))
                if ic == 0 and il == 0:
                    log("  object Par 10 (Lines/um)          = %g   (period %.1f um)" % (1.0 / P, P))
                    for k, v in sorted(slots(s["dll"], names, max_order=s["max_order"], depth_um=d,
                                             n_steps=N, index_grate_r=n).items()):
                        log("  slot %2d %-22s = %g" % (k, names[k - 1] if k <= len(names)
                                                        else "?", v))
                ref = tea.staircase_orders(N, p, [abs(m) for m in orders])
                geo = order_geometry(lam, P, orders, s["detector_z_mm"])
                row = []
                for j, m in enumerate(orders):
                    tab.set_orders(m, m)
                    trace.run()
                    row.append(trace.detector_total(det) / t["source_power_w"])
                eta_r[ic, il], eta_t[ic, il] = row, ref
                j0 = w
                log("  lam %.3f  n %.4f  p %.2f -> p0 %d (x_det %.2f mm): eta_RCWA(p0) %.4f "
                    "TEA %.4f ratio %.3f | window sums %.4f / %.4f ratio %.3f"
                    % (lam, n, p, p0, geo[j0][2], row[j0], ref[j0],
                       row[j0] / max(ref[j0], 1e-9), sum(row), ref.sum(),
                       sum(row) / max(ref.sum(), 1e-9)))
                if ic == 0 and il == 0 and s.get("convergence_max_order"):
                    slot_mo = slot_of(s["dll"], "max_order", names)
                    tab.set_slot(slot_mo, float(s["convergence_max_order"]))
                    tab.set_orders(orders[j0], orders[j0])
                    trace.run()
                    e2 = trace.detector_total(det) / t["source_power_w"]
                    conv = {"case": [P, d], "lam_um": lam, "order": orders[j0],
                            "eta_max_order_%d" % s["max_order"]: row[j0],
                            "eta_max_order_%d" % s["convergence_max_order"]: e2}
                    log("  convergence at p0: Max Order %d -> %.4f, %d -> %.4f (diff %.4f)"
                        % (s["max_order"], row[j0], s["convergence_max_order"], e2,
                           row[j0] - e2))
                    tab.set_slot(slot_mo, float(s["max_order"]))
        S.save(os.path.join(self.out_dir, "nsc_ladder.zos"))

        log.section("summary: eta_RCWA / eta_TEA at the design order")
        log("  P(um) depth(um) N  | " + "  ".join("%5.0f" % (1000 * l) for l in lams) + "  nm")
        ratio0 = eta_r[:, :, w] / np.maximum(eta_t[:, :, w], 1e-9)
        for ic, (P, frac) in enumerate(cases):
            log("  %5.0f %8.3f %3d | " % (P, frac * s["depth_um"], round(P / s["step_um"]))
                + "  ".join("%5.3f" % v if np.isfinite(v) else "  -- " for v in ratio0[ic]))
        np.savez(os.path.join(self.out_dir, "ladder.npz"), lams_um=np.array(lams),
                 cases=np.array(cases), depth_um=s["depth_um"], step_um=s["step_um"],
                 p=p_tab, p0=p0_tab, eta_rcwa=eta_r, eta_tea=eta_t, ratio_p0=ratio0,
                 order_window=w)
        with open(os.path.join(self.out_dir, "ladder.json"), "w") as fh:
            json.dump({"lams_um": lams, "cases": cases, "depth_um": s["depth_um"],
                       "ratio_p0": ratio0.tolist(), "convergence": conv}, fh, indent=1)
        self.figure(lams, cases, ratio0)
        self.members_found = dict(S.members.found)

    def figure(self, lams: List[float], cases: List[Tuple[float, float]],
               ratio0: np.ndarray) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                     # pragma: no cover
            self.log("figure skipped (%s)" % e)
            return
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for ic, (P, frac) in enumerate(cases):
            ax.plot(1000 * np.array(lams), ratio0[ic], marker="o", ms=3,
                    label="P %.0f um, depth %.1f H_f (N=%d)" % (P, frac, round(P / self.s["step_um"])))
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("eta_RCWA / eta_TEA at the design order")
        ax.set_title("TEA validity ladder, DELTA = %.1f um treads" % self.s["step_um"], fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "fig_ladder.png"), dpi=130)
        plt.close(fig)
        self.log("saved fig_ladder.png")
