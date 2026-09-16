"""null -- the RCWA null test on a stock blazed grating.

System (NSC, mm): collimated Source Ellipse (1 W) at z = -10 ->
Diffraction Grating object at z = 0 (flat, Lines/um = 1/P, DLL
srg_blaze_RCWA on its face) -> Detector Rectangle at z = detector_z.
For each wavelength and each order m the tab's Start = Stop = m, one
trace, and eta_m = detector flux / source power. Reference:
eta_m = sinc^2(m - p), p = (n_grate - n_env) d / lam, d = lam0/(n-1)
(nscval.tea). At P = 50 um = 83 lam the scalar theory is accurate to
~1 % and the RCWA must reproduce it (tol_eta); at 0.75 um (p = 0.8)
expect eta_+1 ~ 0.87, eta_0 ~ 0.10.

Outputs: null_test.npz (eta[lam, m] measured + scalar), null_test.json
(table + verdict), fig_null_test.png, nsc_null_test.zos.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from . import tea
from .base import NscAnalysis
from .dlls import slots
from .nsc import DiffractionTab, NscSystem, NscTrace, order_geometry
from .settings import NULL_SETTINGS, TRACE_SETTINGS, blaze_alpha_deg, blaze_depth_um


class NullTest(NscAnalysis):
    MODE = "null"
    DESCRIPTION = "blazed grating + srg_blaze_RCWA vs sinc^2(m - p): the NSC plumbing check"

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.s: Dict[str, Any] = dict(NULL_SETTINGS)
        self.t: Dict[str, Any] = dict(TRACE_SETTINGS)
        for key, v in self.overrides.items():
            (self.s if key in self.s else self.t)[key] = v
        s = self.s
        s["depth_um"] = blaze_depth_um(s["lam0_um"], s["n_grate"], s["n_env"])
        s["alpha_deg"] = blaze_alpha_deg(s["period_um"], s["depth_um"])

    def settings_record(self) -> Dict[str, Any]:
        return {"null": self.s, "trace": self.t}

    def run(self) -> None:
        log, s, t = self.log, self.s, self.t
        S = NscSystem(self.session, log)
        log.section("build the NSC system", "source -> grating (DLL on face 0) -> detector")
        S.add_source_ellipse(-10.0, s["beam_half_mm"], s["beam_half_mm"],
                             t["analysis_rays"], t["layout_rays"], t["source_power_w"])
        gr = S.add_diffraction_grating(0.0, s["grating_clear_mm"], 1.0 / s["period_um"], 1)
        S.add_detector_rect(s["detector_z_mm"], s["detector_half_mm"], s["detector_half_mm"],
                            s["detector_pixels"])
        tab = DiffractionTab(S, gr)
        tab.use_dll(s["dll"], s["orders"][0], s["orders"][-1])
        vals = slots(s["dll"], period_um=s["period_um"], max_order=s["max_order"],
                     fill=s["fill"], alpha_deg=s["alpha_deg"], beta_deg=s["beta_deg"],
                     index_grate_r=s["n_grate"], index_grate_i=0.0,
                     index_env_r=s["n_env"], index_env_i=0.0, n_layer=1,
                     interpolation=0, stochastic=0, only_orders=0)
        tab.set_slots(vals)
        names = tab.names()
        log("DLL %s: blaze depth %.4f um -> alpha %.4f deg (beta %.0f, fill %.1f); "
            "Index Grate %.3f, Env %.3f; Max Order %d"
            % (s["dll"], s["depth_um"], s["alpha_deg"], s["beta_deg"], s["fill"],
               s["n_grate"], s["n_env"], s["max_order"]))
        for k, v in sorted(vals.items()):
            log("  slot %2d %-22s = %g" % (k, names[k - 1] if k <= len(names) else "?", v))
        S.save(os.path.join(self.out_dir, "nsc_null_test.zos"))

        log.section("trace one order at a time", "eta_m = detector flux / source power")
        trace = NscTrace(S, t["split_rays"], t["scatter_rays"], t["use_polarization"],
                         t["ignore_errors"])
        det = S.objects["detector"]
        lams: List[float] = [float(v) for v in s["lams_um"]]
        orders: List[int] = [int(m) for m in s["orders"]]
        eta = np.zeros((len(lams), len(orders)))
        ref = np.zeros_like(eta)
        for i, lam in enumerate(lams):
            S.set_wavelength(lam)
            p = tea.waves_of_depth(s["depth_um"], lam, s["n_grate"], s["n_env"])
            ref[i] = tea.blaze_orders(orders, p)
            geo = order_geometry(lam, s["period_um"], orders, s["detector_z_mm"])
            log.subsection("lam = %.3f um, p = %.3f waves" % (lam, p))
            log("  m   x_det(mm)   eta_RCWA   sinc^2(m-p)   diff")
            for j, m in enumerate(orders):
                tab.set_orders(m, m)
                trace.run()
                eta[i, j] = trace.detector_total(det) / t["source_power_w"]
                log("  %+d   %8.3f    %.4f     %.4f       %+.4f"
                    % (m, geo[j][2], eta[i, j], ref[i, j], eta[i, j] - ref[i, j]))
            log("  sum over traced orders: RCWA %.4f, scalar %.4f"
                % (eta[i].sum(), ref[i].sum()))
        tab.set_orders(orders[0], orders[-1])
        S.save(os.path.join(self.out_dir, "nsc_null_test.zos"))

        log.section("verdict")
        diff = np.abs(eta - ref)
        worst = float(diff.max())
        ok = worst <= float(s["tol_eta"])
        log("max |eta_RCWA - sinc^2| = %.4f (tolerance %.3f) -> %s"
            % (worst, s["tol_eta"], "PASS: the NSC diffraction plumbing reproduces the "
               "scalar blaze; proceed to the ladder" if ok else
               "FAIL: check the log lines above (split type, DLL name, slot labels, "
               "polarization, detector size) before the ladder"))
        np.savez(os.path.join(self.out_dir, "null_test.npz"), lams_um=np.array(lams),
                 orders=np.array(orders), eta_rcwa=eta, eta_scalar=ref,
                 period_um=s["period_um"], depth_um=s["depth_um"])
        with open(os.path.join(self.out_dir, "null_test.json"), "w") as fh:
            json.dump({"lams_um": lams, "orders": orders, "eta_rcwa": eta.tolist(),
                       "eta_scalar": ref.tolist(), "max_abs_diff": worst,
                       "pass": bool(ok)}, fh, indent=1)
        self.figure(lams, orders, eta, ref)
        self.members_found = dict(S.members.found)

    def figure(self, lams: List[float], orders: List[int], eta: np.ndarray,
               ref: np.ndarray) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                     # pragma: no cover
            self.log("figure skipped (%s)" % e)
            return
        fig, axes = plt.subplots(1, len(lams), figsize=(4.5 * len(lams), 3.8), squeeze=False)
        for i, lam in enumerate(lams):
            ax = axes[0, i]
            x = np.arange(len(orders))
            ax.bar(x - 0.2, ref[i], 0.4, label="scalar sinc$^2$(m-p)", color="#bbbbbb")
            ax.bar(x + 0.2, eta[i], 0.4, label="RCWA (NSC trace)", color="#e6550d")
            ax.set_xticks(x)
            ax.set_xticklabels(["%+d" % m for m in orders])
            ax.set_xlabel("order m")
            ax.set_ylabel("efficiency")
            ax.set_title("%.0f nm, P = %.0f um, blaze %.3f um" % (1000 * lam,
                         self.s["period_um"], self.s["depth_um"]), fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "fig_null_test.png"), dpi=130)
        plt.close(fig)
        self.log("saved fig_null_test.png")
