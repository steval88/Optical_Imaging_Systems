"""null -- the RCWA null test on a stock blazed grating.

System (NSC, mm): collimated Source Ellipse (1 W) at z = -10 ->
Diffraction Grating object at z = 0 (flat, Lines/um = 1/P, DLL
srg_blaze_RCWA on its face) -> Detector Rectangle at z = detector_z.
For each wavelength and each PHYSICAL order m the tab's Start = Stop =
order_sign * m (the srg DLLs label their orders mirrored: label -1 is
the ray at sin theta = +lam/P, diag 2026-09-18.02), one trace, and
eta_m = detector flux / source power -- exact, not statistical, since
every ray is identical. Reference (nscval.tea, settings):

    eta_m = T_fresnel * eta_staircase(N = n_layer, p, m),
    p = (n_grate - n_env) d / lam,  d = lam0 / (n_grate - n_env),
    T_fresnel = 4 n n_env / (n + n_env)^2

i.e. the scalar efficiency of the N-level staircase the DLL actually
computes ("# Layer"), times the flat-interface reflection loss. At
P = 5 um (8.3 lam) the rigorous numbers sit within ~6 % of this
(measured 0.819 vs 0.824 at the blaze order, 5 layers; tol_eta 0.08).
At 0.75 um (p = 0.8) expect eta_+1 ~ 0.75, eta_0 ~ 0.10.

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
from .settings import (NULL_SETTINGS, TRACE_SETTINGS, blaze_alpha_deg, blaze_depth_of_angles,
                       blaze_depth_um, check_cutoff, fresnel_transmission)


class NullTest(NscAnalysis):
    MODE = "null"
    DESCRIPTION = "blazed grating + srg_blaze_RCWA vs the scalar staircase x Fresnel: the NSC plumbing check"

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.s: Dict[str, Any] = dict(NULL_SETTINGS)
        self.t: Dict[str, Any] = dict(TRACE_SETTINGS)
        for key, v in self.overrides.items():
            (self.s if key in self.s else self.t)[key] = v
        s = self.s
        s["depth_um"] = blaze_depth_um(s["lam0_um"], s["n_grate"], s["n_env"])
        s["alpha_deg"] = blaze_alpha_deg(s["period_um"], s["depth_um"], s["fill"])
        # the efficiency of a collimated normal-incidence beam is the same for
        # every ray: NULL_SETTINGS["rays"] (20) replaces the statistical count
        self.t["analysis_rays"] = int(self.overrides.get("analysis_rays", s.get("rays", 20)))
        self.sign: int = int(s.get("order_sign", 1))
        self.t_fres: float = fresnel_transmission(float(s["n_grate"]), float(s["n_env"]))

    def settings_record(self) -> Dict[str, Any]:
        return {"null": self.s, "trace": self.t}

    def run(self) -> None:
        log, s, t = self.log, self.s, self.t
        log.section("cut-off check", "the srg DLL refuses every ray when n P / lam is an integer")
        for lam in s["lams_um"]:
            check_cutoff(float(lam), float(s["period_um"]), [float(s["n_env"]), float(s["n_grate"])], log)
        S = NscSystem(self.session, log)
        log.section("build the NSC system", "source -> grating (DLL on face 0) -> detector")
        S.add_source_ellipse(-10.0, s["beam_half_mm"], s["beam_half_mm"],
                             t["analysis_rays"], t["layout_rays"], t["source_power_w"])
        gr = S.add_diffraction_grating(0.0, s["grating_clear_mm"], 1.0 / s["period_um"], 1,
                                       thickness_mm=s["grating_thickness_mm"],
                                       material=s["grating_material"])
        S.add_detector_rect(s["detector_z_mm"], s["detector_half_mm"], s["detector_half_mm"],
                            s["detector_pixels"])
        tab = DiffractionTab(S, gr)
        tab.use_dll(s["dll"], min(self.sign * m for m in s["orders"]), max(self.sign * m for m in s["orders"]))
        names = tab.names()                       # live labels -> slot resolution
        vals = slots(s["dll"], names, period_um=s["period_um"], max_order=s["max_order"],
                     fill=s["fill"], alpha_deg=s["alpha_deg"], beta_deg=s["beta_deg"],
                     index_grate_r=s["n_grate"], index_grate_i=0.0,
                     index_env_r=s["n_env"], index_env_i=0.0, n_layer=int(s["n_layer"]),
                     interpolation=0, stochastic=0, only_orders=0)
        tab.set_slots(vals)
        log("  object Par 10 (Lines/um)          = %g   (period %.1f um: the object sets the "
            "order directions, the DLL's own period parameter the efficiencies)"
            % (1.0 / s["period_um"], s["period_um"]))
        log("DLL %s: blaze depth %.4f um -> alpha %.4f deg FROM THE VERTICAL (beta %.1f, fill %.2f); "
            "the DLL rebuilds depth = fill P / (tan a + tan b) = %.4f um; Index Grate %.3f, Env %.3f; Max Order %d"
            % (s["dll"], s["depth_um"], s["alpha_deg"], s["beta_deg"], s["fill"],
               blaze_depth_of_angles(s["period_um"], s["alpha_deg"], s["beta_deg"], s["fill"]),
               s["n_grate"], s["n_env"], s["max_order"]))
        for k, v in sorted(vals.items()):
            log("  param [%2d] %-22s = %g" % (k, names[k] if k < len(names) else "?", v))
        log("order sign %+d: the tab's Start = Stop = %+d * m for the physical order m; reference = "
            "scalar %d-level staircase x Fresnel %.4f; %d rays per trace"
            % (self.sign, self.sign, int(s["n_layer"]), self.t_fres, self.t["analysis_rays"]))
        S.save(os.path.join(self.out_dir, "nsc_null_test.zos"))

        trace = NscTrace(S, t["split_rays"], t["scatter_rays"], t["use_polarization"],
                         t["ignore_errors"])
        det = S.objects["detector"]
        lams: List[float] = [float(v) for v in s["lams_um"]]
        orders: List[int] = [int(m) for m in s["orders"]]          # physical orders
        labels: List[int] = [self.sign * m for m in orders]        # what the DLL is asked for

        if s.get("geo_precheck", True):
            log.section("geometric pre-check (no DLL)",
                        "Split = DontSplitByOrder: all power follows the object's Diffract "
                        "Order 1 -> the detector must read ~1.0 x source power")
            S.set_wavelength(lams[0])
            tab.set_split_none()
            trace.run()
            log("  " + trace.report())
            e_pix0 = trace.detector_total(det) / t["source_power_w"]
            e_sum = trace.detector_total_pixels(det)
            log("  detector: pixel-0 total %.4f, per-pixel sum %s (fraction of source power)"
                % (e_pix0, "%.4f" % (e_sum / t["source_power_w"]) if e_sum is not None else "n/a"))
            if e_pix0 < 0.9:
                raise SystemExit(
                    "geometric pre-check FAILED (%.4f of the power on the detector without any "
                    "DLL): the problem is geometry / readout, not diffraction. Check the "
                    "trace report above (error text, launched energy), the grating thickness "
                    "and material, and the detector position/size." % e_pix0)
            tab.use_dll(s["dll"], min(labels), max(labels))   # back to the DLL split
            tab.set_slots(vals)

        log.section("trace one order at a time", "eta_m = detector flux / source power")
        eta = np.zeros((len(lams), len(orders)))
        ref = np.zeros_like(eta)
        saw = np.zeros_like(eta)
        for i, lam in enumerate(lams):
            S.set_wavelength(lam)
            p = tea.waves_of_depth(s["depth_um"], lam, s["n_grate"], s["n_env"])
            ref[i] = self.t_fres * tea.staircase_orders(int(s["n_layer"]), p, orders)
            saw[i] = tea.blaze_orders(orders, p)
            geo = order_geometry(lam, s["period_um"], orders, s["detector_z_mm"])
            log.subsection("lam = %.3f um, p = %.3f waves" % (lam, p))
            log("  m  label  x_det(mm)   eta_RCWA   ref(stair x T)   diff    sinc^2(m-p)")
            for j, m in enumerate(orders):
                tab.set_orders(labels[j], labels[j])
                trace.run()
                if i == 0 and j == 0:
                    log("  " + trace.report())
                eta[i, j] = trace.detector_total(det) / t["source_power_w"]
                log("  %+d   %+d   %8.3f    %.4f     %.4f       %+.4f    %.4f"
                    % (m, labels[j], geo[j][2], eta[i, j], ref[i, j], eta[i, j] - ref[i, j], saw[i, j]))
            log("  sum over traced orders: RCWA %.4f, reference %.4f, sawtooth sinc^2 %.4f"
                % (eta[i].sum(), ref[i].sum(), saw[i].sum()))
        tab.set_orders(min(labels), max(labels))
        S.save(os.path.join(self.out_dir, "nsc_null_test.zos"))

        log.section("verdict")
        diff = np.abs(eta - ref)
        worst = float(diff.max())
        ok = worst <= float(s["tol_eta"])
        log("max |eta_RCWA - scalar staircase x Fresnel| = %.4f (tolerance %.3f) -> %s"
            % (worst, s["tol_eta"], "PASS: the NSC diffraction plumbing reproduces the "
               "scalar blaze; proceed to the ladder" if ok else
               "FAIL: check the log lines above (split type, DLL name, slot labels, "
               "polarization, detector size) before the ladder"))
        np.savez(os.path.join(self.out_dir, "null_test.npz"), lams_um=np.array(lams),
                 orders=np.array(orders), labels=np.array(labels), eta_rcwa=eta, eta_scalar=ref,
                 eta_sawtooth=saw, period_um=s["period_um"], depth_um=s["depth_um"],
                 n_layer=int(s["n_layer"]), t_fresnel=self.t_fres)
        with open(os.path.join(self.out_dir, "null_test.json"), "w") as fh:
            json.dump({"lams_um": lams, "orders": orders, "labels": labels, "eta_rcwa": eta.tolist(),
                       "eta_scalar": ref.tolist(), "eta_sawtooth": saw.tolist(), "max_abs_diff": worst,
                       "t_fresnel": self.t_fres, "n_layer": int(s["n_layer"]),
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
            ax.bar(x - 0.2, ref[i], 0.4, label="scalar %d-level staircase $\\times$ Fresnel" % int(self.s["n_layer"]),
                   color="#bbbbbb")
            ax.bar(x + 0.2, eta[i], 0.4, label="RCWA (NSC trace)", color="#e6550d")
            ax.set_xticks(x)
            ax.set_xticklabels(["%+d" % m for m in orders])
            ax.set_xlabel("order m")
            ax.set_ylabel("efficiency")
            ax.set_title("%.0f nm, P = %.0f um, blaze %.3f um, label = %+d m" % (1000 * lam,
                         self.s["period_um"], self.s["depth_um"], self.sign), fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "fig_null_test.png"), dpi=130)
        plt.close(fig)
        self.log("saved fig_null_test.png")
