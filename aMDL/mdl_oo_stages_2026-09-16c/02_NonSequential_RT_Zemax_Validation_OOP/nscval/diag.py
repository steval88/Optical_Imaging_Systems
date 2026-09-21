r"""NullDiag -- which ORDER carries the blaze power? (the sign of m in the DLL)

The convergence probe (nscval 2026-09-18.01, log of 2026-09-18 11:52) cleared
the DLL's energy-conservation check: at Max Order 40/50 (P 8 um) and at P 5 um
the DLL log shows no "Power conservation" error any more, the RCWA runs -- and
order +1 still reads ~0.001. The layer series is the clue:

    layers   detector at +1     scalar staircase of a one-wave blaze (tea.py)
                                 +1 blazed              -1 blazed (mirror)
      2        0.2997            +1 0.405, -1 0.405     the same
      3        0.0000            +1 0.684, -1 0        +1 0,     -1 0.684, +2 0.171
      5        0.0013            +1 0.875              +1 0,     -1 0.875, +4 0.055

A two-level staircase is symmetric and delivers ~0.3-0.4 to BOTH first
orders; three and more levels blaze to ONE side, and the DLL puts nothing
at +1. So the sawtooth of the srg convention (facet "descending toward +x")
diffracts into m = -1 as OpticStudio labels the orders. This version measures
it: order -1 (and +2 for 3 levels) one at a time, then the ZRD histogram of
every order -3..+3 for the three passing configurations, with the scalar
prediction of both hypotheses printed next to each number. Test Mode stays
on and the DLL log is read after every trace.

Outputs: diag.json, diag_<name>.zos (detector > 0.05), diag_<name>.ZRD, the log.
"""
from __future__ import annotations

import json
import math
import os
import time

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import NscAnalysis
from .dlls import slots
from .nsc import DiffractionTab, DllLog, NscSystem, NscTrace, order_histogram, zrd_raw
from .settings import NULL_SETTINGS, TRACE_SETTINGS, blaze_alpha_deg, blaze_depth_um
from .tea import staircase_orders


def _snapshot(root: str) -> Dict[str, float]:
    """{path: mtime} of every file under root (shallow depth 3)."""
    out: Dict[str, float] = {}
    if not os.path.isdir(root):
        return out
    base_depth = root.rstrip("\\/").count(os.sep)
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        if dp.count(os.sep) - base_depth >= 3:
            dn[:] = []
        for f in fn:
            if f.endswith((".pyc", ".db", ".py", ".json", ".zos", ".ZRD")):
                continue
            p = os.path.join(dp, f)
            try:
                out[p] = os.path.getmtime(p)
            except OSError:
                pass
    return out


def n_bk7(lam_um: float) -> float:
    """Sellmeier index of SCHOTT N-BK7 (catalogue coefficients), lam in um."""
    l2 = lam_um * lam_um
    n2 = 1.0 + (1.03961212 * l2 / (l2 - 0.00600069867) + 0.231792344 * l2 / (l2 - 0.0200179144)
                + 1.01046945 * l2 / (l2 - 103.560653))
    return n2 ** 0.5


def scalar_both_signs(n_layer: int, p_waves: float, orders: Sequence[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Scalar (TEA) efficiencies of the n_layer-level staircase of a blaze of
    p_waves, for `orders`: ({m: eta} if the blaze order is +round(p),
    {m: eta} if it is -round(p), i.e. the mirrored profile, eta(-m))."""
    eta = staircase_orders(n_layer, p_waves, list(orders))
    plus = {int(m): float(e) for m, e in zip(orders, eta)}
    eta_m = staircase_orders(n_layer, p_waves, [-int(m) for m in orders])
    minus = {int(m): float(e) for m, e in zip(orders, eta_m)}
    return plus, minus


class NullDiag(NscAnalysis):
    MODE = "diag"
    DESCRIPTION = "which order carries the blaze power: -1 / +2 traces and the full-order ZRD histogram"

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.s: Dict[str, Any] = dict(NULL_SETTINGS)
        self.t: Dict[str, Any] = dict(TRACE_SETTINGS)
        for key, v in self.overrides.items():
            (self.s if key in self.s else self.t)[key] = v
        s = self.s
        s["depth_um"] = blaze_depth_um(s["lam0_um"], s["n_grate"], s["n_env"])
        s["alpha_deg"] = blaze_alpha_deg(s["period_um"], s["depth_um"], s["fill"])
        # cost per ray PER REQUESTED ORDER (2026-09-18.01 log, 61 harmonics unless said):
        # 5 layers 0.34 s, 3 layers ~0.2 s, 2 layers ~0.14 s; Max Order 40: 1.65 s; 50: 3.3 s
        self.t["analysis_rays"] = int(self.overrides.get("analysis_rays", 50))
        self.zrd_rays: int = int(self.overrides.get("zrd_rays", 40))
        self.records: List[Dict[str, Any]] = []

    def settings_record(self) -> Dict[str, Any]:
        return {"null": self.s, "trace": self.t, "zrd_rays": self.zrd_rays}

    def snapshot_roots(self) -> List[str]:
        """Folders a diffraction DLL in Test Mode might write to: the user's
        Zemax data folder, the run folder, the process's working directory,
        ProgramData\\Zemax and whatever directories the application reports."""
        roots = [os.path.join(os.path.expanduser("~"), "Documents", "Zemax"), self.out_dir, os.getcwd(),
                 os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "Zemax")]
        app = getattr(self.session, "app", None)
        the_app = getattr(app, "TheApplication", None)
        for attr in ("ZemaxDataDir", "ProgramDirectory", "ProgramDir", "SamplesDir"):
            try:
                v = getattr(the_app, attr, None)
                if v:
                    roots.append(str(v))
            except Exception:
                pass
        out: List[str] = []
        for r in roots:
            r = os.path.normpath(r)
            if r not in out and os.path.isdir(r):
                out.append(r)
        return out

    # -- one system, rebuilt per variant (cheap) -------------------------------------
    def build(self, material: str = "", extra_detectors: bool = False,
              over: Optional[Dict[str, Any]] = None, flipped: bool = False
              ) -> Tuple[NscSystem, DiffractionTab, Any, int]:
        """The null-test system.

        material   Diffraction Grating object material ('' = air).
        over       null-setting overrides for THIS build only (period_um,
                   detector_z_mm, ...); the object's Lines/um follows.
        flipped    the object turned by 180 deg about Y and shifted by its
                   thickness, so the rays meet the flat face first and the
                   grating face on the way OUT (grating inside the material,
                   diffracting into air) -- the srg DLLs may be written for
                   that side only."""
        s = dict(self.s, **(over or {}))
        t = self.t
        S = NscSystem(self.session, self.log)
        S.add_source_ellipse(-10.0, s["beam_half_mm"], s["beam_half_mm"],
                             t["analysis_rays"], t["layout_rays"], t["source_power_w"])
        gr = S.add_diffraction_grating(0.0, s["grating_clear_mm"], 1.0 / s["period_um"], 1,
                                       thickness_mm=s["grating_thickness_mm"], material=material)
        S.add_detector_rect(s["detector_z_mm"], s["detector_half_mm"], s["detector_half_mm"],
                            s["detector_pixels"])
        if extra_detectors:
            S.add_detector_rect(s["grating_thickness_mm"] + 0.5, 6.0, 6.0, 50, label="near")
            S.add_detector_rect(-5.0, 6.0, 6.0, 50, label="back")
        S.set_wavelength(float(s["lams_um"][0]))
        if flipped:
            try:
                gr.TiltAboutY = 180.0
                gr.ZPosition = float(s["grating_thickness_mm"])
                self.log("  grating object flipped: TiltAboutY 180, Z %.2f mm (grating face at the exit)"
                         % s["grating_thickness_mm"])
            except Exception as exc:
                self.log("  flip failed: %s: %s" % (type(exc).__name__, exc))
        tab = DiffractionTab(S, gr)
        return S, tab, gr, S.objects["detector"]

    def blaze_slots(self, tab: DiffractionTab, **over: float) -> Dict[int, float]:
        s = self.s
        kw: Dict[str, float] = dict(period_um=s["period_um"], max_order=s["max_order"], fill=s["fill"],
                                    alpha_deg=s["alpha_deg"], beta_deg=s["beta_deg"],
                                    index_grate_r=s["n_grate"], index_grate_i=0.0,
                                    index_env_r=s["n_env"], index_env_i=0.0, n_layer=int(s.get("n_layer", 1)),
                                    interpolation=0, stochastic=0, only_orders=0)
        kw.update(over)
        return slots(s["dll"], tab.names(), **kw)

    def trace_and_read(self, S: NscSystem, det: int, split: bool = True,
                       polarization: bool = True, ignore_errors: bool = True) -> Dict[str, Any]:
        t = self.t
        tr = NscTrace(S, split, t["scatter_rays"], polarization, ignore_errors)
        rep: Dict[str, Any]
        t0 = time.time()
        try:
            rep = dict(tr.run())
        except Exception as exc:                       # a refused trace is a result too
            rep = {"exception": "%s: %s" % (type(exc).__name__, exc)}
        # an RCWA that really runs costs milliseconds per ray; microseconds = early return
        rep["us_per_ray"] = 1e6 * (time.time() - t0) / max(1, int(t["analysis_rays"]))
        try:
            rep["detector_fraction"] = tr.detector_total(det) / t["source_power_w"]
        except Exception as exc:
            rep["detector_fraction"] = float("nan")
            rep["detector_error"] = str(exc)
        rep["detector_fraction_pixels"] = tr.detector_total_pixels(det)
        return rep

    def record(self, name: str, note: str, rep: Dict[str, Any]) -> None:
        rep = dict(rep, variant=name, note=note)
        self.records.append(rep)
        self.log("  %-16s detector %.4f | %6.1f us/ray | succeeded %s | error %r%s | %s"
                 % (name, rep.get("detector_fraction", float("nan")), rep.get("us_per_ray", float("nan")),
                    rep.get("succeeded", "?"), rep.get("error", "") or "",
                    (" | EXC %s" % rep["exception"]) if "exception" in rep else "", note))

    # -- the variants --------------------------------------------------------------------
    def zrd_orders(self, S: NscSystem, det: int, period_um: float, name: str, rays: int = 300,
                   note: str = "") -> Dict[int, float]:
        """One trace of `rays` rays saved to a ZRD, read back, and the power per
        diffraction order on the detector (m from the child's direction
        cosine): the DLL's efficiencies read directly, whatever Start/Stop
        and whatever the detector geometry. Returns {m: power}."""
        zrd_name = "diag_%s.ZRD" % name
        S.par(S.NCE.GetObjectAt(S.objects["source"]), 2, rays, integer=True)
        tr = NscTrace(S, True, self.t["scatter_rays"], True, True, save_rays_file=zrd_name)
        t0 = time.time()
        rep = dict(tr.run())
        rep["us_per_ray"] = 1e6 * (time.time() - t0) / rays
        rep["detector_fraction"] = tr.detector_total(det) / self.t["source_power_w"]
        S.par(S.NCE.GetObjectAt(S.objects["source"]), 2, self.t["analysis_rays"], integer=True)
        self.record(name, note or "ZRD of %d rays" % rays, rep)
        cands = [os.path.join(self.out_dir, zrd_name),
                 os.path.join(os.path.expanduser("~"), "Documents", "Zemax", "Samples", zrd_name)]
        zrd = next((c for c in cands if os.path.exists(c)), None)
        if zrd is None:
            for root in (self.out_dir, os.path.join(os.path.expanduser("~"), "Documents", "Zemax")):
                for dp, dn, fn in os.walk(root):
                    if zrd_name.lower() in (f.lower() for f in fn):
                        zrd = os.path.join(dp, zrd_name)
                        break
                if zrd:
                    break
        if zrd is None:
            self.log("    ZRD %s not found" % zrd_name)
            return {}
        raw = zrd_raw(self.session.TheSystem, zrd, self.session.ZOSAPI, max_rays=rays)
        hist, launched, n_par, n_child = order_histogram(raw, period_um, det)
        tot = sum(hist.values())
        self.log("    ZRD %s: %d rays read, launched %.3f W, %d parents, %d child segments (%.2f per parent)"
                 % (os.path.basename(zrd), len(raw), launched, n_par, n_child, n_child / max(1, n_par)))
        self.log("    power per order on the detector (m = round(l P / lam)): %s | total %.4f"
                 % (", ".join("m=%+d: %.4f" % (k, hist[k]) for k in sorted(hist)) or "none", tot))
        for ray_no, lam, segs in raw[:1]:
            for sg in segs[:8]:
                if len(sg) >= 20:
                    self.log("      ray %d seg level %d obj %d face %d status %s | z %.3f l %+.5f | I %.4g"
                             % (ray_no, int(sg[0]), int(sg[2]), int(sg[3]), sg[5], float(sg[8]),
                                float(sg[9]), float(sg[18])))
        self.records[-1]["orders"] = {str(k): float(v) for k, v in hist.items()}
        return {int(k): float(v) for k, v in hist.items()}

    @staticmethod
    def dll_log_path() -> str:
        return DllLog("srg_blaze_RCWA.dll").path

    def dll_log_new(self, start: int) -> Tuple[str, List[str]]:
        """Lines the DLL appended to its log since byte offset `start`, and a
        summary: the first 'Power conservation' error with its R and T."""
        path = self.dll_log_path()
        try:
            with open(path, errors="replace") as fh:
                fh.seek(start)
                new = fh.read()
        except OSError:
            return "log not readable", []
        lines = [ln.rstrip() for ln in new.splitlines() if ln.strip()]
        err = next((ln for ln in lines if "Power conservation" in ln), "")
        r_p = next((ln for ln in lines if ln.strip().startswith("Reflect power")), "")
        t_p = next((ln for ln in lines if ln.strip().startswith("Transmit power")), "")
        other = [ln for ln in lines if "Error" in ln and "Power conservation" not in ln]
        summary = ("%s | %s | %s" % (err.split("]")[-1].strip(), r_p.strip(), t_p.strip())) if err else \
                  ("no conservation error logged; %d new lines%s" % (len(lines),
                   ("; other: " + other[0]) if other else ""))
        return summary, lines

    def dll_log_start(self) -> int:
        try:
            return os.path.getsize(self.dll_log_path())
        except OSError:
            return 0

    def prepare(self, over_b: Dict[str, Any], over_d: Dict[str, float], start: int, stop: int
                ) -> Tuple[NscSystem, DiffractionTab, int, float]:
        """Build + DLL + slots for one variant; returns (system, tab, detector, P)."""
        s = self.s
        Pb = float(over_b.get("period_um", s["period_um"]))
        d = float(s["depth_um"])
        over_b = dict(over_b, alpha_deg=math.degrees(math.atan2(s["fill"] * Pb, d)))
        Sx, tabx, grx, detx = self.build(over=over_b)
        tabx.use_dll(s["dll"], start, stop)
        kw: Dict[str, float] = dict(period_um=Pb, alpha_deg=over_b["alpha_deg"], beta_deg=0.0,
                                    index_coat_r=0.0, test_mode=1)
        kw.update(over_d)
        tabx.set_slots(self.blaze_slots(tabx, **kw))
        return Sx, tabx, detx, Pb

    def run(self) -> None:
        log, s = self.log, self.s
        lam = float(s["lams_um"][0])
        d = float(s["depth_um"])
        p_waves = (float(s["n_grate"]) - float(s["n_env"])) * d / lam
        n_rays = int(self.t["analysis_rays"])
        log.section("single orders", "%d rays per trace, Test Mode on; one requested order per trace; "
                    "next to each number the scalar staircase efficiency if the blaze order is +1 / -1"
                    % n_rays)
        log("  DLL log: %s" % self.dll_log_path())
        log("  blaze depth %.4f um = %.3f waves at %.3f um (n %.3f)" % (d, p_waves, lam, s["n_grate"]))
        for nl in (2, 3, 5, 10):
            plus, minus = scalar_both_signs(nl, p_waves, range(-4, 5))
            log("  scalar %2d levels: blaze at +1 -> %s" % (nl, " ".join("%+d:%.3f" % (m, plus[m]) for m in sorted(plus) if plus[m] > 5e-3)))
            log("                    blaze at -1 -> %s" % " ".join("%+d:%.3f" % (m, minus[m]) for m in sorted(minus) if minus[m] > 5e-3))

        p8: Dict[str, Any] = {}
        p5: Dict[str, Any] = dict(period_um=5.0, detector_z_mm=30.0)     # |m| <= 3 within 12.5 mm at z 30 (m 3 -> 11.6 mm)
        singles: List[Tuple[str, str, Dict[str, Any], Dict[str, float], int]] = [
            # name, note, build overrides, DLL slot overrides, order
            ("mo30_L2_m1", "P 8, Max Order 30, 2 layers, order -1 (symmetric: expect ~0.30 like +1)",
             p8, dict(max_order=30, n_layer=2), -1),
            ("mo30_L3_m1", "P 8, Max Order 30, 3 layers, order -1", p8, dict(max_order=30, n_layer=3), -1),
            ("mo30_L3_p2", "P 8, Max Order 30, 3 layers, order +2 (0.17 if the blaze is at -1)",
             p8, dict(max_order=30, n_layer=3), +2),
            ("mo40_L5_m1", "P 8, Max Order 40, 5 layers, order -1 (~1.7 s/ray)", p8, dict(max_order=40, n_layer=5), -1),
            ("p5_mo30_L5_m1", "P 5 um, Max Order 30, 5 layers, order -1", p5, dict(max_order=30, n_layer=5), -1),
            ("p5_mo30_L10_m1", "P 5 um, Max Order 30, 10 layers, order -1", p5, dict(max_order=30, n_layer=10), -1),
        ]
        first_lines_shown = False
        for name, note, over_b, over_d, m in singles:
            Sx, tabx, detx, Pb = self.prepare(over_b, over_d, m, m)
            start = self.dll_log_start()
            rep = self.trace_and_read(Sx, detx)
            summary, lines = self.dll_log_new(start)
            rep["dll_log"] = summary
            rep["order"] = m
            nl = int(over_d.get("n_layer", 1))
            plus, minus = scalar_both_signs(nl, p_waves, [m])
            rep["scalar_plus"], rep["scalar_minus"] = plus[m], minus[m]
            self.record(name, note, rep)
            log("      scalar %d levels at m=%+d: %.3f (blaze +1) / %.3f (blaze -1) | DLL log: %s"
                % (nl, m, plus[m], minus[m], summary))
            if not first_lines_shown and lines:
                first_lines_shown = True
                for ln in lines[:40]:
                    log("      | " + ln)
            if float(rep.get("detector_fraction") or 0.0) > 0.05:
                Sx.save(os.path.join(self.out_dir, "diag_%s.zos" % name))

        log.section("all orders", "Start -3 / Stop +3 (7 RCWA calls per ray), %d rays, the ZRD read back: "
                    "power per order from the child's direction cosine" % self.zrd_rays)
        fulls: List[Tuple[str, str, Dict[str, Any], Dict[str, float], int]] = [
            ("mo30_L3_all", "P 8, Max Order 30, 3 layers (~0.2 s per ray per order)", p8, dict(max_order=30, n_layer=3), self.zrd_rays),
            ("p5_mo30_L5_all", "P 5 um, Max Order 30, 5 layers (~0.35 s per ray per order)", p5, dict(max_order=30, n_layer=5), self.zrd_rays),
            ("mo40_L5_all", "P 8, Max Order 40, 5 layers (~1.7 s per ray per order)", p8, dict(max_order=40, n_layer=5),
             max(10, int(0.75 * self.zrd_rays))),
        ]
        orders = list(range(-3, 4))
        for name, note, over_b, over_d, rays in fulls:
            Sx, tabx, detx, Pb = self.prepare(over_b, over_d, -3, 3)
            start = self.dll_log_start()
            hist = self.zrd_orders(Sx, detx, Pb, name, rays=rays, note=note)
            summary, lines = self.dll_log_new(start)
            log("      DLL log: %s" % summary)
            nl = int(over_d.get("n_layer", 1))
            plus, minus = scalar_both_signs(nl, p_waves, orders)
            log("      scalar %d levels, blaze +1: %s" % (nl, " ".join("m=%+d: %.3f" % (m, plus[m]) for m in orders)))
            log("      scalar %d levels, blaze -1: %s" % (nl, " ".join("m=%+d: %.3f" % (m, minus[m]) for m in orders)))
            if hist:
                ss_plus = sum((hist.get(m, 0.0) - plus[m]) ** 2 for m in orders)
                ss_minus = sum((hist.get(m, 0.0) - minus[m]) ** 2 for m in orders)
                log("      sum of squares vs blaze +1: %.4f | vs blaze -1: %.4f -> the DLL blazes into m = %s"
                    % (ss_plus, ss_minus, "-1" if ss_minus < ss_plus else "+1"))
                self.records[-1]["ss_plus"], self.records[-1]["ss_minus"] = ss_plus, ss_minus
                Sx.save(os.path.join(self.out_dir, "diag_%s.zos" % name))

        log.section("reading")
        got = {r["variant"]: float(r.get("detector_fraction") or 0.0) for r in self.records}
        log("order -1: L2 %.4f, L3 %.4f, MO40 L5 %.4f, P5 L5 %.4f, P5 L10 %.4f | L3 at +2: %.4f"
            % (got.get("mo30_L2_m1", float("nan")), got.get("mo30_L3_m1", float("nan")),
               got.get("mo40_L5_m1", float("nan")), got.get("p5_mo30_L5_m1", float("nan")),
               got.get("p5_mo30_L10_m1", float("nan")), got.get("mo30_L3_p2", float("nan"))))
        verdict = [str(r["variant"]) + ": " + ("-1" if r["ss_minus"] < r["ss_plus"] else "+1")
                   for r in self.records if "ss_minus" in r]
        log("blaze order by the full-order histograms: %s" % (", ".join(verdict) or "no ZRD read"))
        log("if -1: set LADDER_SETTINGS['order_sign'] = -1 and NULL_SETTINGS['orders'] as they are "
            "(the null test compares eta(m) with sinc^2(-m - p))")
        with open(os.path.join(self.out_dir, "diag.json"), "w") as fh:
            json.dump(self.records, fh, indent=1, default=str)
        log("paste this whole log back")
