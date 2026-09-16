"""huy -- Huygens PSF / MTF on the HYBRID system.

MEASURED 2026-09-07 (DLL debug log): POP evaluates a User Defined
Surface with ~70 rays per wavelength and interpolates -- it cannot carry
the fold-scale residual. The Huygens PSF sums one wavelet PER PUPIL RAY
with that ray's traced OPD, so with the Paraxial lens bending the rays
and the residual UDS cell-averaged on the pupil pitch it is the ray
engine that hosts the MDL. CLOSED 2026-09-08: FWHM Huygens / sub-ring
RS 1.005, 1.006, 1.003, 0.997, corr 1.0000 at 400-1100 nm; Zemax's own
Huygens MTF equals the Hankel transform of its PSF to <= 0.04 and the
sinc-quadrature RS MTF to <= 0.006.

Per line: Huygens PSF (DataGrid -> radial profile about the up-sampled
peak, or about the axis for an annular line), Huygens MTF (DataSeries),
comparison against (a) the run_verify focal slice and (b) the in-script
sub-ring RS reference, MTF numbers on the RS frequency grid, figure,
npz. The DLL debug log (Par 8) is armed for the first line and
summarised afterwards: what the engine passed to the surface.
"""
import os
import time
from collections import Counter

import numpy as np

from .base import Analysis
from .design import dll_surfaces_dir
from .references import (FineReference, RsReference, airy_profile, corr_on,
                         encircled_fraction, fwhm_of, hankel_mtf,
                         radial_profile)
from .report import fig_huygens_line
from .settings import HUY_SETTINGS, HYBRID_SETTINGS
from .zos import samp_index


class HuygensAnalysis(Analysis):
    MODE = "huy"
    VARIANT = "hybrid"
    ZOS_NAME = "mdl_validation_huygens_hybrid.zos"
    ENGINE = "Huygens"

    def __init__(self, design, lams=None, pupil_samp=None, image_samp=None,
                 image_delta_um=None, use_polarization=None, **kw):
        super().__init__(design, **kw)
        self.huy = dict(HUY_SETTINGS)
        self.hy = dict(HYBRID_SETTINGS)
        for k, v in (("lams", lams), ("pupil_samp", pupil_samp),
                     ("image_samp", image_samp),
                     ("image_delta_um", image_delta_um),
                     ("use_polarization", use_polarization)):
            if v is not None:
                self.huy[k] = v
        n = int(self.huy["pupil_samp"])
        if n not in (256, 512, 1024, 2048, 4096, 8192):
            raise SystemExit("pupil sampling must be a power of 2 between "
                             "256 and 8192, got %d" % n)

    # --- configuration -------------------------------------------------------
    def select_lines(self):
        self.d.select_lines(self.huy["lams"])

    def hybrid_settings(self):
        return self.hy

    def settings_record(self):
        return {"HUY_SETTINGS": dict(self.huy), "HYBRID_SETTINGS": dict(self.hy)}

    def describe(self):
        h, d = self.huy, self.d
        self.log("  HUYGENS PSF/MTF on the hybrid: Paraxial f=F + residual UDS "
                 "(Sub ideal=1, Avg cell = pupil pitch), pupil %d^2 -> %.2f um "
                 "rays, image %d^2 at %.2f um; %d wavelength(s). One wavelet "
                 "per traced ray with its OPD -- the ray engine that can host "
                 "the fold-scale residual (POP samples a UDS with ~70 rays: "
                 "DLL log 2026-09-07)"
                 % (h["pupil_samp"], 1000.0 * d.epd_mm / h["pupil_samp"],
                    h["image_samp"], h["image_delta_um"],
                    len(d.wavelengths_um)))

    # --- Huygens runs ----------------------------------------------------------
    def configure(self, an, wi, kind):
        """Typed settings first (echoed), MODIFYSETTINGS codes as a
        fallback. kind = 'psf' | 'mtf'."""
        io, h = self.io, self.huy
        n_p, n_i = int(h["pupil_samp"]), int(h["image_samp"])
        applied = []
        st = io.typed_settings(an, "PupilSampleSize")
        for attr, vals in (("PupilSampleSize", io.enum_candidates(n_p)),
                           ("ImageSampleSize", io.enum_candidates(n_i))):
            nm, v = io.try_set(st, (attr,), vals, verify=False)
            applied.append("%s=%s" % (attr, v if nm else "NOT SET"))
        nm, _ = io.try_set(st, ("ImageDelta",), (float(h["image_delta_um"]),),
                           verify=False)
        applied.append("ImageDelta=%s" % (h["image_delta_um"] if nm else "NOT SET"))
        nm, _ = io.try_set(st, ("UsePolarization",), (bool(h["use_polarization"]),))
        applied.append("UsePolarization=%s"
                       % (h["use_polarization"] if nm else "NOT SET"))
        if kind == "mtf":
            nm, _ = io.try_set(st, ("MaximumFrequency",),
                               (float(h["max_freq_lpmm"]),), verify=False)
            applied.append("MaximumFrequency=%s"
                           % (h["max_freq_lpmm"] if nm else "NOT SET"))
        try:
            if hasattr(st, "Wavelength"):
                st.Wavelength.SetWavelengthNumber(wi)
                applied.append("wave=%d" % wi)
            if hasattr(st, "Field"):
                st.Field.SetFieldNumber(1)
        except Exception as exc:
            applied.append("wave/field(%s)" % exc)
        if any("NOT SET" in a for a in applied):
            pre = "HPS_" if kind == "psf" else "HMF_"
            pairs = [(pre + "WAVE", wi), (pre + "FIELD", 1),
                     (pre + "PUPILSAMP", samp_index(n_p)),
                     (pre + "IMAGESAMP", samp_index(n_i)),
                     (pre + "IMAGEDELTA", float(h["image_delta_um"])),
                     (pre + "POLARIZATION", int(bool(h["use_polarization"])))]
            if kind == "mtf":
                pairs.append((pre + "MAXF", float(h["max_freq_lpmm"])))
            ok = io.modify_settings(an, pairs, kind)
            applied.append("MODIFYSETTINGS %s route: %s"
                           % (pre, "ok" if ok else "failed"))
        return ", ".join(applied)

    def run_psf(self, wi, tag, _retry=True):
        """(arr, dx_um, dy_um, desc) or None. dx unit auto-detected from
        the requested image delta and echoed. An all-zero grid (measured
        with UsePolarization=True) is retried once with the flag
        inverted."""
        an = self.io.new("HuygensPsf")
        desc_set = self.configure(an, wi, "psf")
        t0 = time.time()
        try:
            an.ApplyAndWaitForCompletion()
        except Exception as exc:
            self.log("  Huygens PSF %s FAILED: %s" % (tag, exc))
            self.session.close_analysis(an)
            return None
        grids = self.io.grab_grids(an)
        self.session.close_analysis(an)
        if grids is None:
            self.log("  Huygens PSF %s: no data (%s)" % (tag, desc_set))
            return None
        arr, (dx_g, dy_g, _mx, _my), desc = grids[0]
        if not np.isfinite(arr).any() or np.nansum(np.abs(arr)) == 0:
            self.log("  Huygens PSF %s: grid is ALL ZERO / invalid (settings: "
                     "%s)" % (tag, desc_set))
            if _retry:
                self.huy["use_polarization"] = not self.huy["use_polarization"]
                self.log("  -> retrying once with UsePolarization=%s"
                         % self.huy["use_polarization"])
                return self.run_psf(wi, tag, _retry=False)
            return None
        delta = float(self.huy["image_delta_um"])
        if abs(dx_g * 1000.0 - delta) < abs(dx_g - delta):
            dx_um, dy_um, unit = dx_g * 1000.0, dy_g * 1000.0, "mm"
        else:
            dx_um, dy_um, unit = dx_g, dy_g, "um"
        ny, nx = arr.shape
        self.log("  Huygens PSF %s: grid %dx%d, dx=%.3f um (DataGrid Dx=%.4g "
                 "read as %s; requested %.2f um), extent %.1f x %.1f um, "
                 "sum=%.4g  [%s]  (%.1fs)"
                 % (tag, nx, ny, dx_um, dx_g, unit, delta, nx * dx_um,
                    ny * dy_um, np.nansum(arr), desc, time.time() - t0))
        self.log("  settings: %s" % desc_set)
        return arr, dx_um, dy_um, desc

    def run_mtf(self, wi, tag):
        """(f_lpmm, tangential, sagittal) or None."""
        an = self.io.new("HuygensMtf")
        desc_set = self.configure(an, wi, "mtf")
        t0 = time.time()
        try:
            an.ApplyAndWaitForCompletion()
        except Exception as exc:
            self.log("  Huygens MTF %s FAILED: %s" % (tag, exc))
            self.session.close_analysis(an)
            return None
        series = self.io.grab_series(an)
        self.session.close_analysis(an)
        if not series:
            self.log("  Huygens MTF %s: no DataSeries (%s)" % (tag, desc_set))
            return None
        x, Y, labels, dsc = series[0]
        self.log("  Huygens MTF %s: %d points to %.0f cycles/mm, %d series %s "
                 "[%s]  (%.1fs)" % (tag, x.size, x[-1] if x.size else 0,
                                    Y.shape[1], labels, dsc, time.time() - t0))
        return x, Y[:, 0], (Y[:, 1] if Y.shape[1] > 1 else Y[:, 0])

    # --- DLL debug log ---------------------------------------------------------
    def dll_log_path(self):
        return os.path.join(dll_surfaces_dir(), "us_mdl_rings_log.txt")

    def arm_dll_log(self):
        try:
            if os.path.exists(self.dll_log_path()):
                os.remove(self.dll_log_path())
            self.zs.set_par(8, 1.0)
        except Exception as exc:
            self.log("  (DLL debug log not armed: %s)" % exc)

    def summarise_dll_log(self):
        log = self.log
        log.section("DLL debug log: what %s passed to the residual surface"
                    % self.ENGINE,
                    "first 400 DLL calls after arming (Par 8); type, "
                    "wavelength, indices, ray geometry, returned dz")
        try:
            self.zs.set_par(8, 0.0)
            rows = [ln.split() for ln in open(self.dll_log_path())
                    if ln.strip() and not ln.startswith("#")]
            if not rows:
                log("  log is EMPTY: the DLL was not called after arming "
                    "(or the file is elsewhere)")
                return
            kinds = Counter((r[1], r[-1]) for r in rows)
            log("  %d calls logged: %s" % (len(rows), ", ".join(
                "type %s/%s x%d" % (k[0], k[1], v) for k, v in kinds.items())))
            lams = sorted(set(float(r[5]) for r in rows))
            n1s = sorted(set(round(float(r[6]), 5) for r in rows))
            n2s = sorted(set(round(float(r[7]), 5) for r in rows))
            log("  wavelength(s) passed: %s | n1: %s | n2: %s"
                % (", ".join("%.4f" % v for v in lams[:6]),
                   ", ".join("%.5f" % v for v in n1s[:6]),
                   ", ".join("%.5f" % v for v in n2s[:6])))
            tr = [r for r in rows if r[-1] == "trace"]
            if tr:
                col = lambda i: np.array([float(r[i]) for r in tr])
                rho = np.hypot(col(8), col(9))
                zs_, ns, dzs, trs = col(10), col(13), col(14), col(15)
                log("  trace calls: rho %.4f..%.4f mm, incoming z %.2e..%.2e, "
                    "cos(n) %.5f..%.5f, returned dz %.3e..%.3e mm, tran "
                    "%.3f..%.3f" % (rho.min(), rho.max(), zs_.min(), zs_.max(),
                                    ns.min(), ns.max(), dzs.min(), dzs.max(),
                                    trs.min(), trs.max()))
            log("  first lines (call type numb surf wave lam n1 n2 x y z l m n "
                "dz tran p1..p7 what):")
            for r in rows[:8]:
                log("    " + " ".join(r))
            self.store["dll_log_rows"] = np.array([" ".join(r) for r in rows])
        except Exception as exc:
            log("  (DLL log not readable: %s)" % exc)

    # --- the analysis ----------------------------------------------------------
    def run(self):
        d, log, zs, h, hy = self.d, self.log, self.zs, self.huy, self.hy
        store = self.store
        store.update(variant="hybrid", lam_um=np.array(d.wavelengths_um))
        npz_path = os.path.join(self.out_dir, "huygens_hybrid.npz")
        r_cmp = float(hy["r_max_um"])
        r_prof = max(r_cmp, 30.0) + 5.0

        log.section("hybrid: focus check of the Paraxial lens",
                    "one marginal real ray through the lens, the plate and "
                    "the flat UDS must reach the axis at the image")
        zs.hybrid_focus_check(store)

        log.section("Huygens: pupil ray pitch -> Avg cell",
                    "one wavelet per pupil ray; the residual UDS is "
                    "cell-averaged on the ray pitch EPD/N")
        n_p, n_i = int(h["pupil_samp"]), int(h["image_samp"])
        cell_mm = d.epd_mm / float(n_p)
        zs.set_par(6, cell_mm)
        store.update(avg_cell_mm=cell_mm, huy_pupil_samp=n_p,
                     huy_image_samp=n_i,
                     huy_image_delta_um=float(h["image_delta_um"]))
        log("  pupil sampling %d^2 -> ray pitch %.2f um = Avg cell; image "
            "%d^2 at %.2f um (+-%.1f um); polarization/transmission %s"
            % (n_p, 1000.0 * cell_mm, n_i, h["image_delta_um"],
               0.5 * n_i * float(h["image_delta_um"]),
               "ON" if h["use_polarization"] else "OFF"))

        log.section("reference profiles",
                    "RS focal slice from rs/verify_rzmap.npz (ring quadrature "
                    "as tagged in rs/verify_metrics.json) and the in-script "
                    "sub-ring quadrature")
        rs = RsReference(d)
        log("  RS reference quadrature (rs/verify_rzmap.npz, verify_mtf.npz): "
            "%s  [%s]" % (rs.quad, rs.quad_src))
        fine = None
        if d.run_dir:
            try:
                fine = FineReference(d.ring_table(), d.F_um, r_cmp,
                                     hy["rsf_step_um"]).compute(d.wavelengths_um)
                fine.store_into(store)
                log("  fine-quadrature RS focal profiles computed (%.3f um "
                    "sub-ring step, %d lines) -- the staircase-integrated "
                    "reference" % (fine.step_um, len(fine.I)))
            except Exception as exc:
                log("  (fine-quadrature RS reference failed: %s)" % exc)

        log.section("%s propagation per wavelength" % self.ENGINE,
                    "one Huygens PSF (+ Huygens MTF) run per line to the "
                    "image plane; radial PSF, FWHM and correlation against "
                    "the references")
        log("  lam(um)  peak(x,y)um   FWHM_huy  FWHM_rs(%s)  ratio   "
            "FWHM_rs(fine)  ratio   corr_%s  corr_fine  P(3FWHM)/P"
            % (rs.tag, rs.tag))
        self.arm_dll_log()
        results = []
        for wi, lam in enumerate(d.wavelengths_um, start=1):
            key = int(round(lam * 1000))
            log.subsection("lam = %.2f um (wavelength #%d of %d)"
                           % (lam, wi, len(d.wavelengths_um)))
            res = self.run_psf(wi, "lam=%.2f um" % lam)
            hmtf = self.run_mtf(wi, "lam=%.2f um" % lam)
            if hmtf is not None:
                store["HMTF_f_%d" % key], store["HMTF_T_%d" % key], \
                    store["HMTF_S_%d" % key] = hmtf
            if res is None:
                continue
            arr, dx_um, dy_um, desc = res
            r_um, prof, pk_xy = radial_profile(arr, dx_um, r_prof)
            fwhm = fwhm_of(r_um, prof)
            eff = encircled_fraction(arr, dx_um, dy_um, r_um, prof, fwhm)
            # reference (a): run_verify focal slice, else analytic Airy
            ref = rs.focal_slice(key, d.F_um)
            if ref is not None:
                fwhm_ref = fwhm_of(ref[0], ref[1])
                if not np.isfinite(fwhm_ref):
                    jr = int(np.argmax(ref[1]))
                    log("  (RS focal slice at %.2f um has no on-axis half "
                        "crossing: I(0)/Imax = %.3f, peak at r = %.2f um -- "
                        "annular/defocused at z = F; rz peak plane differs). "
                        "%s profile re-taken ABOUT THE AXIS for the "
                        "comparison." % (lam, ref[1][0] / max(ref[1].max(), 1e-30),
                                         ref[0][jr], self.ENGINE))
                    r_um, prof, _ = radial_profile(arr, dx_um, r_prof,
                                                   about_axis=True)
                    jp = int(np.argmax(prof))
                    log("  %s about the axis: I(0)/Imax = %.3f, peak at r = "
                        "%.2f um" % (self.ENGINE,
                                     prof[0] / max(prof.max(), 1e-30), r_um[jp]))
                corr = corr_on(ref[0], ref[1], r_um, prof)
            else:
                ref = (r_um, airy_profile(r_um, lam, d.na_par))
                fwhm_ref = 1.029 * lam / (2.0 * d.na_par)
                corr = corr_on(ref[0], ref[1], r_um, prof)
            ratio = fwhm / fwhm_ref if (np.isfinite(fwhm) and np.isfinite(fwhm_ref)
                                        and fwhm_ref > 0) else float("nan")
            # reference (b): sub-ring RS
            fwhm_fine = corr_fine = ratio_fine = float("nan")
            fine_ref = None
            if fine is not None and key in fine.I:
                fine_ref = (fine.r_um, fine.I[key])
                fwhm_fine = fwhm_of(fine.r_um, fine.I[key])
                corr_fine = corr_on(fine.r_um, fine.I[key], r_um, prof)
                if np.isfinite(fwhm) and np.isfinite(fwhm_fine) and fwhm_fine > 0:
                    ratio_fine = fwhm / fwhm_fine
            log("  %.3f    (%+.1f,%+.1f)    %6.2f    %6.2f       %5.3f   %6.2f"
                "         %5.3f   %.4f    %.4f    %.3f"
                % (lam, pk_xy[0], pk_xy[1], fwhm, fwhm_ref, ratio, fwhm_fine,
                   ratio_fine, corr, corr_fine, eff))
            # MTF of the engine PSF on the RS frequency grid
            mtf = None
            if rs.mtf is not None:
                f_um = rs.mtf["f_lppmm"] / 1000.0
                r_w = float(rs.mtf["r_max_um"]) if "r_max_um" in rs.mtf.files else 30.0
                selm = r_um <= r_w
                mtf = hankel_mtf(prof[selm], r_um[selm], f_um)
                store["MTF_%d" % key] = mtf
                try:
                    m = rs.mtf_numbers(key, d.na_par, lam, mtf, hmtf)
                    log("  MTF %d nm: quality (area/limit to %.0f lp/mm) "
                        "Hankel(%s PSF) %.3f | Zemax Huygens MTF %.3f | RS "
                        "%.3f ; MTF50 %.0f / %.0f / %.0f lp/mm ; "
                        "max|Zemax-Hankel| %.3f, max|RS-Hankel| %.3f"
                        % (key, m["cutoff"], self.ENGINE, m["q_engine"],
                           m["q_zemax"], m["q_rs"], m["f50_engine"],
                           m["f50_zemax"], m["f50_rs"], m["dev_zemax"],
                           m["dev_rs"]))
                    store["mtf_quality_%d" % key] = np.array(
                        [m["q_engine"], m["q_zemax"], m["q_rs"]])
                except Exception as exc:
                    log("  (MTF numbers not computed: %s)" % exc)
            iy, ix = np.unravel_index(int(np.argmax(arr)), arr.shape)
            y0, x0 = max(iy - 256, 0), max(ix - 256, 0)
            store["I_%d" % key] = arr[y0:y0 + 512, x0:x0 + 512]
            store["dx_um_%d" % key] = dx_um
            store["r_um_%d" % key] = r_um
            store["prof_%d" % key] = prof
            results.append(dict(lam=lam, fwhm=fwhm, fwhm_ref=fwhm_ref, eff=eff,
                                corr=corr, fwhm_fine=fwhm_fine,
                                corr_fine=corr_fine))
            np.savez_compressed(npz_path, **store)
            try:
                fig_huygens_line(
                    os.path.join(self.out_dir, "fig_zemax_huygens_%d.png" % key),
                    key, r_um, prof, fwhm, fwhm_ref, ref, fine_ref,
                    hy["rsf_step_um"], rs.quad, mtf, hmtf, rs.mtf, r_cmp,
                    n_p, dx_um, self.ENGINE)
            except Exception as exc:
                log("  (figure failed: %s)" % exc)

        self.summarise_dll_log()
        np.savez_compressed(npz_path, **store)
        log.section("results and verdict",
                    "npz + figures written; the pass/fail statement")
        log("saved %s" % npz_path)
        self.verdict(results)

    def verdict(self, results):
        """Against the FULL fine reference (phase + transmission: the
        Huygens engine applies rel_surf_tran even with polarization off,
        2026-09-08). Annular lines (no on-axis half crossing in the
        reference) are judged by corr only."""
        if not results:
            self.log("  no Huygens result -- nothing to judge")
            return
        ratios = [r["fwhm"] / r["fwhm_fine"] for r in results
                  if np.isfinite(r["fwhm"]) and np.isfinite(r["fwhm_fine"])
                  and r["fwhm_fine"] > 0]
        corrs = [r["corr_fine"] for r in results if np.isfinite(r["corr_fine"])]
        good = corrs and min(corrs) > 0.98 and ratios and \
            max(abs(x - 1) for x in ratios) < 0.1
        self.log("  HUYGENS verdict (hybrid: Paraxial f=F + residual UDS, Avg "
                 "cell %.4f mm = pupil pitch) against the FINE-quadrature "
                 "(phase + transmission; polarization %s) RS reference: FWHM "
                 "Huygens/RS %s, corr %s -> %s"
                 % (self.store.get("avg_cell_mm", 0.0),
                    "on" if self.huy["use_polarization"] else "off",
                    ", ".join("%.3f" % x for x in ratios),
                    ", ".join("%.4f" % x for x in corrs),
                    "the Huygens PSF reproduces the RS focal PSF: Zemax's "
                    "native Huygens PSF/MTF on this .zos is the colleagues' "
                    "instrument" if good else
                    "MISMATCH: check the settings echo (pupil/image sampling, "
                    "image delta accepted?), the DLL log (one ray per pupil "
                    "sample?) and the transmission (UsePolarization)"))
