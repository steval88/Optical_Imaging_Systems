"""rz -- BATCH-OPD ROUTE: I(r, z) tiles from the ZEMAX-TRACED field.

The batch ray trace has zero settings plumbing. One subtlety, MEASURED
2026-08-31: the batch OPD is referenced to the chief-ray REFERENCE
SPHERE, a wavelength-independent sag of hundreds of waves across this
aperture. Hence a NULL SUBTRACTION per wavelength: the zone DLL's Par 2
is a height scale, so scale 0 makes the surface FLAT while leaving the
pupils and the reference-sphere convention untouched -- trace flat,
trace the design, subtract. Per wavelength:
  1. two batch traces of one ray per ring (py = rho_i / R);
  2. SELF-CHECK: wrapped RMS of (d_opd - ring-table phase) in waves --
     does OpticStudio's model of the surface reproduce the design
     phase? (expect ~0);
  3. propagate the ZEMAX field exp(i 2 pi OPD) to the I(r,z) window of
     rs/verify_rzmap.npz with the IDENTICAL RS-I kernel AND the same
     ring quadrature as that file (tag read from rs/verify_metrics.json:
     "sinc" since 2026-09-08, "midpoint" before), so any difference is
     the surface model, not the propagator. Measured 2026-09-15 with
     mismatched rules (sinc file vs midpoint tiles): corr 0.998-0.9999
     and one-plane (17 um) peak shifts at 550 / 950 nm -- a quadrature
     artefact, the OPD self-check was 0.0000 throughout;
  4. per-line z-peak comparison and tile correlation against rs/.
Output: zemax/zemax_rzmap.npz + fig_zemax_rz_tiles.png +
fig_zemax_onaxis_perlambda.png. Seconds per wavelength.
"""
import os
import time

import numpy as np

from .base import Analysis
from .references import RsReference, rs_tiles
from .report import fig_rz_tiles
from .settings import RZ_SETTINGS


class RzAnalysis(Analysis):
    MODE = "rz"
    VARIANT = "zone"
    ZOS_NAME = "mdl_validation_rz.zos"

    def __init__(self, design, lams=None, **kw):
        super().__init__(design, **kw)
        self.lams = lams or RZ_SETTINGS["lams"]

    def select_lines(self):
        self.d.select_lines(self.lams)

    def settings_record(self):
        return {"lams": self.lams, "rz_span_mm": self.d.rz_span_mm,
                "rz_r_max_um": self.d.rz_r_max_um,
                "rz_r_points": self.d.rz_r_points,
                "rz_z_points": self.d.rz_z_points}

    def describe(self):
        self.log("  rz (batch-OPD route): Zemax traces one ray per ring "
                 "through the DLL; the traced OPD is self-checked against "
                 "the ring table, then the ZEMAX field is propagated to the "
                 "full I(r,z) tile grid (RS-I kernel, same windows as "
                 "rs/verify_rzmap) -- seconds per wavelength, %d wavelengths"
                 % len(self.d.wavelengths_um))

    def run(self):
        d, log, zs = self.d, self.log, self.zs
        log.section("rz: batch-OPD trace per wavelength -> I(r,z) tiles",
                    "one ray per ring through the DLL, OPD self-check "
                    "against the ring table, RS-I propagation of the Zemax "
                    "field")
        tab = d.ring_table()
        if abs(tab.R_um / 1000.0 - d.epd_mm / 2.0) > 1e-6:
            log("  WARNING: ring table aperture %.4f mm != EPD/2 %.4f mm"
                % (tab.R_um / 1000.0, d.epd_mm / 2.0))
        rs = RsReference(d)
        log("  ring quadrature for the Zemax-field tiles: %s  [rs/verify_"
            "metrics.json: %s] -- same rule as the rs/ tiles"
            % (rs.quad, rs.quad_src))
        span_um = d.rz_span_mm * 1000.0
        r0grid = np.linspace(0.0, d.rz_r_max_um, d.rz_r_points)
        zgrid = np.linspace(d.F_um - span_um, d.F_um + span_um, d.rz_z_points)
        store = self.store
        store.update(r0grid=r0grid, zgrid=zgrid,
                     lam_um=np.array(d.wavelengths_um),
                     ring_quadrature=rs.quad)
        npz_path = os.path.join(self.out_dir, "zemax_rzmap.npz")
        phase_rms, zpk = [], {}
        for wi, lam in enumerate(d.wavelengths_um, start=1):
            t0 = time.time()
            zs.set_par(2, 0.0)                    # FLAT: instrument terms
            opd_flat = zs.trace_opd(wi, tab)
            zs.set_par(2, 1.0)                    # the design surface
            opd_dsgn = zs.trace_opd(wi, tab)
            bad = int(np.isnan(opd_flat).sum() + np.isnan(opd_dsgn).sum())
            if bad:
                log("  lam=%.2f um: %d ray results missing/vignetted -- "
                    "SKIPPING" % (lam, bad))
                continue
            d_opd = opd_dsgn - opd_flat           # pure surface phase [waves]
            phi_tab = tab.phase_waves(lam)
            dd = (d_opd - d_opd[0]) - (phi_tab - phi_tab[0])
            d_wrap = dd - np.round(dd)
            rms_w = float(np.sqrt(np.mean(d_wrap ** 2)))
            phase_rms.append(rms_w)
            U = np.exp(1j * 2.0 * np.pi * (d_opd - d_opd[0]))
            M = rs_tiles(U, tab.rho_um, tab.delta_um, lam, zgrid, r0grid,
                         quad=rs.quad)
            key = int(round(lam * 1000))
            store["I_%d" % key] = M
            store["opd_waves_%d" % key] = d_opd
            izm, irm = np.unravel_index(int(np.argmax(M)), M.shape)
            zpk[key] = float(zgrid[izm])
            log("  lam=%.2f um: OPD self-check wrapped RMS = %.4f waves | "
                "tile peak r=%.1f um z=%.3f mm  (%.1fs)"
                % (lam, rms_w, r0grid[irm], zgrid[izm] / 1000.0,
                   time.time() - t0))
            store["phase_rms_waves"] = np.array(phase_rms)
            np.savez_compressed(npz_path, **store)
        log("saved %s" % npz_path)

        rs_path = d.rs_path("verify_rzmap.npz")
        if rs_path:
            rs = np.load(rs_path)
            log.section("rz: Zemax field vs RS tiles",
                        "peak plane and Pearson correlation per line; "
                        "1.0000 = identical structure")
            log("Zemax-field vs RS tiles (%s, both sides %s quadrature):"
                % (rs_path, store["ring_quadrature"]))
            log("  lam(nm)  z_pk_zemax  z_pk_rs   dz(um)   corr")
            for lam in d.wavelengths_um:
                key = int(round(lam * 1000))
                if "I_%d" % key not in rs.files or key not in zpk:
                    continue
                Mr, Mz = rs["I_%d" % key], store["I_%d" % key]
                z_rs = float(rs["zgrid"][int(np.argmax(np.max(Mr, axis=1)))])
                c = float(np.corrcoef(Mr.ravel(), Mz.ravel())[0, 1]) \
                    if Mr.shape == Mz.shape else float("nan")
                log("  %5d   %8.3f   %8.3f   %+6.0f   %.4f"
                    % (key, zpk[key] / 1000.0, z_rs / 1000.0,
                       zpk[key] - z_rs, c))
            log("  (corr = Pearson r over the full tile; 1.0000 = identical "
                "structure. dz in um.)")
        try:
            lams_nm = [k for k in (int(round(l * 1000)) for l in d.wavelengths_um)
                       if "I_%d" % k in store]
            fig_rz_tiles(self.out_dir, store, lams_nm, r0grid, zgrid,
                         d.bfd_mm, log)
        except Exception as exc:
            log("(figure rendering failed: %s -- the npz holds all the data; "
                "re-plot offline)" % exc)
        log("read the tiles against rs\\fig_rz_tiles.png and "
            "bpm\\fig_rz_tiles_bpm.png: same peaks, satellites and ridge at "
            "F = Zemax's surface model agrees with the design at the "
            "intensity level.")
