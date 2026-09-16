"""
VerifyRun -- the four sections of stage 2a on one run folder and the
files they leave in ``<run>/rs/``.

METRICS reported per verification wavelength
---------------------------------------------------------------------------
* z_peak_um   : argmax_z of the on-axis intensity |U(0,z)|^2 over the
                scan F +/- verify_z_span_um -- the focal-shift /
                achromaticity check (the paper's Fig. 2e quantity).
                z_peak_tile_window_um is the same argmax restricted to
                the r-z tile window (F +/- rzmap_z_span_um) and
                I_tilewin_over_global their intensity ratio: 1.0 means
                the main focus IS the global peak, < 1 means a brighter
                SATELLITE focus lies outside the tile window.
* fwhm_um     : full width at half maximum of |U(r0,F)|^2 (first
                half-crossing of the radial profile, doubled).
* eff_3fwhm   : focusing efficiency = power within a disc of DIAMETER
                3x FWHM around the focus / total incident power
                (pi*R^2 for unit amplitude) -- the paper's own
                definition [4]; see Engelberg & Levy [7] on why this
                convention matters.
* strehl_like : on-axis peak intensity / peak of an IDEAL lens of the
                same aperture and focal length (design: sub-ring
                quadrature via the sinc factor; ideal: midpoint, exact
                for its continuous phase). EFFICIENCY-INCLUSIVE: for a
                diffraction-limited core strehl_like ~ encircled
                efficiency, so low values mean halo/other-order loss,
                NOT aberration. Family of the paper's Supplementary
                S3-6 "Normalized Strehl ratio" -- theirs normalizes by
                the power reaching the focal plane, ours (stricter) by
                the total incident power.
* strehl_shape: the paper's Fig. 4f convention (ref. 43 of [4]) --
                Strehl from the PSF normalized to the power CAPTURED
                IN THE MEASUREMENT WINDOW (r <= verify_r_max_um):
                  S = [max I / P_win] / [max I_ideal / P_win,ideal],
                  P_win = INT_window I(r) 2 pi r dr.
                SHAPE-ONLY: diffraction efficiency cancels. Window-
                dependent (a wider window admits more halo). Compare
                strehl_shape to Fig. 4f and strehl_like to the
                Supplementary Normalized Strehl -- NEVER across
                conventions.
* onax_I_at_F : |U(0, F)|^2 (absolute, unit-amplitude illumination).
Also: J on the design objective, on the continuous band and on the
verification comb (arithmetic-mean convention in all three, so runs
with different fom_mode stay comparable), and the ring table is
re-exported so it always matches the vector actually verified.

Outputs (per-solver layout, 2026-08-28)
---------------------------------------------------------------------------
    rs/verify_metrics.json   scalars per wavelength + J summary
    rs/verify_onaxis.npz     zgrid, I_<nm>: on-axis scans
    rs/verify_rzmap.npz      r0grid, zgrid, I_<nm>: r-z intensity maps
                             (raw data of the paper's Fig. 2e / 4a tiles)
    mdl_rings_<n>.txt        run ROOT: fabrication-facing artifact
    scripts/                 run_verify.py + rsval/ as run

[4] Y. Xiao et al., Light Sci. Appl. 11, 323 (2022).
[7] J. Engelberg & U. Levy, Nat. Photonics 16, 171-173 (2022).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import pi

from .base import VERSION, Log, Stage
from .design import DesignState
from .propagator import RSPropagator

FloatVec = np.ndarray
FloatMat = np.ndarray


def nm_key(prefix: str, lam_um: float) -> str:
    """``I_<nm>`` style array names used in every npz of the toolchain."""
    return "%s_%d" % (prefix, int(lam_um * 1000))


# ---------------------------------------------------------------------------
# section results
# ---------------------------------------------------------------------------
@dataclass
class OnAxisScan:
    """[1/4] |U(0, z)|^2 over F +/- verify_z_span_um, per wavelength."""
    zgrid: FloatVec
    I: Dict[float, FloatVec] = field(default_factory=dict)   # lam -> I(z)
    z_peak_um: List[float] = field(default_factory=list)
    z_peak_tile_window_um: List[float] = field(default_factory=list)
    I_tilewin_over_global: List[float] = field(default_factory=list)

    def arrays(self, lams: FloatVec) -> Dict[str, Any]:
        out: Dict[str, Any] = {"zgrid": self.zgrid}
        out.update({nm_key("I", l): self.I[l] for l in lams})
        return out


@dataclass
class PsfMetrics:
    """[2/4] focal-plane metrics, one entry per wavelength (see module doc)."""
    fwhm_um: List[float] = field(default_factory=list)
    eff_3fwhm: List[float] = field(default_factory=list)
    strehl_like: List[float] = field(default_factory=list)
    strehl_shape: List[float] = field(default_factory=list)
    onax_I_at_F: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {"fwhm_um": self.fwhm_um, "eff_3fwhm": self.eff_3fwhm,
                "strehl_like": self.strehl_like, "strehl_shape": self.strehl_shape,
                "onax_I_at_F": self.onax_I_at_F}


@dataclass
class RzMaps:
    """[3/4] I(r, z) tiles around the design focus, per wavelength."""
    r0grid: FloatVec
    zgrid: FloatVec
    I: Dict[float, FloatMat] = field(default_factory=dict)   # lam -> I[z, r]

    def arrays(self, lams: FloatVec) -> Dict[str, Any]:
        out: Dict[str, Any] = {"r0grid": self.r0grid, "zgrid": self.zgrid}
        out.update({nm_key("I", l): self.I[l] for l in lams})
        return out


@dataclass
class JMetrics:
    """[4/4] design-FOM values (arithmetic mean over the wavelength set)."""
    J_continuous: float
    J_verify_comb: float
    J_objective: float


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
class VerifyRun(Stage):
    """One execution of ``run_verify`` on a loaded design.

    Attributes
    ----------
    d          DesignState (config, m, h, MDLProblem)
    rs         RSPropagator on the run's rs_ring_quadrature
    lams       verification wavelengths                               [um]
    results    the dict written to rs/verify_metrics.json
    onaxis, psf, rz, jm   the section results (filled by run())
    """

    def __init__(self, design: DesignState, log: Optional[Log] = None) -> None:
        super().__init__(design.run_dir, log)
        self.d: DesignState = design
        self.rs: RSPropagator = RSPropagator.from_design(design)
        self.lams: FloatVec = design.lams
        c = design.cfg
        self.results: Dict[str, Any] = {
            "run_dir": design.run_dir, "m_file": design.m_file,
            "lam_um": self.lams.tolist(), "F_um": design.F,
            "rs_ring_quadrature": c.rs_ring_quadrature,
            "fom_ring_quadrature": c.ring_quadrature,
            "rsval_version": VERSION}
        self.onaxis: Optional[OnAxisScan] = None
        self.psf: Optional[PsfMetrics] = None
        self.rz: Optional[RzMaps] = None
        self.jm: Optional[JMetrics] = None

    # -- [1/4] ---------------------------------------------------------------------
    def onaxis_scans(self) -> OnAxisScan:
        """On-axis intensity over the full scan; global peak and the peak
        inside the r-z tile window (satellite detection, see module doc)."""
        c, F, log = self.d.cfg, self.d.F, self.log
        zgrid = np.linspace(F - c.verify_z_span_um, F + c.verify_z_span_um,
                            c.verify_z_points)
        res = OnAxisScan(zgrid)
        log("[1/4] on-axis scans |U(0,z)|^2: z = F +/- %.2f mm, %d planes "
            "(exact RS-I on axis; feeds fig_onaxis*.png + z_peak metric)"
            % (c.verify_z_span_um / 1000, zgrid.size))
        tile_span = c.rzmap_z_span_um
        for lam in self.lams:
            I = np.abs(self.rs.onaxis(lam, zgrid)) ** 2
            res.I[lam] = I
            ig = int(np.argmax(I))
            zpk = float(zgrid[ig])
            win = np.where(np.abs(zgrid - F) <= tile_span)[0]
            iw = win[int(np.argmax(I[win]))]
            zpk_w = float(zgrid[iw])
            ratio = float(I[iw] / I[ig])          # 1.0 = main focus IS global peak
            res.z_peak_um.append(zpk)
            res.z_peak_tile_window_um.append(zpk_w)
            res.I_tilewin_over_global.append(ratio)
            if ratio > 0.9999:
                log("  lam=%.2f um: peak at z=%.3f mm (offset %+0.0f um from F)"
                    % (lam, zpk / 1000, zpk - F))
            else:
                log("  lam=%.2f um: main focus z=%.3f mm (in tile window); "
                    "brighter SATELLITE at z=%.3f mm (%+0.0f um), I_main/I_sat=%.2f"
                    % (lam, zpk_w / 1000, zpk / 1000, zpk - F, ratio))
        log("[1/4] on-axis scans done")
        self.results["z_peak_um"] = res.z_peak_um
        self.results["z_peak_tile_window_um"] = res.z_peak_tile_window_um
        self.results["I_tilewin_over_global"] = res.I_tilewin_over_global
        self.onaxis = res
        return res

    # -- [2/4] ---------------------------------------------------------------------
    def psf_metrics(self) -> PsfMetrics:
        """FWHM, 3xFWHM-disc efficiency, strehl_like, strehl_shape at z = F."""
        c, F, na, log, rs = self.d.cfg, self.d.F, self.d.na, self.log, self.rs
        r0grid = np.linspace(0.0, c.verify_r_max_um, c.verify_r_points)
        res = PsfMetrics()
        log("[2/4] focal-plane PSFs |U(r,F)|^2: r = 0..%g um, %d points "
            "(feeds fig_metrics.png)" % (c.verify_r_max_um, r0grid.size))
        log("      columns: FWHM | dl = diffraction limit lam/2NA | eff = power "
            "in 3xFWHM-diameter disc / total incident | S = peak vs ideal lens "
            "(efficiency-inclusive) | Sshape = window-normalized shape Strehl "
            "(paper Fig. 4f convention; efficiency cancels)")
        p_tot = rs.incident_power
        for lam in self.lams:
            I = np.abs(rs.psf(lam, F, r0grid)) ** 2
            Ipk = I.max()
            # FWHM: radius of the first crossing below half the peak, doubled
            idx = np.where(I < Ipk / 2)[0]
            fwhm = 2 * r0grid[idx[0]] if idx.size else np.nan
            # paper convention [4],[7]: power inside a disc of DIAMETER 3 FWHM
            sel = r0grid <= 1.5 * fwhm
            p_in = np.trapezoid(I[sel] * 2 * pi * r0grid[sel], r0grid[sel])
            ideal = rs.ideal_peak(lam)
            # shape Strehl: both PSFs normalized to the in-window power
            I_id = rs.ideal_profile(lam, r0grid)
            p_win = np.trapezoid(I * 2 * pi * r0grid, r0grid)
            p_win_id = np.trapezoid(I_id * 2 * pi * r0grid, r0grid)
            s_shape = float((I.max() / p_win) / (I_id.max() / p_win_id)) \
                if p_win > 0 else float("nan")
            res.fwhm_um.append(float(fwhm))
            res.eff_3fwhm.append(float(p_in / p_tot))
            res.strehl_like.append(float(Ipk / ideal ** 2))
            res.strehl_shape.append(s_shape)
            res.onax_I_at_F.append(float(I[0]))
            log("  lam=%.2f: FWHM=%.2f um (dl %.2f), eff=%.3f, S=%.3f, Sshape=%.3f"
                % (lam, fwhm, lam / (2 * na), res.eff_3fwhm[-1],
                   res.strehl_like[-1], s_shape))
        self.results.update(res.as_dict())
        self.psf = res
        return res

    # -- [3/4] ---------------------------------------------------------------------
    def rz_maps(self) -> RzMaps:
        """I(r, z) of the designed staircase on the rzmap_* window around
        the design focus (raw data of the paper's Fig. 2e / 4a tiles)."""
        c, F, log, rs = self.d.cfg, self.d.F, self.log, self.rs
        rz_r = np.linspace(0.0, c.rzmap_r_max_um, c.rzmap_r_points)
        rz_z = np.linspace(F - c.rzmap_z_span_um, F + c.rzmap_z_span_um,
                           c.rzmap_z_points)
        res = RzMaps(rz_r, rz_z)
        log("[3/4] I(r,z) maps for fig_rz_tiles.png (paper Fig. 2e/4a analogue): "
            "window r = 0..%g um (%d radii; mirrored to +/-r in the plot), "
            "z = F +/- %.2f mm (%d planes)"
            % (rz_r[-1], rz_r.size, (rz_z[-1] - F) / 1000, rz_z.size))
        log("      = %d RS-I integrals over %d rings per wavelength, %d wavelengths"
            % (rz_z.size * rz_r.size, self.d.N, self.lams.size))
        for lam in self.lams:
            t_lam = time.time()
            M = rs.intensity_map(lam, rz_z, rz_r)
            res.I[lam] = M
            izm, irm = np.unravel_index(np.argmax(M), M.shape)
            log("  lam=%.2f um: map done (%.1fs); tile peak at r=%.1f um, z=%.3f mm"
                % (lam, time.time() - t_lam, rz_r[irm], rz_z[izm] / 1000))
        self.write_npz("verify_rzmap.npz", res.arrays(self.lams))
        log("[3/4] r-z maps done -> verify_rzmap.npz (arrays: r0grid, zgrid, "
            "I_<nm> per wavelength)")
        self.rz = res
        return res

    # -- [4/4] ---------------------------------------------------------------------
    def j_metrics(self) -> JMetrics:
        """J on the continuous band (the run's MDLProblem), on the
        verification comb and on the design objective (its own comb when
        the objective was discrete, else = continuous). Arithmetic mean in
        all three (MDLProblem default), whatever fom_mode the design used."""
        d, log = self.d, self.log
        log("[4/4] J metrics (all arithmetic-mean, comparable across runs)...")
        J_cont = d.prob.fom(d.m)
        J_comb = d.comb_problem(self.lams).fom(d.m)
        if d.cfg.target_wavelengths_um is not None:
            J_obj = d.comb_problem(np.asarray(d.cfg.target_wavelengths_um,
                                              dtype=float)).fom(d.m)
        else:
            J_obj = J_cont
        res = JMetrics(J_cont, J_comb, J_obj)
        log("J objective=%.4f   continuous=%.4f   verify-comb=%.4f"
            % (J_obj, J_cont, J_comb))
        self.results.update(J_continuous=J_cont, J_verify_comb=J_comb,
                            J_objective=J_obj)
        self.jm = res
        return res

    # -- outputs -------------------------------------------------------------------
    def save(self) -> None:
        assert self.onaxis is not None
        self.write_npz("verify_onaxis.npz", self.onaxis.arrays(self.lams))
        self.write_json("verify_metrics.json", self.results)
        ring_path = self.d.write_ring_table()
        self.saved_lines([
            ("rs\\verify_metrics.json", "scalar metrics per wavelength + J summary"),
            ("rs\\verify_onaxis.npz", "zgrid + I_<nm>: on-axis scans (fig_onaxis*)"),
            ("rs\\verify_rzmap.npz", "r0grid, zgrid + I_<nm>: r-z maps (fig_rz_tiles)"),
            (os.path.basename(ring_path),
             "ring table re-export (run ROOT), matches the verified vector")])

    # -- driver --------------------------------------------------------------------
    def run(self, script_path: Optional[str] = None) -> Dict[str, Any]:
        """All four sections, the files, and the log block. Returns the
        metrics dict (= rs/verify_metrics.json)."""
        d = self.d
        self.snapshot(script_path)
        d.describe(self.log)
        d.describe_quadratures(self.log)
        self.log("J_continuous(alias-safe, Nw=%d) = %.4f"
                 % (d.cfg.n_wavelengths, d.prob.fom(d.m)))
        self.onaxis_scans()
        self.psf_metrics()
        self.rz_maps()
        self.j_metrics()
        self.save()
        rel = os.path.relpath(self.run_dir)
        self.log("next:  python %s %s   then   python %s %s" % (
            os.path.join("02_Rayleigh_Sommerfeld_Validation_OOP", "mtf_verify.py"), rel,
            os.path.join("02_Rayleigh_Sommerfeld_Validation_OOP", "make_plots.py"), rel))
        return self.results
