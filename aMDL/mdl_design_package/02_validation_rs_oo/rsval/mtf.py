"""
MtfRun -- the modulation transfer function of a designed MDL, computed
the ONLY way that is valid for a diffractive lens in this toolchain: as
the normalized Hankel transform of the trusted focal-plane PSF, NOT from
a ray-based engine.

WHY
---------------------------------------------------------------------------
OpticStudio's native MTF (Geometric and FFT) is INVALID on the zone/rz
surface model: a pure phase screen does not bend rays, so the ray-traced
exit pupil both flavours rely on is degenerate (afocal). The symptom is
an FFT-MTF frequency axis auto-scaled to ~0.25 lp/mm instead of the
physical cutoff 2*NA/lambda ~ 364 lp/mm at 550 nm. The rz route proved
the PHASE is exact (OPD wrapped RMS 0.0000 waves, field corr 1.0000), so
the honest MTF comes from the same focal field VerifyRun propagates:

    incoherent OTF(f) = FT2D{ |U(r,F)|^2 }
                      = 2*pi INT_0^inf I(r) J0(2*pi f r) r dr   (Hankel-0,
                        the PSF being rotationally symmetric)
    MTF(f) = |OTF(f)| / OTF(0),   OTF(0) = 2*pi INT I(r) r dr = window power.

Standard incoherent-imaging MTF (Goodman [1] Sec. 6.3; Born & Wolf [2]
Sec. 9.5). The J0 kernel is validated against the analytic clear-
circular-aperture MTF (2/pi)(arccos nu - nu sqrt(1-nu^2)), nu = f/f_c,
to < 2.1e-3 max error.

READING THE RESULT
---------------------------------------------------------------------------
A diffractive PSF is a diffraction-limited CORE on a broad HALO. Under
MTF(0) = 1 the halo, broad in space, is narrow in frequency: the MTF
drops steeply over the first tens of lp/mm, then settles on a PLATEAU
whose height ~ the core energy fraction and rides it to the full cutoff
2*NA/lambda. A LOW PLATEAU that still reaches the cutoff = sharp core at
low efficiency (the diffractive efficiency limit), NOT a wavefront
defect; aberration would TRUNCATE the cutoff. The paper's own MTF panels
(S1..S5) show exactly this efficiency spread at a common cutoff.

NORMALIZATION / MEASUREMENT MATCH. The MTF is computed over a finite
radial window (mtf_r_max_um, default verify_r_max_um), so OTF(0) is the
IN-WINDOW power -- the convention of a CCD-measured MTF and of
VerifyRun's strehl_shape. Match the window to the camera crop; compare
like-normalized curves only. ``in_window_frac`` (fraction of the incident
power inside the window) says how much halo the normalization excludes.
Equal spectral weights are assumed for the polychromatic curve.

Outputs (into rs/)
---------------------------------------------------------------------------
    rs/verify_mtf.npz   f_lppmm, lam_um, r_max_um, MTF_<nm>, MTFdl_<nm>,
                        MTF_poly, MTFdl_poly, fc_lppmm, mtf_quality
                        (area(MTF)/area(MTF_dl) to the own-lambda cutoff),
                        in_window_frac, mtf_quality_poly
    rs/fig_mtf.png      left: polychromatic design vs polychromatic limit
                        (the bounded comparison); right: per-wavelength
                        design (solid) vs its OWN limit (dashed)

[1] J. W. Goodman, Introduction to Fourier Optics, 3rd ed. (2005), Sec. 6.3.
[2] M. Born & E. Wolf, Principles of Optics, 7th ed. (1999), Sec. 9.5.
[3] Y. Xiao et al., Light Sci. Appl. 11, 323 (2022) (the MTF figure
    being compared against).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import pi
from scipy.special import j0

from .base import Log, Stage
from .design import DesignState
from .propagator import RSPropagator
from .verify import nm_key

FloatVec = np.ndarray


def hankel_mtf(I: FloatVec, r: FloatVec, f: FloatVec) -> FloatVec:
    """MTF(f) = |2 pi INT I(r) J0(2 pi f r) r dr| / (2 pi INT I(r) r dr).

    r in um, f in cycles/um. Trapezoid quadrature on the sampled PSF; the
    denominator is the in-window power (OTF(0)) -> MTF(0) = 1."""
    w = I * r
    dc = 2 * pi * np.trapezoid(w, r)
    out = np.empty(f.size)
    for i, ff in enumerate(f):
        out[i] = 2 * pi * np.trapezoid(w * j0(2 * pi * ff * r), r)
    return np.abs(out) / dc


@dataclass
class MtfGrids:
    """Radial PSF window and frequency axis (config keys mtf_*)."""
    r0: FloatVec            # 0 .. mtf_r_max_um, mtf_r_points          [um]
    f_um: FloatVec          # 0 .. f_max, mtf_f_points            [cyc/um]

    @property
    def f_lppmm(self) -> FloatVec:
        return self.f_um * 1000.0

    @property
    def r_max(self) -> float:
        return float(self.r0[-1])


@dataclass
class MtfResult:
    """Everything written to rs/verify_mtf.npz, per wavelength and poly."""
    f_lppmm: FloatVec
    lam_um: FloatVec
    r_max_um: float
    MTF: Dict[float, FloatVec] = field(default_factory=dict)      # lam -> design
    MTFdl: Dict[float, FloatVec] = field(default_factory=dict)    # lam -> ideal
    fc_lppmm: List[float] = field(default_factory=list)           # 2NA/lam
    mtf_quality: List[float] = field(default_factory=list)        # area ratio
    in_window_frac: List[float] = field(default_factory=list)     # captured power
    MTF_poly: Optional[FloatVec] = None
    MTFdl_poly: Optional[FloatVec] = None
    mtf_quality_poly: float = float("nan")

    def arrays(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"f_lppmm": self.f_lppmm, "lam_um": self.lam_um,
                               "r_max_um": self.r_max_um,
                               "note": "window-normalized incoherent MTF"}
        for lam in self.lam_um:
            out[nm_key("MTF", lam)] = self.MTF[lam]
            out[nm_key("MTFdl", lam)] = self.MTFdl[lam]
        out["MTF_poly"] = self.MTF_poly
        out["MTFdl_poly"] = self.MTFdl_poly
        out["fc_lppmm"] = np.asarray(self.fc_lppmm)
        out["mtf_quality"] = np.asarray(self.mtf_quality)
        out["in_window_frac"] = np.asarray(self.in_window_frac)
        out["mtf_quality_poly"] = self.mtf_quality_poly
        return out


class MtfRun(Stage):
    """One execution of ``mtf_verify`` on a loaded design.

    Attributes
    ----------
    d        DesignState
    rs       RSPropagator on the run's rs_ring_quadrature (the SAME
             kernel VerifyRun uses -- one source)
    lams     verification wavelengths                                [um]
    grids    MtfGrids (filled by run())
    result   MtfResult (filled by run())
    """

    def __init__(self, design: DesignState, log: Optional[Log] = None) -> None:
        super().__init__(design.run_dir, log)
        self.d: DesignState = design
        self.rs: RSPropagator = RSPropagator.from_design(design)
        self.lams: FloatVec = design.lams
        self.grids: Optional[MtfGrids] = None
        self.result: Optional[MtfResult] = None

    def make_grids(self) -> MtfGrids:
        c = self.d.cfg
        r0 = np.linspace(0.0, c.mtf_r_max(), int(c.mtf_r_points))
        f_um = np.linspace(0.0, c.mtf_f_max() / 1000.0, int(c.mtf_f_points))
        g = MtfGrids(r0, f_um)
        self.log("MTF: Hankel-0 of |U(r,F)|^2 over r=0..%.1f um (%d pts), "
                 "f=0..%.0f lp/mm (%d pts)"
                 % (g.r_max, r0.size, g.f_lppmm[-1], f_um.size))
        self.log("     window normalized (OTF(0)=in-window power; CCD/strehl_shape "
                 "convention). cutoff 2NA/lam: %.0f lp/mm @400nm .. %.0f lp/mm "
                 "@1100nm" % (2 * self.d.na / 0.400 * 1000, 2 * self.d.na / 1.100 * 1000))
        self.grids = g
        return g

    def compute(self) -> MtfResult:
        """Per-wavelength design and diffraction-limit MTFs, the quality
        area ratio to the OWN-lambda limit, and the equal-weight
        polychromatic pair (MTF of the summed focal intensities)."""
        g = self.grids or self.make_grids()
        d, rs, log = self.d, self.rs, self.log
        c, F, na = d.cfg, d.F, d.na
        r0, f_um, f_lppmm = g.r0, g.f_um, g.f_lppmm
        res = MtfResult(f_lppmm, self.lams, g.r_max)
        psf_sum = np.zeros(r0.size)
        psf_id_sum = np.zeros(r0.size)
        p_tot = rs.incident_power
        log("  lam(um)  cutoff   in-win%   MTF@50  MTF@100  MTF@200   quality "
            "(design vs OWN-lambda limit)")
        for lam in self.lams:
            I = np.abs(rs.psf(lam, F, r0)) ** 2
            Iid = rs.ideal_profile(lam, r0)
            psf_sum += I
            psf_id_sum += Iid
            mtf = hankel_mtf(I, r0, f_um)
            mtf_dl = hankel_mtf(Iid, r0, f_um)
            res.MTF[lam] = mtf
            res.MTFdl[lam] = mtf_dl
            fc = 2 * na / lam * 1000.0
            res.fc_lppmm.append(fc)
            # fraction of the incident power inside the window at the focus
            cap = float(2 * pi * np.trapezoid(I * r0, r0) / p_tot)
            res.in_window_frac.append(cap)
            # quality = area(design) / area(own-lambda limit) up to that cutoff
            m_cut = f_lppmm <= fc
            q = float(np.trapezoid(mtf[m_cut], f_lppmm[m_cut]) /
                      np.trapezoid(mtf_dl[m_cut], f_lppmm[m_cut]))
            res.mtf_quality.append(q)
            at = lambda freq: float(np.interp(freq, f_lppmm, mtf))  # noqa: E731
            log("  %.3f    %4.0f     %5.1f    %.3f   %.3f    %.3f      %.3f"
                % (lam, fc, 100 * cap, at(50), at(100), at(200), q))
        res.MTF_poly = hankel_mtf(psf_sum, r0, f_um)
        res.MTFdl_poly = hankel_mtf(psf_id_sum, r0, f_um)
        fc_poly = 2 * na / (0.5 * (c.lam_min_um + c.lam_max_um)) * 1000.0
        mpc = f_lppmm <= fc_poly
        res.mtf_quality_poly = float(
            np.trapezoid(res.MTF_poly[mpc], f_lppmm[mpc]) /
            np.trapezoid(res.MTFdl_poly[mpc], f_lppmm[mpc]))
        self.result = res
        return res

    def save(self) -> str:
        assert self.result is not None
        path = self.write_npz("verify_mtf.npz", self.result.arrays())
        self.log("polychromatic (equal weights): quality area ratio = %.3f "
                 "(1.0 = diffraction limit)" % self.result.mtf_quality_poly)
        self.log("saved rs\\verify_mtf.npz")
        return path

    def figure(self) -> Optional[str]:
        """rs/fig_mtf.png: LEFT the bounded polychromatic comparison
        (design <= limit always), RIGHT each wavelength vs its OWN
        monochromatic limit (never read a monochromatic curve against the
        polychromatic dashed line -- that is what makes short lines look
        like they beat the limit). Returns the path, None if matplotlib
        is unavailable."""
        res, g = self.result, self.grids
        assert res is not None and g is not None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                      # pragma: no cover
            self.log("plot skipped (%s); data is in rs\\verify_mtf.npz" % e)
            return None
        f_lppmm, cap = res.f_lppmm, res.in_window_frac
        fig, (axp, axl) = plt.subplots(1, 2, figsize=(12.0, 4.7))
        axp.plot(f_lppmm, res.MTFdl_poly, color="k", lw=1.8, ls="--",
                 label="diffraction limit (polychromatic)")
        axp.plot(f_lppmm, res.MTF_poly, color="#e6550d", lw=2.6,
                 label="design (polychromatic)")
        axp.set_xlim(0, f_lppmm[-1])
        axp.set_ylim(0, 1.0)
        axp.set_xlabel("spatial frequency  (lp/mm)")
        axp.set_ylabel("MTF")
        axp.set_title("Polychromatic MTF (the bounded comparison)\n"
                      "area ratio design/limit = %.3f" % res.mtf_quality_poly,
                      fontsize=9)
        axp.legend(fontsize=8, loc="upper right")
        axp.grid(True, alpha=0.25)
        cmap = plt.get_cmap("turbo")
        n = self.lams.size
        for i, lam in enumerate(self.lams):
            col = cmap(i / max(n - 1, 1))
            axl.plot(f_lppmm, res.MTF[lam], color=col, lw=1.1,
                     label="%d nm" % int(lam * 1000))
            axl.plot(f_lppmm, res.MTFdl[lam], color=col, lw=0.8, ls="--", alpha=0.6)
        axl.set_xlim(0, f_lppmm[-1])
        axl.set_ylim(0, 1.0)
        axl.set_xlabel("spatial frequency  (lp/mm)")
        axl.set_ylabel("MTF")
        axl.set_title("Per-wavelength: design (solid) vs its OWN limit (dashed)\n"
                      "window r<=%.0f um (in-window %.0f-%.0f%% of incident power)"
                      % (g.r_max, 100 * min(cap), 100 * max(cap)), fontsize=9)
        axl.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
        axl.grid(True, alpha=0.25)
        fig.suptitle("MDL '%s': incoherent MTF from RS focal field "
                     "(window-normalized, CCD convention)" % self.d.cfg.name,
                     fontsize=10)
        fig.tight_layout()
        path = self.rs_path("fig_mtf.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        self.log("saved rs\\fig_mtf.png")
        return path

    def run(self, script_path: Optional[str] = None) -> MtfResult:
        d = self.d
        self.snapshot(script_path)
        d.describe(self.log, m_line=True)
        d.describe_quadratures(self.log)
        self.make_grids()
        res = self.compute()
        self.save()
        self.figure()
        self.log("next: overlay rs\\fig_mtf.png on the paper's MTF (same lp/mm "
                 "axis); compare like-normalized curves only.")
        return res
