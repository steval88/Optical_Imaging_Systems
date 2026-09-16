"""
PlotRun -- the figures of stage 2a from the files run_verify / mtf_verify
leave in ``<run>/rs/`` (older run folders with the verify_* files at the
top level are read transparently: subfolder first, root fallback).

    fig_onaxis.png     |U(0, z)|^2 per wavelength over the scan, absolute
                       (left) and normalized (right), F marked -- the
                       focal-shift / achromaticity picture (paper Fig. 2e)
    fig_psf.png        focal-plane PSF |U(r, F)|^2 per wavelength against
                       the ideal-lens PSF of the same aperture, one panel
                       per line (radial, normalized to the ideal peak ->
                       the panel shows the Strehl-like ratio directly)
    fig_rz_tiles.png   I(r, z) tiles per wavelength in ONE ROW, mirrored
                       to +/-r, shared absolute z axis (F dashed), each
                       tile normalized to its own peak, one colour bar
                       (paper Fig. 2e layout, OrRd by default)
    fig_metrics.png    the scalar metrics vs wavelength: focus offset,
                       FWHM vs the diffraction limit, eff / Strehl-like,
                       shape-Strehl, and the MTF quality when
                       verify_mtf.npz exists
    fig_psf_2d.png     the focal spots as 2-D images (revolved radial
                       profiles), one row, one per wavelength, common scale

The design itself is not re-propagated except for the ideal-lens
reference profiles of fig_psf (RSPropagator.ideal_profile on the
rz-map radial grid; a second's work).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Log, Stage
from .design import DesignState
from .propagator import RSPropagator
from .verify import nm_key

FloatVec = np.ndarray


class RsFiles:
    """Locate and load the rs/ outputs (subfolder first, root fallback)."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def path(self, name: str) -> Optional[str]:
        for cand in (os.path.join(self.run_dir, "rs", name), os.path.join(self.run_dir, name)):
            if os.path.exists(cand):
                return cand
        return None

    def metrics(self) -> Dict[str, Any]:
        p = self.path("verify_metrics.json")
        if p is None:
            raise SystemExit("no verify_metrics.json in %s -- run run_verify.py first" % self.run_dir)
        with open(p) as fh:
            return json.load(fh)

    def npz(self, name: str) -> Optional[Dict[str, Any]]:
        p = self.path(name)
        if p is None:
            return None
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}


class PlotRun(Stage):
    """All figures of a run folder.

    Attributes
    ----------
    d          the design state (config, geometry, wavelengths)
    files      locator of the rs/ products (subfolder first, root fallback)
    lams       the wavelengths of the run [um]
    cmap       matplotlib colormap of the intensity images (fig_psf_2d,
               fig_rz_tiles). Default "OrRd": white background, orange to
               dark red, the paper's Fig. 2e / 4a convention and the look
               of the legacy make_plots.py. Any matplotlib name works
               ("inferno", "magma", "viridis", ...); echoed in the log.
    line_cmap  colormap of the per-wavelength line colours ("turbo").
    written    file names written so far (rs/)
    """
    DEFAULT_CMAP = "OrRd"

    def __init__(self, design: DesignState, log: Optional[Log] = None,
                 cmap: str = DEFAULT_CMAP, line_cmap: str = "turbo") -> None:
        super().__init__(design.run_dir, log)
        self.d = design
        self.files = RsFiles(design.run_dir)
        self.lams: FloatVec = design.lams
        self.cmap: str = cmap
        self.line_cmap: str = line_cmap
        self.written: List[str] = []

    # -- helpers -----------------------------------------------------------------
    def _plt(self) -> Any:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    def _colors(self, plt: Any) -> List[Any]:
        cmap = plt.get_cmap(self.line_cmap)
        n = self.lams.size
        return [cmap(i / max(n - 1, 1)) for i in range(n)]

    def _save(self, fig: Any, name: str) -> str:
        path = self.rs_path(name)
        fig.savefig(path, dpi=140)
        self.written.append(name)
        self.log("saved rs\\%s" % name)
        return path

    # -- figures -------------------------------------------------------------------
    def fig_onaxis(self) -> Optional[str]:
        z = self.files.npz("verify_onaxis.npz")
        if z is None:
            self.log("verify_onaxis.npz missing -- fig_onaxis skipped")
            return None
        plt = self._plt()
        F = self.d.F
        zg = z["zgrid"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
        for lam, c in zip(self.lams, self._colors(plt)):
            key = nm_key("I", lam)
            if key not in z:
                continue
            I = z[key]
            a1.plot((zg - F) / 1000, I, color=c, lw=1.0, label="%d nm" % int(lam * 1000))
            a2.plot((zg - F) / 1000, I / I.max(), color=c, lw=1.0)
        for ax in (a1, a2):
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("z - F  (mm)")
            ax.grid(True, alpha=0.25)
        a1.set_ylabel("|U(0, z)|^2  (unit-amplitude illumination)")
        a2.set_ylabel("normalized to each line's peak")
        a1.set_title("on-axis intensity along the scan (exact RS-I)", fontsize=9)
        a2.set_title("focal shift per line (paper Fig. 2e)", fontsize=9)
        a1.legend(fontsize=7, ncol=2)
        fig.suptitle("MDL '%s': on-axis scans, F = %.2f mm, RS quadrature %s"
                     % (self.d.cfg.name, F / 1000, self.d.cfg.rs_ring_quadrature), fontsize=10)
        fig.tight_layout()
        path = self._save(fig, "fig_onaxis.png")
        plt.close(fig)
        return path

    def _focal_profiles(self) -> Optional[Dict[str, Any]]:
        """Focal-plane radial profiles from the rz map row at z = F, with
        the ideal-lens profile on the same grid."""
        z = self.files.npz("verify_rzmap.npz")
        if z is None:
            return None
        r, zg = z["r0grid"], z["zgrid"]
        iz = int(np.argmin(np.abs(zg - self.d.F)))
        rs = RSPropagator.from_design(self.d)
        out: Dict[str, Any] = {"r": r, "I": {}, "I_ideal": {}, "maps": {}, "zgrid": zg}
        for lam in self.lams:
            key = nm_key("I", lam)
            if key not in z:
                continue
            out["maps"][lam] = z[key]
            out["I"][lam] = z[key][iz]
            out["I_ideal"][lam] = rs.ideal_profile(lam, r)
        return out

    def fig_psf(self, prof: Optional[Dict[str, Any]] = None) -> Optional[str]:
        prof = prof or self._focal_profiles()
        if prof is None:
            self.log("verify_rzmap.npz missing -- fig_psf skipped")
            return None
        plt = self._plt()
        m = self.files.metrics()
        n = len(prof["I"])
        ncol = min(5, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow), squeeze=False)
        for k, (lam, I) in enumerate(prof["I"].items()):
            ax = axes[k // ncol, k % ncol]
            Iid = prof["I_ideal"][lam]
            r = prof["r"]
            ax.plot(np.r_[-r[::-1], r], np.r_[Iid[::-1], Iid] / Iid.max(), color="#bbbbbb", lw=1.0,
                    label="ideal lens")
            ax.plot(np.r_[-r[::-1], r], np.r_[I[::-1], I] / Iid.max(), color="#e6550d", lw=1.4,
                    label="design (same scale)")
            ax.plot(np.r_[-r[::-1], r], np.r_[I[::-1], I] / I.max(), color="#e6550d", lw=0.8,
                    ls=":", label="design / own peak")
            i = int(np.argmin(np.abs(self.lams - lam)))
            ax.set_title("%d nm  FWHM %.2f um  S %.3f" % (int(lam * 1000), m["fwhm_um"][i],
                                                          m["strehl_like"][i]), fontsize=8)
            ax.set_xlabel("r (um)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.25)
            if k == 0:
                ax.set_ylabel("I / ideal peak")
                ax.legend(fontsize=7)
        for k in range(n, nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        fig.suptitle("MDL '%s': focal-plane PSFs at z = F vs the ideal lens (same aperture)"
                     % self.d.cfg.name, fontsize=10)
        fig.tight_layout()
        path = self._save(fig, "fig_psf.png")
        plt.close(fig)
        return path

    def fig_psf_2d(self, prof: Optional[Dict[str, Any]] = None) -> Optional[str]:
        prof = prof or self._focal_profiles()
        if prof is None:
            return None
        plt = self._plt()
        n = len(prof["I"])
        r = prof["r"]
        x = np.linspace(-r[-1], r[-1], 2 * r.size - 1)
        X, Y = np.meshgrid(x, x)
        RR = np.hypot(X, Y)
        peak = max(float(I.max()) for I in prof["I"].values())
        # one row, one spot per wavelength, common intensity scale
        fig, axes = plt.subplots(1, n, figsize=(max(6.0, 1.35 * n + 1.6), 2.6), squeeze=False)
        im = None
        for k, (lam, I) in enumerate(prof["I"].items()):
            ax = axes[0, k]
            img = np.interp(RR, r, I, right=0.0)
            im = ax.imshow(img / peak, extent=(x[0], x[-1], x[0], x[-1]), cmap=self.cmap,
                           vmin=0, vmax=1)
            ax.set_title("%d nm" % int(lam * 1000), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.subplots_adjust(left=0.02, right=0.955, top=0.74, bottom=0.04, wspace=0.08)
        if im is not None:
            cax = fig.add_axes([0.963, 0.06, 0.008, 0.66])
            fig.colorbar(im, cax=cax).set_label("I / max over lines", fontsize=8)
            cax.tick_params(labelsize=7)
        fig.suptitle("MDL '%s': focal spots at z = F (common scale, +/-%.0f um)"
                     % (self.d.cfg.name, r[-1]), fontsize=10)
        path = self._save(fig, "fig_psf_2d.png")
        plt.close(fig)
        return path

    def fig_rz_tiles(self, prof: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """I(r, z) tiles, ONE ROW, one tile per wavelength (paper Fig. 2e
        layout): shared z axis in absolute mm (F dashed), r mirrored to
        +/-r_max, each tile normalized to its own peak, one colour bar."""
        prof = prof or self._focal_profiles()
        if prof is None:
            self.log("verify_rzmap.npz missing -- fig_rz_tiles skipped")
            return None
        plt = self._plt()
        F = self.d.F
        r, zg = prof["r"], prof["zgrid"]
        n = len(prof["maps"])
        fig, axes = plt.subplots(1, n, figsize=(max(6.0, 1.35 * n + 1.6), 4.2),
                                 sharey=True, squeeze=False)
        im = None
        for k, (lam, M) in enumerate(prof["maps"].items()):
            ax = axes[0, k]
            tile = np.hstack([M[:, :0:-1], M])
            im = ax.imshow(tile / tile.max(), origin="lower", aspect="auto", cmap=self.cmap,
                           vmin=0, vmax=1,
                           extent=(-r[-1], r[-1], zg[0] / 1000, zg[-1] / 1000))
            ax.axhline(F / 1000, color="k" if self.cmap in ("OrRd", "Reds", "YlOrRd", "Greys")
                       else "w", lw=0.8, ls="--")
            ax.set_title("%d nm" % int(lam * 1000), fontsize=9)
            ax.set_xticks([-r[-1], 0, r[-1]])
            if k == 0:
                ax.set_xlabel("r (um)", fontsize=9)
                ax.set_ylabel("z (mm)", fontsize=9)
                ax.tick_params(labelsize=8)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="y", length=0)
        fig.subplots_adjust(left=0.05, right=0.955, top=0.80, bottom=0.16, wspace=0.12)
        if im is not None:
            cax = fig.add_axes([0.963, 0.16, 0.008, 0.64])
            fig.colorbar(im, cax=cax).set_label("I / per-tile max", fontsize=8)
            cax.tick_params(labelsize=7)
        fig.suptitle("Intensity I(r, z) around the focus -- %s (paper Fig. 2e convention, "
                     "F = %.3f mm dashed)" % (self.d.cfg.name, F / 1000), fontsize=10)
        path = self._save(fig, "fig_rz_tiles.png")
        plt.close(fig)
        return path

    def fig_metrics(self) -> Optional[str]:
        plt = self._plt()
        m = self.files.metrics()
        lam_nm = 1000 * np.asarray(m["lam_um"])
        F = float(m["F_um"])
        mtf = self.files.npz("verify_mtf.npz")
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        ax = axes[0, 0]
        ax.plot(lam_nm, np.asarray(m["z_peak_um"]) - F, "o-", color="#e6550d", ms=4, label="global peak")
        if "z_peak_tile_window_um" in m:
            ax.plot(lam_nm, np.asarray(m["z_peak_tile_window_um"]) - F, "s--", color="#3182bd", ms=3,
                    label="peak in tile window")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("focus offset z_peak - F (um)")
        ax.set_title("focal shift", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[0, 1]
        ax.plot(lam_nm, m["fwhm_um"], "o-", color="#e6550d", ms=4, label="design FWHM")
        ax.plot(lam_nm, np.asarray(m["lam_um"]) / (2 * self.d.na), "--", color="k", lw=1,
                label="diffraction limit lam/2NA")
        ax.set_ylabel("FWHM (um)")
        ax.set_title("spot size", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[0, 2]
        ax.plot(lam_nm, m["eff_3fwhm"], "o-", color="#e6550d", ms=4, label="eff (3xFWHM disc / incident)")
        ax.plot(lam_nm, m["strehl_like"], "s-", color="#3182bd", ms=3, label="Strehl-like (peak / ideal)")
        ax.set_ylabel("fraction")
        ax.set_title("efficiency", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[1, 0]
        ax.plot(lam_nm, m["strehl_shape"], "o-", color="#e6550d", ms=4)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("shape Strehl (window-normalized)")
        ax.set_title("core fidelity (paper Fig. 4f convention)", fontsize=9)
        ax = axes[1, 1]
        if "I_tilewin_over_global" in m:
            ax.plot(lam_nm, m["I_tilewin_over_global"], "o-", color="#e6550d", ms=4)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("I(main focus) / I(global peak)")
            ax.set_title("satellite check (1 = main focus is the brightest)", fontsize=9)
        ax = axes[1, 2]
        if mtf is not None:
            ax.plot(1000 * mtf["lam_um"], mtf["mtf_quality"], "o-", color="#e6550d", ms=4,
                    label="per line vs own limit")
            ax.axhline(float(mtf["mtf_quality_poly"]), color="k", ls="--", lw=1,
                       label="polychromatic %.3f" % float(mtf["mtf_quality_poly"]))
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("MTF area ratio")
            ax.set_title("MTF quality (mtf_verify)", fontsize=9)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "run mtf_verify.py for the MTF panel", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        for ax in axes.ravel():
            ax.set_xlabel("wavelength (nm)")
            ax.grid(True, alpha=0.25)
        fig.suptitle("MDL '%s': verification metrics per wavelength (RS %s, J objective %.4f)"
                     % (self.d.cfg.name, m.get("rs_ring_quadrature", "?"), m.get("J_objective", float("nan"))),
                     fontsize=10)
        fig.tight_layout()
        path = self._save(fig, "fig_metrics.png")
        plt.close(fig)
        return path

    # -- driver ----------------------------------------------------------------------
    def run(self, script_path: Optional[str] = None) -> List[str]:
        self.snapshot(script_path)
        self.d.describe(self.log, m_line=False)
        self.log("image colormap %s (fig_psf_2d, fig_rz_tiles), line colours %s"
                 % (self.cmap, self.line_cmap))
        self.log("reading %s" % (os.path.relpath(os.path.dirname(self.files.path("verify_metrics.json") or ""))))
        self.fig_onaxis()
        prof = self._focal_profiles()
        self.fig_psf(prof)
        self.fig_psf_2d(prof)
        self.fig_rz_tiles(prof)
        self.fig_metrics()
        self.log("figures in %s: %s" % (os.path.relpath(self.out_rs), ", ".join(self.written)))
        return self.written
