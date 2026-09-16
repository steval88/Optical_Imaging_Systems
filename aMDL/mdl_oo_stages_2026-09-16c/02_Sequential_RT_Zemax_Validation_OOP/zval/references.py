"""Numerics shared with 02_validation_rs, kept IDENTICAL to the legacy
script (2026-09-08.04) so that every number is comparable:

* rs_tiles          I(r0, z) of a ring field by the J0-reduced RS-I with
                    the ring-MIDPOINT quadrature -- the same kernel and
                    quadrature on both sides of the rz comparison, so the
                    Zemax-field tiles and the RS tiles differ only by
                    whose exit field goes in.
* FineReference     focal-plane |U(r0, F)|^2 of the staircase on a
                    sub-ring grid (0.125 um): the physical reference for
                    the wave engines (phase + cell-mean transmission).
* radial_profile    azimuthal average of a 2-D irradiance grid about the
                    up-sampled peak (or about the axis).
* fwhm_of           first half-crossing FWHM of a radial profile.
* hankel_mtf        MTF = |Hankel-0 of the PSF| / DC on a frequency grid.
* RsReference       loader of rs/verify_rzmap.npz + rs/verify_mtf.npz +
                    the quadrature tag of rs/verify_metrics.json.
"""
import json
import os

import numpy as np
from scipy.ndimage import zoom as _zoom
from scipy.special import j0, j1

from .settings import n_resist


def rs_tiles(U, rho, delta, lam, zgrid, r0grid, quad="midpoint"):
    """I(r0, z) of exit field U(rho_i) -- the run_verify.rs_psf kernel
    (RS-I, J0-reduced) with the SAME ring quadrature as the rs/ tiles it
    is compared against: quad = "midpoint" (one kernel sample per ring,
    pre-2026-09-08 files) or "sinc" (analytic ring integral of the
    kernel phase ramp, factor sinc(delta rho_i / (lam rbar_i)), the
    run_verify default since 2026-09-08). The tag is read from
    rs/verify_metrics.json by RsReference, never assumed."""
    k = 2.0 * np.pi / lam
    out = np.empty((zgrid.size, r0grid.size))
    for iz, z in enumerate(zgrid):
        for ir, r0 in enumerate(r0grid):
            rb = np.sqrt(z * z + rho * rho + r0 * r0)
            w = rho if quad == "midpoint" else \
                rho * np.sinc(delta * rho / (lam * rb))
            s = np.sum(U * j0(k * rho * r0 / rb)
                       * np.exp(1j * k * rb) / rb ** 2 * w)
            out[iz, ir] = np.abs((z / (1j * lam)) * 2.0 * np.pi
                                 * delta * s) ** 2
    return out


def radial_profile(arr, dx_um, r_max_um, fine_um=0.05, about_axis=False):
    """Azimuthal average around the TRUE peak: (r_um, prof, peak_xy_um
    relative to the grid centre).

    MEASURED 2026-09-07: annular binning at the grid pitch inflates the
    FWHM by 8-10 % at ~5 samples per FWHM (and the axis of an even grid
    sits on a pixel corner). The sub-image is up-sampled (cubic) to
    fine_um before the peak search and the binning: unbiased down to
    ~3 samples per FWHM (1.001-1.005 on a sampled Airy pattern)."""
    ny, nx = arr.shape
    iy, ix = np.unravel_index(int(np.argmax(arr)), arr.shape)
    if about_axis:
        iy, ix = ny // 2, nx // 2
    half = int(np.ceil(r_max_um / dx_um)) + 4
    y0, y1 = max(iy - half, 0), min(iy + half + 1, ny)
    x0, x1 = max(ix - half, 0), min(ix + half + 1, nx)
    sub = np.nan_to_num(arr[y0:y1, x0:x1])
    up = max(int(round(dx_um / fine_um)), 1)
    if up > 1:
        subf = _zoom(sub, up, order=3, grid_mode=True, mode="nearest")
        dxf = dx_um / up
    else:
        subf, dxf = sub, dx_um
    jy, jx = np.unravel_index(int(np.argmax(subf)), subf.shape)
    if about_axis:
        jy = int(round((ny / 2.0 - y0) * up - 0.5 * (up - 1)))
        jx = int(round((nx / 2.0 - x0) * up - 0.5 * (up - 1)))
        jy = min(max(jy, 0), subf.shape[0] - 1)
        jx = min(max(jx, 0), subf.shape[1] - 1)
    yy, xx = np.mgrid[0:subf.shape[0], 0:subf.shape[1]]
    rr = np.hypot((xx - jx) * dxf, (yy - jy) * dxf)
    kk = np.floor(rr / dxf).astype(int)
    nb = int(np.ceil(r_max_um / dxf)) + 1
    sel = kk < nb
    s = np.bincount(kk[sel], weights=subf[sel], minlength=nb)
    c = np.bincount(kk[sel], minlength=nb)
    prof = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    r_um = (np.arange(nb) + 0.5) * dxf
    r_um[0] = 0.0
    pk_x = (x0 + (jx + 0.5) / up - 0.5 - nx / 2.0) * dx_um
    pk_y = (y0 + (jy + 0.5) / up - 0.5 - ny / 2.0) * dx_um
    return r_um, prof, (pk_x, pk_y)


def fwhm_of(r, I):
    """2 x first half-maximum crossing outward from r = 0; NaN when the
    profile has no on-axis half crossing (annular pattern)."""
    if I.size < 3 or I.max() <= 0:
        return float("nan")
    half = 0.5 * I[0] if I[0] >= I.max() else 0.5 * I.max()
    idx = np.where(I < half)[0]
    if idx.size == 0 or idx[0] == 0:
        return float("nan")
    i1 = idx[0]
    r_half = r[i1 - 1] + (r[i1] - r[i1 - 1]) * \
        (I[i1 - 1] - half) / max(I[i1 - 1] - I[i1], 1e-30)
    return float(2.0 * r_half)


def hankel_mtf(I, r, f_um):
    """MTF on the frequency grid f_um [cycles/um] from a radial PSF."""
    w = I * r
    dc = 2 * np.pi * np.trapezoid(w, r)
    if dc <= 0:
        return np.zeros(f_um.size)
    out = np.array([2 * np.pi * np.trapezoid(w * j0(2 * np.pi * ff * r), r)
                    for ff in f_um])
    return np.abs(out) / dc


def airy_profile(r_um, lam_um, na):
    x = 2 * np.pi * na * np.maximum(r_um, 1e-9) / lam_um
    return (2 * j1(x) / x) ** 2


def encircled_fraction(arr, dx_um, dy_um, r_um, prof, fwhm):
    """power within a disc of DIAMETER 3 x FWHM (from the radial
    profile) / total power on the grid."""
    p_grid = float(arr.sum())
    if not np.isfinite(fwhm):
        return float("nan")
    sel = r_um <= 1.5 * fwhm
    p_core = float(2 * np.pi * np.trapezoid(prof[sel] * r_um[sel], r_um[sel]))
    return p_core / max(p_grid * dx_um * dy_um, 1e-30)


def corr_on(ref_r, ref_I, r_um, prof):
    """Pearson r of the engine profile (interpolated to ref_r) against
    the reference, both peak-normalized; NaN if either is empty."""
    p = np.interp(ref_r, r_um, prof)
    if p.max() <= 0 or ref_I.max() <= 0:
        return float("nan")
    return float(np.corrcoef(p / p.max(), ref_I / ref_I.max())[0, 1])


class FineReference:
    """Focal-plane RS profiles of the staircase on a sub-ring grid.

    Across a flat 2 um ring the RS kernel phase ramps by Delta*rho/r =
    0.2 um of OPL at the rim (half a wave at 400 nm); one kernel sample
    per ring cannot see it. Here the identical kernel on a step_um grid,
    h(rho) piecewise constant per ring: the staircase-integrated
    reference the wave engines are judged against."""

    def __init__(self, table, F_um, r_max_um=30.0, step_um=0.125):
        self.step_um = float(step_um)
        self.F_um = float(F_um)
        self.rf = np.arange(0.0, table.N * table.delta_um, self.step_um) \
            + 0.5 * self.step_um
        self.hf = table.h_um[np.minimum((self.rf / table.delta_um).astype(int),
                                        table.N - 1)]
        self.r_um = np.arange(0.0, max(r_max_um, 30.0) + 5.0, 0.05)
        self.I = {}

    def kernel(self, E0, lam):
        k = 2.0 * np.pi / lam
        U = np.empty(self.r_um.size, dtype=complex)
        rf, F = self.rf, self.F_um
        for ir, r0 in enumerate(self.r_um):
            rb = np.sqrt(F * F + rf * rf + r0 * r0)
            U[ir] = (F / (1j * lam)) * 2 * np.pi * self.step_um * np.sum(
                E0 * j0(k * rf * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 * rf)
        return np.abs(U) ** 2

    def compute(self, wavelengths_um):
        for lam in wavelengths_um:
            k = 2.0 * np.pi / lam
            E0 = np.exp(1j * k * (n_resist(lam) - 1.0) * self.hf)
            self.I[int(round(lam * 1000))] = self.kernel(E0, lam)
        return self

    def store_into(self, store):
        store["rsf_r_um"] = self.r_um
        for key, I in self.I.items():
            store["rsf_I_%d" % key] = I


class RsReference:
    """rs/verify_rzmap.npz (focal slices), rs/verify_mtf.npz (MTF on its
    frequency grid) and the ring-quadrature tag written by run_verify
    into rs/verify_metrics.json -- read back, never assumed."""

    def __init__(self, design):
        self.rz = self.mtf = None
        self.quad = "midpoint"                # pre-09-08 files carry no tag
        self.quad_src = "no rs/verify_metrics.json"
        p = design.rs_path("verify_rzmap.npz")
        if p:
            self.rz = np.load(p)
            self.rz_path = p
        p = design.rs_path("verify_mtf.npz")
        if p:
            self.mtf = np.load(p)
        p = design.rs_path("verify_metrics.json")
        if p:
            try:
                vm = json.load(open(p))
                self.quad = str(vm.get("rs_ring_quadrature", "midpoint"))
                self.quad_src = ("rs_ring_quadrature key"
                                 if "rs_ring_quadrature" in vm
                                 else "no key -> pre-2026-09-08 file")
            except Exception as exc:
                self.quad_src = "unreadable (%s)" % exc

    @property
    def tag(self):
        return "mid" if self.quad == "midpoint" else self.quad

    def focal_slice(self, key, F_um):
        """(r0grid, I(r0, z=F)) of line <key> nm, or None."""
        if self.rz is None or "I_%d" % key not in self.rz.files:
            return None
        zg = self.rz["zgrid"]
        izF = int(np.argmin(np.abs(zg - F_um)))
        return self.rz["r0grid"], self.rz["I_%d" % key][izF, :]

    def mtf_numbers(self, key, na, lam, mtf_engine, mtf_zemax=None):
        """quality = area to the OWN diffraction limit's cutoff / area
        of the limit; MTF50; largest deviations on the RS frequency
        grid. Returns dict or None when verify_mtf.npz is absent."""
        if self.mtf is None:
            return None
        fl = self.mtf["f_lppmm"]
        fc = 2.0 * na / (lam * 1e-3)                       # lp/mm
        dl = self.mtf["MTFdl_%d" % key] if "MTFdl_%d" % key in self.mtf.files else None
        rs = self.mtf["MTF_%d" % key] if "MTF_%d" % key in self.mtf.files else None
        selc = fl <= fc

        def q(m):
            if m is None or dl is None:
                return float("nan")
            return float(np.trapezoid(m[selc], fl[selc])
                         / max(np.trapezoid(dl[selc], fl[selc]), 1e-30))

        def f50(m):
            if m is None:
                return float("nan")
            j = np.where(m < 0.5)[0]
            return float(fl[j[0]]) if j.size else float("nan")

        zm = None
        if mtf_zemax is not None:
            zm = np.interp(fl, mtf_zemax[0], mtf_zemax[1])
        dev_z = float(np.nanmax(np.abs(zm - mtf_engine)[selc])) \
            if zm is not None else float("nan")
        dev_rs = float(np.nanmax(np.abs(rs - mtf_engine)[selc])) \
            if rs is not None else float("nan")
        return dict(cutoff=fc, q_engine=q(mtf_engine), q_zemax=q(zm),
                    q_rs=q(rs), f50_engine=f50(mtf_engine), f50_zemax=f50(zm),
                    f50_rs=f50(rs), dev_zemax=dev_z, dev_rs=dev_rs,
                    fl=fl, rs=rs, dl=dl)
