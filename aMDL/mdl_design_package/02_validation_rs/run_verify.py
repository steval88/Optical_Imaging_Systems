"""
run_verify.py -- STAGE 2a: physical verification of a designed MDL by
scalar Rayleigh-Sommerfeld diffraction of the actual staircase profile.

This is the same validation approach as the reference paper (Xiao et
al. [4], Fig. 2e), whose ref. 46 for it is Goodman [1]. Usage (from the
package root):

    python 02_validation_rs\\run_verify.py runs\\<run_folder> [other_m.npy]

All geometry, wavelengths and numerical grids are read from the run
folder's config.json (written by 01_design/run_MDL_design.py); nothing
is hardcoded here. The optional second argument verifies a different
design vector against the same configuration.

===========================================================================
PHYSICAL MODEL -- three steps, each documented against the literature
===========================================================================

STEP 1: exit field via the thin-element approximation (TEA)
---------------------------------------------------------------------------
The lens is a staircase of N concentric rings, ring i of width DELTA
covering [i*DELTA, (i+1)*DELTA) with height h_i = m_i * dh in resist of
index n(lambda). For a unit-amplitude plane wave at normal incidence the
field just after the relief is modeled as a pure phase screen

    U0(rho) = exp[ i k (n(lambda) - 1) h(rho) ],     k = 2*pi/lambda,

i.e. each ray accumulates the optical-path difference (n-1)h of the
resist column it crosses; intra-relief diffraction is neglected. This is
the standard phase-transformation model of Goodman [1] Ch. 5 ("the thin
lens as a phase transformation"; 3rd ed. Sec. 5.1) and the universal
model for multilevel diffractive optics since Swanson [6]. Its accuracy
for THIS geometry (H ~ 15 um relief, >= 2 um rings, NA ~ 0.1) is
quantified numerically by bpm_validate.py in this folder, which
propagates the field THROUGH the finite-thickness staircase with a
beam-propagation method and finds 1-3 % agreement in focal efficiency.

STEP 2: propagation by the first Rayleigh-Sommerfeld solution (RS-I)
---------------------------------------------------------------------------
The field at observation point P0 = (r0, z) behind a planar aperture
carrying the field U0(P1) is (Goodman [1], 1968 ed. pp. 38-53 = Sec.
3-5/3-7 "The Rayleigh-Sommerfeld formulation"; 3rd ed. Sec. 3.5,
first Rayleigh-Sommerfeld solution U_I; equivalently Born & Wolf [2]
Sec. 8.11):

    U(P0) = (1 / i*lambda) * INT_aperture U0(P1)
              * exp(i k r01) / r01 * cos(theta)  dS               (RS-I)

with r01 = |P0 - P1| and the obliquity factor cos(theta) = z / r01
(angle between the aperture normal and P0-P1). Substituting cos(theta)
gives the kernel actually coded below:

    U(P0) = (z / i*lambda) * INT U0(P1) * exp(i k r01) / r01^2  dS.

RS-I is an EXACT solution of the scalar Helmholtz equation for a plane
screen with Kirchhoff boundary values -- no Fresnel or paraxial
approximation is made at this step. (For the numerical direct-
integration approach and its accuracy vs. FFT-based propagators see
Shen & Wang [3].)

STEP 3: axisymmetric (Bessel) reduction of the RS-I integral
---------------------------------------------------------------------------
Both the lens and the illumination are rotationally symmetric, so the
2-D aperture integral reduces to 1-D. In polar aperture coordinates
(rho, phi), the exact distance to the observation point (r0, z) is

    r01 = sqrt( z^2 + rho^2 + r0^2 - 2*rho*r0*cos(phi) ).

Writing rbar = sqrt(z^2 + rho^2 + r0^2) and expanding to first order in
the cross term (valid for rho*r0 << rbar^2),

    r01 ~ rbar - rho*r0*cos(phi) / rbar          (phase),
    1/r01^2 ~ 1/rbar^2                           (amplitude),

the azimuthal integral becomes the standard Bessel identity
INT_0^{2pi} exp(-i a cos(phi)) dphi = 2*pi*J0(a)  (DLMF 10.9.2 /
Abramowitz & Stegun 9.1.21 [5]; the same Fourier-Bessel/Hankel
reduction as Goodman [1] Sec. 2.1.5), giving the formula implemented in
rs_psf():

    U(r0, z) = (z / i*lambda) * SUM_i U0_i * J0(k*rho_i*r0/rbar_i)
               * exp(i k rbar_i) / rbar_i^2 * (2*pi*rho_i*DELTA)

ACCURACY OF THE REDUCTION. The neglected next-order phase term is
bounded by k*(rho*r0)^2 / (2*rbar^3). Worst case for the S3 geometry
(rho = R = 5.12 mm, r0 = 30 um, z = F = 50.94 mm, lambda = 0.4 um):
~1.4e-3 rad -- negligible. ON AXIS (r0 = 0, used by rs_onaxis()) the
reduction is EXACT: no cross term exists and J0(0) = 1.

DISCRETIZATION. Midpoint-rule quadrature over the physical rings: one
sample per ring at its center rho_i = (i+1/2)*DELTA, area weight
2*pi*rho_i*DELTA. The field U0 is exactly piecewise-constant per ring
(staircase), and this is the SAME discretization used by the design FOM
in mdl_core.MDLProblem (tables G, L), so design and verification are
mutually consistent by construction; it was additionally cross-checked
against Zemax Huygens-PSF results (which agree with the RS metrics --
see the project findings doc).

===========================================================================
METRICS reported per verification wavelength
===========================================================================
* z_peak_um   : argmax_z of the on-axis intensity |U(0,z)|^2 over the
                scan F +/- verify_z_span_um -- the focal-shift /
                achromaticity check (the paper's Fig. 2e quantity).
* fwhm_um     : full width at half maximum of |U(r0,F)|^2 (first
                half-crossing of the radial profile).
* eff_3fwhm   : focusing efficiency = power within a disc of DIAMETER
                3x FWHM around the focus / total incident power
                (pi*R^2 for unit amplitude). This is the paper's own
                definition [4] ("power within a diameter of 3 times
                the FWHM"; radius 1.5x FWHM below) -- see also
                Engelberg & Levy [7] on why this convention matters.
* strehl_like : on-axis peak intensity / peak of an IDEAL lens of the
                same aperture and focal length, computed with the same
                quadrature (discretization bias cancels in the ratio).
                "Strehl-like" because the reference is the perfect
                hyperbolic phase at that wavelength, not a best-fit
                sphere. EFFICIENCY-INCLUSIVE: for a diffraction-
                limited core, strehl_like ~ encircled efficiency, so
                low values mean halo/other-order loss, NOT
                aberration. Family of the paper's Supplementary S3-6
                "Normalized Strehl ratio" (S1-S5 avg 0.66/0.49/0.28/
                0.14/0.05) -- theirs normalizes by power reaching the
                focal plane, ours (stricter) by total incident power.
* strehl_shape: the paper's Fig. 4f convention (ref. 43 of [4]) --
                Strehl from the PSF normalized to the power CAPTURED
                IN THE MEASUREMENT WINDOW (the CCD analogue:
                r <= verify_r_max_um):
                  S = [max I / P_win] / [max I_ideal / P_win,ideal],
                  P_win = INT_window I(r) 2 pi r dr,
                ideal = the same-quadrature ideal-lens PSF. SHAPE-
                ONLY: diffraction efficiency cancels; this answers
                "is the focal spot diffraction-limited in form?".
                Window-dependent (a wider window admits more halo and
                lowers it); the paper's CCD frames are ~ +/-10-15 um.
                Compare strehl_shape to Fig. 4f and strehl_like to
                the Supplementary Normalized Strehl -- NEVER across
                conventions.
Also reported: J on the design objective, on the continuous band and on
the verification comb (arithmetic-mean convention in all three, so runs
with different fom_mode remain comparable), and the DLL ring table is
re-exported so it always matches the vector actually verified.

Outputs (per-solver layout, 2026-08-28 -- each solver keeps its data
and figures in its own subfolder of the run folder; zemax/ set the
precedent):
    rs/verify_metrics.json, rs/verify_onaxis.npz, rs/verify_rzmap.npz
    mdl_rings_<n>.txt        (run ROOT: fabrication-facing artifact,
                              consumed by the Zemax and GDS stages)
    scripts/run_verify.py    (snapshot as run)
Older run folders with the verify_* files at the top level are still
read transparently by make_plots.py (subfolder first, root fallback).

verify_rzmap.npz holds full I(r, z) intensity maps per verification
wavelength (the raw data of the paper's Fig. 2e / Fig. 4a tiles),
computed with rs_psf on an (r, z) grid around the design focus.
Optional config keys (defaults in parentheses): rzmap_r_max_um (20),
rzmap_r_points (41), rzmap_z_span_um (1000), rzmap_z_points (121).

===========================================================================
REFERENCES
===========================================================================
[1] J. W. Goodman, Introduction to Fourier Optics. 1st ed., McGraw-Hill
    (1968), pp. 38-53 (scalar diffraction; the exact source cited as
    ref. 46 by Xiao et al. [4]); 3rd ed., Roberts & Company (2005):
    Sec. 3.5 (Rayleigh-Sommerfeld formulation), Sec. 2.1.5
    (Fourier-Bessel transform), Sec. 5.1 (thin phase transformations).
[2] M. Born & E. Wolf, Principles of Optics, 7th ed., Cambridge
    University Press (1999), Sec. 8.11 (Rayleigh-Sommerfeld
    diffraction integrals).
[3] F. Shen & A. Wang, "Fast-Fourier-transform based numerical
    integration method for the Rayleigh-Sommerfeld diffraction
    formula," Appl. Opt. 45, 1102-1110 (2006). (Direct-integration
    numerics and accuracy analysis of exactly this integral.)
[4] Y. Xiao et al., "Large-scale achromatic flat lens by light
    frequency-domain coherence optimization," Light Sci. Appl. 11, 323
    (2022). (The design method being reproduced; RS validation in
    Fig. 2e cites [1] as its ref. 46; measured Strehl in Fig. 4f and
    Supplementary S3-6.)
[5] NIST DLMF Eq. 10.9.2 (https://dlmf.nist.gov/10.9), equivalently
    M. Abramowitz & I. A. Stegun, Handbook of Mathematical Functions,
    Eq. 9.1.21: the integral representation of J0.
[6] G. J. Swanson, "Binary Optics Technology: The Theory and Design of
    Multi-level Diffractive Optical Elements," MIT Lincoln Laboratory
    Technical Report 854 (1989). (TEA design model for multilevel
    diffractive lenses; ref. 36 of [4].)
[7] J. Engelberg & U. Levy, "Standardizing flat lens characterization,"
    Nat. Photonics 16, 171-173 (2022). (Efficiency-metric conventions;
    ref. 47 of [4].)
"""
import json
import os
import shutil
import sys
import time

import numpy as np
from numpy import pi
from scipy.special import j0

# package root = parent of this stage folder; mdl_core.py lives there
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from mdl_core import MDLProblem

T0 = time.time()


def log(s):
    print("[%6.1fs] %s" % (time.time() - T0, s), flush=True)


# ---- load run configuration ----------------------------------------------
if len(sys.argv) < 2:
    raise SystemExit("usage: python run_verify.py runs/<run_folder> "
                     "[optional/other_m.npy]")
run_dir = sys.argv[1].rstrip("/\\")
if not os.path.exists(os.path.join(run_dir, "config.json")):
    # also accept a path relative to the package root (any CWD)
    alt = os.path.join(PKG_ROOT, run_dir)
    if os.path.exists(os.path.join(alt, "config.json")):
        run_dir = alt
cfg_path = os.path.join(run_dir, "config.json")
if not os.path.exists(cfg_path):
    # typo helper: suggest the closest existing run folders
    hint = ""
    runs_root = os.path.join(PKG_ROOT, "runs")
    if os.path.isdir(runs_root):
        import difflib
        cands = sorted(os.listdir(runs_root))
        close = difflib.get_close_matches(os.path.basename(run_dir),
                                          cands, n=3, cutoff=0.5)
        if close:
            hint = "\n  did you mean: " + "  ".join(
                os.path.join("runs", c) for c in close)
        elif cands:
            hint = "\n  most recent run folders: " + "  ".join(
                os.path.join("runs", c) for c in cands[-3:])
    raise SystemExit("no config.json in %r -- is this a run folder made "
                     "by run_MDL_design.py?%s" % (run_dir, hint))
cfg = json.load(open(cfg_path))
der = cfg["derived"]

D, na, F = cfg["diameter_um"], der["na"], der["focal_um"]
lmin, lmax = cfg["lam_min_um"], cfg["lam_max_um"]
lams_obj = cfg["target_wavelengths_um"]        # None = continuous objective
lam_list = np.asarray(cfg["verify_wavelengths_um"], dtype=float)

m_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir,
                                                            "m_final.npy")
m = np.load(m_file)

# ---- DESIGN STATE (module-level, read-only for the rest of the run) -----
# The optimized design enters here and ONLY here: m (gray levels from
# m_final.npy, the Search/Smooth/Gradient/HJA output of stage 1) and the
# geometry rebuilt from config. The propagator functions below
# (exit_field, rs_onaxis, rs_psf, ideal_peak) deliberately CLOSE OVER
# these variables instead of taking them as arguments: within one run
# they are constants, they are shared by three different analysis
# sections, and closing over a single copy makes it impossible to mix
# the heights of one design with the radii of another. Only quantities
# that vary between calls (wavelength, plane z, observation grid) are
# function parameters. (A reusable library would wrap this state in a
# class; this is a run-once pipeline stage.)
prob = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"],
                  cfg["h_max_um"], cfg["dh_um"],
                  n_wavelengths=cfg["n_wavelengths"])
if m.size != prob.N:
    raise SystemExit("%s has %d rings; config expects %d"
                     % (m_file, m.size, prob.N))
h = m * prob.dh                      # ring heights h_i = m_i * dh   [um]
rho = prob.rho                       # ring center radii (i+1/2)*DELTA [um]
drho = prob.delta                    # ring width DELTA               [um]
R = prob.R                           # aperture radius                [um]

# snapshot this script into the run folder (traceability)
os.makedirs(os.path.join(run_dir, "scripts"), exist_ok=True)
shutil.copy2(__file__, os.path.join(run_dir, "scripts",
                                    os.path.basename(__file__)))

# per-solver output subfolder (see header "Outputs"): everything this
# solver produces lands in rs/, mirroring zemax/ for the Zemax stage
out_rs = os.path.join(run_dir, "rs")
os.makedirs(out_rs, exist_ok=True)

log("run: %s" % run_dir)
log("design '%s': D=%.2f mm, F=%.2f mm, NA=%.4f | %d rings x %.2f um, "
    "H=%.1f um (%d levels x %g nm)"
    % (cfg["name"], D / 1000, F / 1000, na, prob.N, prob.delta,
       cfg["h_max_um"], prob.M, cfg["dh_um"] * 1000))
log("objective: %s, fom_mode=%s | band %.0f-%.0f nm | verifying at %d "
    "wavelengths"
    % ("discrete comb (%d lines)" % len(lams_obj) if lams_obj is not None
       else "continuous band", cfg.get("fom_mode", "mean"),
       lmin * 1000, lmax * 1000, lam_list.size))
log("design vector: %s (levels 0..%d used, h_max=%.2f um)"
    % (m_file, int(m.max()), float(h.max())))
log("J_continuous(alias-safe, Nw=%d) = %.4f"
    % (cfg["n_wavelengths"], prob.fom(m)))


def exit_field(lam):
    """STEP 1 -- thin-element exit field U0(rho_i) (header STEP 1).

    U0_i = exp[i k (n(lam)-1) h_i] : unit plane wave picks up the
    optical-path difference of the resist column at each ring.
    Refs: Goodman [1] Sec. 5.1; Swanson [6]. TEA error quantified by
    bpm_validate.py (1-3 % in focal efficiency at this geometry).
    """
    n = float(prob.n_func(lam))
    k = 2 * pi / lam
    return np.exp(1j * k * (n - 1.0) * h)


def rs_onaxis(lam, zgrid):
    """STEP 2 -- exact on-axis RS-I field U(0, z) for each z in zgrid.

    On axis (r0 = 0) the axisymmetric reduction is EXACT (header
    STEP 3): rbar = sqrt(z^2 + rho^2) is the true source-observer
    distance and J0(0) = 1, so this is the unapproximated RS-I
    integral (Goodman [1] Sec. 3.5) evaluated with the midpoint ring
    quadrature:

        U(0,z) = (z/(i lam)) SUM_i U0_i e^{ik rbar_i}/rbar_i^2
                 * 2 pi rho_i DELTA
    """
    E0 = exit_field(lam)
    k = 2 * pi / lam
    out = np.empty(zgrid.size, dtype=complex)
    for iz, z in enumerate(zgrid):
        rb = np.sqrt(z * z + rho * rho)          # exact r01 at r0 = 0
        integ = E0 * np.exp(1j * k * rb) / rb ** 2 * rho
        out[iz] = (z / (1j * lam)) * 2 * pi * drho * np.sum(integ)
    return out


def rs_psf(lam, z, r0grid):
    """STEPS 2+3 -- RS-I field U(r0, z) on the plane z, near-axis form.

    Axisymmetric Bessel reduction of RS-I (header STEP 3):

        U(r0,z) = (z/(i lam)) SUM_i U0_i J0(k rho_i r0 / rbar_i)
                  * e^{ik rbar_i}/rbar_i^2 * 2 pi rho_i DELTA,
        rbar_i  = sqrt(z^2 + rho_i^2 + r0^2).

    Neglected phase term <= k (rho r0)^2/(2 rbar^3): ~1.4e-3 rad worst
    case at the S3 geometry for r0 <= 30 um (see header). J0 identity:
    DLMF 10.9.2 [5].
    """
    E0 = exit_field(lam)
    k = 2 * pi / lam
    out = np.empty(r0grid.size, dtype=complex)
    for ir, r0 in enumerate(r0grid):
        rb = np.sqrt(z * z + rho * rho + r0 * r0)
        integ = E0 * j0(k * rho * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 \
            * rho
        out[ir] = (z / (1j * lam)) * 2 * pi * drho * np.sum(integ)
    return out


def ideal_peak(lam):
    """Reference peak amplitude of an IDEAL lens (same R, F, lam).

    The ideal element carries the exact hyperbolic phase
    exp[-ik(sqrt(rho^2+F^2)-F)] which makes the RS-I integrand phase
    stationary at the focus; its on-axis focal amplitude, computed with
    the SAME quadrature as rs_onaxis, normalizes the strehl_like
    metric (discretization bias cancels in the ratio).
    """
    k = 2 * pi / lam
    rb = np.sqrt(F * F + rho * rho)
    integ = np.exp(-1j * k * (rb - F)) * np.exp(1j * k * rb) / rb ** 2 * rho
    return abs((F / (1j * lam)) * 2 * pi * drho * np.sum(integ))


def ideal_profile(lam, r0grid):
    """Focal-plane PSF |U_ideal(r0, F)|^2 of the IDEAL lens (same R,
    F, quadrature): the exact hyperbolic phase propagated by the same
    J0-reduced RS-I as rs_psf. Reference for the strehl_shape metric
    (paper Fig. 4f convention -- see header METRICS): computing the
    ideal PSF with the identical quadrature makes discretization bias
    cancel in the shape-Strehl ratio, exactly as ideal_peak does for
    strehl_like."""
    k = 2 * pi / lam
    rb0 = np.sqrt(rho * rho + F * F)
    E0 = np.exp(-1j * k * (rb0 - F))
    out = np.empty(r0grid.size, dtype=complex)
    for ir, r0 in enumerate(r0grid):
        rb = np.sqrt(F * F + rho * rho + r0 * r0)
        integ = E0 * j0(k * rho * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 \
            * rho
        out[ir] = (F / (1j * lam)) * 2 * pi * drho * np.sum(integ)
    return np.abs(out) ** 2


# ---- on-axis scans (focal shift / achromaticity, paper Fig. 2e) ----------
results = {"run_dir": run_dir, "m_file": m_file,
           "lam_um": lam_list.tolist(), "F_um": F}
zgrid = np.linspace(F - cfg["verify_z_span_um"],
                    F + cfg["verify_z_span_um"], cfg["verify_z_points"])
onax = {}
log("[1/4] on-axis scans |U(0,z)|^2: z = F +/- %.2f mm, %d planes "
    "(exact RS-I on axis; feeds fig_onaxis*.png + z_peak metric)"
    % (cfg["verify_z_span_um"] / 1000, zgrid.size))
# NOTE on windows: z_peak_um is the GLOBAL argmax over the full scan
# (F +/- verify_z_span_um). The [3/4] tiles only cover the narrower
# rzmap window, so when a SATELLITE focus outside the tile window is
# brighter on axis than the main focus, the two stages report
# different z. Both use the identical RS-I integral (rs_psf at r0=0
# reduces exactly to rs_onaxis: J0(0)=1); only the search window
# differs. z_peak_tile_window_um / the I_main/I_sat ratio below make
# the comparison explicit.
tile_span = cfg.get("rzmap_z_span_um", 1000.0)
for lam in lam_list:
    I = np.abs(rs_onaxis(lam, zgrid)) ** 2
    onax[lam] = I
    ig = int(np.argmax(I))
    zpk = float(zgrid[ig])
    results.setdefault("z_peak_um", []).append(zpk)
    win = np.where(np.abs(zgrid - F) <= tile_span)[0]
    iw = win[int(np.argmax(I[win]))]
    zpk_w = float(zgrid[iw])
    ratio = float(I[iw] / I[ig])           # 1.0 = main focus IS global peak
    results.setdefault("z_peak_tile_window_um", []).append(zpk_w)
    results.setdefault("I_tilewin_over_global", []).append(ratio)
    if ratio > 0.9999:
        log("  lam=%.2f um: peak at z=%.3f mm (offset %+0.0f um from F)"
            % (lam, zpk / 1000, zpk - F))
    else:
        log("  lam=%.2f um: main focus z=%.3f mm (in tile window); "
            "brighter SATELLITE at z=%.3f mm (%+0.0f um), "
            "I_main/I_sat=%.2f"
            % (lam, zpk_w / 1000, zpk / 1000, zpk - F, ratio))
log("[1/4] on-axis scans done")

# ---- PSF metrics at the design focal plane -------------------------------
r0grid = np.linspace(0.0, cfg["verify_r_max_um"], cfg["verify_r_points"])
psf_metrics = {"fwhm_um": [], "eff_3fwhm": [], "strehl_like": [],
               "strehl_shape": [], "onax_I_at_F": []}
log("[2/4] focal-plane PSFs |U(r,F)|^2: r = 0..%g um, %d points "
    "(feeds fig_metrics.png)"
    % (cfg["verify_r_max_um"], r0grid.size))
log("      columns: FWHM | dl = diffraction limit lam/2NA | eff = power "
    "in 3xFWHM-diameter disc / total incident | S = peak vs ideal lens "
    "(efficiency-inclusive) | Sshape = window-normalized shape Strehl "
    "(paper Fig. 4f convention; efficiency cancels)")
for lam in lam_list:
    E = rs_psf(lam, F, r0grid)
    I = np.abs(E) ** 2
    Ipk = I[0] if I[0] == I.max() else I.max()
    # FWHM: radius of the first crossing below half the peak, doubled.
    half = Ipk / 2
    idx = np.where(I < half)[0]
    fwhm = 2 * r0grid[idx[0]] if idx.size else np.nan
    # focusing efficiency, paper convention [4],[7]: power inside a disc
    # of DIAMETER 3x FWHM (radius 1.5x FWHM) / total incident power.
    # Power integral in polar coordinates: INT I(r0) 2 pi r0 dr0.
    r_int = 1.5 * fwhm
    sel = r0grid <= r_int
    p_in = np.trapezoid(I[sel] * 2 * pi * r0grid[sel], r0grid[sel])
    p_tot = pi * R * R          # unit-amplitude plane wave over aperture
    ideal = ideal_peak(lam)
    # shape Strehl (header METRICS, paper Fig. 4f convention): both
    # PSFs normalized to the power captured in the r0grid window (the
    # CCD analogue), so diffraction efficiency cancels and only core
    # fidelity remains. Ideal reference: same-quadrature ideal-lens
    # PSF on the same window.
    I_id = ideal_profile(lam, r0grid)
    p_win = np.trapezoid(I * 2 * pi * r0grid, r0grid)
    p_win_id = np.trapezoid(I_id * 2 * pi * r0grid, r0grid)
    s_shape = float((I.max() / p_win) / (I_id.max() / p_win_id)) \
        if p_win > 0 else float("nan")
    psf_metrics["fwhm_um"].append(float(fwhm))
    psf_metrics["eff_3fwhm"].append(float(p_in / p_tot))
    psf_metrics["strehl_like"].append(float(Ipk / ideal ** 2))
    psf_metrics["strehl_shape"].append(s_shape)
    psf_metrics["onax_I_at_F"].append(float(I[0]))
    log("  lam=%.2f: FWHM=%.2f um (dl %.2f), eff=%.3f, S=%.3f, "
        "Sshape=%.3f"
        % (lam, fwhm, lam / (2 * na), psf_metrics["eff_3fwhm"][-1],
           psf_metrics["strehl_like"][-1], s_shape))
results.update(psf_metrics)

# ---- r-z intensity maps (raw data of the paper's Fig. 2e/4a tiles) -------
# For each verification wavelength, the intensity I(r0, z) of the field
# diffracted by the DESIGNED staircase (h, via exit_field inside rs_psf)
# is evaluated on an observation grid around the design focus. The grid
# scalars come from config.json (rzmap_* keys; cfg.get defaults exist
# only for run folders created before those keys did) and define WHERE
# the field is sampled -- the field itself comes from the design vector.
# Each tile pixel is one J0-reduced RS-I integral over all N rings.
rz_r = np.linspace(0.0, cfg.get("rzmap_r_max_um", 20.0),
                   cfg.get("rzmap_r_points", 41))
rz_z = np.linspace(F - cfg.get("rzmap_z_span_um", 1000.0),
                   F + cfg.get("rzmap_z_span_um", 1000.0),
                   cfg.get("rzmap_z_points", 121))
rzmaps = {"r0grid": rz_r, "zgrid": rz_z}
log("[3/4] I(r,z) maps for fig_rz_tiles.png (paper Fig. 2e/4a analogue): "
    "window r = 0..%g um (%d radii; mirrored to +/-r in the plot), "
    "z = F +/- %.2f mm (%d planes)"
    % (rz_r[-1], rz_r.size, (rz_z[-1] - F) / 1000, rz_z.size))
log("      = %d RS-I integrals over %d rings per wavelength, "
    "%d wavelengths" % (rz_z.size * rz_r.size, prob.N, lam_list.size))
for lam in lam_list:
    t_lam = time.time()
    Mrz = np.empty((rz_z.size, rz_r.size))
    for iz, z in enumerate(rz_z):
        Mrz[iz] = np.abs(rs_psf(lam, z, rz_r)) ** 2
    rzmaps["I_%d" % int(lam * 1000)] = Mrz
    # peak of the map = brightest point in the tile window (r, z)
    izm, irm = np.unravel_index(np.argmax(Mrz), Mrz.shape)
    log("  lam=%.2f um: map done (%.1fs); tile peak at r=%.1f um, "
        "z=%.3f mm" % (lam, time.time() - t_lam, rz_r[irm],
                       rz_z[izm] / 1000))
np.savez(os.path.join(out_rs, "verify_rzmap.npz"), **rzmaps)
log("[3/4] r-z maps done -> verify_rzmap.npz (arrays: r0grid, zgrid, "
    "I_<nm> per wavelength)")

# ---- J metrics: objective / continuous / verification comb ---------------
# All three use the ARITHMETIC-mean convention (MDLProblem default),
# regardless of the fom_mode the design was optimized with, so numbers
# stay comparable across runs. J_objective additionally evaluates on the
# design's own target wavelengths when those were a discrete comb.
log("[4/4] J metrics (all arithmetic-mean, comparable across runs)...")
results["J_continuous"] = prob.fom(m)
comb = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"], cfg["h_max_um"],
                  cfg["dh_um"],
                  n_wavelengths=lam_list.size).set_wavelengths(lam_list)
results["J_verify_comb"] = comb.fom(m)
if lams_obj is not None:
    obj = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"],
                     cfg["h_max_um"], cfg["dh_um"],
                     n_wavelengths=len(lams_obj)).set_wavelengths(
                         np.asarray(lams_obj, dtype=float))
    results["J_objective"] = obj.fom(m)
else:
    results["J_objective"] = results["J_continuous"]
log("J objective=%.4f   continuous=%.4f   verify-comb=%.4f"
    % (results["J_objective"], results["J_continuous"],
       results["J_verify_comb"]))

# ---- save into the run folder --------------------------------------------
np.savez(os.path.join(out_rs, "verify_onaxis.npz"), zgrid=zgrid,
         **{"I_%d" % int(l * 1000): onax[l] for l in lam_list})
json.dump(results, open(os.path.join(out_rs, "verify_metrics.json"), "w"),
          indent=1)

# re-export ring table for the Zemax DLL (heights in mm!) so the table
# always matches the vector actually verified
ring_path = os.path.join(run_dir, "mdl_rings_%d.txt" % cfg["dll_file_no"])
with open(ring_path, "w") as f:
    f.write("%d %.9f\n" % (prob.N, prob.delta / 1000.0))
    f.writelines("%.9f\n" % (v / 1000.0) for v in h)
log("saved into %s:" % os.path.relpath(run_dir))
log("  rs\\verify_metrics.json  scalar metrics per wavelength + J summary")
log("  rs\\verify_onaxis.npz    zgrid + I_<nm>: on-axis scans "
    "(fig_onaxis*)")
log("  rs\\verify_rzmap.npz     r0grid, zgrid + I_<nm>: r-z maps "
    "(fig_rz_tiles)")
log("  %s      ring table re-export (run ROOT), matches the verified "
    "vector" % os.path.basename(ring_path))
rel = os.path.relpath(run_dir)
log("next:  python %s %s"
    % (os.path.join("02_validation_rs", "make_plots.py"), rel))