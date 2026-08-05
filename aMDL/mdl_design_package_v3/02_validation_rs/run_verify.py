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
                sphere.
Also reported: J on the design objective, on the continuous band and on
the verification comb (arithmetic-mean convention in all three, so runs
with different fom_mode remain comparable), and the DLL ring table is
re-exported so it always matches the vector actually verified.

Outputs (into the run folder):
    verify_metrics.json, verify_onaxis.npz, mdl_rings_<n>.txt,
    scripts/run_verify.py (snapshot of this script as run)

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
    Fig. 2e cites [1] as its ref. 46.)
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
    raise SystemExit("no config.json in %r -- is this a run folder made "
                     "by run_MDL_design.py?" % run_dir)
cfg = json.load(open(cfg_path))
der = cfg["derived"]

D, na, F = cfg["diameter_um"], der["na"], der["focal_um"]
lmin, lmax = cfg["lam_min_um"], cfg["lam_max_um"]
lams_obj = cfg["target_wavelengths_um"]        # None = continuous objective
lam_list = np.asarray(cfg["verify_wavelengths_um"], dtype=float)

m_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir,
                                                            "m_final.npy")
m = np.load(m_file)

# Rebuild the design problem for this geometry. Used for (a) the ring
# center radii rho_i and widths (the SAME quadrature nodes as the design
# FOM, see STEP 3 in the header) and (b) the alias-safe continuous-band
# J metric.
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

log("run: %s" % run_dir)
log("profile %s: J_continuous = %.4f" % (m_file, prob.fom(m)))


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


# ---- on-axis scans (focal shift / achromaticity, paper Fig. 2e) ----------
results = {"run_dir": run_dir, "m_file": m_file,
           "lam_um": lam_list.tolist(), "F_um": F}
zgrid = np.linspace(F - cfg["verify_z_span_um"],
                    F + cfg["verify_z_span_um"], cfg["verify_z_points"])
onax = {}
log("on-axis scans...")
for lam in lam_list:
    I = np.abs(rs_onaxis(lam, zgrid)) ** 2
    onax[lam] = I
    zpk = float(zgrid[np.argmax(I)])
    results.setdefault("z_peak_um", []).append(zpk)
log("on-axis done")

# ---- PSF metrics at the design focal plane -------------------------------
r0grid = np.linspace(0.0, cfg["verify_r_max_um"], cfg["verify_r_points"])
psf_metrics = {"fwhm_um": [], "eff_3fwhm": [], "strehl_like": [],
               "onax_I_at_F": []}
log("PSFs at z=F...")
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
    psf_metrics["fwhm_um"].append(float(fwhm))
    psf_metrics["eff_3fwhm"].append(float(p_in / p_tot))
    psf_metrics["strehl_like"].append(float(Ipk / ideal ** 2))
    psf_metrics["onax_I_at_F"].append(float(I[0]))
    log("  lam=%.2f: FWHM=%.2f um (dl %.2f), eff=%.3f, S=%.3f"
        % (lam, fwhm, lam / (2 * na), psf_metrics["eff_3fwhm"][-1],
           psf_metrics["strehl_like"][-1]))
results.update(psf_metrics)

# ---- J metrics: objective / continuous / verification comb ---------------
# All three use the ARITHMETIC-mean convention (MDLProblem default),
# regardless of the fom_mode the design was optimized with, so numbers
# stay comparable across runs. J_objective additionally evaluates on the
# design's own target wavelengths when those were a discrete comb.
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
np.savez(os.path.join(run_dir, "verify_onaxis.npz"), zgrid=zgrid,
         **{"I_%d" % int(l * 1000): onax[l] for l in lam_list})
json.dump(results, open(os.path.join(run_dir, "verify_metrics.json"), "w"),
          indent=1)

# re-export ring table for the Zemax DLL (heights in mm!) so the table
# always matches the vector that was actually verified
ring_path = os.path.join(run_dir, "mdl_rings_%d.txt" % cfg["dll_file_no"])
with open(ring_path, "w") as f:
    f.write("%d %.9f\n" % (prob.N, prob.delta / 1000.0))
    f.writelines("%.9f\n" % (v / 1000.0) for v in h)
log("saved: verify_metrics.json, verify_onaxis.npz, %s"
    % os.path.basename(ring_path))
rel = os.path.relpath(run_dir)
log("next:  python %s %s"
    % (os.path.join("02_validation_rs", "make_plots.py"), rel))