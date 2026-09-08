"""
mtf_verify.py -- STAGE 2a (MTF): the modulation transfer function of a
designed MDL, computed the ONLY way that is valid for a diffractive lens
in this toolchain -- as the normalized Hankel transform of the trusted
focal-plane PSF, NOT from a ray-based engine.

WHY THIS SCRIPT EXISTS
======================================================================
OpticStudio's native MTF (both Geometric and FFT) is INVALID on our
zone/rz surface model. That surface is a pure phase screen: it does not
bend rays (every diffraction order stays in phase), so the ray-traced
exit pupil / reference sphere that both MTF flavors rely on is
degenerate -- afocal. The symptom is unmistakable: the FFT-MTF frequency
axis auto-scales to ~0.25 lp/mm instead of the physical cutoff
2*NA/lambda ~ 364 lp/mm at 550 nm (a factor ~1500 too small), and the
curve collapses at the origin. That is Zemax reporting "I cannot find
the image", not a statement about the lens.

The rz route already proved the PHASE is exact (OPD self-check wrapped
RMS = 0.0000 waves, Zemax field vs RS corr = 1.0000). So the honest MTF
is obtained from the same focal field run_verify.py propagates:

    incoherent OTF(f) = FT2D{ |U(r,F)|^2 }
                      = 2*pi INT_0^inf I(r) J0(2*pi f r) r dr   (Hankel-0,
                        because the PSF is rotationally symmetric)
    MTF(f) = |OTF(f)| / OTF(0),   OTF(0) = 2*pi INT I(r) r dr = window power.

This is the standard incoherent-imaging MTF (Goodman [1] Sec. 6.3;
Born & Wolf [2] Sec. 9.5). The J0 kernel is validated against the
analytic clear-circular-aperture MTF (2/pi)(arccos nu - nu sqrt(1-nu^2)),
nu = f/f_c, to < 2.1e-3 max error.

READING THE RESULT (the physics we expect, and why it is NOT aberration)
======================================================================
A diffractive PSF is a diffraction-limited CORE on a broad HALO. Under
the (definitional) normalization MTF(0)=1:
  * the halo, being broad in space, is narrow in frequency: it makes the
    MTF drop steeply over the first few tens of lp/mm;
  * after that drop the MTF settles onto a PLATEAU whose height ~ the
    core energy fraction, and rides it out to the full cutoff 2*NA/lambda.
So a LOW PLATEAU that still reaches the correct cutoff = a sharp core at
low efficiency (light in the halo/other orders), the diffractive
efficiency limit -- NOT a wavefront defect. The aberration signature
would instead be a TRUNCATED cutoff (curve dying well before 2*NA/lambda).
This mirrors the reference paper's own Fig. (S1..S5): the spread from
near-diffraction-limited (efficient) to steep-drop-low-plateau
(inefficient achromat) IS the efficiency spread, and the low-plateau
samples are published results, reaching the same cutoff.

NORMALIZATION / MEASUREMENT MATCH. The MTF is computed over a finite
radial window (mtf_r_max_um, default = verify_r_max_um), so OTF(0) is the
IN-WINDOW power -- exactly the convention of a CCD-measured MTF and of
run_verify's strehl_shape. To compare against a measured MTF, match the
window to the camera crop; a wider window admits more halo and lowers the
plateau. Compare like-normalized curves only (same lesson as the Strehl
conventions: never mix window-normalized with total-power-normalized).

USAGE (from the package root)
======================================================================
    python 02_validation_rs\\mtf_verify.py runs\\<run_folder> [other_m.npy]

Reads config.json + m_final.npy exactly as run_verify.py does; nothing is
hardcoded. Optional config keys (defaults): mtf_r_max_um (verify_r_max_um),
mtf_r_points (1601), mtf_f_points (400), mtf_f_max_lppmm (auto = 1.05 x
2*NA/lam_min). Equal spectral weights are assumed for the polychromatic
curve (state a weighting in config later if needed).

OUTPUTS (per-solver layout, into rs/)
    rs/verify_mtf.npz   f_lppmm, MTF_<nm>, MTFdl_<nm> per wavelength,
                        MTF_poly, MTFdl_poly, fc_lppmm per wavelength,
                        and the scalar mtf_quality = area(MTF)/area(MTFdl)
    rs/fig_mtf.png      per-wavelength design MTF (thin) + polychromatic
                        (bold) + polychromatic diffraction limit (dashed),
                        on an lp/mm axis for direct overlay on the paper.

The RS focal kernel (exit_field / rs_psf / ideal_profile) is copied
VERBATIM from the validated run_verify.py -- run_verify.py remains the
single source of truth for that physics; this module only adds the
Hankel-MTF transform on top of it. (When convenient these can be merged:
run_verify saves the focal PSF, make_plots draws fig_mtf from the npz.)

REFERENCES
    [1] J. W. Goodman, Introduction to Fourier Optics, 3rd ed. (2005),
        Sec. 6.3 (frequency response of an incoherent imaging system).
    [2] M. Born & E. Wolf, Principles of Optics, 7th ed. (1999), Sec. 9.5.
    [3] Y. Xiao et al., Light Sci. Appl. 11, 323 (2022) (the MTF figure
        being compared against).
"""
import json
import os
import sys
import time

import numpy as np
from numpy import pi
from scipy.special import j0

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from mdl_core import MDLProblem

T0 = time.time()


def log(s):
    print("[%6.1fs] %s" % (time.time() - T0, s), flush=True)


# ---- load run configuration (same contract as run_verify.py) -------------
if len(sys.argv) < 2:
    raise SystemExit("usage: python mtf_verify.py runs/<run_folder> "
                     "[optional/other_m.npy]")
run_dir = sys.argv[1].rstrip("/\\")
if not os.path.exists(os.path.join(run_dir, "config.json")):
    alt = os.path.join(PKG_ROOT, run_dir)
    if os.path.exists(os.path.join(alt, "config.json")):
        run_dir = alt
cfg_path = os.path.join(run_dir, "config.json")
if not os.path.exists(cfg_path):
    raise SystemExit("no config.json in %r -- not a run folder?" % run_dir)
cfg = json.load(open(cfg_path))
der = cfg["derived"]

D, na, F = cfg["diameter_um"], der["na"], der["focal_um"]
lmin, lmax = cfg["lam_min_um"], cfg["lam_max_um"]
lam_list = np.asarray(cfg["verify_wavelengths_um"], dtype=float)

m_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir,
                                                            "m_final.npy")
m = np.load(m_file)

prob = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"],
                  cfg["h_max_um"], cfg["dh_um"],
                  n_wavelengths=cfg["n_wavelengths"])
if m.size != prob.N:
    raise SystemExit("%s has %d rings; config expects %d"
                     % (m_file, m.size, prob.N))
h = m * prob.dh                      # ring heights h_i = m_i * dh   [um]
rho = prob.rho                       # ring center radii              [um]
drho = prob.delta                    # ring width DELTA               [um]
R = prob.R                           # aperture radius                [um]
# ring quadrature of the design field -- same rule and same config key
# as run_verify.py (2026-09-08): "sinc" = analytic ring integral of the
# kernel phase ramp (physical staircase, default), "midpoint" = one
# kernel sample per ring (paper Eq. 4). The ideal-lens reference keeps
# the midpoint sum, which is exact for a continuous phase.
RS_QUAD = str(cfg.get("rs_ring_quadrature", "sinc")).lower()
if RS_QUAD not in ("sinc", "midpoint"):
    raise SystemExit("rs_ring_quadrature must be 'sinc' or 'midpoint'")


def ring_factor(lam, rb):
    if RS_QUAD == "midpoint":
        return 1.0
    return np.sinc(drho * rho / (lam * rb))

out_rs = os.path.join(run_dir, "rs")
os.makedirs(out_rs, exist_ok=True)

log("run: %s" % run_dir)
log("RS ring quadrature: %s" % RS_QUAD)
log("design '%s': D=%.2f mm, F=%.2f mm, NA=%.4f | %d rings x %.2f um"
    % (cfg["name"], D / 1000, F / 1000, na, prob.N, prob.delta))


# ---- RS focal kernel: VERBATIM from run_verify.py (single source) --------
def exit_field(lam):
    """U0_i = exp[i k (n(lam)-1) h_i] -- thin-element exit field."""
    n = float(prob.n_func(lam))
    k = 2 * pi / lam
    return np.exp(1j * k * (n - 1.0) * h)


def rs_psf(lam, z, r0grid):
    """J0-reduced RS-I field U(r0, z) -- identical to run_verify.rs_psf
    (incl. the ring_factor of 2026-09-08)."""
    E0 = exit_field(lam)
    k = 2 * pi / lam
    out = np.empty(r0grid.size, dtype=complex)
    for ir, r0 in enumerate(r0grid):
        rb = np.sqrt(z * z + rho * rho + r0 * r0)
        integ = E0 * j0(k * rho * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 \
            * rho * ring_factor(lam, rb)
        out[ir] = (z / (1j * lam)) * 2 * pi * drho * np.sum(integ)
    return out


def ideal_profile(lam, r0grid):
    """|U_ideal(r0, F)|^2 -- identical to run_verify.ideal_profile."""
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


# ---- MTF transform: normalized Hankel-0 of the intensity PSF --------------
def mtf_of(I, r, f):
    """MTF(f) = |2 pi INT I(r) J0(2 pi f r) r dr| / (2 pi INT I(r) r dr).

    r in um, f in cyc/um. Trapezoid quadrature on the sampled PSF; the
    denominator is the in-window power (OTF(0)) -> MTF(0) = 1. Validated
    against the analytic clear-aperture MTF to < 2.1e-3.
    """
    w = I * r
    dc = 2 * pi * np.trapezoid(w, r)
    out = np.empty(f.size)
    for i, ff in enumerate(f):
        out[i] = 2 * pi * np.trapezoid(w * j0(2 * pi * ff * r), r)
    return np.abs(out) / dc


# ---- grids ---------------------------------------------------------------
r_max = float(cfg.get("mtf_r_max_um", cfg["verify_r_max_um"]))
r_pts = int(cfg.get("mtf_r_points", 1601))
r0 = np.linspace(0.0, r_max, r_pts)
fc_min_um = 2 * na / lmin                 # highest cutoff (shortest lambda)
f_max_lppmm = float(cfg.get("mtf_f_max_lppmm", 1.05 * fc_min_um * 1000.0))
f_um = np.linspace(0.0, f_max_lppmm / 1000.0, int(cfg.get("mtf_f_points",
                                                          400)))
f_lppmm = f_um * 1000.0

log("MTF: Hankel-0 of |U(r,F)|^2 over r=0..%.1f um (%d pts), "
    "f=0..%.0f lp/mm (%d pts)"
    % (r_max, r_pts, f_lppmm[-1], f_um.size))
log("     window normalized (OTF(0)=in-window power; CCD/strehl_shape "
    "convention). cutoff 2NA/lam: %.0f lp/mm @400nm .. %.0f lp/mm @1100nm"
    % (2 * na / 0.400 * 1000, 2 * na / 1.100 * 1000))

out = {"f_lppmm": f_lppmm, "lam_um": lam_list,
       "r_max_um": r_max, "note": "window-normalized incoherent MTF"}
psf_sum = np.zeros(r0.size)
psf_id_sum = np.zeros(r0.size)
fc_list, q_list, cap_list = [], [], []
p_tot = pi * R * R                        # total incident power (unit ampl.)
log("  lam(um)  cutoff   in-win%   MTF@50  MTF@100  MTF@200   quality "
    "(design vs OWN-lambda limit)")
for lam in lam_list:
    I = np.abs(rs_psf(lam, F, r0)) ** 2
    Iid = ideal_profile(lam, r0)
    psf_sum += I
    psf_id_sum += Iid
    mtf = mtf_of(I, r0, f_um)
    mtf_dl = mtf_of(Iid, r0, f_um)
    out["MTF_%d" % int(lam * 1000)] = mtf
    out["MTFdl_%d" % int(lam * 1000)] = mtf_dl
    fc = 2 * na / lam * 1000.0
    fc_list.append(fc)
    # fraction of the total incident power that lands INSIDE the window at
    # the focal plane: low value => much halo excluded => the window
    # normalization inflates the plateau (see header NORMALIZATION note).
    cap = float(2 * pi * np.trapezoid(I * r0, r0) / p_tot)
    cap_list.append(cap)
    # quality = area(design MTF) / area(OWN-lambda diffraction-limit MTF),
    # each integrated to THAT wavelength's cutoff -> 1.0 = at the limit.
    # (Always <= 1 for a physical system; this is the honest per-lambda
    # comparison. In the plot, compare each curve to its OWN dashed limit,
    # NOT to the polychromatic one.)
    m_cut = f_lppmm <= fc
    q = float(np.trapezoid(mtf[m_cut], f_lppmm[m_cut]) /
              np.trapezoid(mtf_dl[m_cut], f_lppmm[m_cut]))
    q_list.append(q)

    def at(freq):
        return float(np.interp(freq, f_lppmm, mtf))
    log("  %.3f    %4.0f     %5.1f    %.3f   %.3f    %.3f      %.3f"
        % (lam, fc, 100 * cap, at(50), at(100), at(200), q))

# polychromatic (equal weights): MTF of the summed focal intensity
mtf_poly = mtf_of(psf_sum, r0, f_um)
mtf_poly_dl = mtf_of(psf_id_sum, r0, f_um)
out["MTF_poly"] = mtf_poly
out["MTFdl_poly"] = mtf_poly_dl
out["fc_lppmm"] = np.asarray(fc_list)
out["mtf_quality"] = np.asarray(q_list)
out["in_window_frac"] = np.asarray(cap_list)
fc_poly = 2 * na / (0.5 * (lmin + lmax)) * 1000.0
mpc = f_lppmm <= fc_poly
q_poly = float(np.trapezoid(mtf_poly[mpc], f_lppmm[mpc]) /
               np.trapezoid(mtf_poly_dl[mpc], f_lppmm[mpc]))
out["mtf_quality_poly"] = q_poly

np.savez(os.path.join(out_rs, "verify_mtf.npz"), **out)
log("polychromatic (equal weights): quality area ratio = %.3f "
    "(1.0 = diffraction limit)" % q_poly)
log("saved rs\\verify_mtf.npz")

# ---- figure: two panels, each curve vs its OWN limit ---------------------
# LEFT  = the headline, bounded comparison (paper style): polychromatic
#         design vs polychromatic diffraction limit -- both chromatically
#         blurred identically, so design <= limit always.
# RIGHT = per-wavelength design (solid) each with its OWN monochromatic
#         diffraction limit (dashed, same color). A monochromatic curve
#         must be read against its OWN dashed line, never against the
#         polychromatic one -- that mismatch is what makes short-lambda
#         curves look like they beat the limit.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axp, axl) = plt.subplots(1, 2, figsize=(12.0, 4.7))

    axp.plot(f_lppmm, mtf_poly_dl, color="k", lw=1.8, ls="--",
             label="diffraction limit (polychromatic)")
    axp.plot(f_lppmm, mtf_poly, color="#e6550d", lw=2.6,
             label="design (polychromatic)")
    axp.set_xlim(0, f_lppmm[-1])
    axp.set_ylim(0, 1.0)
    axp.set_xlabel("spatial frequency  (lp/mm)")
    axp.set_ylabel("MTF")
    axp.set_title("Polychromatic MTF (the bounded comparison)\n"
                  "area ratio design/limit = %.3f" % q_poly, fontsize=9)
    axp.legend(fontsize=8, loc="upper right")
    axp.grid(True, alpha=0.25)

    cmap = plt.get_cmap("turbo")
    n = lam_list.size
    for i, lam in enumerate(lam_list):
        c = cmap(i / max(n - 1, 1))
        axl.plot(f_lppmm, out["MTF_%d" % int(lam * 1000)], color=c, lw=1.1,
                 label="%d nm" % int(lam * 1000))
        axl.plot(f_lppmm, out["MTFdl_%d" % int(lam * 1000)], color=c,
                 lw=0.8, ls="--", alpha=0.6)
    axl.set_xlim(0, f_lppmm[-1])
    axl.set_ylim(0, 1.0)
    axl.set_xlabel("spatial frequency  (lp/mm)")
    axl.set_ylabel("MTF")
    axl.set_title("Per-wavelength: design (solid) vs its OWN limit "
                  "(dashed)\nwindow r<=%.0f um (in-window %.0f-%.0f%% of "
                  "incident power)"
                  % (r_max, 100 * min(cap_list), 100 * max(cap_list)),
                  fontsize=9)
    axl.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
    axl.grid(True, alpha=0.25)

    fig.suptitle("MDL '%s': incoherent MTF from RS focal field "
                 "(window-normalized, CCD convention)" % cfg["name"],
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_rs, "fig_mtf.png"), dpi=140)
    log("saved rs\\fig_mtf.png")
except Exception as e:
    log("plot skipped (%s); data is in rs\\verify_mtf.npz" % e)

log("next: overlay rs\\fig_mtf.png on the paper's MTF (same lp/mm axis); "
    "compare like-normalized curves only.")