"""
bpm_validate.py -- STAGE 2a (optional): quantify the thin-element
approximation (TEA) error by propagating the field THROUGH the actual
finite-thickness staircase relief with a finite-difference beam
propagation method (FD-BPM), and comparing against the phase-screen
model used everywhere else (design FOM, run_verify.py, Zemax zone DLL).

Usage (from the package root):

    python 02_validation_rs\\bpm_validate.py runs\\<run_folder>

Reads geometry, ring table and wavelengths from the run folder's
config.json; writes bpm_validation.json into the same folder. By
default a coarse wavelength subset (every 3rd verification wavelength)
is used because the BPM is the expensive step; override by adding
"bpm_wavelengths_um": [...] to the config dictionary.

===========================================================================
WHY THIS SCRIPT EXISTS
===========================================================================
run_verify.py models the H ~ 15 um staircase as an infinitely thin
phase screen U0 = exp[i k (n-1) h(rho)] (TEA). That neglects what
happens INSIDE the relief: diffraction of the field between the ring
side-walls, edge shadowing, and walk-off across ring boundaries over
the finite thickness. This script computes those effects with a
one-way scalar propagator and reports how much they change the exit
field and the focal metrics -- i.e. it puts an error bar on the TEA.
(Result on record for the S3 geometry: exit-field overlap 0.82-0.92 --
the difference is dominated by a smooth intra-relief propagation phase
that barely refocuses -- while the metrics that matter, focal
efficiency and FWHM, agree to 1-3 %: the TEA is adequate there.)

What it deliberately does NOT capture: Fresnel interface reflections
(a nearly uniform few-% amplitude factor at n ~ 1.6 -> air; does not
reshape the PSF), backward waves, and vector/polarization effects
(rings >= 2 um >= 2*lambda, NA ~ 0.1: scalar regime). For those, the
next tool up is a rigorous solver (FDTD/RCWA) on the outermost zones.

===========================================================================
NUMERICAL METHOD -- paraxial one-way FD-BPM, Crank-Nicolson scheme
===========================================================================
Model equation. Write the scalar field as E_total = E(r, z) *
exp(i k0 nref z) with a fixed reference index nref; the slowly-varying
envelope E obeys the paraxial (Fresnel) one-way wave equation

    2 i k0 nref dE/dz = Laplacian_T E + k0^2 (n(r,z)^2 - nref^2) E

(standard FD-BPM formulation: Van Roey, van der Donk & Lagasse [B1];
Chung & Dagli [B2]; review: Scarmozzino et al. [B3], Sec. II). In
cylindrical coordinates with rotational symmetry the transverse
Laplacian is

    Laplacian_T = d^2/dr^2 + (1/r) d/dr .

The index landscape is the actual staircase: n(r, z) = n_resist(lam)
for z < h(r), else 1 (air), sampled on the BPM grid from the run's
ring table. nref = (n_resist + 1)/2 (midpoint of the two media)
minimizes the worst-case |n^2 - nref^2|, keeping the envelope slowly
varying in both materials [B3].

Discretization. Uniform grid, DR = DZ = 0.05 um (>= 40 points per
2 um ring width, ~300 z-steps through a 15 um relief). Writing
L = (1/(2 k0 nref)) Laplacian_T + V with the "potential"
V(r, z) = k0 (n^2 - nref^2) / (2 nref), one z-step is advanced by the
Crank-Nicolson (trapezoidal) scheme

    (I - i DZ/2 L) E(z+DZ) = (I + i DZ/2 L) E(z),

which is unconditionally stable and second-order accurate in DZ and DR
[B2, B3]. The three-point stencil for Laplacian_T makes the left-hand
side tridiagonal; each step is one banded solve
(scipy.linalg.solve_banded).

Boundary conditions.
* Axis r -> 0: the grid is staggered (first node at r = DR/2) and the
  symmetry condition dE/dr = 0 is imposed via a ghost node
  E[-1] = E[0], folded into the diagonal (Neumann BC).
* Outer edge: a PAD = 200 um "numerical beach" beyond the aperture --
  a super-Gaussian amplitude taper absorbs outgoing light so nothing
  reflects back into the aperture (absorbing boundary as in [B3],
  Sec. II-E; simpler and sufficient here compared to a transparent BC
  [B4] because the relief only weakly scatters outward).

Validity. One-way and paraxial: inside the 15 um relief the field is
still essentially collimated (it left a plane wave <= 15 um earlier;
the lens's NA ~ 0.1 develops over millimeters, not micrometers), so
paraxiality holds where it is used. Scalar: feature size >= 2 um
>= 2*lambda across the band.

===========================================================================
COMPARISON PROTOCOL
===========================================================================
For each test wavelength:
1. E_bpm(r)  = BPM exit field at z = H (through the real staircase).
2. E_te(r)   = exp[i k0 (n-1) h(r)] (thin-element field, identical to
               run_verify.py's exit_field on the same grid).
3. overlap   = |<E_te, E_bpm>| / (||E_te|| ||E_bpm||), inner product
               weighted by r dr over the aperture (axisymmetric area
               element) -- a 1.000 means TEA and BPM agree perfectly
               in shape AND phase.
4. Both fields are then propagated to the design focus with the SAME
   first Rayleigh-Sommerfeld kernel as run_verify.py (axisymmetric
   J0-reduced RS-I -- see run_verify.py header STEPS 2-3 and Goodman
   [1]/Shen & Wang [3] cited there), and the focal metrics (on-axis
   intensity, FWHM, focusing efficiency in the 3x-FWHM-diameter disc)
   are compared per wavelength: eff_te vs eff_bpm, fwhm_te vs
   fwhm_bpm. Differences ARE the thin-element model error.

Output: bpm_validation.json in the run folder with all of the above.

===========================================================================
REFERENCES (BPM; for the RS references see run_verify.py)
===========================================================================
[B1] J. Van Roey, J. van der Donk, P. E. Lagasse, "Beam-propagation
     method: analysis and assessment," J. Opt. Soc. Am. 71, 803-810
     (1981).
[B2] Y. Chung & N. Dagli, "An assessment of finite difference beam
     propagation method," IEEE J. Quantum Electron. 26, 1335-1339
     (1990). (The Crank-Nicolson FD-BPM used here.)
[B3] R. Scarmozzino, A. Gopinath, R. Pregla, S. Helfert, "Numerical
     techniques for modeling guided-wave photonic devices," IEEE J.
     Sel. Top. Quantum Electron. 6, 150-162 (2000). (Review: scheme,
     reference-index choice, absorbing boundaries.)
[B4] G. R. Hadley, "Transparent boundary condition for beam
     propagation," Opt. Lett. 16, 624-626 (1991).
"""
import json
import os
import sys
import time

import numpy as np
from numpy import pi
from scipy.linalg import solve_banded
from scipy.special import j0

# package root = parent of this stage folder; mdl_core.py lives there
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from mdl_core import n_az4562

T0 = time.time()


def log(s):
    print("[%6.1fs] %s" % (time.time() - T0, s), flush=True)


# ---- load run configuration ----------------------------------------------
if len(sys.argv) < 2:
    raise SystemExit("usage: python bpm_validate.py runs/<run_folder>")
run_dir = sys.argv[1].rstrip("/\\")
if not os.path.exists(os.path.join(run_dir, "config.json")):
    alt = os.path.join(PKG_ROOT, run_dir)          # package-root relative
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
F = cfg["derived"]["focal_um"]                     # design focus [um]

RINGS = os.path.join(run_dir, "mdl_rings_%d.txt" % cfg["dll_file_no"])
with open(RINGS) as fh:
    N, delta_mm = fh.readline().split()
    N, delta = int(N), float(delta_mm) * 1000.0    # ring width [um]
    h_ring = np.array([float(fh.readline()) for _ in range(N)]) * 1000.0

R = N * delta                                      # aperture radius [um]
H = float(h_ring.max())                            # max relief height [um]
log("run: %s" % run_dir)
log("rings: %d x %.2f um, R=%.2f mm, H=%.2f um, F=%.2f mm"
    % (N, delta, R / 1000, H, F / 1000))

# BPM wavelengths -- selection priority:
#   1. CLI:  python bpm_validate.py runs/<folder> all
#            -> the FULL verify_wavelengths_um list (14 lines for the
#               comb designs; tiles then match fig_rz_tiles line for
#               line; budget ~90-100 s per wavelength)
#      or:   python bpm_validate.py runs/<folder> 0.85,0.95
#            -> an explicit list [um]
#   2. config key "bpm_wavelengths_um" (list)
#   3. default: every 3rd verification lambda (cheap subset)
if len(sys.argv) > 2:
    if sys.argv[2].strip().lower() == "all":
        LAMS = [float(v) for v in cfg["verify_wavelengths_um"]]
    else:
        LAMS = [float(v) for v in sys.argv[2].split(",")]
else:
    LAMS = cfg.get("bpm_wavelengths_um",
                   list(np.asarray(cfg["verify_wavelengths_um"])[::3]))
log("wavelengths (%d): %s um" % (len(LAMS),
                                 ", ".join("%.2f" % l for l in LAMS)))
if len(LAMS) > 6:
    log("NOTE: %d wavelengths x ~90-100 s each -- expect ~%d min total"
        % (len(LAMS), int(len(LAMS) * 100 / 60) + 1))

# ---- BPM grid (see header: NUMERICAL METHOD) ------------------------------
DR = 0.05          # transverse step [um] (>= 40 pts per 2 um ring)
DZ = 0.05          # longitudinal step [um] (~300 steps through H=15 um)
PAD = 200.0        # absorbing "beach" beyond the aperture [um]
r = np.arange(0.0, R + PAD, DR) + 0.5 * DR         # staggered: r0 = DR/2
NR = r.size

# sample the staircase h(r) from the ring table onto the BPM grid
h_of_r = np.zeros(NR)
idx = np.minimum((r / delta).astype(int), N - 1)
inside = r < R
h_of_r[inside] = h_ring[idx[inside]]

# super-Gaussian amplitude absorber in the pad (Scarmozzino [B3] II-E)
absorb = np.ones(NR)
pad_zone = r > R + 10.0
absorb[pad_zone] = np.exp(-((r[pad_zone] - (R + 10.0)) / (0.6 * PAD)) ** 4
                          * 0.5)


def bpm_exit_field(lam):
    """Crank-Nicolson FD-BPM through the staircase -> E(r) at z = H.

    Advances the paraxial envelope (header eq.) step by step through
    the index landscape n(r,z) = n_resist for z < h(r) else 1, with
    axis Neumann BC and the outer absorber. Tridiagonal solve per step
    (banded storage: ab[0]=super, ab[1]=diag, ab[2]=sub).
    """
    n_res = float(n_az4562(lam))
    k0 = 2 * pi / lam
    n_ref = 0.5 * (n_res + 1.0)      # midpoint reference index [B3]
    nz = int(np.ceil(H / DZ))
    E = np.ones(NR, dtype=complex)   # unit plane wave at z = 0

    # transverse operator coefficients: (1/(2 k0 nref)) (d2/dr2+(1/r)d/dr)
    inv2k = 1.0 / (2.0 * k0 * n_ref)
    lo = inv2k * (1.0 / DR ** 2 - 1.0 / (2.0 * r * DR))  # sub-diagonal
    di = inv2k * (-2.0 / DR ** 2) * np.ones(NR)          # diagonal
    up = inv2k * (1.0 / DR ** 2 + 1.0 / (2.0 * r * DR))  # super-diagonal
    # axis (staggered node at DR/2): Neumann dE/dr = 0 via ghost node
    # E[-1] = E[0] -> fold the sub-diagonal coefficient into the diagonal
    di0_extra = lo[0]

    for iz in range(nz):
        z = (iz + 0.5) * DZ                       # midpoint of the slab
        n_loc = np.where(z < h_of_r, n_res, 1.0)  # staircase cross-section
        V = 0.5 * k0 * (n_loc ** 2 - n_ref ** 2) / n_ref
        diag = di + V
        diag_ax = diag.copy()
        diag_ax[0] += di0_extra

        # explicit half step: RHS = (I + i DZ/2 L) E
        rhs = E.copy()
        rhs[1:-1] += 0.5j * DZ * (lo[1:-1] * E[:-2] + diag_ax[1:-1] * E[1:-1]
                                  + up[1:-1] * E[2:])
        rhs[0] += 0.5j * DZ * (diag_ax[0] * E[0] + up[0] * E[1])
        rhs[-1] += 0.5j * DZ * (lo[-1] * E[-2] + diag_ax[-1] * E[-1])

        # implicit half step: solve (I - i DZ/2 L) E_new = RHS
        ab = np.zeros((3, NR), dtype=complex)
        ab[0, 1:] = -0.5j * DZ * up[:-1]
        ab[1, :] = 1.0 - 0.5j * DZ * diag_ax
        ab[2, :-1] = -0.5j * DZ * lo[1:]
        E = solve_banded((1, 1), ab, rhs)
        E *= absorb
    # NOTE on the phase reference: the envelope is defined against
    # exp(i k0 nref z); the TEA field below is written in the SAME
    # convention up to a global (r-independent) phase, which drops out
    # of the |overlap| and of all intensity-based metrics.
    return E


def thin_element_field(lam):
    """TEA exit field on the same grid (run_verify.py exit_field)."""
    n_res = float(n_az4562(lam))
    k0 = 2 * pi / lam
    return np.exp(1j * k0 * (n_res - 1.0) * h_of_r) * (r < R)


def rs_focus(E_r, lam, z, r0grid):
    """Axisymmetric RS-I propagation to the plane z (J0-reduced form).

    Same kernel and same J0 reduction as run_verify.rs_psf -- see that
    script's header (STEPS 2-3) for the derivation, validity bound and
    references (Goodman; Shen & Wang; DLMF 10.9.2). Quadrature here is
    the midpoint rule on the fine BPM grid (weight 2 pi r DR).
    """
    k = 2 * pi / lam
    sel = r < R + 5.0
    rr, EE = r[sel], E_r[sel]
    out = np.empty(r0grid.size, dtype=complex)
    for i, r0 in enumerate(r0grid):
        rb = np.sqrt(z * z + rr * rr + r0 * r0)
        out[i] = (z / (1j * lam)) * 2 * pi * DR * np.sum(
            EE * j0(k * rr * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 * rr)
    return out


# radial grid for the focal-plane PSF profiles (saved for plotting)
R0_GRID = np.linspace(0.0, 40.0, 401)

# on-axis scan grid: same span/sampling as run_verify.py (from config),
# so the BPM on-axis maps are directly comparable to verify_onaxis.npz
Z_GRID = np.linspace(F - cfg["verify_z_span_um"],
                     F + cfg["verify_z_span_um"], cfg["verify_z_points"])


def focus_profile(E_r, lam):
    """Radial focal-plane PSF |U(r0, F)|^2 on R0_GRID (RS-I propagated)."""
    return np.abs(rs_focus(E_r, lam, F, R0_GRID)) ** 2


def onaxis_profile(E_r, lam):
    """On-axis intensity |U(0, z)|^2 over Z_GRID (exact RS-I at r0=0;
    see run_verify.rs_onaxis -- the axisymmetric reduction is exact on
    axis). Quadrature: midpoint rule on the fine BPM grid."""
    k = 2 * pi / lam
    sel = r < R + 5.0
    rr, EE = r[sel], E_r[sel]
    out = np.empty(Z_GRID.size)
    for iz, z in enumerate(Z_GRID):
        rb = np.sqrt(z * z + rr * rr)
        out[iz] = abs((z / (1j * lam)) * 2 * pi * DR * np.sum(
            EE * np.exp(1j * k * rb) / rb ** 2 * rr)) ** 2
    return out


def metrics_from_profile(I):
    """Focal-plane metrics (same definitions as run_verify.py):
    on-axis intensity, FWHM (first half-crossing), focusing efficiency
    = power in the 3x-FWHM-diameter disc / total incident power."""
    Ipk = I.max()
    half = Ipk / 2
    below = np.where(I < half)[0]
    fwhm = 2 * R0_GRID[below[0]] if below.size else np.nan
    selr = R0_GRID <= 1.5 * fwhm
    p_in = np.trapezoid(I[selr] * 2 * pi * R0_GRID[selr], R0_GRID[selr])
    p_tot = pi * R * R
    return I[0], fwhm, p_in / p_tot


# ---- run the comparison ---------------------------------------------------
results = {"run_dir": run_dir, "lam_um": list(map(float, LAMS)),
           "F_um": F, "rings": RINGS, "DR_um": DR, "DZ_um": DZ,
           "eff_te": [], "eff_bpm": [], "fwhm_te": [], "fwhm_bpm": [],
           "field_overlap": []}
psf_store = {"r0grid": R0_GRID, "zgrid": Z_GRID}   # profiles for plotting
for lam in LAMS:
    t0 = time.time()
    E_bpm = bpm_exit_field(lam)
    E_te = thin_element_field(lam)
    # normalized complex overlap over the aperture, area weight r dr
    sel = r < R
    w = r[sel]
    ov = abs(np.sum(np.conj(E_te[sel]) * E_bpm[sel] * w)) / np.sqrt(
        np.sum(np.abs(E_te[sel]) ** 2 * w) *
        np.sum(np.abs(E_bpm[sel]) ** 2 * w))
    I_te = focus_profile(E_te, lam)
    I_bpm = focus_profile(E_bpm, lam)
    psf_store["I_te_%d" % int(lam * 1000)] = I_te
    psf_store["I_bpm_%d" % int(lam * 1000)] = I_bpm
    # on-axis intensity vs z for both fields (Fig. 2e-style raw data)
    psf_store["Iz_te_%d" % int(lam * 1000)] = onaxis_profile(E_te, lam)
    psf_store["Iz_bpm_%d" % int(lam * 1000)] = onaxis_profile(E_bpm, lam)
    # I(r, z) tile maps around the focus for BOTH fields, on the same
    # window as run_verify's verify_rzmap.npz (rzmap_* config keys):
    # the BPM only crosses the relief; from the exit plane rs_focus
    # (RS-I) reaches every (r0, z), so the BPM tile = rigorous relief
    # + identical propagation -- directly comparable to fig_rz_tiles.
    # npz keys (contract for make_plots.py): rz_r0grid, rz_zgrid,
    # Irz_te_<nm>, Irz_bpm_<nm>.
    if "rz_r0grid" not in psf_store:
        psf_store["rz_r0grid"] = np.linspace(
            0.0, cfg.get("rzmap_r_max_um", 20.0),
            int(cfg.get("rzmap_r_points", 41)))
        span = cfg.get("rzmap_z_span_um", 1000.0)
        psf_store["rz_zgrid"] = F + np.linspace(
            -span, span, int(cfg.get("rzmap_z_points", 121)))
    rz_r0, rz_z = psf_store["rz_r0grid"], psf_store["rz_zgrid"]
    t_rz = time.time()
    for tag_f, E_f in (("te", E_te), ("bpm", E_bpm)):
        # stored shape (nz, nr) -- SAME orientation as run_verify's
        # verify_rzmap.npz maps, so the plotting code is shared
        M = np.empty((rz_z.size, rz_r0.size))
        for iz, zz in enumerate(rz_z):
            M[iz, :] = np.abs(rs_focus(E_f, lam, zz, rz_r0)) ** 2
        psf_store["Irz_%s_%d" % (tag_f, int(lam * 1000))] = M
    log("  r-z tiles (%dx%d, TE+BPM) done (%.0fs)"
        % (rz_r0.size, rz_z.size, time.time() - t_rz))
    I0t, ft, et = metrics_from_profile(I_te)
    I0b, fb, eb = metrics_from_profile(I_bpm)
    results["eff_te"].append(float(et))
    results["eff_bpm"].append(float(eb))
    results["fwhm_te"].append(float(ft))
    results["fwhm_bpm"].append(float(fb))
    results["field_overlap"].append(float(ov))
    log("lam=%.2f um: overlap=%.3f | eff TE=%.3f BPM=%.3f | "
        "FWHM TE=%.2f BPM=%.2f um  (%.0fs)"
        % (lam, ov, et, eb, ft, fb, time.time() - t0))

out_json = os.path.join(run_dir, "bpm_validation.json")
json.dump(results, open(out_json, "w"), indent=1)
np.savez(os.path.join(run_dir, "bpm_psf.npz"), **psf_store)
log("saved bpm_validation.json, bpm_psf.npz")
log("reading: eff/FWHM differences TE vs BPM = the thin-element model "
    "error; overlap ~ 1 means the phase-screen model is faithful.")
log("next:  python %s %s   (adds fig_bpm_psf.png)"
    % (os.path.join("02_validation_rs", "make_plots.py"),
       os.path.relpath(run_dir)))