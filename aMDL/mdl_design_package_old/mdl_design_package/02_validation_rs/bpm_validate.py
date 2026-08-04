"""
bpm_validate.py
===============

"FDTD-lite" validation of the thin-element (zone-decomposition) model
used in both the design code and the Zemax UDS DLL.

The scalar field is propagated THROUGH the actual staircase relief with
a Crank-Nicolson beam-propagation method (BPM) in cylindrical
coordinates -- this captures intra-relief diffraction, edge shadowing
and the finite-thickness character of the 15 um structure, i.e. the
leading effects a rigorous solver would add over the thin-element
phase mask. (One-way propagation: Fresnel reflections and true vector
effects are outside scope; at NA ~ 0.1 and ring width >= 2 um they are
the next-order corrections.)

Then both exit fields -- BPM and thin-element -- are propagated to the
design focus with the same Rayleigh-Sommerfeld kernel, and the focal
metrics are compared per wavelength.

Geometry (S3 design): resist below (n(lam)), relief h(r) in [0, H],
air above. Field enters at z=0 as a unit plane wave inside the resist.

Usage:
    python bpm_validate.py [out/mdl_rings_2.txt]
"""
import json
import os
import sys
import time

import numpy as np
from numpy import pi
from scipy.linalg import solve_banded
from scipy.special import j0

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aMDL.old.mdl_design_package.mdl_core import n_az4562

T0 = time.time()
os.makedirs("out", exist_ok=True)


def log(s):
    print("[%6.1fs] %s" % (time.time() - T0, s), flush=True)


if len(sys.argv) < 2:
    raise SystemExit("usage: python bpm_validate.py runs/<run_folder>/mdl_rings_<n>.txt")
RINGS = sys.argv[1]
with open(RINGS) as fh:
    N, delta_mm = fh.readline().split()
    N, delta = int(N), float(delta_mm) * 1000.0          # um
    h_ring = np.array([float(fh.readline()) for _ in range(N)]) * 1000.0

R = N * delta
F = 50940.0                                              # S3 focal [um]
H = float(h_ring.max())                                  # relief height
log("rings: %d x %.2f um, R=%.2f mm, H=%.2f um" % (N, delta, R / 1000, H))

# BPM grid
DR = 0.05                                                # um
DZ = 0.05                                                # um
PAD = 200.0                                              # absorber [um]
r = np.arange(0.0, R + PAD, DR) + 0.5 * DR
NR = r.size
h_of_r = np.zeros(NR)
idx = np.minimum((r / delta).astype(int), N - 1)
inside = r < R
h_of_r[inside] = h_ring[idx[inside]]

# absorber (super-Gaussian loss ramp in the pad)
absorb = np.ones(NR)
pad_zone = r > R + 10.0
absorb[pad_zone] = np.exp(-((r[pad_zone] - (R + 10.0)) / (0.6 * PAD)) ** 4
                          * 0.5)

LAMS = [0.45, 0.55, 0.70, 0.85, 1.05]                    # on/off resonance


def bpm_exit_field(lam):
    """CN-BPM through the staircase; returns E(r) at z = H (in air)."""
    n_res = float(n_az4562(lam))
    k0 = 2 * pi / lam
    n_ref = 0.5 * (n_res + 1.0)
    nz = int(np.ceil(H / DZ))
    E = np.ones(NR, dtype=complex)                       # plane wave @ z=0

    # transverse operator: (1/(2 k0 n_ref)) * (d2/dr2 + (1/r) d/dr)
    # CN: (I - i dz/2 L) E+ = (I + i dz/2 L) E  with L = A + V(z)
    inv2k = 1.0 / (2.0 * k0 * n_ref)
    lo = inv2k * (1.0 / DR ** 2 - 1.0 / (2.0 * r * DR))  # sub-diag coeff
    di = inv2k * (-2.0 / DR ** 2) * np.ones(NR)
    up = inv2k * (1.0 / DR ** 2 + 1.0 / (2.0 * r * DR))  # super-diag
    # axis r ~ DR/2: Neumann (E[-1]=E[0]) -> fold sub into diag at i=0
    di0_extra = lo[0]

    for iz in range(nz):
        z = (iz + 0.5) * DZ
        n_loc = np.where(z < h_of_r, n_res, 1.0)
        V = 0.5 * k0 * (n_loc ** 2 - n_ref ** 2) / n_ref
        diag = di + V
        diag_ax = diag.copy()
        diag_ax[0] += di0_extra

        # RHS = (I + i dz/2 L) E
        rhs = E.copy()
        rhs[1:-1] += 0.5j * DZ * (lo[1:-1] * E[:-2] + diag_ax[1:-1] * E[1:-1]
                                  + up[1:-1] * E[2:])
        rhs[0] += 0.5j * DZ * (diag_ax[0] * E[0] + up[0] * E[1])
        rhs[-1] += 0.5j * DZ * (lo[-1] * E[-2] + diag_ax[-1] * E[-1])

        # banded matrix (I - i dz/2 L)
        ab = np.zeros((3, NR), dtype=complex)
        ab[0, 1:] = -0.5j * DZ * up[:-1]
        ab[1, :] = 1.0 - 0.5j * DZ * diag_ax
        ab[2, :-1] = -0.5j * DZ * lo[1:]
        E = solve_banded((1, 1), ab, rhs)
        E *= absorb
    # common phase reference: remove the uniform-propagation phase of a
    # ray through air-only (h=0) so fields compare to thin-element form
    return E


def thin_element_field(lam):
    n_res = float(n_az4562(lam))
    k0 = 2 * pi / lam
    return np.exp(1j * k0 * (n_res - 1.0) * h_of_r) * (r < R)


def rs_focus(E_r, lam, z, r0grid):
    """RS1 propagation of an axisymmetric field sampled on the BPM grid."""
    k = 2 * pi / lam
    sel = r < R + 5.0
    rr, EE = r[sel], E_r[sel]
    out = np.empty(r0grid.size, dtype=complex)
    for i, r0 in enumerate(r0grid):
        rb = np.sqrt(z * z + rr * rr + r0 * r0)
        out[i] = (z / (1j * lam)) * 2 * pi * DR * np.sum(
            EE * j0(k * rr * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 * rr)
    return out


def focus_metrics(E_r, lam):
    r0g = np.linspace(0.0, 40.0, 401)
    I = np.abs(rs_focus(E_r, lam, F, r0g)) ** 2
    Ipk = I.max()
    half = Ipk / 2
    below = np.where(I < half)[0]
    fwhm = 2 * r0g[below[0]] if below.size else np.nan
    selr = r0g <= 1.5 * fwhm
    p_in = np.trapezoid(I[selr] * 2 * pi * r0g[selr], r0g[selr])
    p_tot = pi * R * R
    return I[0], fwhm, p_in / p_tot


results = {"lam_um": LAMS, "F_um": F, "rings": RINGS,
           "eff_te": [], "eff_bpm": [], "fwhm_te": [], "fwhm_bpm": [],
           "field_overlap": []}
for lam in LAMS:
    t0 = time.time()
    E_bpm = bpm_exit_field(lam)
    E_te = thin_element_field(lam)
    # complex overlap of the two exit fields over the aperture
    sel = r < R
    w = r[sel]
    ov = abs(np.sum(np.conj(E_te[sel]) * E_bpm[sel] * w)) / np.sqrt(
        np.sum(np.abs(E_te[sel]) ** 2 * w) *
        np.sum(np.abs(E_bpm[sel]) ** 2 * w))
    I0t, ft, et = focus_metrics(E_te, lam)
    I0b, fb, eb = focus_metrics(E_bpm, lam)
    results["eff_te"].append(float(et))
    results["eff_bpm"].append(float(eb))
    results["fwhm_te"].append(float(ft))
    results["fwhm_bpm"].append(float(fb))
    results["field_overlap"].append(float(ov))
    log("lam=%.2f um: overlap=%.3f | eff TE=%.3f BPM=%.3f | "
        "FWHM TE=%.2f BPM=%.2f um  (%.0fs)"
        % (lam, ov, et, eb, ft, fb, time.time() - t0))

out_json = os.path.join(os.path.dirname(os.path.abspath(RINGS)),
                        "bpm_validation.json")
json.dump(results, open(out_json, "w"), indent=1)
log("saved %s" % out_json)
