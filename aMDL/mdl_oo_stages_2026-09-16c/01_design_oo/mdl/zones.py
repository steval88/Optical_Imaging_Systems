"""Local linear-grating decomposition and the rigorous-efficiency hook.

An MDL is a circularly chirped grating: at radius rho the zone structure
is locally a linear grating of period Lambda(rho), and focusing is the
local grating equation steering one order to F. The thin-element phasor
model ignores how efficiently each local grating actually diffracts; a
BPM audit showed the scalar model overestimates balanced-design
efficiency by ~20 % in the outer zones where Lambda is a few
wavelengths. The grating community's fix is the local linear grating
approximation: compute each zone's order efficiency RIGOROUSLY (RCWA)
and fold it back in as per-ring amplitude weights. The workflow is
file-decoupled:

  1. extract_local_gratings(prob, m)   -> list of Zone
  2. write_zone_table(path, zones, ...) -> npz for ANY rigorous solver
     (Lumerical RCWA, torcwa, grcwa, RETICOLO ...), which sweeps each
     zone over the design wavelengths and writes an efficiency table
     (contract in load_efficiency_table)
  3. tea_efficiency_table(prob, m, lams) -> the matching SCALAR baseline
  4. relative_correction_table(eta_rigorous, eta_scalar) -> corr
  5. MDLProblem.apply_efficiency(lam, r, corr) -> G scaled by sqrt(corr)

The RATIO in step 4 is essential: the phasor sum over a zone's rings
already computes the scalar zone response, so only the rigorous-minus-
scalar difference may be injected (absolute eta would count the scalar
loss twice: J collapsed ~15x on the S3 comb design when tried). With
ring_quadrature = "sinc" the scalar baseline must be sub-ring
sampled, because tea_zone_efficiency samples one height per ring
and does not contain the ring-ramp loss that S already carries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .material import IndexModel, n_az4562
from .problem import IntVec, MDLProblem

#: one sawtooth zone as a plain dict (keys documented in extract_local_gratings)
Zone = Dict[str, Any]


def extract_local_gratings(prob: MDLProblem, m_vec: IntVec,
                           reset_frac: float = 0.4,
                           min_zone_rings: int = 2) -> List[Zone]:
    """Split the design into sawtooth zones.

    A zone boundary is detected where the height jumps UP by more than
    reset_frac * h_max between adjacent rings (the fold reset of the
    local blaze); zones narrower than min_zone_rings are merged forward.
    Inner rings before the first reset form zone 0 (quasi-flat paraxial
    core; its 'period' is not a meaningful grating period).

    Returns a list of dicts, one per zone, with keys
      i0, i1       : ring index range [i0, i1)
      r_in, r_out  : zone radii [um]
      r_center     : centre radius [um]
      period_um    : zone width = local grating period Lambda
      h_um         : (i1 - i0,) height profile across the zone [um]
      opd_span_um  : sqrt(r_out^2 + F^2) - sqrt(r_in^2 + F^2) -- its
                     ratio to lam is the local order alpha_loc(lam)
    """
    if m_vec.size != prob.N:
        raise ValueError("design vector has %d rings but the problem "
                         "has N=%d -- geometry mismatch" % (m_vec.size,
                                                            prob.N))
    h = m_vec.astype(float) * prob.dh
    jumps = np.where(np.diff(h) > reset_frac * prob.h_max)[0] + 1
    bounds = [0] + [int(j) for j in jumps] + [prob.N]
    # merge too-narrow zones forward
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < min_zone_rings and b != prob.N:
            continue
        merged.append(b)
    zones = []
    for i0, i1 in zip(merged[:-1], merged[1:]):
        r_in = i0 * prob.delta
        r_out = i1 * prob.delta
        rc = 0.5 * (r_in + r_out)
        opd = (np.sqrt(r_out ** 2 + prob.F ** 2)
               - np.sqrt(r_in ** 2 + prob.F ** 2))
        zones.append(dict(i0=int(i0), i1=int(i1), r_in=r_in,
                          r_out=r_out, r_center=rc,
                          period_um=r_out - r_in,
                          h_um=h[i0:i1].copy(),
                          opd_span_um=float(opd)))
    return zones


def write_zone_table(path: str, zones: List[Zone], lams_um: NDArray[np.float64],
                     prob: Optional[MDLProblem] = None) -> str:
    """Save the zone decomposition for an external rigorous solver.

    npz contents: zone_id (Z,), r_in / r_out / r_center / period_um (Z,),
    h_profile (object array of per-zone height samples [um], one per
    ring), ring_width_um, lams_um (K,), n_real (K,). The solver's job per
    zone and wavelength: build the sawtooth h_profile[z] on period
    period_um[z], normal incidence from the substrate side, and return
    the efficiency of the FOCUSING order, the one whose deflection
    matches sin(theta) = r_center / sqrt(r_center^2 + F^2). Returns path.
    """
    z = zones
    np.savez(path,
             zone_id=np.arange(len(z)),
             r_in=np.array([q["r_in"] for q in z]),
             r_out=np.array([q["r_out"] for q in z]),
             r_center=np.array([q["r_center"] for q in z]),
             period_um=np.array([q["period_um"] for q in z]),
             h_profile=np.array([q["h_um"] for q in z], dtype=object),
             ring_width_um=(z[0]["h_um"].size and
                            (z[0]["r_out"] - z[0]["r_in"])
                            / z[0]["h_um"].size),
             lams_um=np.asarray(lams_um, float),
             n_real=(prob.n_func(np.asarray(lams_um, float))
                     if prob is not None else
                     n_az4562(np.asarray(lams_um, float))),
             allow_pickle=True)
    return path


def load_efficiency_table(path: str) -> Tuple[NDArray[np.float64],
                                              NDArray[np.float64],
                                              NDArray[np.float64]]:
    """Load an efficiency table (npz with exactly lam_um (K,), r_um (Z,),
    eta (K, Z)): raw efficiencies in [0, 1] from a solver, or the
    rigorous/scalar ratio (may exceed 1) for apply_efficiency.
    Returns (lam_um, r_um, eta)."""
    d = np.load(path)
    return d["lam_um"], d["r_um"], d["eta"]


def tea_zone_efficiency(zone: Zone, lams_um: NDArray[np.float64], F_um: float,
                        n_func: IndexModel = n_az4562) -> NDArray[np.float64]:
    """Scalar (thin-element) efficiency of one zone into its focusing
    order: |(1/Lambda) INT_zone exp{ i [k (n-1) h(x) - k x sin theta_c] } dx|^2
    with sin theta_c = r_c / sqrt(r_c^2 + F^2), sampled ONE height per
    ring (ring-midpoint rule). Returns (K,)."""
    lams = np.asarray(lams_um, float)
    n = n_func(lams)
    k = 2.0 * np.pi / lams
    nx = zone["h_um"].size
    if nx == 0:
        return np.ones_like(lams)
    dx = zone["period_um"] / nx
    x = (np.arange(nx) + 0.5) * dx
    rc = zone["r_center"]
    s = rc / np.sqrt(rc ** 2 + F_um ** 2)
    ph = (k[:, None] * (n - 1.0)[:, None] * zone["h_um"][None, :]
          - k[:, None] * s * x[None, :])
    c = np.mean(np.exp(1j * ph), axis=1)
    return np.abs(c) ** 2


def tea_efficiency_table(prob: MDLProblem, m_vec: IntVec,
                         lams_um: NDArray[np.float64], reset_frac: float = 0.4
                         ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                                    NDArray[np.float64]]:
    """Full (lam, r) scalar table over all zones: (lams (K,), r_um (Z,), eta (K, Z))."""
    zones = extract_local_gratings(prob, m_vec, reset_frac=reset_frac)
    lams = np.asarray(lams_um, float)
    r_um = np.array([z["r_center"] for z in zones])
    eta = np.stack([tea_zone_efficiency(z, lams, prob.F, prob.n_func)
                    for z in zones], axis=1)          # (K, Z)
    return lams, r_um, eta


def relative_correction_table(eta_rigorous: NDArray[np.float64],
                              eta_scalar: NDArray[np.float64],
                              eta_floor: float = 0.02,
                              corr_max: float = 2.0) -> NDArray[np.float64]:
    """corr = eta_rigorous / eta_scalar, the table apply_efficiency
    expects, both inputs (K, Z) on identical axes. Zones whose scalar
    efficiency is below eta_floor get corr = 1 (the ratio of two near-
    zeros is noise); the ratio is clipped to [0, corr_max]."""
    er = np.asarray(eta_rigorous, float)
    es = np.asarray(eta_scalar, float)
    corr = np.where(es > eta_floor, er / np.maximum(es, 1e-12), 1.0)
    return np.clip(corr, 0.0, corr_max)
