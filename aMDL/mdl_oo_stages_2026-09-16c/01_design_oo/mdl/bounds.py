"""Upper bound of J_w(F) from pairwise coherence (paper Eqs. S14-S15).

The bound asks how coherent two rings can be made at best, for every
pair, and contracts that pairwise map with the paraxial aperture
weights; it is a ceiling for the continuous-band objective of any
design with the given geometry, band, height range and material,
independent of the optimizer. The candidate-anchor evaluation was
brute-force validated to <= 3 % (validate_bound.py).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .material import C_UM_PER_S, IndexModel, n_az4562


def pairwise_bound_matrix(diameter_um: float, na: float, lam_min_um: float,
                          lam_max_um: float, h_max_um: float, dh_um: float,
                          n_rho: int = 256, n_wavelengths: int = 512,
                          n_func: IndexModel = n_az4562,
                          n_candidates: int = 4
                          ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pairwise coherence map B[i, j] = max_dm Re < J_w(rho_i, rho_j) >_w.

    The paper's Fig. 1b/1c quantity (Eq. S14): for each ring pair the
    band-averaged mutual coherence maximized over the height difference
    dm in gray levels. max Re J depends on the heights only through
    delta_h = h1 - h2 = dm dh,

        max Re J = max_dm < cos( (w/c) [(n(w) - 1) dh dm + (r1 - r2)] ) >_w.

    A brute-force traversal over dm with coarse w sampling aliases badly
    (path differences r1 - r2 reach hundreds of um), so only CANDIDATE
    dm values near the stationary points are evaluated on a dense w
    grid: phase-matched dm* = -(r1 - r2)/((n_bar - 1) dh), group-delay-
    matched (group index), and the clipped boundaries +-M, each with a
    window wide enough to tune the carrier phase through 2 pi. Pairs
    whose group-delay residual exceeds a cut are ~0 and reported as 0.

    Returns
    -------
    rho_norm : (n_rho,)      rho / R in (0, 1]
    B : (n_rho, n_rho)       values in [0, 1]
    """
    R = 0.5 * diameter_um
    F = R * np.sqrt(1.0 / na ** 2 - 1.0)
    M = int(round(h_max_um / dh_um))

    w_min = 2.0 * np.pi * C_UM_PER_S / lam_max_um
    w_max = 2.0 * np.pi * C_UM_PER_S / lam_min_um
    omega = np.linspace(w_min, w_max, n_wavelengths)
    lam = 2.0 * np.pi * C_UM_PER_S / omega
    k = 2.0 * np.pi / lam                                 # (Nw,)
    n = n_func(lam)
    kn = k * (n - 1.0)                                    # (Nw,)
    n_bar = float(np.mean(n))

    rho = (np.arange(n_rho) + 0.5) * (R / n_rho)
    r = np.sqrt(rho ** 2 + F ** 2)

    # geometric path difference for pairs (unique upper triangle incl diag)
    p_idx, q_idx = np.triu_indices(n_rho)
    Lgeo = r[p_idx] - r[q_idx]                            # (Np,) [um] >= 0 dirs mixed
    Np = Lgeo.size

    # group index (what actually governs broadband coherence of a height step)
    n_group = n + omega * np.gradient(n, omega)
    ng_bar = float(np.mean(n_group))

    # candidate anchors: phase-matched, group-delay-matched, and the clipped
    # boundary +-M; around each anchor a window wide enough to tune the
    # carrier phase through a full 2*pi (phase step/level ~ k*(n-1)*dh)
    dm_phase = np.rint(-Lgeo / ((n_bar - 1.0) * dh_um)).astype(np.int64)
    dm_group = np.rint(-Lgeo / ((ng_bar - 1.0) * dh_um)).astype(np.int64)
    k_bar = float(np.mean(k))
    win = max(n_candidates,
              int(np.ceil(2.0 * np.pi / (k_bar * (n_bar - 1.0) * dh_um))) + 2)

    best = np.zeros(Np)   # max_dm <cos> is essentially never negative
    cand_arrays = []
    for anchor in (dm_phase, dm_group,
                   np.where(Lgeo >= 0, -M, M).astype(np.int64)):
        for o in range(-win, win + 1):
            cand_arrays.append(np.clip(anchor + o, -M, M))

    # deduplicate work by evaluating each candidate array; skip pairs whose
    # *group-delay* residual is large (their true contribution is ~0, and
    # skipping keeps the w grid alias-free for what we do evaluate)
    d_inv_lam = 1.0 / lam_min_um - 1.0 / lam_max_um       # [1/um]
    L_GROUP_CUT = max(12.0, 8.0 / d_inv_lam)              # [um]
    for cand in cand_arrays:
        dh_eff = cand * dh_um                             # (Np,)
        gd_res = (ng_bar - 1.0) * dh_eff + Lgeo           # group-delay path
        sel = np.abs(gd_res) <= L_GROUP_CUT
        if not np.any(sel):
            continue
        dh_s, Lg_s = dh_eff[sel], Lgeo[sel]
        acc = np.zeros(dh_s.size)
        chunk = 64
        for s in range(0, n_wavelengths, chunk):
            kk = k[s:s + chunk]
            kkn = kn[s:s + chunk]
            acc += np.cos(np.multiply.outer(dh_s, kkn)
                          + np.multiply.outer(Lg_s, kk)).sum(axis=1)
        best[sel] = np.maximum(best[sel], acc / n_wavelengths)

    # scatter back to full symmetric matrix
    B = np.empty((n_rho, n_rho))
    B[p_idx, q_idx] = best
    B[q_idx, p_idx] = best
    return np.asarray(rho / R, dtype=np.float64), B


def upper_bound_jf(diameter_um: float, na: float, lam_min_um: float,
                   lam_max_um: float, h_max_um: float, dh_um: float,
                   n_rho: int = 256, n_wavelengths: int = 512,
                   n_func: IndexModel = n_az4562, n_candidates: int = 4) -> float:
    """max J_w(F) upper bound (Eq. S15): the pairwise coherence matrix
    contracted with the paraxial aperture weights
    w(rho) = (2F/R^2) rho drho / r (normalized to unit sum)."""
    R = 0.5 * diameter_um
    F = R * np.sqrt(1.0 / na ** 2 - 1.0)
    rho_norm, B = pairwise_bound_matrix(
        diameter_um, na, lam_min_um, lam_max_um, h_max_um, dh_um,
        n_rho=n_rho, n_wavelengths=n_wavelengths, n_func=n_func,
        n_candidates=n_candidates)
    rho = rho_norm * R
    r = np.sqrt(rho ** 2 + F ** 2)
    wgt = (2.0 * F / R ** 2) * rho * (R / n_rho) / r
    wgt = wgt / np.sum(wgt)
    return float(wgt @ B @ wgt)


# ---------------------------------------------------------------------------
# Optimizers (paper S2-2): GA + HJA blocks -> Smooth -> Gradient
# ---------------------------------------------------------------------------
