"""profiles -- ring-height profiles as lamellar layer stacks, and the test
structures the solver is validated on.

An MDL zone is N rings of width DELTA with heights h_i (resist, index n)
on a substrate of the same index, relief toward the air side. Light comes
from the substrate: incident half space n, exit half space 1. Slicing by
height: the distinct heights t_1 < ... < t_K (plus 0) define K layers;
layer k spans (t_{k-1}, t_k] and is resist where h(x) >= t_k, air
elsewhere -- so the layer count is the number of DISTINCT heights in the
zone (<= the ring count), never the number of quantization levels.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

Heights = Union[Sequence[float], np.ndarray]

from .rcwa1d import Grating1D, Interval, Layer


def staircase(heights_um: Heights, step_um: float, n_grate: complex, n_inc: complex,
              n_out: complex = 1.0, period_um: float = 0.0, n_void: complex = 1.0,
              from_substrate: bool = True) -> Grating1D:
    """N steps of width step_um with the given heights (resist n_grate, the
    space above a step filled with n_void), light from the n_inc side,
    exiting into n_out. period_um = N step_um unless given (a longer period
    leaves height 0 over the rest). from_substrate: the light meets the
    base of the relief first (the lens: substrate -> resist -> air); False
    = it meets the tips first (air -> relief -> substrate), i.e. the layer
    stack is reversed."""
    h = np.asarray(heights_um, float)
    N = h.size
    P = float(period_um) if period_um else N * float(step_um)
    levels = sorted(set(float(v) for v in h if v > 0.0))
    layers: List[Layer] = []
    t_prev = 0.0
    for t in levels:
        ivs: List[Interval] = []
        for i in range(N):
            x0, x1 = i * step_um, (i + 1) * step_um
            ivs.append((x0, x1, n_grate if h[i] >= t - 1e-12 else n_void))
        if P > N * step_um:
            ivs.append((N * step_um, P, n_void))
        layers.append(Layer(t - t_prev, merge(ivs)))
        t_prev = t
    if not from_substrate:
        layers = layers[::-1]
    return Grating1D(P, n_inc, n_out, layers)


def merge(ivs: List[Interval]) -> List[Interval]:
    """Adjacent intervals of equal index fused (fewer Fourier terms to sum)."""
    out: List[Interval] = []
    for iv in ivs:
        if out and out[-1][2] == iv[2] and abs(out[-1][1] - iv[0]) < 1e-12:
            out[-1] = (out[-1][0], iv[1], iv[2])
        else:
            out.append(iv)
    return out


def sawtooth(period_um: float, depth_um: float, n_grate: complex, n_inc: complex, n_out: complex = 1.0,
             n_slices: int = 40, descending: bool = True, n_void: complex = 1.0,
             from_substrate: bool = True) -> Grating1D:
    """A continuous blaze approximated by n_slices equal-height slices
    (midpoint widths). descending: height falls with x (phase decreasing,
    blaze into m < 0 in the solver's convention)."""
    xs = (np.arange(n_slices) + 0.5) / n_slices
    h = depth_um * (1.0 - xs if descending else xs)
    return staircase(h, period_um / n_slices, n_grate, n_inc, n_out, n_void=n_void,
                     from_substrate=from_substrate)


def flat(thickness_um: float, n_film: complex, n_inc: complex, n_out: complex, period_um: float = 1.0
         ) -> Grating1D:
    """A homogeneous film (Fresnel / thin-film check)."""
    return Grating1D(period_um, n_inc, n_out, [Layer(thickness_um, [(0.0, period_um, n_film)])])
