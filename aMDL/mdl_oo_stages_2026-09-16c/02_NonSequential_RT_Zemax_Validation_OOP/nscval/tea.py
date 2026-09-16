"""Scalar (thin-element) diffraction efficiencies -- the references the
RCWA numbers are compared with.

A one-dimensional phase grating of period P at normal incidence sends
into order m the fraction

    eta_m = | (1/P) INT_0^P exp[ i phi(x) - 2 pi i m x / P ] dx |^2

of the transmitted power (Goodman, Introduction to Fourier Optics,
Sec. 4.5 / the thin sinusoidal-grating derivation generalized to any
phase profile). For a piecewise-constant profile (N steps of width
P/N, phase phi_k on step k) the integral is a finite sum,

    eta_m = | (1/N) SUM_k exp(i phi_k) exp(-2 pi i m (k + 1/2)/N) |^2
            * sinc^2(m / N),                  sinc(x) = sin(pi x)/(pi x),

the sinc^2 being the flat-step factor (the same factor the design FOM
carries per ring, there with the kernel ramp instead of the order
number). For a continuous sawtooth of phase depth 2 pi p (a blaze with
p = (n - n_env) d / lam) the classical result is

    eta_m = sinc^2(m - p).

These are the ONLY approximations the ladder tests: the RCWA DLLs solve
Maxwell's equations for the same profiles.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

FloatVec = np.ndarray


def blaze_orders(orders: Sequence[int], p: float) -> FloatVec:
    """sinc^2(m - p) for each order m; p = (n - n_env) d / lam."""
    m = np.asarray(orders, dtype=float)
    return np.sinc(m - p) ** 2


def staircase_phases(n_steps: int, p: float) -> FloatVec:
    """Phases of an N-step equal staircase approximating a blaze of
    depth p waves: step k (k = 0..N-1) carries phi_k = 2 pi p k / N
    (heights increase toward +x; the RCWA 'descending toward +x'
    convention only flips the sign of m)."""
    k = np.arange(n_steps)
    return 2.0 * np.pi * p * k / n_steps


def profile_orders(phases: FloatVec, orders: Sequence[int]) -> FloatVec:
    """eta_m of a piecewise-constant profile with the given step phases
    (equal step widths), for each order m -- the general formula above."""
    phi = np.asarray(phases, dtype=float)
    N = phi.size
    k = np.arange(N) + 0.5
    out = np.empty(len(orders))
    for j, m in enumerate(orders):
        amp = np.mean(np.exp(1j * phi) * np.exp(-2j * np.pi * m * k / N))
        out[j] = abs(amp) ** 2 * np.sinc(m / N) ** 2
    return out


def staircase_orders(n_steps: int, p: float, orders: Sequence[int]) -> FloatVec:
    """eta_m of the N-step staircase blaze of depth p waves."""
    return profile_orders(staircase_phases(n_steps, p), orders)


def design_order(p: float) -> int:
    """The order a blaze of depth p waves is meant for: round(p)."""
    return int(round(p))


def waves_of_depth(depth_um: float, lam_um: float, n_grate: float,
                   n_env: float = 1.0) -> float:
    """p = (n - n_env) d / lam."""
    return (n_grate - n_env) * depth_um / lam_um
