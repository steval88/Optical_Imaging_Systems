"""
Scalar Rayleigh-Sommerfeld propagation of the staircase MDL -- the one
physics kernel behind every stage-2a number (run_verify, mtf_verify and,
through ``zval.references``, the Zemax cross-checks).

This is the validation approach of the reference paper (Xiao et al. [4],
Fig. 2e), whose ref. 46 for it is Goodman [1]. Three steps:

STEP 1: exit field via the thin-element approximation (TEA)
---------------------------------------------------------------------------
The lens is a staircase of N concentric rings, ring i of width DELTA
covering [i*DELTA, (i+1)*DELTA) with height h_i = m_i * dh in resist of
index n(lambda). For a unit-amplitude plane wave at normal incidence the
field just after the relief is a pure phase screen

    U0(rho) = exp[ i k (n(lambda) - 1) h(rho) ],     k = 2*pi/lambda,

i.e. each ray accumulates the optical-path difference (n-1)h of the
resist column it crosses; intra-relief diffraction is neglected. This is
the phase-transformation model of Goodman [1] Ch. 5 (3rd ed. Sec. 5.1)
and the universal model for multilevel diffractive optics since Swanson
[6]. Its accuracy for THIS geometry (H ~ 15 um relief, >= 2 um rings,
NA ~ 0.1) is quantified numerically by bpm_validate.py (beam propagation
THROUGH the finite-thickness staircase: 1-3 % agreement in focal
efficiency).

STEP 2: propagation by the first Rayleigh-Sommerfeld solution (RS-I)
---------------------------------------------------------------------------
The field at observation point P0 = (r0, z) behind a planar aperture
carrying the field U0(P1) is (Goodman [1] 3rd ed. Sec. 3.5, first
Rayleigh-Sommerfeld solution U_I; Born & Wolf [2] Sec. 8.11):

    U(P0) = (1 / i*lambda) * INT_aperture U0(P1)
              * exp(i k r01) / r01 * cos(theta)  dS               (RS-I)

with r01 = |P0 - P1| and the obliquity factor cos(theta) = z / r01.
Substituting cos(theta) gives the kernel actually coded below:

    U(P0) = (z / i*lambda) * INT U0(P1) * exp(i k r01) / r01^2  dS.

RS-I is an EXACT solution of the scalar Helmholtz equation for a plane
screen with Kirchhoff boundary values -- no Fresnel or paraxial
approximation is made at this step (direct-integration numerics and
accuracy: Shen & Wang [3]).

STEP 3: axisymmetric (Bessel) reduction of the RS-I integral
---------------------------------------------------------------------------
Lens and illumination are rotationally symmetric, so the 2-D aperture
integral reduces to 1-D. In polar aperture coordinates (rho, phi),

    r01 = sqrt( z^2 + rho^2 + r0^2 - 2*rho*r0*cos(phi) ).

Writing rbar = sqrt(z^2 + rho^2 + r0^2) and expanding to first order in
the cross term (valid for rho*r0 << rbar^2),

    r01 ~ rbar - rho*r0*cos(phi) / rbar          (phase),
    1/r01^2 ~ 1/rbar^2                           (amplitude),

the azimuthal integral is the Bessel identity
INT_0^{2pi} exp(-i a cos(phi)) dphi = 2*pi*J0(a) (DLMF 10.9.2 / Abramowitz
& Stegun 9.1.21 [5]), giving the formula of ``RSPropagator.psf``:

    U(r0, z) = (z / i*lambda) * SUM_i U0_i * J0(k*rho_i*r0/rbar_i)
               * exp(i k rbar_i) / rbar_i^2 * (2*pi*rho_i*DELTA)

ACCURACY OF THE REDUCTION. The neglected next-order phase term is
bounded by k*(rho*r0)^2 / (2*rbar^3): worst case for the S3 geometry
(rho = R = 5.12 mm, r0 = 30 um, z = F = 50.94 mm, lambda = 0.4 um)
~1.4e-3 rad -- negligible. ON AXIS (r0 = 0, ``RSPropagator.onaxis``) the
reduction is EXACT: no cross term exists and J0(0) = 1.

DISCRETIZATION -- ring quadrature (revised 2026-09-08)
---------------------------------------------------------------------------
U0 is exactly piecewise-constant per ring (staircase), so the aperture
integral is a sum of ring integrals

    SUM_i U0_i INT_{rho_i - DELTA/2}^{rho_i + DELTA/2}
               J0(k rho r0/rbar) e^{ik rbar(rho)} / rbar^2  rho d rho.

The KERNEL is not constant across a ring: its phase k*rbar(rho) ramps by
k*DELTA*rho/rbar, an optical path of DELTA*rho/rbar = 0.200 um at the
rim of the S3 geometry -- half a wave at 400 nm. A flat ring cannot
follow that ramp; the light it fails to blaze is the staircase
quantization loss (Swanson [6]), locally sinc^2(DELTA rho/(lambda rbar)):
1.00 on axis, 0.81 at R/2, 0.41 at the rim at 400 nm. The MIDPOINT rule
(one kernel sample per ring at rho_i -- the discretization of the paper's
Eq. 4) does not see the ramp and OVERSTATES the focal field: on the s3
softmin design the midpoint focal PSF is 1.5-3 % too narrow at 400-750
nm and its on-axis intensity too high by up to ~1.5x at 400 nm.

The ring integral of a linear phase ramp is analytic; with the slowly
varying amplitude and J0 held at the ring centre,

    INT_ring (...) rho d rho  ~  [midpoint term] * sinc(DELTA rho_i /
                                                 (lambda rbar_i)),
    sinc(x) = sin(pi x)/(pi x)   (numpy convention),

exact to first order in the ramp (curvature of rbar across DELTA and the
J0 variation, k DELTA r0/rbar <= 0.02 rad in the window, negligible).
Validated against a 0.125 um sub-ring quadrature on a synthetic s3-like
lens: peak within 0.15 % (409 nm) .. 0.02 % (1125 nm), FWHM within
0.001 um, max profile deviation 1.4e-3 of the peak -- versus 50 % peak
error for the midpoint rule at 409 nm. The factor is applied to the
DESIGN field (``quad = "sinc"``, the default; ``"midpoint"`` reproduces
pre-2026-09-08 numbers) and NOT to the ideal-lens references: the ideal
element carries the continuous hyperbolic phase, which cancels the
kernel ramp at the focus, so the midpoint sum is already exact for it.
Independent check: OpticStudio's Huygens PSF on the hybrid system agrees
with the sub-ring quadrature to corr 1.0000 and FWHM 0.6 % at 400-1100
nm (findings doc, 2026-09-07/08). Since 2026-09-15 the design FOM
(``mdl.MDLProblem`` tables, ``ring_quadrature = "sinc"``) folds the same
factor into G[w, i].

REFERENCES
---------------------------------------------------------------------------
[1] J. W. Goodman, Introduction to Fourier Optics, 3rd ed. (2005): Sec.
    3.5 (Rayleigh-Sommerfeld), 2.1.5 (Fourier-Bessel), 5.1 (thin phase
    transformations); 1st ed. (1968) pp. 38-53 is ref. 46 of [4].
[2] M. Born & E. Wolf, Principles of Optics, 7th ed. (1999), Sec. 8.11.
[3] F. Shen & A. Wang, Appl. Opt. 45, 1102-1110 (2006).
[4] Y. Xiao et al., Light Sci. Appl. 11, 323 (2022).
[5] NIST DLMF Eq. 10.9.2; Abramowitz & Stegun Eq. 9.1.21.
[6] G. J. Swanson, MIT Lincoln Laboratory Technical Report 854 (1989).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Union

import numpy as np
from numpy import pi
from scipy.special import j0

Quadrature = Literal["sinc", "midpoint"]
FloatVec = np.ndarray          # 1-D float64
ComplexVec = np.ndarray        # 1-D complex128
FloatMat = np.ndarray          # 2-D float64

if TYPE_CHECKING:              # import cycle only matters to the type checker
    from .design import DesignState


class RSPropagator:
    """RS-I propagation of one staircase design (STEPS 1-3 above).

    The design enters once, through the constructor; only the quantities
    that vary between calls (wavelength, plane z, observation grid) are
    method parameters, so the heights of one design can never be mixed
    with the radii of another.

    Attributes
    ----------
    h       ring heights h_i                                          [um]
    rho     ring centre radii rho_i = (i + 1/2) DELTA                  [um]
    delta   ring width DELTA                                          [um]
    R       aperture radius                                           [um]
    F       design focal length (plane of the PSF metrics)            [um]
    n_func  resist index n(lam_um)
    quad    ring rule for the DESIGN field: "sinc" (analytic ring
            integral of the kernel ramp) | "midpoint" (paper Eq. 4)
    """

    def __init__(self, h: FloatVec, rho: FloatVec, delta: float, R: float,
                 F: float, n_func: Callable[[float], float],
                 quad: Quadrature = "sinc") -> None:
        if quad not in ("sinc", "midpoint"):
            raise ValueError("quad must be 'sinc' or 'midpoint', got %r" % quad)
        self.h: FloatVec = np.asarray(h, dtype=float)
        self.rho: FloatVec = np.asarray(rho, dtype=float)
        self.delta: float = float(delta)
        self.R: float = float(R)
        self.F: float = float(F)
        self.n_func = n_func
        self.quad: Quadrature = quad

    @classmethod
    def from_design(cls, state: "DesignState") -> "RSPropagator":
        """The propagator of a loaded run folder (rsval.design.DesignState),
        on the run's ``rs_ring_quadrature``."""
        return cls(state.h, state.rho, state.delta, state.R, state.F,
                   state.prob.n_func, state.cfg.rs_ring_quadrature)

    # -- STEP 1 ---------------------------------------------------------------
    def exit_field(self, lam: float) -> ComplexVec:
        """U0_i = exp[i k (n(lam) - 1) h_i]: the thin-element exit field
        (Goodman [1] Sec. 5.1; Swanson [6])."""
        n = float(self.n_func(lam))
        k = 2 * pi / lam
        return np.exp(1j * k * (n - 1.0) * self.h)

    # -- DISCRETIZATION -------------------------------------------------------
    def ring_factor(self, lam: float, rb: FloatVec) -> Union[float, FloatVec]:
        """sinc(DELTA rho_i / (lam rbar_i)) per ring for the design field
        (1.0 in midpoint mode). ``rb`` = rbar_i on the same rings as rho."""
        if self.quad == "midpoint":
            return 1.0
        return np.sinc(self.delta * self.rho / (lam * rb))

    # -- STEP 2 (on axis, exact) ------------------------------------------------
    def onaxis(self, lam: float, zgrid: FloatVec) -> ComplexVec:
        """U(0, z) for each z of ``zgrid``: the unapproximated RS-I integral
        (rbar = sqrt(z^2 + rho^2) is the true distance, J0(0) = 1),

            U(0,z) = (z/(i lam)) SUM_i U0_i e^{ik rbar_i}/rbar_i^2
                     * 2 pi rho_i DELTA * ring_factor_i.
        """
        E0 = self.exit_field(lam)
        k = 2 * pi / lam
        rho, drho = self.rho, self.delta
        out = np.empty(zgrid.size, dtype=complex)
        for iz, z in enumerate(zgrid):
            rb = np.sqrt(z * z + rho * rho)
            integ = E0 * np.exp(1j * k * rb) / rb ** 2 * rho * self.ring_factor(lam, rb)
            out[iz] = (z / (1j * lam)) * 2 * pi * drho * np.sum(integ)
        return out

    # -- STEPS 2+3 (off axis, J0-reduced) --------------------------------------------
    def psf(self, lam: float, z: float, r0grid: FloatVec) -> ComplexVec:
        """U(r0, z) on the plane z for each r0 of ``r0grid``,

            U(r0,z) = (z/(i lam)) SUM_i U0_i J0(k rho_i r0 / rbar_i)
                      * e^{ik rbar_i}/rbar_i^2 * 2 pi rho_i DELTA
                      * ring_factor_i,        rbar_i = sqrt(z^2+rho_i^2+r0^2).
        """
        E0 = self.exit_field(lam)
        k = 2 * pi / lam
        rho, drho = self.rho, self.delta
        out = np.empty(r0grid.size, dtype=complex)
        for ir, r0 in enumerate(r0grid):
            rb = np.sqrt(z * z + rho * rho + r0 * r0)
            integ = E0 * j0(k * rho * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 \
                * rho * self.ring_factor(lam, rb)
            out[ir] = (z / (1j * lam)) * 2 * pi * drho * np.sum(integ)
        return out

    def intensity_map(self, lam: float, zgrid: FloatVec, r0grid: FloatVec) -> FloatMat:
        """I(z, r0) = |U(r0, z)|^2 on the (zgrid x r0grid) window: one
        J0-reduced RS-I integral over all N rings per pixel."""
        M = np.empty((zgrid.size, r0grid.size))
        for iz, z in enumerate(zgrid):
            M[iz] = np.abs(self.psf(lam, z, r0grid)) ** 2
        return M

    # -- ideal-lens references (midpoint sum, exact for a continuous phase) ------
    def ideal_peak(self, lam: float) -> float:
        """|U_ideal(0, F)| of an IDEAL lens (same R, F, lam): the exact
        hyperbolic phase exp[-ik(sqrt(rho^2+F^2)-F)] makes the RS-I
        integrand stationary at the focus. Normalizes ``strehl_like``.
        No ring factor (see DISCRETIZATION)."""
        k = 2 * pi / lam
        rho, drho, F = self.rho, self.delta, self.F
        rb = np.sqrt(F * F + rho * rho)
        integ = np.exp(-1j * k * (rb - F)) * np.exp(1j * k * rb) / rb ** 2 * rho
        return abs((F / (1j * lam)) * 2 * pi * drho * np.sum(integ))

    def ideal_profile(self, lam: float, r0grid: FloatVec) -> FloatVec:
        """|U_ideal(r0, F)|^2 of the IDEAL lens on the focal plane, by the
        same J0-reduced RS-I as ``psf``. Reference of ``strehl_shape``
        (paper Fig. 4f convention) and of the diffraction-limit MTF.
        No ring factor (see DISCRETIZATION)."""
        k = 2 * pi / lam
        rho, drho, F = self.rho, self.delta, self.F
        rb0 = np.sqrt(rho * rho + F * F)
        E0 = np.exp(-1j * k * (rb0 - F))
        out = np.empty(r0grid.size, dtype=complex)
        for ir, r0 in enumerate(r0grid):
            rb = np.sqrt(F * F + rho * rho + r0 * r0)
            integ = E0 * j0(k * rho * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 \
                * rho
            out[ir] = (F / (1j * lam)) * 2 * pi * drho * np.sum(integ)
        return np.abs(out) ** 2

    # -- convenience ------------------------------------------------------------------
    @property
    def incident_power(self) -> float:
        """pi R^2: the power of the unit-amplitude plane wave over the aperture."""
        return pi * self.R * self.R

    def rim_ramp_um(self) -> float:
        """DELTA R / sqrt(R^2 + F^2): kernel path ramp across the outermost ring."""
        return float(self.delta * self.R / np.sqrt(self.R * self.R + self.F * self.F))

    def __repr__(self) -> str:
        return ("RSPropagator(N=%d, delta=%.3f um, R=%.1f um, F=%.1f um, quad=%s)"
                % (self.rho.size, self.delta, self.R, self.F, self.quad))
