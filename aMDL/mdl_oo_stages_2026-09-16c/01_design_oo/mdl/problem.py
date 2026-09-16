"""The MDL design problem: geometry, band, phasor tables, objective.

Physical model (Xiao et al., Light Sci. Appl. 11:323, 2022, Eqs. 3-4;
their ref. 45 = Thompson, Moran & Swenson, *Interferometry and Synthesis
in Radio Astronomy*, 3rd ed., Eq. 15.32)
=======================================================================

The lens is a set of N concentric rings of equal width DELTA; ring i
covers [i DELTA, (i+1) DELTA) and carries one quantized height

    h_i = m_i dh,      m_i in {0, ..., M}                (gray levels)

on a flat substrate. A unit-amplitude plane wave through the relief is
a pure phase screen (thin-element approximation),

    phi_lens(rho, w) = (w/c) (n(w) - 1) h(rho),

and the phase distortion with respect to the ideal hyperbolic profile
that would focus at distance F is

    dphi(rho, w) = phi_lens(rho, w) + (w/c) (sqrt(rho^2 + F^2) - F).   (A)

Note that dphi contains the PROPAGATION phase k r(rho) of the ring to
the focus, r(rho) = sqrt(rho^2 + F^2), k = w/c = 2 pi / lam.

The paper's mutual intensity of the exit field is the frequency average
(their Eq. 3)

    J_w(rho_1, rho_2) = < exp(i dphi(rho_1, w)) exp(-i dphi(rho_2, w)) >_w

and its propagation to the focal point is Zernike's law for mutual
intensity (TMS Eq. 15.32) with the two observation points merged, the
propagation phase already inside dphi, no obliquity factor, and the
prefactor lam^-2 replaced by F^2 / (pi^2 R^4) (their Eq. 4):

    J_w(F) = F^2/(pi^2 R^4) INT INT_Sigma J_w(rho_1, rho_2)
                                 dSigma_1 dSigma_2 / (r_1 r_2).          (B)

Substituting Eq. 3 into (B) and exchanging the frequency average with
the two surface integrals -- both are finite linear operations -- the
product kernel (1/r_1)(1/r_2) with conjugate integrands is the modulus
squared of ONE integral, so

    J_w(F) = < |U(w)|^2 >_w,                                            (C)

    U(w) = (F / pi R^2) INT_Sigma exp(i dphi) dSigma / r
         = (2F / R^2) INT_0^R exp(i dphi(rho, w)) rho drho / r(rho).   (D)

U(w) is therefore nothing but the inner integral of the paper's Eq. 4
at one frequency: the monochromatic on-axis complex amplitude of a
unit-amplitude plane wave through the phase screen, in the paper's
normalization (dimensionless; no 1/(i lam), no obliquity; a perfect
achromat gives |U| = 1 at every w because INT rho drho / r = sqrt(R^2 +
F^2) - F ~ R^2 / 2F). Averaging |U|^2 over w is then an average of
Strehl-like ratios, one per frequency, exactly as the paper's F^2/(pi^2
R^4) normalization intends.

Discretisation on the rings and the phasor tables
-------------------------------------------------
With one height per ring, (D) becomes a sum over rings, and everything
is vectorized on two lookup tables plus a quadrature factor:

    L[w, m] = exp( i k_w (n_w - 1) m dh )                 (Nw, M+1)
    G[w, i] = (2F/R^2) rho_i DELTA / r_i
              * exp( i k_w (r_i - F) ) * S[w, i]            (Nw, N)
    U(w)    = SUM_i G[w, i] L[w, m_i]                       (Nw,)

so one objective evaluation is O(Nw N) and a SINGLE-RING change is
O(Nw) (``delta_field``), which is what makes the Hooke-Jeeves search
cheap.

Ring quadrature S[w, i]  (``ring_quadrature`` = "midpoint" | "sinc")
-------------------------------------------------------------------
The paper samples the kernel of (D) ONCE per ring, at rho_i ("midpoint"
rule, S = 1). Inside a ring the lens phase is constant, but the
propagation phase k (r - F) is not: across the ring it ramps by

    dphi_ramp,i = k DELTA rho_i / r_i

(0.200 um of path at the rim of the S3 geometry, DELTA = 2 um, R = 5.12
mm, F = 50.94 mm: half a wave at 400 nm). The ring integral of that
ramp is analytic, amplitude and lens phase held at the ring centre:

    (1/DELTA) INT_{rho_i - DELTA/2}^{rho_i + DELTA/2}
        exp( i k (rho - rho_i) rho_i / r_i ) drho = sinc( DELTA rho_i / (lam r_i) )

with numpy's sinc(u) = sin(pi u)/(pi u). That is the "sinc" rule: the
same integral (D) evaluated correctly for a piecewise-constant relief
(S = 0.637 at the S3 rim at 400 nm, 0.947 at 1100 nm; S^2 is the
scalar staircase-quantization efficiency of a flat step against a
linear blaze). Measured 2026-09-08 on the S3 softmin design: the
midpoint focal field is up to 1.5x too bright on axis at 400-450 nm;
with the sinc rule the scalar Rayleigh-Sommerfeld propagation and
OpticStudio's Huygens PSF/MTF coincide (corr 1.0000, FWHM within
0.6 %). "midpoint" remains the default so that every run folder
without the key reproduces the paper's Eq. 4 and its own J values. S
is a real amplitude factor: gradients and delta updates stay exact. S
is NOT applied to the ideal-lens denominator of the overlap objective
(the continuous ideal phase cancels the ramp).

Objectives and aggregation
--------------------------
``objective`` selects the per-frequency quantity I_w:

* "onaxis"  : I_w = |U(w)|^2, Eq. (C) -- the paper's objective.
* "overlap" : I_w = eta_w, the ENCIRCLED ENERGY in a focal disc of
  radius r_enc(w), normalized to the ideal lens at the same frequency
  (a grating-coupler style mode overlap; our extension, see
  ``enable_overlap_fom``). The off-axis field uses the same kernel as
  (D) azimuthally averaged to first order in r_0 (Bessel J_0):

      U(r_0, w) = (2F/R^2) SUM_i G[w,i] L[w,m_i] J_0( k rho_i r_0 / r_i ).

``fom_mode`` selects how the I_w are aggregated into the scalar J:

* "mean"    : J = < I >_w                       (paper Eq. 2 / Eq. C)
* "geomean" : J = exp < ln(I + eps) >_w         (every line must focus)
* "softmin" : J = -(1/beta) ln < exp(-beta I) >_w, the smooth minimum
              (log-sum-exp; Boyd & Vandenberghe sec. 3.1.5). beta -> 0
              is the mean, beta -> inf the worst line; the gradient
              weights are a softmax on the worst lines (the minimax
              objective of broadband grating-coupler inverse design,
              Lalau-Keraly et al. Opt. Express 21, 21693 (2013)).

Rigorous efficiency correction (``apply_efficiency``)
----------------------------------------------------
Optionally the phasor amplitudes are weighted by sqrt of a per-(lam, rho)
RELATIVE efficiency eta_rigorous / eta_scalar from an external RCWA
sweep of the local-grating zones (see ``mdl.zones``). With the sinc
rule the scalar baseline must be sub-ring sampled, otherwise the ramp
loss is counted twice.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

from .material import C_UM_PER_S, IndexModel, n_az4562

#: gray-level design vector, one integer per ring, shape (N,)
IntVec = NDArray[np.int32]
#: continuous heights [um], shape (N,)
FloatVec = NDArray[np.float64]
#: complex field samples, shape (Nw,) or (Nw, n_r0)
Field = NDArray[np.complexfloating]

FomMode = Literal["mean", "geomean", "softmin"]
Objective = Literal["onaxis", "overlap"]
RingQuadrature = Literal["midpoint", "sinc"]


class MDLProblem:
    """Geometry, band, material and the phasor tables of one design problem.

    Parameters
    ----------
    diameter_um : float
        Lens diameter D [um]; the aperture radius is R = D/2.
    na : float
        Numerical aperture in air, NA = R / sqrt(R^2 + F^2); the focal
        length follows as F = R sqrt(1/NA^2 - 1).
    lam_min_um, lam_max_um : float
        Working band [um]. The default wavelength grid samples this band
        uniformly in ANGULAR FREQUENCY (the paper's <.>_w); use
        ``set_wavelengths`` for an explicit comb.
    ring_width_um : float
        Ring width DELTA [um]. The number of rings is N = round(R/DELTA).
    h_max_um : float
        Maximum relief height H [um].
    dh_um : float
        Height quantum dh [um]; the number of gray levels is
        M = round(H/dh), so m_i in {0, ..., M}.
    n_wavelengths : int
        Number of frequency samples of the uniform-in-w grid.
    n_func : callable
        Refractive-index model lam_um -> n (default AZ4562).
    ring_quadrature : "midpoint" | "sinc"
        Quadrature of the ring integral (module docstring). Default
        "midpoint" = the paper's Eq. 4 verbatim.

    Attributes
    ----------
    R : float            aperture radius [um]
    F : float            focal length [um]
    na : float           numerical aperture
    delta : float        ring width DELTA [um]
    N : int              number of rings
    h_max : float        maximum height H [um]
    dh : float           height quantum [um]
    M : int              maximum gray level (levels are 0..M, i.e. M+1 values)
    lam_min, lam_max : float
    n_func : callable    the index model
    ring_quadrature : str
    omega : (Nw,)        angular frequencies [rad/s], ascending
    lam : (Nw,)          wavelengths [um] of the frequency grid
    k : (Nw,)            wavenumbers 2 pi / lam [1/um]
    n : (Nw,)            refractive index at each wavelength
    Nw : int             number of frequency samples
    rho : (N,)           ring centre radii (i + 1/2) DELTA [um]
    r_ring : (N,)        distances ring centre -> focus, sqrt(rho^2 + F^2) [um]
    S : (Nw, N)          ring-quadrature factor (all ones for "midpoint")
    G : (Nw, N)          geometric phasor table, module docstring (complex)
    L : (Nw, M+1)        gray-level phasor table (complex)
    fom_mode : str       "mean" | "geomean" | "softmin"
    objective : str      "onaxis" | "overlap"
    softmin_beta : float sharpness of the softmin aggregation
    K : (Nw, n_r0, N)    overlap kernel table G[w,i] J0(...) (overlap only)
    r_enc : (Nw,)        disc radius per frequency [um] (overlap only)

    Notes
    -----
    The public methods keep the names of the original mdl_core module
    so that run_verify.py, mtf_verify.py and fom_quadrature_check.py are
    unaffected: ``field``, ``fom``, ``per_wavelength``, ``fom_from_field``,
    ``delta_field``, ``field_h``, ``fom_h``, ``grad_h``,
    ``enable_overlap_fom``, ``disable_overlap_fom``, ``apply_efficiency``,
    ``reset_efficiency``, ``set_wavelengths``. ``G`` and ``L`` may be cast
    to complex64 by the caller (``use_single_precision``) to halve the
    memory of the overlap kernel; every number reported in the findings
    documents was computed in that precision.
    """

    #: aggregation over frequencies (module docstring)
    fom_mode: FomMode = "mean"
    #: per-frequency quantity (module docstring)
    objective: Objective = "onaxis"
    #: regularizer inside the geomean logarithm
    _geo_eps: float = 1e-6
    #: softmin sharpness; with I ~ 0.1, beta = 50-300 is the useful range
    softmin_beta: float = 100.0

    def __init__(self,
                 diameter_um: float,
                 na: float,
                 lam_min_um: float,
                 lam_max_um: float,
                 ring_width_um: float,
                 h_max_um: float,
                 dh_um: float,
                 n_wavelengths: int = 25,
                 n_func: IndexModel = n_az4562,
                 ring_quadrature: RingQuadrature = "midpoint") -> None:
        if ring_quadrature not in ("midpoint", "sinc"):
            raise ValueError("ring_quadrature must be 'midpoint' or 'sinc', "
                             "got %r" % (ring_quadrature,))
        self.ring_quadrature: RingQuadrature = ring_quadrature
        self.R: float = 0.5 * diameter_um
        self.na: float = na
        self.F: float = self.R * np.sqrt(1.0 / na ** 2 - 1.0)
        self.delta: float = ring_width_um
        self.N: int = int(round(self.R / ring_width_um))
        self.h_max: float = h_max_um
        self.dh: float = dh_um
        self.M: int = int(round(h_max_um / dh_um))
        self.lam_min: float = lam_min_um
        self.lam_max: float = lam_max_um
        self.n_func: IndexModel = n_func

        # uniform sampling in angular frequency (the paper's <.>_w)
        w_min = 2.0 * np.pi * C_UM_PER_S / lam_max_um
        w_max = 2.0 * np.pi * C_UM_PER_S / lam_min_um
        self.omega: FloatVec = np.linspace(w_min, w_max, n_wavelengths)
        self.lam: FloatVec = 2.0 * np.pi * C_UM_PER_S / self.omega
        self.k: FloatVec = 2.0 * np.pi / self.lam
        self.n: FloatVec = np.asarray(n_func(self.lam), dtype=np.float64)

        self.rho: FloatVec = (np.arange(self.N, dtype=np.float64) + 0.5) * self.delta
        self._build_tables()
        self.Nw: int = n_wavelengths

    # -- readable aliases ---------------------------------------------------
    @property
    def n_rings(self) -> int:
        """N, the number of rings."""
        return self.N

    @property
    def n_levels(self) -> int:
        """M + 1, the number of admissible gray levels (0 .. M)."""
        return self.M + 1

    @property
    def n_freq(self) -> int:
        """Nw, the number of frequency samples."""
        return self.Nw

    # -- tables -----------------------------------------------------------------
    def _build_tables(self) -> None:
        """Build S, G, L (module docstring) for the current ``self.lam``.

        Also keeps the ideal-lens amplitude ``_amp_ideal[i]`` (no S: the
        ideal phase is continuous) for the overlap denominator.
        """
        r = np.sqrt(self.rho ** 2 + self.F ** 2)
        self.r_ring: FloatVec = r
        geo_phase = self.k[:, None] * (r[None, :] - self.F)
        amp = (2.0 * self.F / self.R ** 2) * self.rho * self.delta / r
        if self.ring_quadrature == "sinc":
            self.S: NDArray[np.float64] = np.sinc(
                self.delta * self.rho[None, :] / (self.lam[:, None] * r[None, :]))
        else:
            self.S = np.ones((self.lam.size, self.rho.size))
        self._amp_ideal: FloatVec = amp
        self.G: Field = amp[None, :] * self.S * np.exp(1j * geo_phase)
        kn = self.k * (self.n - 1.0)                          # [1/um]
        m = np.arange(self.M + 1)
        self.L: Field = np.exp(1j * kn[:, None] * self.dh * m[None, :])

    def set_wavelengths(self, lams_um: NDArray[np.float64]) -> "MDLProblem":
        """Rebuild the tables on an EXPLICIT wavelength list [um].

        Use this for comb objectives (e.g. the paper's 14 lines). Any
        efficiency correction and the overlap objective are rebuilt on
        the new grid. Returns self.
        """
        self.lam = np.asarray(lams_um, dtype=float)
        self.omega = 2.0 * np.pi * C_UM_PER_S / self.lam
        self.k = 2.0 * np.pi / self.lam
        self.n = np.asarray(self.n_func(self.lam), dtype=np.float64)
        self._build_tables()
        self.Nw = self.lam.size
        if hasattr(self, "_eta_table"):
            self._G_raw = self.G.copy()
            self._reapply_eta()
        if self.objective == "overlap":
            self.enable_overlap_fom(*self._overlap_cfg)
        return self

    def use_single_precision(self) -> "MDLProblem":
        """Cast G and L to complex64 (halves the overlap kernel memory).

        The design driver does this before optimizing; the overlap
        kernel inherits the dtype of G. Returns self.
        """
        self.G = self.G.astype(np.complex64)
        self.L = self.L.astype(np.complex64)
        if self.objective == "overlap":
            self.enable_overlap_fom(*self._overlap_cfg)
        return self

    # -- aggregation over frequencies -------------------------------------------
    def _aggregate(self, I: FloatVec) -> float:
        """Scalar J from the per-frequency values I, shape (Nw,)."""
        if self.fom_mode == "geomean":
            return float(np.exp(np.mean(np.log(I + self._geo_eps))))
        if self.fom_mode == "softmin":
            b = self.softmin_beta
            a = -b * (I - I.min())               # <= 0, numerically stable
            return float(I.min() - np.log(np.mean(np.exp(a))) / b)
        return float(np.mean(I))

    def _aggregate_weights(self, I: FloatVec) -> FloatVec:
        """Weights w_w such that dJ/dh = SUM_w w_w dI_w/dh, shape (Nw,)."""
        Nw = I.size
        if self.fom_mode == "geomean":
            Ieps = I + self._geo_eps
            J = float(np.exp(np.mean(np.log(Ieps))))
            return J / (Nw * Ieps)
        if self.fom_mode == "softmin":
            a = -self.softmin_beta * (I - I.min())
            e = np.exp(a)
            return e / e.sum()                   # softmax on the worst lines
        return np.full(Nw, 1.0 / Nw)

    # -- discrete design vector -------------------------------------------------
    def field(self, m_vec: IntVec) -> Field:
        """Focal field of the gray-level vector m_vec (ints 0..M, shape (N,)).

        Returns U(w), shape (Nw,), for objective "onaxis" (Eq. D of the
        module docstring), or U(w, r0_q) on the encircled-energy
        quadrature nodes, shape (Nw, n_r0), for objective "overlap".
        """
        if self.objective == "overlap":
            return np.einsum("wqi,wi->wq", self.K, self.L[:, m_vec])
        return np.sum(self.G * self.L[:, m_vec], axis=1)

    def per_wavelength(self, U: Field) -> FloatVec:
        """Per-frequency values I_w, shape (Nw,): |U|^2 or encircled eta_w."""
        if U.ndim == 2:                          # overlap field (Nw, nq)
            return np.einsum("wq,wq->w", self._enc_w,
                             np.abs(U) ** 2) / self._enc_D
        return np.abs(U) ** 2

    def fom_from_field(self, U: Field) -> float:
        """Aggregated J from a field returned by ``field`` / ``field_h``."""
        return self._aggregate(self.per_wavelength(U))

    def fom(self, m_vec: IntVec) -> float:
        """Aggregated J (fom_mode, objective) of the gray-level vector."""
        return self.fom_from_field(self.field(m_vec))

    def delta_field(self, U: Field, i: int, m_old: int, m_new: int) -> Field:
        """Field after changing ring i from level m_old to m_new -- O(Nw)."""
        dL = self.L[:, m_new] - self.L[:, m_old]
        if self.objective == "overlap":
            return U + self.K[:, :, i] * dL[:, None]
        return U + self.G[:, i] * dL

    # -- continuous heights (gradient stage) -------------------------------------
    def field_h(self, h_vec: FloatVec) -> Field:
        """Focal field of CONTINUOUS heights h_vec [um], shape (N,)."""
        kn = self.k * (self.n - 1.0)
        E = np.exp(1j * kn[:, None] * h_vec[None, :])
        if self.objective == "overlap":
            return np.einsum("wqi,wi->wq", self.K, E)
        return np.sum(self.G * E, axis=1)

    def fom_h(self, h_vec: FloatVec) -> float:
        """Aggregated J of continuous heights."""
        return self.fom_from_field(self.field_h(h_vec))

    def grad_h(self, h_vec: FloatVec) -> FloatVec:
        """Analytic gradient dJ/dh_i, shape (N,), all fom modes and objectives.

        onaxis : dI_w/dh_i = 2 Re{ conj(U_w) i kn_w G[w,i] E[w,i] }
        overlap: dI_w/dh_i = (2/D_w) SUM_q wq Re{ conj(P_wq) i kn_w K[w,q,i] E[w,i] }
        then dJ/dh = SUM_w w_w dI_w/dh with the aggregation weights.
        """
        kn = self.k * (self.n - 1.0)                     # (Nw,)
        E = np.exp(1j * kn[:, None] * h_vec[None, :])    # (Nw, N)
        if self.objective == "overlap":
            P = np.einsum("wqi,wi->wq", self.K, E)       # (Nw, nq)
            I = self.per_wavelength(P)                   # eta_w
            T = np.einsum("wq,wq,wqi->wi", self._enc_w,
                          np.conj(P), self.K)            # (Nw, N)
            dI = (2.0 * kn / self._enc_D)[:, None] \
                * np.real(1j * T * E)
        else:
            EG = self.G * E                              # (Nw, N)
            U = np.sum(EG, axis=1)                       # (Nw,)
            I = np.abs(U) ** 2
            dI = 2.0 * np.real(np.conj(U)[:, None]
                               * 1j * kn[:, None] * EG)
        w = self._aggregate_weights(I)
        return np.sum(w[:, None] * dI, axis=0)

    # -- overlap (encircled-energy) objective ------------------------------------
    def enable_overlap_fom(self, r_enc_um: Optional[float] = None,
                           n_r0: int = 24,
                           airy_factor: float = 2.0) -> "MDLProblem":
        """Switch to the encircled-energy objective (module docstring).

            eta_w = INT_0^{r_enc(w)} |U(r0,w)|^2 r0 dr0
                    / INT_0^{r_enc(w)} |U_ideal(r0,w)|^2 r0 dr0

        with U(r0, w) the Bessel-reduced field of the module docstring and
        U_ideal the same sum with all phase errors zero and no ring
        factor (continuous ideal phase). The disc is CHROMATIC by
        default, r_enc(w) = airy_factor * 0.61 lam_w / NA (the same
        shape in Airy units at every line; 1.0 = main lobe to the first
        dark ring, 2.0 = main lobe + first bright ring). A number in
        r_enc_um forces one fixed disc for all lines (not recommended:
        a lam_max-sized disc lets short lines pass with halo, measured
        eta 0.18 vs true efficiency 0.004 at 400 nm). The disc also sets
        the tolerated axial defocus, dz ~ r_enc / NA: a spot displaced
        along the axis still scores if it fits the disc, which is why a
        large airy_factor lets the optimizer trade core sharpness for
        disc energy (measured 2026-09-15: airy_factor 2.0 accepted a
        5.8 um FWHM at 700 nm with the true focus 0.56 mm short).

        The radial integral is a Gauss-Legendre quadrature on n_r0 nodes
        per frequency. Memory: K is (Nw, n_r0, N) in the dtype of G --
        14 x 24 x 2560 complex64 = 7 MB for a comb, ~250 MB for a
        500-sample continuous band. Returns self.
        """
        from numpy.polynomial.legendre import leggauss
        self._overlap_cfg: Tuple[Optional[float], int, float] = (
            None if r_enc_um is None else float(r_enc_um), int(n_r0),
            float(airy_factor))
        if r_enc_um is None:
            r_enc = airy_factor * 0.61 * self.lam / self.na  # (Nw,)
        else:
            r_enc = np.full(self.lam.size, float(r_enc_um))
        x, wgl = leggauss(int(n_r0))                     # nodes on [-1, 1]
        r0 = 0.5 * r_enc[:, None] * (x + 1.0)[None, :]   # (Nw, nq) [um]
        wq = 0.5 * r_enc[:, None] * wgl[None, :]         # GL weights
        self._enc_r0: NDArray[np.float64] = r0
        self._enc_w: NDArray[np.float64] = wq * r0       # includes r0 dr0
        self.r_enc: FloatVec = r_enc
        r = np.sqrt(self.rho ** 2 + self.F ** 2)         # (N,)
        arg = (self.k[:, None, None] * self.rho[None, None, :]
               * r0[:, :, None] / r[None, None, :])      # (Nw, nq, N)
        B = j0(arg)
        self.K: NDArray[np.complexfloating] = (self.G[:, None, :] * B).astype(self.G.dtype)
        # ideal-lens denominator: the CONTINUOUS ideal phase, no ring
        # factor S, no efficiency weight; bit-identical to |G| whenever
        # S = 1 and no efficiency table is active (all pre-2026-09-15 runs)
        if self.ring_quadrature == "midpoint" and not hasattr(self, "_eta_table"):
            ideal_amp = np.abs(self.G)
        else:
            ideal_amp = np.broadcast_to(
                self._amp_ideal.astype(self.G.real.dtype)[None, :], self.G.shape)
        Uid = np.einsum("wqi->wq", ideal_amp[:, None, :] * B)
        self._enc_D: FloatVec = np.einsum("wq,wq->w", self._enc_w, Uid ** 2)
        self.objective = "overlap"
        return self

    def disable_overlap_fom(self) -> "MDLProblem":
        """Back to the on-axis objective (frees the kernel table)."""
        self.objective = "onaxis"
        for attr in ("K", "_enc_D", "_enc_w", "_enc_r0",
                     "_overlap_cfg", "r_enc"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self

    # -- rigorous local-grating efficiency correction -----------------------------
    def apply_efficiency(self, lam_um: NDArray[np.float64],
                         r_um: NDArray[np.float64],
                         corr: NDArray[np.float64]) -> "MDLProblem":
        """Weight G by sqrt(corr), corr = eta_rigorous / eta_scalar.

        corr has shape (len(lam_um), len(r_um)) and is interpolated onto
        (self.lam, self.rho) with edge clamping. The RATIO (not the
        absolute efficiency) is required: the phasor sum over a zone's
        rings already computes the scalar zone response, so only the
        rigorous-minus-scalar difference may be injected. The table is
        stored, survives ``set_wavelengths``; ``reset_efficiency``
        reverts; calling again replaces the previous table. See
        ``mdl.zones`` for the file contract. Returns self.
        """
        if not hasattr(self, "_G_raw"):
            self._G_raw = self.G.copy()
        self._eta_table = (np.asarray(lam_um, float),
                           np.asarray(r_um, float),
                           np.asarray(corr, float))
        self._reapply_eta()
        if self.objective == "overlap":
            self.enable_overlap_fom(*self._overlap_cfg)
        return self

    def reset_efficiency(self) -> "MDLProblem":
        """Remove the efficiency correction (bare scalar model)."""
        if hasattr(self, "_G_raw"):
            self.G = self._G_raw
            del self._G_raw
        if hasattr(self, "_eta_table"):
            del self._eta_table
        if self.objective == "overlap":
            self.enable_overlap_fom(*self._overlap_cfg)
        return self

    def _reapply_eta(self) -> None:
        lam_t, r_t, eta_t = self._eta_table
        et = np.empty((self.lam.size, r_t.size))
        order = np.argsort(lam_t)
        for j in range(r_t.size):
            et[:, j] = np.interp(self.lam, lam_t[order], eta_t[order, j])
        w = np.empty((self.lam.size, self.rho.size))
        order_r = np.argsort(r_t)
        for i in range(self.lam.size):
            w[i] = np.interp(self.rho, r_t[order_r], et[i, order_r])
        self.G = self._G_raw * np.sqrt(np.clip(w, 0.0, None)).astype(
            self._G_raw.real.dtype)

    # -- conveniences -----------------------------------------------------------
    def heights_um(self, m_vec: IntVec) -> FloatVec:
        """h_i = m_i dh [um] for a gray-level vector."""
        return np.asarray(m_vec, dtype=float) * self.dh

    def levels_from_heights(self, h_vec: FloatVec) -> IntVec:
        """Round continuous heights [um] to admissible gray levels."""
        return np.clip(np.rint(np.asarray(h_vec, float) / self.dh),
                       0, self.M).astype(np.int32)

    def __repr__(self) -> str:
        return ("MDLProblem(D=%.3f mm, F=%.3f mm, NA=%.4f, N=%d rings x %.2f um, "
                "M=%d levels x %.3f um, Nw=%d in %.3f-%.3f um, %s, %s, %s)"
                % (2 * self.R / 1000, self.F / 1000, self.na, self.N,
                   self.delta, self.M, self.dh, self.Nw, self.lam.min(),
                   self.lam.max(), self.ring_quadrature, self.objective,
                   self.fom_mode))
