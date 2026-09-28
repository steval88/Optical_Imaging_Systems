"""rcwa1d -- rigorous coupled-wave analysis of a one-dimensional (lamellar
multilayer) grating, planar incidence, TE and TM, S-matrix recursion.

The structure is a stack of layers between two half spaces; every layer
is periodic in x with period P and piecewise constant in x (a list of
(x_start, x_end, n) intervals on [0, P)). The relief of an MDL zone is
such a stack: a staircase of rings, sliced by height (profiles.py).

Formulation (Moharam, Grann, Pommet & Gaylord, JOSA A 12, 1068 (1995),
in the scattering-matrix form of Rumpf, PIER B 35, 241 (2011), with the
inverse rule of Li, JOSA A 13, 1870 (1996) for TM):

    x-wavevectors    kx_m = n_inc sin(theta) + m lam / P,  m = -N..N,
                     normalized to k0 = 2 pi / lam (Kx = diag(kx_m))
    E   = Toeplitz of the Fourier coefficients of eps(x) = n(x)^2
    E_i = Toeplitz of the Fourier coefficients of 1 / eps(x)
    TE (E along the grooves, y):    Omega^2 = Kx^2 - E,           V = (E - Kx^2) W / lam
    TM (H along the grooves):       Omega^2 = (Kx E^-1 Kx - I) E_i^-1,  V = -E_i^-1 W / lam
        (E_i^-1 where the product eps Ex is continuous -- Li's rule;
        E^-1 where Ez = Dz / eps is continuous)
    W, lam^2 = eigenvectors / eigenvalues of Omega^2 (mode fields and
    propagation constants), layer S-matrix from W, V, exp(-lam k0 L),
    global S = star product from the incident half space through the
    layers to the exit half space.
    Efficiencies: R_m = Re(kz_ref,m) / Re(kz_inc) |r_m|^2,
                  T_m = Re(kz_trn,m) / Re(kz_inc) |t_m|^2,
    |r_m|^2 = |ex|^2 + |ez|^2 for TM (ez = -kx ex / kz), |ey|^2 for TE.
    Unpolarized = (TE + TM) / 2.

Sign convention of the order index: field ~ exp(+i kx_m x), so a
transmission function t(x) = sum c_m exp(2 pi i m x / P) diffracts into
order m with amplitude c_m -- the same m as nscval.tea.profile_orders
(a phase DEcreasing with x, the lens's converging blaze, feeds m < 0).

Validation (tests/test_rcwa1d.py): energy conservation to 1e-10 for a
lossless stack; a flat interface reproduces the Fresnel coefficients;
a thin sawtooth at P = 50 lam reproduces sinc^2(m - p) to 1 %; the
5-level staircase of the OpticStudio null test (P 5 um, 600 nm, n 1.632)
gives 0.823 in the blaze order as srg_blaze_RCWA did (2026-09-18).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla

Complex = np.complex128
Interval = Tuple[float, float, complex]        # (x_start, x_end, refractive index)


@dataclass
class Layer:
    """One lamellar layer.

    Attributes
    ----------
    thickness_um   L
    intervals      (x0, x1, n) on [0, P), covering the period without gaps
                   (the last one may end at P); n complex allowed
    """
    thickness_um: float
    intervals: List[Interval]


@dataclass
class Grating1D:
    """The structure: period, incident and exit half spaces, the layers
    from the incident side to the exit side.

    Attributes
    ----------
    period_um   P
    n_inc       index of the incident half space (the light comes from there)
    n_out       index of the exit half space
    layers      Layer list, first = next to the incident half space
    """
    period_um: float
    n_inc: complex
    n_out: complex
    layers: List[Layer] = field(default_factory=list)

    def depth_um(self) -> float:
        return float(sum(l.thickness_um for l in self.layers))


@dataclass
class Efficiencies:
    """The result of one solve.

    Attributes
    ----------
    orders      m = -N..N
    kx          normalized x-wavevector of each order (n sin theta_m)
    R_te, T_te  reflected / transmitted efficiency per order, TE
    R_tm, T_tm  the same, TM
    n_harm      N (2N+1 harmonics)
    """
    orders: np.ndarray
    kx: np.ndarray
    R_te: np.ndarray
    T_te: np.ndarray
    R_tm: np.ndarray
    T_tm: np.ndarray
    n_harm: int

    @property
    def T(self) -> np.ndarray:
        """Unpolarized transmitted efficiency per order."""
        return 0.5 * (self.T_te + self.T_tm)

    @property
    def R(self) -> np.ndarray:
        return 0.5 * (self.R_te + self.R_tm)

    def order(self, m: int, pol: str = "avg") -> float:
        j = int(m) + self.n_harm
        if j < 0 or j >= self.orders.size:
            return 0.0
        return float({"te": self.T_te, "tm": self.T_tm, "avg": self.T}[pol][j])

    def balance(self) -> Tuple[float, float]:
        """(sum R + T) for TE and TM -- 1.0 when lossless."""
        return float(self.R_te.sum() + self.T_te.sum()), float(self.R_tm.sum() + self.T_tm.sum())

    def as_dict(self) -> Dict[str, List[float]]:
        return {"orders": self.orders.tolist(), "kx": self.kx.tolist(), "T_te": self.T_te.tolist(),
                "T_tm": self.T_tm.tolist(), "R_te": self.R_te.tolist(), "R_tm": self.R_tm.tolist()}


# --- Fourier coefficients of a piecewise-constant function -------------------
def fourier_coefficients(intervals: Sequence[Interval], period: float, n_harm: int,
                         func) -> np.ndarray:
    """c_m, m = -2N..2N, of f(x) = func(n) on each interval (analytic)."""
    ms = np.arange(-2 * n_harm, 2 * n_harm + 1)
    c = np.zeros(ms.size, dtype=Complex)
    for x0, x1, n in intervals:
        v = complex(func(n))
        w = (x1 - x0) / period
        c[ms == 0] += v * w
        nz = ms != 0
        arg = 2.0 * np.pi * ms[nz] / period
        c[nz] += v * (np.exp(-1j * arg * x1) - np.exp(-1j * arg * x0)) / (-1j * arg * period)
    return c


def toeplitz_of(coefs: np.ndarray, n_harm: int) -> np.ndarray:
    """T[i, j] = c_{i - j} from c_m, m = -2N..2N."""
    M = 2 * n_harm + 1
    idx = np.arange(M)
    return coefs[(idx[:, None] - idx[None, :]) + 2 * n_harm]


# --- the solver -------------------------------------------------------------------
def _sqrt_branch(z: np.ndarray) -> np.ndarray:
    """sqrt on the branch of the forward / decaying mode: Re >= 0 (evanescent
    modes decay with exp(-lam k0 z)); for propagating modes (Re ~ 0, the
    eigenvalue is a negative real up to round-off noise) Im >= 0, so that the
    noise never flips a forward mode into a backward one -- that flip is
    what makes the S-matrix recursion blow up at random harmonic counts."""
    s = np.sqrt(z.astype(Complex))
    tol = 1e-9 * np.maximum(1.0, np.abs(s))
    flip = (s.real < -tol) | ((np.abs(s.real) <= tol) & (s.imag < 0))
    s[flip] = -s[flip]
    return s


def _homogeneous(kx: np.ndarray, n: complex, pol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W = I, lam, V of a homogeneous region of index n."""
    M = kx.size
    lam = _sqrt_branch(kx * kx - n * n)
    lam[np.abs(lam) < 1e-10] = 1e-10           # an order exactly at cut-off: no power, no singularity
    W = np.eye(M, dtype=Complex)
    if pol == "te":
        V = np.diag(-lam)
    else:
        V = np.diag(-(n * n) / lam)
    return W, lam, V


def _layer_modes(kx: np.ndarray, E: np.ndarray, E_i: np.ndarray, pol: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Kx = np.diag(kx.astype(Complex))
    if pol == "te":
        Om2 = Kx @ Kx - E
        Q = E - Kx @ Kx
    else:
        E_inv = np.linalg.inv(E)
        Ei_inv = np.linalg.inv(E_i)
        Om2 = (Kx @ E_inv @ Kx - np.eye(kx.size)) @ Ei_inv
        Q = -Ei_inv
    lam2, W = sla.eig(Om2)
    lam = _sqrt_branch(lam2)
    V = Q @ W @ np.diag(1.0 / lam)
    return W, lam, V


def _star(SA: Tuple[np.ndarray, ...], SB: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    """Redheffer star product of two S-matrices (S11, S12, S21, S22)."""
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    M = A11.shape[0]
    I = np.eye(M, dtype=Complex)
    D = np.linalg.solve(I - B11 @ A22, np.eye(M, dtype=Complex))
    F = np.linalg.solve(I - A22 @ B11, np.eye(M, dtype=Complex))
    S11 = A11 + A12 @ D @ B11 @ A21
    S12 = A12 @ D @ B12
    S21 = B21 @ F @ A21
    S22 = B22 + B21 @ F @ A22 @ B12
    return S11, S12, S21, S22


def solve(g: Grating1D, lam_um: float, n_harm: int, theta_deg: float = 0.0,
          pols: Sequence[str] = ("te", "tm")) -> Efficiencies:
    """Diffraction efficiencies of `g` at wavelength lam_um, 2N+1 harmonics,
    plane wave from the incident half space at theta_deg from the normal."""
    P = float(g.period_um)
    k0 = 2.0 * np.pi / lam_um
    N = int(n_harm)
    M = 2 * N + 1
    ms = np.arange(-N, N + 1)
    n_inc, n_out = complex(g.n_inc), complex(g.n_out)
    kx0 = (n_inc * np.sin(np.radians(theta_deg))).real
    kx = kx0 + ms * lam_um / P
    kz_inc = np.sqrt(n_inc * n_inc - kx0 * kx0)
    kz_ref = _sqrt_branch(n_inc * n_inc - kx * kx)               # sqrt(n^2 - kx^2), Re >= 0
    kz_trn = _sqrt_branch(n_out * n_out - kx * kx)
    # layer Fourier matrices (independent of polarization)
    mats: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for layer in g.layers:
        cE = fourier_coefficients(layer.intervals, P, N, lambda n: n * n)
        cI = fourier_coefficients(layer.intervals, P, N, lambda n: 1.0 / (n * n))
        mats.append((toeplitz_of(cE, N), toeplitz_of(cI, N), float(layer.thickness_um)))
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # the gap medium of the S-matrix recursion is arbitrary (Rumpf): free space,
    # unless an order sits exactly at its cut-off (kx_m = 1 -- e.g. P = 50 lam),
    # in which case a nearby index that no order hits is used
    n_gap = 1.0
    while np.min(np.abs(kx * kx - n_gap * n_gap)) < 1e-6:
        n_gap += 0.013
    for pol in pols:
        W0, lam0, V0 = _homogeneous(kx, n_gap, pol)
        # incident half space
        Wr, lamr, Vr = _homogeneous(kx, n_inc, pol)
        A = np.linalg.solve(W0, Wr) + np.linalg.solve(V0, Vr)
        B = np.linalg.solve(W0, Wr) - np.linalg.solve(V0, Vr)
        Ai = np.linalg.inv(A)
        S: Tuple[np.ndarray, ...] = (-Ai @ B, 2.0 * Ai, 0.5 * (A - B @ Ai @ B), B @ Ai)
        for E, E_i, L in mats:
            Wl, laml, Vl = _layer_modes(kx, E, E_i, pol)
            A = np.linalg.solve(Wl, W0) + np.linalg.solve(Vl, V0)
            B = np.linalg.solve(Wl, W0) - np.linalg.solve(Vl, V0)
            X = np.diag(np.exp(-laml * k0 * L))
            Ai = np.linalg.inv(A)
            D = np.linalg.inv(A - X @ B @ Ai @ X @ B)
            S11 = D @ (X @ B @ Ai @ X @ A - B)
            S12 = D @ X @ (A - B @ Ai @ B)
            S = _star(S, (S11, S12, S12, S11))
        Wt, lamt, Vt = _homogeneous(kx, n_out, pol)
        A = np.linalg.solve(W0, Wt) + np.linalg.solve(V0, Vt)
        B = np.linalg.solve(W0, Wt) - np.linalg.solve(V0, Vt)
        Ai = np.linalg.inv(A)
        S = _star(S, (B @ Ai, 0.5 * (A - B @ Ai @ B), 2.0 * Ai, -Ai @ B))
        # source: unit amplitude in the 0th harmonic
        e_src = np.zeros(M, dtype=Complex)
        e_src[N] = 1.0
        c_inc = np.linalg.solve(Wr, e_src)
        c_ref = S[0] @ c_inc
        c_trn = S[2] @ c_inc
        e_ref = Wr @ c_ref
        e_trn = Wt @ c_trn
        if pol == "te":
            r2 = np.abs(e_ref) ** 2
            t2 = np.abs(e_trn) ** 2
        else:                                            # ex given; ez = -kx ex / kz
            with np.errstate(divide="ignore", invalid="ignore"):
                ez_r = np.where(np.abs(kz_ref) > 0, -kx * e_ref / kz_ref, 0.0)
                ez_t = np.where(np.abs(kz_trn) > 0, -kx * e_trn / kz_trn, 0.0)
            r2 = np.abs(e_ref) ** 2 + np.abs(ez_r) ** 2
            t2 = np.abs(e_trn) ** 2 + np.abs(ez_t) ** 2
        R = (kz_ref.real / kz_inc.real) * r2
        T = (kz_trn.real / kz_inc.real) * t2
        out[pol] = (R.real, T.real)
    zero = np.zeros(M)
    Rte, Tte = out.get("te", (zero, zero))
    Rtm, Ttm = out.get("tm", (zero, zero))
    return Efficiencies(orders=ms, kx=kx.real, R_te=Rte, T_te=Tte, R_tm=Rtm, T_tm=Ttm, n_harm=N)


def harmonics_for(period_um: float, lam_um: float, n_max: float, margin: float = 1.3,
                  n_min: int = 10, n_cap: Optional[int] = None) -> int:
    """N such that 2N+1 harmonics cover every order propagating in the
    densest medium with a margin: N = ceil(margin n_max P / lam)."""
    N = int(np.ceil(margin * n_max * period_um / lam_um))
    N = max(n_min, N)
    return min(N, n_cap) if n_cap else N
