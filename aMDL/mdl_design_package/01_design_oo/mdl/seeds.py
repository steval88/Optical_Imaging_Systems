"""Analytic starting points for the optimizer ("Initialize m" of Fig. S3).

Both seeds are the ideal hyperbolic optical path, folded back into the
available relief height and quantized:

    h(rho) = ( -OPD(rho) mod (n_ref - 1) H_f ) / (n_ref - 1),
    OPD(rho) = sqrt(rho^2 + F^2) - F,

i.e. a Fresnel-lens sawtooth whose teeth have optical depth
(n_ref - 1) H_f. The only free parameter is the fold height H_f.

* ``harmonic_seed``: H_f = p lam_0 / (n(lam_0) - 1) -- the "harmonic
  diffractive lens" (Sweeney & Sommargren, Appl. Opt. 34, 2469 (1995);
  Faklis & Morris, ibid. 2462): exactly blazed at ONE design wavelength
  lam_0 in order p.
* ``LadderSeed``: the fold height is chosen for a whole COMB of target
  lines at once. A sawtooth of optical depth D = (n(lam_j) - 1) H_f is
  blazed for line j when alpha_j = D / lam_j is an integer m_j, the
  order in which that line focuses; the scalar efficiency into the
  nearest order is sinc^2(alpha_j - m_j). The integers m_j form a
  descending "order ladder" across the comb (24 at 400 nm ... 8 at
  1100 nm for the S3 fold of 14.377 um), hence the name; the selection
  chart is the blaze chart of a high-order (echelle) grating applied to
  the fold height. The chart is only a proxy (it ignores quantization,
  dispersion across the fold and the objective's aggregation), so every
  candidate fold is built into a seed and judged by the REAL objective
  of the problem; the best wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .material import IndexModel, n_az4562
from .problem import IntVec, MDLProblem


def fold_seed(prob: MDLProblem, h_fold_um: float, n_ref: float) -> IntVec:
    """Quantized hyperbolic OPD folded at optical depth (n_ref - 1) H_f."""
    opl_fold = (n_ref - 1.0) * h_fold_um            # fold period in OPD [um]
    r = np.sqrt(prob.rho ** 2 + prob.F ** 2)
    opd = r - prob.F
    h = ((-opd) % opl_fold) / (n_ref - 1.0)
    return np.clip(np.rint(h / prob.dh), 0, prob.M).astype(np.int32)


def harmonic_seed(prob: MDLProblem, lam0_um: float, p_harmonic: int = 1) -> IntVec:
    """Harmonic-lens seed: blazed at lam0 in order p (fold p lam0/(n0-1))."""
    n0 = float(prob.n_func(lam0_um))
    r = np.sqrt(prob.rho ** 2 + prob.F ** 2)
    opd = r - prob.F                                  # [um]
    h = ((-opd) % (p_harmonic * lam0_um)) / (n0 - 1.0)
    return np.clip(np.rint(h / prob.dh), 0, prob.M).astype(np.int32)


def max_harmonic_seed(prob: MDLProblem, lam0_um: float) -> IntVec:
    """Harmonic seed with the largest order p whose fold fits in h_max."""
    n0 = float(prob.n_func(lam0_um))
    step = lam0_um / (n0 - 1.0)
    p = max(1, int(np.floor(prob.h_max / step)))
    return harmonic_seed(prob, lam0_um, p_harmonic=p)


@dataclass
class FoldScan:
    """Blaze chart of the fold height over a comb (``ladder_fold_scan``).

    Attributes
    ----------
    h_fold_um : float        best fold height
    score : float            its geometric-mean sinc^2 (1 = all lines blazed)
    orders : (J,) int        m_j = round(alpha_j) at the best fold, the ladder
    detune : (J,) float      alpha_j - m_j per line
    eff_line : (J,) float    sinc^2(detune) per line (scalar sawtooth efficiency)
    h_grid : (H,) float      scanned fold heights
    score_grid : (H,) float  score at each scanned height
    """
    h_fold_um: float
    score: float
    orders: NDArray[np.int64]
    detune: NDArray[np.float64]
    eff_line: NDArray[np.float64]
    h_grid: NDArray[np.float64]
    score_grid: NDArray[np.float64]


def ladder_fold_scan(target_lams_um: NDArray[np.float64], h_min_um: float,
                     h_max_um: float, n_func: IndexModel = n_az4562,
                     n_scan: int = 20001,
                     weights: Optional[NDArray[np.float64]] = None) -> FoldScan:
    """Scan fold heights in [h_min, h_max] and score each by the geometric
    mean of sinc^2(alpha_j - round(alpha_j)) over the target lines
    (optionally weighted in the log domain)."""
    lams = np.asarray(target_lams_um, dtype=float)
    w = (np.full(lams.size, 1.0 / lams.size) if weights is None
         else np.asarray(weights, float) / np.sum(weights))
    hg = np.linspace(h_min_um, h_max_um, int(n_scan))
    n = n_func(lams)
    alpha = (n - 1.0)[None, :] * hg[:, None] / lams[None, :]  # (H, J)
    det = alpha - np.rint(alpha)
    eff = np.sinc(det) ** 2
    score = np.exp(np.sum(w[None, :] * np.log(eff + 1e-12), axis=1))
    ib = int(np.argmax(score))
    return FoldScan(float(hg[ib]), float(score[ib]),
                    np.rint(alpha[ib]).astype(int), det[ib], eff[ib], hg, score)


@dataclass
class LadderSeedResult:
    """Outcome of ``LadderSeed.run``.

    Attributes
    ----------
    m : (N,) int32                  the seed vector
    J : float                       its objective value
    h_fold_um : float               the winning fold height
    orders : (J,) int               the order ladder m_j of the winner
    detune : (J,) float             alpha_j - m_j per line
    eff_line : (J,) float           sinc^2(detune) per line
    blaze_score : float             geometric mean of eff_line
    candidates : list of (h_fold, blaze_score, J)   every fold judged,
                                    best J first
    scan : FoldScan or None         the full chart (None if the fold was
                                    given explicitly)
    """
    m: IntVec
    J: float
    h_fold_um: float
    orders: NDArray[np.int64]
    detune: NDArray[np.float64]
    eff_line: NDArray[np.float64]
    blaze_score: float
    candidates: List[Tuple[float, float, float]]
    scan: Optional[FoldScan]

    def record(self) -> dict:
        """JSON-ready seed record for design_metrics.json."""
        return {"mode": "ladder", "h_fold_um": self.h_fold_um,
                "orders": [int(o) for o in self.orders],
                "blaze_score": self.blaze_score}


class LadderSeed:
    """Comb seed: fold height selected over ALL target lines (module docstring).

    Candidate folds come from two families, both scored by the real
    ``prob.fom`` on the problem's current wavelength grid / fom_mode
    (set them before calling):

    (a) every harmonic anchor H = p lam_j / (n_j - 1), lam_j a target
        line, for every integer p with h_min_frac h_max <= H <= h_max
        (the full generalization of a single-lam0 harmonic seed);
    (b) the ``n_candidates`` best distinct local maxima of the blaze
        chart (compromise folds blazed at no single line), folds within
        5 nm of a harmonic anchor dropped as duplicates.

    Parameters
    ----------
    prob : MDLProblem
    target_lams_um : (J,)   the comb
    h_min_frac : float      lower end of the scan as a fraction of h_max
    n_candidates : int      chart maxima judged by the objective
    log : callable or None  progress printer
    """

    def __init__(self, prob: MDLProblem, target_lams_um: NDArray[np.float64],
                 h_min_frac: float = 0.5, n_candidates: int = 16,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.prob = prob
        self.lams = np.asarray(target_lams_um, dtype=float)
        self.h_min_frac = float(h_min_frac)
        self.n_candidates = int(n_candidates)
        self.log = log
        lam_mid = 0.5 * (self.lams.min() + self.lams.max())
        #: band-centre index used to build the folded profile (dispersion
        #: breaks exact congruence anyway; the optimizer absorbs the rest)
        self.n_ref: float = float(prob.n_func(lam_mid))

    def line_info(self, h_fold_um: float) -> Tuple[NDArray[np.int64],
                                                   NDArray[np.float64],
                                                   NDArray[np.float64]]:
        """(orders, detune, sinc^2) of every target line at a fold height."""
        n = self.prob.n_func(self.lams)
        alpha = (n - 1.0) * h_fold_um / self.lams
        det = alpha - np.rint(alpha)
        return np.rint(alpha).astype(int), det, np.sinc(det) ** 2

    def blaze_score(self, h_fold_um: float) -> float:
        _, _, eff = self.line_info(h_fold_um)
        return float(np.exp(np.mean(np.log(eff + 1e-12))))

    def run(self, h_fold_um: Optional[float] = None) -> LadderSeedResult:
        """Select the fold (or use ``h_fold_um`` verbatim) and build the seed."""
        prob, lams = self.prob, self.lams
        if h_fold_um is not None:
            m = fold_seed(prob, h_fold_um, self.n_ref)
            orders, det, eff = self.line_info(h_fold_um)
            return LadderSeedResult(m, prob.fom(m), float(h_fold_um), orders,
                                    det, eff, self.blaze_score(h_fold_um),
                                    [], None)
        scan = ladder_fold_scan(lams, self.h_min_frac * prob.h_max, prob.h_max,
                                n_func=prob.n_func)
        hg, sc = scan.h_grid, scan.score_grid
        n_all = prob.n_func(lams)
        # family (a): harmonic anchors, every target line, every order p
        fam: List[Tuple[float, float]] = []            # (H_f, n used to build)
        for lam_j, n_j in zip(lams, n_all):
            step = lam_j / (n_j - 1.0)
            for p in range(max(1, int(np.ceil(self.h_min_frac * prob.h_max / step))),
                           int(np.floor(prob.h_max / step)) + 1):
                fam.append((p * step, n_j))
        # family (b): chart maxima, best blaze score first, duplicates dropped
        imax = np.where((sc[1:-1] > sc[:-2]) & (sc[1:-1] >= sc[2:]))[0] + 1
        scanc = sorted([(float(hg[i]), self.n_ref) for i in imax],
                       key=lambda c: -self.blaze_score(c[0]))
        anchors = np.array([hf for hf, _ in fam])
        scanc = [(hf, nr) for hf, nr in scanc
                 if np.min(np.abs(anchors - hf)) > 5e-3]
        cands: List[Tuple[float, float, float]] = []
        best: Optional[Tuple[float, float, IntVec]] = None
        for hf, nr in fam + scanc[:self.n_candidates]:
            m = fold_seed(prob, hf, nr)
            J = prob.fom(m)
            cands.append((float(hf), self.blaze_score(hf), float(J)))
            if best is None or J > best[0]:
                best = (J, float(hf), m)
        assert best is not None
        J_best, hf_best, m_best = best
        orders, det, eff = self.line_info(hf_best)
        if self.log:
            self.log("ladder seed: %d candidate folds, winner H_fold = %.4f um "
                     "(seed J = %.4g, fom_mode=%s)"
                     % (len(cands), hf_best, J_best, prob.fom_mode))
            self.log("  line[um]  order m  detune   sinc^2")
            for lam, m_j, d, e in zip(lams, orders, det, eff):
                self.log("  %8.4f  %7d  %+.3f   %.3f" % (lam, m_j, d, e))
        return LadderSeedResult(m_best, J_best, hf_best, orders, det, eff,
                                float(np.exp(np.mean(np.log(eff + 1e-12)))),
                                sorted(cands, key=lambda c: -c[2]), scan)
