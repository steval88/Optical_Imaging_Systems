"""The pre-refactor function names of mdl_core.py, as thin wrappers.

For scripts and notebooks written against the old module. Each wrapper
builds the corresponding class and returns what the old function
returned (mostly ``(m, J)`` tuples). ``seed_echelle`` is the old name of
the ladder seed; its return value keeps the old dict layout.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .optimizers import (BinaryGeneticAlgorithm, GeneticAlgorithm,
                         GradientRefine, HistoryEntry, HookeJeeves,
                         MultistepGAHJACombo, SearchGAHJA, Smooth)
from .problem import IntVec, MDLProblem
from .seeds import LadderSeed, harmonic_seed, ladder_fold_scan, max_harmonic_seed

seed_harmonic = harmonic_seed
seed_max_harmonic = max_harmonic_seed


def genetic_algorithm(prob: MDLProblem, m_init: IntVec, pop_size: int = 40,
                      epochs: int = 200, p_cross: float = 0.8,
                      p_mut: float = 0.01, elite: int = 2,
                      rng: Optional[np.random.Generator] = None,
                      log_every: int = 0,
                      log_list: Optional[List[HistoryEntry]] = None
                      ) -> Tuple[IntVec, float]:
    r = GeneticAlgorithm(prob, pop_size, epochs, p_cross, p_mut, elite, rng,
                         log_every).run(m_init)
    if log_list is not None:
        log_list += r.history
    return r.m, r.J


def hooke_jeeves(prob: MDLProblem, m_init: IntVec, d0: Optional[int] = None,
                 alpha: float = 1.0, max_sweeps: int = 200, rng=None,
                 log_list: Optional[List[HistoryEntry]] = None
                 ) -> Tuple[IntVec, float]:
    r = HookeJeeves(prob, d0, alpha, max_sweeps).run(m_init)
    if log_list is not None:
        log_list += r.history
    return r.m, r.J


def hooke_jeeves_verbatim(prob: MDLProblem, m_init: IntVec,
                          d0: Optional[int] = None, alpha: float = 1.0
                          ) -> Tuple[IntVec, float]:
    r = HookeJeeves(prob, d0, alpha, verbatim=True).run(m_init)
    return r.m, r.J


def search(prob: MDLProblem, m_init: Optional[IntVec] = None, blocks: int = 4,
           ga_epochs: int = 60, pop_size: int = 40,
           rng: Optional[np.random.Generator] = None,
           log_list: Optional[List[HistoryEntry]] = None,
           verbose: Callable[[str], None] = print, chain: str = "best"
           ) -> Tuple[IntVec, float]:
    r = SearchGAHJA(prob, blocks, ga_epochs, pop_size, rng, chain,
                    log=verbose).run(m_init)
    if log_list is not None:
        log_list += r.history
    return r.m, r.J


def genetic_algorithm_binary(prob: MDLProblem, m_init: Optional[IntVec] = None,
                             pop_size: int = 40, epochs: int = 200,
                             p_cross: float = 0.8, p_mut: Optional[float] = None,
                             elite: int = 2,
                             rng: Optional[np.random.Generator] = None
                             ) -> Tuple[IntVec, float]:
    r = BinaryGeneticAlgorithm(prob, pop_size, epochs, p_cross, p_mut, elite,
                               rng).run(m_init)
    return r.m, r.J


def multistep_GA_HJA_combo(prob: MDLProblem, m_init: Optional[IntVec] = None,
                           s: int = 4, p: int = 60, pop_size: int = 40,
                           d0: Optional[int] = None, alpha: float = 1.0,
                           rng: Optional[np.random.Generator] = None,
                           verbose: Callable[[str], None] = print,
                           parallel_hja: bool = False) -> Tuple[IntVec, float]:
    r = MultistepGAHJACombo(prob, s, p, pop_size, d0, alpha, rng, verbose,
                            parallel_hja).run(m_init)
    return r.m, r.J


def smooth(prob: MDLProblem, m_vec: IntVec, alpha0: float = 2.0,
           beta: float = 7.0) -> IntVec:
    return Smooth(prob, alpha0, beta).run(m_vec).m


def gradient_refine(prob: MDLProblem, m_vec: IntVec,
                    step0: Optional[float] = None, iters: int = 300,
                    shrink: float = 0.5, grow: float = 1.1,
                    log_list: Optional[List[HistoryEntry]] = None,
                    softmin_beta_final: Optional[float] = None
                    ) -> Tuple[IntVec, float]:
    r = GradientRefine(prob, step0, iters, shrink, grow,
                       softmin_beta_final).run(m_vec)
    if log_list is not None:
        log_list += r.history
    return r.m, r.J


def echelle_fold_scan(target_lams_um, h_min_um, h_max_um, n_func=None,
                      n_scan: int = 20001, weights=None, verbose=None) -> dict:
    from .material import n_az4562
    s = ladder_fold_scan(target_lams_um, h_min_um, h_max_um,
                         n_func or n_az4562, n_scan, weights)
    if verbose:
        verbose("ladder fold scan: best H_fold = %.4f um (geomean sinc^2 = %.3f)"
                % (s.h_fold_um, s.score))
    return dict(h_fold_um=s.h_fold_um, score=s.score, orders=s.orders,
                detune=s.detune, eff_line=s.eff_line,
                scan=(s.h_grid, s.score_grid))


def seed_echelle(prob: MDLProblem, target_lams_um, h_fold_um: Optional[float] = None,
                 h_min_frac: float = 0.5, n_candidates: int = 16,
                 verbose=None) -> Tuple[IntVec, dict]:
    """Old name of the ladder seed; returns (m, info dict) as before."""
    r = LadderSeed(prob, target_lams_um, h_min_frac, n_candidates,
                   log=verbose).run(h_fold_um)
    return r.m, dict(h_fold_um=r.h_fold_um, score=r.blaze_score,
                     orders=r.orders, detune=r.detune, eff_line=r.eff_line,
                     scan=(None if r.scan is None
                           else (r.scan.h_grid, r.scan.score_grid)),
                     candidates=(r.candidates or None))


seed_ladder = seed_echelle
