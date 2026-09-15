"""Optimizers of the paper's Fig. 2a pipeline, one class per box.

Search  = s blocks of [genetic algorithm (p epochs) -> Hooke-Jeeves]
          (paper S2-2, Fig. S3): ``SearchGAHJA`` (engineering wiring,
          integer-coded GA) and ``MultistepGAHJACombo`` (verbatim Fig. S3
          wiring, binary-coded GA as in S2-2).
Smooth  = aspect-ratio reduction of the height profile (S2-4, Eq. 24,
          Fig. S5): ``Smooth``.
Gradient= gradient ascent on continuous heights then rounding (S2-2,
          Fig. S2): ``GradientRefine``; a final integer Hooke-Jeeves
          polish (``HookeJeeves``) closes the pipeline in the driver.

Every optimizer exposes ``run(m_init) -> OptResult`` and keeps its
working state as documented attributes (for the GA: ``population``,
``fitness``, ``generation``). The arithmetic and the order in which
random numbers are drawn are identical to the pre-refactor functions of
mdl_core.py, so a run with the same ``rng_seed`` reproduces the same
design vector.

Notation follows the supplementary: m is the gray-level vector
("distribution m"), an individual of the GA population is one such
vector, N its length, M the maximum gray level, J = f(m) the objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .problem import Field, IntVec, MDLProblem

#: a (stage, iteration, J) triple appended to the history of a run
HistoryEntry = Tuple[str, int, float]
#: print-like callback for progress lines
Logger = Callable[[str], None]


def _silent(_: str) -> None:
    pass


@dataclass
class OptResult:
    """Outcome of one optimizer run.

    Attributes
    ----------
    m : (N,) int32
        The best gray-level vector found.
    J : float
        Its objective value under the problem's current fom_mode /
        objective (for a softmin problem: at the beta in force when the
        run ended).
    history : list of (stage, iteration, J)
        Progress records, if the optimizer logs them.
    """
    m: IntVec
    J: float
    history: List[HistoryEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
class GeneticAlgorithm:
    """Integer-coded genetic algorithm on the gray-level vector.

    Each individual of the population is a gray-level vector m (N
    integers in 0..M). Generation update: sort by fitness, copy the
    ``elite`` best unchanged, then fill the population with children of
    tournament-selected parents (two random draws each, the fitter
    wins), single-point crossover with probability ``p_cross``, and
    mutation that resets each gene to a random level with probability
    ``p_mut``. This is the engineering variant used by ``SearchGAHJA``;
    ``BinaryGeneticAlgorithm`` is the paper's binary-coded GA.

    Parameters
    ----------
    prob : MDLProblem
    pop_size : int          individuals per generation
    epochs : int            generations to run
    p_cross : float         crossover probability per pair
    p_mut : float           mutation probability per gene
    elite : int             individuals copied unchanged each generation
    rng : numpy Generator   random source (shared with the caller so a
                            run is reproducible from one seed)
    log_every : int         record the best J every log_every generations
                            (0 = never)

    Attributes
    ----------
    population : (pop_size, N) int32   current individuals
    fitness : (pop_size,) float        J of each individual
    generation : int                   generations completed
    """

    def __init__(self, prob: MDLProblem, pop_size: int = 40,
                 epochs: int = 200, p_cross: float = 0.8,
                 p_mut: float = 0.01, elite: int = 2,
                 rng: Optional[np.random.Generator] = None,
                 log_every: int = 0) -> None:
        self.prob = prob
        self.pop_size = int(pop_size)
        self.epochs = int(epochs)
        self.p_cross = float(p_cross)
        self.p_mut = float(p_mut)
        self.elite = int(elite)
        self.rng = rng or np.random.default_rng()
        self.log_every = int(log_every)
        self.population: NDArray[np.int32] = np.empty((0, prob.N), np.int32)
        self.fitness: NDArray[np.float64] = np.empty(0)
        self.generation: int = 0

    def initialize(self, m_init: IntVec) -> None:
        """Seed the population: the seed itself, pop_size//2 - 1 copies
        with 5 % of the genes randomized, the rest fully random."""
        N, M = self.prob.N, self.prob.M
        pop = np.empty((self.pop_size, N), dtype=np.int32)
        pop[0] = m_init
        for p in range(1, self.pop_size):
            if p < self.pop_size // 2:
                pop[p] = m_init
                idx = self.rng.random(N) < 0.05
                pop[p, idx] = self.rng.integers(0, M + 1, idx.sum())
            else:
                pop[p] = self.rng.integers(0, M + 1, N)
        self.population = pop
        self.fitness = np.array([self.prob.fom(ind) for ind in pop])
        self.generation = 0

    def _tournament(self) -> IntVec:
        a, b = self.rng.integers(0, self.pop_size, 2)
        return self.population[a] if self.fitness[a] > self.fitness[b] \
            else self.population[b]

    def step(self) -> None:
        """One generation: elitism, tournament selection, crossover, mutation."""
        N, M = self.prob.N, self.prob.M
        order = np.argsort(self.fitness)[::-1]
        self.population, self.fitness = self.population[order], self.fitness[order]
        new_pop = [self.population[i].copy() for i in range(self.elite)]
        while len(new_pop) < self.pop_size:
            pa = self._tournament()
            pb = self._tournament()
            c1, c2 = pa.copy(), pb.copy()
            if self.rng.random() < self.p_cross:
                cut = self.rng.integers(1, N)
                c1[cut:], c2[cut:] = pb[cut:].copy(), pa[cut:].copy()
            for c in (c1, c2):
                mask = self.rng.random(N) < self.p_mut
                c[mask] = self.rng.integers(0, M + 1, mask.sum())
                new_pop.append(c)
        self.population = np.array(new_pop[:self.pop_size], dtype=np.int32)
        self.fitness = np.array([self.prob.fom(ind) for ind in self.population])
        self.generation += 1

    @property
    def best(self) -> Tuple[IntVec, float]:
        """(m, J) of the fittest individual."""
        i = int(np.argmax(self.fitness))
        return self.population[i].copy(), float(self.fitness[i])

    def run(self, m_init: IntVec) -> OptResult:
        self.initialize(m_init)
        history: List[HistoryEntry] = []
        for _ in range(self.epochs):
            self.step()
            if self.log_every and self.generation % self.log_every == 0:
                history.append(("GA", self.generation, float(self.fitness.max())))
        # the original returned pop[argsort(fit)[::-1][0]]: the FIRST
        # index of the descending sort, i.e. the last maximum in the
        # array for ties -- reproduced here for bit-identical replays
        order = np.argsort(self.fitness)[::-1]
        return OptResult(self.population[order[0]].copy(),
                         float(self.fitness[order[0]]), history)


# ---------------------------------------------------------------------------
class BinaryGeneticAlgorithm:
    """The paper's binary-coded GA (supplementary S2-2).

    Each gray level is encoded in nb = ceil(log2(M+1)) bits (6/7/8 bits
    for M = 32/64/192), an individual is the bit string of length nb N,
    single-point crossover and bit-flip mutation act on the bit string,
    and decoded values above M are clamped to M. Used by
    ``MultistepGAHJACombo`` (verbatim Fig. S3).

    Attributes
    ----------
    population : (pop_size, nb N) uint8   bit strings
    fitness : (pop_size,) float
    """

    def __init__(self, prob: MDLProblem, pop_size: int = 40,
                 epochs: int = 200, p_cross: float = 0.8,
                 p_mut: Optional[float] = None, elite: int = 2,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.prob = prob
        self.pop_size = int(pop_size)
        self.epochs = int(epochs)
        self.p_cross = float(p_cross)
        self.elite = int(elite)
        self.rng = rng or np.random.default_rng()
        self.nb: int = int(np.ceil(np.log2(prob.M + 1)))
        self.L: int = prob.N * self.nb                  # bits per individual
        self.p_mut: float = 1.0 / self.L if p_mut is None else float(p_mut)
        self.population: NDArray[np.uint8] = np.empty((0, self.L), np.uint8)
        self.fitness: NDArray[np.float64] = np.empty(0)

    def decode(self, bits: NDArray[np.uint8]) -> IntVec:
        """Bit string (nb N,) -> gray-level vector (N,), clamped to M."""
        w = (1 << np.arange(self.nb - 1, -1, -1)).astype(np.int64)
        vals = bits.reshape(self.prob.N, self.nb) @ w
        return np.minimum(vals, self.prob.M).astype(np.int32)

    def encode(self, m_vec: IntVec) -> NDArray[np.uint8]:
        """Gray-level vector -> bit string (big-endian per gene)."""
        out = np.empty(m_vec.size * self.nb, dtype=np.uint8)
        for b in range(self.nb):
            out[b::self.nb] = (m_vec >> (self.nb - 1 - b)) & 1
        return out

    def run(self, m_init: Optional[IntVec] = None) -> OptResult:
        rng, L = self.rng, self.L
        pop = rng.integers(0, 2, size=(self.pop_size, L), dtype=np.uint8)
        if m_init is not None:
            pop[0] = self.encode(m_init.astype(np.int32))
        fit = np.array([self.prob.fom(self.decode(ind)) for ind in pop])
        for _ in range(self.epochs):
            order = np.argsort(fit)[::-1]
            pop, fit = pop[order], fit[order]
            new_pop = [pop[i].copy() for i in range(self.elite)]
            while len(new_pop) < self.pop_size:
                a, b = rng.integers(0, self.pop_size, 2)
                pa = pop[a] if fit[a] > fit[b] else pop[b]
                a, b = rng.integers(0, self.pop_size, 2)
                pb = pop[a] if fit[a] > fit[b] else pop[b]
                c1, c2 = pa.copy(), pb.copy()
                if rng.random() < self.p_cross:
                    cut = int(rng.integers(1, L))
                    c1[cut:], c2[cut:] = pb[cut:].copy(), pa[cut:].copy()
                for c in (c1, c2):
                    mask = rng.random(L) < self.p_mut
                    c[mask] ^= 1
                    new_pop.append(c)
            pop = np.array(new_pop[:self.pop_size], dtype=np.uint8)
            fit = np.array([self.prob.fom(self.decode(ind)) for ind in pop])
        self.population, self.fitness = pop, fit
        ib = int(np.argmax(fit))
        return OptResult(self.decode(pop[ib]), float(fit[ib]))


# ---------------------------------------------------------------------------
class HookeJeeves:
    """Integer Hooke-Jeeves direct search with O(Nw) delta evaluation.

    Exploratory sweep: ring by ring, try +d then -d gray levels and keep
    the FIRST improving move (``first_improvement=True``, the engineering
    variant) or the better of the two (``first_improvement=False``, the
    literal reading of the paper's Fig. S1). Pattern (acceleration) move
    after an improving sweep: Y = m_new + [alpha (m_new - m_prev)],
    clipped to [0, M] (rint for the engineering variant, floor for the
    verbatim one). Failed sweep: d halves. Stopping: ``max_sweeps`` or
    d < 1 (engineering), or the d = 1 sweep failing (verbatim, with
    d = ceil(d/2)).

    Parameters
    ----------
    prob : MDLProblem
    d0 : int or None        initial step in gray levels (None = M // 8)
    alpha : float           acceleration factor of the pattern move
    max_sweeps : int        cap on exploratory sweeps (engineering variant)
    verbatim : bool         paper Fig. S1 literal reading (see above)
    """

    def __init__(self, prob: MDLProblem, d0: Optional[int] = None,
                 alpha: float = 1.0, max_sweeps: int = 200,
                 verbatim: bool = False) -> None:
        self.prob = prob
        self.d0 = d0
        self.alpha = float(alpha)
        self.max_sweeps = int(max_sweeps)
        self.verbatim = bool(verbatim)

    def _explore(self, m_base: IntVec, U_base: Field, f_base: float,
                 d: int) -> Tuple[IntVec, Field, float]:
        prob, M = self.prob, self.prob.M
        m_c, U_c, f_c = m_base.copy(), U_base.copy(), f_base
        for i in range(prob.N):
            mi = m_c[i]
            if self.verbatim:
                best_f: float = f_c
                best_m: Optional[int] = None
                best_U: Optional[Field] = None
                for step in (d, -d):
                    mn = mi + step
                    if mn < 0 or mn > M:
                        continue
                    U_try = prob.delta_field(U_c, i, m_c[i], mn)
                    f_try = prob.fom_from_field(U_try)
                    if f_try > best_f:
                        best_f, best_m, best_U = f_try, mn, U_try
                if best_m is not None and best_U is not None:
                    m_c[i], U_c, f_c = best_m, best_U, best_f
            else:
                for step in (d, -d):
                    mn = mi + step
                    if mn < 0 or mn > M:
                        continue
                    U_try = prob.delta_field(U_c, i, m_c[i], mn)
                    f_try = prob.fom_from_field(U_try)
                    if f_try > f_c:
                        m_c[i], U_c, f_c = mn, U_try, f_try
                        break
        return m_c, U_c, f_c

    def run(self, m_init: IntVec) -> OptResult:
        prob, M = self.prob, self.prob.M
        d = self.d0 if self.d0 is not None else max(1, M // 8)
        m = m_init.astype(np.int32).copy()
        U = prob.field(m)
        f = prob.fom_from_field(U)
        m_prev = m.copy()
        sweeps = 0
        history: List[HistoryEntry] = []
        while True:
            if not self.verbatim and (d < 1 or sweeps >= self.max_sweeps):
                break
            m_new, U_new, f_new = self._explore(m, U, f, d)
            if f_new > f + 1e-12:
                move = self.alpha * (m_new - m_prev)
                patt = m_new + (np.floor(move) if self.verbatim
                                else np.rint(move)).astype(np.int32)
                np.clip(patt, 0, M, out=patt)
                U_p = prob.field(patt)
                f_p = prob.fom_from_field(U_p)
                m_prev = m.copy()
                if f_p > f_new:
                    m, U, f = patt, U_p, f_p
                else:
                    m, U, f = m_new, U_new, f_new
            else:
                if self.verbatim:
                    if d == 1:
                        break
                    d = (d + 1) // 2                 # ceil(d/2)
                else:
                    d //= 2
            if not self.verbatim:
                sweeps += 1
                history.append(("HJA", sweeps, f))
        return OptResult(m, f, history)


# ---------------------------------------------------------------------------
class SearchGAHJA:
    """The paper's *Search* step: s blocks of [GA (p epochs) -> HJA].

    Mapping to Fig. S3: "Initialize m" -> m_init (random per S2-1 or a
    seed); per block, GA gives m* and HJA(m*) gives m*(b); "stop"
    returns the best m*(b). Two wirings of the blocks:

    * chain = "ga"   : paper-exact -- block b+1's GA continues from
      block b's GA output (the down-arrow of Fig. S3); the HJA results
      are only collected at the end.
    * chain = "best" : engineering variant (default) -- block b+1's GA
      is seeded with the best HJA result so far; monotone per block and
      identical to "ga" whenever the GA does not beat its elite.

    Parameters
    ----------
    prob : MDLProblem
    blocks : int            s
    ga_epochs : int         p
    pop_size : int
    rng : numpy Generator
    chain : "best" | "ga"
    hja_max_sweeps : int    cap for the in-block Hooke-Jeeves
    log : callable          progress printer
    """

    def __init__(self, prob: MDLProblem, blocks: int = 4, ga_epochs: int = 60,
                 pop_size: int = 40, rng: Optional[np.random.Generator] = None,
                 chain: str = "best", hja_max_sweeps: int = 200,
                 log: Logger = print) -> None:
        self.prob = prob
        self.blocks = int(blocks)
        self.ga_epochs = int(ga_epochs)
        self.pop_size = int(pop_size)
        self.rng = rng or np.random.default_rng(0)
        self.chain = chain
        self.hja_max_sweeps = int(hja_max_sweeps)
        self.log = log

    def run(self, m_init: Optional[IntVec] = None) -> OptResult:
        prob, rng = self.prob, self.rng
        if m_init is None:
            m_init = rng.integers(0, prob.M + 1, prob.N).astype(np.int32)
        m_ga = m_init
        best: Optional[OptResult] = None
        history: List[HistoryEntry] = []
        for b in range(self.blocks):
            ga = GeneticAlgorithm(prob, pop_size=self.pop_size,
                                  epochs=self.ga_epochs, rng=rng, log_every=10)
            r_ga = ga.run(m_ga)
            m_ga = r_ga.m
            r_h = HookeJeeves(prob, max_sweeps=self.hja_max_sweeps).run(m_ga)
            history += r_ga.history + r_h.history
            self.log("  block %d/%d: GA %.4f -> HJA %.4f"
                     % (b + 1, self.blocks, r_ga.J, r_h.J))
            if best is None or r_h.J > best.J:
                best = OptResult(r_h.m.copy(), r_h.J)
            if self.chain == "best":
                m_ga = best.m.copy()
        assert best is not None
        return OptResult(best.m, best.J, history)


class MultistepGAHJACombo:
    """Verbatim Fig. S3 of the supplementary: binary GA + verbatim HJA.

        start -> Initialize m
        block b = 1..s:  m*_(b) = GA(m*_(b-1), p epochs)   [GA chain]
                         m*^(b) = HJA(m*_(b))              [side branch]
        stop: return argmax_b J(m*^(b))

    HJA never feeds back into the GA chain, so with ``parallel_hja`` the
    HJA of block b runs concurrently with the GA of block b+1 (identical
    result, shorter wall clock).
    """

    def __init__(self, prob: MDLProblem, s: int = 4, p: int = 60,
                 pop_size: int = 40, d0: Optional[int] = None,
                 alpha: float = 1.0, rng: Optional[np.random.Generator] = None,
                 log: Logger = print, parallel_hja: bool = False) -> None:
        self.prob, self.s, self.p = prob, int(s), int(p)
        self.pop_size, self.d0, self.alpha = int(pop_size), d0, float(alpha)
        self.rng = rng or np.random.default_rng(0)
        self.log = log
        self.parallel_hja = bool(parallel_hja)

    def run(self, m_init: Optional[IntVec] = None) -> OptResult:
        prob, rng = self.prob, self.rng
        if m_init is None:
            m_init = rng.integers(0, prob.M + 1, prob.N).astype(np.int32)
        outputs: List[OptResult] = []
        m_chain = m_init
        hja = HookeJeeves(prob, d0=self.d0, alpha=self.alpha, verbatim=True)
        if not self.parallel_hja:
            for b in range(1, self.s + 1):
                r_ga = BinaryGeneticAlgorithm(prob, pop_size=self.pop_size,
                                              epochs=self.p, rng=rng).run(m_chain)
                m_chain = r_ga.m
                r_b = hja.run(m_chain)
                outputs.append(r_b)
                self.log("  block %d/%d: GA m*=%.4f -> HJA m*(%d)=%.4f"
                         % (b, self.s, r_ga.J, b, r_b.J))
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = []
                for b in range(1, self.s + 1):
                    r_ga = BinaryGeneticAlgorithm(prob, pop_size=self.pop_size,
                                                  epochs=self.p, rng=rng).run(m_chain)
                    m_chain = r_ga.m
                    self.log("  block %d/%d: GA m*=%.4f (HJA dispatched)"
                             % (b, self.s, r_ga.J))
                    futures.append(ex.submit(hja.run, m_chain.copy()))
                outputs = [fut.result() for fut in futures]
        best = max(range(len(outputs)), key=lambda i: outputs[i].J)
        self.log("  stop: best is m*(%d) with J=%.4f" % (best + 1, outputs[best].J))
        return OptResult(outputs[best].m.copy(), outputs[best].J)


# ---------------------------------------------------------------------------
class Smooth:
    """The paper's *Smooth* step (S2-4, Eq. 24, Fig. S5c).

    A 'structure' is a maximal run of consecutive rings with equal gray
    level; its aspect ratio is alpha = (protrusion height) / (run width),
    the protrusion measured from the lower of the two neighbouring runs.
    For alpha > alpha0 the height is reduced to

        alpha_new = alpha0 + (alpha - alpha0) exp(-(alpha - alpha0) / beta)

    (identity below alpha0, saturating reduction above -- the qualitative
    curves of Fig. S5c; the paper suggests beta = 5..10).
    """

    def __init__(self, prob: MDLProblem, alpha0: float = 2.0,
                 beta: float = 7.0) -> None:
        self.prob, self.alpha0, self.beta = prob, float(alpha0), float(beta)

    def run(self, m_vec: IntVec) -> OptResult:
        prob = self.prob
        m = m_vec.astype(np.int32).copy()
        h = m * prob.dh
        N = prob.N
        runs: List[Tuple[int, int]] = []
        start = 0
        for i in range(1, N + 1):
            if i == N or m[i] != m[start]:
                runs.append((start, i))
                start = i
        for ri, (a, b) in enumerate(runs):
            width = (b - a) * prob.delta
            h0 = h[a]
            left = h[runs[ri - 1][0]] if ri > 0 else 0.0
            right = h[runs[ri + 1][0]] if ri < len(runs) - 1 else 0.0
            base = min(left, right)
            prot = h0 - base
            if prot <= 0:
                continue
            alpha = prot / width
            if alpha <= self.alpha0:
                continue
            alpha_new = self.alpha0 + (alpha - self.alpha0) * np.exp(
                -(alpha - self.alpha0) / self.beta)
            h_new = base + alpha_new * width
            m_new = int(round(h_new / prob.dh))
            m[a:b] = np.clip(m_new, 0, prob.M)
        return OptResult(m, prob.fom(m))


# ---------------------------------------------------------------------------
class GradientRefine:
    """The paper's *Gradient* step: adaptive-step ascent on continuous
    heights (S2-2, Fig. S2), then rounding to gray levels.

    Each iteration moves along the normalized analytic gradient
    (``MDLProblem.grad_h``) with the current step; an improving step is
    kept and the step grows by ``grow``, otherwise the step shrinks by
    ``shrink``; stop when the step falls below 1e-4 dh or after
    ``iters`` iterations.

    ``softmin_beta_final`` (softmin problems only) anneals
    ``prob.softmin_beta`` geometrically from its current value to this
    value over the iterations -- the standard minimax continuation
    (start soft / mean-like, end nearly worst-case). Since J changes
    meaning as beta moves, the accept/reject reference is recomputed at
    every beta update; ``prob.softmin_beta`` is left at the final value.
    """

    def __init__(self, prob: MDLProblem, step0: Optional[float] = None,
                 iters: int = 300, shrink: float = 0.5, grow: float = 1.1,
                 softmin_beta_final: Optional[float] = None) -> None:
        self.prob = prob
        self.step0 = step0
        self.iters = int(iters)
        self.shrink, self.grow = float(shrink), float(grow)
        self.softmin_beta_final = softmin_beta_final

    def run(self, m_vec: IntVec) -> OptResult:
        prob = self.prob
        h = m_vec.astype(float) * prob.dh
        f = prob.fom_h(h)
        step = self.step0 if self.step0 is not None else 0.05 * prob.dh * prob.M
        beta_final = self.softmin_beta_final
        anneal = (beta_final is not None and prob.fom_mode == "softmin"
                  and beta_final != prob.softmin_beta)
        if anneal and beta_final is not None:
            b0 = float(prob.softmin_beta)
            ratio = (float(beta_final) / b0) ** (1.0 / max(self.iters - 1, 1))
        history: List[HistoryEntry] = []
        for it in range(self.iters):
            if anneal:
                prob.softmin_beta = b0 * ratio ** it
                f = prob.fom_h(h)            # J is beta-dependent: re-anchor
            g = prob.grad_h(h)
            gn = np.max(np.abs(g))
            if gn == 0:
                break
            h_try = np.clip(h + step * g / gn, 0.0, prob.h_max)
            f_try = prob.fom_h(h_try)
            if f_try > f:
                h, f = h_try, f_try
                step *= self.grow
            else:
                step *= self.shrink
                if step < 1e-4 * prob.dh:
                    break
            history.append(("GD", it + 1, f))
        m = np.clip(np.rint(h / prob.dh), 0, prob.M).astype(np.int32)
        return OptResult(m, prob.fom(m), history)
