"""
mdl_core.py
===========

Core physics + optimization machinery for designing an Achromatic
Multilevel Diffractive Lens (AMDL / MDL) by light frequency-domain
coherence optimization, following:

    Xiao et al., "Large-scale achromatic flat lens by light
    frequency-domain coherence optimization",
    Light: Science & Applications (2022) 11:323.
    https://doi.org/10.1038/s41377-022-01024-y

Model
-----
The lens is a set of N concentric rings of equal width DELTA, ring i
covering [i*DELTA, (i+1)*DELTA), with quantized height

    h_i = m_i * dh,      m_i in {0 .. M}      (gray levels)

on a flat substrate; the lens provides phase-only modulation

    phi_lens(rho, w) = (w/c) * (n(w) - 1) * h(rho)

The phase distortion w.r.t. the ideal hyperbolic profile is

    dphi(rho, w) = phi_lens(rho, w) + (w/c) * (sqrt(rho^2+F^2) - F)

Figure of merit (Eq. 4 of the paper, paraxial-normalized so that a
perfect achromat gives 1):

    J_w(F) = < | U(w) |^2 >_w
    U(w)   = (2F/R^2) * SUM_i  e^{i dphi_i(w)} * rho_i * DELTA / r_i

with r_i = sqrt(rho_i^2 + F^2), <.>_w a uniform average over angular
frequency samples spanning the working band.

Everything is vectorized on a per-gray-level phasor lookup table:

    L[w, m] = exp(i k(w) (n(w)-1) m dh)
    G[w, i] = (2F/R^2) * rho_i * DELTA / r_i * exp(i k(w) (sqrt(rho_i^2+F^2)-F))
    U(w)    = SUM_i G[w, i] * L[w, m_i]

so one FOM evaluation is O(Nw * N) and a *single-ring* update is O(Nw)
(delta evaluation), which makes Hooke-Jeeves cheap.
"""

from __future__ import annotations

import numpy as np


C_UM_PER_S = 2.99792458e14  # speed of light [um/s]


# ---------------------------------------------------------------------------
# Material: AZ4562 photoresist (Cauchy fit to the dispersion shown in the
# paper's Fig. S8; average n ~ 1.63 over 400-1100 nm as stated in S2-1)
# ---------------------------------------------------------------------------
def n_az4562(lam_um):
    """Refractive index of AZ4562 (real part), lam in micrometers."""
    lam = np.asarray(lam_um, dtype=float)
    return 1.594 + 0.01152 / lam ** 2


# the paper's 14 measurement wavelengths for S3 (um): 400-1100 nm in
# 50 nm steps, 800 nm omitted (Fig. 2e / Fig. 4)
PAPER_COMB_14 = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                          0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10])


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
class MDLProblem:
    """Holds geometry, band, material and the precomputed phasor tables."""

    def __init__(self,
                 diameter_um: float,
                 na: float,
                 lam_min_um: float,
                 lam_max_um: float,
                 ring_width_um: float,
                 h_max_um: float,
                 dh_um: float,
                 n_wavelengths: int = 25,
                 n_func=n_az4562):
        self.R = 0.5 * diameter_um
        self.na = na
        # focal length from NA (in air): NA = R / sqrt(R^2 + F^2)
        self.F = self.R * np.sqrt(1.0 / na ** 2 - 1.0)
        self.delta = ring_width_um
        self.N = int(round(self.R / ring_width_um))
        self.h_max = h_max_um
        self.dh = dh_um
        self.M = int(round(h_max_um / dh_um))          # gray levels 0..M
        self.lam_min = lam_min_um
        self.lam_max = lam_max_um
        self.n_func = n_func

        # sample uniformly in angular frequency (as the paper's <.>_w)
        w_min = 2.0 * np.pi * C_UM_PER_S / lam_max_um
        w_max = 2.0 * np.pi * C_UM_PER_S / lam_min_um
        self.omega = np.linspace(w_min, w_max, n_wavelengths)
        self.lam = 2.0 * np.pi * C_UM_PER_S / self.omega     # [um]
        self.k = 2.0 * np.pi / self.lam                       # [1/um]
        self.n = n_func(self.lam)

        # ring center radii
        self.rho = (np.arange(self.N) + 0.5) * self.delta
        r = np.sqrt(self.rho ** 2 + self.F ** 2)

        # geometric phasor table G[w, i]  (includes ideal-lens defocus term)
        geo_phase = self.k[:, None] * (r[None, :] - self.F)
        amp = (2.0 * self.F / self.R ** 2) * self.rho * self.delta / r
        self.G = amp[None, :] * np.exp(1j * geo_phase)

        # gray-level phasor table L[w, m], m = 0..M
        kn = self.k * (self.n - 1.0)                          # [1/um]
        m = np.arange(self.M + 1)
        self.L = np.exp(1j * kn[:, None] * self.dh * m[None, :])

        # per-w normalization sanity: sum_i |G| = (2F/R^2) sum rho d / r ~ <1
        self.Nw = n_wavelengths

    def set_wavelengths(self, lams_um):
        """Rebuild the phasor tables on an EXPLICIT wavelength list [um].

        Use this for comb objectives (e.g. the paper's 14 measurement
        wavelengths). The default constructor samples uniformly in
        angular frequency, which lands on different wavelengths.
        """
        self.lam = np.asarray(lams_um, dtype=float)
        self.omega = 2.0 * np.pi * C_UM_PER_S / self.lam
        self.k = 2.0 * np.pi / self.lam
        self.n = self.n_func(self.lam)
        r = np.sqrt(self.rho ** 2 + self.F ** 2)
        geo_phase = self.k[:, None] * (r[None, :] - self.F)
        amp = (2.0 * self.F / self.R ** 2) * self.rho * self.delta / r
        self.G = amp[None, :] * np.exp(1j * geo_phase)
        kn = self.k * (self.n - 1.0)
        m = np.arange(self.M + 1)
        self.L = np.exp(1j * kn[:, None] * self.dh * m[None, :])
        self.Nw = self.lam.size
        return self

    # -- FOM ---------------------------------------------------------------
    # fom_mode selects how per-wavelength intensities are aggregated:
    #   "mean"    : J = < |U|^2 >_w        (the paper's Eq. 2; default).
    #               Indifferent to balance -- may zero some wavelengths.
    #   "geomean" : J = exp< ln(|U|^2+eps) >_w  (product of intensities).
    #               Any wavelength going dark kills J, so every target
    #               wavelength keeps a focus (balanced comb, paper
    #               Fig. 2e behavior).
    fom_mode = "mean"
    _geo_eps = 1e-6

    def field(self, m_vec):
        """U(w) for gray-level vector m_vec (ints 0..M). Shape (Nw,)."""
        return np.sum(self.G * self.L[:, m_vec], axis=1)

    def fom(self, m_vec):
        """Aggregated J (see fom_mode)."""
        return self.fom_from_field(self.field(m_vec))

    def fom_from_field(self, U):
        I = np.abs(U) ** 2
        if self.fom_mode == "geomean":
            return float(np.exp(np.mean(np.log(I + self._geo_eps))))
        return float(np.mean(I))

    def delta_field(self, U, i, m_old, m_new):
        """Field after changing ring i from m_old to m_new (O(Nw))."""
        return U + self.G[:, i] * (self.L[:, m_new] - self.L[:, m_old])

    # -- continuous-height versions (for gradient step) --------------------
    def field_h(self, h_vec):
        kn = self.k * (self.n - 1.0)
        return np.sum(self.G * np.exp(1j * kn[:, None] * h_vec[None, :]),
                      axis=1)

    def fom_h(self, h_vec):
        return self.fom_from_field(self.field_h(h_vec))

    def grad_h(self, h_vec):
        """Analytic dJ/dh_i (both fom modes)."""
        kn = self.k * (self.n - 1.0)                     # (Nw,)
        E = self.G * np.exp(1j * kn[:, None] * h_vec[None, :])  # (Nw, N)
        U = np.sum(E, axis=1)                            # (Nw,)
        # dI_w/dh_i = 2 Re{ conj(U_w) * i * kn_w * E_wi }
        dI = 2.0 * np.real(np.conj(U)[:, None] * 1j * kn[:, None] * E)
        if self.fom_mode == "geomean":
            I = np.abs(U) ** 2 + self._geo_eps           # (Nw,)
            J = float(np.exp(np.mean(np.log(I))))
            # dJ/dh_i = J * < dI_w/dh_i / I_w >_w
            return J * np.mean(dI / I[:, None], axis=0)
        return np.mean(dI, axis=0)


# ---------------------------------------------------------------------------
# Upper bound of J_w(F)  (paper Eq. S14-S15)
# ---------------------------------------------------------------------------
def upper_bound_jf(diameter_um, na, lam_min_um, lam_max_um,
                   h_max_um, dh_um, n_rho=256, n_wavelengths=512,
                   n_func=n_az4562, n_candidates=4):
    """
    max J_w(F) upper bound: exchange max and integration (Eq. S15).

    max Re J_w(rho1, rho2) depends on the heights only through
    delta_h = h1 - h2 = delta_m * dh:

        max Re J = max_dm < cos( (w/c)[(n(w)-1) dh dm + (r1 - r2)] ) >_w

    Brute-force traversal over dm with coarse w-sampling aliases badly
    (path differences r1-r2 reach hundreds of um), so instead we evaluate
    only the *candidate* dm near the stationary value

        dm* = -(r1 - r2) / ((n_bar - 1) dh)

    (plus the clipped boundaries +-M), each on a dense w grid. Far from
    dm*, the band-average phasor is negligible, so the max is attained
    within a few gray levels of dm*. This also captures material
    dispersion, which is what limits wide-band coherence at large |dh|.
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
    seen = set()
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

    # integrate with paraxial weights w(rho) = 2 rho / R^2 with F/r obliquity
    wgt = (2.0 * F / R ** 2) * rho * (R / n_rho) / r
    wgt = wgt / np.sum(wgt)
    bound = float(wgt @ B @ wgt)
    return bound


# ---------------------------------------------------------------------------
# Optimizers (paper S2-2): GA + HJA blocks -> Smooth -> Gradient
# ---------------------------------------------------------------------------
def genetic_algorithm(prob: MDLProblem, m_init, pop_size=40, epochs=200,
                      p_cross=0.8, p_mut=0.01, elite=2, rng=None,
                      log_every=0, log_list=None):
    """Integer-vector GA (mutation flips to random level; crossover 1-pt)."""
    rng = rng or np.random.default_rng()
    N, M = prob.N, prob.M

    pop = np.empty((pop_size, N), dtype=np.int32)
    pop[0] = m_init
    for p in range(1, pop_size):
        if p < pop_size // 2:
            # perturbed copies of the seed
            pop[p] = m_init
            idx = rng.random(N) < 0.05
            pop[p, idx] = rng.integers(0, M + 1, idx.sum())
        else:
            pop[p] = rng.integers(0, M + 1, N)

    fit = np.array([prob.fom(ind) for ind in pop])

    for ep in range(epochs):
        order = np.argsort(fit)[::-1]
        pop, fit = pop[order], fit[order]
        new_pop = [pop[i].copy() for i in range(elite)]
        while len(new_pop) < pop_size:
            # tournament selection
            a, b = rng.integers(0, pop_size, 2)
            pa = pop[a] if fit[a] > fit[b] else pop[b]
            a, b = rng.integers(0, pop_size, 2)
            pb = pop[a] if fit[a] > fit[b] else pop[b]
            c1, c2 = pa.copy(), pb.copy()
            if rng.random() < p_cross:
                cut = rng.integers(1, N)
                c1[cut:], c2[cut:] = pb[cut:].copy(), pa[cut:].copy()
            for c in (c1, c2):
                mask = rng.random(N) < p_mut
                c[mask] = rng.integers(0, M + 1, mask.sum())
                new_pop.append(c)
        pop = np.array(new_pop[:pop_size], dtype=np.int32)
        fit = np.array([prob.fom(ind) for ind in pop])
        if log_every and (ep + 1) % log_every == 0:
            best = float(fit.max())
            if log_list is not None:
                log_list.append(("GA", ep + 1, best))
    order = np.argsort(fit)[::-1]
    return pop[order[0]].copy(), float(fit[order[0]])


def hooke_jeeves(prob: MDLProblem, m_init, d0=None, alpha=1.0,
                 max_sweeps=200, rng=None, log_list=None):
    """
    Integer Hooke-Jeeves direct search with delta-evaluation.
    Exploratory move: each coordinate tries +d and -d gray levels.
    Pattern move: m + [alpha*(m_new - m_old)].
    Step halves (integer) when no improvement; stops at d < 1.
    """
    N, M = prob.N, prob.M
    d = d0 if d0 is not None else max(1, M // 8)
    m = m_init.astype(np.int32).copy()
    U = prob.field(m)
    f = prob.fom_from_field(U)
    m_prev = m.copy()
    sweeps = 0

    def explore(m_base, U_base, f_base):
        m_c = m_base.copy()
        U_c = U_base.copy()
        f_c = f_base
        for i in range(N):
            mi = m_c[i]
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

    while d >= 1 and sweeps < max_sweeps:
        m_new, U_new, f_new = explore(m, U, f)
        if f_new > f + 1e-12:
            # pattern (acceleration) move
            patt = m_new + np.rint(alpha * (m_new - m_prev)).astype(np.int32)
            np.clip(patt, 0, M, out=patt)
            U_p = prob.field(patt)
            f_p = prob.fom_from_field(U_p)
            m_prev = m.copy()
            if f_p > f_new:
                m, U, f = patt, U_p, f_p
            else:
                m, U, f = m_new, U_new, f_new
        else:
            d //= 2
        sweeps += 1
        if log_list is not None:
            log_list.append(("HJA", sweeps, f))
    return m, f


def search(prob: MDLProblem, m_init=None, blocks=4, ga_epochs=60,
           pop_size=40, rng=None, log_list=None, verbose=print,
           chain="best"):
    """
    Paper's *Search* (Fig. S3): s blocks of [GA (p epochs) -> HJA].

    Mapping to the Fig. S3 boxes:
      "Initialize m"  -> m_init (random per S2-1, or a harmonic seed)
      per block:  GA  -> genetic_algorithm(...)   giving m*   (= m_ga)
                  HJA -> hooke_jeeves(m*)         giving m*(b) (= m_h)
      "stop"      -> return the best m*(b) over all blocks

    chain = "ga"   : paper-exact wiring -- block b+1's GA continues from
                     block b's GA output m* (the down-arrow in Fig. S3);
                     the HJA results m*(1..s) are only collected at the
                     end. (In the paper the HJA of block b can then run
                     in parallel with the GA of block b+1; here blocks
                     run sequentially, which changes wall-clock only,
                     not the algorithm.)
    chain = "best" : engineering variant (default) -- block b+1's GA is
                     seeded with the best HJA result so far. Departs
                     from Fig. S3 but is monotonically at least as good
                     per block; with a strong seed both wirings give
                     identical results (GA rarely beats the elite).
    """
    rng = rng or np.random.default_rng(0)
    if m_init is None:
        m_init = rng.integers(0, prob.M + 1, prob.N).astype(np.int32)
    m_ga = m_init
    best_m, best_f = None, -np.inf
    for b in range(blocks):
        m_ga, f_ga = genetic_algorithm(prob, m_ga, pop_size=pop_size,
                                       epochs=ga_epochs, rng=rng,
                                       log_every=10, log_list=log_list)
        m_h, f_h = hooke_jeeves(prob, m_ga, rng=rng, log_list=log_list)
        verbose("  block %d/%d: GA %.4f -> HJA %.4f" % (b + 1, blocks,
                                                        f_ga, f_h))
        if f_h > best_f:
            best_m, best_f = m_h.copy(), f_h
        if chain == "best":
            m_ga = best_m.copy()
        # chain == "ga": m_ga stays the GA output (paper Fig. S3)
    return best_m, best_f


# ---------------------------------------------------------------------------
# VERBATIM Fig. S3 implementation: multistep GA + HJA combination
# ---------------------------------------------------------------------------
def _n_bits(M):
    """Bits per gray level, as in S2-2 (6/7/8 bits for M=32/64/192)."""
    return int(np.ceil(np.log2(M + 1)))


def _decode(bits, N, nb, M):
    """Binary string (N*nb,) -> integer vector (N,), clamped to M (S2-2)."""
    w = (1 << np.arange(nb - 1, -1, -1)).astype(np.int64)
    vals = bits.reshape(N, nb) @ w
    return np.minimum(vals, M).astype(np.int32)


def _encode(m_vec, nb):
    """Integer vector -> binary string (big-endian per gene)."""
    N = m_vec.size
    out = np.empty(N * nb, dtype=np.uint8)
    for b in range(nb):
        out[b::nb] = (m_vec >> (nb - 1 - b)) & 1
    return out


def genetic_algorithm_binary(prob: MDLProblem, m_init=None, pop_size=40,
                             epochs=200, p_cross=0.8, p_mut=None,
                             elite=2, rng=None):
    """
    Conventional binary-coded GA exactly as described in S2-2:
    each gray level encoded in nb bits, individuals are binary vectors of
    length nb*N, single-point crossover and bit-flip mutation act on the
    bit string, decoded values exceeding M are clamped to M.
    Returns (m_elite, J_elite).
    """
    rng = rng or np.random.default_rng()
    N, M = prob.N, prob.M
    nb = _n_bits(M)
    L = N * nb
    if p_mut is None:
        p_mut = 1.0 / L                     # conventional ~1/L bit-flip rate

    pop = rng.integers(0, 2, size=(pop_size, L), dtype=np.uint8)
    if m_init is not None:
        pop[0] = _encode(m_init.astype(np.int32), nb)

    def fitness(ind):
        return prob.fom(_decode(ind, N, nb, M))

    fit = np.array([fitness(ind) for ind in pop])

    for _ in range(epochs):
        order = np.argsort(fit)[::-1]
        pop, fit = pop[order], fit[order]
        new_pop = [pop[i].copy() for i in range(elite)]
        while len(new_pop) < pop_size:
            a, b = rng.integers(0, pop_size, 2)
            pa = pop[a] if fit[a] > fit[b] else pop[b]
            a, b = rng.integers(0, pop_size, 2)
            pb = pop[a] if fit[a] > fit[b] else pop[b]
            c1, c2 = pa.copy(), pb.copy()
            if rng.random() < p_cross:
                cut = int(rng.integers(1, L))
                c1[cut:], c2[cut:] = pb[cut:].copy(), pa[cut:].copy()
            for c in (c1, c2):
                mask = rng.random(L) < p_mut
                c[mask] ^= 1
                new_pop.append(c)
        pop = np.array(new_pop[:pop_size], dtype=np.uint8)
        fit = np.array([fitness(ind) for ind in pop])

    ib = int(np.argmax(fit))
    return _decode(pop[ib], N, nb, M), float(fit[ib])


def hooke_jeeves_verbatim(prob: MDLProblem, m_init, d0=None, alpha=1.0):
    """
    HJA per S2-2 / Fig. S1, literal reading:
      * exploratory sweep: each coordinate tries +d and -d gray levels
        (both evaluated; the better improving one is kept);
      * improving sweep -> acceleration (pattern) move
            Y = m_new + floor(alpha * (m_new - m_prev)),  clipped to [0,M];
      * failed sweep -> d = ceil(d/2); terminate after the d = 1 sweep
        fails (ceil keeps d >= 1, so d == 1 is the explicit exit level).
    Uses O(Nw) delta-evaluation per trial move.
    """
    N, M = prob.N, prob.M
    d = d0 if d0 is not None else max(1, M // 8)
    m = m_init.astype(np.int32).copy()
    U = prob.field(m)
    f = prob.fom_from_field(U)
    m_prev = m.copy()

    def explore(m_base, U_base, f_base):
        m_c, U_c, f_c = m_base.copy(), U_base.copy(), f_base
        for i in range(N):
            mi = m_c[i]
            best_f, best_m, best_U = f_c, None, None
            for step in (d, -d):
                mn = mi + step
                if mn < 0 or mn > M:
                    continue
                U_try = prob.delta_field(U_c, i, m_c[i], mn)
                f_try = prob.fom_from_field(U_try)
                if f_try > best_f:
                    best_f, best_m, best_U = f_try, mn, U_try
            if best_m is not None:
                m_c[i], U_c, f_c = best_m, best_U, best_f
        return m_c, U_c, f_c

    while True:
        m_new, U_new, f_new = explore(m, U, f)
        if f_new > f + 1e-12:
            patt = m_new + np.floor(alpha * (m_new - m_prev)).astype(np.int32)
            np.clip(patt, 0, M, out=patt)
            U_p = prob.field(patt)
            f_p = prob.fom_from_field(U_p)
            m_prev = m.copy()
            if f_p > f_new:
                m, U, f = patt, U_p, f_p
            else:
                m, U, f = m_new, U_new, f_new
        else:
            if d == 1:
                break
            d = (d + 1) // 2                # ceil(d/2)
    return m, f


def multistep_GA_HJA_combo(prob: MDLProblem, m_init=None, s=4, p=60,
                           pop_size=40, d0=None, alpha=1.0, rng=None,
                           verbose=print, parallel_hja=False):
    """
    VERBATIM implementation of Fig. S3 (supplementary of Xiao et al.,
    Light Sci. Appl. 11:323, 2022):

        start -> Initialize m
        block b = 1..s:
            m*_(b)   = GA( m*_(b-1), p epochs )     # GA chain: fed by the
                                                    # PREVIOUS BLOCK'S GA
                                                    # output (down-arrow)
            m*^(b)   = HJA( m*_(b) )                # side branch only
        stop: return argmax_b J( m*^(b) )

    HJA never feeds back into the GA chain; hence (optionally,
    parallel_hja=True) the HJA of block b runs concurrently with the GA
    of block b+1 -- identical results, shorter wall clock.
    Initialize m: random per S2-1 if m_init is None.
    """
    rng = rng or np.random.default_rng(0)
    if m_init is None:
        m_init = rng.integers(0, prob.M + 1, prob.N).astype(np.int32)

    outputs = []                            # [(m*^(b), J)]
    m_chain = m_init

    if not parallel_hja:
        for b in range(1, s + 1):
            m_chain, f_ga = genetic_algorithm_binary(
                prob, m_init=m_chain, pop_size=pop_size, epochs=p, rng=rng)
            m_b, f_b = hooke_jeeves_verbatim(prob, m_chain, d0=d0,
                                             alpha=alpha)
            outputs.append((m_b, f_b))
            verbose("  block %d/%d: GA m*=%.4f -> HJA m*(%d)=%.4f"
                    % (b, s, f_ga, b, f_b))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = []
            for b in range(1, s + 1):
                m_chain, f_ga = genetic_algorithm_binary(
                    prob, m_init=m_chain, pop_size=pop_size, epochs=p,
                    rng=rng)
                verbose("  block %d/%d: GA m*=%.4f (HJA dispatched)"
                        % (b, s, f_ga))
                futures.append(ex.submit(hooke_jeeves_verbatim, prob,
                                         m_chain.copy(), d0, alpha))
            outputs = [fut.result() for fut in futures]

    best = max(range(len(outputs)), key=lambda i: outputs[i][1])
    m_star, f_star = outputs[best]
    verbose("  stop: best is m*(%d) with J=%.4f" % (best + 1, f_star))
    return m_star.copy(), f_star


# -- Smooth (paper S2-4, qualitative reproduction of Fig. S5c) --------------
def smooth(prob: MDLProblem, m_vec, alpha0=2.0, beta=7.0):
    """
    Remove high-aspect-ratio structures.

    A 'structure' is a maximal run of consecutive rings with equal gray
    level. Its aspect ratio is alpha = (protrusion height) / (run width),
    protrusion measured w.r.t. the lower of the two neighbor runs.
    For alpha > alpha0 the height is reduced following

        alpha_new = alpha0 + (alpha - alpha0) * exp(-(alpha - alpha0)/beta)

    which reproduces the qualitative alpha_new/alpha curves of Fig. S5c
    (identity below alpha0, saturating reduction above).
    """
    m = m_vec.astype(np.int32).copy()
    h = m * prob.dh
    N = prob.N

    # find runs
    runs = []
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
        if alpha <= alpha0:
            continue
        alpha_new = alpha0 + (alpha - alpha0) * np.exp(-(alpha - alpha0)
                                                       / beta)
        h_new = base + alpha_new * width
        m_new = int(round(h_new / prob.dh))
        m[a:b] = np.clip(m_new, 0, prob.M)
    return m


# -- Gradient (paper S2-2, gradient descent on continuous h, then round) ----
def gradient_refine(prob: MDLProblem, m_vec, step0=None, iters=300,
                    shrink=0.5, grow=1.1, log_list=None):
    h = m_vec.astype(float) * prob.dh
    f = prob.fom_h(h)
    step = step0 if step0 is not None else 0.05 * prob.dh * prob.M
    for it in range(iters):
        g = prob.grad_h(h)
        gn = np.max(np.abs(g))
        if gn == 0:
            break
        h_try = np.clip(h + step * g / gn, 0.0, prob.h_max)
        f_try = prob.fom_h(h_try)
        if f_try > f:
            h, f = h_try, f_try
            step *= grow
        else:
            step *= shrink
            if step < 1e-4 * prob.dh:
                break
        if log_list is not None:
            log_list.append(("GD", it + 1, f))
    m = np.clip(np.rint(h / prob.dh), 0, prob.M).astype(np.int32)
    return m, prob.fom(m)


# ---------------------------------------------------------------------------
# Heuristic seeds
# ---------------------------------------------------------------------------
def seed_harmonic(prob: MDLProblem, lam0_um, p_harmonic=1):
    """
    Quantized 'harmonic diffractive lens' seed: ideal hyperbolic phase at
    design wavelength lam0 folded to p*2pi, converted to height.
    """
    n0 = float(prob.n_func(lam0_um))
    r = np.sqrt(prob.rho ** 2 + prob.F ** 2)
    opd = r - prob.F                                  # [um]
    # need (n0-1)*h == -opd  (mod p*lam0)
    h = ((-opd) % (p_harmonic * lam0_um)) / (n0 - 1.0)
    m = np.clip(np.rint(h / prob.dh), 0, prob.M).astype(np.int32)
    return m


def seed_max_harmonic(prob: MDLProblem, lam0_um):
    """Harmonic seed using the largest fold height that fits in h_max."""
    n0 = float(prob.n_func(lam0_um))
    step = lam0_um / (n0 - 1.0)
    p = max(1, int(np.floor(prob.h_max / step)))
    return seed_harmonic(prob, lam0_um, p_harmonic=p)