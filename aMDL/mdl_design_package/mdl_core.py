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
        # keep any efficiency correction / overlap tables consistent
        # with the new wavelength grid
        if hasattr(self, "_eta_table"):
            self._G_raw = self.G.copy()
            self._reapply_eta()
        if self.objective == "overlap":
            self.enable_overlap_fom(*self._overlap_cfg)
        return self

    # -- FOM ---------------------------------------------------------------
    # fom_mode selects how per-wavelength values are aggregated:
    #   "mean"    : J = < I >_w            (the paper's Eq. 2; default).
    #               Indifferent to balance -- may zero some wavelengths.
    #   "geomean" : J = exp< ln(I+eps) >_w (product of intensities).
    #               Any wavelength going dark kills J, so every target
    #               wavelength keeps a focus (balanced comb, paper
    #               Fig. 2e behavior).
    #   "softmin" : J = -(1/beta) ln < exp(-beta I) >_w, the smooth
    #               minimum (log-sum-exp; Boyd & Vandenberghe, Convex
    #               Optimization, sec. 3.1.5). Interpolates from "mean"
    #               (beta -> 0) to the true worst-case min_w I
    #               (beta -> inf); the gradient weights are a softmax
    #               concentrated on the WORST wavelengths. This is the
    #               epigraph/minimax formulation standard in broadband
    #               grating-coupler inverse design (Lalau-Keraly et al.,
    #               Opt. Express 21, 21693 (2013); Michaels &
    #               Yablonovitch, Opt. Express 26, 4766 (2018)).
    #               Set softmin_beta relative to typical I values: with
    #               I ~ 0.1, beta = 50-300 is a useful range; anneal
    #               upward (see gradient_refine's softmin_beta_final).
    #
    # objective selects WHAT per-wavelength value I is:
    #   "onaxis"  : I_w = |U(w)|^2, the on-axis focal intensity in the
    #               paper's paraxial normalization (default; identical
    #               to the pre-refactor behavior).
    #   "overlap" : I_w = eta_w, the ENCIRCLED ENERGY inside a focal
    #               disc of radius r_enc, normalized per wavelength to
    #               the same integral for the ideal (perfectly phase-
    #               matched) lens -- a diffraction-grating-coupler style
    #               mode-overlap objective (couplers maximize overlap
    #               with the target fiber mode, not intensity at a
    #               point). Energy parked in halo, sidelobes or
    #               satellite foci reduces eta_w by construction.
    #               Enable with enable_overlap_fom(); adds a J0 kernel
    #               table K[w, q, i] (memory: Nw*n_r0*N complex).
    fom_mode = "mean"
    objective = "onaxis"
    _geo_eps = 1e-6
    softmin_beta = 100.0

    # -- aggregation over wavelengths (shared by both objectives) ----------
    def _aggregate(self, I):
        """Scalar J from per-wavelength values I (shape (Nw,))."""
        if self.fom_mode == "geomean":
            return float(np.exp(np.mean(np.log(I + self._geo_eps))))
        if self.fom_mode == "softmin":
            b = self.softmin_beta
            a = -b * (I - I.min())               # <= 0, stable
            return float(I.min() - np.log(np.mean(np.exp(a))) / b)
        return float(np.mean(I))

    def _aggregate_weights(self, I):
        """w_w such that dJ/dh = sum_w w_w dI_w/dh (same modes)."""
        Nw = I.size
        if self.fom_mode == "geomean":
            Ieps = I + self._geo_eps
            J = float(np.exp(np.mean(np.log(Ieps))))
            return J / (Nw * Ieps)
        if self.fom_mode == "softmin":
            a = -self.softmin_beta * (I - I.min())
            e = np.exp(a)
            return e / e.sum()                   # softmax on worst lines
        return np.full(Nw, 1.0 / Nw)

    def field(self, m_vec):
        """Focal field for gray-level vector m_vec (ints 0..M).

        objective "onaxis":  U(w), shape (Nw,).
        objective "overlap": U(w, r0_q) on the encircled-energy
        quadrature nodes, shape (Nw, n_r0).
        """
        if self.objective == "overlap":
            return np.einsum("wqi,wi->wq", self.K, self.L[:, m_vec])
        return np.sum(self.G * self.L[:, m_vec], axis=1)

    def fom(self, m_vec):
        """Aggregated J (see fom_mode / objective)."""
        return self.fom_from_field(self.field(m_vec))

    def per_wavelength(self, U):
        """Per-wavelength values I_w (intensity or encircled eta)."""
        if U.ndim == 2:                          # overlap field (Nw, nq)
            return np.einsum("wq,wq->w", self._enc_w,
                             np.abs(U) ** 2) / self._enc_D
        return np.abs(U) ** 2

    def fom_from_field(self, U):
        return self._aggregate(self.per_wavelength(U))

    def delta_field(self, U, i, m_old, m_new):
        """Field after changing ring i from m_old to m_new (O(Nw))."""
        dL = self.L[:, m_new] - self.L[:, m_old]
        if self.objective == "overlap":
            return U + self.K[:, :, i] * dL[:, None]
        return U + self.G[:, i] * dL

    # -- continuous-height versions (for gradient step) --------------------
    def field_h(self, h_vec):
        kn = self.k * (self.n - 1.0)
        E = np.exp(1j * kn[:, None] * h_vec[None, :])
        if self.objective == "overlap":
            return np.einsum("wqi,wi->wq", self.K, E)
        return np.sum(self.G * E, axis=1)

    def fom_h(self, h_vec):
        return self.fom_from_field(self.field_h(h_vec))

    def grad_h(self, h_vec):
        """Analytic dJ/dh_i (all fom modes, both objectives)."""
        kn = self.k * (self.n - 1.0)                     # (Nw,)
        E = np.exp(1j * kn[:, None] * h_vec[None, :])    # (Nw, N)
        if self.objective == "overlap":
            P = np.einsum("wqi,wi->wq", self.K, E)       # (Nw, nq)
            I = self.per_wavelength(P)                   # eta_w
            # deta_w/dh_i = (2/D_w) sum_q wq Re{conj(P) i kn K E}
            T = np.einsum("wq,wq,wqi->wi", self._enc_w,
                          np.conj(P), self.K)            # (Nw, N)
            dI = (2.0 * kn / self._enc_D)[:, None] \
                * np.real(1j * T * E)
        else:
            EG = self.G * E                              # (Nw, N)
            U = np.sum(EG, axis=1)                       # (Nw,)
            I = np.abs(U) ** 2
            # dI_w/dh_i = 2 Re{ conj(U_w) * i * kn_w * EG_wi }
            dI = 2.0 * np.real(np.conj(U)[:, None]
                               * 1j * kn[:, None] * EG)
        w = self._aggregate_weights(I)
        return np.sum(w[:, None] * dI, axis=0)

    # -- overlap (encircled-energy) objective ------------------------------
    def enable_overlap_fom(self, r_enc_um=None, n_r0=24,
                           airy_factor=2.0):
        """
        Switch the objective to grating-coupler-style mode overlap:
        per-wavelength encircled energy in a focal disc, normalized to
        the ideal phase-matched lens at the same wavelength (a perfect
        achromat scores eta_w = 1 for all w).

            eta_w = int_0^r_enc(w) |U(r0,w)|^2 r0 dr0
                    / int_0^r_enc(w) |U_ideal(r0,w)|^2 r0 dr0

        The disc is CHROMATIC by default:

            r_enc(w) = airy_factor * 0.61 * lam_w / NA

        (airy_factor Airy radii at EACH wavelength), so the target
        mode has the same shape in diffraction units at every line.
        This matters for two coupled reasons, both measured on the
        first s3_comb_softmin_overlap run, which used one fixed disc
        of 1.22 lam_max/NA = 13.4 um:
          * a fixed lam_max-sized disc is ~5.5 Airy radii wide at
            400 nm -- the optimizer parked a diffuse halo inside it
            (softmin eta 0.18 while the true 3xFWHM-disc efficiency
            was 0.004: the objective was satisfied, the focus wasn't);
          * the disc radius sets a tolerated axial defocus
            dz ~ r_enc/NA (133 um for the fixed disc -- exactly the
            observed z-peak offsets and PSF elongation). The chromatic
            disc tightens this to ~airy_factor*0.61*lam/NA^2 per line
            (48 um at 400 nm), attacking the elongation at the source.

        U(r0,w) uses the paraxial Fourier-Bessel kernel: ring i
        contributes G[w,i] * J0(k rho_i r0 / r_i) (the r0=0 limit
        reproduces the on-axis U exactly). The radial integral is a
        Gauss-Legendre quadrature on n_r0 nodes per wavelength.

        Parameters
        ----------
        r_enc_um : None (default) = chromatic disc as above; a number
                   forces one FIXED disc radius for all wavelengths
                   (kept for comparability with the first run).
        n_r0     : quadrature nodes (default 24).
        airy_factor : disc radius in Airy-radius units (default 2.0 =
                   main lobe + first ring), used only when
                   r_enc_um is None.

        Memory: K is Nw x n_r0 x N complex (same dtype as G) -- e.g.
        14 x 24 x 2560 complex64 = 7 MB for a comb problem, but ~250 MB
        for a 500-sample continuous band; for continuous designs prefer
        running the discrete search on-axis and only the gradient
        polish with the overlap objective.

        set_wavelengths() / apply_efficiency() rebuild the tables with
        the SAME settings automatically. disable_overlap_fom() reverts.
        """
        from numpy.polynomial.legendre import leggauss
        self._overlap_cfg = (None if r_enc_um is None else float(r_enc_um),
                             int(n_r0), float(airy_factor))
        if r_enc_um is None:
            r_enc = airy_factor * 0.61 * self.lam / self.na  # (Nw,)
        else:
            r_enc = np.full(self.lam.size, float(r_enc_um))
        x, wgl = leggauss(int(n_r0))                     # on [-1, 1]
        r0 = 0.5 * r_enc[:, None] * (x + 1.0)[None, :]   # (Nw, nq)
        wq = 0.5 * r_enc[:, None] * wgl[None, :]         # GL weights
        self._enc_r0 = r0
        self._enc_w = wq * r0                            # includes r0 dr0
        self.r_enc = r_enc                               # per-w array
        r = np.sqrt(self.rho ** 2 + self.F ** 2)         # (N,)
        from scipy.special import j0    # same dependency as run_verify.py
        arg = (self.k[:, None, None] * self.rho[None, None, :]
               * r0[:, :, None] / r[None, None, :])      # (Nw, nq, N)
        B = j0(arg)
        self.K = (self.G[:, None, :] * B).astype(self.G.dtype)
        # ideal-lens denominator: all phase errors zero -> |G| in kernel
        Uid = np.einsum("wqi->wq", np.abs(self.G)[:, None, :] * B)
        self._enc_D = np.einsum("wq,wq->w", self._enc_w, Uid ** 2)
        self.objective = "overlap"
        return self

    def disable_overlap_fom(self):
        """Back to the on-axis point objective (frees the K table)."""
        self.objective = "onaxis"
        for attr in ("K", "_enc_D", "_enc_w", "_enc_r0",
                     "_overlap_cfg", "r_enc"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self

    # -- rigorous local-grating efficiency correction ----------------------
    def apply_efficiency(self, lam_um, r_um, corr):
        """
        Weight the phasor amplitudes by the square root of a per-
        (wavelength, radius) RELATIVE efficiency correction

            corr = eta_rigorous / eta_scalar        (1.0 = no change)

        i.e. the ratio of the RCWA efficiency of each local grating
        zone into its focusing order (computed externally: Ansys
        Lumerical RCWA, torcwa, grcwa; see write_zone_table /
        load_efficiency_table for the file contract) to the SCALAR
        (TEA) efficiency of the same zone (tea_efficiency_table).

        The ratio -- not the absolute eta -- is required because the
        phasor sum over the rings of a zone already computes the
        scalar zone response: multiplying by absolute eta would count
        the scalar diffraction loss twice. relative_correction_table()
        builds the ratio from the two tables with sane clipping.

        corr has shape (len(lam_um), len(r_um)); it is interpolated
        onto (self.lam, self.rho) with edge clamping, then
        G *= sqrt(corr). The table is stored, survives
        set_wavelengths(), and reset_efficiency() reverts. Calling
        apply_efficiency again REPLACES the previous table (no
        stacking). The correction is quasi-static (local-linear-
        grating assumption: the outer-zone period is pinned by the
        geometry, not the fine design) -- regenerate it after major
        design changes.
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

    def reset_efficiency(self):
        """Remove the efficiency correction (bare TEA phasor model)."""
        if hasattr(self, "_G_raw"):
            self.G = self._G_raw
            del self._G_raw
        if hasattr(self, "_eta_table"):
            del self._eta_table
        if self.objective == "overlap":
            self.enable_overlap_fom(*self._overlap_cfg)
        return self

    def _reapply_eta(self):
        lam_t, r_t, eta_t = self._eta_table
        # interp along wavelength for each table radius, then radius
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


# ---------------------------------------------------------------------------
# Upper bound of J_w(F)  (paper Eq. S14-S15)
# ---------------------------------------------------------------------------
def pairwise_bound_matrix(diameter_um, na, lam_min_um, lam_max_um,
                          h_max_um, dh_um, n_rho=256, n_wavelengths=512,
                          n_func=n_az4562, n_candidates=4):
    """
    Pairwise coherence map  B[i, j] = max_dm Re < J_w(rho_i, rho_j) >_w
    (the paper's Fig. 1b/1c quantity, Eq. S14): for each ring pair the
    band-averaged mutual coherence maximized over the height difference
    dm in gray levels. Returns (rho_norm, B) with rho_norm = rho/R in
    (0, 1] and B of shape (n_rho, n_rho), values in [0, 1] (pairs whose
    group-delay residual exceeds the cut are ~0 and reported as 0).

    upper_bound_jf() below contracts this matrix with the paraxial
    aperture weights to get the scalar ceiling max J_w(F) (Eq. S15).
    Numerics identical to the original upper_bound_jf (candidate-anchor
    method, brute-force validated to <= 3%; see validate_bound.py).

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
    return rho / R, B


def upper_bound_jf(diameter_um, na, lam_min_um, lam_max_um,
                   h_max_um, dh_um, n_rho=256, n_wavelengths=512,
                   n_func=n_az4562, n_candidates=4):
    """
    max J_w(F) upper bound (Eq. S15): the pairwise coherence matrix of
    pairwise_bound_matrix() contracted with the paraxial aperture
    weights w(rho) = (2F/R^2) rho drho / r  (obliquity F/r included).
    See pairwise_bound_matrix for the method and validation notes.
    """
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
                    shrink=0.5, grow=1.1, log_list=None,
                    softmin_beta_final=None):
    """
    Adaptive-step gradient ascent on continuous heights, then rounding.

    softmin_beta_final (only used when prob.fom_mode == "softmin"):
    anneal prob.softmin_beta geometrically from its current value to
    this value over the iterations -- the standard minimax continuation
    scheme (start soft/mean-like, end nearly worst-case). The FOM value
    changes meaning as beta moves, so the accept/reject reference f is
    recomputed at every beta update; prob.softmin_beta is left at the
    final value on return (recorded by the caller's config).
    """
    h = m_vec.astype(float) * prob.dh
    f = prob.fom_h(h)
    step = step0 if step0 is not None else 0.05 * prob.dh * prob.M
    anneal = (softmin_beta_final is not None
              and prob.fom_mode == "softmin"
              and softmin_beta_final != prob.softmin_beta)
    if anneal:
        b0 = float(prob.softmin_beta)
        ratio = (float(softmin_beta_final) / b0) ** (1.0 / max(iters - 1,
                                                               1))
    for it in range(iters):
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


# ---------------------------------------------------------------------------
# Echelle-style fold-height selection for COMB designs
# ---------------------------------------------------------------------------
# A comb MDL operates like an echelle grating: each target line lam_j is
# served by an integer diffraction order m_j of the fold, and the fold
# height H_f is "blazed" for line j when
#
#     alpha_j = (n(lam_j) - 1) H_f / lam_j       is an integer  (= m_j).
#
# The scalar efficiency of a perfect sawtooth of optical depth
# (n-1) H_f at lam_j, into its nearest order, is sinc^2(alpha_j - m_j)
# (multi-order / harmonic diffractive lens theory: Faklis & Morris,
# Appl. Opt. 34, 2462 (1995); Sweeney & Sommargren, Appl. Opt. 34,
# 2469 (1995)). Scanning H_f and scoring the GEOMETRIC mean of these
# sinc^2 values over the target lines picks the fold that serves ALL
# lines at once -- the echelle designer's blaze chart, applied to the
# fold height. The result seeds the comb optimizer analytically
# instead of the heuristic single-lam0 harmonic seed.
def echelle_fold_scan(target_lams_um, h_min_um, h_max_um,
                      n_func=n_az4562, n_scan=20001, weights=None,
                      verbose=None):
    """
    Scan fold heights in [h_min, h_max] and score each by the geometric
    mean of sinc^2(alpha_j - round(alpha_j)) over the target lines.

    Returns dict with:
      h_fold_um : best fold height
      score     : its geometric-mean sinc^2 (1.0 = all lines exactly
                  blazed)
      orders    : m_j = round(alpha_j) at the best fold (the "order
                  ladder"; compare the Zemax order-decomposition DLL)
      detune    : alpha_j - m_j per line
      eff_line  : sinc^2 detune per line (scalar sawtooth efficiency)
      scan      : (h_grid, score_grid) for plotting the blaze chart
    weights: optional per-line weights (default uniform) applied in the
    log domain.
    """
    lams = np.asarray(target_lams_um, dtype=float)
    w = (np.full(lams.size, 1.0 / lams.size) if weights is None
         else np.asarray(weights, float) / np.sum(weights))
    hg = np.linspace(h_min_um, h_max_um, int(n_scan))
    n = n_func(lams)
    alpha = (n - 1.0)[None, :] * hg[:, None] / lams[None, :]  # (H, J)
    det = alpha - np.rint(alpha)
    eff = np.sinc(det) ** 2                                   # np.sinc(x)=sin(pi x)/(pi x)
    score = np.exp(np.sum(w[None, :] * np.log(eff + 1e-12), axis=1))
    ib = int(np.argmax(score))
    out = dict(h_fold_um=float(hg[ib]), score=float(score[ib]),
               orders=np.rint(alpha[ib]).astype(int),
               detune=det[ib], eff_line=eff[ib],
               scan=(hg, score))
    if verbose:
        verbose("echelle fold scan: best H_fold = %.4f um "
                "(geomean sinc^2 = %.3f)" % (out["h_fold_um"],
                                             out["score"]))
        verbose("  line[um]  order m  detune   sinc^2")
        for lam, m_j, d, e in zip(lams, out["orders"], out["detune"],
                                  out["eff_line"]):
            verbose("  %8.4f  %7d  %+.3f   %.3f" % (lam, m_j, d, e))
    return out


def _fold_seed(prob: MDLProblem, h_fold_um, n_ref):
    """Quantized hyperbolic-OPD seed folded at (n_ref-1)*h_fold."""
    opl_fold = (n_ref - 1.0) * h_fold_um            # fold period in OPD
    r = np.sqrt(prob.rho ** 2 + prob.F ** 2)
    opd = r - prob.F
    h = ((-opd) % opl_fold) / (n_ref - 1.0)
    return np.clip(np.rint(h / prob.dh), 0, prob.M).astype(np.int32)


def seed_echelle(prob: MDLProblem, target_lams_um, h_fold_um=None,
                 h_min_frac=0.5, n_candidates=16, verbose=None):
    """
    Analytic comb seed: hyperbolic OPD folded at an echelle-selected
    fold height,

        h(rho) = ( -OPD(rho) mod (n_ref - 1) H_f ) / (n_ref - 1)

    with n_ref the band-center index (dispersion breaks exact
    congruence anyway; the optimizer absorbs the residual).

    Fold selection is CANDIDATE-BASED (same philosophy as the
    candidate-anchor bound code): the sinc^2 blaze chart from
    echelle_fold_scan is only a proxy -- it ranks folds by sawtooth
    congruence but ignores quantization, dispersion across the fold
    and the actual FOM aggregation. Candidates come from two families:
    (a) every harmonic anchor h = p lam_j/(n_j-1) with lam_j a target
    line (the full generalization of the seed_lam0_um heuristic --
    all are cheap enough to judge exhaustively), and (b) the
    n_candidates best distinct local maxima of the blaze chart
    (compromise folds congruent at no single line). Every candidate is
    built into a seed and scored with the real prob.fom() on the
    problem's CURRENT wavelength grid / fom_mode (call
    set_wavelengths + set fom_mode first); the winner is returned.
    Pass h_fold_um to skip selection entirely.

    Returns (m_seed, info): info carries h_fold_um, the order ladder
    m_j, per-line detune and sinc^2 for the WINNING fold, the full
    scan, and 'candidates' [(h_fold, blaze_score, J_seed), ...].
    """
    lams = np.asarray(target_lams_um, dtype=float)
    lam_mid = 0.5 * (lams.min() + lams.max())
    n_ref = float(prob.n_func(lam_mid))

    def line_info(hf):
        n = prob.n_func(lams)
        alpha = (n - 1.0) * hf / lams
        det = alpha - np.rint(alpha)
        return (np.rint(alpha).astype(int), det, np.sinc(det) ** 2)

    if h_fold_um is not None:
        m = _fold_seed(prob, h_fold_um, n_ref)
        orders, det, eff = line_info(h_fold_um)
        return m, dict(h_fold_um=float(h_fold_um),
                       score=float(np.exp(np.mean(np.log(eff + 1e-12)))),
                       orders=orders, detune=det, eff_line=eff,
                       scan=None, candidates=None)

    scan = echelle_fold_scan(lams, h_min_frac * prob.h_max, prob.h_max,
                             n_func=prob.n_func, verbose=None)
    hg, sc = scan["scan"]
    # candidate folds, two families:
    # (a) the HARMONIC family: fold exactly congruent at anchor line
    #     lam_j in order p, h = p lam_j / (n_j - 1) -- every target
    #     line acts as an anchor (generalizes the seed_lam0_um loop);
    # (b) local maxima of the blaze chart (compromise folds congruent
    #     at no single line but decent at all).
    # All are scored analytically by the blaze geomean sinc^2, and only
    # the n_candidates best are built and judged by the real prob.fom.
    n_all = prob.n_func(lams)

    def blaze_score(hf):
        det = (n_all - 1.0) * hf / lams
        det = det - np.rint(det)
        return float(np.exp(np.mean(np.log(np.sinc(det) ** 2 + 1e-12))))

    # family (a) is small (n_lines x ~n_orders) and a seed fom() eval
    # is O(Nw N) ~ milliseconds, so ALL harmonic anchors are judged by
    # the real fom; the blaze ranking only prunes family (b), whose
    # chart maxima come in dense near-duplicate clusters.
    fam = []                                     # (hf, n_ref_for_build)
    for lam_j, n_j in zip(lams, n_all):
        step = lam_j / (n_j - 1.0)
        for p in range(max(1, int(np.ceil(h_min_frac * prob.h_max
                                          / step))),
                       int(np.floor(prob.h_max / step)) + 1):
            fam.append((p * step, n_j))
    imax = np.where((sc[1:-1] > sc[:-2]) & (sc[1:-1] >= sc[2:]))[0] + 1
    scanc = sorted([(float(hg[i]), n_ref) for i in imax],
                   key=lambda c: -blaze_score(c[0]))
    # drop scan folds within 5 nm of a harmonic anchor (duplicates)
    anchors = np.array([hf for hf, _ in fam])
    scanc = [(hf, nr) for hf, nr in scanc
             if np.min(np.abs(anchors - hf)) > 5e-3]
    cands = []
    best = None
    for hf, nr in fam + scanc[:int(n_candidates)]:
        m = _fold_seed(prob, hf, nr)
        J = prob.fom(m)
        cands.append((float(hf), blaze_score(hf), float(J)))
        if best is None or J > best[0]:
            best = (J, float(hf), m)
    J_best, hf_best, m_best = best
    orders, det, eff = line_info(hf_best)
    if verbose:
        verbose("echelle seed: %d candidate folds, winner "
                "H_fold = %.4f um (seed J = %.4g, fom_mode=%s)"
                % (len(cands), hf_best, J_best, prob.fom_mode))
        verbose("  line[um]  order m  detune   sinc^2")
        for lam, m_j, d, e in zip(lams, orders, det, eff):
            verbose("  %8.4f  %7d  %+.3f   %.3f" % (lam, m_j, d, e))
    return m_best, dict(h_fold_um=hf_best, score=float(
        np.exp(np.mean(np.log(eff + 1e-12)))), orders=orders,
        detune=det, eff_line=eff, scan=scan["scan"],
        candidates=sorted(cands, key=lambda c: -c[2]))


# ---------------------------------------------------------------------------
# Local linear-grating decomposition + rigorous-efficiency correction
# ---------------------------------------------------------------------------
# An MDL is a circularly chirped grating: at radius rho the zone
# structure is locally a linear grating of period Lambda(rho), and
# focusing is the local grating equation steering order m to F. The
# thin-element approximation (TEA) behind the phasor tables ignores
# how efficiently each local grating actually diffracts -- our BPM
# audit showed TEA overestimates balanced-design efficiency by ~20 %
# in the outer zones where Lambda -> a few wavelengths. The grating
# community's fix is the local linear grating approximation: compute
# each zone's order efficiencies RIGOROUSLY (RCWA) and fold them back
# in as per-ring amplitude weights.
#
# The workflow here is deliberately FILE-DECOUPLED (files-as-interface,
# like the rest of the package):
#
#   1. extract_local_gratings(prob, m)  -> zone list (period, profile)
#   2. write_zone_table(path, zones)    -> npz handed to ANY rigorous
#      solver: Ansys Lumerical RCWA via its Python API, torcwa (Kim &
#      Lee, Comput. Phys. Commun. 282, 108552 (2023)), grcwa (Jin et
#      al., Phys. Rev. B 101, 245418 (2020)), RETICOLO, ...
#      The solver sweeps each zone profile over the design wavelengths
#      and writes an efficiency table (see load_efficiency_table for
#      the exact npz contract).
#   3. tea_efficiency_table(prob, m)    -> matching SCALAR baseline
#   4. relative_correction_table(...)   -> corr = eta_rcwa / eta_tea
#   5. prob.apply_efficiency(lam, r, corr) -> G scaled by sqrt(corr);
#      every FOM/gradient/search then optimizes the corrected model.
#      reset_efficiency() reverts.
#
# The RATIO in step 4 is essential: the phasor sum over a zone's rings
# already computes the scalar zone response, so only the rigorous-
# minus-scalar DIFFERENCE may be injected -- applying absolute eta
# would count the scalar diffraction loss twice (verified: it
# collapses J by ~15x on the S3 comb design; the ratio leaves J
# untouched when eta_rcwa == eta_tea).
def extract_local_gratings(prob: MDLProblem, m_vec, reset_frac=0.4,
                           min_zone_rings=2):
    """
    Split the design into sawtooth zones. A zone boundary is detected
    where the height jumps UP by more than reset_frac * h_max between
    adjacent rings (the fold reset of the local blaze). Zones narrower
    than min_zone_rings rings are merged forward.

    Returns a list of dicts, one per zone:
      i0, i1        : ring index range [i0, i1)
      r_in, r_out   : zone radii [um]
      r_center      : center radius [um]
      period_um     : zone width = local grating period Lambda
      h_um          : height profile across the zone (one sample per
                      ring, inner to outer) [um]
      m_local       : round(Lambda * r_center / (lam * r_bar)) is the
                      caller's job per lam; here we store the OPD span
      opd_span_um   : geometric OPD change across the zone
                      (sqrt(r_out^2+F^2) - sqrt(r_in^2+F^2)) -- its
                      ratio to lam is the local order alpha_loc(lam).
    Inner rings before the first reset form zone 0 (quasi-flat paraxial
    core; its 'period' is not a meaningful grating period -- rigorous
    correction matters in the OUTER zones).
    """
    if m_vec.size != prob.N:
        raise ValueError("design vector has %d rings but the problem "
                         "has N=%d -- geometry mismatch" % (m_vec.size,
                                                            prob.N))
    h = m_vec.astype(float) * prob.dh
    jumps = np.where(np.diff(h) > reset_frac * prob.h_max)[0] + 1
    bounds = [0] + [int(j) for j in jumps] + [prob.N]
    # merge too-narrow zones forward
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < min_zone_rings and b != prob.N:
            continue
        merged.append(b)
    zones = []
    for i0, i1 in zip(merged[:-1], merged[1:]):
        r_in = i0 * prob.delta
        r_out = i1 * prob.delta
        rc = 0.5 * (r_in + r_out)
        opd = (np.sqrt(r_out ** 2 + prob.F ** 2)
               - np.sqrt(r_in ** 2 + prob.F ** 2))
        zones.append(dict(i0=int(i0), i1=int(i1), r_in=r_in,
                          r_out=r_out, r_center=rc,
                          period_um=r_out - r_in,
                          h_um=h[i0:i1].copy(),
                          opd_span_um=float(opd)))
    return zones


def write_zone_table(path, zones, lams_um, prob: MDLProblem = None):
    """
    Save the zone decomposition for an external rigorous solver.

    npz contents:
      zone_id (Z,), r_in/r_out/r_center/period_um (Z,),
      h_profile: object array of per-zone height samples [um]
        (one per ring, sample pitch = ring width),
      ring_width_um, lams_um (K,): wavelengths to sweep,
      n_real (K,): material index at those wavelengths (the solver
        still needs its own k/absorption model if any).

    The solver's job per zone z and wavelength lam: build the sawtooth
    profile h_profile[z] on a period period_um[z], incidence normal
    from the substrate side, and return the diffraction efficiency of
    the FOCUSING order -- the order whose deflection angle matches
    sin(theta) = r_center / sqrt(r_center^2 + F^2); with Lumerical's
    RCWA this is one 'grating characterization' sweep per zone. Write
    the result with axes (lams_um, r_center) -- see
    load_efficiency_table.
    """
    z = zones
    np.savez(path,
             zone_id=np.arange(len(z)),
             r_in=np.array([q["r_in"] for q in z]),
             r_out=np.array([q["r_out"] for q in z]),
             r_center=np.array([q["r_center"] for q in z]),
             period_um=np.array([q["period_um"] for q in z]),
             h_profile=np.array([q["h_um"] for q in z], dtype=object),
             ring_width_um=(z[0]["h_um"].size and
                            (z[0]["r_out"] - z[0]["r_in"])
                            / z[0]["h_um"].size),
             lams_um=np.asarray(lams_um, float),
             n_real=(prob.n_func(np.asarray(lams_um, float))
                     if prob is not None else
                     n_az4562(np.asarray(lams_um, float))),
             allow_pickle=True)
    return path


def load_efficiency_table(path):
    """
    Load an efficiency table written by the external solver (or by
    tea_efficiency_table). npz contract -- exactly three arrays:

      lam_um : (K,)   wavelengths [um]
      r_um   : (Z,)   radial positions (zone centers) [um]
      eta    : (K, Z) efficiency into the focusing order in [0, 1]
               for a raw solver table, or the rigorous/scalar ratio
               (may exceed 1) for a correction file destined for
               apply_efficiency

    Returns (lam_um, r_um, eta).
    """
    d = np.load(path)
    return d["lam_um"], d["r_um"], d["eta"]


def tea_zone_efficiency(zone, lams_um, F_um, n_func=n_az4562):
    """
    Scalar (TEA) reference efficiency of one zone into its focusing
    order: the squared modulus of the Fourier coefficient of the zone's
    complex transmission at the local grating frequency,

        eta(lam) = | (1/Lambda) int_zone exp{ i [ k (n-1) h(x)
                     - k x sin(theta_c) ] } dx |^2,

    sin(theta_c) = r_c / sqrt(r_c^2 + F^2) the deflection that sends
    light from the zone center to the focus. This is the same physics
    as the phasor model (scalar, thin), just organized per zone -- use
    it to validate the correction plumbing and as the baseline that a
    rigorous RCWA table (Lumerical / torcwa) replaces.
    """
    lams = np.asarray(lams_um, float)
    n = n_func(lams)
    k = 2.0 * np.pi / lams
    nx = zone["h_um"].size
    if nx == 0:
        return np.ones_like(lams)
    dx = zone["period_um"] / nx
    x = (np.arange(nx) + 0.5) * dx
    rc = zone["r_center"]
    s = rc / np.sqrt(rc ** 2 + F_um ** 2)
    ph = (k[:, None] * (n - 1.0)[:, None] * zone["h_um"][None, :]
          - k[:, None] * s * x[None, :])
    c = np.mean(np.exp(1j * ph), axis=1)
    return np.abs(c) ** 2


def tea_efficiency_table(prob: MDLProblem, m_vec, lams_um,
                         reset_frac=0.4):
    """Full (lam, r) TEA table over all zones -- reference/baseline."""
    zones = extract_local_gratings(prob, m_vec, reset_frac=reset_frac)
    lams = np.asarray(lams_um, float)
    r_um = np.array([z["r_center"] for z in zones])
    eta = np.stack([tea_zone_efficiency(z, lams, prob.F, prob.n_func)
                    for z in zones], axis=1)          # (K, Z)
    return lams, r_um, eta


def relative_correction_table(eta_rigorous, eta_scalar, eta_floor=0.02,
                              corr_max=2.0):
    """
    corr = eta_rigorous / eta_scalar, the table apply_efficiency()
    expects. Zones where the SCALAR efficiency is below eta_floor are
    set to corr = 1 (the ratio of two near-zeros is noise, and such
    zones barely contribute to the field anyway); the ratio is clipped
    to [0, corr_max] to keep one bad RCWA point from dominating.
    Both inputs shape (K, Z) on identical (lam, r) axes.
    """
    er = np.asarray(eta_rigorous, float)
    es = np.asarray(eta_scalar, float)
    corr = np.where(es > eta_floor, er / np.maximum(es, 1e-12), 1.0)
    return np.clip(corr, 0.0, corr_max)