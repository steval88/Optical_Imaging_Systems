"""Validate the candidate-based upper bound and the achievable FOM."""
import time
import numpy as np

from aMDL.old.mdl_design_package.mdl_core import (MDLProblem, upper_bound_jf, seed_harmonic,
                      seed_max_harmonic, hooke_jeeves)

C = 2.99792458e14


def brute_bound(D, na, l1, l2, H, dh, n_rho=200, Nw=4001):
    """Full traversal over all dm on a dense (alias-free) w grid."""
    R = 0.5 * D
    F = R * np.sqrt(1 / na ** 2 - 1)
    M = int(round(H / dh))
    om = np.linspace(2 * np.pi * C / l2, 2 * np.pi * C / l1, Nw)
    lam = 2 * np.pi * C / om
    k = 2 * np.pi / lam
    n = 1.594 + 0.01152 / lam ** 2
    kn = k * (n - 1)
    rho = (np.arange(n_rho) + 0.5) * (R / n_rho)
    r = np.sqrt(rho ** 2 + F ** 2)
    p, q = np.triu_indices(n_rho)
    L = r[p] - r[q]
    # phasor accumulation: mean_w exp(i(kn*dh*dm + k*L)) for all dm
    # do it as outer products in chunks of dm
    best = np.full(L.size, -np.inf)
    dm = np.arange(-M, M + 1)
    PL = np.exp(1j * np.multiply.outer(L, k))          # (Np, Nw)
    for s in range(0, dm.size, 16):
        dmb = dm[s:s + 16]
        A = np.exp(1j * np.multiply.outer(kn * dh, dmb))   # (Nw, ndm)
        vals = np.real(PL @ A) / Nw                        # (Np, ndm)
        best = np.maximum(best, vals.max(axis=1))
    B = np.empty((n_rho, n_rho))
    B[p, q] = best
    B[q, p] = best
    wgt = (2 * F / R ** 2) * rho * (R / n_rho) / r
    wgt /= wgt.sum()
    return float(wgt @ B @ wgt)


# ---- case A: D=1mm NA=0.3 400-1100 H=10 --------------------------------
t0 = time.time()
bb = brute_bound(1000, 0.3, 0.40, 1.10, 10.0, 0.078)
t1 = time.time()
bc = upper_bound_jf(1000, 0.3, 0.40, 1.10, 10.0, 0.078,
                    n_rho=200, n_wavelengths=1024, n_candidates=6)
t2 = time.time()
print("case A  brute=%.4f (%.0fs)   candidate=%.4f (%.0fs)"
      % (bb, t1 - t0, bc, t2 - t1))

# ---- case B: paper S1 (D=1.024mm NA=0.1 400-1100 H=15) -----------------
bb = brute_bound(1024, 0.1, 0.40, 1.10, 15.0, 0.078)
bc = upper_bound_jf(1024, 0.1, 0.40, 1.10, 15.0, 0.078,
                    n_rho=200, n_wavelengths=1024, n_candidates=6)
print("case B (paper S1, theirs 0.72)  brute=%.4f  candidate=%.4f" % (bb, bc))

# ---- achievability on S1: dense-FOM optimization ------------------------
prob = MDLProblem(diameter_um=1024, na=0.1, lam_min_um=0.40, lam_max_um=1.10,
                  ring_width_um=2.0, h_max_um=15.0, dh_um=0.078,
                  n_wavelengths=257)
print("S1 problem: N=%d rings, M=%d levels, F=%.2f mm"
      % (prob.N, prob.M, prob.F / 1000))
# dense-check helper
dense = MDLProblem(diameter_um=1024, na=0.1, lam_min_um=0.40,
                   lam_max_um=1.10, ring_width_um=2.0, h_max_um=15.0,
                   dh_um=0.078, n_wavelengths=2001)

best = (None, -1, None)
for lam0 in (0.50, 0.55, 0.60):
    for p in (1, 5, 10, 17):
        m = seed_harmonic(prob, lam0, p)
        f = prob.fom(m)
        if f > best[1]:
            best = (m, f, (lam0, p))
print("best harmonic seed: lam0=%.2f p=%d  J(257)=%.4f  J(2001)=%.4f"
      % (best[2][0], best[2][1], best[1], dense.fom(best[0])))

t0 = time.time()
m_opt, f_opt = hooke_jeeves(prob, best[0], max_sweeps=120)
print("after HJA: J(257)=%.4f  J(2001)=%.4f  (%.0fs)"
      % (f_opt, dense.fom(m_opt), time.time() - t0))
