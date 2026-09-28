"""The solver against what is known exactly, and against the OpticStudio
srg DLLs where they were measured (NSC_TRACK.md). Run from the package root:

    python 02_RCWA_Validation_OOP\\tests\\test_rcwa1d.py        (~1 min)

1. Fresnel: a flat resist -> air interface transmits 4 n / (n + 1)^2.
2. A thin film in air reproduces the Airy (thin-film) reflectance.
3. Energy: sum R + T = 1 to 1e-8 on every structure below.
4. A subwavelength lamellar grating is the effective film of the
   Rytov limits (TE <eps>, TM <1/eps>^-1) to 1 %.
5. Two independent 1-D solvers agree: the 5-level staircase of the null test
   (P 5 um, 600 nm, n 1.632, free-standing) gave TE 0.7721 / TM 0.8036 with
   grcwa 0.1.2 (2026-09-21); this solver must reproduce them to 0.5 %.
6. srg_step ladder lines (2026-09-18 17:08, free-standing relief, which is
   what the srg DLLs compute for an object in air): the short staircases
   of the S3 ladder, |ratio - 1| <= 5 % (the DLL's own staircase rule and
   harmonic count are not ours; 4.4 % worst on 2026-09-21).
7. Order-sign convention: a height DEscending with x feeds m > 0 in this
   solver and m < 0 in nscval.tea.profile_orders.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.dirname(HERE)
PKG = os.path.dirname(STAGE)
sys.path.insert(0, STAGE)
sys.path.insert(0, os.path.join(PKG, "02_NonSequential_RT_Zemax_Validation_OOP"))
sys.path.insert(0, os.path.join(PKG, "01_design_oo"))

from rcwaval.rcwa1d import Grating1D, Layer, harmonics_for, solve      # noqa: E402
from rcwaval.profiles import flat, staircase                            # noqa: E402


def check(name: str, ok: bool, detail: str) -> bool:
    print("  %-52s %s  %s" % (name, "OK  " if ok else "FAIL", detail))
    return ok


def main() -> bool:
    t0 = time.time()
    ok = True
    n = 1.632
    # 1. Fresnel
    e = solve(flat(0.0, n, n, 1.0), 0.6, 5)
    T = 4 * n / (n + 1) ** 2
    ok &= check("Fresnel resist -> air, TE and TM", abs(e.T_te[5] - T) < 1e-9 and abs(e.T_tm[5] - T) < 1e-9,
                "%.6f %.6f vs %.6f" % (e.T_te[5], e.T_tm[5], T))
    # 2. thin film
    nf, d, lam = 1.3, 0.5, 0.6
    e = solve(flat(d, nf, 1.0, 1.0), lam, 3)
    r12 = (1 - nf) / (1 + nf)
    delta = 2 * np.pi * nf * d / lam
    r = (r12 * (1 - np.exp(2j * delta))) / (1 - r12 ** 2 * np.exp(2j * delta))
    ok &= check("thin film reflectance", abs(e.R_te[3] - abs(r) ** 2) < 1e-6, "%.6f vs %.6f" % (e.R_te[3], abs(r) ** 2))
    # 4. EMT
    P, f, dd = 0.03, 0.5, 0.5
    g = Grating1D(P, 1.0, 1.0, [Layer(dd, [(0.0, f * P, n), (f * P, P, 1.0)])])
    e = solve(g, lam, 10)
    ok &= check("energy balance (subwavelength lamellar)", all(abs(b - 1) < 1e-8 for b in e.balance()), "%s" % (e.balance(),))
    eps_te = f * n ** 2 + (1 - f)
    eps_tm = 1.0 / (f / n ** 2 + (1 - f))

    def film_r(nfilm: float) -> float:
        r12 = (1 - nfilm) / (1 + nfilm)
        de = 2 * np.pi * nfilm * dd / lam
        rr = (r12 * (1 - np.exp(2j * de))) / (1 - r12 ** 2 * np.exp(2j * de))
        return abs(rr) ** 2
    ok &= check("EMT TE (<eps>)", abs(e.R_te[10] - film_r(np.sqrt(eps_te))) < 0.01 * film_r(np.sqrt(eps_te)) + 1e-4,
                "%.5f vs %.5f" % (e.R_te[10], film_r(np.sqrt(eps_te))))
    ok &= check("EMT TM (<1/eps>^-1)", abs(e.R_tm[10] - film_r(np.sqrt(eps_tm))) < 5e-4,
                "%.5f vs %.5f" % (e.R_tm[10], film_r(np.sqrt(eps_tm))))
    # 5. the null-test staircase, free-standing, vs grcwa
    dblaze = 0.6 / 0.632
    h = (dblaze * np.arange(5) / 5.0)[::-1]
    e = solve(staircase(h, 1.0, n, 1.0, 1.0), 0.6, 30)
    ok &= check("energy balance (5-level staircase)", all(abs(b - 1) < 1e-8 for b in e.balance()), "%s" % (e.balance(),))
    ok &= check("null staircase vs grcwa TE 0.7721", abs(e.order(1, "te") - 0.7721) < 0.004, "%.4f" % e.order(1, "te"))
    ok &= check("null staircase vs grcwa TM 0.8036", abs(e.order(1, "tm") - 0.8036) < 0.004, "%.4f" % e.order(1, "tm"))
    # 7. sign convention
    from nscval import tea
    tea_eta = tea.profile_orders(2 * np.pi * 0.632 * h / 0.6, [-1, 1])
    ok &= check("descending profile: tea feeds m = -1, solver m = +1",
                tea_eta[0] > 0.8 and e.order(1) > 0.7 and e.order(-1) < 0.02,
                "tea(-1) %.3f solver(+1) %.3f solver(-1) %.4f" % (tea_eta[0], e.order(1), e.order(-1)))
    # 6. srg_step ladder lines (free-standing), design-order efficiencies from the 17:08 log
    from mdl.material import n_az4562
    riser, step = 0.3254996867942404, 2.01
    lines = [(3, 0.600, 1, 0.6016), (3, 0.850, 1, 0.4536), (3, 1.100, 1, 0.3182),
             (4, 0.700, 1, 0.6399), (4, 0.850, 1, 0.6554), (4, 1.100, 1, 0.5574),
             (6, 0.850, 1, 0.4553), (6, 1.050, 1, 0.8109), (6, 1.100, 1, 0.8037)]
    worst = 0.0
    for Nt, lam_l, p0, srg in lines:
        nl = float(n_az4562(lam_l))
        hh = (riser * np.arange(Nt))[::-1]
        ee = solve(staircase(hh, step, nl, 1.0, 1.0), lam_l, harmonics_for(Nt * step, lam_l, nl))
        worst = max(worst, abs(ee.order(p0) / srg - 1.0))
    ok &= check("srg_step ladder, 9 lines, |ratio - 1| <= 5 %", worst <= 0.05, "worst %.3f" % worst)
    print("ALL OK (%.0f s)" % (time.time() - t0) if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
