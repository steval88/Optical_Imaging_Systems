"""Regression of the mdl package against the frozen mdl_core.py at the
package root: tables, objectives, gradients, delta updates, seeds,
bounds, zones and the optimizer trajectories with the same RNG must be
BIT-IDENTICAL. Run from anywhere:

    python 01_design_oo\\tests\\test_against_mdl_core.py

Takes ~30 s (the small optimizer replays). Prints ALL OK or FAILED.
The full-pipeline replay (S3 softmin config, rng_seed 7) was done on
2026-09-16: m_final identical, J stages 0.0340 / 0.1680 / 0.1505 /
0.1581 / 0.1601 identical to 10 decimals on both drivers.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))          # the package root
# the frozen mdl_core.py: at this package root, or in the frozen
# mdl_design_package next to an OOP package, or under MDL_LEGACY_ROOT
for _cand in ([os.environ["MDL_LEGACY_ROOT"]] if os.environ.get("MDL_LEGACY_ROOT") else []) + \
        [PKG, os.path.join(os.path.dirname(PKG), "mdl_design_package")]:
    if os.path.exists(os.path.join(_cand, "mdl_core.py")):
        sys.path.insert(0, _cand)
        break
else:
    raise SystemExit("mdl_core.py not found next to this package nor in ../mdl_design_package; "
                     "set MDL_LEGACY_ROOT")
sys.path.insert(0, os.path.dirname(HERE))
import mdl_core as old                                 # the frozen original
import mdl
from mdl import compat

# the S3 geometry and the paper's 14-line comb
D, F = 10240.0, 50940.0; R = D / 2; na = R / np.sqrt(R * R + F * F)
lams = np.asarray(mdl.PAPER_COMB_14)

def build(mod, quad, overlap=True, single=True, mode="softmin", beta=300.0):
    p = mod.MDLProblem(D, na, 0.4, 1.1, 2.0, 15.0, 0.078, n_wavelengths=14, ring_quadrature=quad)
    p.set_wavelengths(lams); p.fom_mode = mode; p.softmin_beta = beta
    if single:
        p.G = p.G.astype(np.complex64); p.L = p.L.astype(np.complex64)
    if overlap:
        p.enable_overlap_fom(None, 24, 2.0)
    return p

def same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))

ok = True
for quad in ("midpoint", "sinc"):
    po, pn = build(old, quad), build(mdl, quad)
    for name in ("G", "L", "S", "K", "_enc_D", "_enc_w", "rho", "lam", "k", "n"):
        if not same(getattr(po, name), getattr(pn, name)):
            print("DIFF", quad, name); ok = False
    m, info = compat.seed_echelle(pn, lams)
    m_old, info_old = old.seed_echelle(po, lams)
    print("%s: seed identical %s (H_fold %.4f), J old %.6f new %.6f" % (
        quad, same(m, m_old), info["h_fold_um"], po.fom(m_old), pn.fom(m)))
    ok &= same(m, m_old) and po.fom(m_old) == pn.fom(m)
    h = m * 0.078 + 0.01
    ok &= same(po.grad_h(h), pn.grad_h(h)) and po.fom_h(h) == pn.fom_h(h)
    for mode in ("mean", "geomean"):
        po.fom_mode = pn.fom_mode = mode
        ok &= po.fom(m) == pn.fom(m) and same(po.grad_h(h), pn.grad_h(h))
    po.fom_mode = pn.fom_mode = "softmin"
    # onaxis objective
    po.disable_overlap_fom(); pn.disable_overlap_fom()
    ok &= po.fom(m) == pn.fom(m) and same(po.grad_h(h), pn.grad_h(h)) and same(po.field(m), pn.field(m))
    U = pn.field(m); ok &= same(po.delta_field(U, 100, m[100], 5), pn.delta_field(U, 100, m[100], 5))
print("tables / objectives / gradients / delta / seed:", "OK" if ok else "FAILED")

# --- optimizers, small settings, same rng seed -----------------------------
po, pn = build(old, "sinc"), build(mdl, "sinc")
po.softmin_beta = pn.softmin_beta = 20.0
m0, _ = old.seed_echelle(po, lams)
t0 = time.time()
ro = old.search(po, m_init=m0, blocks=2, ga_epochs=3, pop_size=6, rng=np.random.default_rng(7), verbose=lambda s: None)
rn = compat.search(pn, m_init=m0, blocks=2, ga_epochs=3, pop_size=6, rng=np.random.default_rng(7), verbose=lambda s: None)
print("search (2 blocks x 3 epochs x pop 6): identical m %s, J %.6f / %.6f  (%.1fs)" % (same(ro[0], rn[0]), ro[1], rn[1], time.time() - t0))
ok &= same(ro[0], rn[0]) and ro[1] == rn[1]
so, sn = old.smooth(po, ro[0]), compat.smooth(pn, rn[0])
ok &= same(so, sn); print("smooth identical", same(so, sn))
go = old.gradient_refine(po, so, iters=5, softmin_beta_final=300.0)
gn = compat.gradient_refine(pn, sn, iters=5, softmin_beta_final=300.0)
ok &= same(go[0], gn[0]) and go[1] == gn[1] and po.softmin_beta == pn.softmin_beta
print("gradient identical", same(go[0], gn[0]), go[1], gn[1], "beta", po.softmin_beta, pn.softmin_beta)
ho = old.hooke_jeeves(po, go[0], d0=2, max_sweeps=3)
hn = compat.hooke_jeeves(pn, gn[0], d0=2, max_sweeps=3)
ok &= same(ho[0], hn[0]) and ho[1] == hn[1]; print("hooke_jeeves identical", same(ho[0], hn[0]), ho[1], hn[1])
# verbatim combo (binary GA + verbatim HJA), tiny
vo = old.multistep_GA_HJA_combo(po, m_init=m0, s=1, p=2, pop_size=4, rng=np.random.default_rng(3), verbose=lambda s: None)
vn = compat.multistep_GA_HJA_combo(pn, m_init=m0, s=1, p=2, pop_size=4, rng=np.random.default_rng(3), verbose=lambda s: None)
ok &= same(vo[0], vn[0]) and vo[1] == vn[1]; print("verbatim combo identical", same(vo[0], vn[0]))
# harmonic seeds
ok &= same(old.seed_harmonic(po, 0.6, 15), mdl.harmonic_seed(pn, 0.6, 15)) and same(old.seed_max_harmonic(po, 0.55), mdl.max_harmonic_seed(pn, 0.55))
# bounds (reduced) and zones
bo = old.upper_bound_jf(D, na, 0.4, 1.1, 15.0, 0.078, n_rho=64, n_wavelengths=128)
bn = mdl.upper_bound_jf(D, na, 0.4, 1.1, 15.0, 0.078, n_rho=64, n_wavelengths=128)
ok &= bo == bn; print("bound identical", bo == bn, bo)
zo, zn = old.extract_local_gratings(po, ho[0]), mdl.extract_local_gratings(pn, hn[0])
ok &= len(zo) == len(zn) and all(same(a["h_um"], b["h_um"]) and a["period_um"] == b["period_um"] for a, b in zip(zo, zn))
to, tn = old.tea_efficiency_table(po, ho[0], lams), mdl.tea_efficiency_table(pn, hn[0], lams)
ok &= same(to[2], tn[2]); print("zones + TEA table identical", same(to[2], tn[2]), len(zn), "zones")
print("ALL OK" if ok else "FAILED")
