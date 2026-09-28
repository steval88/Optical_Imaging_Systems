"""
efficiency_corr_check.py -- what the rigorous efficiency correction does to
an EXISTING design, separated from what the optimizer did with it.

Two runs of the same preset, one bare (scalar) and one with
efficiency_corr_npz, differ for two reasons: the tables the FOM is built
from, and the optimizer's trajectory on those tables (GA with the same
seed, different landscape). This script rebuilds the problem of the bare
run twice -- without and with the correction table of the corrected run
-- and evaluates BOTH design vectors under BOTH tables. Nothing is
optimized, nothing is written. Usage (from the package root):

    python 01_design_oo\\efficiency_corr_check.py runs\\<bare_run> runs\\<corrected_run> [--table <npz>]

--table replaces the corrected run's table by another one (e.g. the table
swept from the corrected run's OWN zone profiles): the self-consistency
check of a re-design -- its vector scored under the losses of its own
zones rather than under those of the run it started from.

Printed: the 2 x 2 table of J (the run's own fom_mode, final softmin
beta) and the per-line eta_w of each vector under each table, with the
ratios that matter:

    same vector, corrected / bare tables   = what the correction costs
                                             (the physics)
    corrected vector / bare vector, corrected tables
                                           = what the re-optimization
                                             recovered (the optimizer)

The diagonal must reproduce each run's design_metrics.json J_final to
float32 precision (the check that the rebuilt problems are the runs').

Version 2026-09-21.14 (--table added; 2026-09-21.01 first).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the mdl package
from mdl import MDLProblem, load_efficiency_table                # noqa: E402


def build(cfg, corr_npz=None):
    D = float(cfg["diameter_um"])
    R = 0.5 * D
    if cfg.get("focal_um"):
        F = float(cfg["focal_um"])
        na = R / np.sqrt(R * R + F * F)
    else:
        na = float(cfg["na"])
    lams = cfg.get("target_wavelengths_um")
    nw = len(lams) if lams is not None else int(cfg["n_wavelengths"])
    prob = MDLProblem(D, na, cfg["lam_min_um"], cfg["lam_max_um"],
                      cfg["ring_width_um"], cfg["h_max_um"], cfg["dh_um"],
                      n_wavelengths=nw, ring_quadrature=cfg.get("ring_quadrature", "midpoint"))
    if lams is not None:
        prob.set_wavelengths(np.asarray(lams, float))
    prob.fom_mode = cfg.get("fom_mode", "mean")
    if prob.fom_mode == "softmin":
        prob.softmin_beta = float(cfg.get("softmin_beta_final")
                                  or cfg.get("softmin_beta", 20.0))
    prob.use_single_precision()
    if corr_npz:
        lam_c, r_c, corr_c = load_efficiency_table(corr_npz)
        prob.apply_efficiency(lam_c, r_c, corr_c)
    if cfg.get("overlap_fom"):
        prob.enable_overlap_fom(cfg.get("overlap_r_enc_um"),
                                cfg.get("overlap_n_r0", 24),
                                cfg.get("overlap_airy_factor", 2.0))
    return prob


def load_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    m = np.load(os.path.join(run_dir, "m_final.npy")).astype(np.int64)
    dm = os.path.join(run_dir, "design_metrics.json")
    j = json.load(open(dm)).get("J_final") if os.path.exists(dm) else None
    return cfg, m, j


def resolve(path, run_dir):
    """The correction path as the corrected run recorded it, tried as given,
    then relative to the package root, then inside the run folder."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (path, os.path.join(pkg_root, path), os.path.join(run_dir, os.path.basename(path))):
        if os.path.exists(cand):
            return cand
    raise SystemExit("efficiency_corr_npz of the corrected run not found: %s" % path)


SCRIPT_VERSION = "2026-09-21.14"


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    print("efficiency_corr_check %s | %s" % (SCRIPT_VERSION, " ".join(sys.argv)))
    bare_dir, corr_dir = sys.argv[1], sys.argv[2]
    table_override = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else None
    cfg_b, m_b, j_b = load_run(bare_dir)
    cfg_c, m_c, j_c = load_run(corr_dir)
    if table_override:
        corr_npz = resolve(table_override, corr_dir)
        j_c = None                      # the run's J_final was scored on another table
        print("table override: %s (the corrected run's own J_final is not compared)" % corr_npz)
    else:
        if not cfg_c.get("efficiency_corr_npz"):
            raise SystemExit("%s has no efficiency_corr_npz in its config.json" % corr_dir)
        corr_npz = resolve(cfg_c["efficiency_corr_npz"], corr_dir)
    same = all(cfg_b.get(k) == cfg_c.get(k) for k in
               ("diameter_um", "focal_um", "na", "lam_min_um", "lam_max_um", "target_wavelengths_um",
                "ring_width_um", "h_max_um", "dh_um", "fom_mode", "softmin_beta_final", "overlap_fom",
                "overlap_airy_factor", "overlap_r_enc_um", "ring_quadrature"))
    print("bare:      %s  (%s)" % (bare_dir, cfg_b.get("name")))
    print("corrected: %s  (%s), table %s" % (corr_dir, cfg_c.get("name"), corr_npz))
    if not same:
        print("  WARNING: the two runs differ in more than the correction table -- the 2 x 2 "
              "still holds, the reading does not")
    if m_b.size != m_c.size:
        raise SystemExit("the two vectors have different lengths (%d / %d rings)" % (m_b.size, m_c.size))
    probs = {"bare": build(cfg_b), "corr": build(cfg_b, corr_npz)}
    vecs = {"bare": m_b, "corr": m_c}
    lam_c, r_c, corr_c = load_efficiency_table(corr_npz)
    print("  table: %d wavelengths x %d radii (%s um), eta %.3f-%.3f"
          % (lam_c.size, r_c.size, ", ".join("%.0f" % r for r in r_c), corr_c.min(), corr_c.max()))
    p0 = probs["bare"]
    print("  N=%d rings x %.2f um, F=%.2f mm, NA=%.4f | fom_mode=%s%s | objective=%s | quadrature %s"
          % (p0.N, p0.delta, p0.F / 1000.0, p0.na, p0.fom_mode,
             " (beta=%.0f)" % p0.softmin_beta if p0.fom_mode == "softmin" else "",
             p0.objective, cfg_b.get("ring_quadrature", "(absent -> midpoint)")))

    J = {(t, v): probs[t].fom(vecs[v]) for t in probs for v in vecs}
    I = {(t, v): probs[t].per_wavelength(probs[t].field(vecs[v])) for t in probs for v in vecs}
    changed = int(np.count_nonzero(m_b != m_c))
    print("  vectors: %d of %d rings differ (%.1f %%), mean |dh| %.3f levels over the changed rings"
          % (changed, m_b.size, 100.0 * changed / m_b.size,
             float(np.abs(m_b - m_c)[m_b != m_c].mean()) if changed else 0.0))
    print("  J (%s):                bare vector   corrected vector" % p0.fom_mode)
    print("    bare tables            %.4f        %.4f" % (J[("bare", "bare")], J[("bare", "corr")]))
    print("    corrected tables       %.4f        %.4f" % (J[("corr", "bare")], J[("corr", "corr")]))
    for t, v, j_ref, name in (("bare", "bare", j_b, "bare"), ("corr", "corr", j_c, "corrected")):
        if j_ref is not None:
            d = J[(t, v)] - float(j_ref)
            print("    %s run's design_metrics J_final %.4f -> rebuild %s (diff %.1e)"
                  % (name, j_ref, "MATCHES" if abs(d) < 2e-4 else "DIFFERS", d))
    print("  the correction on the bare vector:   J x %.3f   (physics: what the scalar model overstated)"
          % (J[("corr", "bare")] / max(J[("bare", "bare")], 1e-30)))
    print("  the re-optimization, corrected tables: J x %.3f   (what the optimizer recovered)"
          % (J[("corr", "corr")] / max(J[("corr", "bare")], 1e-30)))
    print("  the corrected vector on the bare tables: J x %.3f of the bare vector (its scalar-model cost)"
          % (J[("bare", "corr")] / max(J[("bare", "bare")], 1e-30)))
    label = "eta_w (encircled)" if p0.objective == "overlap" else "I_w (on-axis)"
    print("  per target line, %s:" % label)
    print("    nm   bare/bare  corr/bare  ratio | bare/corr  corr/corr  ratio     (tables/vector)")
    for w, lam in enumerate(p0.lam):
        a, b = I[("bare", "bare")][w], I[("corr", "bare")][w]
        c, d = I[("bare", "corr")][w], I[("corr", "corr")][w]
        print("  %5.0f    %.4f     %.4f    %.3f |  %.4f     %.4f    %.3f"
              % (1000 * lam, a, b, b / max(a, 1e-30), c, d, d / max(c, 1e-30)))
    for t in probs:
        for v in vecs:
            vals = I[(t, v)]
            print("  %s tables, %s vector: mean %.4f, worst %.0f nm (%.4f), best %.0f nm (%.4f)"
                  % (t, v, vals.mean(), 1000 * p0.lam[np.argmin(vals)], vals.min(),
                     1000 * p0.lam[np.argmax(vals)], vals.max()))


if __name__ == "__main__":
    main()
