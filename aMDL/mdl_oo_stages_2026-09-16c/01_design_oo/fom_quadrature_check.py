"""
fom_quadrature_check.py -- what the design FOM sees with the midpoint
rule (paper Eq. 4) versus the sinc ring-quadrature rule, for an EXISTING
design vector. Nothing is optimized; nothing in the run folder is
written. Usage (from the package root):

    python 01_design_oo\\fom_quadrature_check.py runs\\<run_folder> [other_m.npy]

The problem is rebuilt exactly as run_MDL_design.py built it for that
run (geometry, target lines, fom_mode, softmin beta -- the FINAL
annealed value, since design_metrics.json's J_final was evaluated with
it -- and the overlap objective with its chromatic disc), once per
quadrature rule. Printed per target line: the per-wavelength objective
value I_w (on-axis |U|^2, or the encircled eta_w for the overlap
objective) under both rules and their ratio; then the aggregated J of
both rules, and the J under the run's OWN rule (config key
ring_quadrature, midpoint when absent) against design_metrics.json's
J_final (they must agree to float32 precision -- this is the check that
the rebuilt problem IS the run's problem). Read: the sinc/midpoint ratio
per line is the fraction of the midpoint-predicted focal signal the
rings can physically blaze; the optimizer that produced this vector
was rewarded for the rest.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the mdl package
from mdl import MDLProblem                            # noqa: E402


def build(cfg, quad):
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
                      n_wavelengths=nw, ring_quadrature=quad)
    if lams is not None:
        prob.set_wavelengths(np.asarray(lams, float))
    prob.fom_mode = cfg.get("fom_mode", "mean")
    if prob.fom_mode == "softmin":
        prob.softmin_beta = float(cfg.get("softmin_beta_final")
                                  or cfg.get("softmin_beta", 20.0))
    prob.use_single_precision()
    if cfg.get("overlap_fom"):
        prob.enable_overlap_fom(cfg.get("overlap_r_enc_um"),
                                cfg.get("overlap_n_r0", 24),
                                cfg.get("overlap_airy_factor", 2.0))
    return prob


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run_dir = sys.argv[1]
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    m_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir,
                                                                "m_final.npy")
    m = np.load(m_file).astype(np.int64)
    dm_path = os.path.join(run_dir, "design_metrics.json")
    j_final = json.load(open(dm_path)).get("J_final") if os.path.exists(dm_path) \
        else None

    probs = {q: build(cfg, q) for q in ("midpoint", "sinc")}
    p0 = probs["midpoint"]
    print("design: %s  (%s)" % (cfg.get("name", "?"), m_file))
    print("  N=%d rings x %.2f um, F=%.2f mm, NA=%.4f | fom_mode=%s%s | "
          "objective=%s%s | run's ring_quadrature key: %s"
          % (p0.N, p0.delta, p0.F / 1000.0, p0.na, p0.fom_mode,
             " (beta=%.0f)" % p0.softmin_beta if p0.fom_mode == "softmin"
             else "", p0.objective,
             " (chromatic disc %.1f Airy radii)" % cfg.get(
                 "overlap_airy_factor", 2.0)
             if p0.objective == "overlap" and not cfg.get("overlap_r_enc_um")
             else "", cfg.get("ring_quadrature", "(absent -> midpoint)")))
    I = {q: probs[q].per_wavelength(probs[q].field(m)) for q in probs}
    J = {q: probs[q].fom(m) for q in probs}
    S_rim = {q: probs[q].S[:, -1] for q in probs}
    label = "eta_w (encircled)" if p0.objective == "overlap" else "I_w (on-axis)"
    print("  per target line: %s under both ring quadratures" % label)
    print("   nm    S_rim(sinc)   midpoint     sinc     sinc/midpoint")
    for w, lam in enumerate(p0.lam):
        print("  %4.0f     %.3f       %.4f     %.4f      %.3f"
              % (1000 * lam, S_rim["sinc"][w], I["midpoint"][w], I["sinc"][w],
                 I["sinc"][w] / max(I["midpoint"][w], 1e-30)))
    print("  J (%s): midpoint %.4f | sinc %.4f | ratio %.3f"
          % (p0.fom_mode, J["midpoint"], J["sinc"],
             J["sinc"] / max(J["midpoint"], 1e-30)))
    if j_final is not None:
        own = cfg.get("ring_quadrature", "midpoint")      # the run's rule
        d = J[own] - float(j_final)
        print("  design_metrics.json J_final = %.4f (run's rule: %s) -> %s "
              "rebuild %s (diff %.1e)"
              % (j_final, own, own, "MATCHES" if abs(d) < 2e-4 else
                 "DIFFERS: rebuilt problem != run's problem", d))
    print("  mean over lines: midpoint %.4f | sinc %.4f   (softmin J is the "
          "smooth minimum, mean shows the balance)"
          % (I["midpoint"].mean(), I["sinc"].mean()))
    print("  worst line: midpoint %.0f nm (%.4f) | sinc %.0f nm (%.4f)"
          % (1000 * p0.lam[np.argmin(I["midpoint"])], I["midpoint"].min(),
             1000 * p0.lam[np.argmin(I["sinc"])], I["sinc"].min()))


if __name__ == "__main__":
    main()
