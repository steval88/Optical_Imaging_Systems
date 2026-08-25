"""
run_MDL_design.py -- STAGE 1: MDL design (Search GA+HJA -> Smooth ->
Gradient). Unified driver (replaces run_design.py and run_s3.py).

All settings live in ONE dictionary. Edit `SETTINGS = ...` at the top of
the CONFIG section and, FROM THE PACKAGE ROOT, run:

    python 01_design\\run_MDL_design.py

Every run creates its own timestamped folder under the package root

    runs/<YYYYMMDD_HHMMSS>_<name>/
        config.json          the exact settings used (incl. derived values)
        scripts/             snapshot of the scripts as they were at run time
        m_final.npy          optimized gray-level vector
        mdl_rings_<n>.txt    ring table for the Zemax UDS DLLs / export_gds.py
        design_metrics.json  J at every pipeline stage

so results are never overwritten and every result can be traced back to
the code and settings that produced it. Downstream (stage 2/3):

    python 02_validation_rs\\run_verify.py  runs/<folder>
    python 02_validation_rs\\make_plots.py  runs/<folder>
    python 03_tapeout\\export_gds.py --rings runs/<folder>/mdl_rings_<n>.txt ...

Objective modes (choose via "target_wavelengths_um"):
    None      -> continuous-band objective, uniform in angular frequency
                 with "n_wavelengths" samples (alias-checked below)
    [list]    -> discrete comb objective on EXACTLY those wavelengths (um),
                 e.g. the paper's 14 measurement lines (PAPER_COMB_14) or
                 any custom set such as laser lines [0.450, 0.520, 0.638]
"""
import json
import os
import shutil
import sys
import time

import numpy as np

# package root = parent of this stage folder; mdl_core.py lives there
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from mdl_core import (MDLProblem, PAPER_COMB_14, multistep_GA_HJA_combo,
                      search, smooth, gradient_refine, hooke_jeeves,
                      seed_harmonic, upper_bound_jf)

# =========================================================================
# CONFIG -- edit the dictionary(ies) below, then set  SETTINGS = <one of them>
# All lengths in micrometers unless the key says otherwise.
# =========================================================================

S3_CONTINUOUS = {
    "name": "s3_continuous",
    # -- geometry ---------------------------------------------------------
    "diameter_um": 10240.0,
    "focal_um": 50940.0,          # give focal_um OR na (set the other None)
    "na": None,
    # -- band / objective -------------------------------------------------
    "lam_min_um": 0.40,
    "lam_max_um": 1.10,
    "target_wavelengths_um": None,   # None = continuous band
    "n_wavelengths": 1001,           # continuous sampling (alias-checked)
    "fom_mode": "geomean",              # "mean": J=<I>_w (paper Eq. 2; may
                                     #   sacrifice wavelengths entirely)
                                     # "geomean": product-type J -- keeps a
                                     #   focus at EVERY target wavelength
                                     #   (balanced comb, paper Fig. 2e look)
    # -- fabrication ------------------------------------------------------
    "ring_width_um": 2.0,
    "h_max_um": 15.0,
    "dh_um": 0.078,
    # -- optimizer --------------------------------------------------------
    "seed_lam0_um": [0.45, 0.50, 0.55, 0.60, 0.70],
    "verbatim_fig_s3": False, #True,      # True: paper Fig. S3 GA+HJA combo
    "ga_blocks": 3,               # s (blocks) in the combo / legacy search
    "ga_epochs": 30,              # p (epochs per block)
    "pop_size": 24,
    "gradient_iters": 300,
    "hja_d0": 2,
    "hja_max_sweeps": 80,
    "rng_seed": 7,
    "compute_upper_bound": True,  # alias-free coherence ceiling (info only)
    "bound_n_rho": 256,
    "bound_n_wavelengths": 512,
    # -- verification (consumed later by run_verify.py) -------------------
    "verify_wavelengths_um": list(PAPER_COMB_14),
    "verify_z_span_um": 2500.0,   # on-axis scan: F +/- span
    "verify_z_points": 401,
    "verify_r_max_um": 30.0,      # PSF radial grid
    "verify_r_points": 601,
    # -- packaging --------------------------------------------------------
    "dll_file_no": 2,             # ring table name: mdl_rings_<n>.txt
    "reuse_design_npy": None,     # path to an existing m.npy: skip the
                                  # optimizer, just (re)package that design
    "runs_dir": "runs",           # created under the package root
    "snapshot_scripts": [         # paths relative to the package root
        "mdl_core.py",
        "01_design/run_MDL_design.py",
        "02_validation_rs/run_verify.py",
        "02_validation_rs/make_plots.py",
        "02_validation_zemax/mdl_zemax_validation.py",
        "03_tapeout/export_gds.py",
    ],
}

# paper-style comb objective: same geometry, optimize on the 14 lines
S3_COMB = dict(S3_CONTINUOUS,
               name="s3_comb",
               target_wavelengths_um=list(PAPER_COMB_14),
               dll_file_no=3)

# the original D = 1 cm, NA 0.3 broadband design point
NA03 = dict(S3_CONTINUOUS,
            name="na03_broadband",
            diameter_um=10000.0, focal_um=None, na=0.3,
            ring_width_um=0.65, h_max_um=28.0,
            n_wavelengths=2001, ga_epochs=25, pop_size=20,
            rng_seed=42, gradient_iters=250, hja_max_sweeps=60,
            dll_file_no=1)


# --- SWIR band 1100-1800 nm on the S3 geometry ---------------------------
# Band ratio 1.64 (vs 2.75 in the visible): easier problem. Ceilings at
# this geometry: 0.088 (H=15 um), 0.155 (H=30), 0.216 (H=45) -- raising
# H is the biggest lever if the writer allows it. CAUTION: n_az4562 is
# a Cauchy fit to data ending at 1100 nm; n ~ 1.60 extrapolates safely,
# but resist ABSORPTION (C-H/O-H overtones near 1.4/1.7 um) is not in
# the model -- verify n AND k before tape-out. Note every band-coupled
# key changes here, not just lam_min/lam_max: seeds, verify list,
# aperture of the verification grids, and the DLL file number.
SWIR_LAMS_15 = [round(1.10 + 0.05 * i, 3) for i in range(15)]
SWIR_CONTINUOUS = dict(S3_CONTINUOUS,
                       name="swir_continuous",
                       lam_min_um=1.10, lam_max_um=1.80,
                       target_wavelengths_um=None,
                       n_wavelengths=501,        # alias-free min is 91
                       fom_mode="mean",
                       seed_lam0_um=[1.20, 1.30, 1.45, 1.60, 1.75],
                       verify_wavelengths_um=SWIR_LAMS_15,
                       verify_r_max_um=50.0,     # spots ~2x larger here
                       rzmap_r_max_um=30.0,
                       dll_file_no=4)
# balanced comb on 15 SWIR lines (all-wavelengths-locked behavior)
SWIR_COMB_GEO = dict(SWIR_CONTINUOUS,
                     name="swir_comb_geo",
                     target_wavelengths_um=SWIR_LAMS_15,
                     fom_mode="geomean", verbatim_fig_s3=False,
                     dll_file_no=5)


SETTINGS = SWIR_COMB_GEO #S3_COMB #S3_CONTINUOUS          # <-- EDIT: pick / customize a dictionary

# =========================================================================
# no user-serviceable parts below
# =========================================================================
T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def resolve_geometry(cfg):
    """Return (D_um, na, F_um) from diameter + (focal_um XOR na)."""
    D = float(cfg["diameter_um"])
    R = 0.5 * D
    if cfg.get("focal_um"):
        F = float(cfg["focal_um"])
        na = R / np.sqrt(R * R + F * F)
    elif cfg.get("na"):
        na = float(cfg["na"])
        F = R * np.sqrt(1.0 / na ** 2 - 1.0)
    else:
        raise SystemExit("config error: set focal_um or na")
    return D, na, F


def main(cfg):
    D, na, F = resolve_geometry(cfg)
    R = 0.5 * D
    lmin, lmax = cfg["lam_min_um"], cfg["lam_max_um"]
    lams = cfg["target_wavelengths_um"]
    comb_mode = lams is not None

    # ---- run folder + snapshots + config -------------------------------
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PKG_ROOT, cfg["runs_dir"],
                           "%s_%s" % (stamp, cfg["name"]))
    scripts_dir = os.path.join(run_dir, "scripts")
    os.makedirs(scripts_dir)
    for fn in cfg["snapshot_scripts"]:
        src = os.path.join(PKG_ROOT, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(scripts_dir,
                                           os.path.basename(fn)))
    log("run folder: %s" % os.path.relpath(run_dir))

    # ---- alias safety check (continuous mode only) ----------------------
    L_max = np.sqrt(R * R + F * F) - F
    nw_min = int(np.ceil(L_max * (1.0 / lmin - 1.0 / lmax)))
    if not comb_mode and cfg["n_wavelengths"] < nw_min:
        raise SystemExit(
            "alias error: n_wavelengths=%d < %d required for L_max=%.0f um "
            "-- increase n_wavelengths" % (cfg["n_wavelengths"], nw_min,
                                           L_max))
    log("geometry: D=%.2f mm  F=%.2f mm  NA=%.4f  (L_max=%.0f um, "
        "alias-free Nw >= %d)" % (D / 1000, F / 1000, na, L_max, nw_min))

    # ---- objective ------------------------------------------------------
    nw = len(lams) if comb_mode else cfg["n_wavelengths"]
    prob = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"],
                      cfg["h_max_um"], cfg["dh_um"], n_wavelengths=nw)
    if comb_mode:
        prob.set_wavelengths(np.asarray(lams, dtype=float))
        log("objective: discrete comb on %d wavelengths: %s"
            % (len(lams), ", ".join("%.0f" % (l * 1000) for l in lams)))
    else:
        log("objective: continuous band %.0f-%.0f nm, Nw=%d"
            % (lmin * 1000, lmax * 1000, nw))
    prob.fom_mode = cfg.get("fom_mode", "mean")
    prob.G = prob.G.astype(np.complex64)
    prob.L = prob.L.astype(np.complex64)
    log("problem: N=%d rings, M=%d levels, fom_mode=%s"
        % (prob.N, prob.M, prob.fom_mode))

    # write config (incl. derived values) before the long part
    cfg_out = dict(cfg)
    cfg_out["derived"] = {"na": float(na), "focal_um": float(F),
                          "N_rings": int(prob.N), "M_levels": int(prob.M),
                          "L_max_um": float(L_max),
                          "alias_free_nw_min": nw_min,
                          "timestamp": stamp}
    json.dump(cfg_out, open(os.path.join(run_dir, "config.json"), "w"),
              indent=1)

    # ---- optional ceiling ------------------------------------------------
    bound = None
    if cfg["compute_upper_bound"]:
        bound = upper_bound_jf(D, na, lmin, lmax, cfg["h_max_um"],
                               cfg["dh_um"], n_rho=cfg["bound_n_rho"],
                               n_wavelengths=cfg["bound_n_wavelengths"])
        log("alias-free continuous-band ceiling: %.4f" % bound)

    # ---- design ---------------------------------------------------------
    if cfg["reuse_design_npy"]:
        m_f = np.load(cfg["reuse_design_npy"])
        if m_f.size != prob.N:
            raise SystemExit("reuse error: %s has %d rings, config needs %d"
                             % (cfg["reuse_design_npy"], m_f.size, prob.N))
        f_f = prob.fom(m_f)
        stages = {"J_final": f_f,
                  "reused_from": cfg["reuse_design_npy"]}
        log("reused %s: J=%.4f (no optimization)"
            % (cfg["reuse_design_npy"], f_f))
    else:
        best = (None, -1.0, None)
        for lam0 in cfg["seed_lam0_um"]:
            n0 = float(prob.n_func(lam0))
            pmax = max(1, int(np.floor(cfg["h_max_um"] * (n0 - 1.0) / lam0)))
            for p in sorted({1, 2, 4, 8, pmax // 2, pmax}):
                if p < 1:
                    continue
                m = seed_harmonic(prob, lam0, p)
                f = prob.fom(m)
                if f > best[1]:
                    best = (m, f, (lam0, p))
        log("best harmonic seed: lam0=%.2f p=%d  J=%.4f"
            % (best[2][0], best[2][1], best[1]))

        rng = np.random.default_rng(cfg["rng_seed"])
        if cfg["verbatim_fig_s3"]:
            m_s, f_s = multistep_GA_HJA_combo(
                prob, m_init=best[0], s=cfg["ga_blocks"],
                p=cfg["ga_epochs"], pop_size=cfg["pop_size"],
                rng=rng, verbose=log)
        else:
            m_s, f_s = search(prob, m_init=best[0],
                              blocks=cfg["ga_blocks"],
                              ga_epochs=cfg["ga_epochs"],
                              pop_size=cfg["pop_size"], rng=rng,
                              verbose=log)
        log("Search: J=%.4f" % f_s)
        m_sm = smooth(prob, m_s)
        f_sm = prob.fom(m_sm)
        log("Smooth: %.4f -> %.4f" % (f_s, f_sm))
        m_g, f_g = gradient_refine(prob, m_sm, iters=cfg["gradient_iters"])
        log("Gradient: J=%.4f" % f_g)
        m_f, f_f = hooke_jeeves(prob, m_g, d0=cfg["hja_d0"],
                                max_sweeps=cfg["hja_max_sweeps"])
        log("Polish: J=%.4f" % f_f)
        stages = {"J_seed": best[1],
                  "seed": {"lam0_um": best[2][0], "p": best[2][1]},
                  "J_search": f_s, "J_smooth": f_sm,
                  "J_gradient": f_g, "J_final": f_f}

    # ---- outputs --------------------------------------------------------
    np.save(os.path.join(run_dir, "m_final.npy"), m_f)
    ring_name = "mdl_rings_%d.txt" % cfg["dll_file_no"]
    with open(os.path.join(run_dir, ring_name), "w") as fh:
        fh.write("%d %.9f\n" % (prob.N, prob.delta / 1000.0))
        fh.writelines("%.9f\n" % (v * prob.dh / 1000.0) for v in m_f)
    stages["upper_bound_continuous"] = bound
    json.dump(stages, open(os.path.join(run_dir, "design_metrics.json"),
                           "w"), indent=1)
    log("saved: m_final.npy, %s, design_metrics.json" % ring_name)
    rel = os.path.relpath(run_dir)
    log("done -> %s" % rel)
    log("next:  python %s %s"
        % (os.path.join("02_validation_rs", "run_verify.py"), rel))
    return run_dir


if __name__ == "__main__":
    main(SETTINGS)