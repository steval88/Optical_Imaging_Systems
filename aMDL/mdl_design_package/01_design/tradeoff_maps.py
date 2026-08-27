"""
tradeoff_maps.py -- STAGE 0 (design-space exploration): reproduce the
paper's Fig. 1b/1c and Fig. 1d trade-off analyses for ANY band,
material and fabrication constraints, BEFORE committing to a design.

Usage (from the package root; edit the SETTINGS dictionary first):

    python 01_design\\tradeoff_maps.py

Two outputs, matching the paper's two panels (Xiao et al., Light Sci.
Appl. 11:323 (2022), Fig. 1; theory: their Eqs. S14-S15, implemented in
mdl_core.pairwise_bound_matrix / upper_bound_jf, brute-force validated
-- see tools/validate_bound.py):

(a) PAIR MAPS (Fig. 1b/1c analogue). For each requested (H, D) panel,
    the map  max_dm Re < J_w(rho_1, rho_2) >_w  over the normalized
    ring-pair plane (rho_1/R, rho_2/R): the best band-averaged mutual
    coherence any height profile could give that ring pair, maximized
    over their height difference (dm gray levels). Red (=1) regions can
    interfere constructively across the WHOLE band; the red wedge
    shrinking toward the diagonal as D grows (or H shrinks) is the
    visual form of the aperture/bandwidth/thickness trade-off: distant
    ring pairs acquire geometric path spreads L = r1 - r2 larger than
    the (n_g - 1) H group-delay budget and decohere.

(b) D-H MAP (Fig. 1d analogue). The scalar ceiling  max J_w(F)
    (pair maps contracted with the paraxial aperture weights, Eq. S15)
    swept over a (thickness H, diameter D) grid at fixed NA, with the
    user's sample points starred. This is THE chart for choosing a
    design point: it tells you the best achievable J before any
    optimization is run.

Outputs go into a timestamped run folder (same convention as stage 1):

    runs/<stamp>_<name>/
        config.json           settings + derived values
        tradeoff_pairmaps.npz rho grid + B matrix per panel
        tradeoff_dh_map.npz   H grid, D grid, ceiling matrix
        fig_pair_maps.png     panel row, Fig. 1b/c style
        fig_dh_map.png        D-H heatmap with sample stars
        scripts/              snapshot of this script + mdl_core.py

Cost note: each D-H grid point is one full bound evaluation; the sweep
dominates runtime. Grid resolution and bound accuracy (bound_n_rho /
bound_n_wavelengths) are the knobs; progress + ETA are logged per row.
Accuracy note: the ceiling is alias-free and dispersion-aware but still
an UPPER bound -- achieved designs land at roughly 40-50 % of it (see
project findings); read the D-H map comparatively, not absolutely.
"""
import json
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# package root = parent of this stage folder; mdl_core.py lives there
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from mdl_core import pairwise_bound_matrix, upper_bound_jf

# =========================================================================
# CONFIG -- edit, then run. All lengths in micrometers unless stated.
# =========================================================================

# The paper's own study (Fig. 1b/c/d): visible band, NA of the S-series,
# panels S1..S5. Reproducing this validates the tool against the paper.
PAPER_FIG1 = {
    "name": "tradeoff_paper_fig1",
    "lam_min_um": 0.40,
    "lam_max_um": 1.10,
    "na": 0.1006,                 # S-series NA (D and F scale together)
    "dh_um": 0.078,
    # -- (a) pair maps: one panel per (H, D), Fig. 1b/1c style ------------
    "pair_panels": [
        {"label": "S1", "h_max_um": 15.0, "diameter_um": 1024.0},
        {"label": "S2", "h_max_um": 15.0, "diameter_um": 3072.0},
        {"label": "S3", "h_max_um": 15.0, "diameter_um": 10240.0},
        {"label": "S4", "h_max_um": 5.0,  "diameter_um": 10240.0},
        {"label": "S5", "h_max_um": 1.0,  "diameter_um": 10240.0},
    ],
    "pair_n_rho": 200,            # (rho1, rho2) resolution per panel
    "pair_n_wavelengths": 512,
    # -- (b) D-H ceiling map, Fig. 1d style -------------------------------
    "h_grid_um":  [0.5, 20.0, 40],    # [min, max, points]
    "d_grid_mm":  [0.3, 11.0, 36],    # [min, max, points]
    # Per-grid-point bound accuracy. 64/192 is ~4-5% above the converged
    # 256/512 value -- fine for a comparative map (sweep ~5-10 min).
    # Small-D points dominate runtime (more coherent pairs to evaluate).
    "bound_n_rho": 64,
    "bound_n_wavelengths": 192,
    "star_samples": [             # points marked on the D-H map
        {"label": "S1", "h_max_um": 15.0, "diameter_um": 1024.0},
        {"label": "S2", "h_max_um": 15.0, "diameter_um": 3072.0},
        {"label": "S3", "h_max_um": 15.0, "diameter_um": 10240.0},
        {"label": "S4", "h_max_um": 5.0,  "diameter_um": 10240.0},
        {"label": "S5", "h_max_um": 1.0,  "diameter_um": 10240.0},
    ],
    "runs_dir": "runs",
}

# Same analysis for the SWIR band on the S3 geometry family
SWIR_TRADEOFF = dict(PAPER_FIG1,
                     name="tradeoff_swir",
                     lam_min_um=1.10, lam_max_um=1.80,
                     pair_panels=[
                         {"label": "H15", "h_max_um": 15.0,
                          "diameter_um": 10240.0},
                         {"label": "H30", "h_max_um": 30.0,
                          "diameter_um": 10240.0},
                         {"label": "H45", "h_max_um": 45.0,
                          "diameter_um": 10240.0},
                     ],
                     h_grid_um=[2.0, 50.0, 40],
                     star_samples=[
                         {"label": "S3-SWIR", "h_max_um": 15.0,
                          "diameter_um": 10240.0},
                     ])

SETTINGS = PAPER_FIG1             # <-- EDIT: pick / customize a dictionary

# =========================================================================
# no user-serviceable parts below
# =========================================================================
T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def main(cfg):
    lmin, lmax = cfg["lam_min_um"], cfg["lam_max_um"]
    na, dh = cfg["na"], cfg["dh_um"]

    # ---- run folder + snapshots + config --------------------------------
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PKG_ROOT, cfg["runs_dir"],
                           "%s_%s" % (stamp, cfg["name"]))
    os.makedirs(os.path.join(run_dir, "scripts"))
    for fn in ("01_design/tradeoff_maps.py", "mdl_core.py"):
        src = os.path.join(PKG_ROOT, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, "scripts",
                                           os.path.basename(fn)))
    cfg_out = dict(cfg)
    cfg_out["derived"] = {"timestamp": stamp}
    json.dump(cfg_out, open(os.path.join(run_dir, "config.json"), "w"),
              indent=1)
    log("run folder: %s" % os.path.relpath(run_dir))
    log("band %.0f-%.0f nm, NA=%.4f, dh=%g nm"
        % (lmin * 1000, lmax * 1000, na, dh * 1000))

    # ---- (a) pairwise coherence maps ------------------------------------
    panels = cfg["pair_panels"]
    log("[1/2] pair maps max Re J_w(rho1, rho2): %d panels, "
        "%d x %d rho grid each" % (len(panels), cfg["pair_n_rho"],
                                   cfg["pair_n_rho"]))
    pm_store = {}
    for p in panels:
        t0 = time.time()
        rho_norm, B = pairwise_bound_matrix(
            p["diameter_um"], na, lmin, lmax, p["h_max_um"], dh,
            n_rho=cfg["pair_n_rho"],
            n_wavelengths=cfg["pair_n_wavelengths"])
        pm_store["rho_norm"] = rho_norm
        pm_store["B_%s" % p["label"]] = B
        frac_coherent = float((B > 0.9).mean())
        log("  %s (H=%g um, D=%.2f mm): done (%.1fs); fully-coherent "
            "pair fraction (B>0.9): %.1f%%"
            % (p["label"], p["h_max_um"], p["diameter_um"] / 1000,
               time.time() - t0, 100 * frac_coherent))
    np.savez(os.path.join(run_dir, "tradeoff_pairmaps.npz"), **pm_store)

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.0 * len(panels) + 1.0, 3.2),
                             squeeze=False)
    for j, p in enumerate(panels):
        ax = axes[0][j]
        B = pm_store["B_%s" % p["label"]]
        im = ax.imshow(B.T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="turbo", vmin=0, vmax=1, aspect="equal")
        ax.set_title("%s:  H=%g µm  D=%.2f mm"
                     % (p["label"], p["h_max_um"],
                        p["diameter_um"] / 1000), fontsize=9)
        ax.set_xlabel(r"$\rho_1/R$")
        if j == 0:
            ax.set_ylabel(r"$\rho_2/R$")
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes[0][-1], pad=0.04, fraction=0.05)
    cb.set_label(r"max Re $J_\omega(\rho_1,\rho_2)$", fontsize=8)
    fig.suptitle("Pairwise band coherence (paper Fig. 1b/c analogue) — "
                 "%.0f–%.0f nm, NA %.3f"
                 % (lmin * 1000, lmax * 1000, na), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_pair_maps.png"), dpi=150,
                bbox_inches="tight")
    log("[1/2] fig_pair_maps.png saved")

    # ---- (b) D-H ceiling sweep ------------------------------------------
    h0, h1, nh = cfg["h_grid_um"]
    d0, d1, nd = cfg["d_grid_mm"]
    H_grid = np.linspace(h0, h1, int(nh))
    D_grid = np.linspace(d0, d1, int(nd)) * 1000.0        # -> um
    log("[2/2] D-H ceiling map max J_w(F): %d x %d grid = %d bound "
        "evaluations (n_rho=%d, Nw=%d each)"
        % (int(nh), int(nd), int(nh) * int(nd), cfg["bound_n_rho"],
           cfg["bound_n_wavelengths"]))
    M = np.empty((int(nd), int(nh)))
    t_sweep = time.time()
    for i, Dum in enumerate(D_grid):
        t_row = time.time()
        for k, Hum in enumerate(H_grid):
            M[i, k] = upper_bound_jf(
                Dum, na, lmin, lmax, Hum, dh,
                n_rho=cfg["bound_n_rho"],
                n_wavelengths=cfg["bound_n_wavelengths"])
        done = i + 1
        eta = (time.time() - t_sweep) / done * (int(nd) - done)
        log("  D=%.2f mm row done (%.1fs)  [%d/%d, ETA %.0fs]"
            % (Dum / 1000, time.time() - t_row, done, int(nd), eta))
    np.savez(os.path.join(run_dir, "tradeoff_dh_map.npz"),
             h_grid_um=H_grid, d_grid_mm=D_grid / 1000, ceiling=M)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(M, origin="lower", aspect="auto",
                   extent=[H_grid[0], H_grid[-1],
                           D_grid[0] / 1000, D_grid[-1] / 1000],
                   cmap="RdBu_r", vmin=0, vmax=1)
    for s in cfg["star_samples"]:
        ax.plot(s["h_max_um"], s["diameter_um"] / 1000, marker="*",
                ms=13, mec="black", mfc="red", lw=0)
        ax.annotate(s["label"],
                    (s["h_max_um"], s["diameter_um"] / 1000),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=9)
    ax.set_xlabel("max relief thickness H (µm)")
    ax.set_ylabel("diameter D (mm)")
    ax.set_title("Achievability ceiling max $J_\\omega(F)$ "
                 "(paper Fig. 1d analogue) — %.0f–%.0f nm, NA %.3f"
                 % (lmin * 1000, lmax * 1000, na), fontsize=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("max $J_\\omega(F)$")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_dh_map.png"), dpi=150,
                bbox_inches="tight")
    log("[2/2] fig_dh_map.png saved")

    # star-point ceilings, printed for direct use in design decisions
    for s in cfg["star_samples"]:
        b = upper_bound_jf(s["diameter_um"], na, lmin, lmax,
                           s["h_max_um"], dh,
                           n_rho=cfg["bound_n_rho"],
                           n_wavelengths=cfg["bound_n_wavelengths"])
        log("  ceiling at %s (H=%g um, D=%.2f mm): %.4f"
            % (s["label"], s["h_max_um"], s["diameter_um"] / 1000, b))

    log("done -> %s" % os.path.relpath(run_dir))
    log("next: pick a (H, D) with an acceptable ceiling, put it in "
        "run_MDL_design.py's SETTINGS, and run stage 1")
    return run_dir


if __name__ == "__main__":
    main(SETTINGS)