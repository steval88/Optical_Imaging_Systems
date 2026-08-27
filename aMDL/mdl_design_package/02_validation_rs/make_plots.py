"""Report figures for an MDL design run.

STAGE 2a (figures). Usage (from the package root):
    python 02_validation_rs\\make_plots.py runs/<run_folder>

Reads config.json, verify_metrics.json and verify_onaxis.npz from the
run folder (produced by run_MDL_design.py + run_verify.py) and writes,
into the same folder:

    fig_onaxis.png    on-axis intensity vs z per wavelength, GLOBALLY
                      normalized (honest relative brightness)
    fig_onaxis_perlambda.png
                      same data, each wavelength normalized to its own
                      peak -- this is how the paper's Fig. 2e is drawn,
                      so use THIS one for a like-for-like comparison
                      (it hides the efficiency differences between
                      wavelengths; the global one shows them)
    fig_rz_tiles.png  per-wavelength I(r, z) intensity maps around the
                      focus, one tile per wavelength, each normalized
                      to its own max -- the DIRECT analogue of the
                      paper's Fig. 2e (simulation) / Fig. 4a
                      (measurement) tile rows; requires
                      verify_rzmap.npz from the current run_verify.py
    fig_metrics.png   focus efficiency and PSF FWHM vs wavelength
    fig_profile.png   designed height profile (full aperture + rim zoom)
    fig_bpm_psf.png   (only if bpm_validate.py was run on this folder)
                      focal-plane PSF radial cuts per wavelength,
                      thin-element vs BPM overlaid -- the visual form
                      of the thin-element model-error check
    fig_onaxis_bpm.png / fig_onaxis_perlambda_bpm.png
                      (same condition) on-axis intensity vs z maps of
                      the BPM-propagated field, global / per-lambda
                      normalized -- compare against fig_onaxis(.png /
                      _perlambda.png), which use the thin-element
                      field on the full verification comb; the BPM
                      maps cover only the BPM wavelength subset

Everything (geometry, wavelengths, labels) comes from config.json;
only the plot styling below is fixed by design.
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- load run ------------------------------------------------------------
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2:
    raise SystemExit("usage: python make_plots.py runs/<run_folder>")
run_dir = sys.argv[1].rstrip("/\\")
if not os.path.exists(os.path.join(run_dir, "config.json")):
    # also accept a path relative to the package root (any CWD)
    alt = os.path.join(PKG_ROOT, run_dir)
    if os.path.exists(os.path.join(alt, "config.json")):
        run_dir = alt
cfg_path = os.path.join(run_dir, "config.json")
if not os.path.exists(cfg_path):
    # typo helper: suggest the closest existing run folders
    hint = ""
    runs_root = os.path.join(PKG_ROOT, "runs")
    if os.path.isdir(runs_root):
        import difflib
        cands = sorted(os.listdir(runs_root))
        close = difflib.get_close_matches(os.path.basename(run_dir),
                                          cands, n=3, cutoff=0.5)
        if close:
            hint = "\n  did you mean: " + "  ".join(
                os.path.join("runs", c) for c in close)
        elif cands:
            hint = "\n  most recent run folders: " + "  ".join(
                os.path.join("runs", c) for c in cands[-3:])
    raise SystemExit("no config.json in %r -- is this a run folder made "
                     "by run_MDL_design.py?%s" % (run_dir, hint))
cfg = json.load(open(cfg_path))
der = cfg["derived"]


def _need(name):
    p = os.path.join(run_dir, name)
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run run_verify.py %s first"
                         % (p, run_dir))
    return p


onaxis = np.load(_need("verify_onaxis.npz"))
metrics = json.load(open(_need("verify_metrics.json")))
m = np.load(_need("m_final.npy"))

NA = der["na"]
F_mm = der["focal_um"] / 1000.0
D_mm = cfg["diameter_um"] / 1000.0
ring_um = cfg["ring_width_um"]
dh_um = cfg["dh_um"]
label = "%s  (D=%.2f mm, f=%.2f mm, NA=%.3f, H=%g um)" % (
    cfg["name"], D_mm, F_mm, NA, cfg["h_max_um"])

# snapshot this script alongside the others
os.makedirs(os.path.join(run_dir, "scripts"), exist_ok=True)
shutil.copy2(__file__, os.path.join(run_dir, "scripts",
                                    os.path.basename(__file__)))

# ---- styling (fixed by design) -------------------------------------------
C_BLUE, C_ORANGE, C_TEAL = "#3366BB", "#CC5500", "#00887A"
INK, MUTED, GRID = "#333333", "#666666", "#DDDDDD"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK,
})

# ------------------------------------------------------------- fig_onaxis
# On-axis intensity vs z per wavelength -- the paper's Fig. 2e analogue.
z = onaxis["zgrid"] / 1000.0                      # mm
lams = sorted(int(k.split("_")[1]) for k in onaxis.files
              if k.startswith("I_"))
M = np.array([onaxis["I_%d" % l] for l in lams])
M = M / M.max()

fig, ax = plt.subplots(figsize=(8.6, 3.2))
ext = [0, len(lams), z[0], z[-1]]
im = ax.imshow(M.T, aspect="auto", origin="lower", extent=ext,
               cmap="magma", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(lams)) + 0.5)
ax.set_xticklabels(["%d" % l for l in lams], fontsize=7)
ax.axhline(F_mm, color="white", lw=0.8, ls="--", alpha=0.8)
ax.text(0.15, F_mm + 0.02 * (z[-1] - z[0]),
        "design focus F = %.2f mm" % F_mm, color="white", fontsize=7)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("on-axis distance z (mm)")
ax.set_title("On-axis intensity vs z — %s" % label, fontsize=9)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("normalized intensity", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(run_dir, "fig_onaxis.png"), bbox_inches="tight")

# same data, per-wavelength normalization (the paper's Fig. 2e convention:
# every panel scaled to its own maximum, efficiency differences hidden)
Mn = np.array([onaxis["I_%d" % l] for l in lams])
Mn = Mn / Mn.max(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8.6, 3.2))
im = ax.imshow(Mn.T, aspect="auto", origin="lower", extent=ext,
               cmap="magma", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(lams)) + 0.5)
ax.set_xticklabels(["%d" % l for l in lams], fontsize=7)
ax.axhline(F_mm, color="white", lw=0.8, ls="--", alpha=0.8)
ax.text(0.15, F_mm + 0.02 * (z[-1] - z[0]),
        "design focus F = %.2f mm" % F_mm, color="white", fontsize=7)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("on-axis distance z (mm)")
ax.set_title("On-axis intensity, per-λ normalized (paper Fig. 2e "
             "convention) — %s" % cfg["name"], fontsize=9)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("intensity / per-λ max", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(run_dir, "fig_onaxis_perlambda.png"),
            bbox_inches="tight")

# ------------------------------------------------------------ fig_rz_tiles
# Paper Fig. 2e / 4a convention: one r-z intensity tile per wavelength,
# each tile normalized to its own maximum, dashed line at the design
# focus. The radial half-profile is mirrored to +/-r for display.
rz_npz = os.path.join(run_dir, "verify_rzmap.npz")
if os.path.exists(rz_npz):
    rz = np.load(rz_npz)
    r0 = rz["r0grid"]
    zz = rz["zgrid"] / 1000.0                       # mm
    tlams = sorted(int(k.split("_")[1]) for k in rz.files
                   if k.startswith("I_"))
    n = len(tlams)
    fig, axes = plt.subplots(1, n, figsize=(0.95 * n + 0.9, 3.0),
                             sharey=True, squeeze=False)
    for j, lnm in enumerate(tlams):
        ax = axes[0][j]
        Mrz = rz["I_%d" % lnm]
        Mrz = Mrz / Mrz.max()                       # per-tile normalization
        sym = np.concatenate([Mrz[:, ::-1], Mrz[:, 1:]], axis=1)
        im = ax.imshow(sym, aspect="auto", origin="lower",
                       extent=[-r0[-1], r0[-1], zz[0], zz[-1]],
                       cmap="OrRd", vmin=0, vmax=1)
        ax.axhline(F_mm, color="black", lw=0.7, ls="--", alpha=0.7)
        ax.set_title("%d nm" % lnm, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        if j == 0:
            ax.set_ylabel("z (mm)")
            ax.set_xlabel("r (µm)")
        else:
            ax.set_xticklabels([])
    cb = fig.colorbar(im, ax=axes[0][-1], pad=0.04, fraction=0.15)
    cb.set_label("I / per-tile max", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Intensity I(r, z) around the focus — %s "
                 "(paper Fig. 2e convention)" % cfg["name"],
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_rz_tiles.png"),
                bbox_inches="tight")

# -------------------------------------------------------- fig_rz_tiles_bpm
# Same tile convention, but sourced from the CN-BPM exit field (rigorous
# relief propagation) + RS-I to each (r0, z): the direct visual check of
# the thin-element error on the paper-style maps. Data written by
# bpm_validate.py into bpm_psf.npz (keys rz_r0grid, rz_zgrid,
# Irz_bpm_<nm>; Irz_te_<nm> is the TE twin on the same grids). Skipped
# silently for bpm_psf.npz files from before this key existed.
bpm_npz_p = os.path.join(run_dir, "bpm_psf.npz")
if os.path.exists(bpm_npz_p):
    bz = np.load(bpm_npz_p)
    if "rz_r0grid" in bz.files:
        r0 = bz["rz_r0grid"]
        zz = bz["rz_zgrid"] / 1000.0                # mm
        tlams = sorted(int(k.split("_")[2]) for k in bz.files
                       if k.startswith("Irz_bpm_"))
        n = len(tlams)
        fig, axes = plt.subplots(1, n, figsize=(0.95 * n + 0.9, 3.0),
                                 sharey=True, squeeze=False)
        for j, lnm in enumerate(tlams):
            ax = axes[0][j]
            Mrz = bz["Irz_bpm_%d" % lnm]
            Mrz = Mrz / Mrz.max()                   # per-tile normalization
            sym = np.concatenate([Mrz[:, ::-1], Mrz[:, 1:]], axis=1)
            im = ax.imshow(sym, aspect="auto", origin="lower",
                           extent=[-r0[-1], r0[-1], zz[0], zz[-1]],
                           cmap="OrRd", vmin=0, vmax=1)
            ax.axhline(F_mm, color="black", lw=0.7, ls="--", alpha=0.7)
            ax.set_title("%d nm" % lnm, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(False)
            if j == 0:
                ax.set_ylabel("z (mm)")
                ax.set_xlabel("r (µm)")
            else:
                ax.set_xticklabels([])
        cb = fig.colorbar(im, ax=axes[0][-1], pad=0.04, fraction=0.15)
        cb.set_label("I / per-tile max", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle("I(r, z) around the focus, CN-BPM relief + RS-I "
                     "propagation — %s" % cfg["name"],
                     fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_rz_tiles_bpm.png"),
                    bbox_inches="tight")
        print("fig_rz_tiles_bpm.png written (BPM wavelengths: %s nm)"
              % ", ".join(str(t) for t in tlams))

# ------------------------------------------------------------ fig_metrics
lam = np.array(metrics["lam_um"]) * 1000

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
ax = axes[0]
ax.plot(lam, metrics["eff_3fwhm"], "-o", color=C_BLUE, lw=2, ms=4)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("focus efficiency")
ax.set_title("Focus efficiency at z = F", fontsize=9)
ax.set_ylim(0, min(1.0, 1.25 * max(0.4, np.nanmax(metrics["eff_3fwhm"]))))
for x, y in zip(lam, metrics["eff_3fwhm"]):
    if y > 0.3:
        ax.annotate("%d nm" % x, (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color=MUTED)

ax = axes[1]
ax.plot(lam, metrics["fwhm_um"], "-o", color=C_ORANGE, lw=2, ms=4,
        label="designed MDL")
ax.plot(lam, lam / 1000 / (2 * NA), "--", color=MUTED, lw=1.5,
        label="diffraction limit λ/2NA")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("PSF FWHM (µm)")
ax.set_title("Spot size at z = F (where a focus exists)", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_ylim(0, max(3.0, 1.3 * np.nanmax(metrics["fwhm_um"])))
fig.tight_layout()
fig.savefig(os.path.join(run_dir, "fig_metrics.png"), bbox_inches="tight")

# ------------------------------------------------------------ fig_profile
h = m * dh_um
rho = (np.arange(m.size) + 0.5) * ring_um / 1000.0   # mm
R_mm = rho[-1] + 0.5 * ring_um / 1000.0
zoom_mm = 50 * ring_um / 1000.0                      # outermost 50 rings

fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.2), sharey=True)
axes[0].plot(rho, h, color=C_TEAL, lw=0.4)
axes[0].set_title("Designed height profile h(ρ) — full aperture (%s)"
                  % cfg["name"], fontsize=9)
axes[0].set_ylabel("height (µm)")
sel = rho > (R_mm - zoom_mm)
axes[1].step(rho[sel] * 1000, h[sel], color=C_TEAL, lw=1.2, where="mid")
axes[1].set_title("Zoom: outermost %.0f µm (%.2f µm rings, %g nm levels)"
                  % (zoom_mm * 1000, ring_um, dh_um * 1000), fontsize=9)
axes[1].set_xlabel("radius (µm  /  mm in top panel)")
axes[1].set_ylabel("height (µm)")
fig.tight_layout()
fig.savefig(os.path.join(run_dir, "fig_profile.png"), bbox_inches="tight")

# ------------------------------------------------------------ fig_bpm_psf
# Focal-plane PSF cuts, thin-element vs BPM (optional: only when
# bpm_validate.py has been run on this folder and saved bpm_psf.npz).
# Each panel is one wavelength; both curves share the panel's TE peak
# normalization so the BPM curve directly shows the model error in
# peak height as well as in shape.
bpm_npz = os.path.join(run_dir, "bpm_psf.npz")
if os.path.exists(bpm_npz):
    bp = np.load(bpm_npz)
    r0 = bp["r0grid"]
    blams = sorted(int(k.split("_")[2]) for k in bp.files
                   if k.startswith("I_te_"))
    ncol = min(len(blams), 5)
    nrow = int(np.ceil(len(blams) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(1.9 * ncol + 0.8, 2.4 * nrow),
                             sharex=True, squeeze=False)
    for j, lnm in enumerate(blams):
        ax = axes[j // ncol][j % ncol]
        I_te = bp["I_te_%d" % lnm]
        I_bpm = bp["I_bpm_%d" % lnm]
        norm = I_te.max()
        ax.plot(r0, I_te / norm, color=C_BLUE, lw=1.8,
                label="thin element")
        ax.plot(r0, I_bpm / norm, color=C_ORANGE, lw=1.4, ls="--",
                label="BPM")
        ax.set_title("%d nm" % lnm, fontsize=9)
        ax.set_xlim(0, r0[-1])
        ax.set_ylim(0, 1.15)
        if j % ncol == 0:
            ax.set_ylabel("I / TE peak")
        if j // ncol == nrow - 1:
            ax.set_xlabel("r (µm)")
    # hide any unused panels
    for j in range(len(blams), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Focal-plane PSF at z = F: thin-element vs BPM "
                 "through the real relief — %s" % cfg["name"],
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_bpm_psf.png"),
                bbox_inches="tight")

    # BPM on-axis maps (only when bpm_validate.py stored the z-scans;
    # older bpm_psf.npz files without "zgrid" are silently skipped)
    if "zgrid" in bp.files:
        zb = bp["zgrid"] / 1000.0                      # mm
        Mb = np.array([bp["Iz_bpm_%d" % l] for l in blams])
        for tag, Mn_, cbl in (
                ("fig_onaxis_bpm.png", Mb / Mb.max(),
                 "normalized intensity"),
                ("fig_onaxis_perlambda_bpm.png",
                 Mb / Mb.max(axis=1, keepdims=True),
                 "intensity / per-λ max")):
            fig, ax = plt.subplots(figsize=(8.6, 3.2))
            extb = [0, len(blams), zb[0], zb[-1]]
            im = ax.imshow(Mn_.T, aspect="auto", origin="lower",
                           extent=extb, cmap="magma", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(blams)) + 0.5)
            ax.set_xticklabels(["%d" % l for l in blams], fontsize=7)
            ax.axhline(F_mm, color="white", lw=0.8, ls="--", alpha=0.8)
            ax.text(0.1, F_mm + 0.02 * (zb[-1] - zb[0]),
                    "design focus F = %.2f mm" % F_mm, color="white",
                    fontsize=7)
            ax.set_xlabel("wavelength (nm)")
            ax.set_ylabel("on-axis distance z (mm)")
            ax.set_title("On-axis intensity vs z, BPM through the real "
                         "relief — %s" % cfg["name"], fontsize=9)
            ax.grid(False)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label(cbl, fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, tag), bbox_inches="tight")

rel = os.path.relpath(run_dir)
ring = "mdl_rings_%d.txt" % cfg["dll_file_no"]
print("figures saved into %s" % rel)
print("next (Zemax):   copy %s + the DLLs from 02_validation_zemax/dll "
      "into Documents/Zemax/DLL/Surfaces, then run "
      "02_validation_zemax/mdl_zemax_validation.py on the Zemax machine"
      % os.path.join(rel, ring))
print("next (tape-out): python %s --rings %s --out %s"
      % (os.path.join("03_tapeout", "export_gds.py"),
         os.path.join(rel, ring), os.path.join(rel, "mdl.gds")))