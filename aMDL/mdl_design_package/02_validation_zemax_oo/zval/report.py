"""The (deliberately verbose) log as a numbered sequence of stages, and
the figures. All figures fail soft: the npz holds every number."""
import os

import numpy as np


class Log:
    """print with numbered section banners so the log reads as steps."""

    def __init__(self):
        self.n = 0

    def __call__(self, msg=""):
        print(msg)

    def section(self, title, sub=None):
        self.n += 1
        print("")
        print("=" * 78)
        print("%d. %s" % (self.n, title.upper()))
        if sub:
            print("   %s" % sub)
        print("=" * 78)

    def subsection(self, title):
        print("")
        print("--- %s " % title + "-" * max(0, 74 - len(title)))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_huygens_line(path, key, r_um, prof, fwhm, fwhm_ref, ref, fine,
                     fine_step_um, rs_quad, mtf, hmtf, rs_mtf, r_max_um,
                     pupil_samp, dx_um, engine="Huygens"):
    """Radial PSF (engine vs run_verify slice vs sub-ring RS) + MTF
    (Hankel of the engine PSF vs Zemax Huygens MTF vs RS)."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(r_um, prof / max(prof.max(), 1e-30), color="k", lw=2,
            label="%s PSF (hybrid, %d^2 pupil)" % (engine, pupil_samp))
    if ref is not None:
        a1.plot(ref[0], ref[1] / max(ref[1].max(), 1e-30), color="#e6550d",
                lw=1.6, ls="--",
                label="RS focal slice (run_verify, %s)" % rs_quad)
    if fine is not None:
        a1.plot(fine[0], fine[1] / max(fine[1].max(), 1e-30),
                color="#3182bd", lw=1.4, ls=":",
                label="RS, sub-ring quadrature (%.3f um)" % fine_step_um)
    a1.set_xlim(0, r_max_um)
    a1.set_xlabel("r (um)")
    a1.set_ylabel("I / peak")
    a1.set_title("%d nm: radial PSF at z=F  FWHM %.2f um (ref %.2f)"
                 % (key, fwhm, fwhm_ref), fontsize=9)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.25)
    if mtf is not None and rs_mtf is not None:
        fl = rs_mtf["f_lppmm"]
        a2.plot(fl, mtf, color="k", lw=2, label="%s PSF -> Hankel MTF" % engine)
        if hmtf is not None:
            a2.plot(hmtf[0], hmtf[1], color="#31a354", lw=1.4,
                    label="Zemax Huygens MTF (tangential)")
        if "MTF_%d" % key in rs_mtf.files:
            a2.plot(fl, rs_mtf["MTF_%d" % key], color="#e6550d", lw=1.6,
                    ls="--", label="RS design")
        if "MTFdl_%d" % key in rs_mtf.files:
            a2.plot(fl, rs_mtf["MTFdl_%d" % key], color="gray", lw=1.2,
                    ls=":", label="diffraction limit")
        a2.set_xlim(0, fl[-1])
        a2.set_ylim(0, 1)
        a2.set_xlabel("spatial frequency (lp/mm)")
        a2.set_ylabel("MTF")
        a2.set_title("MTF: Hankel of the %s PSF (same window as rs) vs "
                     "Zemax Huygens MTF vs RS" % engine, fontsize=9)
        a2.legend(fontsize=8)
        a2.grid(alpha=0.25)
    else:
        a2.text(0.5, 0.5, "rs/verify_mtf.npz not found\n(run "
                "02_validation_rs\\mtf_verify.py)", ha="center",
                va="center", transform=a2.transAxes)
    fig.suptitle("Huygens PSF on the hybrid (Paraxial + cell-averaged "
                 "residual), pupil %d^2, image %.2f um/sample"
                 % (pupil_samp, dx_um), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_rz_tiles(out_dir, store, lams_nm, r0grid, zgrid, bfd_mm, log):
    """I(r,z) tiles per line + on-axis map, same conventions as the
    rs/ and bpm/ tiles."""
    plt = _plt()
    n = len(lams_nm)
    if not n:
        return
    zz_mm = zgrid / 1000.0
    fig, axes = plt.subplots(1, n, figsize=(0.95 * n + 0.9, 3.0),
                             sharey=True, squeeze=False)
    for j, lnm in enumerate(lams_nm):
        ax = axes[0][j]
        M = store["I_%d" % lnm] / store["I_%d" % lnm].max()
        sym = np.concatenate([M[:, ::-1], M[:, 1:]], axis=1)
        im = ax.imshow(sym, aspect="auto", origin="lower",
                       extent=[-r0grid[-1], r0grid[-1], zz_mm[0], zz_mm[-1]],
                       cmap="OrRd", vmin=0, vmax=1)
        ax.axhline(bfd_mm, color="black", lw=0.7, ls="--", alpha=0.7)
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
    fig.suptitle("I(r, z) around the focus -- ZEMAX-traced field (batch "
                 "OPD through the zone DLL) + RS-I propagation",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_zemax_rz_tiles.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.62 * n + 2.6, 3.2))
    Miz = np.array([store["I_%d" % l][:, 0] for l in lams_nm])
    mx = Miz.max(axis=1, keepdims=True)
    mx[mx == 0] = 1.0
    ax.imshow((Miz / mx).T, aspect="auto", origin="lower",
              extent=[0, n, zz_mm[0], zz_mm[-1]], cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(["%d" % l for l in lams_nm], fontsize=7)
    ax.axhline(bfd_mm, color="white", lw=0.8, ls="--", alpha=0.8)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("On-axis intensity vs z -- Zemax-traced field (per-λ norm)",
                 fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_zemax_onaxis_perlambda.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("figures -> fig_zemax_rz_tiles.png, fig_zemax_onaxis_perlambda.png "
        "(in zemax\\)")
