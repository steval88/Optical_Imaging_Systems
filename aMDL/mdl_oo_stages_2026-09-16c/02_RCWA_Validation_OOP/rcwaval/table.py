"""table -- a robust efficiency_corr.npz from a finished (or partial) zone sweep.

The raw per-cell ratio eta_RCWA / eta_scalar is exact for the cell but a
poor input for the design: where the scalar efficiency is small the ratio
is the quotient of two small numbers (3.7 on a reference of 0.009, 2.6 on
0.11 in the S3 sweep of 2026-09-21), and the optimizer's zones are
irregular enough that neighbouring zones scatter around the physical
trend. The design's apply_efficiency needs that trend -- a slowly varying
function of radius and wavelength (the fold depth, the local period and
the riser all vary smoothly with r) -- not the cell noise. This module:

    1. keeps a cell only where eta_scalar >= eta_min (default 0.10) and the
       energy balance closed (sum R + T within 1e-3 of 1);
    2. clips the surviving ratios to [clip_lo, clip_hi] (default 0.3 .. 1.3);
    3. fills the (line, zone) gaps by interpolation over wavelength within a
       zone, then over radius across zones;
    4. smooths over radius with a running median of `smooth` solved zones
       (default 5), per line;
    5. writes the table on the design's contract: an r = 0 anchor at 1.0 and
       the smoothed solved zones. The unsolved wide zones (P > p_max) are NOT
       written as 1.0 between solved neighbours -- that made a comb of spikes
       in the first version; the design interpolates them in r from their
       solved neighbours, which overstates their (smaller) fold loss a little
       and is the conservative side. Inside the first solved radius the
       interpolation runs to the anchor, i.e. the paraxial core is 1.0.

Every threshold is echoed and stored in run_info.json; the raw ratio map
and the smoothed map are drawn side by side (fig_corr_table.png) so the
smoothing can be judged, and the per-line mean of the table (weighted by
the zone widths) is printed -- the number the design's J will move by, to
first order.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .zonesweep import ZoneRecord, focusing_order, load_zone_table


def running_median(values: np.ndarray, width: int) -> np.ndarray:
    """Running median over `width` points (odd), edges shrunk; nan-aware."""
    n = values.size
    half = max(0, int(width) // 2)
    out = np.full(n, np.nan)
    for i in range(n):
        seg = values[max(0, i - half): min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


class CorrTable:
    """Build the design table from a sweep folder.

    Attributes
    ----------
    run_dir, sweep_dir   the run folder and its rcwa/<stamp>_zones folder
    zones, lams, n_real  the zone table
    focal_um, p_max_um   from config.json / the sweep's run_info.json
    eta_min, clip, smooth, balance_tol   the thresholds above
    """

    def __init__(self, run_dir: str, sweep_dir: str, eta_min: float = 0.10,
                 clip: Tuple[float, float] = (0.3, 1.3), smooth: int = 5, balance_tol: float = 1e-3,
                 log: Any = print) -> None:
        self.run_dir, self.sweep_dir, self.log = run_dir, sweep_dir, log
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        self.focal_um = float(cfg.get("focal_um") or cfg["derived"]["focal_um"])
        self.zones, self.step_um, self.lams, self.n_real = load_zone_table(os.path.join(run_dir, "zone_table.npz"))
        info_path = os.path.join(sweep_dir, "run_info.json")
        info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        self.p_max_um = float(info.get("settings", {}).get("p_max_um", 60.0))
        with open(os.path.join(sweep_dir, "zone_sweep_results.json")) as fh:
            self.results: Dict[str, Dict[str, Any]] = json.load(fh)
        self.eta_min, self.clip, self.smooth, self.balance_tol = float(eta_min), clip, int(smooth), float(balance_tol)

    def raw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(eta_rcwa, eta_scalar, ratio, balance) per (line, zone); nan = not solved."""
        K, Z = self.lams.size, len(self.zones)
        eta_r = np.full((K, Z), np.nan)
        eta_s = np.full((K, Z), np.nan)
        bal = np.full((K, Z), np.nan)
        index = {z.zone_id: i for i, z in enumerate(self.zones)}
        for key, res in self.results.items():
            zid, il = (int(v) for v in key.split(":"))
            zi = index[zid]
            lam, n = float(self.lams[il]), float(self.n_real[il])
            eta_r[il, zi] = res["eta_rcwa"]
            eta_s[il, zi] = focusing_order(self.zones[zi], lam, n, self.focal_um)[2]
            bal[il, zi] = res["sum_T"] + res["sum_R"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(eta_r) & (eta_s > 0), eta_r / np.maximum(eta_s, 1e-12), np.nan)
        return eta_r, eta_s, ratio, bal

    def build(self) -> Dict[str, Any]:
        log = self.log
        eta_r, eta_s, ratio_raw, bal = self.raw()
        K, Z = ratio_raw.shape
        solved = np.isfinite(ratio_raw)
        keep = solved & (eta_s >= self.eta_min) & (np.abs(bal - 1.0) <= self.balance_tol)
        log("cells solved %d; kept %d (eta_scalar >= %.2f and |R + T - 1| <= %.0e); dropped %d weak, %d unbalanced"
            % (int(solved.sum()), int(keep.sum()), self.eta_min, self.balance_tol,
               int((solved & (eta_s < self.eta_min)).sum()),
               int((solved & (eta_s >= self.eta_min) & (np.abs(bal - 1.0) > self.balance_tol)).sum())))
        ratio = np.where(keep, np.clip(ratio_raw, self.clip[0], self.clip[1]), np.nan)
        n_clip = int((keep & ((ratio_raw < self.clip[0]) | (ratio_raw > self.clip[1]))).sum())
        log("kept ratios: %.3f-%.3f, median %.3f; %d clipped to [%.2f, %.2f]"
            % (np.nanmin(ratio_raw[keep]), np.nanmax(ratio_raw[keep]), np.nanmedian(ratio_raw[keep]), n_clip,
               self.clip[0], self.clip[1]))
        # zones with at least one kept cell, in radius order
        r_all = np.array([z.r_center for z in self.zones])
        zi_kept = [zi for zi in range(Z) if np.isfinite(ratio[:, zi]).any()]
        zi_kept.sort(key=lambda i: r_all[i])
        if not zi_kept:
            raise SystemExit("no cell survives the thresholds")
        # 3. gaps: over wavelength within a zone, then over radius across zones
        filled = np.full((K, len(zi_kept)), np.nan)
        for j, zi in enumerate(zi_kept):
            col = ratio[:, zi]
            ok = np.isfinite(col)
            filled[:, j] = np.interp(self.lams, self.lams[ok], col[ok]) if ok.sum() < K else col
        r_kept = r_all[zi_kept]
        # 4. running median over radius, per line
        smoothed = np.vstack([running_median(filled[il], self.smooth) for il in range(K)])
        # 5. the design table: the r = 0 anchor and the solved zones
        cols: List[np.ndarray] = [np.ones(K)]
        radii: List[float] = [0.0]
        kinds: List[str] = ["anchor"]
        for j, zi in enumerate(zi_kept):
            cols.append(smoothed[:, j])
            radii.append(float(r_kept[j]))
            kinds.append("solved")
        order = np.argsort(radii)
        eta = np.stack(cols, axis=1)[:, order]
        r_um = np.asarray(radii)[order]
        # the width-weighted mean per line over the solved zones: J moves ~ by this
        widths = np.array([self.zones[zi].period_um for zi in zi_kept])
        w_mean = (smoothed * widths[None, :]).sum(axis=1) / widths.sum()
        log("  lam(um)  raw median   smoothed median   width-weighted mean (solved zones)")
        for il, lam in enumerate(self.lams):
            rr = ratio_raw[il][keep[il]]
            log("  %.3f     %.3f          %.3f              %.3f"
                % (lam, np.median(rr) if rr.size else float("nan"), np.nanmedian(smoothed[il]), w_mean[il]))
        n_wide = sum(1 for zi, z in enumerate(self.zones) if z.period_um > self.p_max_um and zi not in zi_kept)
        log("table: %d lines x %d radii (r = 0 anchor + %d solved zones smoothed over %d; %d wide zones with "
            "P > %.0f um left to the design's interpolation in r); eta %.3f-%.3f, band mean of the "
            "width-weighted means %.3f"
            % (K, r_um.size, len(zi_kept), self.smooth, n_wide, self.p_max_um, eta.min(), eta.max(),
               float(w_mean.mean())))
        return {"lam_um": self.lams, "r_um": r_um, "eta": eta, "kinds": [kinds[i] for i in order],
                "ratio_raw": ratio_raw, "ratio_kept": ratio, "eta_rcwa": eta_r, "eta_scalar": eta_s,
                "balance": bal, "zone_r_um": r_all, "r_kept": r_kept, "smoothed": smoothed, "filled": filled,
                "w_mean": w_mean}

    def write(self, out_dir: str, t: Dict[str, Any]) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "efficiency_corr.npz")
        np.savez(path, lam_um=t["lam_um"], r_um=t["r_um"], eta=t["eta"])
        np.savez(os.path.join(out_dir, "corr_table.npz"), lam_um=t["lam_um"], r_um=t["r_um"], eta=t["eta"],
                 kinds=np.array(t["kinds"]), ratio_raw=t["ratio_raw"], ratio_kept=t["ratio_kept"],
                 eta_rcwa=t["eta_rcwa"], eta_scalar=t["eta_scalar"], balance=t["balance"],
                 zone_r_um=t["zone_r_um"], r_kept=t["r_kept"], smoothed=t["smoothed"], w_mean=t["w_mean"],
                 eta_min=self.eta_min, clip=np.array(self.clip), smooth=self.smooth, p_max_um=self.p_max_um)
        self.log("wrote %s (config efficiency_corr_npz)" % path)
        self.figure(out_dir, t)
        return path

    def figure(self, out_dir: str, t: Dict[str, Any]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:                        # pragma: no cover
            self.log("figure skipped (%s)" % exc)
            return
        lam_nm = 1000 * t["lam_um"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        ext = [t["zone_r_um"].min(), t["zone_r_um"].max(), lam_nm.min(), lam_nm.max()]
        im = axes[0].imshow(t["ratio_raw"], aspect="auto", origin="lower", vmin=0.3, vmax=1.3, cmap="RdBu_r", extent=ext)
        axes[0].set_title("raw eta_RCWA / eta_scalar per zone", fontsize=9)
        ext2 = [t["r_kept"].min(), t["r_kept"].max(), lam_nm.min(), lam_nm.max()]
        axes[1].imshow(t["smoothed"], aspect="auto", origin="lower", vmin=0.3, vmax=1.3, cmap="RdBu_r", extent=ext2)
        axes[1].set_title("kept (eta_scalar >= %.2f), clipped, median over %d zones" % (self.eta_min, self.smooth), fontsize=9)
        for ax in axes[:2]:
            ax.set_xlabel("zone centre radius (um)")
            ax.set_ylabel("wavelength (nm)")
        fig.colorbar(im, ax=axes[:2], fraction=0.03)
        ax = axes[2]
        for il in range(0, t["lam_um"].size, max(1, t["lam_um"].size // 6)):
            ax.plot(t["r_um"], t["eta"][:, :][il], marker=".", ms=3, label="%.0f nm" % lam_nm[il])
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("radius (um)")
        ax.set_ylabel("corr for the design")
        ax.set_title("the table apply_efficiency reads", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(out_dir, "fig_corr_table.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        self.log("saved fig_corr_table.png")


def latest_sweep_dir(run_dir: str) -> Optional[str]:
    import glob
    cands = sorted(glob.glob(os.path.join(run_dir, "rcwa", "*_zones")))
    cands = [c for c in cands if os.path.exists(os.path.join(c, "zone_sweep_results.json"))]
    return cands[-1] if cands else None


__all__ = ["CorrTable", "latest_sweep_dir", "running_median", "ZoneRecord"]
