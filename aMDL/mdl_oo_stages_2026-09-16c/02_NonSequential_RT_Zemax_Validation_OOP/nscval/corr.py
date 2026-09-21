"""corr -- the ladder's ratios folded into the design's efficiency table.

The design (01_design_oo, MdlProblem.apply_efficiency) takes an npz with
exactly lam_um (K,), r_um (Z,) and eta (K, Z) = eta_rigorous / eta_scalar
of the local grating into its focusing order, interpolated onto its own
(lam, rho) grid with edge clamping, and weights the ring phasors by
sqrt(eta). The ladder measures that ratio for the local staircase at a
few slopes sin(theta) = fractions of the rim NA; the radius of a slope is

    r = F sin(theta) / sqrt(1 - sin^2(theta)).

Per (case, line) the ratio used is the design-order ratio eta_RCWA(p0) /
eta_ref(p0) when the staircase's design order is well defined (p >=
p_min, default 0.75 waves) AND carries a meaningful share of the power
(eta_ref(p0) >= ref_min, default 0.25: a ratio of two small numbers is
noise -- N3 at 0.4 um read 1.255 on a reference of 0.104), and the
window-sum ratio otherwise (a short staircase whose p is below half a
wave has no single order at the focusing direction; the sum over its
orders is the honest measure).
Cases at the same slope (N' = 3, 4, 6) are averaged, lines without data
are interpolated over wavelength, r = 0 gets 1.0 (a flat tread: the
reference already carries the Fresnel loss), and the table is clipped
to [0.5, 1.2].

Outputs: efficiency_corr.npz (the design's contract), efficiency_corr.json
(the same with provenance), fig_efficiency_corr.png. Run it as
`mdl_nsc_validation.py corr runs\\<run> [--ladder <folder>]` on an existing
ladder folder (no OpticStudio needed); the ladder also writes it itself.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import NscAnalysis, RunContext


def latest_ladder_dir(run_dir: str) -> Optional[str]:
    """The newest <run>/nsc/*_ladder folder that holds a ladder.npz."""
    cands = sorted(glob.glob(os.path.join(run_dir, "nsc", "*_ladder")))
    cands = [c for c in cands if os.path.exists(os.path.join(c, "ladder.npz"))]
    return cands[-1] if cands else None


def slope_radius_um(slope: float, focal_um: float) -> float:
    """r of the ring whose local blaze deflects by sin(theta) = slope."""
    return focal_um * slope / math.sqrt(max(1e-12, 1.0 - slope * slope))


def correction_from_ladder(ladder_npz: str, focal_um: float, p_min: float = 0.75,
                           clip: Tuple[float, float] = (0.5, 1.2), log: Any = print,
                           ref_min: float = 0.25) -> Dict[str, Any]:
    """Build the (lam_um, r_um, eta) table from a ladder.npz. Returns a dict
    with the arrays and the per-(slope, line) provenance."""
    z = np.load(ladder_npz)
    lams = np.asarray(z["lams_um"], float)
    slopes = np.asarray(z["slopes"], float)
    n_treads = np.asarray(z["n_treads"], int)
    p = np.asarray(z["p"], float)
    ratio0 = np.asarray(z["ratio_p0"], float)
    ratio_sum = np.asarray(z["ratio_sum"], float)
    refused = np.asarray(z["refused"], bool) if "refused" in z.files else np.zeros_like(ratio0, dtype=bool)
    if "eta_ref" in z.files and "order_window" in z.files:
        ref0 = np.asarray(z["eta_ref"], float)[:, :, int(z["order_window"])]
    else:
        ref0 = np.full_like(ratio0, 1.0)
    uniq = sorted(set(float(s) for s in slopes))
    used: List[Dict[str, Any]] = []
    table = np.full((len(lams), len(uniq)), np.nan)
    for js, s in enumerate(uniq):
        rows = [i for i in range(len(slopes)) if abs(float(slopes[i]) - s) < 1e-12]
        for il in range(len(lams)):
            vals: List[float] = []
            for i in rows:
                if refused[i, il]:
                    continue
                r0, rs, pp = ratio0[i, il], ratio_sum[i, il], p[i, il]
                if pp >= p_min and ref0[i, il] >= ref_min and np.isfinite(r0) and r0 > 0.0:
                    vals.append(float(r0))
                    kind = "p0"
                elif np.isfinite(rs) and rs > 0.0:
                    vals.append(float(rs))
                    kind = "sum"
                else:
                    continue
                used.append({"slope": s, "lam_um": float(lams[il]), "n_treads": int(n_treads[i]),
                             "p": float(pp), "ref_p0": float(ref0[i, il]), "kind": kind, "ratio": vals[-1]})
            if vals:
                table[il, js] = float(np.mean(vals))
        col = table[:, js]
        ok = np.isfinite(col)
        if ok.sum() == 0:
            raise SystemExit("no usable ladder line at slope %.4f" % s)
        if ok.sum() < len(lams):
            table[~ok, js] = np.interp(lams[~ok], lams[ok], col[ok])
    table = np.clip(table, clip[0], clip[1])
    r_um = np.array([0.0] + [slope_radius_um(s, focal_um) for s in uniq])
    eta = np.hstack([np.ones((len(lams), 1)), table])
    log("  slope -> radius: " + ", ".join("%.4f -> %.0f um" % (s, r) for s, r in zip(uniq, r_um[1:])))
    log("  lam(um) | " + "  ".join("r=%5.0f" % r for r in r_um))
    for il, lam in enumerate(lams):
        log("  %.3f   | " % lam + "  ".join("%7.3f" % v for v in eta[il]))
    return {"lam_um": lams, "r_um": r_um, "eta": eta, "used": used, "slopes": uniq,
            "p_min": p_min, "ref_min": ref_min, "clip": clip, "source": os.path.abspath(ladder_npz)}


def write_correction(out_dir: str, corr: Dict[str, Any], log: Any = print) -> str:
    path = os.path.join(out_dir, "efficiency_corr.npz")
    np.savez(path, lam_um=corr["lam_um"], r_um=corr["r_um"], eta=corr["eta"])
    with open(os.path.join(out_dir, "efficiency_corr.json"), "w") as fh:
        json.dump({"lam_um": corr["lam_um"].tolist(), "r_um": corr["r_um"].tolist(),
                   "eta": corr["eta"].tolist(), "slopes": corr["slopes"], "p_min": corr["p_min"],
                   "ref_min": corr["ref_min"],
                   "clip": list(corr["clip"]), "source": corr["source"], "used": corr["used"]},
                  fh, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for j, r in enumerate(corr["r_um"]):
            ax.plot(1000 * corr["lam_um"], corr["eta"][:, j], marker="o", ms=3, label="r = %.0f um" % r)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("eta_rigorous / eta_scalar")
        ax.set_title("efficiency correction for the design (apply_efficiency)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_efficiency_corr.png"), dpi=130)
        plt.close(fig)
    except Exception as exc:                        # pragma: no cover
        log("figure skipped (%s)" % exc)
    log("wrote %s (set config efficiency_corr_npz to this path and re-run the design)" % path)
    return path


class LadderCorrection(NscAnalysis):
    """CLI mode `corr`: rebuild the table from an existing ladder folder."""
    MODE = "corr"
    DESCRIPTION = "ladder ratios -> efficiency_corr.npz for the design (no OpticStudio)"

    def __init__(self, ctx: RunContext, gui: bool = False, app: Any = None,
                 overrides: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ctx, gui, app, overrides)
        self.ladder_dir: Optional[str] = self.overrides.get("ladder_dir") or (
            latest_ladder_dir(ctx.run_dir) if ctx.run_dir else None)
        self.p_min = float(self.overrides.get("p_min", 0.75))

    def settings_record(self) -> Dict[str, Any]:
        return {"ladder_dir": self.ladder_dir, "p_min": self.p_min}

    def connect(self) -> Any:                        # no session needed
        return None

    def run(self) -> None:
        if not self.ladder_dir:
            raise SystemExit("no ladder folder with ladder.npz under the run's nsc\\; pass --ladder <folder>")
        F = float(self.ctx.cfg.get("focal_um") or 0.0)
        if F <= 0.0:
            raise SystemExit("the run's config.json has no focal_um")
        self.log.section("correction table", "from %s" % self.ladder_dir)
        corr = correction_from_ladder(os.path.join(self.ladder_dir, "ladder.npz"), F, self.p_min, log=self.log)
        write_correction(self.out_dir, corr, log=self.log)
