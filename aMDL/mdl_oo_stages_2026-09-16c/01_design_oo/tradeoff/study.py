"""The two figures of the paper's Fig. 1 as run-folder studies.

* PairMapStudy  -- Fig. 1b/1c: the pairwise coherence map
  max Re J_w(rho1, rho2) for a list of (D, H) panels at fixed NA, band,
  material (paper: S1-S3 = D 1/3/10 mm at H 15 um; S3-S5 = H 15/5/1 um
  at D 10 mm).
* SweepStudy    -- Fig. 1d: the ceiling max J_w(F) over a (D, H) grid,
  with the reference points as stars.

Both write into runs/<stamp>_tradeoff_<name>/: config.json, the npz
tables, the figures, and a summary text. Presets PAPER_FIG1 and
SWIR_TRADEOFF reproduce the 2026-08-24 studies (ceilings at 64 x 192
sweep accuracy: S1 0.52, S2 0.21, S3 0.069, S4 0.028, S5 0.023).
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .space import REFERENCE_POINTS, Ceiling, LensSpec

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class StudyConfig:
    """A trade-off study: the fixed spec (NA/F#, band, dh, DELTA, material)
    and what is swept."""
    name: str
    base: LensSpec                                  # D and H of base are ignored
    pair_panels: List[Tuple[float, float]] = field(default_factory=list)   # (D_mm, H_um)
    sweep_d_mm: List[float] = field(default_factory=list)
    sweep_h_um: List[float] = field(default_factory=list)
    stars: Dict[str, Tuple[float, float]] = field(default_factory=dict)   # name -> (D, H)
    pair_n_rho: int = 192
    pair_n_wavelengths: int = 384
    sweep_n_rho: int = 64
    sweep_n_wavelengths: int = 192
    runs_dir: str = "runs"

    def spec_at(self, d_mm: float, h_um: float) -> LensSpec:
        return replace(self.base, diameter_mm=float(d_mm), h_max_um=float(h_um),
                       name="%s_D%g_H%g" % (self.name, d_mm, h_um))

    def to_json(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["base"] = self.base.to_dict()
        return d


_S3 = LensSpec(10.24, 0.40, 1.10, h_max_um=15.0, dh_um=0.078, ring_width_um=2.0,
               na=0.1, material="AZ4562", name="paper_fig1")

PAPER_FIG1 = StudyConfig(
    name="paper_fig1", base=_S3,
    pair_panels=[(1.0, 15.0), (3.0, 15.0), (10.0, 15.0), (10.0, 5.0), (10.0, 1.0)],
    sweep_d_mm=list(np.round(np.geomspace(0.5, 50.0, 22), 3)),
    sweep_h_um=list(np.round(np.geomspace(0.5, 60.0, 26), 3)),
    stars={k: (v[0], v[1]) for k, v in REFERENCE_POINTS.items()})

SWIR_TRADEOFF = StudyConfig(
    name="swir_tradeoff",
    base=replace(_S3, lam_min_um=1.10, lam_max_um=1.80, name="swir"),
    pair_panels=[(10.24, 15.0), (10.24, 30.0), (10.24, 45.0)],
    sweep_d_mm=list(np.round(np.geomspace(1.0, 50.0, 18), 3)),
    sweep_h_um=list(np.round(np.geomspace(2.0, 60.0, 18), 3)),
    stars={"H15": (10.24, 15.0), "H30": (10.24, 30.0), "H45": (10.24, 45.0)})

PRESETS: Dict[str, StudyConfig] = {"PAPER_FIG1": PAPER_FIG1, "SWIR_TRADEOFF": SWIR_TRADEOFF}


class StudyFolder:
    """runs/<stamp>_tradeoff_<name>/ with json / npz / png writers."""

    def __init__(self, cfg: StudyConfig, pkg_root: str) -> None:
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(pkg_root, cfg.runs_dir, "%s_tradeoff_%s" % (self.stamp, cfg.name))
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "config.json"), "w") as fh:
            json.dump(cfg.to_json(), fh, indent=1)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)


class PairMapStudy:
    """Fig. 1b/1c panels: one pairwise coherence map per (D, H)."""

    def __init__(self, cfg: StudyConfig, folder: StudyFolder, log: LogFn = print) -> None:
        self.cfg, self.folder, self.log = cfg, folder, log
        self.maps: List[Tuple[LensSpec, np.ndarray, np.ndarray, float]] = []

    def run(self) -> List[Tuple[LensSpec, np.ndarray, np.ndarray, float]]:
        for d_mm, h_um in self.cfg.pair_panels:
            spec = self.cfg.spec_at(d_mm, h_um)
            t0 = time.time()
            c = Ceiling.compute(spec, numeric=True, n_rho=self.cfg.pair_n_rho,
                                n_wavelengths=self.cfg.pair_n_wavelengths)
            rho, B = c.pair_map(self.cfg.pair_n_rho, self.cfg.pair_n_wavelengths)
            assert c.numeric is not None
            self.maps.append((spec, rho, B, c.numeric))
            self.log("pair map D=%g mm H=%g um: red area (B > 0.9) %.1f %% of pairs, ceiling "
                     "%.4f (analytic %.4f) [%.1fs]"
                     % (d_mm, h_um, 100 * np.mean(B > 0.9), c.numeric, c.analytic, time.time() - t0))
        arrays: Dict[str, Any] = {"B_D%g_H%g" % (s.diameter_mm, s.h_max_um): B
                                  for s, _, B, _ in self.maps}
        arrays.update(rho_norm=self.maps[0][1] if self.maps else np.zeros(0),
                      ceilings=np.array([m[3] for m in self.maps]),
                      panels=np.array(self.cfg.pair_panels))
        np.savez(self.folder.path("pair_maps.npz"), **arrays)
        self.figure()
        return self.maps

    def figure(self) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:                          # pragma: no cover
            return None
        n = len(self.maps)
        fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 1.2, 3.6), squeeze=False,
                                 sharey=True, constrained_layout=True)
        for ax, (spec, rho, B, J) in zip(axes[0], self.maps):
            im = ax.imshow(B, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1, cmap="jet")
            ax.set_title("D %g mm, H %g um\nmax J(F) <= %.3f" % (spec.diameter_mm, spec.h_max_um, J),
                         fontsize=8)
            ax.set_xlabel("rho1 / R")
        axes[0, 0].set_ylabel("rho2 / R")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="max Re J(rho1, rho2)")
        b = self.cfg.base
        fig.suptitle("pairwise coherence ceilings, NA %.3f, %.0f-%.0f nm, %s (paper Fig. 1b/c)"
                     % (b.NA, 1000 * b.lam_min_um, 1000 * b.lam_max_um, b.material), fontsize=9)
        path = self.folder.path("fig_pair_maps.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


class SweepStudy:
    """Fig. 1d: ceiling over the (D, H) grid, stars at the references."""

    def __init__(self, cfg: StudyConfig, folder: StudyFolder, log: LogFn = print,
                 progress: Optional[Callable[[int, int], None]] = None) -> None:
        self.cfg, self.folder, self.log, self.progress = cfg, folder, log, progress
        self.J: np.ndarray = np.zeros((len(cfg.sweep_h_um), len(cfg.sweep_d_mm)))
        self.J_analytic: np.ndarray = np.zeros_like(self.J)
        self.star_values: Dict[str, float] = {}

    def run(self) -> np.ndarray:
        cfg = self.cfg
        total = self.J.size
        k = 0
        t0 = time.time()
        for i, h in enumerate(cfg.sweep_h_um):
            for j, d in enumerate(cfg.sweep_d_mm):
                spec = cfg.spec_at(d, h)
                c = Ceiling.compute(spec, numeric=True, n_rho=cfg.sweep_n_rho,
                                    n_wavelengths=cfg.sweep_n_wavelengths)
                assert c.numeric is not None
                self.J[i, j], self.J_analytic[i, j] = c.numeric, c.analytic
                k += 1
                if self.progress:
                    self.progress(k, total)
            self.log("H = %6.2f um: ceiling %.3f (D %g) .. %.3f (D %g)  [%.0fs]"
                     % (h, self.J[i, 0], cfg.sweep_d_mm[0], self.J[i, -1], cfg.sweep_d_mm[-1],
                        time.time() - t0))
        for name, (d, h) in cfg.stars.items():
            c = Ceiling.compute(cfg.spec_at(d, h), numeric=True, n_rho=cfg.sweep_n_rho,
                                n_wavelengths=cfg.sweep_n_wavelengths)
            assert c.numeric is not None
            self.star_values[name] = c.numeric
            self.log("star %s (D %g mm, H %g um): ceiling %.4f" % (name, d, h, c.numeric))
        np.savez(self.folder.path("sweep.npz"), d_mm=np.array(cfg.sweep_d_mm),
                 h_um=np.array(cfg.sweep_h_um), J_ceiling=self.J, J_analytic=self.J_analytic,
                 star_names=np.array(list(cfg.stars)), star_d_mm=np.array([v[0] for v in cfg.stars.values()]),
                 star_h_um=np.array([v[1] for v in cfg.stars.values()]),
                 star_J=np.array([self.star_values[n] for n in cfg.stars]))
        self.figure()
        return self.J

    def figure(self) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:                          # pragma: no cover
            return None
        cfg = self.cfg
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        D, H = np.meshgrid(cfg.sweep_d_mm, cfg.sweep_h_um)
        levels = np.linspace(0, 1, 21)
        cf = ax.contourf(D, H, self.J, levels=levels, cmap="jet")
        ax.contour(D, H, self.J, levels=[0.05, 0.1, 0.2, 0.4, 0.7], colors="k", linewidths=0.6)
        for name, (d, h) in cfg.stars.items():
            ax.plot(d, h, marker="*", ms=12, color="w", mec="k")
            ax.annotate("%s %.2f" % (name, self.star_values.get(name, float("nan"))), (d, h),
                        textcoords="offset points", xytext=(6, 4), fontsize=7, color="w")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("diameter D (mm)")
        ax.set_ylabel("relief height H (um)")
        b = cfg.base
        ax.set_title("ceiling max J_w(F), NA %.3f, %.0f-%.0f nm, %s (paper Fig. 1d)"
                     % (b.NA, 1000 * b.lam_min_um, 1000 * b.lam_max_um, b.material), fontsize=9)
        fig.colorbar(cf, ax=ax, label="max J_w(F)")
        fig.tight_layout()
        path = self.folder.path("fig_sweep.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def run_study(cfg: StudyConfig, pkg_root: str, log: LogFn = print, pair: bool = True,
              sweep: bool = True) -> str:
    folder = StudyFolder(cfg, pkg_root)
    log("trade-off study '%s' -> %s" % (cfg.name, os.path.relpath(folder.dir)))
    for line in cfg.base.describe():
        log("  " + line)
    if pair and cfg.pair_panels:
        PairMapStudy(cfg, folder, log).run()
    if sweep and cfg.sweep_d_mm and cfg.sweep_h_um:
        SweepStudy(cfg, folder, log).run()
    return folder.dir


def sequence(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]
