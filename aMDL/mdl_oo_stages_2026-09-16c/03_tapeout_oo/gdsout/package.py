"""The tape-out package of a run: everything a foundry / litho engineer
needs about the designed achromatic MDL, in one time-stamped folder

    <run>/tapeout/<YYYYMMDD_HHMMSS>_<mode>/
        <name>.gds                 the layout (gdsout.export)
        <name>.gds.layers.csv      gds_layer -> gray level -> height (dose map)
        height_map.csv             ONE ROW PER RING: index, r_in, r_out,
                                   gray level, height, gds layer (index mode)
        height_map.npz             the same as arrays (+ the annuli)
        height_profile.png         h(rho) staircase, full aperture + rim zoom
        mdl_rings_<n>.txt          the ring table as exported (byte copy)
        m_final.npy                the gray-level vector of the design
        design_config.json         the run's config.json (geometry, band,
                                   objective, fabrication quanta)
        tapeout_info.json          provenance + the design in numbers:
                                   D, F, NA, band, target comb, N, M,
                                   dh, DELTA, H_max, table SHA256, J at
                                   every pipeline stage, and the RS
                                   verification metrics per wavelength
                                   when stage 2a has run (rs/)
        fabrication_spec.txt       THE SHEET FOR THE FOUNDRY (gdsout.spec):
                                   optical function, material + index
                                   model, relief geometry and tolerances,
                                   layout encoding + dose map, verified
                                   performance, file checksums, and the
                                   full per-ring height map
        README.txt                 file index

The GDS is built from the run's own ring table (the file the Zemax DLLs
read and run_verify.py re-exports from the verified vector), so the
layout, the height map and the validation numbers all describe the same
design.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import __version__
from .encode import Encoding
from .export import ExportSettings, GdsExport
from .rings import RingTable
from .spec import FabricationSpec, MaterialSpec

LogFn = Callable[[str], None]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TapeoutPackage:
    """Build the tape-out folder of one run.

    Attributes
    ----------
    run_dir    the run folder (config.json, m_final.npy, mdl_rings_<n>.txt,
               optionally design_metrics.json and rs/verify_metrics.json)
    cfg        the run's config.json dict
    table      RingTable (dh from config.json unless overridden)
    encoding   Encoding (index | terrace)
    stamp      time stamp of this package
    out_dir    <run>/tapeout/<stamp>_<mode>
    files      what was written: {label: path}
    """

    def __init__(self, run_dir: str, encoding: Encoding, dh_um: Optional[float] = None,
                 cell: str = "MDL", tol_um: float = 0.02, precision_m: float = 1e-9,
                 material: Optional[MaterialSpec] = None,
                 log: Optional[LogFn] = None) -> None:
        self.run_dir: str = run_dir.rstrip("/\\")
        with open(os.path.join(self.run_dir, "config.json")) as fh:
            self.cfg: Dict[str, Any] = json.load(fh)
        self.table: RingTable = RingTable.from_run(self.run_dir, dh_um)
        self.encoding: Encoding = encoding
        self.stamp: str = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir: str = os.path.join(self.run_dir, "tapeout",
                                         "%s_%s" % (self.stamp, encoding.NAME))
        self.name: str = str(self.cfg["name"])
        self.settings = ExportSettings(
            out=os.path.join(self.out_dir, "%s.gds" % self.name),
            cell=cell, tol_um=tol_um, precision_m=precision_m)
        self.material: MaterialSpec = material or MaterialSpec.design_default()
        self.log: LogFn = log or print
        self.files: Dict[str, str] = {}
        self.t0 = time.time()

    # -- pieces -------------------------------------------------------------------
    def gds(self) -> GdsExport:
        ex = GdsExport(self.table, self.encoding, self.settings, self.log)
        ex.describe()
        self.files["gds"] = ex.write()
        self.files["layer_map"] = self.settings.layer_map_path
        self._export = ex
        return ex

    def height_map(self) -> str:
        """height_map.csv / .npz: one row per ring i -- r_in = i DELTA,
        r_out = (i+1) DELTA, gray level m_i, height h_i = m_i dh, and the
        GDS layer the ring is drawn on (index encoding: m_i, 0 = not
        drawn; terrace encoding: the top layer 1..m_i that covers it)."""
        t = self.table
        m = t.levels()
        idx = np.arange(t.n)
        r_in = idx * t.delta_um
        r_out = (idx + 1) * t.delta_um
        layer = m.copy()
        if self.encoding.NAME == "index" and not self.encoding.draw_zero:
            layer[m == 0] = 0
        path = os.path.join(self.out_dir, "height_map.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ring", "r_in_um", "r_out_um", "gray_level", "height_um",
                        "height_table_um", "gds_layer"])
            for i in range(t.n):
                w.writerow([i, "%.4f" % r_in[i], "%.4f" % r_out[i], int(m[i]),
                            "%.4f" % (m[i] * t.dh_um), "%.6f" % t.h_um[i], int(layer[i])])
        ann = getattr(self, "_export", None)
        annuli = ann.annuli if ann is not None else self.encoding.annuli(m)
        np.savez(os.path.join(self.out_dir, "height_map.npz"),
                 ring=idx, r_in_um=r_in, r_out_um=r_out, gray_level=m,
                 height_um=m * t.dh_um, height_table_um=t.h_um, gds_layer=layer,
                 dh_um=t.dh_um, delta_um=t.delta_um,
                 annulus_layer=np.array([a.layer for a in annuli], dtype=int),
                 annulus_i0=np.array([a.i0 for a in annuli], dtype=int),
                 annulus_i1=np.array([a.i1 for a in annuli], dtype=int))
        self.files["height_map"] = path
        self.files["height_map_npz"] = path[:-4] + ".npz"
        self.log("height map -> %s (%d rings, levels %d..%d, %d annuli)"
                 % (path, t.n, int(m.min()), int(m.max()), len(annuli)))
        return path

    def height_profile_png(self) -> Optional[str]:
        """height_profile.png: the staircase h(rho) over the whole aperture
        and a zoom on the outermost 100 rings."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                      # pragma: no cover
            self.log("height_profile.png skipped (%s)" % e)
            return None
        t = self.table
        edges = np.arange(t.n + 1) * t.delta_um
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6))
        a1.stairs(t.h_um, edges / 1000.0, lw=0.5, color="#2c5aa0")
        a1.set_xlabel("radius  (mm)")
        a1.set_ylabel("height  (um)")
        a1.set_title("%s: ring heights h_i = m_i dh  (N=%d, DELTA=%.2f um, dh=%.3f um, "
                     "%d levels used)" % (self.name, t.n, t.delta_um, t.dh_um,
                                          int(t.levels().max()) + 1), fontsize=9)
        a1.grid(True, alpha=0.25)
        k = min(100, t.n)
        a2.stairs(t.h_um[-k:], edges[-k - 1:], lw=1.0, color="#2c5aa0")
        a2.set_xlabel("radius  (um)  -- outermost %d rings" % k)
        a2.set_ylabel("height  (um)")
        a2.grid(True, alpha=0.25)
        fig.tight_layout()
        path = os.path.join(self.out_dir, "height_profile.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        self.files["height_profile"] = path
        self.log("height profile -> %s" % path)
        return path

    def copies(self) -> None:
        """Byte copies of the ring table, the design vector and config.json."""
        shutil.copy2(self.table.path, os.path.join(self.out_dir,
                                                   os.path.basename(self.table.path)))
        self.files["ring_table"] = os.path.join(self.out_dir, os.path.basename(self.table.path))
        m_path = os.path.join(self.run_dir, "m_final.npy")
        if os.path.exists(m_path):
            shutil.copy2(m_path, os.path.join(self.out_dir, "m_final.npy"))
            self.files["m_final"] = os.path.join(self.out_dir, "m_final.npy")
        shutil.copy2(os.path.join(self.run_dir, "config.json"),
                     os.path.join(self.out_dir, "design_config.json"))
        self.files["design_config"] = os.path.join(self.out_dir, "design_config.json")

    def info(self) -> Dict[str, Any]:
        """tapeout_info.json: the design in numbers plus provenance."""
        c, t = self.cfg, self.table
        der = c.get("derived", {})
        m = t.levels()
        info: Dict[str, Any] = {
            "gdsout_version": __version__, "stamp": self.stamp,
            "run_dir": os.path.abspath(self.run_dir), "design_name": self.name,
            "encoding": self.encoding.describe(),
            "gds": {"file": os.path.basename(self.settings.out), "cell": self.settings.cell,
                    "unit_m": 1e-6, "precision_m": self.settings.precision_m,
                    "chord_tolerance_um": self.settings.tol_um,
                    "polygons": getattr(self, "_export", None) and self._export.n_polygons,
                    "annuli": getattr(self, "_export", None) and len(self._export.annuli)},
            "geometry": {"diameter_um": c.get("diameter_um"),
                         "focal_um": der.get("focal_um"), "na": der.get("na"),
                         "radius_from_table_um": t.radius_um},
            "band": {"lam_min_um": c.get("lam_min_um"), "lam_max_um": c.get("lam_max_um"),
                     "target_wavelengths_um": c.get("target_wavelengths_um"),
                     "fom_mode": c.get("fom_mode"),
                     "ring_quadrature": c.get("ring_quadrature")},
            "fabrication": {"n_rings": t.n, "ring_width_um": t.delta_um,
                            "dh_um": t.dh_um, "dh_source": t.dh_source,
                            "h_max_config_um": c.get("h_max_um"),
                            "h_max_table_um": float(t.h_um.max()),
                            "levels_used": int(m.max()) + 1, "m_max": int(m.max()),
                            "level_residual_um": t.level_residual_um()},
            "ring_table": {"file": os.path.basename(t.path), "sha256": sha256_of(t.path),
                           "dll_file_no": c.get("dll_file_no")},
        }
        dm = os.path.join(self.run_dir, "design_metrics.json")
        if os.path.exists(dm):
            with open(dm) as fh:
                info["design_metrics"] = json.load(fh)
        vm = os.path.join(self.run_dir, "rs", "verify_metrics.json")
        if os.path.exists(vm):
            with open(vm) as fh:
                v = json.load(fh)
            keep = ["lam_um", "z_peak_um", "fwhm_um", "eff_3fwhm", "strehl_like",
                    "strehl_shape", "J_objective", "J_continuous", "J_verify_comb",
                    "rs_ring_quadrature", "fom_ring_quadrature"]
            info["rs_verification"] = {k: v[k] for k in keep if k in v}
        mt = os.path.join(self.run_dir, "rs", "verify_mtf.npz")
        if os.path.exists(mt):
            z = np.load(mt)
            info["rs_mtf"] = {"mtf_quality": z["mtf_quality"].tolist(),
                              "mtf_quality_poly": float(z["mtf_quality_poly"]),
                              "r_max_um": float(z["r_max_um"])}
        path = os.path.join(self.out_dir, "tapeout_info.json")
        with open(path, "w") as fh:
            json.dump(info, fh, indent=1)
        self.files["info"] = path
        return info

    def checksums(self) -> Dict[str, str]:
        """SHA256 of every file written so far, by base name."""
        return {os.path.basename(p): sha256_of(p) for p in self.files.values()
                if os.path.isfile(p)}

    def spec(self) -> str:
        """fabrication_spec.txt (gdsout.spec.FabricationSpec)."""
        path = os.path.join(self.out_dir, "fabrication_spec.txt")
        FabricationSpec(self.run_dir, self.cfg, self.table, self.encoding,
                        self.material).write(path, self.files, self.checksums())
        self.files["spec"] = path
        self.log("fabrication spec -> %s (material: %s, from %s)"
                 % (path, self.material.name, self.material.source))
        return path

    def readme(self) -> str:
        t = self.table
        lines = [
            "Tape-out package of MDL design '%s'  (%s, gdsout %s)" % (self.name, self.stamp, __version__),
            "run folder: %s" % os.path.abspath(self.run_dir), "",
            "READ FIRST: fabrication_spec.txt -- the complete sheet for the foundry.", "",
            "Files", "-----",
            "%-28s GDSII layout, encoding: %s" % (os.path.basename(self.settings.out), self.encoding.describe()),
            "%-28s gds_layer -> gray level -> height (um) -> height/max (dose table)" % os.path.basename(self.settings.layer_map_path),
            "%-28s one row per ring: r_in, r_out, gray level, height, gds layer" % "height_map.csv",
            "%-28s the same as numpy arrays (+ the merged annuli)" % "height_map.npz",
            "%-28s h(rho) staircase plot (full aperture + rim zoom)" % "height_profile.png",
            "%-28s the ring table the OpticStudio DLLs read (N delta_mm; h_mm per ring)" % os.path.basename(t.path),
            "%-28s the design's gray-level vector (numpy)" % "m_final.npy",
            "%-28s the design run's configuration" % "design_config.json",
            "%-28s provenance + design numbers + validation summary (machine-readable)" % "tapeout_info.json",
            "%-28s the foundry sheet (human-readable, self-contained)" % "fabrication_spec.txt",
            "", "GDS units: user unit 1 um, database unit %g m; cell '%s'." % (self.settings.precision_m, self.settings.cell),
        ]
        path = os.path.join(self.out_dir, "README.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        self.files["readme"] = path
        return path

    # -- driver --------------------------------------------------------------------
    def build(self) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        self.log("tape-out package -> %s" % self.out_dir)
        self.gds()
        self.height_map()
        self.height_profile_png()
        self.copies()
        self.info()
        self.spec()
        self.readme()
        self.log("package complete: %d files in %s (%.1fs)"
                 % (len(self.files), self.out_dir, time.time() - self.t0))
        return self.out_dir
