"""Annuli -> GDSII.

* Annuli are drawn with ``gdstk.ellipse(inner_radius=...)`` using a chord
  tolerance (``tol_um``, default 0.02 um) so circle facets stay well
  below the writer's address grid; polygons above 8190 vertices are
  fractured (the GDSII record limit).
* Units: user unit 1 um, database unit ``precision_m`` (default 1 nm).
* A sidecar CSV ``<out>.layers.csv`` maps gds_layer -> gray_level ->
  height (um) -> height / max height, for the lithography tool's dose
  table.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import gdstk

from .encode import Annulus, Encoding
from .rings import RingTable

LogFn = Callable[[str], None]
MAX_POINTS = 8190


@dataclass(frozen=True)
class ExportSettings:
    """Everything the writer needs besides the table and the encoding
    (all echoed by ``GdsExport.describe``)."""
    out: str                            # .gds path (folder is created)
    cell: str = "MDL"                   # top cell name
    library: str = "MDL"                # GDS library name
    tol_um: float = 0.02                # circle chord tolerance
    precision_m: float = 1e-9           # database unit (1 nm)
    max_points: int = MAX_POINTS        # fracture limit per polygon

    @property
    def layer_map_path(self) -> str:
        return self.out + ".layers.csv"


class GdsExport:
    """Write one ring table with one encoding.

    Attributes
    ----------
    table      RingTable
    encoding   Encoding (index | terrace)
    settings   ExportSettings
    annuli     the merged annuli (filled by build())
    n_polygons polygons written (after fracturing; filled by build())
    """

    def __init__(self, table: RingTable, encoding: Encoding,
                 settings: ExportSettings, log: Optional[LogFn] = None) -> None:
        self.table = table
        self.encoding = encoding
        self.settings = settings
        self.log: LogFn = log or print
        self.annuli: List[Annulus] = []
        self.n_polygons: int = 0
        self.library: Optional[Any] = None

    # -- geometry ------------------------------------------------------------------
    def polygons(self, a: Annulus) -> List[Any]:
        """gdstk polygon(s) of one annulus (a disc when r_in = 0)."""
        s = self.settings
        r_in, r_out = a.radii_um(self.table.delta_um)
        if r_in <= 0.0:
            p = gdstk.ellipse((0, 0), r_out, layer=a.layer, tolerance=s.tol_um)
        else:
            p = gdstk.ellipse((0, 0), r_out, inner_radius=r_in, layer=a.layer,
                              tolerance=s.tol_um)
        if p.size > s.max_points:
            return list(p.fracture(max_points=s.max_points))
        return [p]

    def build(self) -> Any:
        """The gdstk.Library with the top cell filled; sets annuli /
        n_polygons."""
        s = self.settings
        lib = gdstk.Library(name=s.library, unit=1e-6, precision=s.precision_m)
        cell = lib.new_cell(s.cell)
        self.annuli = self.encoding.annuli(self.table.levels())
        self.n_polygons = 0
        for a in self.annuli:
            polys = self.polygons(a)
            cell.add(*polys)
            self.n_polygons += len(polys)
        self.library = lib
        return lib

    # -- files -------------------------------------------------------------------------
    def write(self) -> str:
        """Build (if not yet), write the .gds and the layer-map CSV; returns
        the .gds path."""
        s = self.settings
        os.makedirs(os.path.dirname(os.path.abspath(s.out)), exist_ok=True)
        lib = self.library or self.build()
        lib.write_gds(s.out)
        size_mb = os.path.getsize(s.out) / 1e6
        self.log("wrote %s  (%.1f MB, %d polygons from %d annuli, mode=%s)"
                 % (s.out, size_mb, self.n_polygons, len(self.annuli),
                    self.encoding.NAME))
        self.write_layer_map()
        return s.out

    def write_layer_map(self) -> str:
        """``<out>.layers.csv``: gds_layer, gray_level, height_um,
        height_frac_of_max (for the dose table)."""
        t, path = self.table, self.settings.layer_map_path
        m_max = int(t.levels().max())
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gds_layer", "gray_level", "height_um", "height_frac_of_max"])
            for L, lev in self.encoding.layer_levels(m_max):
                w.writerow([L, lev, "%.4f" % (lev * t.dh_um), "%.5f" % (lev / m_max)])
        self.log("layer map -> %s" % path)
        return path

    def describe(self) -> None:
        s = self.settings
        self.log(self.table.describe())
        self.log("encoding: %s" % self.encoding.describe())
        self.log("gds: cell %s, library %s, unit 1 um, database unit %g m, chord "
                 "tolerance %.3f um, fracture at %d points -> %s"
                 % (s.cell, s.library, s.precision_m, s.tol_um, s.max_points, s.out))
