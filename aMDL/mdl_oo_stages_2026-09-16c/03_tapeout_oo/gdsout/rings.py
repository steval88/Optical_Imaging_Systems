"""The ring-height table -- the fabrication-facing artifact of a run.

Format (the file the OpticStudio user-defined-surface DLLs read; written
by run_MDL_design.py and re-written by run_verify.py so it always
matches the verified vector):

    line 1 : N delta_mm
    then N : h_mm           (one ring height per line, in millimetres)

Ring i covers [i DELTA, (i+1) DELTA) and has height h_i = m_i dh with
integer gray level m_i in 0..M. The GDS stage works on the levels, so it
needs dh: from the run folder's config.json (``RingTable.from_run``),
from the caller, or inferred from the table itself.
"""
from __future__ import annotations

import json
import os
from typing import Literal, Optional, Tuple

import numpy as np

FloatVec = np.ndarray
IntVec = np.ndarray
DhSource = Literal["argument", "config.json", "inferred"]


class RingTable:
    """N ring heights on a DELTA pitch, and their gray levels.

    Attributes
    ----------
    path        the table file
    delta_um    ring width DELTA                                  [um]
    h_um        ring heights h_i, i = 0..N-1                       [um]
    dh_um       height quantum used to recover the gray levels   [um]
    dh_source   where dh came from ("argument" | "config.json" |
                "inferred")
    """

    def __init__(self, path: str, dh_um: Optional[float] = None,
                 dh_source: DhSource = "argument") -> None:
        self.path: str = path
        with open(path) as fh:
            first = fh.readline().split()
            n, delta_mm = int(first[0]), float(first[1])
            h_mm = np.array([float(fh.readline()) for _ in range(n)])
        self.delta_um: float = delta_mm * 1000.0
        self.h_um: FloatVec = h_mm * 1000.0
        if dh_um is None:
            dh_um, dh_source = self.infer_dh(), "inferred"
        self.dh_um: float = float(dh_um)
        self.dh_source: DhSource = dh_source

    # -- constructors -------------------------------------------------------------
    @classmethod
    def from_run(cls, run_dir: str, dh_um: Optional[float] = None) -> "RingTable":
        """The table of a run folder: ``mdl_rings_<dll_file_no>.txt`` with
        ``dh_um`` from its config.json (an explicit ``dh_um`` wins)."""
        with open(os.path.join(run_dir, "config.json")) as fh:
            cfg = json.load(fh)
        path = os.path.join(run_dir, "mdl_rings_%d.txt" % int(cfg["dll_file_no"]))
        if dh_um is not None:
            return cls(path, dh_um, "argument")
        return cls(path, float(cfg["dh_um"]), "config.json")

    # -- derived --------------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.h_um.size)

    @property
    def radius_um(self) -> float:
        """Outer radius N DELTA of the lens."""
        return self.n * self.delta_um

    def infer_dh(self) -> float:
        """The smallest non-zero spacing between the distinct heights of
        the table: equals dh unless every neighbouring level pair is
        skipped (never the case for a 100+ level staircase)."""
        vals = np.unique(self.h_um)
        gaps = np.diff(vals)
        gaps = gaps[gaps > 1e-9]
        if gaps.size == 0:
            raise ValueError("%s: all rings at one height, cannot infer dh" % self.path)
        return float(gaps.min())

    def levels(self) -> IntVec:
        """Gray levels m_i = round(h_i / dh)."""
        return np.rint(self.h_um / self.dh_um).astype(int)

    def level_residual_um(self) -> float:
        """max |h_i - m_i dh|: 0 for a table written from integer levels;
        a value near dh/2 means dh is wrong for this table."""
        m = self.levels()
        return float(np.max(np.abs(self.h_um - m * self.dh_um)))

    def ring_bounds_um(self, i0: int, i1: int) -> Tuple[float, float]:
        """[r_in, r_out) of the annulus made of rings i0 .. i1-1."""
        return i0 * self.delta_um, i1 * self.delta_um

    def describe(self) -> str:
        m = self.levels()
        return ("rings: %d  ring width: %.4f um  levels: 0..%d  R = %.3f mm  "
                "(dh = %.4f um from %s, residual %.2e um)"
                % (self.n, self.delta_um, int(m.max()), self.radius_um / 1000.0,
                   self.dh_um, self.dh_source, self.level_residual_um()))
