"""How gray levels become GDS layers.

Two encodings, chosen with ``--mode``:

  index     layer L contains the annuli whose gray level == L. This is
            the common Heidelberg grayscale conversion convention "one
            GDS layer = one gray / dose value". Level 0 (no resist left /
            full-exposure reference) is NOT drawn unless asked
            (``draw_zero``), for dose tables that need an explicit
            polygon for it.

  terrace   layer L contains the region where gray level >= L (nested
            "topographic contour" masks, L = 1..M). For etch-back /
            multi-mask flows and for conversion tools that build height
            by stacking.

Both merge adjacent rings of equal membership into one annulus (fewer,
cleaner polygons). An ``Encoding`` turns the level vector into a list
of ``Annulus`` records; drawing them is ``export.GdsExport``'s job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Type

import numpy as np

IntVec = np.ndarray


@dataclass(frozen=True)
class Annulus:
    """Rings i0 .. i1-1 (radii [i0 DELTA, i1 DELTA)) drawn on ``layer``.
    ``level`` is the gray level the annulus stands for (== layer in
    index mode; the threshold in terrace mode)."""
    layer: int
    level: int
    i0: int
    i1: int

    def radii_um(self, delta_um: float) -> Tuple[float, float]:
        return self.i0 * delta_um, self.i1 * delta_um


def runs_of_equal(levels: IntVec) -> Iterator[Tuple[int, int, int]]:
    """Consecutive equal levels: yields (level, i_start, i_end_excl)."""
    start = 0
    for i in range(1, levels.size + 1):
        if i == levels.size or levels[i] != levels[start]:
            yield int(levels[start]), start, i
            start = i


def runs_of_true(mask: np.ndarray) -> Iterator[Tuple[int, int]]:
    """Contiguous True stretches of a boolean vector: (i_start, i_end_excl)."""
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


class Encoding:
    """Base: ``annuli(levels)`` and the layer -> level map of the sidecar."""
    NAME = "?"

    def __init__(self, draw_zero: bool = False) -> None:
        self.draw_zero: bool = draw_zero      # index mode: draw level 0 too

    def annuli(self, levels: IntVec) -> List[Annulus]:
        raise NotImplementedError

    def layer_levels(self, m_max: int) -> List[Tuple[int, int]]:
        """(gds_layer, gray_level) rows of the layer map, in layer order."""
        return [(L, L) for L in range(0 if self.draw_zero else 1, m_max + 1)]

    def describe(self) -> str:
        return self.NAME


class IndexEncoding(Encoding):
    """layer L = {rings with level == L}; level 0 skipped unless draw_zero."""
    NAME = "index"

    def annuli(self, levels: IntVec) -> List[Annulus]:
        out: List[Annulus] = []
        for lev, i0, i1 in runs_of_equal(levels):
            if lev == 0 and not self.draw_zero:
                continue
            out.append(Annulus(layer=lev, level=lev, i0=i0, i1=i1))
        return out

    def describe(self) -> str:
        return "index (layer L = rings at gray level L%s)" % (
            ", level 0 drawn" if self.draw_zero else ", level 0 not drawn")


class TerraceEncoding(Encoding):
    """layer L = {rings with level >= L}, L = 1..M (nested contours)."""
    NAME = "terrace"

    def annuli(self, levels: IntVec) -> List[Annulus]:
        out: List[Annulus] = []
        m_max = int(levels.max())
        for L in range(1, m_max + 1):
            for i0, i1 in runs_of_true(levels >= L):
                out.append(Annulus(layer=L, level=L, i0=i0, i1=i1))
        return out

    def describe(self) -> str:
        return "terrace (layer L = rings at gray level >= L, L = 1..M)"


ENCODINGS: Dict[str, Type[Encoding]] = {IndexEncoding.NAME: IndexEncoding,
                                        TerraceEncoding.NAME: TerraceEncoding}


def encoding_by_name(name: str, draw_zero: bool = False) -> Encoding:
    try:
        return ENCODINGS[name](draw_zero)
    except KeyError:
        raise ValueError("unknown encoding %r (choose from %s)"
                         % (name, ", ".join(ENCODINGS))) from None
