"""rcwaval -- rigorous (RCWA) efficiency of the design's own zone profiles,
without OpticStudio: the fold-reset question the srg DLLs cannot reach
(harmonic cap 50, energy-balance refusals on deep layered profiles).

    rcwa1d.py    the 1-D RCWA solver (TE/TM, S-matrix, Li's rule)
    profiles.py  a ring-height profile -> lamellar layer stack; test structures
    zonesweep.py the zone_table.npz sweep (every zone x line, checkpointed)
    table.py     the sweep -> the design's efficiency_corr.npz (weak cells
                 dropped, clipped, gaps filled, median-smoothed over r)
"""
SCRIPT_VERSION = "2026-09-21.11"

__all__ = ["SCRIPT_VERSION"]
