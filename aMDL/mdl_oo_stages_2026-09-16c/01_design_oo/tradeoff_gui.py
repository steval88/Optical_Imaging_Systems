"""
tradeoff_gui.py -- STAGE 0: the design-space GUI (Tkinter).

    python 01_design_oo\\tradeoff_gui.py

Type a specification (diameter in mm or inch -- any size --, NA or
F-number, band, relief height, quanta, material) and:
  * Check feasibility: derived geometry, the analytic (Eq. 7) and the
    numeric alias-free ceiling of max J_w(F), the verdict against the
    target J, the H the target needs, the paper reference it resembles;
  * Pair map: the Fig. 1b/c coherence map of the spec;
  * Run full study: the Fig. 1d (D, H) sweep -- a preset (PAPER_FIG1,
    SWIR_TRADEOFF) or a sweep around the typed spec -- written to
    runs\\<stamp>_tradeoff_<name>\\ (config.json, npz, png).
The physics is tradeoff/space.py (bounds from mdl.bounds); the command
line twin is tradeoff_maps.py.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tradeoff.gui import TradeoffApp  # noqa: E402

if __name__ == "__main__":
    TradeoffApp(pkg_root=os.path.dirname(HERE)).run()
