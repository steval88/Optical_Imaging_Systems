"""
tradeoff -- STAGE 0: the design space of an achromatic MDL before any
optimization (paper Fig. 1b/c/d and Eq. 7 of Xiao et al., Light Sci.
Appl. 11:323).

Given a lens specification (diameter, NA or F-number, band, relief
height H, height quantum dh, ring width DELTA, resist) the package
answers, in seconds:

  * the derived geometry (F, N rings, fold waves per line, rim period,
    alias-free wavelength sampling, aspect ratio, table sizes of a run);
  * the UPPER BOUND of the continuous-band figure of merit max J_w(F):
    numerically from the pairwise coherence matrix (paper Eq. S14-S15,
    mdl.bounds) and analytically from the paper's Eq. 7,
        D_max = 4 (n_max - 1) H / [(1 - sqrt(1 - max J)) NA];
  * a FEASIBILITY verdict against a target J (or the paper's S1..S5
    reference points), with the H the target would need at this D / NA;
  * the Fig. 1b/1c pairwise coherence maps and the Fig. 1d D-H sweep of
    the ceiling (tradeoff_maps.py, run-folder outputs).

Module map
----------
    space.py    LensSpec (typed inputs + derived quantities), Ceiling
                (numeric + analytic bounds), Feasibility (verdict)
    study.py    PairMapStudy (Fig. 1b/c panels), SweepStudy (Fig. 1d),
                presets PAPER_FIG1 / SWIR_TRADEOFF, run-folder writer
    gui.py      TradeoffApp: the Tkinter front end (tradeoff_gui.py)
"""
from .space import Ceiling, Feasibility, LensSpec, REFERENCE_POINTS
from .study import PAPER_FIG1, PRESETS, SWIR_TRADEOFF, PairMapStudy, StudyConfig, SweepStudy, run_study

__version__ = "2026-09-16.01"

__all__ = ["Ceiling", "Feasibility", "LensSpec", "REFERENCE_POINTS", "PAPER_FIG1", "PRESETS",
           "SWIR_TRADEOFF", "PairMapStudy", "StudyConfig", "SweepStudy", "run_study", "__version__"]
