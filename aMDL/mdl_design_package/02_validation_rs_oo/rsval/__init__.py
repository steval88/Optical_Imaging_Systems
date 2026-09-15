"""
rsval -- STAGE 2a, object-oriented: Rayleigh-Sommerfeld verification of
a designed MDL (the physics of the paper's Fig. 2e check) and its
Hankel-transform MTF.

Typed, documented port of ``02_validation_rs/run_verify.py`` and
``02_validation_rs/mtf_verify.py`` (2026-09-16). The two legacy scripts
stay untouched in their folder; this package is the sibling stage
``02_validation_rs_oo`` and produces byte-for-byte the same
``rs/verify_*.npz`` / ``rs/verify_metrics.json`` / ring-table files, so
``make_plots.py`` and the Zemax stage consume its run folders unchanged
(regression: tests/test_against_legacy.py).

Module map
----------
    design.py      VerifyConfig  -- every config.json key this stage reads,
                                    typed, with the defaults it falls back
                                    to (and the list of keys that did)
                   DesignState   -- the run folder loaded: geometry, gray
                                    levels m, heights h, the MDLProblem
                                    (tables of the design FOM) and the
                                    quadrature rules in force
    propagator.py  RSPropagator  -- the scalar RS-I physics (thin-element
                                    exit field, exact on-axis integral,
                                    J0-reduced off-axis integral, ideal-lens
                                    references, ring quadrature); the ONE
                                    source of this kernel for both drivers
    verify.py      VerifyRun     -- on-axis scans, focal PSF metrics, r-z
                                    maps, J metrics -> rs/verify_*.npz,
                                    rs/verify_metrics.json, ring table
    mtf.py         MtfRun        -- window-normalized incoherent MTF from
                                    the focal PSF -> rs/verify_mtf.npz,
                                    rs/fig_mtf.png
    base.py        Log, Stage    -- timing log, run-folder resolution,
                                    rs/ output folder, script snapshot

The design FOM (``MDLProblem``) is imported from ``01_design_oo/mdl``
when that folder exists, else from the frozen root ``mdl_core.py`` (the
two are bit-identical; the choice is echoed in the log).
"""
from .base import VERSION, Log, Stage, parse_cli, resolve_run_dir
from .design import DesignState, VerifyConfig, import_mdl_problem
from .mtf import MtfRun, hankel_mtf
from .propagator import RSPropagator
from .verify import VerifyRun

__version__ = VERSION

__all__ = ["Log", "Stage", "parse_cli", "resolve_run_dir", "DesignState", "VerifyConfig",
           "import_mdl_problem", "MtfRun", "hankel_mtf", "RSPropagator",
           "VerifyRun", "__version__"]
