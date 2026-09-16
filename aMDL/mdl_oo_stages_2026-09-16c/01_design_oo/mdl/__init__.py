"""
mdl -- design of an achromatic multilevel diffractive lens (AMDL / MDL)
by light frequency-domain coherence optimization, after

    Xiao et al., "Large-scale achromatic flat lens by light
    frequency-domain coherence optimization",
    Light: Science & Applications (2022) 11:323,
    https://doi.org/10.1038/s41377-022-01024-y

Object-oriented, typed and documented port (2026-09-16) of the single
file mdl_core.py that lives at the package root (frozen there; every
run folder created before this date holds a snapshot of it). The
arithmetic and the order of random draws are identical, so a run with
the same settings and rng_seed reproduces the same design vector
(regression: old vs new, bit-identical tables, objectives, seeds,
bounds, zones and optimizer trajectories).

Module map
----------
material.py    n_az4562 (Cauchy fit of the resist), PAPER_COMB_14
problem.py     MDLProblem -- geometry, band, phasor tables G / L / S,
               objectives (on-axis Eq. 4 | encircled-energy overlap),
               aggregation (mean | geomean | softmin), analytic gradient,
               rigorous-efficiency hook. The physics derivation (paper
               Eqs. 3-4 <- TMS Eq. 15.32, the ring quadrature, the
               Bessel reduction) is in its module docstring.
optimizers.py  GeneticAlgorithm (integer-coded, explicit `population`),
               BinaryGeneticAlgorithm (paper S2-2), HookeJeeves,
               SearchGAHJA (blocks of GA -> HJA), MultistepGAHJACombo
               (verbatim Fig. S3), Smooth (S2-4), GradientRefine (S2-2);
               each `run(m) -> OptResult`.
seeds.py       harmonic_seed, LadderSeed (fold height chosen over a comb
               of lines; the per-line diffraction orders form a ladder)
bounds.py      pairwise_bound_matrix, upper_bound_jf (Eqs. S14-S15)
zones.py       local-grating decomposition and the RCWA correction files
compat.py      the pre-refactor function names, for scripts that still
               call e.g. search(prob, ...) or seed_echelle(prob, ...)
"""
from .material import C_UM_PER_S, PAPER_COMB_14, IndexModel, n_az4562
from .problem import (Field, FloatVec, FomMode, IntVec, MDLProblem, Objective,
                      RingQuadrature)
from .optimizers import (BinaryGeneticAlgorithm, GeneticAlgorithm,
                         GradientRefine, HookeJeeves, MultistepGAHJACombo,
                         OptResult, SearchGAHJA, Smooth)
from .seeds import (FoldScan, LadderSeed, LadderSeedResult, fold_seed,
                    harmonic_seed, ladder_fold_scan, max_harmonic_seed)
from .bounds import pairwise_bound_matrix, upper_bound_jf
from .zones import (Zone, extract_local_gratings, load_efficiency_table,
                    relative_correction_table, tea_efficiency_table,
                    tea_zone_efficiency, write_zone_table)

__version__ = "2026-09-16.03"

__all__ = [
    "C_UM_PER_S", "PAPER_COMB_14", "IndexModel", "n_az4562",
    "Field", "FloatVec", "FomMode", "IntVec", "MDLProblem", "Objective",
    "RingQuadrature",
    "BinaryGeneticAlgorithm", "GeneticAlgorithm", "GradientRefine",
    "HookeJeeves", "MultistepGAHJACombo", "OptResult", "SearchGAHJA", "Smooth",
    "FoldScan", "LadderSeed", "LadderSeedResult", "fold_seed", "harmonic_seed",
    "ladder_fold_scan", "max_harmonic_seed",
    "pairwise_bound_matrix", "upper_bound_jf",
    "Zone", "extract_local_gratings", "load_efficiency_table",
    "relative_correction_table", "tea_efficiency_table", "tea_zone_efficiency",
    "write_zone_table", "__version__",
]
