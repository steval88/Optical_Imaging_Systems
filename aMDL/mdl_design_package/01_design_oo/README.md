# 01_design_oo — MDL design, object-oriented (2026-09-16, mdl 2026-09-16.02)

Typed, documented port of `mdl_core.py` + `01_design/run_MDL_design.py`.
Both originals stay untouched at their places (every run folder created
before this date holds a snapshot of them); this folder is a sibling
stage, exactly like `02_validation_zemax_oo` next to `02_validation_zemax`.

    01_design_oo\
        run_MDL_design.py        typed DesignConfig (dataclass) + DesignRun
        fom_quadrature_check.py  midpoint vs sinc objective of an existing run
        mdl\                     the package (module map in mdl/__init__.py)
            material.py          n_az4562, PAPER_COMB_14
            problem.py           MDLProblem: tables G / L / S, objectives,
                                 aggregation, analytic gradient, RCWA hook.
                                 The physics (paper Eqs. 3-4 <- TMS 15.32,
                                 ring quadrature, Bessel reduction) is derived
                                 in its module docstring.
            optimizers.py        GeneticAlgorithm (explicit `population`),
                                 BinaryGeneticAlgorithm, HookeJeeves,
                                 SearchGAHJA, MultistepGAHJACombo, Smooth,
                                 GradientRefine  -- each run(m) -> OptResult
            seeds.py             harmonic_seed, LadderSeed (fold height chosen
                                 over a comb; the per-line orders form a ladder)
            bounds.py            pairwise_bound_matrix, upper_bound_jf
            zones.py             local-grating decomposition, RCWA files
            compat.py            the old function names (search, smooth, ...)
        tests\
            test_against_mdl_core.py   bit-identical regression vs the
                                       frozen mdl_core.py (~30 s)

## Run

    python 01_design_oo\run_MDL_design.py          # SETTINGS preset
    python 01_design_oo\fom_quadrature_check.py runs\<run>
    python 01_design_oo\tests\test_against_mdl_core.py

Presets are `DesignConfig` instances; variants use `dataclasses.replace`:

    S3_COMB_SOFTMIN_A1 = replace(S3_COMB_SOFTMIN, name="s3_comb_softmin_a1",
                                 overlap_airy_factor=1.0, dll_file_no=7)

`config.json` keeps the key set of the old driver, so `02_validation_rs_oo`,
`02_validation_zemax_oo` and `03_tapeout_oo` consume the run folders
unchanged. `scripts/` snapshots the code of every `_oo` stage: the list is
the module constant `STAGE_SNAPSHOT` (the one place stage folder names
appear in the driver; `DesignConfig.snapshot_scripts` defaults to it and
is echoed in `config.json`), and the log prints what was copied and what
was missing. The "next" hint points at `02_validation_rs_oo/run_verify.py`.

## Renames

* `seed_mode "echelle"` -> `"ladder"` (old configs are accepted and mapped;
  old seed records are read by the validation stage as before). The seed
  picks the fold height for a whole comb of lines; each line then focuses
  in its own integer diffraction order and those orders form a descending
  ladder (24 at 400 nm ... 8 at 1100 nm for the S3 fold). The selection
  chart is the blaze chart of a high-order ("echelle", French *échelle* =
  ladder) grating applied to the fold height.
* `mdl_core_Dev_v1.*` mentions removed.

## Regression (2026-09-16)

* `tests/test_against_mdl_core.py`: tables, objectives (mean / geomean /
  softmin x onaxis / overlap), gradients, delta updates, harmonic and
  ladder seeds, small GA + HJA + Smooth + Gradient + verbatim-combo
  replays with the same RNG, bound, zones, TEA table: all bit-identical.
* Full pipeline, S3 softmin preset, rng_seed 7, old driver vs new driver:
  `m_final.npy` identical; J_seed 0.0339618569, J_search 0.1680064458,
  J_smooth 0.1504816995, J_gradient 0.1580930905, J_final 0.1600510785
  identical to 10 decimals (and equal to run 20260915_140205).
* `mypy --ignore-missing-imports mdl run_MDL_design.py fom_quadrature_check.py`:
  no issues (2026-09-16.02: the driver's config fields are `Literal` types,
  `FomMode` / `RingQuadrature` / `SeedMode`); `pyflakes`: clean.
