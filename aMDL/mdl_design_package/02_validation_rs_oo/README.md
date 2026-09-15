# 02_validation_rs_oo — RS verification, object-oriented (2026-09-16)

Typed, documented port of `02_validation_rs/run_verify.py` and
`mtf_verify.py`. The legacy folder stays untouched; this is the sibling
stage, like `01_design_oo` and `02_validation_zemax_oo`. Same files in
`<run>/rs/`, same keys, same numbers (bit-identical regression below),
so `make_plots.py` and the Zemax stage consume the run folders as before.

    02_validation_rs_oo\
        run_verify.py            thin CLI  -> rsval.VerifyRun
        mtf_verify.py            thin CLI  -> rsval.MtfRun
        rsval\
            base.py              Log, run-folder resolution (+ typo hint),
                                 Stage (rs/ folder, scripts/ snapshot)
            design.py            VerifyConfig: every config.json key read,
                                 typed, with the legacy defaults and the
                                 list of keys that were absent (echoed);
                                 DesignState: m, h, rho, geometry, MDLProblem
            propagator.py        RSPropagator: TEA exit field, exact on-axis
                                 RS-I, J0-reduced off-axis RS-I, ideal-lens
                                 references, sinc ring quadrature. THE ONE
                                 copy of the kernel (the two legacy scripts
                                 carried duplicates). Physics + references
                                 in its module docstring.
            verify.py            VerifyRun: [1/4] on-axis scans, [2/4] PSF
                                 metrics, [3/4] r-z maps, [4/4] J metrics;
                                 section results as dataclasses
            mtf.py               MtfRun: window-normalized Hankel MTF + fig
        tests\
            test_against_legacy.py   runs both stages on copies of a run
                                     folder and diffs every output

## Run

    python 02_validation_rs_oo\run_verify.py runs\<run> [other_m.npy]
    python 02_validation_rs_oo\mtf_verify.py runs\<run> [other_m.npy]
    python 02_validation_rs_oo\tests\test_against_legacy.py runs\<run>

`MDLProblem` (the design-FOM tables for the J metrics) comes from
`01_design_oo/mdl` when that folder exists, else from the root
`mdl_core.py`; the choice is printed. Both stages snapshot themselves
(`run_verify.py` / `mtf_verify.py` + the whole `rsval/` folder) into
`<run>/scripts/`.

## What changed besides the structure

* `verify_metrics.json` gains one key, `rsval_version`; every legacy key
  is unchanged.
* The log prints which optional config keys were absent and the default
  used (`rs_ring_quadrature=sinc, mtf_r_points=1601, ...`).
* `run_verify.py`'s "next" hint now points at `mtf_verify.py` in this
  folder (the legacy pointed at `make_plots.py`, which has not been
  ported yet: it still runs from `02_validation_rs` on these run folders).

## Regression (2026-09-16)

`tests/test_against_legacy.py` on the synthetic harmonic lens and on the
replay of run 20260915_140205 (S3 softmin, sinc): `verify_onaxis.npz`,
`verify_rzmap.npz`, `verify_mtf.npz`, all 17 legacy keys of
`verify_metrics.json` and the re-exported ring table identical (numpy
`array_equal`, byte compare for the table). J objective 0.0927580270,
polychromatic MTF quality 0.578 — the numbers of the 140205 run.
`mypy --ignore-missing-imports rsval run_verify.py mtf_verify.py`: no
issues; `pyflakes`: clean.
