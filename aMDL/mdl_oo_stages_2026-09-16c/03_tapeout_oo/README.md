# 03_tapeout_oo — tape-out package, object-oriented (2026-09-16)

Typed, documented port of `03_tapeout/export_gds.py` (which stays
untouched), extended into a full tape-out package per run: the GDS of
the designed achromatic MDL plus the per-ring height map and a
plain-text specification sheet for the foundry.

    03_tapeout_oo\
        export_gds.py            thin CLI -> gdsout.cli.main
        gdsout\
            rings.py             RingTable: the DLL-format table (N, DELTA,
                                 h_i) from a file or a run folder; gray
                                 levels m_i = round(h_i/dh) with dh from the
                                 run's config.json / the caller / inferred
                                 (source and residual |h - m dh| echoed)
            encode.py            IndexEncoding (layer L = level L) and
                                 TerraceEncoding (layer L = level >= L) ->
                                 merged Annulus records
            export.py            GdsExport: annuli -> gdstk polygons (chord
                                 tolerance, fracture at 8190 pts), .gds and
                                 the <out>.layers.csv dose map
            spec.py              MaterialSpec (AZ4562 model of the design)
                                 and FabricationSpec -> fabrication_spec.txt
            package.py           TapeoutPackage: <run>/tapeout/<stamp>_<mode>/
            cli.py               argparse front end
        tests\
            test_against_legacy.py   same polygons / layer map as the
                                     legacy script, both encodings

## Run

    python 03_tapeout_oo\export_gds.py --run runs\<run> [--mode index|terrace]
    python 03_tapeout_oo\export_gds.py --rings mdl_rings_2.txt --out lens.gds [--dh-um 0.078]
    python 03_tapeout_oo\tests\test_against_legacy.py runs\<run>

`--run` builds the package (run `02_Rayleigh_Sommerfeld_Validation_OOP\run_verify.py` and
`mtf_verify.py` first so the verified performance lands in the sheet):

    <run>\tapeout\<YYYYMMDD_HHMMSS>_<mode>\
        <name>.gds                 the layout  (a)
        <name>.gds.layers.csv      gds layer -> gray level -> height (dose map)
        fabrication_spec.txt       the foundry sheet  (b): optical function,
                                   material + index model + n at every design
                                   line, relief geometry (N, DELTA, dh, levels,
                                   H_max), tolerance guidance, encoding + dose
                                   map, verified performance per wavelength,
                                   SHA256 of every file, and the full height
                                   map (one line per ring)
        height_map.csv / .npz      ring, r_in, r_out, gray level, height, layer
        height_profile.png         h(rho), full aperture + outermost 100 rings
        mdl_rings_<n>.txt          byte copy of the ring table
        m_final.npy                the design vector
        design_config.json         the run's config.json
        tapeout_info.json          the same numbers, machine-readable
        README.txt                 file index

The material block is taken from `01_design_oo/mdl/material.py`
(`n_az4562`, the model the design was optimized with); a different
resist is a `MaterialSpec(...)` passed to `TapeoutPackage`. Items the
design does not fix (substrate, resist lot, AR coating, tolerances) are
printed as explicit TO-BE-AGREED lines.

## Not hardcoded any more

The legacy CLI defaulted `--dh-um` to 0.078 silently. Now dh comes from
the run's `config.json` (`--run`), from `--dh-um`, or is inferred from
the table (smallest non-zero spacing of the distinct heights), and the
log says which; a residual |h_i - m_i dh| above dh/4 raises a warning.

## Regression (2026-09-16)

`tests/test_against_legacy.py` on the 2560-ring, 193-level S3 table:
index (35 152 polygons), index --draw-zero (35 218), terrace (251 885)
— every polygon's layer and vertex array identical to the legacy
script's, layer-map CSV identical. `mypy` / `pyflakes` clean.
