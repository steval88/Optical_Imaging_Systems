# MDL design package

Design, validation and tape-out of achromatic Multilevel Diffractive
Lenses (MDL) by frequency-domain coherence optimization, after
Xiao et al., Light Sci. Appl. 11:323 (2022).

## Folder layout (mirrors the workflow)

```
mdl_design_package/
├── mdl_core.py               shared physics + optimizers (imported by all stages)
├── 01_design/
│   └── run_MDL_design.py     STAGE 1: Search (GA+HJA) -> Smooth -> Gradient
├── 02_validation_rs/
│   ├── run_verify.py         STAGE 2a: Rayleigh-Sommerfeld diffraction check
│   ├── make_plots.py         figures (incl. paper-Fig.2e-style on-axis map)
│   └── bpm_validate.py       optional: BPM through the real staircase relief
├── 02_validation_zemax/
│   ├── mdl_zemax_validation.py  STAGE 2b: ZOS-API driver (zone + od modes)
│   ├── zos_connection.py        ZOS-API standalone connection helper
│   └── dll/                     UDS C++ sources, compiled x64 DLLs, usersurf.h
├── 03_tapeout/
│   ├── export_gds.py         STAGE 3: GDSII for maskless litho (Heidelberg DWL)
│   └── npy_to_rings.py       ad-hoc .npy -> ring-table converter
├── tools/
│   └── validate_bound.py     upper-bound self-check (brute force vs candidate)
└── runs/                     one timestamped folder per design run (never overwritten)
```

## Workflow (run everything from THIS folder, the package root)

STAGE 1 — design. Edit the `SETTINGS` dictionary at the top of
`01_design/run_MDL_design.py` (geometry, band, continuous vs discrete
target wavelengths, `fom_mode` mean/geomean, optimizer settings), then:

    python 01_design\run_MDL_design.py

This creates `runs/<timestamp>_<name>/` with `config.json`, a
`scripts/` snapshot, `m_final.npy`, `mdl_rings_<n>.txt` and
`design_metrics.json`. Each script prints the suggested next command.

STAGE 2a — RS validation + figures:

    python 02_validation_rs\run_verify.py  runs\<folder>
    python 02_validation_rs\make_plots.py  runs\<folder>

STAGE 2b — Zemax validation (on the machine with OpticStudio):
copy `runs\<folder>\mdl_rings_<n>.txt` and the DLLs from
`02_validation_zemax\dll\` into `Documents\Zemax\DLL\Surfaces\`, then

    python 02_validation_zemax\mdl_zemax_validation.py s3 zone   (Huygens/FFT PSF)
    python 02_validation_zemax\mdl_zemax_validation.py s3 od     (ray-based chromatic plots)

zone mode = us_mdl_rings.dll (staircase; only diffraction analyses are
physical). od mode = us_mdl_rings_od.dll (order decomposition; Order
m = 0 means auto/dominant order — the multi-order "camera view").

STAGE 3 — tape-out:

    python 03_tapeout\export_gds.py --rings runs\<folder>\mdl_rings_<n>.txt --out runs\<folder>\mdl.gds

writes the GDS plus a `.layers.csv` dose map (one GDS layer per 78 nm
gray level, Heidelberg convention).

## Conventions

- All lengths in the config dictionary are micrometers.
- Ring-table format (DLL + GDS): line 1 `N delta_mm`, then N lines `h_mm`.
- A run folder is self-contained: config, code snapshot, design vector,
  ring table, verification data and figures all live inside it.
