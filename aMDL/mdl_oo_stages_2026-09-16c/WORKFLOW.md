# Designing an achromatic MDL with this package — the steps in order

All commands are run FROM THE PACKAGE ROOT (`mdl_design_package\`) in
the `ZOS_API_Zemax` conda environment. `<run>` stands for the run folder
that step 2 creates, e.g. `runs\20260915_231624_s3_comb_softmin_a1`.
Every stage reads everything it needs from `<run>\config.json` and
writes only into `<run>\` (per-solver subfolders `rs\`, `zemax\`, `nsc\`,
`tapeout\`), so the run folder is the complete record of one design.

## 0. Explore the design space  (before choosing anything)

    python 01_design_oo\tradeoff_gui.py
    python 01_design_oo\tradeoff_maps.py check --d-inch 2 --fnum 5 --h 45 --band 400,1100 --target 0.05
    python 01_design_oo\tradeoff_maps.py study --preset PAPER_FIG1

Type any aperture (mm or inch, no size limit), NA or F-number, band,
relief height H, quanta and material: the tool prints the derived
geometry (F, rings, fold orders, rim period, alias-free sampling, table
sizes of a run), the ceiling of the continuous-band figure of merit
max J_w(F) — analytic (paper Eq. 7) and numeric alias-free (Eqs.
S14-S15) — and a feasibility verdict against a target J, with the H the
target needs at that D / NA. Pair maps (paper Fig. 1b/c) and the (D, H)
sweep (Fig. 1d) go to `runs\<stamp>_tradeoff_<name>\`.

## 1. Choose the design settings  (edit, no command)

File: `01_design_oo\run_MDL_design.py`, CONFIG section.
Pick or derive a `DesignConfig` preset and point `SETTINGS` at it:

    SETTINGS = S3_COMB_SOFTMIN_A1        # or replace(S3_COMB_SOFTMIN_A1, name=..., ...)

The fields that define the design (all documented in the dataclass):
geometry (`diameter_um`, `focal_um` xor `na`), band and objective
(`lam_min_um`, `lam_max_um`, `target_wavelengths_um`, `fom_mode`,
`overlap_fom`, `overlap_airy_factor`, `ring_quadrature`), fabrication
quanta (`ring_width_um`, `h_max_um`, `dh_um`), optimizer (`seed_mode`,
`ga_blocks`, `ga_epochs`, `pop_size`, `gradient_iters`, `hja_*`,
`rng_seed`), verification grids (`verify_*`, `rzmap_*`) and packaging
(`dll_file_no` — one number per design, never reuse one). The settings
of record (run 3, 2026-09-16): sinc ring quadrature, softmin objective
on the 14-line comb, main-lobe encircled energy (`overlap_airy_factor
= 1.0`), ladder seed, file number 7.

## 2. Design  (Search → Smooth → Gradient → Polish)

    python 01_design_oo\run_MDL_design.py

Creates `runs\<stamp>_<name>\` with `config.json`, `m_final.npy`,
`mdl_rings_<n>.txt`, `design_metrics.json` (J at every stage + the seed
record), `zone_table.npz` and `scripts\` (snapshot of every stage's
code). Runtime: tens of minutes for the S3 presets. The last log line
is the next command.

## 3. Verify the design physically  (scalar Rayleigh–Sommerfeld)

    python 02_Rayleigh_Sommerfeld_Validation_OOP\run_verify.py <run>
    python 02_Rayleigh_Sommerfeld_Validation_OOP\mtf_verify.py <run>
    python 01_design_oo\fom_quadrature_check.py <run>

`run_verify`: on-axis scans (focal shift, satellites), focal PSF
metrics per line (FWHM vs diffraction limit, eff in the 3×FWHM disc,
Strehl-like, shape-Strehl), r–z tiles, J metrics → `<run>\rs\`
(`verify_metrics.json`, `verify_onaxis.npz`, `verify_rzmap.npz`) and
the ring table re-export. `mtf_verify`: window-normalized Hankel MTF
per line and polychromatic → `rs\verify_mtf.npz`, `rs\fig_mtf.png`.
`fom_quadrature_check`: η per line under midpoint and sinc rules,
confirms J_final. About one minute in total.

What to look at: every line's focus at F (offsets within a few tens of
µm, no satellites), FWHM/limit ≈ 1.0–1.2, eff flat across the band,
`fom_quadrature_check` MATCHES, polychromatic MTF quality.

Figures (after run_verify / mtf_verify):

    python 02_Rayleigh_Sommerfeld_Validation_OOP\make_plots.py <run>

→ `rs\fig_onaxis.png` (focal shift), `fig_psf.png` (focal PSF per line vs
the ideal lens), `fig_psf_2d.png` (spots), `fig_rz_tiles.png` (Fig. 2e
tiles), `fig_metrics.png` (all scalar metrics vs wavelength).

## 4. Cross-check in OpticStudio  (ZOS-API, the OpticStudio machine)

Prerequisites once: `us_mdl_rings.dll`, `us_mdl_rings_od.dll` in
`{Documents}\Zemax\DLL\Surfaces\`; `pip install pythonnet`;
`zos_connection.py` in `02_validation_zemax\`. The design's ring table
is synced into the DLL folder automatically (SHA256-checked).

    python 02_Sequential_RT_Zemax_Validation_OOP\mdl_zemax_validation.py <run> rz
    python 02_Sequential_RT_Zemax_Validation_OOP\mdl_zemax_validation.py <run> huy

`rz`: the DLL's phase against the design (OPD self-check 0.0000 waves)
and Zemax-field r–z tiles against `rs\verify_rzmap.npz` (dz 0, corr
1.0000). `huy`: OpticStudio's native Huygens PSF/MTF on the hybrid
system (Paraxial f = F + residual UDS) against the RS reference (FWHM
ratios ≈ 1.00, corr 1.0000, MTF quality three ways). Optional:
`zone` (surface build only) and `od --orders 10,15,0` (per-order
chromatic plots). Outputs: `<run>\zemax\<stamp>_<mode>\` with
`run_info.json`, npz, figures and the `.zos` file.

GUI mode — for looking at the results with Zemax's own windows:

  1. Open OpticStudio.
  2. Programming tab → Interactive Extension → the tile must show
     "waiting for connection".
  3. Run the same command with `--gui`:

         python 02_Sequential_RT_Zemax_Validation_OOP\mdl_zemax_validation.py <run> huy --gui

  4. Every analysis (Huygens PSF, Huygens MTF per line) opens as a native
     window and STAYS OPEN after the script exits; the system is saved as
     `mdl_validation_huygens_hybrid.zos` in the run's zemax subfolder for
     colleagues (open it, keep the Paraxial + zero-thickness plate + UDS
     recipe: Sub ideal 1, Avg cell = pupil pitch, OPD law 1, dz gain 1,
     dz offset 1.5 µm; Huygens pupil ≥ 512², polarization OFF).

  If the connection fails: the tile was not waiting when the script
  started, or an orphaned headless OpticStudio from an earlier headless
  run holds the licence seat (Task Manager).

## 4b. Non-sequential cross-check  (efficiency, stray orders, power budget)

    python 02_NonSequential_RT_Zemax_Validation_OOP\mdl_nsc_validation.py probe
    python 02_NonSequential_RT_Zemax_Validation_OOP\mdl_nsc_validation.py null
    python 02_NonSequential_RT_Zemax_Validation_OOP\mdl_nsc_validation.py ladder <run>

The complementary stage (plan and rungs: its `NSC_TRACK.md`): NSC ray
splitting with the RCWA diffraction DLLs answers the order-resolved
efficiency / halo / radiometric questions; `probe` first (ZOS-API member
discovery), `null` (stock blaze vs sinc², the plumbing check), `ladder`
(thin-element validity of DELTA-wide treads at fold depth). Outputs in
`<run>\nsc\<stamp>_<mode>\`. Later rungs: real zone profiles, the
custom chirped diffraction DLL on a custom User Defined Object, the
whole-lens power budget.

## 5. Tape-out package  (GDS + foundry sheet)

    python 03_tapeout_oo\export_gds.py --run <run>            # index encoding
    python 03_tapeout_oo\export_gds.py --run <run> --mode terrace

Run after step 3 so the verified performance lands in the sheet.
Creates `<run>\tapeout\<stamp>_<mode>\`: `<name>.gds` (the lens),
`fabrication_spec.txt` (the sheet for the foundry: optical function,
material + index model, relief, tolerance guidance, encoding + dose map,
performance, SHA256s, full per-ring height map), `height_map.csv/.npz`,
`height_profile.png`, the ring table, `m_final.npy`, `design_config.json`,
`tapeout_info.json`, `README.txt`. Fill the TO-BE-AGREED lines of the
sheet with the foundry (substrate, resist lot, tolerances).

## 6. Iterate  (when the numbers say so)

* Objective / fabrication change → step 1 with a new preset name and a
  new `dll_file_no`, then 2–5 again. Warm start from an existing
  design: `init_design_npy="runs\<run>\m_final.npy"`.
* Rigorous check of the thin-element model (open item): sweep
  `<run>\zone_table.npz` with an RCWA solver, build the correction with
  `mdl.zones.relative_correction_table`, re-design with
  `efficiency_corr_npz=...` (sub-ring sampled table when the design
  uses the sinc rule).

## Regression tests (after any code change)

    python 01_design_oo\tests\test_against_mdl_core.py
    python 02_Rayleigh_Sommerfeld_Validation_OOP\tests\test_against_legacy.py <run>
    python 03_tapeout_oo\tests\test_against_legacy.py <run>
    python 02_NonSequential_RT_Zemax_Validation_OOP\tests\test_mock.py <run>

Each must end with `ALL OK`.
