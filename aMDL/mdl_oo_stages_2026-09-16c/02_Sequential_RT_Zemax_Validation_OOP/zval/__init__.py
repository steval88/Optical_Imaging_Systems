"""
zval -- STAGE 2b: Zemax OpticStudio validation of a designed MDL
=================================================================

Object-oriented refactor (2026-09-09) of mdl_zemax_validation.py.
This package lives in its OWN stage folder, 02_Sequential_RT_Zemax_Validation_OOP,
next to the untouched pre-refactor folder 02_validation_zemax (the
3159-line 2026-09-08.04 script: every result quoted in the findings
documents up to that date is reproducible from it; zos_connection.py
is imported from there). Every run writes into its own time-stamped
subfolder <run-folder>\\zemax\\<YYYYMMDD_HHMMSS>_<mode>\\ (run_info.json
inside: version, command line, settings).

Live modes (see the CLI in ../mdl_zemax_validation.py):

    zone  us_mdl_rings.dll, staircase decomposition, rays STRAIGHT:
          FFT PSF per wavelength (coarse energy localization only).
    od    us_mdl_rings_od.dll, order decomposition: ray-based chromatic
          analyses one order per pass.
    rz    batch-OPD route: Zemax traces one ray per ring through the
          zone DLL, null-subtracted OPD self-check against the ring
          table, RS-I propagation of the ZEMAX field to the I(r,z)
          tiles of rs/verify_rzmap.npz.  The ray-trace closure.
    huy   Huygens PSF / MTF on the HYBRID system (Paraxial f=F, zero-
          thickness glass plate, residual UDS cell-averaged on the
          pupil pitch).  The focal-field closure: OpticStudio's own
          Huygens PSF/MTF reproduce the sub-ring RS PSF (2026-09-08).

Frozen (legacy script only): the POP ladder (rungs 0-4; POP samples a
UDS with ~70 rays and interpolates, closed 2026-09-07), rzfft (FFT
through-focus ladder, decimated DataGrid), the retired 'huygens' mode
on the bare zone surface, the H2/H3/H3b/H4 hypothesis columns, the
dz-gain sweep and the ZOS-API member probes.

Module map
----------
    settings.py    every knob (SETTINGS dicts) + SCRIPT_VERSION
    design.py      Design: geometry/lines from a run folder, ring table,
                   DLL-folder table sync (provenance)
    zos.py         ZosSession (standalone / GUI extension), ZosSystem
                   (surface builder for the zone / od / hybrid variants,
                   DLL parameters, batch traces), analysis plumbing
                   (typed settings, MODIFYSETTINGS fallback, DataGrid
                   and DataSeries readers)
    references.py  numerics shared with 02_validation_rs: RS-I ring
                   tiles, sub-ring focal reference, radial profile,
                   FWHM, Hankel MTF, the rs/ reference loader
    report.py      section banners and the figures
    zone.py, od.py, rz.py, huygens.py   one Analysis class per mode
"""
SCRIPT_VERSION = "2026-09-16.02"     # bumped at every delivery; echoed
                                     # in the configuration section so a
                                     # stale copy is visible at a glance
