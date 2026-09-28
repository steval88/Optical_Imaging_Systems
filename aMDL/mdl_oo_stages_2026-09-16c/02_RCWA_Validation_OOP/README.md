# 02_RCWA_Validation_OOP -- rigorous efficiency of the design's zone profiles (no OpticStudio)

The rung-4 question of the NSC track (`../02_NonSequential_RT_Zemax_Validation_OOP/NSC_TRACK.md`):
how much of the thin-element (TEA) efficiency the real zone profiles deliver,
INCLUDING the fold reset at every zone boundary. The OpticStudio srg DLLs cannot
reach it (harmonic cap 50, energy-balance refusals on deep layered profiles,
and they model a free-standing relief rather than resist on a substrate), so
this stage carries its own 1-D RCWA and sweeps `zone_table.npz` directly.

    rcwaval/rcwa1d.py     the solver: lamellar multilayer, TE + TM, S-matrix
                          recursion, Li's inverse rule for TM, energy-conserving
                          to 1e-8, any harmonic count (numpy + scipy only)
    rcwaval/profiles.py   ring-height profile -> layer stack (one layer per
                          distinct height), test structures
    rcwaval/zonesweep.py  the sweep: focusing order per zone and line, scalar
                          reference (exact staircase integral x Fresnel), the
                          ratio, the design's efficiency_corr.npz, checkpoints
    rcwaval/table.py      the sweep's cells -> the design's table: weak cells
                          dropped (eta_scalar < 0.10), ratios clipped [0.3, 1.3],
                          gaps interpolated, running median over 5 zones in r
    rcwa_zone_sweep.py    CLI of the sweep
    rcwa_corr_table.py    CLI of the table (seconds; the file to give the design)
    lumerical_zone_check.py
                          a few zones through Lumerical FDTD (lumapi, 2-D
                          periodic) as the independent full-wave witness
    rcwa_converge.py      one cell at several harmonic counts (the RCWA's
                          own convergence test; the FDTD's is --dx / --mesh)
    tests/test_rcwa1d.py  the validation record below, re-runnable in 1 s

## Run (from the package root, ZOS_API_Zemax env)

    python 02_RCWA_Validation_OOP\tests\test_rcwa1d.py
    python 02_RCWA_Validation_OOP\rcwa_zone_sweep.py runs\<run> --plan
    python 02_RCWA_Validation_OOP\rcwa_zone_sweep.py runs\<run> [--lam-stride 2] [--workers 11]
    python 02_RCWA_Validation_OOP\rcwa_corr_table.py runs\<run> [--sweep <folder>] [--eta-min 0.10] [--clip 0.3,1.3] [--smooth 5]

`--plan` prints, per zone, the radius, period, rings, distinct levels, the
focusing order at the band ends, how many lines have a well-defined order,
the harmonic range and the estimated seconds -- and the total -- then stops.
The sweep writes into `<run>\rcwa\<stamp>_zones\`: `efficiency_corr.npz`
(`lam_um`, `r_um`, `eta` -- the `efficiency_corr_npz` contract of
`run_MDL_design.py`), `zone_sweep.npz` (eta_rcwa / eta_scalar / ratio per
line x zone), `zone_sweep_results.json` (every solve with TE/TM, the +-3
order window and the energy balance; also the checkpoint: Ctrl+C and
`--resume <folder>` continue), `fig_zone_sweep.png`, `zone_sweep.log`,
`run_info.json`. The correction table then feeds a re-design exactly as the
ladder table did (`01_design_oo\efficiency_corr_check.py` for the 2 x 2).

`rcwa_corr_table.py` turns the sweep's cells into the table the design
should read (`<run>\rcwa\<stamp>_table\efficiency_corr.npz`): the raw
per-cell ratio is exact for its cell but noisy as a design input (a ratio
of two small numbers where the zone barely blazes, and the optimizer's
irregular zones scatter around the physical trend), so weak cells are
dropped, the rest clipped, the gaps interpolated and the radius profile
median-smoothed; `fig_corr_table.png` shows raw and smoothed side by side
and the log prints the per-line width-weighted mean -- the factor J moves
by, to first order. The sweep's own `efficiency_corr.npz` is the
unsmoothed table (kept for reference).

## Lumerical cross-check (lumapi, v241)

    python 02_RCWA_Validation_OOP\lumerical_zone_check.py runs\<run> --lumapi "C:\Program Files\Lumerical\v241\api\python" --check-api
    python 02_RCWA_Validation_OOP\lumerical_zone_check.py runs\<run> --lumapi "..." --zones 34 --auto 1
    python 02_RCWA_Validation_OOP\lumerical_zone_check.py runs\<run> --lumapi "..." --zones 34,80,158 --auto 3

`--dry-run` prints the plan without importing lumapi; `--check-api` imports
it, opens and closes an FDTD session and prints the module path and the
Lumerical version -- the proof the API is reachable. Each cell is a 2-D FDTD
run (one period, periodic in x, PML in y, plane wave from the substrate,
the line through the global source / monitor settings), both polarizations
(polarization angle 90 = E along the grooves = the RCWA's TE; Lumerical's
own log calls that its "TM simulation"), eta at the sweep's focusing order
= transmission x `grating` share. Order sign: Lumerical numbers the orders
physically, the solver's index is the mirror, so the share is read at
n = -m_solver; the mirror side and the +-3 window are logged next to the
sweep's values, and the log prints what was actually simulated (source and
monitor wavelength, mesh cells and dx, accuracy, polarization angle). Every
cell is saved as `<run>\rcwa\<stamp>_lumerical\fsp\zoneNNN_lamL.LLL_pol.fsp`
before it runs (the solver refuses an unsaved project) -- open one in the
CAD to check the geometry. Results: `lumerical_zone_check.json` / `.log`,
`run_info.json`.

First cell, 2026-09-21, zone 34 at 0.400 um (P 22 um, tallest ring
10.374 um = 43 wavelengths of resist): sign confirmed (mirror order 0.009),
T / R 0.908 / 0.092 against the RCWA's 0.902 / 0.098, focusing order 0.508
against 0.441. The RCWA is converged (`rcwa_converge.py`: 0.4410 -> 0.4392
from 110 to 275 harmonics); the FDTD is not -- the Yee grid at 12 nm, 20
points per wavelength in the resist, adds 0.93 rad of phase over the
pillar, the same as solving at n + 0.0049, and the RCWA at n + 0.005
(`rcwa_converge.py --dn 0.005`) reproduces the FDTD: TE 0.5253 / TM 0.4962
against the FDTD's 0.5253 / 0.4910, the +-3 window to 1 %. The log prints
this dispersion-equivalent dn per cell; the FDTD is a witness where it is
small (long wavelengths, low profiles), and no affordable mesh makes it one
on a 10-um pillar at 400 nm (6 nm still leaves 0.0012). Side result: the
same cell moves 8 / 16 / 23 % for dn 0.0025 / 0.005 / 0.0075 -- the
design's short-end efficiency depends on the resist index to +-0.002.
Long end, zone 34 at 1.000 um (mesh-equivalent dn 0.0007): FDTD 0.4494 vs
RCWA 0.4564 (ratio 0.985), the window order by order; zone 158 at 1.000 um
(excluded by the sweep as not well defined) splits 0.386 / 0.182 between
orders +5 and +6 in the FDTD -- the exclusion rule at work.

    python 02_RCWA_Validation_OOP\rcwa_converge.py runs\<run> --zone 34 --lam 0.400 [--margins 1.2,1.5,2,3] [--dn 0.0025,0.005]

## What is solved and what is not

* Structure per zone: substrate (resist index, from the table's `n_real`) ->
  the zone's ring heights as resist steps with air above -> air; normal
  incidence from the substrate; the zone's width is the period (locally
  periodic assumption, the same one the design's zone decomposition makes).
* Focusing order: |m| = round(P sin(theta_c) / lam), sin(theta_c) = r_c /
  sqrt(r_c^2 + F^2); the side is the one the scalar spectrum feeds. A zone
  whose P sin(theta_c) / lam is farther than 0.3 from an integer, or with
  fewer than 4 rings, has no single order at the focusing direction and is
  left out of the table (the design interpolates across it in r).
* Zones with P > `--p-max` (60 um) are not solved and enter the table as 1.0:
  at P/lam > 55 the thin-element model is exact to better than the
  correction's own scatter, and the solver cost ((2N+1)^3 per layer) is
  highest there.
* Reference: the exact scalar integral of the piecewise-constant profile
  (the flat-tread sinc factor the design's S table carries) times the flat-
  interface Fresnel transmission 4n/(n+1)^2. Ratio floored at
  eta_scalar 0.02, clipped to [0, 2] (`mdl.zones.relative_correction_table`).
* Harmonics N = ceil(margin n P / lam), margin 1.2 -- every propagating order
  in the resist plus 20 %; a spot check with `--margin 1.5` on a few zones
  is the convergence test.

## Validation record (tests/test_rcwa1d.py, 2026-09-21)

* Fresnel resist -> air: 0.942342 (exact) TE and TM; thin film in air:
  0.017303 (Airy formula); energy balance 1 - 1e-8 on every structure.
* Subwavelength lamellar grating (P = lam/20) = effective film: TE 0.04696 vs
  0.04661 (<eps>), TM 0.00014 vs 0.00003 (<1/eps>^-1).
* Independent solver (grcwa 0.1.2, 61 harmonics, 2000-point grid) on the
  null-test staircase (5 levels of 0.190 um, P 5 um, 600 nm, n 1.632, free-
  standing): TE 0.7721 / TM 0.8036; this solver 0.7721 / 0.8067.
* The srg_step ladder of 2026-09-18 (short staircases, free-standing): 9
  lines reproduced within 4.4 %. The srg DLLs compute a FREE-STANDING relief
  (the object's medium, air, on both sides): its reflection 3 % matches the
  DLL log's "Reflect power 0.032", and the free-standing numbers match the
  DLL where resist -> air numbers do not. The lens is resist -> air, and the
  two differ by up to 8 % on a 4-tread staircase (0.714 vs 0.656 at 850 nm)
  -- one more reason for this stage.
* The srg_blaze null-test value (0.8234 at 600 nm) is NOT reproduced by a
  5-level equal staircase (0.789 free-standing); the DLL's own slicing of a
  sawtooth into "# Layer" layers is closer to a 6-level profile with half-
  width end levels (0.832 / 0.667 at 600 / 750 nm vs 0.823 / 0.651). The
  staircase rule of that DLL is undocumented; the srg_step numbers, which are
  what the ladder used, agree with this solver.
* Order sign: a height descending with x feeds m > 0 here and m < 0 in
  `nscval.tea.profile_orders`; the sweep converts.

## Cost

Single BLAS thread per process (set by the CLI before numpy loads; with the
default multi-threaded BLAS four workers were slower than one): 0.02 +
0.7 (M/371)^3 s per eigen-decomposition, M = 2N + 1, two per layer. A
22-um zone of 11 levels at 400 nm (N 110): 4 s; a 46-um zone of 21 levels
(N 230): 50 s; a 60-um zone of 30 levels at 400 nm (N 300): ~3 min. The
plan prints the total; `--lam-stride 2` halves it, `--workers` divides it.
Measured on the S3 table (172 zones, 879 tasks, 11 workers on a 12-thread
CPU): 19 min against an estimate of 7 -- hyperthreads are not cores and the
sandbox core the model was fitted on is faster per thread; read the
estimate as a lower bound and count physical cores.
