# NSC_TRACK — the non-sequential OpticStudio model of the MDL (plan, 2026-09-16)

Purpose agreed on 2026-09-16: order-resolved efficiency and stray-order /
halo maps, RCWA-based efficiency (the thin-element check), and the optical
power budget / radiometric analysis of the lens — the incoherent,
energy-flow questions. The focal PSF/MTF closure stays with the sequential
Huygens hybrid (`02_Sequential_RT_Zemax_Validation_OOP`, mode `huy`); the
two stages are complementary and share the run folder.

Why non-sequential for this: OpticStudio's NSC ray splitting is the
framework the Ansys knowledge base prescribes for diffractive elements.
Every diffractive object is a two-layer model — the OBJECT sets the
direction of each order (grating equation / phase gradient), the
DIFFRACTION DLL on its face sets the energy split among the orders, and
the `srg_*_RCWA` DLLs compute that split rigorously from the local
microstructure (Maxwell, not the thin-element phase screen). Rays carry
energy per order to detectors: exactly a power budget. What NSC does NOT
do is interfere the wavefront across the aperture, so the PSF is not its
job (a coherent detector only sums the rays that reach a pixel; a
phase-only screen sends none to the focus) — hence the split of roles.

## Rungs

1. `probe` — ZOS-API member discovery on the installed build (object,
   diffraction data, ray-trace tool, NCE detector readers, DLL parameter
   labels in slot order, sample `user_grating_data_xx.txt`). The
   adapters in `nscval/nsc.py` resolve names at run time from candidate
   lists; the probe log is how the lists get corrected. **Run first.**

2. `null` — RCWA null test on a stock blazed grating (`srg_blaze_RCWA`,
   P = 50 µm, blaze λ₀ = 600 nm in n = 1.632): one trace per order,
   η_m = detector flux / 1 W against sinc²(m − p). PASS = |Δη| ≤ 0.03
   at 600 and 750 nm. Also tells the DLL's x-convention (power at m = +1
   or −1 → `order_sign` of the ladder).

3. `ladder` — the thin-element-validity map: equal-step staircases of
   DELTA-wide treads at the design's fold depth over the FOLD periods
   P = (n − 1) H_f / sin θ (≈ 90 µm at the S3 rim, longer inward; the
   design order p = (n − 1) H_f / λ = 9–23 waves propagates only for
   P > p λ, which is why the short sub-zone periods of `zone_table.npz`
   are not gratings for this purpose). `srg_step_RCWA` vs the scalar
   staircase formula of the same profile (`nscval/tea.py`), per case and
   design wavelength, orders p₀ ± 3 one at a time; ratio tables, a
   convergence pair (Max Order 50 vs 30; the DLL caps at 50 and P/λ
   reaches 450 here — if the pair disagrees the ladder is limited to the
   longer wavelengths / shorter periods and says so).

4. Real zone profiles — `srg_user_defined_RCWA` with each zone's actual
   `h_profile` written to `user_grating_data_xx.txt` (format to be pinned
   from the probe's sample file or the DLL's help), TEA reference =
   `mdl.zones.tea_zone_efficiency`. Output = the `efficiency_corr_npz`
   table for the design (`mdl.zones.relative_correction_table`), i.e.
   the RCWA-corrected re-design loop. (Sub-ring sampled when the design
   uses the sinc rule — mdl docs.)

5. Custom chirped diffraction DLL `us_mdl_rings_diffraction.dll`
   (`UserDiffraction` + `UserParamNames`, ring table by file number as
   the UDS DLLs): at the ray's local (x, y) look up the ring / zone,
   return the per-order energy (TEA, or the rung-4 RCWA table) and the
   local phase derivatives dP/dx, dP/dy so the object's direction law is
   the design's own local grating — the whole lens in ONE NSC object.
   Host first on the stock Diffraction Grating object (direction law
   overridden by the DLL, data[31] = 1 mechanism), then on the custom UDO.

6. Custom User Defined Object `us_mdl_rings_udo.dll` (UDO DLL API,
   distinct from the sequential `UserDefinedSurface3`): a disc of the
   substrate material carrying the staircase relief for drawings and the
   mechanical envelope, its front face hosting the rung-5 DLL; the
   physical Fresnel reflections of the substrate faces (the UDS ignores
   them) enter the power budget here.

7. Whole-lens NSC analyses: collimated source of known power → lens
   object → Detector Rectangles at the focal plane (encircled energy per
   order and per wavelength = the efficiency the design FOM predicts),
   at ±dz, and a large detector for the halo / stray orders; ray
   database for order bookkeeping. Deliverable: the power budget table
   (in-focus, other orders, Fresnel, absorbed/lost) per wavelength and
   the halo maps — the numbers a system engineer needs before the lens
   goes into an optical train with housing and other components.

## Run

    python 02_NonSequential_RT_Zemax_Validation_OOP\mdl_nsc_validation.py probe
    python 02_NonSequential_RT_Zemax_Validation_OOP\mdl_nsc_validation.py null   [--gui]
    python 02_NonSequential_RT_Zemax_Validation_OOP\mdl_nsc_validation.py ladder runs\<run> [--gui]
    python 02_NonSequential_RT_Zemax_Validation_OOP\mdl_nsc_validation.py corr   runs\<run> [--ladder <folder>]
    python 02_NonSequential_RT_Zemax_Validation_OOP\tests\test_mock.py runs\<run>    (offline plumbing)

Outputs: `<run>\nsc\<YYYYMMDD_HHMMSS>_<mode>\` (probe / null without a run
folder: `<package>\runs\_standalone_nsc\`, git-ignored like every run), each with `run_info.json`
(version, command, every setting, the ZOS-API member names that were
resolved), npz/json tables, figures and the `.zos`.

Prerequisites: `srg_*_RCWA.dll` present in the Diffraction-tab DLL list
(Premium/Enterprise; confirmed in the 2024 R1 build on 2026-09-02),
`pythonnet`, the sequential stage folder next to this one (its `zval`
package supplies the connection and the log).

## Facts already on file (zemax_doe_primitives.md, 2026-09-02)

* Diffraction tab: Split = "Split by DLL function", DLL, Start/Stop Order,
  then per-DLL parameters in twin Reflect | Transmit columns (keep them
  identical). Labels are DLL cosmetics over numbered slots — address by
  slot (`nscval/dlls.py` holds the verbatim slot maps of blaze and step).
* srg family: 1-D gratings, planar substrate, single period per object,
  Max Order ≤ 50, P ≲ 100 λ recommended, deep gratings need # Layer and
  Max Order raised together; Stochastic mode avoids ray-count explosion;
  `Only these orders` bitmask; coating files must be UTF-8.
* Depth for RCWA is the PHYSICAL relief height (the (n − 1) scaling is
  the phase, not the geometry) — corrected here against an earlier note.

## Probe log (OpticStudio 2024 R1, 2026-09-16, first real run)

* `ZOSAPI.Editors.NCE.ObjectType` lists `DiffractionGrating`,
  `UserDefinedObject`, `SourceEllipse`, `DetectorRectangle`,
  `RectangularVolumeGrating`, `RectangularPipeGrating`, `SourceDiffractive`
  (names as spelled by the API; 140 object types in all).
* `ObjectColumn` has `Par1`…`Par250` plus `Comment, Material, RefObject,
  InsideOf, XPosition, YPosition, ZPosition, TiltX, TiltY, TiltZ`.
* `DiffractionSplitType` = `DontSplitByOrder, SplitByTable, SplitByDLL`
  (NOT `SplitByDLLFunction`; adapter order fixed, mock renamed).
* Diffraction Grating Par 11 (Diffract Order) is a DOUBLE cell: the
  `IntegerValue` setter raises `Expected Integer, got 'Double'`. The
  adapter now reads `cell.DataType` and uses the matching setter (nscval
  2026-09-16.02). Sections 3–5 of the probe still to be seen.
* Second probe run (nscval .02) reached the end. `IDiffractionData`
  members: `DLL, GetAvailableDLLs, Split, StartOrder, StopOrder,
  NumberOfParameters, IsDLLRequired, IsDiffractionAvailable,
  Get/SetReflectParameterValue, Get/SetTransmitParameterValue,
  GetReflectParameterName, GetTransmitParamaterName` (sic). No `Face`
  member (one tab per object). The manual's `SplitType / SetTransmitValue /
  GetParameterName` do not exist → adapter rewritten on the real names
  (nscval 2026-09-16.03), mock aligned. The "0 parameter labels" line of
  that run was the old name getter finding nothing, not an empty DLL.
* NSC ray trace tool: `SplitNSCRays, ScatterNSCRays, UsePolarization,
  IgnoreErrors, ClearDetectors, RunAndWaitForCompletion, GetTotalRayEnergy,
  NumberOfCores, RayMultiplier, SetRandomSeed/ResetRandomSeed, SaveRays`.
  NCE readers: `GetDetectorData, GetAllDetectorData(Safe),
  GetDetectorDimensions, GetDetectorSize, GetCoherentData`.
* `{Documents}\Zemax\DLL\Diffractive` (24 entries): srg_blaze_RCWA,
  srg_step_RCWA, srg_step2/3_RCWA, srg_trapezoid(2)_RCWA,
  srg_user_defined_RCWA, srg_GridWirePolarizer_RCWA, hologram_kogelnik,
  Diff2DSample, diff_samp_1, lumerical-sub-wavelength (2023R2, 2024R1).
  No `user_grating_data_*.txt` sample present → rung 4 needs the profile
  file format from the srg_user_defined manual page.
* Third probe (nscval .03): the tab works end to end — `Split` →
  SplitByDLL, DLL set and verified, 23 labels read, slot 1 written and
  read back on both columns, `GetAvailableDLLs` lists 14 DLLs. Cell
  types: source rays Integer, source distance Double, grating Lines/µm
  and Diffract Order Double, detector pixels Integer.
* THE SLOT MAP OF 2026-09-02 WAS SHIFTED BY ONE: the srg DLLs have no
  period slot (slot 1 = Max Order, 2 = Unused, 3 = Fill factor, 4 = Alpha,
  5 = Beta, 6/7 = Coat thick top/side, 8 = # Layer, 9 = Use Coating File,
  10–15 = Index Grate/Env/Coat (R,I), 16 = Rotate, 17 = Interpolation,
  18 = Test Mode, 19 = Only these orders, 20 = Stochastic, 21 = Coat mode,
  22 = NIL Thick). The period comes from the OBJECT's Lines/µm (Par 10 =
  1/P). With the old map the null test would have written P = 50 into
  Max Order and the ladder P = 90 (> the cap of 50). nscval 2026-09-16.04
  resolves every key against the live labels (`dlls.resolve`), echoes
  Par 10 next to the slots and refuses `period_um` as a slot key. The
  step-DLL labels on file are PROVISIONAL until the next probe lists
  them (the probe now prints the labels of every srg_*.dll).
* Fourth probe (nscval .04): verbatim labels of all srg DLLs on file
  (`dlls.py`): step (22: Max Order, Depth, Number of Steps, Layers per
  step, Alpha, coat top/side, Unused, Use Coating File, indices, Rotate,
  Interpolation, Test Mode, "Only theseorders", Stochastic, NIL Thick),
  step3 (slot 4 = Number of Layers), step2 (32 slots: A/B/C of four
  sub-steps, four grating indices), trapezoid / trapezoid2, wire-grid,
  user_defined (9: Max Order, File number, ...).
* FIRST REAL NULL TEST (nscval .04): every order read 0.0000 at both
  wavelengths, 19 s for 14 traces. The tab and slots were right (echo
  matches the labels). Prime suspect: the Diffraction Grating object is a
  LENS-type volume (two faces + edge) and was built with THICKNESS 0 —
  a degenerate slab whose rays are lost silently under IgnoreErrors.
  nscval .05: thickness 1 mm (setting `grating_thickness_mm`, echoed),
  column headers of the object logged (`cell.Header`), the trace tool's
  Succeeded / ErrorMessage / GetTotalRayEnergy logged, detector total
  cross-checked pixel-0 vs per-pixel sum, and a GEOMETRIC PRE-CHECK
  (Split = DontSplitByOrder, object order 1, no DLL) that must put
  ~1.0 of the power on the detector before any DLL trace; it stops the
  run with the diagnosis otherwise.
* Null test with nscval .05 (thickness 1 mm): geometric pre-check
  1.0000 (pixel-0 and per-pixel sum agree, trace succeeded, 1.0 W
  launched, columns confirmed: 5 Thickness, 10 Lines/µm, 11 Diff Order)
  — every Split-by-DLL order still 0.0000. The zero is made by the DLL
  split itself. nscval .06 adds mode `diag`: one wavelength, order +1,
  single-change variants (IgnoreErrors off → error text; # Layer 20;
  all orders at once; polarization off as control; N-BK7 object;
  Zemax's Diff2DSample.dll; srg Test Mode = 1 with a scan for files the
  DLL writes) → diag.json + log. Result pending.
* diag v1 (nscval .06) on the real build: baseline 0, errors_on 0 with
  error text '' (no error is raised), layers_20 0, all_orders 0,
  glass_object 0, test_mode 0 (no file written), Diff2DSample 0 (but
  its T(m,0) defaults are 0 — inconclusive), and no_polarization 1.0000:
  with polarization off OpticStudio cannot split, follows the object's
  Diffract Order geometrically and everything arrives. So the SPLIT
  MECHANISM itself loses the power, silently, whatever the DLL. diag v2
  (nscval .07): NonSequentialData dump (ray-intensity cut-offs, simple
  ray splitting), Diff2DSample with T(1,0) = 1 and its period, near
  (z = 1.5 mm) and back (z = -5 mm) detectors, and the baseline trace's
  ray database (ZRD) read back segment by segment (object, face, status,
  intensity).
* diag v2 (nscval .08): NonSequentialData defaults (min relative 0.001,
  absolute 0, simple splitting off, 500 segments); Diff2DSample with
  T(1,0) = 1 -> 0; near detector 0, back detector 1.0000 = the incident
  beam only (transparent detector), i.e. no reflected orders either: with
  the split on, every ray ENDS at the grating, no error, whatever the
  DLL. ZRD written (8 kB) but the reader was called on IZRDReader
  instead of GetResults() -> fixed in .09; .09 also loads OpticStudio's
  own diffraction sample files, prints their grating object (type,
  material, params, Diffraction tab, coatings) and traces them with
  split + polarization through the same path.
* diag v2 on the real build (.09): ZRD of the baseline: ray 1 has TWO
  segments (launch, hit on the grating) — no children are created at
  the split, no error. Zemax's own `Colorimetry\Example 3 Grating Splits
  up Color.zos` (Diffraction Grating, SplitByTable -1..+1, 0.33 each,
  coatings None) also reads 0 on its detector through our trace path
  (split + polarization), while a sample without a grating traces
  normally (0.7066). So the split itself produces nothing through the
  API in this build, for Zemax's file too. Next (diag v3, .10): segment
  status flags (reader fixed for the enum out-parameter), source
  polarization data set explicitly, SplitByTable on our grating, all
  samples (Diffractives first); GUI cross-check of Example 3 requested.
* diag v3 (.11) — THE CAUSE. Split by TABLE on our grating delivers
  power (0.5000: values written at "slots" 1 and 2 landed on order +1
  and out of range), Zemax's table-split samples deliver power through
  our trace (fringes 0.9998, multiple orders 0.95 + 0.05, Boolean
  0.46 + 0.54); ZRD: 2 segments, status Terminated at the grating;
  source polarization explicit -> still 0; ABSORB detector -> 0. So the
  split works and only the DLL path gives 0: THE DIFFRACTION-TAB
  PARAMETERS ARE INDEXED FROM 0. The 1-based listing showed 22 labels +
  a blank (index 23 does not exist) and never showed index 0, which for
  the srg DLLs is the PERIOD (2026-09-02 transcription was right; the
  "shift" of 2026-09-16 morning was the wrong correction) and for
  Diff2DSample "X Period". Period 0 -> RCWA with no grating -> zero
  efficiency in every order -> children below the cut-off, no error.
  nscval .12: 0-based everywhere (names(), slots, echo "param [i]"),
  period_um back as a DLL key at index 0 AND on the object (Lines/um);
  the probe writes/reads param [0]. Next: probe (index-0 label), null.
* probe .12 / diag .13: index 0 IS `+Period/-Freq (um)`; with it set
  (50) the srg_blaze variants STILL read 0.0000 in every order (beta 89,
  alpha 30/beta 60, layers 20, max order 5, interpolation 1, stochastic
  1, fill 0.5, N-BK7 object, explicit source polarization, ABSORB
  detector, test mode writes nothing under Documents\Zemax); every
  trace costs ~7 us/ray (an RCWA that runs costs ms/ray: the DLL
  returns at once). Diff2DSample with its two periods and T(1,0) = 1
  delivers 0.5000 (T and R both written -> very likely T/(T+R)),
  SplitByTable 1.0000. So the whole Split-by-DLL path is alive through
  the API and the zero is specific to the srg RCWA DLLs. The one thing
  every srg run so far had in common: an object in AIR (or N-BK7) while
  the DLL believed Index Grate 1.632 — a DLL that checks the ray's
  medium against its own indices, or that only handles the ray coming
  from INSIDE the grating material, would return zero. diag .14 adds:
  glass_matched (N-BK7 + Index Grate = n_BK7), flipped_glass /
  flipped_swap / air_flipped (grating face on the exit side), period_5um
  (P/lam 8: fully resolved by 20 harmonics), coat_index_1, max_order_50,
  freq_neg (negative index 0 = lines/um), sample_dll_Tonly (R = 0 ->
  1.0 confirms the normalisation), us/ray per variant, Test Mode watch
  on ProgramData\Zemax, cwd and the application folders; the baseline
  now echoes the grating object (Lines/um, Diff Order, DLL values).
  Decisive in parallel: the GUI trace of diag_baseline.zos (Ray Trace,
  Split + Polarization, Detector Viewer): GUI 0 = physics/parameters,
  GUI > 0 = API-hosted-process issue with the RCWA DLLs.
* Diff2DSample.cpp (Documents\Zemax\DLL\Diffractive, read 2026-09-16
  evening) settles two points. (i) Parameter storage: data[200] = max
  parameters, data[201] = parameter 1 REFLECT, data[202] = parameter 1
  TRANSMIT, ... -- one value pair per parameter, the API's index i is
  the DLL's parameter i+1, and OpticStudio asks the DLL for each name
  TWICE (reflect / transmit); a DLL may rewrite the values it is handed
  in that call (Diff2DSample forces R = T on its first three). (ii) The
  0.5 of sample_dll_T1 is exactly the DLL's own normalisation: it sums
  T and R over all orders and divides when the sum exceeds 1, so
  T(1,0) = R(1,0) = 1 -> 0.5. The DLL path through the API is therefore
  fully alive (sample_dll_Tonly of .14 should read 1.0000). (iii) The
  DLL is handed the approach-side index data[12] and the exit-side
  index data[13] from the OBJECT and its surroundings, independent of
  its own parameters -- which is what the glass_matched / flipped
  variants of .14 probe for the srg DLLs.
* GUI cross-check of diag_baseline.zos (2026-09-16 19:00) — THE ERROR.
  Ray Trace with Split + Polarization, Ignore Errors ON: run time 0.09 s,
  lost energy (thresholds) 0, lost energy (ERRORS) 2.0 = every ray (T and
  R child each counted). Ignore Errors OFF: "Error 10561: NSC group
  surface 1: Geometry error object 2 detected. Start xyz = 0, 0, -10;
  Start lmn = 0, 0, 1" on the first ray, then the trace halts on the
  dialog (0.00 % done, clock running — not a loop; Terminate -> 5e-5 W
  lost = one ray). So the srg DLL raises a ray error on every call; the
  API's IgnoreErrors=False never surfaced it. Ansys article
  42661666095891 ("Simulating diffraction efficiency of surface-relief
  grating using the RCWA method") lists when the srg DLLs "cannot
  calculate and return geometric error": (L + m lam/P)^2 + M^2 equal to
  1, n_env^2 or n_grate^2 for some order m (an order exactly at cut-off),
  grazing incidence > 89 deg, non-physical parameters, RAM. Ours:
  1.632 * 50 / 0.6 = 136.000 EXACTLY -> order 136 at cut-off inside the
  grating medium. Remedy per the article: nudge the period. Also from
  the article: the grating is always on Face 1 of the object; Index
  Grate (R) = 0 means "use the substrate index", Index Env (R) = 0 the
  outside material; "Error Log"/Test Mode non-zero writes
  {Zemax}\DLL\Diffractive\<dll name>.txt; Max Order is the ORDER limit
  (10 before 2023 R2.2, 50 after), the harmonic count follows P/lam
  (P = 50 um at 0.6 um: ~136 propagating harmonics inside -> ms per ray
  without Interpolation; the ladder's 90-180 um periods need
  Interpolation or shorter periods). nscval .15: settings.cutoff_orders
  / check_cutoff (null test stops with the message, ladder skips the
  case and says so), NULL period 50.0 -> 50.5 um, diag keeps 50.0 to
  reproduce and adds period_50p5 and lam_075. Open: the null test's
  0.75 um line (m = 108.8, no coincidence) also read 0 — lam_075 will
  say whether a second cause exists.
* diag .15 / null .15 (2026-09-16 20:31): EVERY srg variant still 0 —
  including period 50.5 um, lam 0.75 um and period 5 um, all off the
  cut-off coincidence — so the coincidence was at most one cause. The
  us/ray column is useless (51 us for every variant, table and
  Diff2DSample included: tool overhead). sample_dll_Tonly = 1.0000
  confirms the T/(T+R) reading. The ZRD shows the ray terminated on
  object 2 FACE 1 (the article: "the grating is always on Face 1").
  THE SECOND CAUSE, from the article's own definition: "Alpha and Beta
  are positive in direction rotating from -z to +x" — the wall angles
  are measured FROM THE VERTICAL, and for the blaze "the depth is
  automatically calculated inside depending on the given parameters
  Alpha and Beta", i.e. depth = fill P / (tan a + tan b). Our recipe
  (2026-09-02) put alpha = atan(depth / P) = 1.08 deg (from the
  SURFACE) and beta = 90: in the DLL's convention a nearly vertical
  wall plus a wall lying flat, depth 0 — non-physical, and "one of the
  most common reasons for geometric errors are a non-physical grating
  parameter". Right-angle sawtooth in srg terms: beta = 0 (vertical
  back wall), alpha = atan(fill P / depth) = 88.92 deg. nscval .16:
  blaze_alpha_deg() returns the srg alpha, NULL beta 0, the log echoes
  the depth the DLL will rebuild; diag baseline = NULL_SETTINGS (50.5,
  88.9/0) with variants old_convention (1.08/90, expected 0), beta_1,
  alpha_neg (mirrored sawtooth), sym_30, period_50_exact (P = 50.0,
  expected 0 if the cut-off condition is real), lam_075. The mock
  follows the same convention (depth from alpha/beta/fill).
* diag .16 (2026-09-16 20:48, partial, still running after 2 h): THE
  RCWA RUNS. With alpha 88.9 / beta 0 the baseline costs 17.6 ms per
  ray (20 layers: 197 ms; Max Order 5: 0.38 ms -> the cost scales as
  (2N+1)^3, so "Max Order" IS the RCWA truncation, not only an order
  limit), old_convention / alpha_neg / sym_30 / interp_1 stay at 51 us
  (refused). stochastic_1 puts 0.925 of the power on the detector and
  fill_05 0.18 at +1 -- yet the deterministic split to +1 (baseline,
  layers_20, beta_1, coat_index_1) still reads 0.0000. So the
  efficiencies are computed and mostly land within the detector when
  the DLL picks the order itself; the question is now WHICH order the
  DLL labels +1 (sign) and whether 20 harmonics at P/lam = 84 (n P/lam
  = 137 propagating orders inside) are anywhere near converged -- they
  are not. nscval .17: diag rewritten around that question (2000 rays,
  ~35 s per trace): +1 / -1 / 0 / all orders on one system, the ZRD of
  the all-orders trace read back into power-per-order (m from the
  child's direction cosine: nsc.zrd_raw / order_histogram), Max Order
  30 and 5, controls, P = 5 um (n P/lam = 13.6 < 20: converged) with
  +1 / -1 / ZRD, lam 0.75, P = 50.0 exact, N-BK7 matched. The null test
  is retuned to a period the RCWA can converge at within the cap:
  P 8.0 um (n P/lam 21.8 / 17.4), Max Order 30, 3000 rays, detector
  40 mm / +-12.5 mm, tol 0.08 (TEA itself ~5 % at P/lam 13).
  CONSEQUENCE FOR THE LADDER: its cases at 45-180 um need n P/lam up to
  ~500 harmonics; the cap is 50. The RCWA can only validate TEA at
  P/lam <~ 20 (P <~ 12 um at 0.6 um), which is where TEA is doubtful
  anyway; the ladder must be redesigned on short periods (next).
* 2026-09-17: dll_direct.py / nscval.direct -- the DLL called through
  ctypes with a hand-built data[] (layout from the help table "Data[]
  values for Bulk Scatter, Diffraction, Surface Scatter DLLs" and
  diff_samp_1.c: [10] lam, [11] transmit/reflect, [12]/[13] indices,
  [14] order, [15]/[16] start/stop, [20..25] E field, [30] energy OUT,
  [31] flag, [33]/[34] phase derivatives, [51+i] parameter i+1 and the
  [201+2i]/[202+2i] reflect/transmit pairs -- both blocks are filled).
  No OpticStudio in the loop: rc -1 is visible, efficiencies are exact,
  a call costs what the RCWA costs. Self-test on Diff2DSample.dll;
  then the srg blaze per order (transmit / reflect, TE / TM) with sinc^2
  beside, glass exit index, the old angle pair and P = 50.0 (both
  expected rc -1), and the Max Order sweep 5..50 (convergence, ms).
  dP/dx at m = +1 is the DLL's sign convention. Verified on Linux
  against a fake srg DLL with the same interface.
* diag .17 (2026-09-17 17:40, P 8 um, Max Order 30, # Layer 1): +1 = 0,
  -1 = 0, ORDER 0 = 0.860, all orders 0.873; 50-70 ms per ray. So the
  RCWA runs and puts everything into the zeroth order: "# Layer" = 1
  staircases the sawtooth into ONE slab, i.e. a flat plate (0.87 = one
  minus two Fresnel reflections at n 1.632). The blaze needs the
  profile sliced into several layers; the cost is linear in the count
  (.16: 197 ms at 20 layers vs 17.6 at 1). The ZRD histogram of .17
  read garbage because zrd_raw kept the leading success flag that
  read_zrd strips (sr[1:]) -- fixed. nscval .19 (patch 2026-09-17.03):
  diag = layer sweep 1/5/10/20 at +1, then -1 / 0 / all / ZRD at the
  settings' count, Interpolation 1 (cost), lam 0.75, P 5 um at Max
  Order 20; 400 rays per trace. NULL_SETTINGS: n_layer 10, 1000 rays.
* diag .18 (2026-09-17 18:16, P 8, Max Order 30): with 5 / 10 / 20
  layers at alpha 83.2 / beta 0 the orders +1, -1 AND 0 all read ~0
  (20 layers: 0.0017 at +1); at 1 layer order 0 held 0.86. So with
  layers the structure diffracts -- but not into the orders of a
  one-wave sawtooth. The depth the DLL builds from (Alpha, Beta) is not
  0.95 um: if the angles are measured from the SURFACE and beta = 0
  means "no second facet", 83.2 deg gives depth = P tan(alpha) = 67 um
  (a 70-wave sawtooth, power in orders far outside +-3 / evanescent),
  which matches every observation, including the 1-layer slab. Cost:
  0.11 / 0.34 / 0.51 / 1.03 s per ray at 1 / 5 / 10 / 20 layers (Max
  Order 30, 61 harmonics). nscval 2026-09-17.03: diag = the four
  candidate pairs (vertical 83.2/0; surface 6.77/0; 6.77/89; 6.77/90)
  at orders +1 and 0 with 5 layers and 300 rays, then for the best:
  order -1, EVERY propagating order -13..+13 from the ZRD (40 rays),
  layers 1/10/20, Interpolation 1 cost. Note: the DLL computes the RCWA
  once per requested ORDER per ray (all_orders of .17 cost 7x a single
  order) -- Start/Stop ranges are expensive.
* diag .19 (2026-09-17 18:53): the four Alpha/Beta pairs. C and D (beta
  89 / 90) refused at 3.4 ms per ray -> the angles ARE from the vertical
  (a wall at 90 is flat) and pair A (83.2 / 0) is the one-wave sawtooth.
  A and B compute (320 ms) and deliver nothing at +1, -1, 0. The ZRD of
  every propagating order (-13..+13, 40 rays): each ray = launch + hit
  on face 1, status Terminated, FULL energy, not one child. So with
  layers the DLL returns 0 (or NaN) for every order, while 1 layer gave
  a slab and (fill 0.5, 1 layer) a binary grating. Interpolation 1
  costs MORE (711 ms/ray), no cheap path. 20 layers: 1.1 s/ray.
  nscval 2026-09-17.04: single-parameter probe of the staircase path at
  5 layers, orders +1 and 0: Beta 1 / 5 (cot(beta) at 0), Index Coat 1
  / n_grate (zero-index coat layer singular), fill 0.9 (zero-width top
  layer), both, and Test Mode with the DLL's error log read back.
* 2026-09-18, diag .20 (single-parameter probe) — THE DLL'S OWN LOG.
  Test Mode = 1 writes DLL\Diffractive\srg_blaze_RCWA_log.txt, and for
  every layered ray it says: "Error: Power conservation. (error =
  -0.200816 %)  Reflect power = 0.0320468  Transmit power = 0.965945",
  followed by the full input echo (period 0.008 mm, 61 harmonics,
  approaching / exit index 1, incident LMN (0,0,1), the 23 parameters).
  So the RCWA is right -- a 5-layer one-wave sawtooth transmits 96.6 %
  -- but the DLL REJECTS the ray when its energy balance misses by more
  than its tolerance (< 0.2 %). A single slab conserves exactly, hence
  order 0 = 0.86 at 1 layer; every staircase carries the truncation
  error of 61 harmonics at P/lam = 13 and is refused. Beta 1/5, Index
  Coat, fill 0.9 change nothing (all ~0.0014 or 0). nscval
  2026-09-18.01: convergence probe with Test Mode on and the DLL log
  read after each trace (50 rays): Max Order 30/40/50 at P 8, 2/3
  layers, polarization off, P 5 um at Max Order 20/30 and 5/10 layers.
  The DLL log also echoes the parameter block it received -- the
  definitive check of the 0-based slot mapping (period 8, Max Order 30,
  fill 1, alpha 83.23, beta 0, # Layer 5, Index Grate 1.632, Env 1).
* diag 2026-09-18.01 (log of 11:52) -- THE CONSERVATION CHECK IS CLEARED,
  THE ORDER IS NOT +1. Max Order 40 and 50 at P 8 um (81 / 101
  harmonics) and every P 5 um case run with NO "Power conservation"
  error in the DLL log; only the 61-harmonic / 5-layer reference still
  fails (-0.2008 %). Detector at +1 (50 rays): 2 layers 0.2997, 3
  layers 0.0000, MO40 5 layers 0.0013, MO50 0.0015, P5 0.004-0.006.
  The scalar staircase (tea.py) of a one-wave blaze: 2 levels 0.405 at
  BOTH +1 and -1 (symmetric binary); 3 levels 0.684 at the blaze order
  and 0 at its mirror; 5 levels 0.875 / 0. The measured pattern is the
  mirror: the srg sawtooth diffracts into m = -1 as OpticStudio labels
  the orders (its facet "descends toward +x"). Polarization off gives
  1.0000 at 20 ms/ray (no DLL split applied -- not informative).
  nscval 2026-09-18.02: single-order traces at -1 (L2, L3, MO40 L5, P5
  L5, P5 L10) and +2 (L3, expect 0.17), then the ZRD histogram of
  -3..+3 for L3, P5 L5 and MO40 L5 with a least-squares verdict against
  the scalar staircase of both signs. If confirmed: LADDER_SETTINGS
  order_sign = -1, null test compares eta(m) with sinc^2(-m - p).
  Also fixed: the '\D' SyntaxWarning (raw docstring).
* diag 2026-09-18.02 (log of 12:25, 17 min) -- FOUND. Requesting the
  label -1 delivers the blaze power: 2 layers 0.361, 3 layers 0.587,
  MO40 5 layers 0.815, P5 MO30 5 layers 0.819, P5 10 layers 0.784;
  the 3-layer label +2 reads 0.117 (scalar 0.171 x Fresnel = 0.161).
  The full-order ZRD histograms (m from the child's direction cosine)
  put that power at l = +lam/P, i.e. the PHYSICAL +1 of OpticStudio's
  grating equation (least squares vs the staircase TEA: 0.003 for +1,
  1.43 for -1): the srg DLL labels its orders mirrored, label m <->
  sin(theta) = -m lam/P. Not a bug, a convention -- the ray directions
  come from the DLL together with the efficiencies, so label and
  direction are consistent, only the sign of the label differs from
  the object's Diffract Order convention. Numbers vs scalar staircase x
  Fresnel (0.942): 5 layers 0.99 (both P 8 / MO40 and P 5 / MO30), 3
  layers 0.91, 2 layers 0.95, 10 layers at 61 harmonics 0.86 (not yet
  converged). Every ray is identical: 40 and 50 rays agree to 4 digits.
  nscval 2026-09-18.03: NULL_SETTINGS -> P 5 um, Max Order 30, 5
  layers, order_sign -1 (label = -m), 20 rays, detector z 30 / half 20;
  the null test traces the PHYSICAL orders -3..+3 at 0.60 and 0.75 um
  (labels mirrored) against the scalar N-level staircase x Fresnel
  (tol 0.08), prints the sawtooth sinc^2 beside it; LADDER_SETTINGS
  order_sign -1; the mock models the mirrored label and the staircase.
  Expected: ~4 min, +1 at 0.60 um ~0.82 vs 0.824, at 0.75 um ~0.73 vs
  0.726, order 0 at 0.75 um ~0.05.
* 2026-09-18 14:39 -- NULL TEST PASSED (nscval 2026-09-18.03, 237 s).
  P 5 um, Max Order 30, 5 layers, labels mirrored, 20 rays. 0.60 um
  (p = 1.0): physical +1 = 0.8234 vs scalar staircase x Fresnel 0.8247
  (diff -0.0013), every other order < 0.004, sum 0.839. 0.75 um
  (p = 0.8): +1 = 0.6508 vs 0.7255 (-0.075, inside tol 0.08), order 0
  0.0458 vs 0.0561, sidebands within 0.009, sum 0.763 vs 0.830. The
  detuned line at P/lam = 6.7 (m = 3 at 27 deg) is where the rigorous
  result drifts from the scalar one -- the pattern is right, the level
  is 8 % lower: the TEA error the ladder is meant to map, not plumbing.
  Geometric pre-check 1.0000. The NSC plumbing is closed: slot map,
  angle convention, cut-off check, conservation check (harmonics),
  layer count, order label sign, deterministic per-ray efficiency.
  NEXT: the ladder cannot run at the fold periods (rim P = 119 um at
  the SWIR 1-inch F/3 design: n P / lam_min = 173 harmonics, cap 50).
  Redesign on short periods: keep the tread DELTA and the riser
  h_s = DELTA sin(theta) / (n - 1) of the local blaze, reduce the
  tread count N' to 4 / 6 / 8 (P' 13 / 20 / 26 um, harmonics 19 / 29
  / 38 at 1.1 um), reference = TEA of the same N'-step staircase x
  Fresnel, 5 rays (identical rays), window +/-3; the fold reset (H_f,
  18 waves) stays a geometric estimate (edge zone ~ sqrt(lam H_f) ~ 6
  um ~ 5 % per fold at the rim).
* nscval 2026-09-18.04 -- THE LADDER REDESIGNED ON SHORT STAIRCASES.
  The S3 design (D 10.24 mm, F 50.94 mm, NA 0.100, ring 2.0 um, fold
  14.4 um, 0.4-1.1 um): rim fold period 89 um = 45 treads, n P/lam_min
  = 360 harmonics -- out of reach (cap 50, cost ~ (2N+1)^3). The ladder
  now keeps the LOCAL geometry, tread DELTA and riser h_s = DELTA
  sin(theta)/(n-1) (0.326 um at the rim, 0.163 um at half radius), and
  shortens the staircase to N' = 3 / 4 / 6 treads (P' 6 / 8 / 12 um):
  same tread/lam and riser/lam physics, a period the RCWA converges on.
  Per line: Max Order = ceil(1.5 n P'/lam) clipped to [20, 50] ('!'
  where the margin is not met: N' 6 at 0.4-0.5 um), design order p0 =
  round((n-1) d'/lam) (0..3), window +/-3, evanescent orders skipped,
  reference = scalar N'-step staircase x Fresnel, 4 rays (identical
  rays), order-label sign measured on the first case (labels +p0 / -p0),
  the tread stretched by the smallest of 0.5..4 % that clears every
  cut-off coincidence (2.0 um treads on a 50-nm comb hit 6/0.4 = 15
  etc.), convergence pair Max Order / Max Order - 10, cost estimate
  printed before tracing (S3, 8 lines: ~28 min). CLI: --treads --slopes
  --na --riser --lam-stride. The fold reset (H_f) is not in the ladder.
  Mock updated (mirrored label, Fresnel on srg_step too): ALL OK.
* 2026-09-18 16:00 -- FIRST LADDER RUN on 20260916_071220_s3_comb_softmin_a1
  (nscval .04, 24 min, 6 cases x 8 lines). Two things were wrong in the
  harness and both are pinned by the data:
  (1) srg_step "Depth (um)" is the SPAN of the staircase (lowest to
      highest level), riser = Depth / (N - 1) -- not N x riser. Every
      returned line fits the scalar N-level staircase only with
      p x N/(N-1): 34 lines, ratio at the design order 0.86-1.04, e.g.
      N4 at 1.1 um: orders 0 / +1 / -3 / +3 = 0.444 / 0.268 / 0.033 /
      0.0065 measured vs 0.44 / 0.30 / 0.033 / 0.0067 scalar x Fresnel.
      (The KB wording "Depth = total staircase height" meant exactly
      that.) So the run measured risers 1.5x / 1.33x / 1.2x the intended
      ones (0.49 / 0.43 / 0.39 um at the rim).
  (2) the srg_step order label is the PHYSICAL order (+1) -- opposite
      to srg_blaze (-1). The auto-sign had been measured on the 0.4 um
      line, which the DLL refused, and defaulted to -1: the tables of
      the run must be read with label = -printed m.
  (3) 14 of 48 lines returned 0 at every order -- the DLL's
      energy-balance refusal, non-monotonic in the harmonic count (N6:
      Max Order 35 fine, 31 refused, 28 partial, 27 refused).
  WHAT THE RUN SAYS ANYWAY (re-read with the right model): for 2.0 um
  treads and risers 0.39-0.49 um the rigorous efficiency at the design
  order is 0.86-0.95 of the scalar staircase x Fresnel over 0.5-1.1 um
  (typically 0.92), window sums 0.90-0.98; no wavelength trend inside
  the converged range. At half slope (0.20-0.24 um risers): 0.90-1.01.
  nscval 2026-09-18.05: Depth = (N' - 1) h_s written to the DLL
  (depth_convention "span"), order_sign +1 for srg_step, auto-sign only
  on a line with power, the design order traced first and a refused
  line retried at Max Order -3/+3/-6/+6/-9/+9 (first that returns power
  kept, 'x' in the table if none), Test Mode on with the DLL log
  (srg_step_RCWA_log.txt) summarized per line, shared DllLog reader in
  nsc.py. Mock models the span convention: ALL OK.
* 2026-09-18 17:08 -- THE LADDER RUNS (nscval .05, 19 min, 6 cases x 8
  lines, 34 lines returned, 14 refused). S3 design, 2.0 um treads,
  risers 0.326 um (rim, sin theta 0.100) and 0.163 um (half radius).
  eta_RCWA / (scalar N'-level staircase x Fresnel) at the design order:
    rim   N3: 0.94 0.93 0.96 0.92 1.03 1.01 0.98 (0.5..1.1 um)
          N4: 1.09 0.87 0.90 0.87 0.97 0.99 0.94
          N6: 0.91 0.90 0.99 0.95 (0.85..1.1 um)
    half  N3: 0.90 1.04 0.93 0.98 0.95 0.98 0.99 1.01 (0.4..1.1 um)
          N4: 1.01 0.93 0.95 0.99 0.98 0.97 0.95
          N6: 0.92 0.91 0.93 0.93 (0.85..1.1 um)
  window sums 0.90-1.08. Convergence pair at N3 / 0.4 um: Max Order 35
  vs 25 differ by 0.0015. The refusals are the DLL's energy-balance
  check with errors of 3-676 % at Max Order >= 41 with 4-6 layers (the
  RCWA itself breaks down there, retries at 47/44/41 all refused): N6
  is out of reach below 0.85 um, N4 at 0.4 um. Every refused line
  logged "Error: Power conservation" (8 per attempt = 4 rays x TE/TM).
  READING: for the S3 geometry the thin-element model overestimates the
  local staircase efficiency by ~5-10 % at the rim and ~2-6 % at half
  radius, with no wavelength trend inside 0.5-1.1 um; the N3 window
  sums above 1 at 1.05-1.1 um say the graded interface reflects less
  than the flat Fresnel factor assumes. nscval 2026-09-18.06: corr.py
  folds the ladder into the design's efficiency_corr_npz contract
  (lam_um, r_um, eta): design-order ratio where p >= 0.75 and eta_ref
  (p0) >= 0.25, window-sum ratio otherwise, cases averaged per slope,
  r = 0 -> 1.0, radius of a slope r = F s / sqrt(1 - s^2) (2550 and
  5120 um); written by the ladder itself and by the new `corr` mode on
  an existing ladder folder (no OpticStudio). From this run:
    lam    r=0   2550   5120 um
    0.40   1.00  0.904  0.916
    0.50   1.00  1.032  1.015
    0.60   1.00  0.994  0.904
    0.70   1.00  0.959  0.925
    0.85   1.00  0.969  0.919
    0.95   1.00  0.975  0.975
    1.05   1.00  0.961  1.019
    1.10   1.00  0.973  1.017
  Next: re-run the design with efficiency_corr_npz pointing at it (a
  few-percent, nearly achromatic discount: J will move little), and
  the fold reset (rung 4, srg_user_defined with the real zone profile)
  remains the open item.
* 2026-09-21 -- THE CORRECTION APPLIED TO THE S3 DESIGN. corr mode on the
  ladder folder -> efficiency_corr.npz (0.904-1.044). Run
  20260921_100830_s3_comb_softmin_a1_corr (file 11): same fold, pipeline
  x 0.909 throughout (J_final 0.1272 vs 0.1400). The 2 x 2 of
  01_design_oo/efficiency_corr_check.py (both vectors, both tables):
  run-3 vector 0.1400 (bare) / 0.1355 (corrected); corrected-run vector
  0.1292 / 0.1272. So the PHYSICS costs 3.2 % (per line 0.925 at 400 nm
  .. 0.994 at 1100 nm, worst line 1100 -> 400 nm) and the re-optimized
  vector is simply a worse GA sample (-6 % under both tables, 84 % of
  the rings moved). Design of record stays run 3 at J 0.1355 corrected.
  The tread-level TEA error is NOT what limits the S3 lens; the fold
  reset (rung 4) is the open item. Next: warm start (init_design_npy =
  run 3's m_final) on the corrected tables, then rung 4.
* 2026-09-21 -- RUNG 4 MOVES OUT OF OPTICSTUDIO: 02_RCWA_Validation_OOP
  (rcwaval 2026-09-21.01). Decision (Stefano): the fold-reset question
  is answered with a Python RCWA on zone_table.npz, the srg DLLs kept as
  a cross-check. Own 1-D solver (Moharam-Gaylord, Rumpf S-matrix, Li's
  rule; numpy + scipy), validated: Fresnel / thin film exact, energy 1 -
  1e-8, EMT limits, grcwa 0.1.2 on the null staircase (TE 0.7721 =
  0.7721, TM 0.8036 vs 0.8067), the srg_step ladder lines within 4.4 %.
  FINDING ON THE srg DLLs: they compute a FREE-STANDING relief (the
  object's medium on both sides): free-standing numbers match them,
  resist -> air numbers do not, and their "Reflect power 0.032" is the
  free-standing 3 %. The srg_blaze null value 0.8234 sits between the
  5-level (0.789) and the 6-level-half-ends (0.832) slicings of the
  sawtooth -- the DLL's staircase rule is undocumented. The lens is
  resist -> air; the two geometries differ by up to 8 % on a 4-tread
  staircase, so the ladder table of 2026-09-18 is the right order of
  magnitude but not the right geometry. First zone sweep on the S3-like
  smoke table (31 zones): P 22 um / 11 rings at 1.05-1.10 um: ratio 0.85
  (the fold wall at a low-order blaze -- exactly what the ladder could
  not see), 0.95-1.01 elsewhere; P 46 um: 0.89-0.97. Instabilities
  found and fixed on the way: the sqrt branch of propagating modes
  (round-off flipped forward modes -> S-matrix blow-up at random N),
  an order exactly at cut-off in the gap medium (P = 50 lam), BLAS
  thread oversubscription across workers (4 workers slower than 1).
* 2026-09-21 11:14 -- THE S3 ZONE SWEEP (rcwaval .01, 19 min on 11
  workers; estimate said 7 -- hyperthreads): 172 zones, 108 solved (879
  cells), 21 wide zones (P 62-312 um) unsolved, 43 zones without a
  well-defined order (2-3-ring fragments and the inner zones 1-11).
  Raw ratio eta_RCWA / (scalar staircase x Fresnel) per cell: median
  0.805, range 0.02-2.6 -- the extremes where the scalar reference is
  small (3.7 on 0.009). Reflection 8-20 % per zone (vs 5.8 % flat):
  the deep relief reflects more. THE HEADLINE: the real zones deliver
  about 80 % of what the scalar model promises -- the fold reset and
  the deep irregular profiles cost ~20 %, seven times the tread-level
  3 % the srg ladder measured. rcwaval 2026-09-21.02: table.py /
  rcwa_corr_table.py turn the cells into the design's table (cells with
  eta_scalar < 0.10 dropped, clipped [0.3, 1.3], gaps interpolated,
  running median over 5 zones in r, unsolved wide zones left to the
  design's interpolation, r = 0 anchor at 1.0) with raw / smoothed maps
  side by side.
* 2026-09-21 14:31 -- LUMERICAL FDTD WITNESS OF THE RCWA (rcwaval .08-.11,
  lumerical_zone_check.py through lumapi v241, 2-D periodic cell, plane
  wave from the substrate, T / R line monitors, `grating` per order).
  Zone 34 at 0.400 um (P 22 um, 11 rings, tallest ring 10.374 um = 43
  wavelengths of resist): sign convention confirmed (Lumerical's order
  is the physical one, n = -m_solver; mirror order 0.009 vs 0.508), T / R
  0.908 / 0.092 against the RCWA's 0.902 / 0.098, but the focusing order
  0.508 against 0.441 (+15 %). The RCWA is converged (110 -> 275
  harmonics: 0.4410 -> 0.4392, rcwa_converge.py). The FDTD is not: the
  Yee grid at 12 nm (20 points per wavelength in the resist) accumulates
  0.93 rad of excess phase over the pillar (0.12 rad over the air path
  beside it), equivalent to solving at n + 0.0049. The RCWA at n + 0.005
  gives TE 0.5253 / TM 0.4962 -- the FDTD gave TE 0.5253 / TM 0.4910, and
  the +-3 window agrees to 1 %. Two solvers agree once the mesh error is
  put into the other one: the sweep's numbers stand, and the FDTD is a
  usable witness only where dn_equivalent is small (the log now prints
  it per cell: 0.0007 at 1.0 um on the same pillar with the same mesh;
  6 nm at 0.4 um still leaves 0.0012). DESIGN FINDING from the same
  table: this zone's efficiency moves +8 % / +16 % / +23 % for dn =
  0.0025 / 0.005 / 0.0075 -- the 17-wave fold-reset pillars make the
  short-end efficiency of the S3 design depend on the resist index to
  +-0.002, inside the uncertainty of a dispersion fit. A measured n(lam)
  of the actual AZ4562 batch should replace the fit before fabrication,
  and the tolerance belongs in the findings doc.
* 2026-09-21 14:38 -- WITNESS CLOSED AT THE LONG END. Zone 34 at 1.000 um
  (same pillar, mesh-equivalent dn 0.0007): FDTD 0.4494 vs RCWA 0.4564
  unpolarized (ratio 0.985; TE 0.4621 / 0.4661, TM 0.4367 / 0.4468), the
  +-3 window order by order (0.034 / 0.035, 0.047 / 0.051, 0.208 / 0.215,
  0.462 / 0.456, 0.038 / 0.036). Zone 158 (P 58 um, 29 rings, h_max 11.3
  um) at 1.000 um, a cell the sweep excluded as "order not well defined":
  the FDTD splits the power 0.386 / 0.182 between orders +5 and +6 --
  the focusing direction falls between two orders of that period, which
  is what the exclusion rule is for. The sweep's table is witnessed at
  both ends of the band on the tallest pillars it contains; the Lumerical
  stage is done unless a specific zone is questioned.
* 2026-09-21 15:11 -- THIRD WITNESS POINT AND THE RUNG-4 TABLE. Zone 158
  at 1.000 um in the RCWA (rcwa_converge.py, no exclusion rule): +5 0.3941,
  +6 0.1791 vs the FDTD's 0.3818 / 0.181 -- 3 % and 1 % on the most
  complex cell in the table (P 58 um, 26 levels). rcwa_corr_table.py on
  the S3 sweep (eta_min 0.10, clip [0.3, 1.3], smooth 5, balance 1e-3):
  879 cells, 782 kept, 97 weak dropped, 24 clipped; kept-ratio median
  0.803. Width-weighted mean of the correction per line: 0.872 (0.40),
  0.868 (0.45), 0.839, 0.821, 0.843, 0.827, 0.808, 0.795 (0.75), 0.778
  (0.85), 0.768, 0.757, 0.784, 0.753, 0.771 (1.10); band mean 0.806.
  The loss grows toward the long end, where a zone spans fewer
  wavelengths and the fold walls are a larger share of the aperture --
  the fold reset, not the tread shape. Table: 14 lines x 109 radii,
  runs\20260916_071220_s3_comb_softmin_a1\rcwa\20260921_151142_table\
  efficiency_corr.npz. Next: warm start of run 3 under this table
  (dll_file_no 13) and the 2 x 2 (efficiency_corr_check.py); expected J
  of run 3 under the table ~0.11 with a tilt against the long lines.
* 2026-09-21 16:00 -- RUNG 4 CLOSED (findings: 02_RCWA_Validation_OOP\
  FINDINGS_2026-09-21_rung4.md). Warm start of run 3 under the RCWA
  table (run 20260921_152146_s3_comb_softmin_a1_rcwa, file 13): 0.1134
  (beta 20) -> Search 0.1258 -> Smooth 0.1012 -> Gradient 0.1130 ->
  Polish 0.1143; 28 % of the rings moved by 9 levels. Its own zone sweep
  (138 zones, 75 solvable, 611 cells, 14 min): median 0.809, per-line
  means 0.856 (0.40) .. 0.743 (1.10), band mean 0.809. The 2 x 2 under
  the new design's own table (efficiency_corr_check.py --table, new in
  2026-09-21.14): run-3 vector 0.1400 bare / 0.1074 corrected, re-designed
  vector 0.1386 / 0.1134; under run 3's table 0.1070 / 0.1143 -- < 1 %
  between the tables, fixed point in one iteration. Physics x 0.767,
  optimizer x 1.056 (balance: worst line 1050 nm 0.0997 -> 0.1094, all 14
  lines within 0.1094-0.1229), scalar cost x 0.990. Candidate design of
  record: file 13 at J 0.1134, pending run_verify / mtf_verify / huy on
  its folder. Pipeline note: Smooth costs 20 % in every run and nothing
  re-enforces the aspect-ratio rule after it (10.4 um pillar on one 2 um
  ring in run 3's zone 34).
