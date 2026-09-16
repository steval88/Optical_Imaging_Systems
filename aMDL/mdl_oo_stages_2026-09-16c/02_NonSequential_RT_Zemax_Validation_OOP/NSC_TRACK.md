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
    python 02_NonSequential_RT_Zemax_Validation_OOP\tests\test_mock.py runs\<run>    (offline plumbing)

Outputs: `<run>\nsc\<YYYYMMDD_HHMMSS>_<mode>\` (probe / null without a run
folder: `{Documents}\Zemax_MDL_NSC\`), each with `run_info.json`
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
