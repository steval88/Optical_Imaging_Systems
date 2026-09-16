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
