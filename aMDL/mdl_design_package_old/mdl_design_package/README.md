# MDL design + Zemax UDS validation package

Achromatic Multilevel Diffractive Lens (AMDL/MDL) designed by light
frequency-domain coherence optimization (methodology of Xiao et al.,
*Light Sci. Appl.* **11**:323, 2022) + a Zemax OpticStudio User-Defined
Surface DLL implementing the zone-decomposition staircase model
(approach of the Ansys KB article "Realistic modeling of relief-type
diffractive intraocular lenses using User-Defined Surface DLLs").

## Design specs (as agreed)

| quantity | value |
|---|---|
| Diameter D | 10 mm |
| NA (air) | 0.3  →  f = 15.899 mm |
| Band | 400–1100 nm (uniform in ω) |
| Material | AZ4562 photoresist, Cauchy n = 1.594 + 0.01152/λ² [µm] |
| Max height H | 28 µm |
| Gray levels | 359 × 78 nm |
| Ring width Δ | 0.65 µm (Nyquist λ_min/2NA), N = 7692 rings |

## Key result (be aware!)

The **alias-free** coherence upper bound for these joint specs is
max Jω(F) ≈ 0.034 — the requested bandwidth+NA+size combination cannot
produce a good *continuous-band* white-light achromat (the paper's own
Eq. 7 agrees; its headline ~0.2 figures for cm-scale lenses trace back
to discrete/coarse wavelength sampling). The optimized design reaches
J = 0.0091 and physically behaves as a **harmonic diffractive lens**:
diffraction-limited foci at its resonance wavelengths (600, 700,
1050 nm: 40–58 % focus efficiency, FWHM ≈ λ/2NA) and little in
between. Excellent for laser-line / multi-line use at f/1.6 across a
1 cm aperture; poor for broadband white-light imaging. To get a true
broadband achromat at NA 0.3, shrink the aperture (see
`out/fig1_tradeoff.png`: D = 1–2 mm reaches J = 0.16–0.40).

## Contents

```
mdl_core.py              physics + optimizers (FOM, GA+HJA, Smooth,
                         Gradient, alias-free upper bound)
run_design.py            the design run (seeds -> Search -> Smooth ->
                         Gradient -> polish)
run_verify.py            Rayleigh-Sommerfeld verification + ring-table
                         export
make_plots.py            report figures
validate_bound.py        brute-force validation of the upper bound
mdl_zemax_validation.py  ZOS-API script: builds the system with the UDS
                         DLL, ray checks + Huygens PSFs
out/m_final.npy          designed gray-level vector (7692 ints, 0..359)
out/mdl_rings_1.txt      ring table for the DLL (N, Δ[mm]; h[mm] lines)
out/verify_metrics.json  efficiency/FWHM/Strehl-like per wavelength
out/fig*.png             report figures
dll/us_mdl_rings.cpp     UDS DLL source (zone decomposition staircase)
dll/usersurf.h           reconstructed v1 UDS header (see note below)
dll/us_mdl_rings.dll     cross-compiled Windows x64 DLL (mingw-w64)
dll/test_uds.cpp         native unit-test harness (matches Python model
                         to 1e-9)
```

## Installing the DLL in OpticStudio

1. Copy `dll/us_mdl_rings.dll` **and** `out/mdl_rings_1.txt` into
   `{Documents}\Zemax\DLL\Surfaces\`.
2. In the LDE set the surface type to *User Defined*, DLL
   `us_mdl_rings.dll`.
3. Parameters: `File # = 1`, `Height scale = 1`, `Z sign = ±1`
   (relief orientation), `Parax f` = paraxial focal length for solves
   (0 = plane plate).
4. Or run `mdl_zemax_validation.py` (needs `zos_connection.py` from the
   project) to build + analyze the system automatically.

**usersurf.h note**: the header here is a faithful reconstruction of
the classic v1 interface. If OpticStudio rejects or misreads the DLL,
rebuild against the `usersurf.h` from your own installation
(`{Documents}\Zemax\DLL\Surfaces\usersurf.h`):
`cl /LD /O2 us_mdl_rings.cpp` (VS) or the mingw line in the source
header. The .cpp only uses documented fields, so it compiles against
any version.

## Physics notes

* Design FOM: Jω(F) = band-average (uniform in ω, 2001 samples —
  alias-free for a 1 cm aperture) of the normalized on-axis focal
  intensity; identical to the paper's Eq. (4) in the paraxial limit.
* The staircase DLL models vertical side walls as absent
  (thin-relief/zone-decomposition assumption) and refracts on the flat
  ring tops; diffraction then emerges from the wavefront staircase in
  any diffraction-based analysis (Huygens PSF, POP). Geometric spot
  diagrams are not meaningful for this surface.
* Huygens/FFT PSF must sample the pupil finely enough to resolve
  0.65 µm rings over 10 mm — use the highest sampling the license
  allows and expect long runtimes; validate first at reduced aperture
  or on the resonance wavelengths.
