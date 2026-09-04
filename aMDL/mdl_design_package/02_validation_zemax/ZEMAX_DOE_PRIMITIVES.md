# Zemax primitives for modeling a DOE / Multilevel Diffractive Lens (MDL)

Reference for the NSC (non-sequential) primitive choice and the Diffraction-panel
DLL parameter sets, compiled 2026-09-02.

Sources: user screenshots of OpticStudio 2024 R1 (srg_blaze_RCWA panel on the
Diffraction Grating object — parameter names below are VERBATIM from those
screenshots), the Ansys KB article "Simulating diffraction efficiency of
surface-relief grating using the RCWA method" (optics.ansys.com article
42661666095891), "Custom DLLs in OpticStudio" (article 42661741799699), and the
official sample sources `diff_samp_1.c`, `Diff2DSample.cpp`, `us_stand.c`
already analyzed in `VALIDATION_MODES.md` / `MDL_design_findings.md`.

Confidence flags used below:
  [V] verbatim from our screenshots       [KB] from the Ansys KB article
  [?] inferred / to be confirmed with a screenshot of that DLL's panel

---------------------------------------------------------------------------
## 0. TL;DR — the recommendation and its scope

There is NO single Zemax primitive that models the whole MDL rigorously.
Every diffractive object in NSC is a two-layer model:

  LAYER 1 (the OBJECT):    macroscopic geometry + the BASE diffraction law
                           that sets the propagation DIRECTION of each order
                           (grating equation or phase-gradient equation).
  LAYER 2 (Diffraction tab DLL): the local MICROSTRUCTURE, which sets the
                           ENERGY SPLIT among orders (RCWA efficiencies).

The srg RCWA DLLs model a UNIFORM, 1-D, single-period grating — the Period is
one fixed parameter of the DLL, and the KB states explicitly: *1-D gratings
only, planar substrate, "no diffractive lens support"*. So:

* For RCWA characterization (per-zone efficiency, the pre-tape-out insurance
  role): **Diffraction Grating object + srg_*_RCWA.dll**, sweeping Period over
  the local ring spacing. This is the KB-prescribed host for circular
  apertures and the only choice where Layer 1 and Layer 2 describe the SAME
  physical structure.
* For whole-lens behavior in Zemax: our calibrated sequential UDS pair
  (us_mdl_rings v4 = wave/zone, us_mdl_rings_od v2 = ray/od) remains the
  primary model — it ingests the exact ring table with no polynomial fit.
  Binary 1/2/2A cannot do that.

---------------------------------------------------------------------------
## 1. Candidate host objects, compared

All of these activate the Object Properties -> Diffraction tab (confirmed in
the user's build): Diffraction Grating, Binary 1, Binary 2, Binary 2A,
Hologram Lens, Hologram Surface. Also: User Defined Object +
DiffractionGrating.dll (rectangular host, KB-prescribed), Boolean
Native/CAD + Extruded (arbitrary aperture, KB).

### 1.1 Diffraction Grating object  — RECOMMENDED for RCWA work
Editor columns [V]: Radius 1, Conic 1, Clear 1, Edge 1, Thickness, Radius 2,
Conic 2, Clear 2, Edge 2, **Lines/µm**, **Diff Order**, Formula.

Base model: a UNIFORM LINEAR grating on a (possibly curved) circular
substrate. Order m leaves with transverse cosine
l' = (l·n1 + m·λ·T)/n2, T = Lines/µm — i.e. exactly the classical grating
equation, the same law the official diffraction-DLL interface applies via
data[33]=dP/dx (see `Diff2DSample.cpp`).

Why it wins for the RCWA role, against Binary 1/2/2A:

1. **Model consistency.** Its macroscopic law IS a fixed-period linear
   grating — identical to the structure the srg DLL solves. Directions
   (object) and energies (DLL) then describe the same grating: set
   Lines/µm = 1/Period(µm) and the null test closes on itself.
   On Binary 1/2, the object launches orders along the LOCAL gradient of a
   lens-like polynomial phase while the DLL weights them with efficiencies
   of a uniform grating of one fixed period — two different structures
   everywhere except at one radius. Physically inconsistent for a chirped
   lens; acceptable only zone-by-zone.
2. **KB prescription.** The RCWA article names the Diffraction Grating
   object as the host for circular apertures (UDO+DiffractionGrating.dll for
   rectangular). The srg DLLs were written and tested against it.
3. **No extraneous phase.** Nothing else contributes to ray bending, so a
   detector reads pure order efficiencies — the sinc² null test is clean.
4. **Controllability.** One Period knob ↔ one ring-table local period; a
   Period sweep (Universal Plot 1D or MCE) maps efficiency vs zone radius.

Limitations: single period per object (it cannot chirp); 1-D grooves.

### 1.2 Binary 1 object
Base model: XY-polynomial phase Φ = M·Σ Aᵢ·xᵖyᵠ (M = Diff Order). General
free-form carrier; for a rotationally symmetric MDL the XY basis is the
wrong parameterization (needs many terms, breaks exact symmetry).

### 1.3 Binary 2 object — the closest macro-analogue of the MDL carrier
Base model: rotationally symmetric phase Φ = M·Σ Aᵢ·ρ^(2i) (even powers of
normalized radius). This is the natural host for a smooth lens-like carrier
phase, i.e. the same role our od UDS plays in sequential mode.

Why it still loses to the UDS for the full lens:
* It cannot ingest `mdl_rings_<n>.txt`. Representing the 9-term-Chebyshev,
  78 nm-quantized, 15λ₀-folded profile as an even-ρ polynomial is a FIT with
  its own error budget — strictly worse than the ring-exact UDS we
  calibrated to 1.7e-11 µm.
* The phase is smooth and unwrapped: order splitting/fold discontinuities
  (the physics that costs us the 10–15 % TEA error) are exactly what it
  omits — same limitation as our od DLL, without the exactness.
* Attaching an srg DLL to its Diffraction tab does NOT fix this: the DLL's
  Period is a global constant, not the local zone spacing (inconsistency #1
  above).

### 1.4 Binary 2A object
Binary 2 variant on an aspheric substrate (adds asphere/annular terms to the
editor) [?]. Same phase law and hence the same verdict as Binary 2 for our
purpose. (Worth one screenshot of its editor columns if we ever need it.)

### 1.5 Hologram Lens / Hologram Surface objects
Base model: two-construction-point (optically recorded) hologram. The phase
is whatever two recording beams interfere to — elegant for HOEs, but our
profile is lithographic, not holographic; mapping the ring table onto
construction points is an inverse problem with no benefit. Useful only as
another Diffraction-tab activator.

### 1.6 User Defined Object + DiffractionGrating.dll
The rectangular-aperture twin of §1.1 (KB). Same verdict, rectangular
substrate. (This is one of the 4 UDO DLLs the user's build lists.)

### 1.7 What none of them replace
The sequential UDS pair stays the whole-lens truth model in Zemax:
zone (v4) for wave-domain analyses (FFT PSF/MTF, Wavefront Map, rz
regression) and od (v2) for ray-domain analyses (focal shift, fans,
layouts). The NSC/RCWA track supplies the per-zone EFFICIENCY correction
those models cannot know. This division mirrors the object/DLL split above
— it is the same physics dichotomy at a different scale.

---------------------------------------------------------------------------
## 2. The Diffraction tab: framework parameters (all DLLs)

Panel header [V]: **Split: "Split by DLL function"** (required for the DLL
to act; also enable Split NSC Rays in the ray-trace/analysis settings),
**DLL** dropdown, **Start Order / Stop Order**, then per-DLL parameters in
twin **Reflect | Transmit** columns with a **"Copy →"** button.

* Start/Stop Order [V]: the order range actually traced (subset of what the
  DLL computes). Keep |orders| ≤ Max Order.
* Reflect and Transmit columns MUST be kept identical for the srg DLLs [KB]
  — the "Copy →" button exists for exactly this.
* Every energy the DLL returns rides on a ray whose DIRECTION the host
  object computes from its own base law (§1).

### Common srg parameter block (appears in every srg_*_RCWA DLL)
| Parameter [V names] | Meaning [KB] |
|---|---|
| +Period/-Freq (um) | >0: groove period in µm; <0: |value| = spatial frequency in 1/µm |
| Max Order | # of RCWA harmonics retained; accuracy vs time; **cap 50** (was 10 pre-23R2.2) |
| Unused | placeholder row [V] |
| Rotate Grating | groove-line rotation, deg; 0 = lines along Y; + = CCW |
| Interpolation | efficiency cache: 0 = off; N>0 = grid size (min 21); N<0 = alternate slower method |
| Test Mode | diagnostic/verbose mode for debugging a setup [?semantics] |
| Only these orders | bitmask Σ2^n selecting which orders to trace |
| Stochastic mode | ≠0: probabilistic splitting — each ray takes ONE order with prob = its efficiency (Monte-Carlo friendly, no ray-count explosion) |
| Coat mode | how the conformal coating is applied to the profile [?semantics] |
| NIL Thick | residual layer thickness for nano-imprint-type profiles [?semantics] |
| Use Coating File | ≠0: read material dispersion from COATING_xx.dat (**file must be UTF-8**, not UTF-16 LE) |
| Index Grate (R)/(I) | grating-region complex index; **R=0 ⇒ use substrate index**; **negative (e.g. −2) ⇒ read material RCWA02 from coating file** |
| Index Env (R)/(I) | environment-region index; R=0 ⇒ outside/ambient index; negative ⇒ coating file |
| Index Coat (R)/(I) | coating complex index (I ≤ 0 for absorbers) |
| Coat Thick Top(um) | coating thickness on top (adds to total depth) |
| Coat Thick Side(um) | coating thickness on sidewalls |
| # Layer | # of z-slices used by RCWA for sloped profiles (accuracy vs time) |
| (Error Log) [KB] | ≠0: writes diagnostics to a .txt in {Zemax}\DLL\Diffractive\ (not visible on the blaze panel we screenshotted — may be per-DLL) |

---------------------------------------------------------------------------
## 3. Per-DLL shape parameters

### 3.1 srg_blaze_RCWA.dll  [V — full panel screenshotted]
Shape parameters: **Fill factor**, **Alpha (deg)**, **Beta (deg)**.
* NO Depth row: depth is derived internally from Alpha, Beta and the period
  [KB]. For a right-angle blaze (Beta = 90°): depth = Period·tan(Alpha).
* Alpha: facet angle, positive rotating −z → +x; Beta: back-facet angle,
  negative rotating +x → −z (so 90° = vertical back wall).
* Fill factor: ratio of the profiled base to the period (1 = full sawtooth).
* Full panel row order [V]: +Period/-Freq (um), Max Order, Unused, Fill
  factor, Alpha (deg), Beta (deg), Coat Thick Top(um), Coat Thick Side(um),
  # Layer, Use Coating File, Index Grate (R), Index Grate (I), Index Env
  (R), Index Env (I), Index Coat (R), Index Coat (I), Rotate Grating,
  Interpolation, Test Mode, Only these orders, Stochastic mode, Coat mode,
  NIL Thick.

### 3.2 srg_step_RCWA.dll  [V — full panel screenshotted 2026-09-02]
— THE STAIRCASE THAT MATCHES AN MDL ZONE
Verbatim row order (Reflect|Transmit twins, after Start/Stop Order):
+Period/-Freq (um), Max Order, **Depth (um)**, **Number of Steps**,
**Layers per step**, **Alpha (deg)**, Coat Thick Top(um),
Coat Thick Side(um), Unused, Use Coating File, Index Grate (R),
Index Grate (I), Index Env (R), Index Env (I), Index Coat (R),
Index Coat (I), Rotate Grating, Interpolation, Test Mode,
Only these orders, Stochastic mode, NIL Thick.

KB meanings: Depth = total staircase height; Number of Steps = steps per
period, descending toward +x; Alpha = sidewall oblique angle (+ = rotation
+z → −x; 0 = vertical walls); Layers per step = z-slices per step,
meaningful only when Alpha ≠ 0. Index/coat semantics per the common block
(§2). MDL mapping: Number of Steps = levels per fold (78 nm quantization),
Depth = local (n−1)-scaled fold height, Alpha = SEM sidewall angle —
the DLL that carries the fabrication-error study.

**step vs step3 [V]: the two panels are IDENTICAL except one label —
`Layers per step` (step) vs `Number of Layers` (step3).** With the same
parameter set, whatever distinguishes them is INTERNAL (profile
construction and/or slicing semantics — e.g. per-step vs total z-slices,
possibly staircase orientation). Discriminate empirically: render both in
RCWAvisualization.exe with identical inputs (e.g. Period 10, Depth 1,
Steps 4, Alpha 10°, Layers 3) and compare the drawn profiles; or compare
η(m) DLL-vs-DLL on the null system. Until then treat them as
interchangeable at Alpha = 0 (the layers row is inert there per the KB).

### 3.3 srg_step2_RCWA.dll  [V — full panel screenshotted 2026-09-02]
A FOUR-LAYER STACKED grating: each layer i = 1..4 has a geometry triple
(Ai, Bi, Ci) in µm plus its OWN complex index `Index Grate i (R)/(I)`.
Layer semantics confirmed by the grating-tools update log (community
article "OpticStudio grating tools beta function update history"): since
rev 2022-10-28, `Index Grate 2/3/4 (R) = 0` ⇒ that layer takes the
SUBSTRATE index (previously: copied layer 1); setting a layer's index to
zero "turns the layer off". So: srg_step = equal-step staircase, one
material; srg_step2 = unequal/multi-material stack, MAX 4 LAYERS.

Verbatim row order (Reflect|Transmit twins, after Start/Stop Order):
+Period/-Freq (um), Max Order, B1 (um), C1 (um), A2 (um), B2 (um),
C2 (um), A3 (um), B3 (um), C3 (um), A4 (um), B4 (um), C4 (um),
Coat Th Top(um), Coat Th Side(um), Use Coat File, Index Grate 1 (R),
Index Grate 1 (I), Index Coat (R), Index Coat (I), Thick coat (um),
Rotate Grating, Interpolation, Test Mode, Only these orders,
Stochastic mode, Index Grate 2 (R), Index Grate 2 (I),
Index Grate 3 (R), Index Grate 3 (I), Index Grate 4 (R),
Index Grate 4 (I).

A/B/C geometry [?]: NOT documented in the KB. A1 is absent — layer 1 has
only two numbers, layers 2-4 have three — consistent with A = lateral
offset of the layer's ridge (layer 1 anchors the origin), B/C = layer
thickness and ridge width in some order. PIN IT EMPIRICALLY with
RCWAvisualization.exe (renders the profile from the parameters): populate
only layer 1 with distinguishable values (B1=2, C1=0.5, Period 10) to see
which is height vs width, then set A2≠0 on layer 2 and check for lateral
shift. Update this section with the measured semantics.

Panel differences vs blaze [V]: NO Index Env rows (environment fixed to
the outside material), NO # Layer (geometry already layer-rectangular, no
slicing), NO Coat mode / NIL Thick / Unused / Fill/Alpha/Beta/Depth rows;
label variants `Use Coat File`, `Thick coat (um)`. Total depth = sum of
layer thicknesses (implied).

MDL relevance: 4-layer cap rules it out for an 8-level zone (srg_step
stays the zone workhorse); step2 is the tool for fabrication-error
studies with unequal steps, a conformal coat, or a residual-layer /
two-material (NIL-type) stack.

### 3.3b srg_step3_RCWA.dll  [V — full panel screenshotted 2026-09-02]
Verbatim row order (Reflect|Transmit twins, after Start/Stop Order):
+Period/-Freq (um), Max Order, **Depth (um)**, **Number of Steps**,
**Number of Layers**, **Alpha (deg)**, Coat Thick Top(um),
Coat Thick Side(um), Unused, Use Coating File, Index Grate (R),
Index Grate (I), Index Env (R), Index Env (I), Index Coat (R),
Index Coat (I), Rotate Grating, Interpolation, Test Mode,
Only these orders, Stochastic mode, NIL Thick.

Reading: a SINGLE-MATERIAL EQUAL-STEP STAIRCASE. Now that srg_step's own
panel is also verbatim-confirmed (§3.2), the two interfaces are IDENTICAL
except the slicing-row label (`Number of Layers` here vs `Layers per
step` there) — the difference between the DLLs is internal; see the
discrimination test in §3.2. MDL mapping is the same for both: Depth =
(n−1)-scaled fold height span, Number of Steps = levels per fold (78 nm
quantization), Alpha = SEM sidewall angle, Index Grate 1.632/0 (or 0 ⇒
substrate), Index Env 1/0. At Alpha = 0 use either; once the internal
difference is pinned, record the choice here.

Family logic as now established: srg_step(3) = equal-step, one material;
srg_step2 = up-to-4-layer unequal/multi-material stack; srg_blaze =
sawtooth (depth derived); srg_trapezoid(2) = slanted lamellar.
ALL EIGHT srg panels are now verbatim-confirmed (2026-09-02): blaze,
step, step2, step3, trapezoid, trapezoid2, GridWirePolarizer,
user_defined. Remaining [?] items are semantic only: step2 A/B/C
geometry, step-vs-step3 internal difference, trapezoid2 Filling/Wedge,
Test Mode / Coat mode / NIL Thick exact behavior — all pinnable via
RCWAvisualization.exe renders when needed.

### 3.4 srg_trapezoid_RCWA.dll  [V — full panel screenshotted 2026-09-02]
Verbatim row order (Reflect|Transmit twins, after Start/Stop Order):
+Period/-Freq (um), Max Order, **Depth (um)**, **Fill factor**,
**Alpha (deg)**, **Beta (deg)**, Coat Thick Top(um), Coat Thick Side(um),
**Number of layers**, Use Coating File, Index Grate (R), Index Grate (I),
Index Env (R), Index Env (I), Index Coat (R), Index Coat (I),
Rotate Grating, Interpolation, Test Mode, Only these orders,
Stochastic mode, NIL Thick.

KB meanings: Depth = groove height; Alpha = left sidewall slant (+ from
−z to +x); Beta = right sidewall slant (− from +x to −z); Fill factor =
bottom base / period; Number of layers = z-slicing (accuracy vs time,
matters when the walls are sloped). Full common block present, Index Env
included. **Alpha = Beta = 0 ⇒ binary (lamellar) grating — the preferred
dielectric 2-level limit check** (keeps Env/Coat freedom, unlike
GridWirePolarizer).

Note on panel structure (now visible across all eight [V] panels): the
DLLs share a fixed parameter-slot layout and only relabel slots — e.g.
the slot that is `Unused` in step/step3 is `Number of layers` here and
`# Layer` in the KB's naming; blaze's `Unused` sits where step's Depth
does not. Labels drift (Grate/Grating, Coat/Coating, per-DLL
capitalization); when scripting via ZOS-API, address parameters BY SLOT
INDEX after confirming against these verbatim lists, not by name.
ROOT CAUSE (official Help "Creating a New Diffraction DLL", 2025 R1):
every diffraction DLL exports `UserParamNames`, which supplies the label
strings the Diffraction tab shows — labels are DLL-authored cosmetics
over numbered data slots; only the slot indices are structural.

### 3.4b srg_trapezoid2_RCWA.dll  [V — full panel screenshotted 2026-09-02]
Verbatim row order (Reflect|Transmit twins, after Start/Stop Order):
+Period/-Freq (um), Max Order, **Depth (um)**, **Slant Angle (deg)**,
**a (Bottom Base) (um)**, **b (Top Base) (um)**, **Filling**, **Wedge**,
**Number of Layers**, Use Coating File, Index Grating (R),
Index Grating (I), Index Coating (R), Index Coating (I),
**Coat Thick Top (um)**, **Coat Thick Left (um)**,
**Coat Thick Right (um)**, Rotate Grating, Interpolation, Test Mode,
Only these orders, Stochastic mode, NIL Thick.

Reading: a SLANTED trapezoid — different parameterization from the base
trapezoid DLL: bases given directly in µm (`a` bottom, `b` top) plus a
`Slant Angle` that tilts the whole tooth (the AR-waveguide "slanted SRG"
geometry), instead of Alpha/Beta sidewall angles + Fill factor. The
asymmetric coating split `Top/Left/Right` matches the grating-tools
update log entry (2021-01-11) that introduced trapezoid2 with "separate
coating thickness options for top, left side, and right side" —
independent confirmation of the log↔panel mapping. NO Index Env rows
(environment fixed to outside material); label variants `Index Grating`/
`Index Coating` (vs Grate/Coat elsewhere); `Number of Layers` = z-slicing.
[?] rows: `Filling` (redundant with a/b — possibly an alternative
fill-factor input where a/b=0, or a duty-cycle override) and `Wedge`
(possibly asymmetry/wedge of the tooth or substrate). Pin both with
RCWAvisualization renders (vary one at a time from a plain trapezoid)
if this DLL is ever needed — it is NOT on our MDL path (slanted teeth
are an AR-waveguide geometry, not a lithographic staircase).

### 3.5 srg_GridWirePolarizer_RCWA.dll  [V — full panel screenshotted 2026-09-02]
Verbatim row order (Reflect|Transmit twins, after Start/Stop Order):
+Period/-Freq (um), Max Order, **Depth (um)**, **Fill factor**,
Use Coating File, **Index (R)**, **Index (I)**, Rotate Grating,
Interpolation, Test Mode, Only these orders, Stochastic mode, NIL Thick.

Matches the KB exactly: a lamellar (binary) grating of ONE material —
Depth, Fill factor (= grate width / period), single Index (R)/(I) pair
(the wire metal, e.g. Al: complex index with I < 0 by their convention
check). The leanest panel of the family: NO Index Env rows (environment
fixed to outside material), NO Index Coat / coat thickness rows, no
Alpha/Beta, no # Layer (vertical walls, one z-layer). Intended use is
polarization-resolved wire-grid work; incidentally it is also the
SIMPLEST binary-grating check in the family (2-level limit with the
fewest knobs) — but for our dielectric 2-level tests prefer
srg_trapezoid with Alpha=Beta=0, which keeps the Env/Coat freedom.
Not needed for the MDL zone ladder; catalogued for completeness.

### 3.6 srg_user_defined_RCWA.dll  [V — full panel screenshotted 2026-09-02]
Verbatim row order (Reflect|Transmit twins, after Start/Stop Order):
+Period/-Freq (um), Max Order, **File number**, Use Coating File,
Rotate Grating, Interpolation, Test Mode, Only these orders,
Stochastic mode. — That is the WHOLE panel: no index rows, no coat rows,
no NIL Thick; everything geometric/material beyond the period lives in
the profile file. Note Period stays panel-side even here.

KB semantics: File number (1–99) → reads the profile from
`{Documents}\Zemax\DLL\Diffractive\user_grating_data_xx.txt`; a NEGATIVE
value forces a reload after editing the file. This is the escape hatch if
a measured (SEM-derived) zone profile must be fed to RCWA directly — same
"table file + file number" pattern as our UDS ring table, same stale-file
failure class: apply the byte-compare sync discipline.

### 3.7 Non-RCWA diffraction DLLs — OFFICIAL INVENTORY
[Source: Help topic "Creating a New Diffraction DLL" (2025 R1), uploaded
2026-09-04 — the authoritative list of {Zemax}\DLL\Diffractive\:]
* `diff_samp_1.dll` (+ source): sample, simple 1-D grating — idealized
  per-order energies; the didactic skeleton of the interface.
* `Diff2DSample.dll` (+ source): sample, simple 2-D grating — X/Y
  periods, slant, per-order T/R power table.
* `Grid_rect_windows.dll`: 2-D grating = grid of rectangular apertures
  (windows) on a mask, adjustable window size; efficiency from SCALAR
  diffraction theory. Premium/Enterprise.
* `hologram_kogelnik.dll`: Kogelnik theory for holograms — **DEPRECATED**:
  the function is now built into the Hologram 1 and Hologram 2 surfaces.
  Premium/Enterprise.
* the 8 `srg_*_RCWA.dll` (§§3.1–3.6): "simulate 1D gratings … efficiency
  calculated by the algorithm RCWA. Different DLLs provide different
  parametric models to create the grating shape at each period."
  Premium/Enterprise.
* `Lumerical_RCWA_dynamic_link.dll`: **DEPRECATED**, replaced by
  lumerical-sub-wavelength-dynamic-link.dll.
* `lumerical-sub-wavelength.dll`: static bridge — imports a .JSON model
  exported from Lumerical; "can correctly simulate the intensity AND
  polarization state of the output diffraction rays."
  Professional/Premium/Enterprise (Ansys builds only).
* `lumerical-sub-wavelength_dynamic_link.dll`: live bridge — requires
  Lumerical FDTD ≥ 2023 R1 installed on the same PC; Premium/Enterprise
  (Ansys builds only). KB: "Dynamic workflow between Lumerical RCWA and
  Zemax OpticStudio".
* (Sequential side, for reference: Michael Cheng's community RCWA 1-D
  grating UDS; `lumerical-metalens-2024R1.dll` in DLL\Surfaces.)

### 3.8 THE ESCAPE HATCH: a custom diffraction DLL can chirp
[Official interface, same Help topic:] a diffraction DLL exports
`UserDiffraction` + `UserParamNames`. On every diffractive-surface hit
OpticStudio passes the DLL **the ray's x, y, z IN THE OBJECT'S LOCAL
FRAME**, the wavelength, and the order currently being traced; the DLL
returns that order's relative energy — and may OPTIONALLY compute ALL
output ray properties (direction cosines, E-field vector, energy) and
flag them as authoritative (the data[31]=1 mechanism seen in
Diff2DSample.cpp; phase via data[32] in lens units, local phase
derivatives via data[33]/data[34]).

Consequence: the "single global Period / no chirp" limitation (§0, §1.1)
belongs to the STOCK srg DLLs, not to the interface. A custom
`us_mdl_rings_diffraction.dll` could read our ring table, look up the
LOCAL zone at (x,y), and return per-order energies (TEA or precomputed
per-zone RCWA) plus local dP/dx,dP/dy from the table — a chirped,
position-dependent MDL model in one NSC object, unifying the od (ray
direction) and efficiency pictures. We already hold every ingredient:
the interface map from diff_samp_1/Diff2DSample, the ring-table loader
from the UDS DLLs, and the mingw cross-compile chain. NOT scheduled —
the sequential UDS + per-zone srg ladder covers current needs — but
this is the designated route if a full-aperture NSC efficiency model is
ever required (e.g. stray-order/halo maps on a Detector Rectangle).

---------------------------------------------------------------------------
## 4. Operating constraints & readout (srg family)

* 1-D grooves only; planar substrate assumed at the grating; **no
  diffractive-lens (chirped) support** — per-zone use only. [KB]
* Period ≲ 100×λ recommended (harmonic count explodes beyond). Our rim
  local period ≈ 90 µm ≈ 150λ @ 0.6 µm is BORDERLINE — push Max Order to
  50 and convergence-test, or characterize at 2× the true period per
  half-profile symmetry arguments cautiously. Inner zones are worse
  (larger periods) but also nearer the TEA-valid regime where RCWA is not
  needed; the rim is exactly where RCWA matters and it is just inside reach.
* Total depth 14.4 µm at λ≈0.6 µm is a DEEP grating: raise # Layer (step
  slicing) and Max Order together; runtime grows fast — use Interpolation
  caching for sweeps.
* Readout: Programming → User Extensions → **RCWAvisualization.exe**
  (renders the profile + Fourier reconstruction — catches parameter errors
  before physics), Universal Plot 1D (efficiency vs angle/λ/Period), merit
  operands / MCE for sweeps. Detector + Split NSC Rays for in-system use.
* License: srg DLLs require Premium/Enterprise; presence in the DLL
  dropdown (confirmed in the user's 2024 R1 build) = enabled.

---------------------------------------------------------------------------
## 5. Mapping to OUR MDL (null test → zone ladder)

Null test (`rcwa_null_test.zos`, recipe in VALIDATION_MODES.md):
Source Ellipse → **Diffraction Grating object** (Lines/µm = 1/Period,
Diff Order 1) → Detector; DLL = srg_blaze_RCWA; Period 50 µm; blaze depth
λ₀/(n−1) = 0.95 µm ⇒ Alpha = atan(0.95/50) ≈ 1.09°, Beta 90°, Fill 1;
Index Grate 1.632/0, Env 1/0; Start/Stop −3..+3, Max Order 20.
Expect η(m) = sinc²(m−1) at 0.60 µm (η₊₁ ≈ 1); detuned at 0.75 µm expect
η₊₁ ≈ 0.87, η₀ ≈ 0.10.

Zone ladder (after the null closes): switch to **srg_step_RCWA** (or
step3 — identical interfaces, see §3.2; both verbatim-confirmed), Number of
steps = levels per fold, Depth = (n−1)-scaled fold height, sweep Period
down the ring table toward the rim; compare η₊₁(zone) against the BPM
overlap discount (0.776–0.812) — agreement validates BPM as the cheap
bound; divergence at the rim quantifies the extra vectorial penalty for the
tape-out margin.