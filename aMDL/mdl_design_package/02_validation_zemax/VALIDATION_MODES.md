# Zemax validation modes: zone / od / rz (+ rzfft, gui)

`mdl_zemax_validation.py <run-folder> [zone|od|rz|rzfft] [gui]`

Four modes, four different *questions* about the same lens, plus a
`gui` flag that runs any of them against the OPEN OpticStudio (see
"gui mode" below). None of them is "the" validation on its own: each
answers a question the others cannot, and each is silent (or actively
misleading) on the questions it was not built for. All numbers below
were measured on the `s3_comb_softmin_overlap` design (D = 10.24 mm,
F = 50.94 mm, 2560 x 2 um rings, H = 15 um) during the 2026-08-26 and
2026-08-31 sessions.

![the three models of the same lens](validation_modes_figs/fig_modes_models.png)
*Fig. 1 -- the three models of one lens, from the actual design vector:
(a) the staircase whose phase the zone DLL imprints; (b) the smooth
carrier the od DLL traces (red) vs the sub-fold residual it cannot see
(green) -- the optimizer's work; (c) the image-sampling problem: FFT
pixels (red squares) vs the RS 950 nm annulus -- why fine PSF shape
belongs to the rz route / RS / BPM.*

---

## 0. THE MEASURED UDS OPD LAW (2026-08-31) -- read this first

OpticStudio's OPD accounting for User-Defined Surfaces is
undocumented; it was CALIBRATED by a four-run experiment (batch ray
trace, null-subtracted against a flat height-scale-0 reference,
compared to the canonical ring table):

| DLL version | mechanism tried            | imprinted OPD (measured) |
|---|---|---|
| v1 | intercept at z=+h, `UD->path` = step | **-1.000 x h**, lambda-FLAT |
| v2 | flat intercept, `UD->path` = OPL/n1  | **0 exactly**            |
| v3 | intercept at z = -(n-1)h/n2          | -(h - h0) to 1.7e-11 um  |
| v4 | intercept at z = +(n-1)h/n2          | +(n-1)(h - h0): EXACT    |

Unique law consistent with all four:

    OPD_contribution = +n2 * z_intercept        (UD->path is IGNORED)

Neither the substrate index nor `UD->path` ever enter the OPD. Both
DLLs now inject their phase THROUGH THE INTERCEPT POSITION
(us_mdl_rings.cpp v4; us_mdl_rings_od.cpp v2 for its Add OPL
parameter), with the exact AZ4562 Cauchy hardcoded in the DLLs -- the
LDE substrate model is out of the phase loop entirely.

Consequences: every zone-mode OPD-based result produced BEFORE v4
(FFT PSFs of 2026-08-03 and 2026-08-26 included) traced the CONJUGATE
phase -h/lam and is retro-invalidated; ray-geometry checks (err = 0)
were always fine; od ray-direction results were never affected.

Diagnostic fingerprints (hard-won -- reuse them): a wrapped-residual
RMS of 0.2887 = 1/sqrt(12) means the two phases compared are
UNRELATED (unmodeled many-wave term, wrong table, or a sign flip),
not slightly off. A traced profile that is perfectly 78nm-quantized,
perfectly dispersive and spans exactly H, yet random against the
table, means WRONG TABLE or a sign error -- cross-correlate offline
against candidates before touching Zemax again.

---

## 1. `zone` -- staircase phase (`us_mdl_rings.dll`, v4)

**Model.** The DLL imprints the transmitted phase of the physical
staircase, phi(rho, lam) = (2 pi / lam)(n(lam) - 1) h(rho), on every
ray via the intercept-position law above; rays exit STRAIGHT (all
focusing lives in the OPD, the defining feature of the zone
decomposition). Diffraction analyses (FFT PSF/MTF, wavefront map)
reconstruct the interference of ALL orders physically, with no order
or fold assumption. The sag still reports the staircase for drawings.

**Equations.** Sag and transmitted phase:

$$z(\rho)=h_i \;\;\text{for}\;\; \rho\in[i\Delta,(i{+}1)\Delta),
\qquad \varphi(\rho,\lambda)=\tfrac{2\pi}{\lambda}\,[n(\lambda)-1]\,h(\rho)$$

Sampling requirements:

$$N_{pupil}\;\ge\;\frac{2D}{\Delta_{ring}}=10240
\;\;(\text{ring Nyquist}),\qquad
\delta_{FFT}=2\,\lambda\,F/\#\;\;(\text{image pitch, FIXED})$$

With FWHM $\approx\lambda F/\#$, the FFT image grid has ~0.5 samples
per FWHM -- the geometric reason FFT PSFs cannot show shapes here.

**What it validates.** The fabrication-facing truth: the exact ring
table that will be written to GDS, the height scale, the phase sign
(Z sign +1, resist -> air), and -- with v4 -- an OPD/wavefront that is
the design's to machine precision (the GUI Wavefront Map now shows the
real folded design phase). Single-ray OPD checks vs the analytic
staircase: exact.

**What is meaningless here.** Every ray-based system analysis: spot
diagrams, ray fans, longitudinal aberration, chromatic focal shift,
geometric MTF, and the layout's "collimator" look. Rays go straight by
construction -- these views show a blind instrument, not a bad lens.
The deep reason: a diffractive surface splits each ray into MANY
orders; a ray trace must pick one direction. zone refuses to pick
(all orders live in the phase; wave analyses recover them), od picks
one (rays bend; ray analyses become meaningful).

**Accuracy limits (measured).** Pupil sampling >= 8192 (1.6 samples
per 2 um ring). FFT image pitch pinned at ~2 lam F/# (3.9-10.7 um) at
ANY sampling -- energy-localization checks, not PSF shapes; OutputSize
only DECIMATES the full +/-16-44 mm window; export via DataGrid ->
numpy npz (~330 KB), never GetTextFile (6.8 GB). ImageDelta below the
default ABORTS the analysis (measured 2026-08-26) -- leave it 0.

**Status note.** The pre-v4 FFT npz sets are invalid (conjugate
phase); one clean zone re-run with the v4 DLL is the outstanding
housekeeping item.

**Cost.** ~1-4 min per wavelength at 8192^2 (FFT).

---

## 2. `od` -- order decomposition (`us_mdl_rings_od.dll`, v2)

**Model.** At load time the ring table is unfolded to the smooth design
OPD and fitted with a 9-term Chebyshev carrier in (r/R)^2. The surface
bends rays by the local grating equation of ONE diffraction order m
(wavelength-scaled phase, the classic V ~ -3.45 diffractive
dispersion), with optional scalar sinc^2 order efficiency as a
transmission. `Order m = 0` = AUTO: each wavelength uses its dominant
(blazed) order m*(lam) = round(alpha), alpha = (n-1) h_fold / lam --
the "camera view" of the multi-order achromat.

**Equations.** The staircase is the fold of a smooth carrier
$O_u(\rho)$; the DOE transmission decomposes into orders

$$t(\rho)=\sum_m c_m\,e^{\,i\,m\,\Phi(\rho)/P},\qquad
\Phi=\tfrac{2\pi}{\lambda_0}O_u,\qquad
|c_m|^2=\mathrm{sinc}^2\!\big(m-\alpha(\lambda)\big)$$

$$\alpha(\lambda)=\frac{[n(\lambda)-1]\,h_{fold}}{\lambda},\qquad
m^*(\lambda)=\mathrm{round}\,\alpha
\;\;(\text{auto order}),\qquad
h_{fold}=\frac{P\,\lambda_0}{n_0-1}=14.377\,\mu m$$

Ray bending (one order per ray) and the resulting focus:

$$n_2\sin\theta' = n_1\sin\theta
+\frac{m}{P}\frac{\lambda}{\lambda_0}\frac{dO_u}{d\rho},
\qquad f(m,\lambda)\propto\frac{1}{m\,\lambda}$$

so order resets sit at half-integer $\alpha$ and the congruence
lattice is $m\lambda = P\lambda_0 = 9.0\,\mu m$. Fig. 1b shows the
carrier (red) and the sub-fold residual (green, wrapped mod
$(n_0{-}1)h_{fold}$): RMS 2.2 um = 0.24 fold -- this residual IS the
design, and od cannot see it.

![od ladder vs RS](validation_modes_figs/fig_zemax_od_ladder.png)
*Fig. 2 -- the od auto ladder (black, congruence-exact) against the RS
diffraction peaks (red): the gap at detuned lines is the residual of
Fig. 1b doing its job.*

**What it validates (measured on this design).** The auto chromatic
ladder matched f prop. 1/(m*(lam) lam) with RMS 0.000 mm over 121
wavelength samples; all 16 order resets at half-integer alpha;
m* = 24..8; congruence-lattice lines (500/600/750/900/1000 nm) shift
exactly 0. Satellite provenance: the 850 nm RS satellite sits near
the adjacent-order (m = 11) focus. In the GUI: converging chromatic
layouts, meaningful longitudinal aberration / focal shift / ray fans.
Since v2 the Add OPL parameter genuinely reaches the OPD (position-
based injection), so od OPD fans are usable; pre-v2 od OPD output was
silently zero (ray-direction results unaffected).

**What it cannot see.** One order per ray; carrier only -- the
optimizer's sub-fold detunings live in the residual the fit ignores
(measured: carrier ladder swings +/-2.6 mm at detuned lines while the
true RS/rz peaks stay within +/-0.11 mm of F at 13/14 lines -- the
difference IS the optimizer's work). Known artifacts: near-axis
carrier-fit bulge (pupil < 0.2; EFFL ~52.8 vs 50.94), scalar sinc^2
efficiencies.

**Cost.** Seconds per order.

---

## 3. `rz` -- Zemax-traced field intensity tiles (the closure mode)

**Model.** For each wavelength the batch ray trace (the one ZOS-API
interface with zero settings plumbing) traces one ray per ring
through the zone DLL; the traced OPD is SELF-CHECKED against the run
folder's ring table (wrapped RMS in waves -- THE Zemax validation
content: does OpticStudio's model of the fabrication file reproduce
the design phase?); the ZEMAX field exp(i 2 pi OPD) is then
propagated to the full I(r, z) tile grid with the IDENTICAL RS-I
kernel and windows as `rs/verify_rzmap.npz`, so any difference from
the RS tiles isolates the surface model, not the propagator. If the
RS tiles exist, a per-wavelength comparison table (z-peak offsets +
tile Pearson correlations) is printed.

**Closure result (2026-08-31, v4 DLL).** OPD self-check RMS =
0.000000 waves at ALL 14 comb lines; traced profile == ring table to
1.8e-11 um; tile peaks IDENTICAL to the RS tile peaks line for line
(dz = 0 everywhere, 850's satellite and 950's +112 um displacement
included); the on-axis map is the paper-Fig.-2e achromatic ridge at F
across 400-1100 nm, from a field traced inside OpticStudio. This is
the intensity-level Zemax <-> RS+BPM agreement.

**Built-in guards.** The script byte-compares the run folder's
mdl_rings_<n>.txt against the copy next to the DLLs and SYNCS it
(file numbers are reused across design iterations -- a stale table
next to the DLLs silently validates the wrong design; the self-check
caught exactly this). The npz is re-saved after every wavelength.

**Output.** zemax/zemax_rzmap.npz (r0grid, zgrid, I_<nm>,
opd_waves_<nm>, phase_rms_waves), fig_zemax_rz_tiles.png,
fig_zemax_onaxis_perlambda.png. **Cost.** ~1 s per wavelength.

**`rzfft` (EXPERIMENTAL, kept for the record).** The through-focus
FFT PSF ladder. Measured 2026-08-31: the DataGrid returned the 256-pt
DECIMATED display grid (dx 171.8 um) and all z-planes came back
BIT-IDENTICAL -- do not trust its output unless the dx echo reads
~2 lam F/# AND per-plane peaks differ. Use `rz`.

---

## 4. `huygens` -- RETIRED (incompatible with the zone DLL)

**Why it cannot work.** OpticStudio's Huygens PSF propagates each
traced ray as a small PLANE-WAVE wavefront segment along the ray's
direction; its contribution at image point r carries the phase

$$U(\mathbf r)=\sum_j A_j\,
e^{\,i[\varphi_j + k\,\hat s_j\cdot(\mathbf r-\mathbf r_j)]}$$

The transverse variation over the image plane comes entirely from the
ray direction $\hat s_j$. The zone DLL's rays exit STRAIGHT, so every
wavelet's phase is constant across the plane and the coherent sum is
the same number at every pixel: a flat field, Strehl 0.000. Verified
at every setting tried (2026-08-26) AND in the 2026-08-01 archives
(8192^2, 512^2, 0.2 um: 262144 samples, ONE distinct value). The
historical Strehl-zero outputs were THIS instrument artifact, never a
design property (the design's shape-Strehl, paper Fig. 4f convention,
averages 0.63 -- see the findings doc's Strehl-conventions section).

---

## gui mode (interactive extension)

Add `gui` to any mode: the script connects to the OPEN OpticStudio
(Programming tab -> Interactive Extension must show "Waiting for
connection" AT LAUNCH), builds the system live, and every analysis
opens as a NATIVE OpticStudio window that STAYS OPEN after the script
exits -- assessment with Zemax's own plotting tools. Checklist when
the connection throws LicenseException (the license is fine if
headless runs work): (1) the waiting screen must be up at the moment
of launch (re-click if it timed out); (2) kill orphaned headless
OpticStudio processes in Task Manager (they hold the seat). Without
`gui` all modes run headless exactly as before; the saved .zos files
open manually for the same windows.

Reading the GUI: **ray windows -> od system, wave windows -> zone/rz
system.** The zone layout legitimately shows a collimator; the
Wavefront Map on the zone system now shows the true design phase.

## Division of labor (the four-way chain)

| question                          | instrument                     |
|-----------------------------------|--------------------------------|
| quantitative diffraction truth    | RS (run_verify.py)             |
| thin-element model error          | BPM (bpm_validate.py)          |
| fabrication file + phase truth    | Zemax `zone` (v4) + `rz` check |
| chromatic/order architecture      | Zemax `od` (+ GUI ray views)   |
| Zemax<->RS intensity agreement    | `rz` tiles (closed: dz = 0)    |
| fine sub-Airy PSF morphology      | RS + BPM (FFT pitch-limited)   |

Practical cadence per design: `rz` always (seconds -- it is the
Zemax regression test), `od` for architecture, one `zone` FFT set
for the record; RS/BPM remain the quantitative reference.

Process caveats that cost us runs (all fixed in the script/DLLs, kept
as warnings): the UDS OPD law of section 0; OpticStudio resolves
relative paths against ITS OWN process cwd (abspath out_dir);
FFT-PSF settings take the `PsfSampling` enum, not `SampleSizes`;
`GetTextFile` dumps the full computation grid (use DataGrid -> numpy);
ModifySettings parses decimals with the SYSTEM LOCALE (integers only);
ring tables are auto-synced from the run folder (never hand-copy);
Ctrl+C only lands when control returns from .NET, and a hard kill can
orphan the headless OpticStudio process (which then also blocks
extension connections).