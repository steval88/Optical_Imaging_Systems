# 2026-09-21 -- rung 4 closed: the fold-reset loss measured, witnessed, and designed against

Continues `MDL_design_findings_2026-09-07.md` (whose last entry is the
tread-level ladder correction of 2026-09-18/21). Day-by-day numbers in
`02_NonSequential_RT_Zemax_Validation_OOP/NSC_TRACK.md`; the tools in
`02_RCWA_Validation_OOP/README.md`.

## 1. The question and the instrument

The ladder table of 2026-09-18 measured the thin-element error of the
TREADS (2 um steps, ~0.3 um risers): 3 % on J. It could not see the fold
reset -- the 14 um wall at every zone boundary, 18 waves deep -- because
the srg DLLs cap the harmonics at 50 and refuse deep layered profiles.
Stage `02_RCWA_Validation_OOP` (rcwaval 2026-09-21.02 .. .11) carries its
own 1-D RCWA (Moharam-Gaylord, S-matrix, Li's rule for TM, any harmonic
count; validated on Fresnel, thin film, EMT limits, grcwa 0.1.2 and the
srg_step ladder lines to 4.4 %) and sweeps the design's own
`zone_table.npz`: substrate -> the zone's ring heights -> air, normal
incidence, the zone width as the period, the focusing order
|m| = round(P sin theta_c / lam), reference = exact scalar staircase
integral x Fresnel 4n/(n+1)^2.

## 2. The measurement (run 3, 20260916_071220_s3_comb_softmin_a1)

172 zones, 108 solvable (879 cells, 19 min on 11 workers): raw ratio
eta_RCWA / eta_scalar median 0.805. Table builder (weak cells with
eta_scalar < 0.10 dropped, clipped [0.3, 1.3], gaps interpolated, running
median over 5 zones in r): width-weighted mean per line

    0.40 0.872 | 0.45 0.868 | 0.50 0.839 | 0.55 0.821 | 0.60 0.843
    0.65 0.827 | 0.70 0.808 | 0.75 0.795 | 0.85 0.778 | 0.90 0.768
    0.95 0.757 | 1.00 0.784 | 1.05 0.753 | 1.10 0.771     band mean 0.806

The loss grows toward the long end, where a zone spans fewer wavelengths
and the fold walls are a larger share of the aperture: the fold reset,
not the tread shape (which the ladder had put at 3 %).

## 3. The witness (Lumerical FDTD 2024 R1 through lumapi, 2-D periodic cells)

* Zone 34 at 0.400 um (P 22 um, tallest ring 10.374 um = 43 wavelengths
  of resist): sign convention confirmed (Lumerical numbers orders
  physically, n = -m_solver; mirror order 0.009 vs 0.508), T / R
  0.908 / 0.092 vs RCWA 0.902 / 0.098, but the focusing order 0.508 vs
  0.441. The RCWA is converged (110 -> 275 harmonics: 0.4410 -> 0.4392).
  The FDTD is not: the Yee grid at 12 nm (20 points per wavelength in the
  resist) accumulates 0.93 rad of excess phase over the pillar --
  equivalent to solving at n + 0.0049 -- and the RCWA at n + 0.005
  reproduces the FDTD: TE 0.5253 / TM 0.4962 vs 0.5253 / 0.4910, the
  +-3 window to 1 %.
* Zone 34 at 1.000 um (mesh-equivalent dn 0.0007): FDTD 0.4494 vs RCWA
  0.4564 (ratio 0.985), the window order by order.
* Zone 158 at 1.000 um (P 58 um, 26 levels, a cell the sweep excludes as
  "order not well defined"): both solvers split the power between
  orders +5 and +6, 0.386 / 0.181 (FDTD) vs 0.394 / 0.179 (RCWA).
Two independent full-wave solvers agree on the tallest pillars of the
table at both ends of the band; the one discrepancy found was a
quantified mesh artefact. The FDTD log now prints the dispersion-
equivalent dn per cell; it is a witness where that number is small.

## 4. Side result: index tolerance of the tall pillars

`rcwa_converge.py --dn` on zone 34 at 0.400 um: eta +8 % / +16 % / +23 %
for dn = 0.0025 / 0.005 / 0.0075. The 17-wave fold-reset pillars make the
short-end efficiency of the S3 design depend on the resist index to
+-0.002 -- inside the uncertainty of a dispersion fit. A measured n(lam)
of the actual AZ4562 batch should replace `mdl/material.py`'s fit before
fabrication, and the tolerance belongs on the spec sheet.

## 5. The design against it (run 20260921_152146_s3_comb_softmin_a1_rcwa, file 13)

Warm start from run 3's vector under the RCWA table (preset
S3_COMB_SOFTMIN_A1_RCWA = S3_COMB_SOFTMIN_A1 + efficiency_corr_npz +
init_design_npy): warm start 0.1134 (softmin beta 20) -> Search 0.1258
-> Smooth 0.1012 -> Gradient 0.1130 -> Polish 0.1143 (beta 300), 18 s;
28 % of the rings moved by 9 levels on average (the cold corrected run of
the ladder table had moved 84 % by 41).

Self-consistency: the new design's own zone table (138 zones, 75
solvable, 14 min) gives median 0.809 and per-line means 0.856 (0.40) ..
0.743 (1.10), band mean 0.809 -- the same loss as run 3's zones, slightly
lower at both ends. THE 2 x 2 under the new design's OWN table
(efficiency_corr_check.py --table):

                          run-3 vector   re-designed vector
    bare tables             0.1400          0.1386
    corrected tables        0.1074          0.1134

and under run 3's table 0.1070 / 0.1143: both vectors move < 1 % between
the tables, one iteration is the fixed point. Physics x 0.767; optimizer
x 1.056; scalar-model cost of the re-design x 0.990. Per line (own
table): run 3 is tilted 0.1193 (400 nm) -> 0.0997 (1050 nm), the
re-design is flat 0.1094 .. 0.1229, worst line 1050 nm up from 0.0997 to
0.1094. The optimizer bought balance, not throughput -- what a softmin
objective is for, and real because the loss it balanced against is
witnessed.

## 6. Readings

1. The thin-element model overstates the S3 design by 23 % on J
   (x 0.767), of which 3 % is the treads (ladder) and the rest the fold
   reset; the physical J of the design of record is 0.107-0.113, not
   0.140. The ~12 % encircled energy of 2026-09-16 (run_verify) becomes
   ~9 % once the folds are counted.
2. Candidate design of record: 20260921_152146_s3_comb_softmin_a1_rcwa
   (file 13), J 0.1134 under its own table, flat across the 14 lines.
   Owed before it replaces run 3: run_verify / mtf_verify (RS, sinc) and
   huy on this folder, and the gdsout package.
3. Pipeline observation: Smooth (S2-4 aspect-ratio reduction) costs
   20 % in both runs (0.1530 -> 0.1197; 0.1258 -> 0.1012) and Gradient +
   Polish rebuild the profile with nothing enforcing the rule afterwards
   -- run 3 ends with a 10.4 um pillar on a single 2 um ring (aspect ~4)
   in zone 34. The final designs of this pipeline do not satisfy the
   aspect-ratio rule the pipeline applies half-way; to be settled with
   the foundry's tolerance lines, and the same pillars carry the index
   sensitivity of section 4.
4. The srg DLL route (rungs 1-3) is closed as a validation instrument for
   this design class; the own RCWA + Lumerical witness replaces it for
   efficiency. NSC rungs 5-7 (chirped DLL, physical UDO, power budget)
   remain for stray light and radiometry.
