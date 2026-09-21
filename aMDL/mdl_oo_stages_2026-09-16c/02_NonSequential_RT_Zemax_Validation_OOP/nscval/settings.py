"""Every knob of the non-sequential stage, echoed by each run into
run_info.json. Nothing below is read from anywhere else; the CLI
options override single values for one run."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

# --- ray trace -------------------------------------------------------------
TRACE_SETTINGS: Dict[str, Any] = {
    # rays per trace. The srg RCWA costs ~17 ms per ray at Max Order 20 and
    # ~ (2 N + 1)^3 (2026-09-16 .16 log), so 2e5 rays is hours per trace;
    # and linear in "# Layer"; 1000 rays = 3.2 % statistical noise on a full
    # order (tol_eta 0.08), ~10 min per trace at Max Order 30 / 10 layers
    "analysis_rays": 1000,
    "layout_rays": 100,
    "source_power_w": 1.0,           # every efficiency = detector flux / this
    "split_rays": True,              # REQUIRED for the diffraction DLL to act
    "scatter_rays": False,
    # the RCWA DLLs are polarization-resolved; an unpolarized source with
    # polarization ON averages TE/TM. OFF -> the DLL's unpolarized average
    # (if it has one). Echoed; compare both once on the null system.
    "use_polarization": True,
    "ignore_errors": True,
}

# --- null test: blazed grating, srg_blaze_RCWA -------------------------------
# recipe of zemax_doe_primitives.md sec. 5 (2026-09-02), settled by the diag
# series of 2026-09-16..18 (NSC_TRACK.md):
#   * the RCWA must CONVERGE within the DLL's Max Order cap (50) AND pass the
#     DLL's own energy-conservation check (< 0.2 %): P 8 um needs 81 harmonics
#     (1.9 s per ray per order), P 5 um passes at 61 (0.83 s) -> P = 5 um.
#   * the DLL's ORDER LABEL is mirrored: its order m leaves with
#     sin(theta) = -m lam / P along OpticStudio's +x (diag 2026-09-18.02: the
#     label -1 carries 0.815 / 0.819 and the ZRD puts that ray at l = +lam/P;
#     label +2 of a 3-level staircase = 0.117 at l = -2 lam/P). order_sign = -1
#     maps the label to the physical order the scalar reference is written for.
#   * "# Layer" is the staircase count: 5 layers gave 0.99 of scalar x Fresnel,
#     10 layers at 61 harmonics only 0.86 (convergence) -> 5.
#   * every ray is identical (collimated, normal incidence), so the detector
#     fraction IS the efficiency: 40 and 50 rays agreed to 4 digits -> 20 rays.
NULL_SETTINGS: Dict[str, Any] = {
    "dll": "srg_blaze_RCWA.dll",
    "period_um": 5.0,                # n P / lam = 13.6 (0.6 um), 10.9 (0.75 um): no cut-off coincidence
    "lam0_um": 0.60,                 # blaze design wavelength
    "n_grate": 1.632,                # AZ4562 near 600 nm (Index Grate)
    "n_env": 1.0,                    # air (Index Env)
    "fill": 1.0,                     # full sawtooth
    "beta_deg": 0.0,                 # vertical back wall: 0 deg FROM THE VERTICAL in the
                                     #   srg convention (90 was the flat wall that made the
                                     #   profile non-physical -> geometry error on every ray)
    "max_order": 30,                 # RCWA harmonics (truncation N: 2N+1 modes), cap 50
    "n_layer": 5,                    # "# Layer": staircase layers of the blaze (cost linear)
    "order_sign": -1,                # DLL label m  <->  physical order -m (see above)
    "orders": list(range(-3, 4)),    # PHYSICAL orders, traced one at a time (label = sign * m)
    "lams_um": [0.60, 0.75],         # blaze-matched (p = 1.0) and detuned (p = 0.8)
    "rays": 20,                      # analysis rays per trace (deterministic efficiency)
    "beam_half_mm": 2.0,             # collimated Source Ellipse half width
    "grating_clear_mm": 5.0,         # Diffraction Grating clear semi-diameter
    "detector_z_mm": 30.0,           # orders separate by lam/P * z = 3.6 mm at 0.6 um
    "detector_half_mm": 20.0,        # m = 3 at 0.75 um lands at 15.1 mm
    "detector_pixels": 200,
    # reference = scalar N-level staircase (tea.staircase_orders, N = n_layer)
    # x Fresnel transmission of the flat interface 4 n n_env / (n + n_env)^2 =
    # 0.942; measured 2026-09-18: 0.815-0.819 vs 0.824 at the blaze order (5
    # layers), 0.587 vs 0.644 and 0.117 vs 0.161 (3 layers, P/lam 13) -- the
    # rigorous result sits within ~6 % of the scalar one at these P/lam
    "tol_eta": 0.08,
    "grating_thickness_mm": 1.0,     # the object is a volume; 0 traps every ray
    "grating_material": "",          # '' = air both sides (no Fresnel, no refraction)
    "geo_precheck": True,            # trace once WITHOUT the DLL (object order 1 only)
}


def fresnel_transmission(n_grate: float, n_env: float = 1.0) -> float:
    """Power transmission of the flat interface at normal incidence,
    4 n1 n2 / (n1 + n2)^2 (0.942 for 1.632 / 1.0): what the RCWA loses to
    reflection, to first order, on top of the scalar order split."""
    return 4.0 * n_grate * n_env / (n_grate + n_env) ** 2


def cutoff_orders(lam_um: float, period_um: float, indices: List[float],
                  l_cos: float = 0.0, m_cos: float = 0.0, tol: float = 2e-3
                  ) -> List[Tuple[float, float, int]]:
    """The srg RCWA DLLs refuse a ray -- "geometric error" on every ray,
    zero power, no efficiency -- when one diffraction order sits EXACTLY at
    cut-off in any of the three media (Ansys article 42661666095891):

        (L + m lam / P)^2 + M^2 = 1, = n_env^2 or = n_grate^2.

    At normal incidence that is m = n P / lam being an integer. The null
    test of 2026-09-16 hit it exactly: 1.632 * 50 / 0.6 = 136.0.
    Returns [(n, m_real, m_int)] for every index whose m is within `tol`
    of an integer, i.e. the coincidences to design away (nudge the period)."""
    out: List[Tuple[float, float, int]] = []
    for n in [1.0] + [float(v) for v in indices]:
        rhs = n * n - m_cos * m_cos
        if rhs <= 0.0:
            continue
        for sgn in (1.0, -1.0):
            m = (sgn * math.sqrt(rhs) - l_cos) * period_um / lam_um
            k = int(round(m))
            if abs(m - k) < tol and (n, k) not in [(o[0], o[2]) for o in out]:
                out.append((n, m, k))
    return out


def check_cutoff(lam_um: float, period_um: float, indices: List[float], log: Any = None) -> None:
    """Stop the run (SystemExit) when `cutoff_orders` finds a coincidence;
    the message names the order and the period nudge that clears it."""
    hits = cutoff_orders(lam_um, period_um, indices)
    if log is not None:
        log("  cut-off check lam %.4f um, P %.4f um, n %s: m = n P / lam -> %s%s"
            % (lam_um, period_um, indices,
               ", ".join("%.3f" % (n * period_um / lam_um) for n in [1.0] + list(indices)),
               "  COINCIDENCE %s" % hits if hits else "  ok"))
    if hits:
        n, m, k = hits[0]
        raise SystemExit("order %d is exactly at cut-off in the medium n = %.4f at lam = %.4f um, "
                         "P = %.4f um (m = n P / lam = %.6f): the srg RCWA DLL returns a geometric "
                         "error on every ray. Nudge the period (e.g. P = %.3f um) or the wavelength."
                         % (k, n, lam_um, period_um, m, period_um * (1.0 + 0.5 / max(1, abs(k)))))


def blaze_alpha_deg(period_um: float, depth_um: float, fill: float = 1.0) -> float:
    """Alpha of a right-angle blaze IN THE srg CONVENTION.

    The srg DLLs measure Alpha and Beta FROM THE VERTICAL: "positive in
    direction rotating from -z to +x" (Ansys article 42661666095891,
    Fig. 19/22), and for the blaze "the depth is automatically calculated
    inside depending on the given parameters Alpha and Beta":

        depth = fill * P / (tan(alpha) + tan(beta)).

    So a wall at 0 deg is vertical and 90 deg is a wall lying flat. The
    recipe of 2026-09-02 put alpha = atan(depth / P) (from the SURFACE)
    and beta = 90: in the DLL's convention that is a flat facet plus a
    flat back wall, depth 0 -- a non-physical profile, and the DLL
    answered with a geometry error on every ray (GUI, 2026-09-16).
    The right-angle sawtooth is beta = 0 (vertical back wall) and

        alpha = atan(fill * P / depth)      (88.9 deg for P 50.5, d 0.949)."""
    return math.degrees(math.atan2(fill * period_um, depth_um))


def blaze_depth_of_angles(period_um: float, alpha_deg: float, beta_deg: float, fill: float = 1.0) -> float:
    """What the srg blaze DLL makes of (alpha, beta): depth = fill P / (tan a + tan b)."""
    den = math.tan(math.radians(alpha_deg)) + math.tan(math.radians(beta_deg))
    return fill * period_um / den if den > 0.0 else float("inf")


def blaze_depth_um(lam0_um: float, n_grate: float, n_env: float = 1.0) -> float:
    """First-order blaze depth: one wave of path at lam0, d = lam0/(n - n_env)."""
    return lam0_um / (n_grate - n_env)


# --- TEA-validity ladder: short equal-step staircases, srg_step_RCWA ----------
# The lens's local structure at radius r is a staircase of DELTA-wide treads
# (ring width) with a riser
#     h_s = DELTA sin(theta) / (n - 1),   sin(theta) = r / sqrt(r^2 + F^2),
# climbing to the fold height H_f and resetting; the fold period is
#     P(r) = (n - 1) H_f / sin(theta)  (~90 um at the S3 rim, NA 0.10),
# i.e. N = P / DELTA = 45 treads and n P / lam_min = 360 harmonics -- far
# beyond the DLL's cap of 50, and the cost goes as (2 Max Order + 1)^3 per ray
# per requested order (2026-09-18: 0.83 / 1.9 / 3.3 s at 30 / 40 / 50, 5
# layers). So the ladder keeps the LOCAL geometry -- the tread DELTA and the
# riser h_s of the design at a given slope -- and shortens the staircase to
# N' treads (P' = N' DELTA, depth d' = N' h_s): the same diffraction physics
# per tread (tread / lam, riser / lam, wall shadowing) at a period the RCWA
# converges on. Each case = (N', slope fraction of the rim NA), at every
# design line: eta_RCWA(m) for the window m in [p0 - w, p0 + w] around the
# staircase's design order p0 = round((n - 1) d' / lam), one trace per order,
# against eta_TEA(m) of the SAME staircase x the Fresnel transmission
# (nscval.tea; the null test of 2026-09-18 reproduced that reference to 1 %
# for a 5-level sawtooth staircase). The fold reset itself (H_f, 18 waves)
# is NOT in the ladder: its loss is the edge zone ~ sqrt(lam H_f) ~ 3-6 um
# per fold, a geometric estimate noted in NSC_TRACK.md.
LADDER_SETTINGS: Dict[str, Any] = {
    "dll": "srg_step_RCWA.dll",
    "treads": [3, 4, 6],             # N' treads per staircase (P' = N' DELTA)
    "slopes": [1.0, 0.5],            # sin(theta) as a fraction of the rim NA
    "na": "design",                  # rim NA = (D/2) / F of config.json, or a number
    "step_um": "ring",               # tread DELTA: "ring" = ring_width_um of the run
    "riser_um": "design",            # h_s = DELTA sin(theta) / (n(lam_mid) - 1), or a number
    "lams_um": "design",             # "design" = target comb, or a list
    "lam_stride": 2,                 # every k-th design line (14 lines -> 7)
    "n_grate": "design",             # "design" = n_az4562(lam) per line
    "n_env": 1.0,
    # RCWA harmonics: "auto" = ceil(harmonic_margin * n P' / lam) clipped to
    # [20, 50]; a line whose n P'/lam exceeds 50 / harmonic_margin is flagged
    # "unconverged" (traced anyway, marked in the tables)
    "max_order": "auto",
    "harmonic_margin": 1.5,
    "max_order_cap": 50,             # the DLL's cap
    "convergence_max_order": -10,    # second value at the first case/line: max_order + this
    "layers_per_step": 1,            # vertical walls
    "alpha_deg": 0.0,                # srg_step "Alpha (deg)" of the stepped side, FROM THE
                                     #   VERTICAL (srg convention): 0 = vertical risers
    # srg_step "Depth (um)" is the SPAN of the staircase, lowest to highest
    # level: N levels at k Depth / (N - 1), riser = Depth / (N - 1). Measured
    # on the first ladder run (2026-09-18 16:00): every returned line fits the
    # scalar staircase only with p x N/(N-1) -- 34 lines, ratios 0.86-1.04,
    # e.g. N4 at 1.1 um: orders 0/+1/-3/+3 = 0.444/0.268/0.033/0.0065 vs
    # 0.44/0.30/0.033/0.0067. So Depth = (N' - 1) h_s is written to the DLL
    # while the scalar reference keeps N' levels of riser h_s.
    "depth_convention": "span",      # "span" (measured) | "levels" (Depth = N' h_s)
    # the srg_step label is the PHYSICAL order (+1): the same run put the
    # design-order power at label +p0 on every returned line (srg_blaze is
    # the opposite, label -m, because its facet descends the other way).
    # "auto" = measure at the first line that returns power (labels +p0 / -p0)
    "order_sign": 1,
    # the DLL refuses every ray of a line when its RCWA energy balance misses
    # the tolerance; that depends on the harmonic count in a non-monotonic
    # way (first run: 14 of 48 lines returned 0 at Max Order 27..50). When
    # the design order reads 0 the line is retried with Max Order + delta,
    # in this sequence, and the first Max Order that returns power is kept
    "max_order_retries": [-3, 3, -6, 6, -9, 9],
    "test_mode": 1,                  # the DLL writes DLL\Diffractive\srg_step_RCWA_log.txt
    "order_window": 3,               # m in [p0 - w, p0 + w]
    "rays": 4,                       # every ray is identical (collimated, normal incidence)
    "beam_half_mm": 1.0,
    "grating_clear_mm": 3.0,
    "detector_z_mm": 10.0,           # short periods diffract wide: sin(theta) up to 0.93
    "detector_half_mm": 25.0,        #   lands within 25 mm at z = 10 mm
    "detector_pixels": 100,
    "grating_thickness_mm": 1.0,     # the object is a volume; 0 traps every ray
    "grating_material": "",          # '' = air both sides
}


def rcwa_seconds_per_ray_order(max_order: int, n_layers: int) -> float:
    """Cost model of the srg RCWA DLLs, fitted 2026-09-18 (P 5-8 um, 5 layers:
    0.83 s at Max Order 30, 1.9 at 40, 3.3 at 50; linear in the layer count):
    0.83 s x ((2 N + 1) / 61)^3 x (layers / 5), per ray per requested order."""
    return 0.83 * ((2 * max_order + 1) / 61.0) ** 3 * (n_layers / 5.0)


# --- what the probe inserts --------------------------------------------------
PROBE_SETTINGS: Dict[str, Any] = {
    "dll": "srg_blaze_RCWA.dll",
    "diffractive_dir_sample": "user_grating_data_01.txt",   # print if present
}


def as_list(v: Any) -> List[float]:
    return [float(x) for x in v]
