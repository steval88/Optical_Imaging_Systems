"""Every knob of the non-sequential stage, echoed by each run into
run_info.json. Nothing below is read from anywhere else; the CLI
options override single values for one run."""
from __future__ import annotations

import math
from typing import Any, Dict, List

# --- ray trace -------------------------------------------------------------
TRACE_SETTINGS: Dict[str, Any] = {
    # rays per trace: 2e5 gives ~0.2 % statistical noise on a 10 % order
    "analysis_rays": 200_000,
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
# recipe of zemax_doe_primitives.md sec. 5 (2026-09-02)
NULL_SETTINGS: Dict[str, Any] = {
    "dll": "srg_blaze_RCWA.dll",
    "period_um": 50.0,               # ~ the outer local period of the S3 lens
    "lam0_um": 0.60,                 # blaze design wavelength
    "n_grate": 1.632,                # AZ4562 near 600 nm (Index Grate)
    "n_env": 1.0,                    # air (Index Env)
    "fill": 1.0,                     # full sawtooth
    "beta_deg": 90.0,                # vertical back facet
    "max_order": 20,                 # RCWA harmonics
    "orders": list(range(-3, 4)),    # traced one at a time
    "lams_um": [0.60, 0.75],         # blaze-matched and detuned
    "beam_half_mm": 2.0,             # collimated Source Ellipse half width
    "grating_clear_mm": 5.0,         # Diffraction Grating clear semi-diameter
    "detector_z_mm": 100.0,          # orders separate by lam/P * z = 1.2 mm
    "detector_half_mm": 10.0,        # catches |m| <= 3 at 0.75 um (3.6 mm)
    "detector_pixels": 200,
    "tol_eta": 0.03,                 # |eta_RCWA - sinc^2| pass threshold
}


def blaze_alpha_deg(period_um: float, depth_um: float) -> float:
    """Facet angle of a right-angle blaze: tan(alpha) = depth / period."""
    return math.degrees(math.atan2(depth_um, period_um))


def blaze_depth_um(lam0_um: float, n_grate: float, n_env: float = 1.0) -> float:
    """First-order blaze depth: one wave of path at lam0, d = lam0/(n - n_env)."""
    return lam0_um / (n_grate - n_env)


# --- TEA-validity ladder: equal-step staircases, srg_step_RCWA ---------------
# The lens's local blaze at radius r is the fold: a staircase of DELTA-wide
# treads climbing the fold height H_f over the local period
#     P(r) = (n - 1) H_f / sin(theta(r)),   sin(theta) = r / sqrt(r^2 + F^2),
# i.e. ~90 um at the S3 rim (NA 0.1) and longer inward. The design order
# is p = (n - 1) H_f / lam (9..23 waves) and propagates only if P > p lam,
# which is why the ladder runs at the fold periods, not at the short
# sub-zone periods of zone_table.npz. Each case = one equal-step staircase
# of N = P / DELTA steps and depth = depth_frac * H_f, at every design
# wavelength: eta_RCWA(m) for m in [p0 - w, p0 + w] (one trace per
# order) against eta_TEA(m) of the SAME staircase (nscval.tea). The RCWA
# harmonic count is capped at 50 by the DLL (P/lam up to 450 here: the
# convergence pair below says whether that is enough).
LADDER_SETTINGS: Dict[str, Any] = {
    "dll": "srg_step_RCWA.dll",
    "cases": [                       # (period_um, depth_frac of H_fold)
        (90.0, 1.0), (120.0, 1.0), (180.0, 1.0),
        (45.0, 0.5), (60.0, 0.5), (90.0, 0.5),
    ],
    "depth_um": "fold",              # "fold" = h_fold_um of the seed record,
                                     #   or a number [um]
    "step_um": "ring",               # "ring" = ring_width_um of the run
    "lams_um": "design",             # "design" = target comb, or a list
    "n_grate": "design",             # "design" = n_az4562(lam) per line
    "n_env": 1.0,
    "max_order": 50,                 # RCWA harmonics (DLL cap 50)
    "convergence_max_order": 30,     # second value at the first case/line
    "layers_per_step": 1,            # vertical walls
    "alpha_deg": 0.0,
    "order_sign": 1,                 # -1 if the null test finds the blaze
                                     #   power at m = -1 (DLL x convention)
    "order_window": 3,               # m in [p0 - w, p0 + w]
    "beam_half_mm": 1.0,
    "grating_clear_mm": 3.0,
    "detector_z_mm": 50.0,
    "detector_half_mm": 25.0,        # sin(theta) ~ 0.1..0.3 -> x <= 16 mm
    "detector_pixels": 100,
}

# --- what the probe inserts --------------------------------------------------
PROBE_SETTINGS: Dict[str, Any] = {
    "dll": "srg_blaze_RCWA.dll",
    "diffractive_dir_sample": "user_grating_data_01.txt",   # print if present
}


def as_list(v: Any) -> List[float]:
    return [float(x) for x in v]
