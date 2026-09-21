"""
mdl_nsc_validation.py -- STAGE 2c: NON-SEQUENTIAL OpticStudio validation (CLI)
==============================================================================

Thin entry point over the nscval package (see nscval/__init__.py for the
mode descriptions and NSC_TRACK.md for the plan). Run on the OpticStudio
machine, from the package root:

    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py probe
    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py null   [--gui]
    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py diag   (why does the DLL split give 0?)
    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py ladder <run> [--gui]
    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py corr   <run> [--ladder <folder>]

Options (one run only; every setting is echoed and stored in run_info.json):
    --rays N            analysis rays per trace (TRACE_SETTINGS)
    --no-polarization   trace with polarization off
    --period P          null: grating period [um]      --lams 0.6,0.75
    --orders -3,3       null: order range
    --treads 3,4,6      ladder: treads per short staircase (P' = N' x ring width)
    --slopes 1,0.5      ladder: sin(theta) as fractions of the rim NA
    --na 0.1            ladder: rim NA (default: (D/2)/F of the run)
    --riser H           ladder: riser per tread [um] (default: ring width x sin(theta) / (n - 1))
    --lam-stride K      ladder: every K-th design line (default 2)
    --max-order N       RCWA harmonics (ladder default: auto from n P'/lam)
    --ladder <folder>   corr: the ladder folder to read (default: the run's newest)
    --p-min P           corr: design-order ratio above this p (waves), window-sum ratio below (0.75)

<run> is a run folder made by 01_design_oo/run_MDL_design.py; outputs go
to <run>\\nsc\\<YYYYMMDD_HHMMSS>_<mode>\\ (null without a run folder:
<package>\\runs\\_standalone_nsc\\<stamp>_<mode>\\). --gui uses the open OpticStudio through the
Interactive Extension (Programming tab -> Interactive Extension waiting)
and leaves the system open.

Prerequisites: the srg_*_RCWA.dll diffraction DLLs (Premium/Enterprise
licence; {Documents}\\Zemax\\DLL\\Diffractive\\), pythonnet, and the
sequential stage folder (zval: connection + logging) next to this one.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nscval import SCRIPT_VERSION                # noqa: E402
from nscval.base import RunContext               # noqa: E402
from nscval.corr import LadderCorrection         # noqa: E402
from nscval.diag import NullDiag                 # noqa: E402
from nscval.ladder import TeaLadder              # noqa: E402
from nscval.nulltest import NullTest             # noqa: E402
from nscval.probe import Probe                   # noqa: E402

MODES = {"probe": Probe, "null": NullTest, "diag": NullDiag, "ladder": TeaLadder,
         "corr": LadderCorrection}


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MDL non-sequential validation (nscval %s)"
                                 % SCRIPT_VERSION)
    ap.add_argument("mode", choices=sorted(MODES))
    ap.add_argument("run", nargs="?", help="run folder (ladder; optional for null)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--rays", type=int)
    ap.add_argument("--no-polarization", action="store_true")
    ap.add_argument("--period", type=float)
    ap.add_argument("--lams")
    ap.add_argument("--orders")
    ap.add_argument("--treads")
    ap.add_argument("--slopes")
    ap.add_argument("--na", type=float)
    ap.add_argument("--riser", type=float)
    ap.add_argument("--lam-stride", type=int)
    ap.add_argument("--max-order", type=int)
    ap.add_argument("--ladder")
    ap.add_argument("--p-min", type=float)
    return ap.parse_args(argv)


def overrides_from(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {}
    if args.rays:
        o["analysis_rays"] = args.rays
    if args.no_polarization:
        o["use_polarization"] = False
    if args.period:
        o["period_um"] = args.period
    if args.lams:
        o["lams_um"] = [float(v) for v in args.lams.split(",")]
    if args.orders:
        a, b = (int(v) for v in args.orders.split(","))
        o["orders"] = list(range(a, b + 1))
    if args.treads:
        o["treads"] = [int(v) for v in args.treads.split(",")]
    if args.slopes:
        o["slopes"] = [float(v) for v in args.slopes.split(",")]
    if args.na:
        o["na"] = args.na
    if args.riser:
        o["riser_um"] = args.riser
    if args.lam_stride:
        o["lam_stride"] = args.lam_stride
    if args.max_order:
        o["max_order"] = args.max_order
    if args.ladder:
        o["ladder_dir"] = args.ladder
    if args.p_min:
        o["p_min"] = args.p_min
    return o


def main(argv=None, app: Any = None) -> None:
    args = parse(argv)
    if args.mode in ("ladder", "corr") and not args.run:
        raise SystemExit("%s needs a run folder (NA, focal length, wavelengths, ring width)" % args.mode)
    ctx = RunContext(args.run)
    MODES[args.mode](ctx, gui=args.gui, app=app, overrides=overrides_from(args)).main()


if __name__ == "__main__":
    main()
