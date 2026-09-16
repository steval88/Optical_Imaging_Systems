"""
mdl_nsc_validation.py -- STAGE 2c: NON-SEQUENTIAL OpticStudio validation (CLI)
==============================================================================

Thin entry point over the nscval package (see nscval/__init__.py for the
mode descriptions and NSC_TRACK.md for the plan). Run on the OpticStudio
machine, from the package root:

    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py probe
    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py null   [--gui]
    python 02_NonSequential_RT_Zemax_Validation_OOP\\mdl_nsc_validation.py ladder <run> [--gui]

Options (one run only; every setting is echoed and stored in run_info.json):
    --rays N            analysis rays per trace (TRACE_SETTINGS)
    --no-polarization   trace with polarization off
    --period P          null: grating period [um]      --lams 0.6,0.75
    --orders -3,3       null: order range
    --depth D           ladder: staircase depth [um] (default: the run's fold)
    --cases 90:1,120:1  ladder: period_um:depth_frac list
    --max-order N       RCWA harmonics

<run> is a run folder made by 01_design_oo/run_MDL_design.py; outputs go
to <run>\\nsc\\<YYYYMMDD_HHMMSS>_<mode>\\ (null without a run folder:
{Documents}\\Zemax_MDL_NSC\\). --gui uses the open OpticStudio through the
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
from nscval.ladder import TeaLadder              # noqa: E402
from nscval.nulltest import NullTest             # noqa: E402
from nscval.probe import Probe                   # noqa: E402

MODES = {"probe": Probe, "null": NullTest, "ladder": TeaLadder}


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
    ap.add_argument("--depth", type=float)
    ap.add_argument("--cases")
    ap.add_argument("--max-order", type=int)
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
    if args.depth:
        o["depth_um"] = args.depth
    if args.cases:
        o["cases"] = [(float(c.split(":")[0]), float(c.split(":")[1])) for c in args.cases.split(",")]
    if args.max_order:
        o["max_order"] = args.max_order
    return o


def main(argv=None, app: Any = None) -> None:
    args = parse(argv)
    if args.mode == "ladder" and not args.run:
        raise SystemExit("ladder needs a run folder (fold depth, wavelengths, ring width)")
    ctx = RunContext(args.run)
    MODES[args.mode](ctx, gui=args.gui, app=app, overrides=overrides_from(args)).main()


if __name__ == "__main__":
    main()
