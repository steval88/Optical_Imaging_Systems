"""
tradeoff_maps.py -- STAGE 0: design-space trade-offs from the command line.

    python 01_design_oo\\tradeoff_maps.py check --d 50.8 --na 0.1 [--h 15] [--band 400,1100] [--target 0.05]
    python 01_design_oo\\tradeoff_maps.py check --d-inch 2 --fnum 5 --h 45
    python 01_design_oo\\tradeoff_maps.py study --preset PAPER_FIG1 [--no-pair] [--no-sweep]
    python 01_design_oo\\tradeoff_maps.py study --d 50.8 --na 0.1 --h 15        (sweep around a spec)

check : geometry, analytic + numeric ceilings, feasibility verdict (seconds).
study : Fig. 1b/c pair maps and the Fig. 1d (D, H) sweep -> runs\\<stamp>_tradeoff_<name>\\.
Same physics as tradeoff_gui.py (tradeoff/space.py, tradeoff/study.py).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from tradeoff import (PRESETS, Ceiling, Feasibility, LensSpec, StudyConfig,  # noqa: E402
                      __version__)
from tradeoff.space import INCH_MM  # noqa: E402
from tradeoff.study import run_study  # noqa: E402


def spec_from(args: argparse.Namespace) -> Optional[LensSpec]:
    d = args.d if args.d else (args.d_inch * INCH_MM if args.d_inch else None)
    if d is None:
        return None
    lo, hi = (float(v) / 1000.0 for v in args.band.split(","))
    kw: Dict[str, Any] = dict(diameter_mm=d, lam_min_um=lo, lam_max_um=hi, h_max_um=args.h,
                              dh_um=args.dh, ring_width_um=args.delta, material=args.material,
                              name=args.name)
    if args.na:
        kw["na"] = args.na
    elif args.fnum:
        kw["fnum"] = args.fnum
    else:
        raise SystemExit("give --na or --fnum")
    return LensSpec(**kw)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="MDL design-space trade-offs (tradeoff %s)" % __version__)
    ap.add_argument("mode", choices=("check", "study"))
    ap.add_argument("--d", type=float, help="diameter [mm]")
    ap.add_argument("--d-inch", type=float, help="diameter [inch]")
    ap.add_argument("--na", type=float)
    ap.add_argument("--fnum", type=float)
    ap.add_argument("--band", default="400,1100", help="lam_min,lam_max [nm]")
    ap.add_argument("--h", type=float, default=15.0, help="relief height [um]")
    ap.add_argument("--dh", type=float, default=0.078)
    ap.add_argument("--delta", type=float, default=2.0, help="ring width [um]")
    ap.add_argument("--material", default="AZ4562")
    ap.add_argument("--name", default="spec")
    ap.add_argument("--target", type=float, default=0.05, help="target continuous-band J")
    ap.add_argument("--frac", type=float, default=0.55, help="achievable fraction of the ceiling")
    ap.add_argument("--n-rho", type=int, default=256)
    ap.add_argument("--n-w", type=int, default=512)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args(argv)

    if args.mode == "check":
        s = spec_from(args)
        if s is None:
            raise SystemExit("check needs --d or --d-inch")
        print("SPECIFICATION")
        for line in s.describe():
            print("  " + line)
        c = Ceiling.compute(s, numeric=True, n_rho=args.n_rho, n_wavelengths=args.n_w)
        print("FEASIBILITY")
        for line in Feasibility(s, c, args.target, args.frac).lines():
            print("  " + line)
        return
    if args.preset:
        cfg = PRESETS[args.preset]
    else:
        s = spec_from(args)
        if s is None:
            raise SystemExit("study needs --preset or a spec (--d/--d-inch + --na/--fnum)")
        d = s.diameter_mm
        cfg = StudyConfig(name=args.name, base=s, pair_panels=[(d, s.h_max_um)],
                          sweep_d_mm=list(np.round(np.geomspace(d / 10, 4 * d, 16), 3)),
                          sweep_h_um=list(np.round(np.geomspace(1.0, max(60.0, 3 * s.h_max_um), 16), 3)),
                          stars={"spec": (d, s.h_max_um)})
    run_study(cfg, PKG_ROOT, pair=not args.no_pair, sweep=not args.no_sweep)


if __name__ == "__main__":
    main()
