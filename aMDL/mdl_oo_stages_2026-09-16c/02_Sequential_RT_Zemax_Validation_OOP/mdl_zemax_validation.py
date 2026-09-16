"""
mdl_zemax_validation.py -- STAGE 2b: Zemax OpticStudio validation (CLI)
========================================================================

Thin entry point over the zval package (see zval/__init__.py for the
module map and the mode descriptions). Run on the OpticStudio machine:

    python mdl_zemax_validation.py <run-folder> zone [--gui]
    python mdl_zemax_validation.py <run-folder> od   [--orders 10,15,0]
    python mdl_zemax_validation.py <run-folder> rz   [--lams all|0.4,0.55]
    python mdl_zemax_validation.py <run-folder> huy  [--lams verify|all|
                                   primary|0.4,0.55] [--pupil 512]
                                   [--image 128] [--delta 0.4]
                                   [--polarization]

<run-folder> is a folder produced by 01_design/run_MDL_design.py:
every parameter (aperture, focal, ring-table file number, test lines,
OD fold, rz windows) is derived from its config.json /
design_metrics.json and all output goes to <run-folder>\\zemax\\. The
legacy names {na03, s3, s3comb} select the hand-kept registry entries
in zval/settings.py.

--gui drives the OPEN OpticStudio through the Interactive Extension
(Programming tab -> Interactive Extension must be waiting): every
analysis opens as a native window and STAYS OPEN after the script
exits -- the route for assessing results with Zemax's own plots.

Defaults live in zval/settings.py (HUY_SETTINGS, HYBRID_SETTINGS,
RZ_SETTINGS, ZONE_SETTINGS) and are echoed by every run; the options
above override them for one run only.

Prerequisites: us_mdl_rings.dll, us_mdl_rings_od.dll and the design's
mdl_rings_<File#>.txt in {Documents}\\Zemax\\DLL\\Surfaces\\ (the table
is synced from the run folder automatically); pip install pythonnet;
zos_connection.py next to this file or in ..\\02_validation_zemax.

This folder (02_Sequential_RT_Zemax_Validation_OOP) sits next to the untouched
pre-refactor folder 02_validation_zemax, whose mdl_zemax_validation.py
(2026-09-08.04: POP ladder, rzfft, hypothesis columns, gain sweep,
positional syntax e.g. '<run> pop 4 verify 4096') stays the reference
for everything measured up to 2026-09-08. zos_connection.py is found
there automatically (echoed at connect time); zval/settings.py holds
the defaults.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zval import SCRIPT_VERSION                       # noqa: E402
from zval.design import Design                        # noqa: E402
from zval.settings import HUY_SETTINGS, RZ_SETTINGS   # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        prog="mdl_zemax_validation.py",
        description="OpticStudio validation of an MDL run folder "
                    "(zval %s)" % SCRIPT_VERSION)
    p.add_argument("design", help="run folder (preferred) or legacy name")
    sub = p.add_subparsers(dest="mode", required=True, metavar="mode")

    def common(sp):
        sp.add_argument("--gui", action="store_true",
                        help="drive the open OpticStudio (Interactive "
                             "Extension); analysis windows stay open")

    common(sub.add_parser("zone", help="FFT PSF per line on the staircase "
                                       "DLL (coarse localization)"))
    sp = sub.add_parser("od", help="ray-based chromatic analyses on the "
                                   "order-decomposition DLL")
    sp.add_argument("--orders", help="comma list of orders (0 = auto); "
                                     "default from the seed record")
    common(sp)
    sp = sub.add_parser("rz", help="batch-OPD trace -> I(r,z) tiles vs rs/")
    sp.add_argument("--lams", default=RZ_SETTINGS["lams"],
                    help="'all' (default), 'verify', 'primary' or a comma "
                         "list in um")
    common(sp)
    sp = sub.add_parser("huy", help="Huygens PSF/MTF on the hybrid system "
                                    "(the focal-field closure)")
    sp.add_argument("--lams", default=HUY_SETTINGS["lams"],
                    help="'verify' (default: the 5 representative lines), "
                         "'all', 'primary' or a comma list in um")
    sp.add_argument("--pupil", type=int, default=HUY_SETTINGS["pupil_samp"],
                    help="pupil sampling per side (ray pitch EPD/N = Avg "
                         "cell); default %d" % HUY_SETTINGS["pupil_samp"])
    sp.add_argument("--image", type=int, default=HUY_SETTINGS["image_samp"],
                    help="image grid per side; default %d"
                         % HUY_SETTINGS["image_samp"])
    sp.add_argument("--delta", type=float,
                    default=HUY_SETTINGS["image_delta_um"],
                    help="image pitch [um]; default %.2f"
                         % HUY_SETTINGS["image_delta_um"])
    sp.add_argument("--polarization", action="store_true",
                    help="UsePolarization on (measured 2026-09-07: all-zero "
                         "grids on 2024 R1 -- off by default)")
    common(sp)
    return p


def make_analysis(args, design, app=None):
    kw = dict(gui=args.gui, app=app, script_path=__file__)
    if args.mode == "zone":
        from zval.zone import ZoneAnalysis
        return ZoneAnalysis(design, **kw)
    if args.mode == "od":
        from zval.od import OdAnalysis
        orders = [int(v) for v in args.orders.split(",")] if args.orders else None
        return OdAnalysis(design, orders=orders, **kw)
    if args.mode == "rz":
        from zval.rz import RzAnalysis
        return RzAnalysis(design, lams=args.lams, **kw)
    from zval.huygens import HuygensAnalysis
    return HuygensAnalysis(design, lams=args.lams, pupil_samp=args.pupil,
                           image_samp=args.image, image_delta_um=args.delta,
                           use_polarization=True if args.polarization else None,
                           **kw)


def main(argv=None, app=None):
    args = build_parser().parse_args(argv)
    design = Design.from_arg(args.design)
    make_analysis(args, design, app=app).main()


if __name__ == "__main__":
    main()
