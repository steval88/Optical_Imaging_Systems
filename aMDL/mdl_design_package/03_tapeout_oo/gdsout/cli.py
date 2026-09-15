"""Command line of export_gds.py.

    python 03_tapeout_oo\\export_gds.py --run runs\\<run_folder> [--mode index|terrace]
    python 03_tapeout_oo\\export_gds.py --rings mdl_rings_2.txt --out mdl_s3.gds --dh-um 0.078

``--run`` builds the full tape-out package of the run (gdsout.package):
the ring table and dh come from the run's config.json, everything lands
in ``<run>/tapeout/<stamp>_<mode>/``. With ``--out`` only the .gds and
its layer map are written to that path. ``--rings`` takes any table; then
dh comes from ``--dh-um`` or is inferred from the table (the source is
echoed, and the residual |h - m dh| tells whether it fits). Nothing is
hardcoded: every setting used is printed before a file is written.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

from . import __version__
from .encode import ENCODINGS, encoding_by_name
from .export import ExportSettings, GdsExport
from .package import TapeoutPackage
from .rings import RingTable


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="export_gds.py",
        description="MDL ring table -> GDSII for grayscale lithography (gdsout %s)"
                    % __version__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run folder (table + dh from its config.json)")
    src.add_argument("--rings", help="ring table file (mdl_rings_<n>.txt)")
    ap.add_argument("--out", help="output .gds; with --run: write only the .gds "
                                  "+ layer map there instead of the package")
    ap.add_argument("--mode", choices=tuple(ENCODINGS), default="index",
                    help="layer encoding (default index)")
    ap.add_argument("--dh-um", type=float, default=None,
                    help="gray-level height quantum; default: the run's "
                         "config.json, else inferred from the table")
    ap.add_argument("--cell", default="MDL", help="top cell name (default MDL)")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="circle chord tolerance in um (default 0.02)")
    ap.add_argument("--precision", type=float, default=1e-9,
                    help="GDS database unit in metres (default 1e-9 = 1 nm)")
    ap.add_argument("--draw-zero", action="store_true",
                    help="index mode: also draw level-0 annuli")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> str:
    t0 = time.time()

    def log(msg: str) -> None:
        print("[%6.1fs] %s" % (time.time() - t0, msg), flush=True)

    args = build_parser().parse_args(argv)
    log("gdsout %s" % __version__)
    enc = encoding_by_name(args.mode, args.draw_zero)
    if args.run and args.out is None:
        pkg = TapeoutPackage(args.run, enc, dh_um=args.dh_um, cell=args.cell,
                             tol_um=args.tol, precision_m=args.precision, log=log)
        return pkg.build()
    if args.run:
        table = RingTable.from_run(args.run.rstrip("/\\"), args.dh_um)
        log("run: %s  table: %s" % (args.run, os.path.basename(table.path)))
    else:
        if args.out is None:
            raise SystemExit("--out is required with --rings")
        table = RingTable(args.rings, args.dh_um,
                          "argument" if args.dh_um is not None else "inferred")
    settings = ExportSettings(out=args.out, cell=args.cell, tol_um=args.tol,
                              precision_m=args.precision)
    ex = GdsExport(table, enc, settings, log)
    ex.describe()
    if table.level_residual_um() > 0.25 * table.dh_um:
        log("WARNING: heights are not multiples of dh = %.4f um (residual %.4f "
            "um) -- check --dh-um" % (table.dh_um, table.level_residual_um()))
    return ex.write()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
