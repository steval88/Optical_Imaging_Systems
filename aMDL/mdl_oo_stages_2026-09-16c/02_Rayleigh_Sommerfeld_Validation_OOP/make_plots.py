"""
make_plots.py -- STAGE 2a (figures): the PSF / focal-shift / r-z / metric
figures of a verified run folder (thin driver over rsval.PlotRun).

    python 02_Rayleigh_Sommerfeld_Validation_OOP\\make_plots.py runs\\<run_folder> [other_m.npy]

Reads rs\\verify_metrics.json, rs\\verify_onaxis.npz, rs\\verify_rzmap.npz
and, when present, rs\\verify_mtf.npz (older run folders with the files
at the top level are read as well); writes fig_onaxis.png, fig_psf.png,
fig_psf_2d.png, fig_rz_tiles.png, fig_metrics.png into rs\\. Run after
run_verify.py (and mtf_verify.py for the MTF panel).
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsval import DesignState, Log, __version__, parse_cli  # noqa: E402
from rsval.plots import PlotRun                              # noqa: E402


def main(argv: Sequence[str] = tuple(sys.argv)) -> None:
    """``make_plots.py runs/<run> [other_m.npy] [--cmap NAME]``: NAME is any
    matplotlib colormap for the intensity images (default OrRd, the
    paper's white-to-red tiles; inferno / magma for a dark background)."""
    args = list(argv)
    cmap = PlotRun.DEFAULT_CMAP
    if "--cmap" in args:
        i = args.index("--cmap")
        if i + 1 >= len(args):
            raise SystemExit("--cmap needs a matplotlib colormap name")
        cmap = args[i + 1]
        del args[i:i + 2]
    run_dir, m_file = parse_cli(args, "make_plots.py")
    log = Log()
    log("rsval %s" % __version__)
    PlotRun(DesignState(run_dir, m_file), log, cmap=cmap).run(script_path=os.path.abspath(__file__))


if __name__ == "__main__":
    main()
