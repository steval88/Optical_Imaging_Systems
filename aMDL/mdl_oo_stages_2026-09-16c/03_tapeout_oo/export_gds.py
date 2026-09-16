"""
export_gds.py -- STAGE 3: MDL ring-height table -> GDSII for maskless
(grayscale) lithography, e.g. Heidelberg Instruments DWL systems (thin
driver over the ``gdsout`` package in this folder).

    python 03_tapeout_oo\\export_gds.py --run runs\\<run_folder> [--mode index|terrace]
    python 03_tapeout_oo\\export_gds.py --rings mdl_rings_2.txt --out mdl_s3.gds [--dh-um 0.078]

Encodings (``gdsout.encode``): index = one layer per gray level (the
Heidelberg "one layer = one dose" convention, level 0 not drawn unless
--draw-zero); terrace = nested contours, layer L = level >= L. Geometry
and files: ``gdsout.export`` (chord tolerance, fracturing at 8190
points, 1 um user unit / 1 nm database unit, ``<out>.layers.csv`` dose
map). Every setting used is echoed before writing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdsout.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
