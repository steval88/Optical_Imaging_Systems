"""
run_verify.py -- STAGE 2a: physical verification of a designed MDL by
scalar Rayleigh-Sommerfeld diffraction of the actual staircase profile
(thin driver over the ``rsval`` package in this folder).

    python 02_Rayleigh_Sommerfeld_Validation_OOP\\run_verify.py runs\\<run_folder> [other_m.npy]

All geometry, wavelengths and numerical grids are read from the run
folder's config.json (``rsval.design.VerifyConfig`` lists every key and
default); the optional second argument verifies a different design
vector against the same configuration. Physics: ``rsval.propagator``;
metrics and files: ``rsval.verify``. Outputs land in ``<run>/rs/`` and
the ring table ``mdl_rings_<n>.txt`` is re-exported into the run root.
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsval import DesignState, Log, VerifyRun, __version__, parse_cli  # noqa: E402


def main(argv: Sequence[str] = tuple(sys.argv)) -> None:
    run_dir, m_file = parse_cli(argv, "run_verify.py")
    log = Log()
    log("rsval %s" % __version__)
    design = DesignState(run_dir, m_file)
    VerifyRun(design, log).run(script_path=os.path.abspath(__file__))


if __name__ == "__main__":
    main()
