"""
mtf_verify.py -- STAGE 2a (MTF): window-normalized incoherent MTF of a
designed MDL as the Hankel transform of the RS focal-plane PSF (thin
driver over the ``rsval`` package in this folder).

    python 02_validation_rs_oo\\mtf_verify.py runs\\<run_folder> [other_m.npy]

Reads config.json + m_final.npy exactly as run_verify.py does (optional
keys mtf_r_max_um, mtf_r_points, mtf_f_points, mtf_f_max_lppmm -- see
``rsval.design.VerifyConfig``). Why a ray-based MTF is invalid here, how
to read the curves and the normalization convention: ``rsval.mtf``.
Outputs: rs/verify_mtf.npz, rs/fig_mtf.png.
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsval import DesignState, Log, MtfRun, __version__, parse_cli  # noqa: E402


def main(argv: Sequence[str] = tuple(sys.argv)) -> None:
    run_dir, m_file = parse_cli(argv, "mtf_verify.py")
    log = Log()
    log("rsval %s" % __version__)
    design = DesignState(run_dir, m_file)
    MtfRun(design, log).run(script_path=os.path.abspath(__file__))


if __name__ == "__main__":
    main()
