r"""Run probe / null / ladder against the mock NSC surface (no OpticStudio):
the plumbing must run end to end, the null test must PASS (scalar vs
scalar) and the ladder must read ratio 1.000 everywhere.

    python 02_NonSequential_RT_Zemax_Validation_OOP\tests\test_mock.py runs\<run>

The run folder is only READ (config.json, design_metrics.json,
zone_table.npz): the three files are copied into a temporary run folder
and every output of the mock lands there, never in runs\<run>\nsc\.
"""
from __future__ import annotations

import json
import os
import sys
import shutil
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.dirname(HERE)
sys.path.insert(0, STAGE)
sys.path.insert(0, HERE)

import mock_nsc                                       # noqa: E402
from nscval import base                               # noqa: E402

base.ZosSession = mock_nsc.MockSession                # inject the mock session

from nscval.base import RunContext                    # noqa: E402
from nscval.ladder import TeaLadder                   # noqa: E402
from nscval.nulltest import NullTest                  # noqa: E402
from nscval.probe import Probe                        # noqa: E402


def main(run_dir: str) -> bool:
    ok = True
    with tempfile.TemporaryDirectory(prefix="nsc_mock_", ignore_cleanup_errors=True) as tmp:
        os.environ["HOME"] = tmp                          # standalone outputs
        os.environ["USERPROFILE"] = tmp
        ctx0 = RunContext(None)
        Probe(ctx0).main()
        nt = NullTest(ctx0, overrides={"analysis_rays": 1000})
        nt.main()
        with open(os.path.join(nt.out_dir, "null_test.json")) as fh:
            res = json.load(fh)
        print("null test on the mock: pass=%s max diff %.2e" % (res["pass"], res["max_abs_diff"]))
        ok &= bool(res["pass"])
        # a throw-away copy of the run's inputs: mock outputs never touch runs\<run>\nsc\
        run_copy = os.path.join(tmp, os.path.basename(run_dir.rstrip("/\\")))
        os.makedirs(run_copy)
        for f in ("config.json", "design_metrics.json", "zone_table.npz"):
            src = os.path.join(run_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, run_copy)
        ctx = RunContext(run_copy)
        ld = TeaLadder(ctx, overrides={"cases": [(90.0, 1.0), (60.0, 0.5)],
                                       "lams_um": [0.4, 0.75, 1.1]})
        ld.main()
        # np.load on an .npz keeps the file open until closed: on Windows an
        # open file cannot be deleted, and the temp-folder cleanup would fail
        with np.load(os.path.join(ld.out_dir, "ladder.npz")) as z:
            r = np.asarray(z["ratio_p0"])
        print("ladder on the mock: ratio_p0 =", np.round(r, 4).tolist())
        ok &= bool(np.allclose(r[np.isfinite(r)], 1.0, atol=1e-9))
        for f in ("run_info.json", "ladder.json", "fig_ladder.png", "nsc_ladder.zos"):
            ok &= os.path.exists(os.path.join(ld.out_dir, f))
    print("ALL OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
