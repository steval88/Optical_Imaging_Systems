"""Regression of 02_Rayleigh_Sommerfeld_Validation_OOP against the legacy 02_validation_rs
scripts: run both on copies of the same run folder(s) and require every
output to be identical -- rs/verify_onaxis.npz, rs/verify_rzmap.npz,
rs/verify_metrics.json (legacy keys), rs/verify_mtf.npz and the
re-exported ring table, bit for bit.

    python 02_Rayleigh_Sommerfeld_Validation_OOP\\tests\\test_against_legacy.py runs\\<run> [runs\\<run2> ...]

Each run folder is copied twice into a temporary directory (without its
rs/ and scripts/), the legacy and the new drivers are run on their own
copy, and the files are compared. Nothing in the given run folders is
modified. Runtime = 2 x (run_verify + mtf_verify) per folder.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)                                  # this stage folder
PKG_ROOT = os.path.dirname(NEW)


def legacy_dir(name: str) -> str:
    """The pre-refactor stage folder: a sibling of this stage, or inside a
    sibling package folder (the frozen mdl_design_package next to the OOP
    package), or given by MDL_LEGACY_ROOT."""
    cands = [os.path.join(PKG_ROOT, name),
             os.path.join(os.path.dirname(PKG_ROOT), "mdl_design_package", name)]
    env = os.environ.get("MDL_LEGACY_ROOT")
    if env:
        cands.insert(0, os.path.join(env, name))
    for c in cands:
        if os.path.isdir(c):
            return c
    raise SystemExit("legacy folder %r not found (tried %s); set MDL_LEGACY_ROOT to the "
                     "frozen package root" % (name, cands))


LEGACY = legacy_dir("02_validation_rs")
LEGACY_JSON_KEYS = ["run_dir", "m_file", "lam_um", "F_um", "rs_ring_quadrature",
                    "fom_ring_quadrature", "z_peak_um", "z_peak_tile_window_um",
                    "I_tilewin_over_global", "fwhm_um", "eff_3fwhm", "strehl_like",
                    "strehl_shape", "onax_I_at_F", "J_continuous", "J_verify_comb",
                    "J_objective"]


def run_drivers(stage_dir: str, run_copy: str) -> None:
    for script in ("run_verify.py", "mtf_verify.py"):
        r = subprocess.run([sys.executable, os.path.join(stage_dir, script), run_copy],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("%s/%s failed on %s:\n%s" % (stage_dir, script, run_copy,
                                                          r.stdout + r.stderr))


def npz_equal(a: str, b: str) -> List[str]:
    """Names of arrays that differ (empty list = identical)."""
    A, B = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    bad = [k for k in A.files if k not in B.files]
    for k in A.files:
        if k in B.files and not np.array_equal(A[k], B[k]):
            bad.append(k)
    return bad


def compare(old: str, new: str) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    ok = True
    for name in ("verify_onaxis.npz", "verify_rzmap.npz", "verify_mtf.npz"):
        bad = npz_equal(os.path.join(old, "rs", name), os.path.join(new, "rs", name))
        notes.append("  %-20s %s" % (name, "identical" if not bad else "DIFFERS: " + ", ".join(bad)))
        ok &= not bad
    jo = json.load(open(os.path.join(old, "rs", "verify_metrics.json")))
    jn = json.load(open(os.path.join(new, "rs", "verify_metrics.json")))
    bad_keys = []
    for k in LEGACY_JSON_KEYS:
        vo, vn = jo.get(k), jn.get(k)
        if k in ("run_dir", "m_file"):        # differ by the copy's folder name
            continue
        if vo != vn:
            bad_keys.append(k)
    notes.append("  %-20s %s" % ("verify_metrics.json",
                                 "identical (%d legacy keys)" % len(LEGACY_JSON_KEYS)
                                 if not bad_keys else "DIFFERS: " + ", ".join(bad_keys)))
    ok &= not bad_keys
    rings = [f for f in os.listdir(old) if f.startswith("mdl_rings_")]
    for f in rings:
        same = open(os.path.join(old, f), "rb").read() == open(os.path.join(new, f), "rb").read()
        notes.append("  %-20s %s" % (f, "identical" if same else "DIFFERS"))
        ok &= same
    notes.append("  J objective %.10f  poly MTF quality %.4f" % (
        jn["J_objective"], float(np.load(os.path.join(new, "rs", "verify_mtf.npz"))["mtf_quality_poly"])))
    return ok, notes


def main(run_dirs: List[str]) -> bool:
    if not run_dirs:
        raise SystemExit(__doc__)
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="rsval_reg_") as tmp:
        for rd in run_dirs:
            rd = rd.rstrip("/\\")
            if not os.path.isdir(rd):
                rd = os.path.join(PKG_ROOT, rd)
            name = os.path.basename(rd)
            copies: Dict[str, str] = {}
            for tag in ("old", "new"):
                dst = os.path.join(tmp, "%s_%s" % (name, tag))
                shutil.copytree(rd, dst, ignore=shutil.ignore_patterns("rs", "scripts", "zemax"))
                copies[tag] = dst
            print("%s: running legacy ..." % name, flush=True)
            run_drivers(LEGACY, copies["old"])
            print("%s: running 02_Rayleigh_Sommerfeld_Validation_OOP ..." % name, flush=True)
            run_drivers(NEW, copies["new"])
            ok, notes = compare(copies["old"], copies["new"])
            print("\n".join(notes))
            print("%s: %s\n" % (name, "OK" if ok else "FAILED"))
            all_ok &= ok
    print("ALL OK" if all_ok else "FAILED")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
