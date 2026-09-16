"""Regression of 03_tapeout_oo against the legacy 03_tapeout/export_gds.py:
both write a GDS from the same ring table in both encodings; the layouts
must hold the same polygons (layer, vertex arrays) in the same order, and
the same layer-map CSV. (The .gds bytes themselves differ only by the
time stamp gdstk writes in the header.)

    python 03_tapeout_oo\\tests\\test_against_legacy.py runs\\<run> [--dh-um 0.078]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Tuple

import gdstk
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.dirname(HERE)
PKG_ROOT = os.path.dirname(STAGE)


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


LEGACY = os.path.join(legacy_dir("03_tapeout"), "export_gds.py")
NEW = os.path.join(STAGE, "export_gds.py")


def polygons(path: str) -> List[Tuple[int, np.ndarray]]:
    lib = gdstk.read_gds(path)
    out = []
    for cell in lib.cells:
        for p in cell.polygons:
            out.append((p.layer, np.asarray(p.points)))
    return out


def same_layout(a: str, b: str) -> Tuple[bool, str]:
    pa, pb = polygons(a), polygons(b)
    if len(pa) != len(pb):
        return False, "polygon count %d vs %d" % (len(pa), len(pb))
    for i, ((la, va), (lb, vb)) in enumerate(zip(pa, pb)):
        if la != lb or va.shape != vb.shape or not np.array_equal(va, vb):
            return False, "polygon %d differs (layer %d/%d)" % (i, la, lb)
    return True, "%d polygons identical" % len(pa)


def main() -> bool:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dh-um", type=float, default=None)
    a = ap.parse_args()
    run = a.run.rstrip("/\\")
    if not os.path.isdir(run):
        run = os.path.join(PKG_ROOT, run)
    cfg = json.load(open(os.path.join(run, "config.json")))
    rings = os.path.join(run, "mdl_rings_%d.txt" % cfg["dll_file_no"])
    dh = a.dh_um if a.dh_um is not None else float(cfg["dh_um"])
    ok = True
    with tempfile.TemporaryDirectory(prefix="gds_reg_") as tmp:
        for mode in ("index", "terrace"):
            for zero in ((), ("--draw-zero",)) if mode == "index" else ((),):
                tag = mode + ("_zero" if zero else "")
                old = os.path.join(tmp, "old_%s.gds" % tag)
                new = os.path.join(tmp, "new_%s.gds" % tag)
                for script, out in ((LEGACY, old), (NEW, new)):
                    r = subprocess.run([sys.executable, script, "--rings", rings, "--out", out,
                                        "--mode", mode, "--dh-um", str(dh), *zero],
                                       capture_output=True, text=True)
                    if r.returncode != 0:
                        raise SystemExit("%s failed:\n%s" % (script, r.stdout + r.stderr))
                same, note = same_layout(old, new)
                csv_same = open(old + ".layers.csv").read() == open(new + ".layers.csv").read()
                print("  %-12s layout: %s | layer map %s" % (tag, note,
                      "identical" if csv_same else "DIFFERS"))
                ok &= same and csv_same
    print("ALL OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
