"""Common skeleton of the stage-2a drivers: the timing log, how a run
folder argument is resolved, where the outputs go (``<run>/rs/``, the
per-solver layout of 2026-08-28) and the provenance snapshot into
``<run>/scripts/``."""
from __future__ import annotations

import difflib
import json
import os
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "2026-09-16.02"

HERE = os.path.dirname(os.path.abspath(__file__))          # .../rsval
STAGE_DIR = os.path.dirname(HERE)                           # 02_Rayleigh_Sommerfeld_Validation_OOP
PKG_ROOT = os.path.dirname(STAGE_DIR)                       # mdl_design_package

LogFn = Callable[[str], None]


class Log:
    """``log("...")`` prints ``[  12.3s] ...`` with the seconds elapsed since
    the object was created (the format of every stage of the toolchain)."""

    def __init__(self, stream: Any = None) -> None:
        self.t0: float = time.time()
        self.stream = stream or sys.stdout

    def __call__(self, msg: str) -> None:
        print("[%6.1fs] %s" % (time.time() - self.t0, msg), flush=True,
              file=self.stream)


def resolve_run_dir(arg: str, pkg_root: str = PKG_ROOT) -> str:
    """The run folder named on the command line, as given or relative to
    the package root (so the drivers work from any CWD). A folder without
    ``config.json`` is refused with the closest existing run folders as a
    typo hint -- the legacy behaviour, kept verbatim."""
    run_dir = arg.rstrip("/\\")
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        alt = os.path.join(pkg_root, run_dir)
        if os.path.exists(os.path.join(alt, "config.json")):
            run_dir = alt
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        hint = ""
        runs_root = os.path.join(pkg_root, "runs")
        if os.path.isdir(runs_root):
            cands = sorted(os.listdir(runs_root))
            close = difflib.get_close_matches(os.path.basename(run_dir),
                                              cands, n=3, cutoff=0.5)
            if close:
                hint = "\n  did you mean: " + "  ".join(
                    os.path.join("runs", c) for c in close)
            elif cands:
                hint = "\n  most recent run folders: " + "  ".join(
                    os.path.join("runs", c) for c in cands[-3:])
        raise SystemExit("no config.json in %r -- is this a run folder made "
                         "by run_MDL_design.py?%s" % (run_dir, hint))
    return run_dir


def parse_cli(argv: Sequence[str], prog: str) -> Tuple[str, Optional[str]]:
    """``prog runs/<run_folder> [other_m.npy]`` -> (run_dir, m_file|None)."""
    if len(argv) < 2:
        raise SystemExit("usage: python %s runs/<run_folder> "
                         "[optional/other_m.npy]" % prog)
    run_dir = resolve_run_dir(argv[1])
    m_file = argv[2] if len(argv) > 2 else None
    return run_dir, m_file


class Stage:
    """Base of VerifyRun and MtfRun: owns the log, the ``rs/`` output
    folder of the run and the provenance snapshot.

    Attributes
    ----------
    run_dir    the run folder (as resolved from the command line)
    out_rs     ``<run>/rs`` -- every file this solver produces lands here,
               mirroring ``zemax/`` of the Zemax stage
    log        the timing log (shared with the design state when given)
    """

    def __init__(self, run_dir: str, log: Optional[Log] = None) -> None:
        self.run_dir: str = run_dir
        self.out_rs: str = os.path.join(run_dir, "rs")
        os.makedirs(self.out_rs, exist_ok=True)
        self.log: Log = log or Log()

    # -- provenance ---------------------------------------------------------
    def snapshot(self, script_path: Optional[str], package_dir: str = HERE) -> None:
        """Copy the driver script and the whole ``rsval`` package into
        ``<run>/scripts/`` (``__pycache__`` excluded), so the run folder
        records the code that produced its ``rs/`` files."""
        dst = os.path.join(self.run_dir, "scripts")
        os.makedirs(dst, exist_ok=True)
        if script_path:
            shutil.copy2(script_path, os.path.join(dst, os.path.basename(script_path)))
        shutil.copytree(package_dir, os.path.join(dst, os.path.basename(package_dir)),
                        ignore=shutil.ignore_patterns("__pycache__"),
                        dirs_exist_ok=True)

    # -- outputs --------------------------------------------------------------
    def rs_path(self, name: str) -> str:
        return os.path.join(self.out_rs, name)

    def write_npz(self, name: str, arrays: Dict[str, Any]) -> str:
        path = self.rs_path(name)
        np.savez(path, **arrays)
        return path

    def write_json(self, name: str, payload: Dict[str, Any]) -> str:
        path = self.rs_path(name)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)
        return path

    def rel(self, path: str) -> str:
        return os.path.relpath(path)

    def saved_lines(self, entries: List[Tuple[str, str]]) -> None:
        """Log the 'saved into <run>:' block (file, what it holds)."""
        self.log("saved into %s:" % os.path.relpath(self.run_dir))
        for name, what in entries:
            self.log("  %-24s %s" % (name, what))
