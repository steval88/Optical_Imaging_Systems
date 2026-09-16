"""Common skeleton of the non-sequential analyses: the ZOS session
(shared with the sequential stage's zval package), the run folder, the
output subfolder <run>/nsc/<YYYYMMDD_HHMMSS>_<mode>/ and run_info.json."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from . import SCRIPT_VERSION

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # this stage
PKG_ROOT = os.path.dirname(HERE)

# the sequential stage (zval: ZosSession, Log, Design) is a sibling folder
SEQ_CANDIDATES = ("02_Sequential_RT_Zemax_Validation_OOP", "02_validation_zemax_oo")


def sequential_stage_dir() -> str:
    for name in SEQ_CANDIDATES:
        cand = os.path.join(PKG_ROOT, name)
        if os.path.isdir(os.path.join(cand, "zval")):
            return cand
    raise SystemExit("the sequential stage folder (%s) with its zval package must "
                     "sit next to this stage" % " | ".join(SEQ_CANDIDATES))


SEQ_DIR = sequential_stage_dir()
if SEQ_DIR not in sys.path:
    sys.path.insert(0, SEQ_DIR)

from zval.report import Log            # noqa: E402
from zval.zos import ZosSession        # noqa: E402


class RunContext:
    """What a mode needs from a run folder (None for the stand-alone
    null test): config.json and the seed record."""

    def __init__(self, run_dir: Optional[str]) -> None:
        self.run_dir = run_dir.rstrip("/\\") if run_dir else None
        self.cfg: Dict[str, Any] = {}
        self.seed: Dict[str, Any] = {}
        if self.run_dir:
            with open(os.path.join(self.run_dir, "config.json")) as fh:
                self.cfg = json.load(fh)
            dm = os.path.join(self.run_dir, "design_metrics.json")
            if os.path.exists(dm):
                with open(dm) as fh:
                    self.seed = json.load(fh).get("seed") or {}

    @property
    def name(self) -> str:
        return os.path.basename(self.run_dir) if self.run_dir else "standalone"

    def design_lams(self) -> List[float]:
        c = self.cfg
        return [float(v) for v in (c.get("target_wavelengths_um")
                                   or c.get("verify_wavelengths_um") or [])]

    def fold_um(self) -> Optional[float]:
        v = self.seed.get("h_fold_um")
        return float(v) if v is not None else None

    def zone_table_path(self) -> Optional[str]:
        if not self.run_dir:
            return None
        p = os.path.join(self.run_dir, "zone_table.npz")
        return p if os.path.exists(p) else None


class NscAnalysis:
    """Base of the modes: session, output folder, run_info.json."""
    MODE = "?"
    DESCRIPTION = ""

    def __init__(self, ctx: RunContext, gui: bool = False, app: Any = None,
                 overrides: Optional[Dict[str, Any]] = None) -> None:
        self.ctx = ctx
        self.gui = gui
        self.app = app
        self.overrides: Dict[str, Any] = overrides or {}
        self.log = Log()
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        base = (os.path.join(ctx.run_dir, "nsc") if ctx.run_dir else
                os.path.join(os.path.expanduser("~"), "Documents", "Zemax_MDL_NSC"))
        self.out_dir = os.path.abspath(os.path.join(base, "%s_%s" % (self.stamp, self.MODE)))
        self.t0 = time.time()
        self.store: Dict[str, Any] = {}
        self.session: Any = None

    def settings_record(self) -> Dict[str, Any]:
        return {}

    def write_run_info(self, extra: Optional[Dict[str, Any]] = None) -> str:
        info = {"mode": self.MODE, "script_version": SCRIPT_VERSION, "stamp": self.stamp,
                "command": " ".join(sys.argv), "design": self.ctx.run_dir or "standalone",
                "gui": self.gui, "settings": self.settings_record(),
                "overrides": self.overrides,
                "zos_members_resolved": getattr(self, "members_found", {}),
                "elapsed_s": round(time.time() - self.t0, 1)}
        if extra:
            info.update(extra)
        path = os.path.join(self.out_dir, "run_info.json")
        with open(path, "w") as fh:
            json.dump(info, fh, indent=1)
        return path

    def connect(self) -> Any:
        self.session = ZosSession(gui=self.gui, log=self.log, app=self.app)
        return self.session

    def run(self) -> None:
        raise NotImplementedError

    def main(self) -> None:
        log = self.log
        log.section("configuration", self.DESCRIPTION)
        log("script version %s | mode %s | design %s" % (SCRIPT_VERSION, self.MODE,
                                                         self.ctx.name))
        log("output -> %s" % self.out_dir)
        for k, v in self.settings_record().items():
            log("  %s = %s" % (k, v))
        os.makedirs(self.out_dir, exist_ok=True)
        self.connect()
        try:
            self.run()
        finally:
            self.write_run_info()
            log("run_info.json written (%.1fs)" % (time.time() - self.t0))
            if self.session is not None and not self.gui:
                try:
                    self.session.close()
                except Exception:
                    pass
