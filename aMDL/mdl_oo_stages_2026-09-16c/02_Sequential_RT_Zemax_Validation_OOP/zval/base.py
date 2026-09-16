"""Common skeleton of the four analyses: connect, build the system for
the mode's surface variant, save the .zos BEFORE any analysis (a failure
never costs the file), run, re-save."""
import json
import os
import sys
import time

from . import SCRIPT_VERSION
from .design import Design
from .report import Log
from .zos import AnalysisIO, ZosSession, ZosSystem


class Analysis:
    MODE = "?"
    VARIANT = "zone"
    ZOS_NAME = "mdl_validation.zos"
    DESCRIPTION = ""

    def __init__(self, design, gui=False, app=None, script_path=None):
        self.d = design
        self.gui = gui
        self.app = app                     # injected session (tests)
        self.script_path = script_path or "mdl_zemax_validation.py"
        self.log = Log()
        # one time-stamped SUBFOLDER per Zemax run (2026-09-15):
        # <run>\zemax\<YYYYMMDD_HHMMSS>_<mode>\ -- runs never overwrite
        # each other and the folder name says what was run; run_info.json
        # inside records version, command line and the settings used.
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        base = (os.path.join(design.run_dir, "zemax") if design.run_dir else
                os.path.join(os.path.expanduser("~"), "Documents",
                             "Zemax_MDL_Validation"))
        self.out_dir = os.path.abspath(
            os.path.join(base, "%s_%s" % (self.stamp, self.MODE)))
        self.t0 = time.time()
        self.store = {}

    # --- hooks -------------------------------------------------------------
    def select_lines(self):
        """Replace the design's line list for this mode (default: keep)."""

    def describe(self):
        """Mode-specific lines of the configuration section."""

    def hybrid_settings(self):
        return None

    def settings_record(self):
        """The instrument settings this run used (for run_info.json)."""
        return {}

    def write_run_info(self):
        info = {"mode": self.MODE, "script_version": SCRIPT_VERSION,
                "stamp": self.stamp, "command": " ".join(sys.argv),
                "design": self.d.run_dir or self.d.name,
                "wavelengths_um": list(self.d.wavelengths_um),
                "gui": self.gui, "settings": self.settings_record(),
                "elapsed_s": round(time.time() - self.t0, 1)}
        with open(os.path.join(self.out_dir, "run_info.json"), "w") as fh:
            json.dump(info, fh, indent=1)

    def run(self):
        raise NotImplementedError

    # --- driver --------------------------------------------------------------
    def dll_line(self):
        return {"zone": "us_mdl_rings.dll", "od": "us_mdl_rings_od.dll",
                "hybrid": "us_mdl_rings.dll (residual) behind a Paraxial "
                          "lens"}[self.VARIANT]

    def main(self):
        d, log = self.d, self.log
        if d.run_dir:
            log.section("ring table provenance",
                        "the table next to the DLLs must be byte-identical "
                        "to the run folder's (SHA256)")
            d.sync_ring_table(log)
        self.select_lines()
        log.section("design and run configuration",
                    "what is being validated, from which run folder, at "
                    "which lines")
        log("script version %s | %s" % (SCRIPT_VERSION,
                                         os.path.abspath(self.script_path)))
        log("design: %s" % (d.run_dir or d.name))
        log("  EPD %.2f mm | BFD %.2f mm | ring table %s"
            % (d.epd_mm, d.bfd_mm, d.table_name))
        log("  wavelengths (um): %s  (primary #%d = %.2f)"
            % (", ".join("%.2f" % w for w in d.wavelengths_um),
               d.primary_idx, d.primary_um))
        self.describe()
        log("  DLL: %s | output -> %s" % (self.dll_line(), self.out_dir))
        log("  REMINDER: %s must sit next to the DLLs in "
            "{Documents}\\Zemax\\DLL\\Surfaces\\" % d.table_name)

        self.session = ZosSession(gui=self.gui, log=log, app=self.app)
        os.makedirs(self.out_dir, exist_ok=True)
        log.section("build the OpticStudio system",
                    "surfaces, DLL parameters, wavelengths; saved as a .zos "
                    "before any analysis")
        self.zs = ZosSystem(self.session, d, self.VARIANT, self.out_dir,
                            log).build(hybrid=self.hybrid_settings())
        self.io = AnalysisIO(self.session, self.out_dir, log)
        self.zs.ray_checks()
        self.zs.save(self.ZOS_NAME)
        log("saved system -> %s (will be re-saved after the analyses)"
            % self.ZOS_NAME)
        try:
            self.run()
        finally:
            self.zs.save(self.ZOS_NAME)
            log("re-saved system -> %s%s" % (
                self.ZOS_NAME,
                "  (analysis windows left open in the GUI and saved with "
                "the file)" if self.gui else
                "  (run with --gui to keep the analysis windows open for "
                "colleagues)"))
            self.write_run_info()
            log("run folder: %s  (run_info.json: version, command line, "
                "settings)" % self.out_dir)
            self.session.close()
