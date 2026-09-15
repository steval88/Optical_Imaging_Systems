"""The DESIGN under validation: geometry, ring table and test lines,
derived from a run folder written by 01_design/run_MDL_design.py (or,
for the pre-run-folder designs, from the legacy registry)."""
import hashlib
import json
import os
import shutil

import numpy as np

from .settings import DESIGNS, n_resist


def dll_surfaces_dir():
    """{Documents}\\Zemax\\DLL\\Surfaces -- where the DLLs load their
    ring table from (their OWN folder, not the run folder)."""
    return os.path.join(os.path.expanduser("~"), "Documents", "Zemax",
                        "DLL", "Surfaces")


class RingTable:
    """mdl_rings_<n>.txt: N rings of width delta, heights h."""

    def __init__(self, path):
        self.path = path
        with open(path) as fh:
            n, delta_mm = fh.readline().split()
            self.N, self.delta_um = int(n), float(delta_mm) * 1000.0
            self.h_um = np.array([float(fh.readline())
                                  for _ in range(self.N)]) * 1000.0
        self.rho_um = (np.arange(self.N) + 0.5) * self.delta_um
        self.R_um = self.N * self.delta_um

    def phase_waves(self, lam_um):
        """(n-1) h / lam per ring -- the design phase the DLL must
        reproduce."""
        return (n_resist(lam_um) - 1.0) * self.h_um / lam_um


class Design:
    """Everything the validation needs to know about the lens."""

    def __init__(self, epd_mm, bfd_mm, file_no, wavelengths_um,
                 primary_idx, fold_P, lam0_um, orders, run_dir=None,
                 name=None, fold_note=None, lams_all_um=None,
                 rz_span_mm=1.0, rz_r_max_um=20.0, rz_r_points=41,
                 rz_z_points=121):
        self.epd_mm = float(epd_mm)
        self.bfd_mm = float(bfd_mm)
        self.file_no = int(file_no)
        self.wavelengths_um = [float(w) for w in wavelengths_um]
        self.primary_idx = int(primary_idx)          # 1-based
        self.fold_P = fold_P
        self.lam0_um = lam0_um
        self.orders = list(orders)
        self.run_dir = run_dir
        self.name = name or (os.path.basename(os.path.normpath(run_dir))
                             if run_dir else "?")
        self.fold_note = fold_note
        self.lams_all_um = ([float(v) for v in lams_all_um]
                            if lams_all_um else list(self.wavelengths_um))
        self.rz_span_mm = float(rz_span_mm)
        self.rz_r_max_um = float(rz_r_max_um)
        self.rz_r_points = int(rz_r_points)
        self.rz_z_points = int(rz_z_points)

    # --- derived ---------------------------------------------------------
    @property
    def F_um(self):
        return self.bfd_mm * 1000.0

    @property
    def na_par(self):
        """paraxial NA = EPD / 2F"""
        return self.epd_mm / (2.0 * self.bfd_mm)

    @property
    def primary_um(self):
        return self.wavelengths_um[self.primary_idx - 1]

    @property
    def table_name(self):
        return "mdl_rings_%d.txt" % self.file_no

    def rs_path(self, name):
        """rs/<name> in the run folder (or <name> at the root, legacy)."""
        if not self.run_dir:
            return None
        for cand in (os.path.join(self.run_dir, "rs", name),
                     os.path.join(self.run_dir, name)):
            if os.path.exists(cand):
                return cand
        return None

    def ring_table(self):
        return RingTable(os.path.join(self.run_dir, self.table_name))

    # --- line selection ('verify' | 'all' | 'primary' | comma list) ------
    def select_lines(self, sel):
        sel = str(sel).strip().lower()
        if sel == "all":
            lams = list(self.lams_all_um)
        elif sel == "primary":
            lams = [self.primary_um]
        elif sel == "verify":
            lams = list(self.wavelengths_um)
        else:
            lams = [float(v) for v in sel.split(",")]
        self.wavelengths_um = lams
        self.primary_idx = (len(lams) + 1) // 2
        return lams

    # --- constructors ----------------------------------------------------
    @classmethod
    def from_arg(cls, arg):
        """A run folder (preferred) or a legacy registry name."""
        if os.path.isdir(arg):
            return cls.from_run_folder(arg)
        if arg in DESIGNS:
            return cls(name=arg, **DESIGNS[arg])
        raise SystemExit("unknown design %r: pass a run folder or one of "
                         "%s" % (arg, sorted(DESIGNS)))

    @classmethod
    def from_run_folder(cls, run_dir):
        """Geometry / file number / wavelengths from config.json; the OD
        fold parameters from the seed record in design_metrics.json
        (echelle seed: (P, lam0) = the target line with the smallest
        blaze detune of h_fold; harmonic seed: stored directly)."""
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise SystemExit("no config.json in %r -- pass a run folder "
                             "made by run_MDL_design, or one of the "
                             "legacy design names %s"
                             % (run_dir, sorted(DESIGNS)))
        cfg = json.load(open(cfg_path))
        der = cfg["derived"]
        lams_all = list(cfg.get("target_wavelengths_um")
                        or cfg["verify_wavelengths_um"])
        # 5 representative wavelengths across the set (PSFs are expensive)
        n = len(lams_all)
        idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        wavelengths = [float(lams_all[i]) for i in idxs]
        primary_idx = (len(wavelengths) + 1) // 2

        dm_path = os.path.join(run_dir, "design_metrics.json")
        seed = {}
        if os.path.exists(dm_path):
            seed = json.load(open(dm_path)).get("seed") or {}
        if seed.get("mode") == "echelle" or "h_fold_um" in seed:
            h_fold = float(seed["h_fold_um"])
            best = None
            for lam in lams_all:
                alpha = (n_resist(lam) - 1.0) * h_fold / lam
                det = abs(alpha - round(alpha))
                if best is None or det < best[0]:
                    best = (det, float(lam), int(round(alpha)))
            det0, lam0, P = best
            ladder = sorted({int(o) for o in seed.get("orders", [])})
            orders = sorted({ladder[0], P, ladder[-1]}) + [0] \
                if ladder else [P, 0]
            fold_note = ("echelle seed: h_fold=%.4f um -> P=%d @ %.2f um "
                         "(detune %.3f)" % (h_fold, P, lam0, det0))
        elif "lam0_um" in seed:
            lam0, P = float(seed["lam0_um"]), int(seed["p"])
            orders = sorted({max(1, P - 4), P, P + 4}) + [0]
            fold_note = "harmonic seed: P=%d @ %.2f um" % (P, lam0)
        else:
            lam0 = P = None
            orders = []
            fold_note = ("design_metrics.json has no seed record: od "
                         "mode needs the fold (P, lam0)")
        return cls(epd_mm=cfg["diameter_um"] / 1000.0,
                   bfd_mm=der["focal_um"] / 1000.0,
                   file_no=int(cfg["dll_file_no"]),
                   wavelengths_um=wavelengths, primary_idx=primary_idx,
                   fold_P=P, lam0_um=lam0, orders=orders,
                   run_dir=run_dir, fold_note=fold_note,
                   lams_all_um=cfg["verify_wavelengths_um"],
                   rz_span_mm=float(cfg.get("rzmap_z_span_um",
                                            1000.0)) / 1000.0,
                   rz_r_max_um=float(cfg.get("rzmap_r_max_um", 20.0)),
                   rz_r_points=int(cfg.get("rzmap_r_points", 41)),
                   rz_z_points=int(cfg.get("rzmap_z_points", 121)))

    # --- provenance --------------------------------------------------------
    def sync_ring_table(self, log):
        """The DLLs load mdl_rings_<n>.txt from THEIR OWN folder and file
        numbers get reused across iterations, so a stale copy silently
        validates the WRONG design (caught 2026-08-31). Compare bytes,
        copy on mismatch, warn if the Surfaces folder is missing."""
        if not self.run_dir:
            return
        src = os.path.join(self.run_dir, self.table_name)
        dst_dir = dll_surfaces_dir()
        dst = os.path.join(dst_dir, self.table_name)
        if not os.path.exists(src):
            raise SystemExit("run folder has no %s -- re-run run_verify.py "
                             "to re-export it" % src)
        if not os.path.isdir(dst_dir):
            log("  WARNING: %s not found -- copy %s there by hand before "
                "trusting any Zemax result" % (dst_dir, self.table_name))
            return
        src_bytes = open(src, "rb").read()
        dst_bytes = open(dst, "rb").read() if os.path.exists(dst) else None
        h_src = hashlib.sha256(src_bytes).hexdigest()
        if dst_bytes == src_bytes:
            log("  ring table in DLL folder matches the run folder "
                "(byte-identical)  SHA256 %s" % h_src[:16])
            return
        h_dst = (hashlib.sha256(dst_bytes).hexdigest()
                 if dst_bytes is not None else "(missing)")
        shutil.copy2(src, dst)
        log("  ring table %s: DLL-folder copy %s -- SYNCED from the run "
            "folder  (run %s  ->  DLL copy was %s)"
            % (self.table_name,
               "was STALE (different design!)" if dst_bytes is not None
               else "was missing", h_src[:16], h_dst[:16]))
