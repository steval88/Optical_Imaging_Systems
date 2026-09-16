"""
run_MDL_design.py -- STAGE 1: MDL design (Search -> Smooth -> Gradient -> Polish)
=================================================================================

Typed, object-oriented driver over the ``mdl`` package in this folder
(01_design_oo). Edit the ``SETTINGS = ...`` line in the CONFIG section
and, FROM THE PACKAGE ROOT, run

    python 01_design_oo\\run_MDL_design.py

Every run creates its own time-stamped folder under the package root:

    runs/<YYYYMMDD_HHMMSS>_<name>/
        config.json          the exact settings used (+ derived values)
        scripts/             snapshot of the code as it was at run time
        m_final.npy          optimized gray-level vector m (N int32)
        mdl_rings_<n>.txt    ring table for the OpticStudio DLLs / export_gds.py
        design_metrics.json  J at every pipeline stage + the seed record
        zone_table.npz       local-grating zones for a rigorous (RCWA) sweep

The keys and values of config.json are the ones the downstream stages
read (02_Rayleigh_Sommerfeld_Validation_OOP/run_verify.py, mtf_verify.py,
02_Sequential_RT_Zemax_Validation_OOP, 01_design_oo/fom_quadrature_check.py), so a run
folder made here is consumed exactly like one made by the pre-refactor
01_design/run_MDL_design.py.

Pipeline (paper Fig. 2a, supplementary S2-2 .. S2-4)
----------------------------------------------------
    seed      analytic start: harmonic lens at one line, or the LADDER
              seed (fold height chosen over the whole comb), or a warm
              start from an existing m.npy
    Search    s blocks of [GA (p epochs) -> Hooke-Jeeves]   (S2-2, Fig. S3)
    Smooth    aspect-ratio reduction                        (S2-4)
    Gradient  ascent on continuous heights, then rounding   (S2-2)
    Polish    integer Hooke-Jeeves from the rounded vector
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)              # the mdl_design_package root
sys.path.insert(0, HERE)                      # the mdl package lives here

from mdl import (PAPER_COMB_14, GradientRefine, HookeJeeves, IntVec,   # noqa: E402
                 LadderSeed, MDLProblem, MultistepGAHJACombo,
                 SearchGAHJA, Smooth, __version__, extract_local_gratings,
                 harmonic_seed, load_efficiency_table, upper_bound_jf,
                 write_zone_table)

SeedMode = Literal["harmonic", "ladder"]
FomMode = Literal["mean", "geomean", "softmin"]
RingQuadrature = Literal["sinc", "midpoint"]

# The code of every stage, snapshotted into <run>/scripts/ for provenance
# (paths relative to the package root; whole folders are copied without
# __pycache__; a missing entry is reported and skipped). This is the ONE
# place the stage folder names appear in this driver; the downstream
# stages themselves find the run folder by its path, not by this list.
STAGE_SNAPSHOT: Tuple[str, ...] = (
    "01_design_oo/run_MDL_design.py",
    "01_design_oo/mdl",
    "01_design_oo/tradeoff",
    "01_design_oo/tradeoff_maps.py",
    "01_design_oo/tradeoff_gui.py",
    "02_Rayleigh_Sommerfeld_Validation_OOP/run_verify.py",
    "02_Rayleigh_Sommerfeld_Validation_OOP/mtf_verify.py",
    "02_Rayleigh_Sommerfeld_Validation_OOP/make_plots.py",
    "02_Rayleigh_Sommerfeld_Validation_OOP/rsval",
    "02_Sequential_RT_Zemax_Validation_OOP/mdl_zemax_validation.py",
    "02_Sequential_RT_Zemax_Validation_OOP/zval",
    "02_NonSequential_RT_Zemax_Validation_OOP/mdl_nsc_validation.py",
    "02_NonSequential_RT_Zemax_Validation_OOP/nscval",
    "03_tapeout_oo/export_gds.py",
    "03_tapeout_oo/gdsout",
)
NEXT_STAGE_CMD = os.path.join("02_Rayleigh_Sommerfeld_Validation_OOP", "run_verify.py")


# =========================================================================
# CONFIG
# =========================================================================
@dataclass(frozen=True)
class DesignConfig:
    """Every setting of one design run. Lengths in micrometres unless the
    field name says otherwise; ``None`` means "not used" where allowed.
    Variants are made with ``dataclasses.replace(preset, name=..., ...)``.
    The field groups follow the pipeline; each field is documented by the
    comment beside it (``DesignConfig.describe()`` prints them all).
    """
    name: str                                      # run label (folder suffix)

    # -- geometry ---------------------------------------------------------
    diameter_um: float = 10240.0                   # lens diameter D
    focal_um: Optional[float] = 50940.0            # focal length F -- give
    na: Optional[float] = None                     #   focal_um XOR na

    # -- band / objective -------------------------------------------------
    lam_min_um: float = 0.40                       # working band
    lam_max_um: float = 1.10
    target_wavelengths_um: Optional[List[float]] = None  # None = continuous
                                                   #   band; a list = comb
                                                   #   objective on those lines
    n_wavelengths: int = 1001                      # continuous sampling
                                                   #   (alias-checked)
    fom_mode: FomMode = "geomean"                  # "mean" (paper Eq. 2),
                                                   #   "geomean", "softmin"
    softmin_beta: float = 20.0                     # softmin: initial beta
    softmin_beta_final: Optional[float] = 300.0    # softmin: annealed to this
                                                   #   in the Gradient stage
    overlap_fom: bool = False                      # encircled-energy objective
    overlap_r_enc_um: Optional[float] = None       # None = chromatic disc
    overlap_airy_factor: float = 2.0               # disc radius in Airy radii
                                                   #   (1.0 = main lobe only)
    overlap_n_r0: int = 24                         # radial quadrature nodes
    ring_quadrature: RingQuadrature = "sinc"       # "sinc" (physical ring
                                                   #   integral) | "midpoint"
                                                   #   (paper Eq. 4 verbatim)
    efficiency_corr_npz: Optional[str] = None      # RCWA correction table

    # -- fabrication ------------------------------------------------------
    ring_width_um: float = 2.0                     # DELTA
    h_max_um: float = 15.0                         # H
    dh_um: float = 0.078                           # height quantum

    # -- optimizer --------------------------------------------------------
    seed_mode: SeedMode = "harmonic"               # "harmonic" | "ladder"
    seed_lam0_um: List[float] = field(             # harmonic: lam0 grid
        default_factory=lambda: [0.45, 0.50, 0.55, 0.60, 0.70])
    verbatim_fig_s3: bool = False                  # True: binary GA + verbatim
                                                   #   HJA exactly as Fig. S3
    ga_blocks: int = 3                             # s
    ga_epochs: int = 30                            # p (per block)
    pop_size: int = 24                             # GA population size
    gradient_iters: int = 300
    hja_d0: int = 2                                # polish: initial step
    hja_max_sweeps: int = 80                       # polish: sweep cap
    rng_seed: int = 7
    compute_upper_bound: bool = True               # continuous ceiling (info)
    bound_n_rho: int = 256
    bound_n_wavelengths: int = 512

    # -- verification (consumed by 02_Rayleigh_Sommerfeld_Validation_OOP) -------------------
    verify_wavelengths_um: List[float] = field(
        default_factory=lambda: [float(v) for v in PAPER_COMB_14])
    verify_z_span_um: float = 2500.0               # on-axis scan F +/- span
    verify_z_points: int = 401
    verify_r_max_um: float = 30.0                  # PSF radial grid
    verify_r_points: int = 601
    rzmap_r_max_um: float = 20.0                   # I(r,z) tiles (Fig. 2e window)
    rzmap_r_points: int = 41
    rzmap_z_span_um: float = 1000.0
    rzmap_z_points: int = 121

    # -- packaging --------------------------------------------------------
    dll_file_no: int = 2                           # ring table mdl_rings_<n>.txt
                                                   #   (one number per design!)
    reuse_design_npy: Optional[str] = None         # repackage this m.npy,
                                                   #   no optimization
    init_design_npy: Optional[str] = None          # WARM START from this m.npy
    runs_dir: str = "runs"                         # under the package root
    snapshot_scripts: List[str] = field(          # files or folders copied
        default_factory=lambda: list(STAGE_SNAPSHOT))   # into <run>/scripts/

    # -- validation and derived quantities ---------------------------------
    def __post_init__(self) -> None:
        if bool(self.focal_um) == bool(self.na):
            raise ValueError("config %r: set exactly one of focal_um / na"
                             % self.name)
        if self.fom_mode not in ("mean", "geomean", "softmin"):
            raise ValueError("fom_mode must be mean | geomean | softmin")
        if self.ring_quadrature not in ("sinc", "midpoint"):
            raise ValueError("ring_quadrature must be sinc | midpoint")
        if str(self.seed_mode) == "echelle":       # pre-2026-09-16 name
            object.__setattr__(self, "seed_mode", "ladder")
        if self.seed_mode not in ("harmonic", "ladder"):
            raise ValueError("seed_mode must be harmonic | ladder")
        if self.seed_mode == "ladder" and self.target_wavelengths_um is None:
            raise ValueError("seed_mode 'ladder' needs a target_wavelengths_um "
                             "comb (the fold chart is built on the lines)")

    @property
    def comb_mode(self) -> bool:
        return self.target_wavelengths_um is not None

    def geometry(self) -> Tuple[float, float, float]:
        """(D_um, NA, F_um) from diameter + (focal_um XOR na)."""
        D = float(self.diameter_um)
        R = 0.5 * D
        if self.focal_um:
            F = float(self.focal_um)
            na = R / np.sqrt(R * R + F * F)
        else:
            assert self.na is not None
            na = float(self.na)
            F = R * np.sqrt(1.0 / na ** 2 - 1.0)
        return D, na, F

    def alias_free_nw_min(self) -> Tuple[float, int]:
        """(L_max_um, minimum n_wavelengths) for an alias-free continuous
        band: the largest optical-path difference across the aperture
        must be sampled at Nw >= L_max (1/lam_min - 1/lam_max)."""
        D, na, F = self.geometry()
        R = 0.5 * D
        L_max = float(np.sqrt(R * R + F * F) - F)
        return L_max, int(np.ceil(L_max * (1.0 / self.lam_min_um
                                            - 1.0 / self.lam_max_um)))

    def to_json(self) -> Dict[str, Any]:
        """Plain dict with the config.json key set of the pre-refactor driver."""
        return dataclasses.asdict(self)

    @classmethod
    def describe(cls) -> str:
        """The field comments, for people: name, type, default."""
        rows = []
        for f in dataclasses.fields(cls):
            d = f.default if f.default is not dataclasses.MISSING else (
                f.default_factory() if f.default_factory is not dataclasses.MISSING
                else "(required)")
            rows.append("  %-24s %s" % (f.name, d))
        return "\n".join(rows)


# --- presets ---------------------------------------------------------------
S3_CONTINUOUS = DesignConfig(name="s3_continuous")

# paper-style comb objective: same geometry, optimize on the 14 lines
S3_COMB = replace(S3_CONTINUOUS, name="s3_comb",
                  target_wavelengths_um=[float(v) for v in PAPER_COMB_14],
                  dll_file_no=3)

# the original D = 1 cm, NA 0.3 broadband design point
NA03 = replace(S3_CONTINUOUS, name="na03_broadband", diameter_um=10000.0,
               focal_um=None, na=0.3, ring_width_um=0.65, h_max_um=28.0,
               n_wavelengths=2001, ga_epochs=25, pop_size=20, rng_seed=42,
               gradient_iters=250, hja_max_sweeps=60, dll_file_no=1)

# SWIR 1100-1800 nm on the S3 geometry (n_az4562 extrapolated; resist
# absorption not modelled -- verify n AND k before tape-out)
SWIR_LAMS_15 = [round(1.10 + 0.05 * i, 3) for i in range(15)]
SWIR_CONTINUOUS = replace(S3_CONTINUOUS, name="swir_continuous",
                          lam_min_um=1.10, lam_max_um=1.80,
                          n_wavelengths=501, fom_mode="mean",
                          seed_lam0_um=[1.20, 1.30, 1.45, 1.60, 1.75],
                          verify_wavelengths_um=SWIR_LAMS_15,
                          verify_r_max_um=50.0, rzmap_r_max_um=30.0,
                          dll_file_no=4)
SWIR_COMB_GEO = replace(SWIR_CONTINUOUS, name="swir_comb_geo",
                        target_wavelengths_um=SWIR_LAMS_15,
                        fom_mode="geomean", dll_file_no=5)

# ladder-seeded softmin + encircled-energy objective on the 14-line comb
S3_COMB_SOFTMIN = replace(S3_COMB, name="s3_comb_softmin_overlap",
                          fom_mode="softmin", seed_mode="ladder",
                          overlap_fom=True, dll_file_no=6)

# run 3 candidate (2026-09-15): main-lobe disc, sinc quadrature
S3_COMB_SOFTMIN_A1 = replace(S3_COMB_SOFTMIN, name="s3_comb_softmin_a1",
                             ring_quadrature="sinc", overlap_airy_factor=1.0,
                             dll_file_no=7)

SETTINGS = S3_COMB_SOFTMIN_A1        # <-- EDIT: pick / customize a preset


# =========================================================================
# no user-serviceable parts below
# =========================================================================
class RunFolder:
    """The time-stamped output folder of one run and its files."""

    def __init__(self, cfg: DesignConfig, pkg_root: str) -> None:
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(pkg_root, cfg.runs_dir,
                                "%s_%s" % (self.stamp, cfg.name))
        self.pkg_root = pkg_root
        os.makedirs(os.path.join(self.dir, "scripts"))

    @property
    def rel(self) -> str:
        return os.path.relpath(self.dir)

    def snapshot(self, entries: Sequence[str], log: Callable[[str], None]) -> List[str]:
        """Copy files (and whole folders) into scripts/ for provenance.
        Returns the entries actually copied; missing ones are logged."""
        copied: List[str] = []
        for fn in entries:
            src = os.path.join(self.pkg_root, fn)
            dst = os.path.join(self.dir, "scripts", os.path.basename(fn))
            if os.path.isdir(src):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            elif os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                log("  (snapshot: %s not found -- skipped)" % fn)
                continue
            copied.append(fn)
        return copied

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        with open(os.path.join(self.dir, name), "w") as fh:
            json.dump(payload, fh, indent=1)

    def write_vector(self, m: IntVec) -> None:
        np.save(os.path.join(self.dir, "m_final.npy"), m)

    def write_ring_table(self, prob: MDLProblem, m: IntVec, file_no: int) -> str:
        """mdl_rings_<n>.txt: 'N delta_mm' then one height per ring in mm."""
        name = "mdl_rings_%d.txt" % file_no
        with open(os.path.join(self.dir, name), "w") as fh:
            fh.write("%d %.9f\n" % (prob.N, prob.delta / 1000.0))
            fh.writelines("%.9f\n" % (v * prob.dh / 1000.0) for v in m)
        return name


class DesignRun:
    """One execution of the design pipeline for a DesignConfig."""

    def __init__(self, cfg: DesignConfig, pkg_root: str = PKG_ROOT) -> None:
        self.cfg = cfg
        self.pkg_root = pkg_root
        self.t0 = time.time()
        self.stages: Dict[str, Any] = {}

    def log(self, msg: str) -> None:
        print("[%7.1fs] %s" % (time.time() - self.t0, msg), flush=True)

    # -- 1. problem ----------------------------------------------------------
    def build_problem(self) -> MDLProblem:
        cfg = self.cfg
        D, na, F = cfg.geometry()
        L_max, nw_min = cfg.alias_free_nw_min()
        if not cfg.comb_mode and cfg.n_wavelengths < nw_min:
            raise SystemExit("alias error: n_wavelengths=%d < %d required for "
                             "L_max=%.0f um" % (cfg.n_wavelengths, nw_min, L_max))
        self.log("geometry: D=%.2f mm  F=%.2f mm  NA=%.4f  (L_max=%.0f um, "
                 "alias-free Nw >= %d)" % (D / 1000, F / 1000, na, L_max, nw_min))
        nw = (len(cfg.target_wavelengths_um) if cfg.target_wavelengths_um is not None
              else cfg.n_wavelengths)
        prob = MDLProblem(D, na, cfg.lam_min_um, cfg.lam_max_um, cfg.ring_width_um,
                          cfg.h_max_um, cfg.dh_um, n_wavelengths=nw,
                          ring_quadrature=cfg.ring_quadrature)
        if cfg.comb_mode:
            lams = np.asarray(cfg.target_wavelengths_um, dtype=float)
            prob.set_wavelengths(lams)
            self.log("objective: discrete comb on %d wavelengths: %s"
                     % (lams.size, ", ".join("%.0f" % (l * 1000) for l in lams)))
        else:
            self.log("objective: continuous band %.0f-%.0f nm, Nw=%d"
                     % (cfg.lam_min_um * 1000, cfg.lam_max_um * 1000, nw))
        prob.fom_mode = cfg.fom_mode
        if cfg.fom_mode == "softmin":
            prob.softmin_beta = float(cfg.softmin_beta)
            self.log("softmin objective: beta=%.0f (annealed to %s in the "
                     "gradient stage)" % (prob.softmin_beta, cfg.softmin_beta_final))
        self.log("ring quadrature in the FOM tables: %s (%s; rim factor S = %.3f "
                 "at %.0f nm, %.3f at %.0f nm)"
                 % (cfg.ring_quadrature,
                    "analytic ring integral of the kernel phase ramp"
                    if cfg.ring_quadrature == "sinc" else
                    "one kernel sample per ring, paper Eq. 4 -- overstates the "
                    "short lines",
                    prob.S[np.argmin(prob.lam), -1], 1000 * prob.lam.min(),
                    prob.S[np.argmax(prob.lam), -1], 1000 * prob.lam.max()))
        prob.use_single_precision()
        if cfg.efficiency_corr_npz:
            lam_c, r_c, corr_c = load_efficiency_table(cfg.efficiency_corr_npz)
            prob.apply_efficiency(lam_c, r_c, corr_c)
            self.log("rigorous efficiency correction applied from %s (%d "
                     "wavelengths x %d zones, ratio %.2f-%.2f)"
                     % (cfg.efficiency_corr_npz, lam_c.size, r_c.size,
                        corr_c.min(), corr_c.max()))
        if cfg.overlap_fom:
            prob.enable_overlap_fom(cfg.overlap_r_enc_um, cfg.overlap_n_r0,
                                    cfg.overlap_airy_factor)
            if cfg.overlap_r_enc_um is None:
                self.log("overlap objective: CHROMATIC disc r_enc(lam) = %.1f Airy "
                         "radii = %.2f-%.2f um across the band (%d radial nodes, "
                         "kernel table %.0f MB)"
                         % (cfg.overlap_airy_factor, prob.r_enc.min(),
                            prob.r_enc.max(), prob._enc_r0.shape[1],
                            prob.K.nbytes / 1e6))
            else:
                self.log("overlap objective: FIXED disc r <= %.2f um (%d radial "
                         "nodes, kernel table %.0f MB)"
                         % (prob.r_enc[0], prob._enc_r0.shape[1], prob.K.nbytes / 1e6))
        self.log("problem: N=%d rings, M=%d levels, fom_mode=%s, objective=%s"
                 % (prob.N, prob.M, prob.fom_mode, prob.objective))
        self.prob = prob
        return prob

    # -- 2. seed ----------------------------------------------------------------
    def make_seed(self) -> Tuple[IntVec, float, Dict[str, Any]]:
        """(m_seed, J_seed, seed record for design_metrics.json)."""
        cfg, prob = self.cfg, self.prob
        if cfg.init_design_npy:
            src = cfg.init_design_npy
            # int64 as the pre-refactor driver (values are what matter; the
            # optimizers accept any integer dtype)
            m = cast(IntVec, np.asarray(np.load(src), dtype=np.int64))
            if m.size != prob.N:
                raise SystemExit("init error: %s has %d rings, config needs %d"
                                 % (src, m.size, prob.N))
            record: Dict[str, Any] = {"mode": "init",
                                      "init_from": os.path.abspath(src)}
            src_dm = os.path.join(os.path.dirname(os.path.abspath(src)),
                                  "design_metrics.json")
            if os.path.exists(src_dm):
                src_seed = json.load(open(src_dm)).get("seed") or {}
                for k in ("h_fold_um", "orders", "lam0_um", "p"):
                    if k in src_seed:
                        record[k] = src_seed[k]
                record["source_seed_mode"] = src_seed.get("mode")
            J = prob.fom(m)
            self.log("warm start from %s: J=%.4f under THIS run's objective "
                     "(levels %d..%d used)" % (src, J, m.min(), m.max()))
            return m, J, record
        if cfg.seed_mode == "ladder":
            res = LadderSeed(prob, np.asarray(cfg.target_wavelengths_um, float),
                             log=self.log).run()
            self.log("ladder seed: H_fold=%.4f um  J=%.4f" % (res.h_fold_um, res.J))
            return res.m, res.J, res.record()
        best: Optional[Tuple[IntVec, float, Tuple[float, int]]] = None
        for lam0 in cfg.seed_lam0_um:
            n0 = float(prob.n_func(lam0))
            pmax = max(1, int(np.floor(cfg.h_max_um * (n0 - 1.0) / lam0)))
            for p in sorted({1, 2, 4, 8, pmax // 2, pmax}):
                if p < 1:
                    continue
                m_h = harmonic_seed(prob, lam0, p)
                f = prob.fom(m_h)
                if best is None or f > best[1]:
                    best = (m_h, f, (lam0, p))
        assert best is not None
        self.log("best harmonic seed: lam0=%.2f p=%d  J=%.4f"
                 % (best[2][0], best[2][1], best[1]))
        return best[0], best[1], {"mode": "harmonic", "lam0_um": best[2][0],
                                  "p": best[2][1]}

    # -- 3. optimize -----------------------------------------------------------
    def optimize(self, m_seed: IntVec) -> IntVec:
        cfg, prob = self.cfg, self.prob
        rng = np.random.default_rng(cfg.rng_seed)
        if cfg.verbatim_fig_s3:
            r_s = MultistepGAHJACombo(prob, s=cfg.ga_blocks, p=cfg.ga_epochs,
                                      pop_size=cfg.pop_size, rng=rng,
                                      log=self.log).run(m_seed)
        else:
            r_s = SearchGAHJA(prob, blocks=cfg.ga_blocks, ga_epochs=cfg.ga_epochs,
                              pop_size=cfg.pop_size, rng=rng, log=self.log).run(m_seed)
        self.log("Search: J=%.4f" % r_s.J)
        r_sm = Smooth(prob).run(r_s.m)
        self.log("Smooth: %.4f -> %.4f" % (r_s.J, r_sm.J))
        beta_final = cfg.softmin_beta_final if cfg.fom_mode == "softmin" else None
        r_g = GradientRefine(prob, iters=cfg.gradient_iters,
                             softmin_beta_final=beta_final).run(r_sm.m)
        self.log("Gradient: J=%.4f%s" % (
            r_g.J, ("  (softmin beta annealed to %.0f)" % prob.softmin_beta)
            if beta_final else ""))
        r_f = HookeJeeves(prob, d0=cfg.hja_d0, max_sweeps=cfg.hja_max_sweeps).run(r_g.m)
        self.log("Polish: J=%.4f" % r_f.J)
        self.stages.update(J_search=r_s.J, J_smooth=r_sm.J, J_gradient=r_g.J,
                           J_final=r_f.J)
        return r_f.m

    # -- 4. the whole run --------------------------------------------------------
    def run(self) -> str:
        cfg = self.cfg
        folder = RunFolder(cfg, self.pkg_root)
        copied = folder.snapshot(cfg.snapshot_scripts, self.log)
        self.log("run folder: %s   (mdl %s); scripts/ <- %s"
                 % (folder.rel, __version__, ", ".join(copied)))
        prob = self.build_problem()
        D, na, F = cfg.geometry()
        L_max, nw_min = cfg.alias_free_nw_min()
        cfg_out = cfg.to_json()
        cfg_out["derived"] = {"na": float(na), "focal_um": float(F),
                              "N_rings": int(prob.N), "M_levels": int(prob.M),
                              "L_max_um": float(L_max),
                              "alias_free_nw_min": nw_min,
                              "timestamp": folder.stamp,
                              "mdl_version": __version__}
        folder.write_json("config.json", cfg_out)

        bound = None
        if cfg.compute_upper_bound:
            bound = upper_bound_jf(D, na, cfg.lam_min_um, cfg.lam_max_um,
                                   cfg.h_max_um, cfg.dh_um, n_rho=cfg.bound_n_rho,
                                   n_wavelengths=cfg.bound_n_wavelengths)
            self.log("alias-free continuous-band ceiling: %.4f" % bound)

        if cfg.reuse_design_npy:
            m_f = np.load(cfg.reuse_design_npy)
            if m_f.size != prob.N:
                raise SystemExit("reuse error: %s has %d rings, config needs %d"
                                 % (cfg.reuse_design_npy, m_f.size, prob.N))
            self.stages = {"J_final": prob.fom(m_f),
                           "reused_from": cfg.reuse_design_npy}
            self.log("reused %s: J=%.4f (no optimization)"
                     % (cfg.reuse_design_npy, self.stages["J_final"]))
        else:
            m_seed, J_seed, record = self.make_seed()
            self.stages = {"J_seed": J_seed, "seed": record}
            m_f = self.optimize(m_seed)
        self.stages["upper_bound_continuous"] = bound

        folder.write_vector(m_f)
        ring_name = folder.write_ring_table(prob, m_f, cfg.dll_file_no)
        folder.write_json("design_metrics.json", self.stages)
        zt_lams = np.asarray(cfg.target_wavelengths_um if cfg.comb_mode
                             else cfg.verify_wavelengths_um, float)
        zones = extract_local_gratings(prob, np.asarray(m_f, np.int32))
        write_zone_table(os.path.join(folder.dir, "zone_table.npz"), zones,
                         zt_lams, prob)
        self.log("zone_table.npz: %d local-grating zones (outermost period %.1f "
                 "um) -- sweep with a rigorous solver, form corr = eta_rcwa / "
                 "eta_tea with mdl.zones.relative_correction_table, then redesign "
                 "with efficiency_corr_npz" % (len(zones), zones[-1]["period_um"]))
        self.log("saved: m_final.npy, %s, design_metrics.json, zone_table.npz"
                 % ring_name)
        self.log("done -> %s" % folder.rel)
        self.log("next:  python %s %s" % (NEXT_STAGE_CMD, folder.rel))
        return folder.dir


def main(cfg: DesignConfig = SETTINGS) -> str:
    return DesignRun(cfg).run()


if __name__ == "__main__":
    main()
