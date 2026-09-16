"""The design under verification, loaded from a run folder.

A run folder made by ``run_MDL_design.py`` (either driver) holds
``config.json`` (settings + derived geometry) and ``m_final.npy`` (the
optimized gray-level vector). Everything this stage needs is read from
those two files; nothing is hardcoded. ``VerifyConfig`` lists the keys
that are read and the defaults used for keys older run folders lack;
``DesignState`` turns them into the arrays and the ``MDLProblem`` the
propagator and the metrics work on.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np

from .base import PKG_ROOT, Log


# ---------------------------------------------------------------------------
# which MDLProblem
# ---------------------------------------------------------------------------
def import_mdl_problem(pkg_root: str = PKG_ROOT) -> Tuple[type, str]:
    """``(MDLProblem, origin)``: the class from ``01_design_oo/mdl`` when
    that stage exists, else from the frozen ``mdl_core.py`` at the package
    root. Both build bit-identical tables (01_design_oo/tests); the origin
    string is echoed in the log so a run's J numbers are traceable."""
    oo = os.path.join(pkg_root, "01_design_oo")
    if os.path.isdir(os.path.join(oo, "mdl")):
        if oo not in sys.path:
            sys.path.insert(0, oo)
        mod = importlib.import_module("mdl")
        return mod.MDLProblem, "01_design_oo/mdl %s" % getattr(mod, "__version__", "?")
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    mod = importlib.import_module("mdl_core")
    return mod.MDLProblem, "mdl_core.py (package root)"


# ---------------------------------------------------------------------------
# config contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifyConfig:
    """Every ``config.json`` key stage 2a reads, typed. Fields without a
    default are required (written by every design driver); the others
    carry the default the legacy scripts used for run folders that
    predate the key, and ``defaulted`` names the keys that were absent.
    Lengths in micrometres.
    """
    # -- identity / geometry (required) --------------------------------------
    name: str
    diameter_um: float
    na: float                          # derived.na
    focal_um: float                    # derived.focal_um
    lam_min_um: float
    lam_max_um: float
    ring_width_um: float               # DELTA
    h_max_um: float
    dh_um: float
    n_wavelengths: int                 # continuous-band sampling of the FOM
    target_wavelengths_um: Optional[List[float]]   # None = continuous objective
    # -- verification grids (required) ---------------------------------------
    verify_wavelengths_um: List[float]
    verify_z_span_um: float            # on-axis scan F +/- span
    verify_z_points: int
    verify_r_max_um: float             # focal-plane PSF radial window
    verify_r_points: int
    dll_file_no: int                   # ring table mdl_rings_<n>.txt
    # -- optional keys (legacy defaults) -------------------------------------
    fom_mode: str = "mean"
    rzmap_r_max_um: float = 20.0       # I(r,z) tiles (paper Fig. 2e window)
    rzmap_r_points: int = 41
    rzmap_z_span_um: float = 1000.0
    rzmap_z_points: int = 121
    ring_quadrature: str = "midpoint"  # ring rule of the DESIGN-FOM tables
                                       #   (J metrics); pre-2026-09-15 runs
                                       #   have no key -> midpoint
    rs_ring_quadrature: Literal["sinc", "midpoint"] = "sinc"   # RS propagation rule
                                       #   (all verification metrics)
    mtf_r_max_um: Optional[float] = None       # None -> verify_r_max_um
    mtf_r_points: int = 1601
    mtf_f_points: int = 400
    mtf_f_max_lppmm: Optional[float] = None    # None -> 1.05 x 2NA/lam_min
    # -- bookkeeping -----------------------------------------------------------
    defaulted: List[str] = field(default_factory=list)   # keys absent in file

    def __post_init__(self) -> None:
        object.__setattr__(self, "rs_ring_quadrature",
                           cast(Any, str(self.rs_ring_quadrature).lower()))
        if self.rs_ring_quadrature not in ("sinc", "midpoint"):
            raise SystemExit("rs_ring_quadrature must be 'sinc' or 'midpoint', "
                             "got %r" % self.rs_ring_quadrature)
        object.__setattr__(self, "ring_quadrature", str(self.ring_quadrature))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "VerifyConfig":
        """Build from the loaded ``config.json`` dict (``derived`` block
        supplies na / focal_um). Optional keys fall back to the class
        defaults and are recorded in ``defaulted``."""
        der = cfg["derived"]
        kw: Dict[str, Any] = {}
        defaulted: List[str] = []
        for f in dataclasses.fields(cls):
            if f.name == "defaulted":
                continue
            src = der if f.name in ("na", "focal_um") else cfg
            if f.name in src:
                kw[f.name] = src[f.name]
            elif f.default is not dataclasses.MISSING:
                kw[f.name] = f.default
                defaulted.append(f.name)
            else:
                raise SystemExit("config.json lacks the required key %r" % f.name)
        kw["defaulted"] = defaulted
        return cls(**kw)

    # -- derived -------------------------------------------------------------
    @property
    def comb_objective(self) -> bool:
        return self.target_wavelengths_um is not None

    @property
    def fom_quadrature_key_present(self) -> bool:
        return "ring_quadrature" not in self.defaulted

    def verify_lams(self) -> np.ndarray:
        return np.asarray(self.verify_wavelengths_um, dtype=float)

    def mtf_r_max(self) -> float:
        return float(self.verify_r_max_um if self.mtf_r_max_um is None
                     else self.mtf_r_max_um)

    def mtf_f_max(self) -> float:
        """Frequency-axis end in lp/mm: 1.05 x the highest cutoff 2NA/lam_min."""
        if self.mtf_f_max_lppmm is not None:
            return float(self.mtf_f_max_lppmm)
        return float(1.05 * (2 * self.na / self.lam_min_um) * 1000.0)


# ---------------------------------------------------------------------------
# the loaded design
# ---------------------------------------------------------------------------
class DesignState:
    """The optimized design and its geometry, read once per run.

    Attributes
    ----------
    run_dir      the run folder
    cfg          VerifyConfig (typed view of config.json)
    raw          the config.json dict as loaded
    m_file       path of the gray-level vector verified (m_final.npy unless
                 another .npy was given on the command line)
    m            gray levels m_i (N integers)
    prob         MDLProblem on the run's own FOM ring rule
                 (``cfg.ring_quadrature``), continuous band, Nw =
                 n_wavelengths -- used for J_continuous and for n(lam)
    backend      where MDLProblem came from (echoed in the log)
    D, R, F, na  diameter, aperture radius, focal length, numerical aperture
    N, delta     number of rings, ring width DELTA
    h            ring heights h_i = m_i dh                               [um]
    rho          ring centre radii (i + 1/2) DELTA                        [um]
    lams         verification wavelengths                                 [um]
    """

    def __init__(self, run_dir: str, m_file: Optional[str] = None,
                 pkg_root: str = PKG_ROOT) -> None:
        self.run_dir: str = run_dir
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path) as fh:
            self.raw: Dict[str, Any] = json.load(fh)
        self.cfg: VerifyConfig = VerifyConfig.from_dict(self.raw)
        c = self.cfg
        self.m_file: str = m_file or os.path.join(run_dir, "m_final.npy")
        self.m: np.ndarray = np.load(self.m_file)

        MDLProblem, self.backend = import_mdl_problem(pkg_root)
        self._MDLProblem = MDLProblem
        self.prob = MDLProblem(c.diameter_um, c.na, c.lam_min_um, c.lam_max_um,
                               c.ring_width_um, c.h_max_um, c.dh_um,
                               n_wavelengths=c.n_wavelengths,
                               ring_quadrature=c.ring_quadrature)
        if self.m.size != self.prob.N:
            raise SystemExit("%s has %d rings; config expects %d"
                             % (self.m_file, self.m.size, self.prob.N))
        self.D: float = float(c.diameter_um)
        self.na: float = float(c.na)
        self.F: float = float(c.focal_um)
        self.R: float = float(self.prob.R)
        self.N: int = int(self.prob.N)
        self.delta: float = float(self.prob.delta)
        self.h: np.ndarray = self.m * self.prob.dh
        self.rho: np.ndarray = self.prob.rho
        self.lams: np.ndarray = c.verify_lams()

    # -- FOM problems on other wavelength sets ------------------------------------
    def comb_problem(self, lams: np.ndarray) -> Any:
        """A second MDLProblem (same geometry and FOM ring rule) evaluated
        on the discrete comb ``lams`` -- J_verify_comb / J_objective."""
        c = self.cfg
        return self._MDLProblem(c.diameter_um, c.na, c.lam_min_um, c.lam_max_um,
                                c.ring_width_um, c.h_max_um, c.dh_um,
                                n_wavelengths=lams.size,
                                ring_quadrature=c.ring_quadrature
                                ).set_wavelengths(lams)

    def n_of(self, lam: float) -> float:
        """Resist index n(lam) of the design's material model."""
        return float(self.prob.n_func(lam))

    # -- descriptions ---------------------------------------------------------------
    def rim_ramp_um(self) -> float:
        """Kernel optical-path ramp across the outermost ring,
        DELTA R / sqrt(R^2 + F^2) (the quantity the sinc rule integrates)."""
        return float(self.delta * self.R / np.sqrt(self.R * self.R + self.F * self.F))

    def describe(self, log: Log, m_line: bool = True) -> None:
        """The configuration block every driver prints."""
        c = self.cfg
        log("run: %s" % self.run_dir)
        log("design '%s': D=%.2f mm, F=%.2f mm, NA=%.4f | %d rings x %.2f um, "
            "H=%.1f um (%d levels x %g nm)"
            % (c.name, self.D / 1000, self.F / 1000, self.na, self.N, self.delta,
               c.h_max_um, self.prob.M, c.dh_um * 1000))
        log("objective: %s, fom_mode=%s | band %.0f-%.0f nm | verifying at %d "
            "wavelengths"
            % ("discrete comb (%d lines)" % len(c.target_wavelengths_um)
               if c.target_wavelengths_um is not None else "continuous band",
               c.fom_mode, c.lam_min_um * 1000, c.lam_max_um * 1000, self.lams.size))
        if m_line:
            log("design vector: %s (levels 0..%d used, h_max=%.2f um)"
                % (self.m_file, int(self.m.max()), float(self.h.max())))
        log("MDLProblem from %s" % self.backend)
        if c.defaulted:
            log("config keys absent -> defaults: %s"
                % ", ".join("%s=%s" % (k, "auto" if getattr(c, k) is None
                                       else getattr(c, k)) for k in c.defaulted))

    def describe_quadratures(self, log: Log) -> None:
        c = self.cfg
        ramp = self.rim_ramp_um()
        log("RS ring quadrature: %s%s"
            % (c.rs_ring_quadrature,
               " (analytic ring integral of the kernel phase ramp; rim ramp "
               "%.3f um = %.2f waves at %.0f nm)"
               % (ramp, ramp / self.lams.min(), 1000 * self.lams.min())
               if c.rs_ring_quadrature == "sinc" else
               " (one kernel sample per ring -- paper Eq. 4; overstates the "
               "focal field at short wavelengths, see rsval.propagator)"))
        log("design-FOM ring quadrature (J metrics): %s%s"
            % (c.ring_quadrature, "" if c.fom_quadrature_key_present else
               "  [config has no ring_quadrature key -> pre-2026-09-15 run]"))

    # -- fabrication-facing re-export ---------------------------------------------
    def write_ring_table(self) -> str:
        """``mdl_rings_<n>.txt`` in the run ROOT (heights in mm!), re-written
        so the table the Zemax DLLs and export_gds.py read always matches
        the vector actually verified. Returns the path."""
        path = os.path.join(self.run_dir, "mdl_rings_%d.txt" % self.cfg.dll_file_no)
        with open(path, "w") as fh:
            fh.write("%d %.9f\n" % (self.N, self.delta / 1000.0))
            fh.writelines("%.9f\n" % (v / 1000.0) for v in self.h)
        return path
