"""The lens specification, its derived quantities, the two ceilings and
the feasibility verdict.

Ceilings
--------
* NUMERIC (paper Eqs. S14-S15, ``mdl.bounds.upper_bound_jf``): the
  pairwise coherence matrix max_dm Re <J_w(rho_i, rho_j)>_w contracted
  with the paraxial aperture weights. Includes the band (through the
  w average) and the resist dispersion; alias-free for the geometry
  (group-delay cut). This is the reference number ("alias-free
  ceiling", 0.0662 for S3 at 256 x 512).
* ANALYTIC (paper Eq. 7 / S17-S19): only the constructive region
  max Re J = 1 is integrated, paraxial, no dispersion, no bandwidth,

      D_max = 4 (n_max - 1) H / [ (1 - sqrt(1 - max J)) NA ]

  inverted here as  x = 4 (n_max - 1) H / (D NA),
                    max J = 1 - (1 - x)^2   (x < 1; else 1).
  Rough by construction (the paper: "just a rough approximation"), but
  instant and monotone, and it gives the H a target needs in closed
  form: H_req = x_t D NA / (4 (n_max - 1)), x_t = 1 - sqrt(1 - J_t).

What a design reaches
---------------------
Measured on this toolchain: continuous-band designs land at ~45-60 % of
the numeric ceiling (S3 visible 0.04 of 0.066; SWIR 0.041 of 0.088);
comb objectives on a line set score higher on their own metric (run 3:
0.14 on 14 lines, main-lobe encircled energy) and are not bounded by
the continuous ceiling. The verdict below therefore compares the
target with ceiling x ``achievable_fraction`` (default 0.55).
"""
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 01_design_oo
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mdl.bounds import pairwise_bound_matrix, upper_bound_jf   # noqa: E402
from mdl.material import n_az4562                                # noqa: E402

IndexFn = Callable[..., np.ndarray]
MATERIALS: Dict[str, IndexFn] = {"AZ4562": n_az4562}

#: the paper's five reference lenses (NA 0.1, 400-1100 nm, AZ4562):
#: name -> (D_mm, H_um, paper max J (Fig. 1d), our alias-free ceiling)
REFERENCE_POINTS: Dict[str, Tuple[float, float, float, float]] = {
    "S1": (1.0, 15.0, 0.72, 0.52), "S2": (3.0, 15.0, 0.43, 0.21),
    "S3": (10.0, 15.0, 0.21, 0.069), "S4": (10.0, 5.0, 0.11, 0.028),
    "S5": (10.0, 1.0, 0.03, 0.023),
}

INCH_MM = 25.4


@dataclass(frozen=True)
class LensSpec:
    """One point of the design space. Give ``na`` OR ``fnum`` (the other
    is derived exactly: NA = 1 / sqrt(1 + 4 F#^2), F# = F / D)."""
    diameter_mm: float
    lam_min_um: float
    lam_max_um: float
    h_max_um: float = 15.0
    dh_um: float = 0.078
    ring_width_um: float = 2.0
    na: Optional[float] = None
    fnum: Optional[float] = None
    material: str = "AZ4562"
    name: str = "spec"

    def __post_init__(self) -> None:
        if (self.na is None) == (self.fnum is None):
            raise ValueError("give exactly one of na / fnum")
        if self.diameter_mm <= 0 or self.lam_min_um <= 0 or self.lam_max_um <= self.lam_min_um:
            raise ValueError("diameter > 0 and lam_min < lam_max required")
        if self.na is not None and not (0 < self.na < 1):
            raise ValueError("NA must be in (0, 1)")
        if self.fnum is not None and self.fnum <= 0:
            raise ValueError("F-number must be > 0")
        if self.material not in MATERIALS:
            raise ValueError("unknown material %r (known: %s)" % (self.material, sorted(MATERIALS)))

    # -- geometry ---------------------------------------------------------------
    @property
    def n_func(self) -> IndexFn:
        return MATERIALS[self.material]

    @property
    def D_um(self) -> float:
        return self.diameter_mm * 1000.0

    @property
    def R_um(self) -> float:
        return 0.5 * self.D_um

    @property
    def NA(self) -> float:
        if self.na is not None:
            return float(self.na)
        assert self.fnum is not None
        return float(1.0 / np.sqrt(1.0 + 4.0 * float(self.fnum) ** 2))

    @property
    def F_um(self) -> float:
        return float(self.R_um * np.sqrt(1.0 / self.NA ** 2 - 1.0))

    @property
    def F_number(self) -> float:
        return self.F_um / self.D_um

    @property
    def n_rings(self) -> int:
        return int(round(self.R_um / self.ring_width_um))

    @property
    def n_levels(self) -> int:
        return int(round(self.h_max_um / self.dh_um))

    @property
    def diameter_inch(self) -> float:
        return self.diameter_mm / INCH_MM

    def n_at(self, lam_um: float) -> float:
        return float(self.n_func(np.asarray(lam_um)))

    @property
    def n_max(self) -> float:
        return self.n_at(self.lam_min_um)

    def fold_waves(self, lam_um: float) -> float:
        """p = (n - 1) H / lam: the order a full-height fold works in."""
        return (self.n_at(lam_um) - 1.0) * self.h_max_um / lam_um

    def rim_period_um(self, lam_um: float) -> float:
        """Local blaze period of a full-height fold at the rim,
        P = (n - 1) H / sin(theta_rim) = (n - 1) H / NA."""
        return (self.n_at(lam_um) - 1.0) * self.h_max_um / self.NA

    def rim_kernel_ramp_um(self) -> float:
        """Kernel path ramp across one rim ring, DELTA R / sqrt(R^2 + F^2)
        = DELTA NA: the quantity the sinc ring rule integrates."""
        return self.ring_width_um * self.NA

    def rim_sinc_factor(self, lam_um: float) -> float:
        """S = sinc(DELTA NA / lam): the staircase factor of the outermost
        ring (0.637 at 400 nm for S3). S^2 = local ring efficiency."""
        return float(np.sinc(self.rim_kernel_ramp_um() / lam_um))

    def alias_free_nw_min(self) -> Tuple[float, int]:
        """(L_max, Nw_min): largest path difference across the aperture and
        the continuous-band sampling it needs."""
        L = float(np.sqrt(self.R_um ** 2 + self.F_um ** 2) - self.F_um)
        return L, int(np.ceil(L * (1.0 / self.lam_min_um - 1.0 / self.lam_max_um)))

    def run_cost(self, n_wavelengths: Optional[int] = None) -> Dict[str, float]:
        """Memory of the FOM tables of a design run at this size (complex64
        G and L, Nw x N each) and a wall-time scale relative to S3."""
        nw = n_wavelengths or max(self.alias_free_nw_min()[1], 64)
        N = self.n_rings
        table_mb = 2 * nw * N * 8 / 1e6
        s3 = 1001 * 2560
        return {"n_wavelengths": nw, "n_rings": N, "tables_mb": table_mb,
                "time_scale_vs_s3": nw * N / s3}

    def diffraction_limit_um(self, lam_um: float) -> float:
        return lam_um / (2.0 * self.NA)

    def describe(self) -> List[str]:
        L, nw = self.alias_free_nw_min()
        c = self.run_cost()
        rows = [
            "D = %.3f mm (%.2f inch), R = %.3f mm" % (self.diameter_mm, self.diameter_inch,
                                                       self.R_um / 1000),
            "NA = %.4f  <->  F/%.2f,  F = %.2f mm" % (self.NA, self.F_number, self.F_um / 1000),
            "band %.0f-%.0f nm (ratio %.2f), %s: n = %.4f .. %.4f"
            % (1000 * self.lam_min_um, 1000 * self.lam_max_um, self.lam_max_um / self.lam_min_um,
               self.material, self.n_at(self.lam_max_um), self.n_at(self.lam_min_um)),
            "relief H = %.2f um in %d levels of %.3f um; rings DELTA = %.2f um -> N = %d "
            "(aspect ratio H/DELTA = %.1f)" % (self.h_max_um, self.n_levels, self.dh_um,
                                              self.ring_width_um, self.n_rings,
                                              self.h_max_um / self.ring_width_um),
            "fold order p = (n-1)H/lam: %.1f at %.0f nm .. %.1f at %.0f nm"
            % (self.fold_waves(self.lam_min_um), 1000 * self.lam_min_um,
               self.fold_waves(self.lam_max_um), 1000 * self.lam_max_um),
            "rim fold period (n-1)H/NA = %.1f um = %.1f rings; rim ring factor S = %.3f "
            "at %.0f nm, %.3f at %.0f nm"
            % (self.rim_period_um(self.lam_min_um),
               self.rim_period_um(self.lam_min_um) / self.ring_width_um,
               self.rim_sinc_factor(self.lam_min_um), 1000 * self.lam_min_um,
               self.rim_sinc_factor(self.lam_max_um), 1000 * self.lam_max_um),
            "diffraction-limited FWHM lam/2NA: %.2f um at %.0f nm .. %.2f um at %.0f nm"
            % (self.diffraction_limit_um(self.lam_min_um), 1000 * self.lam_min_um,
               self.diffraction_limit_um(self.lam_max_um), 1000 * self.lam_max_um),
            "alias-free continuous sampling: L_max = %.0f um -> Nw >= %d" % (L, nw),
            "design-run size: G + L tables %.0f MB at Nw = %d, N = %d; time scale %.2f x S3"
            % (c["tables_mb"], c["n_wavelengths"], c["n_rings"], c["time_scale_vs_s3"]),
        ]
        return rows

    def to_dict(self) -> Dict[str, object]:
        d = dataclasses.asdict(self)
        d.update(NA=self.NA, F_um=self.F_um, F_number=self.F_number, n_rings=self.n_rings)
        return d


# ---------------------------------------------------------------------------
@dataclass
class Ceiling:
    """The two upper bounds of the continuous-band max J_w(F) at one spec."""
    spec: LensSpec
    analytic: float
    numeric: Optional[float] = None
    n_rho: int = 0
    n_wavelengths: int = 0

    @staticmethod
    def analytic_of(spec: LensSpec) -> float:
        """Paper Eq. 7 inverted: max J = 1 - (1 - x)^2, x = 4 (n_max-1) H / (D NA)."""
        x = 4.0 * (spec.n_max - 1.0) * spec.h_max_um / (spec.D_um * spec.NA)
        return 1.0 if x >= 1.0 else 1.0 - (1.0 - x) ** 2

    @staticmethod
    def h_required_um(spec: LensSpec, target_j: float) -> float:
        """H that makes the analytic ceiling equal target_j at this D / NA."""
        xt = 1.0 - np.sqrt(max(0.0, 1.0 - min(target_j, 1.0)))
        return float(xt * spec.D_um * spec.NA / (4.0 * (spec.n_max - 1.0)))

    @staticmethod
    def d_max_mm(spec: LensSpec, target_j: float) -> float:
        """Paper Eq. 7: the largest diameter whose analytic ceiling is target_j."""
        xt = 1.0 - np.sqrt(max(0.0, 1.0 - min(target_j, 1.0)))
        return float(4.0 * (spec.n_max - 1.0) * spec.h_max_um / (xt * spec.NA) / 1000.0) \
            if xt > 0 else float("inf")

    @classmethod
    def compute(cls, spec: LensSpec, numeric: bool = True, n_rho: int = 256,
                n_wavelengths: int = 512) -> "Ceiling":
        c = cls(spec, cls.analytic_of(spec))
        if numeric:
            c.numeric = float(upper_bound_jf(spec.D_um, spec.NA, spec.lam_min_um,
                                             spec.lam_max_um, spec.h_max_um, spec.dh_um,
                                             n_rho=n_rho, n_wavelengths=n_wavelengths,
                                             n_func=spec.n_func))  # type: ignore[arg-type]
            c.n_rho, c.n_wavelengths = n_rho, n_wavelengths
        return c

    def pair_map(self, n_rho: int = 192, n_wavelengths: int = 384) -> Tuple[np.ndarray, np.ndarray]:
        """(rho/R, B) -- the Fig. 1b/1c map of this spec."""
        s = self.spec
        return pairwise_bound_matrix(s.D_um, s.NA, s.lam_min_um, s.lam_max_um, s.h_max_um,
                                     s.dh_um, n_rho=n_rho, n_wavelengths=n_wavelengths,
                                     n_func=s.n_func)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
@dataclass
class Feasibility:
    """Verdict of a spec against a target continuous-band J."""
    spec: LensSpec
    ceiling: Ceiling
    target_j: float = 0.05
    achievable_fraction: float = 0.55

    @property
    def bound(self) -> float:
        return self.ceiling.numeric if self.ceiling.numeric is not None else self.ceiling.analytic

    @property
    def expected_j(self) -> float:
        return self.achievable_fraction * self.bound

    @property
    def feasible(self) -> bool:
        return self.expected_j >= self.target_j

    def nearest_reference(self) -> str:
        b = self.bound
        best = min(REFERENCE_POINTS.items(), key=lambda kv: abs(np.log(max(kv[1][3], 1e-6) / max(b, 1e-6))))
        return best[0]

    def lines(self) -> List[str]:
        s, c = self.spec, self.ceiling
        rows = ["ceiling max J_w(F): analytic (Eq. 7) %.4f%s"
                % (c.analytic, ("; numeric alias-free %.4f (%d x %d)"
                                % (c.numeric, c.n_rho, c.n_wavelengths))
                   if c.numeric is not None else "")]
        rows.append("expected continuous-band design J ~ %.3f (%.0f %% of the ceiling); "
                    "target %.3f -> %s" % (self.expected_j, 100 * self.achievable_fraction,
                                           self.target_j, "FEASIBLE" if self.feasible
                                           else "NOT FEASIBLE at this H / D / NA"))
        ref = self.nearest_reference()
        Dr, Hr, Jp, Jo = REFERENCE_POINTS[ref]
        rows.append("closest paper reference: %s (D %.0f mm, H %.0f um): paper max J %.2f, "
                    "our alias-free ceiling %.3f" % (ref, Dr, Hr, Jp, Jo))
        h_req = Ceiling.h_required_um(s, self.target_j / self.achievable_fraction)
        d_max = Ceiling.d_max_mm(s, self.target_j / self.achievable_fraction)
        rows.append("to reach the target at this D and NA: H >= %.1f um (analytic); at this H "
                    "the diameter could be up to %.1f mm (%.2f inch)"
                    % (h_req, d_max, d_max / INCH_MM))
        rows.append("comb objectives (N lines, encircled energy) are judged on their own scale "
                    "and are NOT bounded by this number (run 3: 0.14 on 14 lines at the S3 point)")
        return rows
