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

#: the paper's five reference lenses (NA 0.1, 400-1100 nm, AZ4562, 2 um rings):
#: name -> (D_mm, H_um, paper max J (Fig. 1d), our alias-free ceiling at 128 x 256)
#: Paper vs ours differ by 1.3-3.7x because the paper samples the band
#: sparsely (findings 2026-07/08); the ORDER of the five is the same on
#: both scales, which is what the class verdict below relies on.
REFERENCE_POINTS: Dict[str, Tuple[float, float, float, float]] = {
    "S1": (1.0, 15.0, 0.72, 0.536), "S2": (3.0, 15.0, 0.43, 0.214),
    "S3": (10.0, 15.0, 0.21, 0.068), "S4": (10.0, 5.0, 0.11, 0.030),
    "S5": (10.0, 1.0, 0.03, 0.014),
}
#: what the paper reports for each class beyond max J (Fig. 1d): S3 is the
#: fabricated cm-scale achromat (measured mean focusing efficiency 0.31 on
#: the 14 lines, all 14 foci at F, Fig. 2e); S5 the thin-relief limit.
REFERENCE_NOTES: Dict[str, str] = {
    "S1": "1 mm, H 15 um: near-ideal broadband achromat (paper max J 0.72)",
    "S2": "3 mm, H 15 um: strong achromat (paper max J 0.43)",
    "S3": "10 mm, H 15 um: the paper's fabricated cm-scale lens (max J 0.21, measured "
          "mean efficiency 0.31 on 14 lines, every line focused at F)",
    "S4": "10 mm, H 5 um: weak achromat (paper max J 0.11)",
    "S5": "10 mm, H 1 um: thin-relief limit, chromatic (paper max J 0.03)",
}
REFERENCE_ORDER: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")
#: the paper's own feasibility criterion (S2-6): "to design an AMDL with
#: relatively good performance, the max J_w(F) needs to be at least 0.2".
#: On the paper's sparsely sampled scale 0.2 is its S3 lens (0.21), so on
#: our alias-free scale the threshold is S3's ceiling, computed at the
#: resolution of the check. Band-independent; needs no user input.
PAPER_THRESHOLD_CLASS = "S3"
PAPER_THRESHOLD_J_PAPER_SCALE = 0.2

INCH_MM = 25.4
#: address grid the automatic (Nyquist) ring width is rounded DOWN to [um]
RING_GRID_UM = 0.05


@dataclass(frozen=True)
class LensSpec:
    """One point of the design space. Give ``na`` OR ``fnum`` (the other
    is derived exactly: NA = 1 / sqrt(1 + 4 F#^2), F# = F / D).

    Attributes
    ----------
    diameter_mm    aperture D (any size; INCH_MM converts inches)
    lam_min_um, lam_max_um   working band
    h_max_um       relief height H
    dh_um          height quantum (levels = H / dh)
    ring_width_um  ring width DELTA, or None = the Nyquist width
                   lam_min / (2 NA) rounded down to RING_GRID_UM (paper
                   S2-6: "the width of ring is set as 0.7 um to satisfy
                   Nyquist-Shannon": 450 nm / (2 x 0.3) = 0.75 -> 0.7;
                   S3: 400 nm / (2 x 0.1) = 2.0 um). Use ``delta_um``
                   for the value in force.
    na / fnum      numerical aperture xor F-number
    material       key of MATERIALS (index model n(lam))
    name           label for the run folder
    """
    diameter_mm: float
    lam_min_um: float
    lam_max_um: float
    h_max_um: float = 15.0
    dh_um: float = 0.078
    ring_width_um: Optional[float] = None
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
        if self.ring_width_um is not None and self.ring_width_um <= 0:
            raise ValueError("ring width must be > 0 um (or None for the Nyquist width)")

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

    # -- ring width and the Nyquist rule -------------------------------------------
    @property
    def nyquist_ring_width_um(self) -> float:
        """The Nyquist-Shannon ring width lam_min / (2 NA), exact.

        The unfolded lens phase phi(rho) = -(2 pi / lam)(sqrt(rho^2 + F^2) - F)
        has its highest local spatial frequency at the rim, sin(theta_rim)/lam
        = NA / lam cycles per um; sampling it with one height per ring needs
        DELTA <= lam / (2 NA), tightest at lam_min. Equivalently the rim
        zone period lam/NA must span at least two rings."""
        return self.lam_min_um / (2.0 * self.NA)

    @property
    def nyquist_ring_width_grid_um(self) -> float:
        """The Nyquist width rounded DOWN to the RING_GRID_UM address grid
        (never above the limit); what ``ring_width_um = None`` selects."""
        w = np.floor(self.nyquist_ring_width_um / RING_GRID_UM + 1e-9) * RING_GRID_UM
        return float(max(w, RING_GRID_UM))

    @property
    def ring_width_is_auto(self) -> bool:
        return self.ring_width_um is None

    @property
    def delta_um(self) -> float:
        """The ring width in force: the given one, or the Nyquist grid width."""
        return self.nyquist_ring_width_grid_um if self.ring_width_um is None \
            else float(self.ring_width_um)

    @property
    def nyquist_ok(self) -> bool:
        """True when DELTA <= lam_min / (2 NA) (no aliasing of the rim zones)."""
        return self.delta_um <= self.nyquist_ring_width_um * (1.0 + 1e-9)

    @property
    def rings_per_rim_zone(self) -> float:
        """lam_min / (NA DELTA): rings across one rim zone at lam_min (>= 2 = Nyquist)."""
        return self.lam_min_um / (self.NA * self.delta_um)

    @property
    def aspect_ratio(self) -> float:
        """H / DELTA, the worst-case aspect ratio of one ring (paper S2-6:
        40:1 at NA 0.3; the Smooth step removes most of them)."""
        return self.h_max_um / self.delta_um

    @property
    def n_rings(self) -> int:
        return int(round(self.R_um / self.delta_um))

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
        return self.delta_um * self.NA

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
            "relief H = %.2f um in %d levels of %.3f um; rings DELTA = %.2f um (%s) -> N = %d "
            "(aspect ratio H/DELTA = %.1f)" % (self.h_max_um, self.n_levels, self.dh_um,
                                              self.delta_um,
                                              "auto: Nyquist lam_min/2NA = %.3f um rounded down "
                                              "to the %.2f um grid" % (self.nyquist_ring_width_um,
                                                                      RING_GRID_UM)
                                              if self.ring_width_is_auto else "given",
                                              self.n_rings, self.aspect_ratio),
            "Nyquist-Shannon ring width lam_min/(2 NA) = %.3f um: DELTA %.2f um is %s "
            "(%.2f rings per rim zone at %.0f nm, >= 2 required)"
            % (self.nyquist_ring_width_um, self.delta_um,
               "OK" if self.nyquist_ok else "TOO WIDE -> the rim zones alias",
               self.rings_per_rim_zone, 1000 * self.lam_min_um),
            "fold order p = (n-1)H/lam: %.1f at %.0f nm .. %.1f at %.0f nm"
            % (self.fold_waves(self.lam_min_um), 1000 * self.lam_min_um,
               self.fold_waves(self.lam_max_um), 1000 * self.lam_max_um),
            "rim fold period (n-1)H/NA = %.1f um = %.1f rings; rim ring factor S = %.3f "
            "at %.0f nm, %.3f at %.0f nm"
            % (self.rim_period_um(self.lam_min_um),
               self.rim_period_um(self.lam_min_um) / self.delta_um,
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
        d.update(NA=self.NA, F_um=self.F_um, F_number=self.F_number, n_rings=self.n_rings,
                 delta_um=self.delta_um, ring_width_auto=self.ring_width_is_auto,
                 nyquist_ring_width_um=self.nyquist_ring_width_um, nyquist_ok=self.nyquist_ok,
                 aspect_ratio=self.aspect_ratio)
        return d


def reference_spec(name: str) -> "LensSpec":
    """The paper's reference lens `name` as a LensSpec (NA 0.1, 400-1100 nm,
    AZ4562, 2 um rings, 78 nm levels)."""
    D_mm, H_um = REFERENCE_POINTS[name][:2]
    return LensSpec(D_mm, 0.40, 1.10, h_max_um=H_um, dh_um=0.078, ring_width_um=2.0,
                    na=0.1, name=name)


# ---------------------------------------------------------------------------
@dataclass
class Ceiling:
    """The two upper bounds of the continuous-band max J_w(F) at one spec."""
    spec: LensSpec
    analytic: float
    numeric: Optional[float] = None
    n_rho: int = 0
    n_wavelengths: int = 0

    # -- the semi-analytic chain of the supplementary, S1-6 --------------------------
    @staticmethod
    def constructive_path_um(spec: LensSpec) -> float:
        """Eq. S16: two rings can be made fully coherent over the band iff
        their path difference to the focus satisfies |r1 - r2| <= (n - 1) H
        (n = n_max, dispersion ignored)."""
        return (spec.n_max - 1.0) * spec.h_max_um

    @staticmethod
    def x_of(spec: LensSpec) -> float:
        """The paper's reduced variable x = 2 (n_max - 1) H f / R^2 (Eq. S17).
        With the paraxial NA = R / f it is 4 (n_max - 1) H / (D NA), the
        variable of Eq. 7 / S19; the two differ by sqrt(1 + (R/f)^2) - 1
        (1.4 % at F/3)."""
        return 2.0 * (spec.n_max - 1.0) * spec.h_max_um * spec.F_um / spec.R_um ** 2

    @classmethod
    def analytic_of(cls, spec: LensSpec) -> float:
        """Paper Eq. S17 verbatim (paraxial, no dispersion, constructive region
        only):

            max J_w(F) = 4 (n-1) H f / R^2 - 4 (n-1)^2 H^2 f^2 / R^4   for R >= sqrt(2 (n-1) H f)
                       = 1                                          otherwise

        i.e. max J = 1 - (1 - x)^2 with x = 2 (n-1) H f / R^2; the threshold
        R = sqrt(2 (n-1) H f) is x = 1 (Eq. S18, the hyperbolic-phase limit)."""
        x = cls.x_of(spec)
        return 1.0 if x >= 1.0 else 1.0 - (1.0 - x) ** 2

    @classmethod
    def analytic_pair_map(cls, spec: LensSpec, n_rho: int = 256
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """The Eq. S16 map: 1 where |r_i - r_j| <= (n_max - 1) H, 0 elsewhere,
        on the rho/R grid (the analytic counterpart of Fig. 1b/c)."""
        R, F = spec.R_um, spec.F_um
        rho = (np.arange(n_rho) + 0.5) * R / n_rho
        r = np.sqrt(rho ** 2 + F ** 2)
        B = (np.abs(r[:, None] - r[None, :]) <= cls.constructive_path_um(spec)).astype(float)
        return rho / R, B

    @classmethod
    def analytic_from_map(cls, spec: LensSpec, n_rho: int = 2048) -> float:
        """Eq. S15 restricted to the Eq. S16 region, integrated numerically
        with the paraxial aperture weights: reproduces Eq. S17 to < 1 %
        (the exact-geometry version of the same approximation)."""
        R, F = spec.R_um, spec.F_um
        rho_n, B = cls.analytic_pair_map(spec, n_rho)
        rho = rho_n * R
        r = np.sqrt(rho ** 2 + F ** 2)
        w = (2.0 * F / R ** 2) * rho * (R / n_rho) / r
        w = w / w.sum()
        return float(w @ B @ w)

    @staticmethod
    def d_max_hyperbolic_mm(spec: LensSpec) -> float:
        """Eq. S18: the largest diameter at this H / NA / n_max for which the
        ceiling is 1 (phase hyperbolic over the whole band):
        D_max = 4 (n_max - 1) H / NA."""
        return float(4.0 * (spec.n_max - 1.0) * spec.h_max_um / spec.NA / 1000.0)

    @staticmethod
    def h_hyperbolic_um(spec: LensSpec) -> float:
        """The relief height that makes the ceiling 1 at this D / NA (Eq. S18
        solved for H): H = D NA / (4 (n_max - 1))."""
        return float(spec.D_um * spec.NA / (4.0 * (spec.n_max - 1.0)))

    @staticmethod
    def h_required_um(spec: LensSpec, target_j: float) -> float:
        """Eq. S19 solved for H: the height that makes the analytic ceiling
        equal target_j at this D / NA, H = x_t D NA / (4 (n_max - 1)),
        x_t = 1 - sqrt(1 - target_j)."""
        xt = 1.0 - np.sqrt(max(0.0, 1.0 - min(target_j, 1.0)))
        return float(xt * spec.D_um * spec.NA / (4.0 * (spec.n_max - 1.0)))

    @staticmethod
    def d_max_mm(spec: LensSpec, target_j: float) -> float:
        """Eq. S19 / Eq. 7: the largest diameter whose analytic ceiling is
        target_j, D_max = 4 (n_max - 1) H / ((1 - sqrt(1 - target_j)) NA)."""
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
    """Verdict of a spec: which of the paper's lenses it is comparable to,
    and whether it reaches the requested class / absolute target.

    Attributes
    ----------
    spec, ceiling      the spec and its ceiling (numeric when computed)
    target_class       the paper lens the spec must at least match
                       ("S3" = the fabricated cm-scale achromat). The
                       comparison is ceiling-to-ceiling ON THE SAME SCALE
                       (both alias-free, same resolution), so the paper's
                       sparse-sampling inflation cancels.
    ref_ceiling        the target class's ceiling at the same resolution
                       (``Ceiling.compute(reference_spec(...))``); when
                       None the table value (128 x 256) is used.
    class_tolerance    ratio >= class_tolerance counts as reaching the
                       class. Default 0.75: the paper's second fabricated
                       design (S2-6: 1 cm, NA 0.3, H 28 um, 450-680 nm),
                       which it calls "relatively good", sits at 0.76 x S3
                       on our scale; the bound's resolution noise is ~2 %.
    target_j           optional requirement on the continuous-band J. By
                       paper Eq. 5 (J ~ <Eff>/w_max^2, w ~ 1) this is the
                       required MEAN FOCUSING EFFICIENCY over the band as
                       a fraction (0.03 = 3 %); the GUI / CLI take it in
                       percent. Only then are H_required and D_max for
                       that number reported. The ceiling is DERIVED from
                       the spec; the requirement is the user's.
    achievable_fraction  what a continuous-band design reaches of its
                       ceiling on this toolchain (measured 45-60 %).
    """
    spec: LensSpec
    ceiling: Ceiling
    target_class: Optional[str] = None
    ref_ceiling: Optional[Ceiling] = None
    class_tolerance: float = 0.75
    target_j: Optional[float] = None
    achievable_fraction: float = 0.55
    show_references: bool = False
    #: use the paper's threshold (max J >= 0.2 on its scale = the S3 ceiling on
    #: ours) as the requirement when no class and no efficiency are given
    paper_threshold: bool = True

    @property
    def effective_class(self) -> Optional[str]:
        """The reference lens the ceiling is compared with: the explicit
        class, else the paper's threshold lens when enabled, else None."""
        if self.target_class is not None:
            return self.target_class
        if self.paper_threshold and self.target_j is None:
            return PAPER_THRESHOLD_CLASS
        return None

    def __post_init__(self) -> None:
        if self.target_class is not None and self.target_class not in REFERENCE_POINTS:
            raise ValueError("target_class must be one of %s or None" % (REFERENCE_ORDER,))

    @property
    def has_target(self) -> bool:
        """Something to judge against: a class, an efficiency, or the paper's threshold."""
        return self.effective_class is not None or self.target_j is not None

    @property
    def bound(self) -> float:
        return self.ceiling.numeric if self.ceiling.numeric is not None else self.ceiling.analytic

    @property
    def ref_bound(self) -> Optional[float]:
        cls = self.effective_class
        if cls is None:
            return None
        if self.ref_ceiling is not None:
            return self.ref_ceiling.numeric if self.ref_ceiling.numeric is not None \
                else self.ref_ceiling.analytic
        return REFERENCE_POINTS[cls][3]

    @property
    def class_ratio(self) -> Optional[float]:
        """ceiling / ceiling of the target class (same scale); None without a class."""
        rb = self.ref_bound
        return None if rb is None else self.bound / max(rb, 1e-9)

    @property
    def reaches_class(self) -> Optional[bool]:
        r = self.class_ratio
        return None if r is None else r >= self.class_tolerance

    @property
    def expected_j(self) -> float:
        return self.achievable_fraction * self.bound

    @property
    def reaches_target_j(self) -> Optional[bool]:
        return None if self.target_j is None else self.expected_j >= self.target_j

    @property
    def feasible(self) -> Optional[bool]:
        """Nyquist satisfied and every target that was set reached; None
        when no target was set (the check is then informational only)."""
        if not self.has_target:
            return None
        ok = self.spec.nyquist_ok
        if self.effective_class is not None:
            ok = ok and bool(self.reaches_class)
        if self.target_j is not None:
            ok = ok and bool(self.reaches_target_j)
        return ok

    @property
    def verdict(self) -> str:
        f = self.feasible
        return "no requirement (informational)" if f is None else ("FEASIBLE" if f else "NOT FEASIBLE")

    def bracket(self) -> Tuple[Optional[str], Optional[str]]:
        """(better, worse) reference lenses around this spec's ceiling on
        our scale: the spec performs between them. (None, 'S1') = beyond
        S1; ('S5', None) = below S5."""
        b = self.bound
        better: Optional[str] = None
        for name in REFERENCE_ORDER:                    # descending ceilings
            if REFERENCE_POINTS[name][3] <= b:
                return better, name
            better = name
        return better, None

    def nearest_reference(self) -> str:
        b = self.bound
        best = min(REFERENCE_POINTS.items(), key=lambda kv: abs(np.log(max(kv[1][3], 1e-6) / max(b, 1e-6))))
        return best[0]

    def class_label(self) -> str:
        hi, lo = self.bracket()
        if hi is None and lo is not None:
            return "beyond %s" % lo
        if lo is None and hi is not None:
            return "below %s" % hi
        if hi == lo:
            return str(hi)
        return "between %s and %s" % (hi, lo)

    def lines(self) -> List[str]:
        s, c = self.spec, self.ceiling
        rows = ["ceiling max J_w(F): semi-analytic Eq. S17 %.4f (x = 2 (n-1) H f / R^2 = %.4f)%s"
                % (c.analytic, Ceiling.x_of(s),
                   ("; numeric alias-free Eq. S15 %.4f (%d x %d)"
                    % (c.numeric, c.n_rho, c.n_wavelengths)) if c.numeric is not None else "")]
        rows.append("Eq. S16 constructive band |r1 - r2| <= (n_max - 1) H = %.1f um of path; "
                    "Eq. S18 hyperbolic-phase limit: D_max(J = 1) = %.2f mm at this H, or "
                    "H = %.1f um at this D" % (Ceiling.constructive_path_um(s),
                                                Ceiling.d_max_hyperbolic_mm(s),
                                                Ceiling.h_hyperbolic_um(s)))
        rows.append("expected continuous-band design J ~ %.3f (%.0f %% of the ceiling, measured "
                    "on this toolchain) = expected MEAN FOCUSING EFFICIENCY over the band ~ %.1f %% "
                    "(paper Eq. 5, J ~ <Eff> / w_max^2 with w ~ 1 for near-diffraction-limited spots; "
                    "the ceiling itself: at best %.1f %%)"
                    % (self.expected_j, 100 * self.achievable_fraction, 100 * self.expected_j,
                       100 * self.bound))
        rows.append("ring width: DELTA %.2f um vs Nyquist %.3f um -> %s; aspect ratio H/DELTA %.1f%s"
                    % (s.delta_um, s.nyquist_ring_width_um,
                       "OK" if s.nyquist_ok else "NOT FEASIBLE as sampled (rim zones alias): "
                       "set DELTA <= %.2f um" % s.nyquist_ring_width_grid_um,
                       s.aspect_ratio,
                       " (high: rely on the Smooth step, paper S2-6 had 40:1 at NA 0.3)"
                       if s.aspect_ratio > 10 else ""))
        if self.target_j is not None:
            rows.append("REQUIREMENT: mean focusing efficiency >= %.1f %% (J >= %.3f) -> %s "
                        "(expected %.1f %%)"
                        % (100 * self.target_j, self.target_j,
                           "OK" if self.reaches_target_j else "NOT REACHED", 100 * self.expected_j))
            h_req_t = Ceiling.h_required_um(s, self.target_j / self.achievable_fraction)
            d_max_t = Ceiling.d_max_mm(s, self.target_j / self.achievable_fraction)
            rows.append("    Eq. S19 for that J at this D and NA: H >= %.1f um; at this H, "
                        "D <= %.1f mm (%.2f inch)" % (h_req_t, d_max_t, d_max_t / INCH_MM))
        if self.target_class is not None or self.show_references:
            hi, lo = self.bracket()
            rows.append("paper references (NA 0.1, 400-1100 nm, AZ4562) on our scale: this spec is %s"
                        % self.class_label())
            for name in (hi, lo):
                if name is not None:
                    rows.append("    %s: %s; alias-free ceiling %.3f"
                                % (name, REFERENCE_NOTES[name], REFERENCE_POINTS[name][3]))
        cls = self.effective_class
        if cls is not None:
            rb, cr = self.ref_bound, self.class_ratio
            assert rb is not None and cr is not None
            if self.target_class is None:
                rows.append("REQUIREMENT (default, from the paper, S2-6): max J_w(F) >= %.1f on the "
                            "paper's scale = its S3 lens = alias-free ceiling >= %.3f%s; this spec: "
                            "%.3f, ratio %.2f -> %s"
                            % (PAPER_THRESHOLD_J_PAPER_SCALE, rb,
                               " (same resolution)" if self.ref_ceiling is not None else " (table)",
                               self.bound, cr,
                               "MEETS the paper's criterion" if self.reaches_class else
                               "BELOW the paper's criterion (tolerance %.2f)" % self.class_tolerance))
            else:
                rows.append("REQUIREMENT: class %s: ceiling ratio %.2f (%s %.3f vs %s %.3f%s) -> %s"
                            % (cls, cr, s.name, self.bound, cls, rb,
                               ", same resolution" if self.ref_ceiling is not None else ", table value",
                               "REACHES the class" if self.reaches_class else
                               "BELOW the class (tolerance %.2f)" % self.class_tolerance))
            j_ref = Ceiling.analytic_of(reference_spec(cls))
            h_req = Ceiling.h_required_um(s, j_ref)
            d_max = Ceiling.d_max_mm(s, j_ref)
            rows.append("    to meet it at this D and NA: H >= %.1f um (Eq. S19, analytic map); at this "
                        "H the diameter could be up to %.1f mm (%.2f inch)"
                        % (h_req, d_max, d_max / INCH_MM))
            rows.append("    scales: the paper's Fig. 1d numbers (S3 0.21) come from sparse band "
                        "sampling; our alias-free ceilings (S3 0.068) are 1.3-3.7x lower but rank "
                        "the lenses identically")
        rows.append("VERDICT: %s" % self.verdict)
        rows.append("note: comb objectives (N lines, encircled energy) are on their own scale "
                    "and are not bounded by the continuous-band ceiling (run 3: 0.14 on 14 "
                    "lines at the S3 point)")
        return rows
