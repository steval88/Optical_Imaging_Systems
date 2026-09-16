"""Parameter slots of the stock RCWA diffraction DLLs (srg family).

The ZOS-API addresses the Diffraction-tab parameters by SLOT INDEX
(1-based); the DLL only decorates the slots with labels. The probe of
2026-09-16 (OpticStudio 2024 R1) listed the labels of srg_blaze_RCWA.dll
verbatim and showed that the transcription of 2026-09-02 was shifted by
one: THE GRATING PERIOD IS NOT A DLL SLOT. The srg DLLs take the period
from the object's own "Lines/µm" parameter (Diffraction Grating: Par 10),
and slot 1 is already "Max Order".

To be immune to such shifts, a run resolves every canonical key against
the labels the DLL reports at run time (`resolve`), and the verbatim
label lists here serve the mock and the documentation. A key whose
pattern matches no label stops the run with the full label list.

    key               pattern (case-insensitive regex on the label)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

# --- verbatim label lists (slot 1 first) --------------------------------------
# srg_blaze_RCWA.dll — probe 2026-09-16, OpticStudio 2024 R1, 23 slots
LABELS_BLAZE: List[str] = [
    "Max Order", "Unused", "Fill factor", "Alpha (deg)", "Beta (deg)",
    "Coat Thick Top(um)", "Coat Thick Side(um)", "# Layer", "Use Coating File",
    "Index Grate (R)", "Index Grate (I)", "Index Env (R)", "Index Env (I)",
    "Index Coat (R)", "Index Coat (I)", "Rotate Grating", "Interpolation",
    "Test Mode", "Only these orders", "Stochastic mode", "Coat mode", "NIL Thick", "",
]
# srg_step_RCWA.dll — PROVISIONAL (same family layout, depth / steps / layers
# in place of fill / alpha / beta); replaced by the probe listing when seen.
LABELS_STEP: List[str] = [
    "Max Order", "Depth (um)", "# Steps", "Layers per step", "Alpha (deg)",
    "Coat Thick Top(um)", "Coat Thick Side(um)", "# Layer", "Use Coating File",
    "Index Grate (R)", "Index Grate (I)", "Index Env (R)", "Index Env (I)",
    "Index Coat (R)", "Index Coat (I)", "Rotate Grating", "Interpolation",
    "Test Mode", "Only these orders", "Stochastic mode", "Coat mode", "NIL Thick", "",
]

LABELS: Dict[str, List[str]] = {
    "srg_blaze_RCWA.dll": LABELS_BLAZE,
    "srg_step_RCWA.dll": LABELS_STEP,
    "srg_step3_RCWA.dll": LABELS_STEP,
}

# --- canonical keys -> label patterns ------------------------------------------
PATTERNS: Dict[str, str] = {
    "max_order": r"^max\s*order",
    "fill": r"^fill",
    "alpha_deg": r"^alpha",
    "beta_deg": r"^beta",
    "depth_um": r"depth",
    "n_steps": r"^(#|num|number\s+of)\s*steps?",
    "layers_per_step": r"layers?\s*(per|/)\s*step",
    "coat_top_um": r"^coat\s*thick\s*top",
    "coat_side_um": r"^coat\s*thick\s*side",
    "n_layer": r"^#\s*layers?$",
    "use_coating_file": r"^use\s*coating\s*file",
    "index_grate_r": r"^index\s*grat\w*\s*\(r\)",
    "index_grate_i": r"^index\s*grat\w*\s*\(i\)",
    "index_env_r": r"^index\s*env\w*\s*\(r\)",
    "index_env_i": r"^index\s*env\w*\s*\(i\)",
    "index_coat_r": r"^index\s*coat\w*\s*\(r\)",
    "index_coat_i": r"^index\s*coat\w*\s*\(i\)",
    "rotate_deg": r"^rotate",
    "interpolation": r"^interpolation",
    "test_mode": r"^test\s*mode",
    "only_orders": r"^only\s*these\s*orders",
    "stochastic": r"^stochastic",
    "coat_mode": r"^coat\s*mode",
    "nil_thick_um": r"^nil\s*thick",
}

# keys that are NOT DLL slots but object parameters (Diffraction Grating)
OBJECT_KEYS: Dict[str, str] = {
    "period_um": "the object's Lines/um parameter (Par 10 = 1 / period_um)",
}


def resolve(labels: Sequence[str], dll: str = "") -> Dict[str, int]:
    """{key: slot} for every canonical key whose pattern matches one of
    `labels` (slot 1 first). Ambiguous keys (two labels match) raise."""
    out: Dict[str, int] = {}
    for key, pat in PATTERNS.items():
        rx = re.compile(pat, re.IGNORECASE)
        hits = [i + 1 for i, lab in enumerate(labels) if rx.search(lab.strip())]
        if len(hits) > 1:
            raise SystemExit("DLL %s: key %r matches slots %s (labels %s)"
                             % (dll, key, hits, [labels[h - 1] for h in hits]))
        if hits:
            out[key] = hits[0]
    return out


def slots(dll: str, labels: Optional[Sequence[str]] = None, **values: float) -> Dict[int, float]:
    """{slot: value} for the named parameters of `dll`, resolved against the
    live `labels` (from ``DiffractionTab.names()``) or, when None, against
    the verbatim list on file. Unknown / unmatched keys stop the run."""
    labs = list(labels) if labels else LABELS[dll]
    m = resolve(labs, dll)
    out: Dict[int, float] = {}
    for k, v in values.items():
        if k in OBJECT_KEYS:
            raise KeyError("%r is not a DLL slot: it is %s" % (k, OBJECT_KEYS[k]))
        if k not in m:
            raise SystemExit("DLL %s reports no parameter for %r; its labels are: %s"
                             % (dll, k, " | ".join("%d:%s" % (i + 1, s) for i, s in enumerate(labs))))
        out[m[k]] = float(v)
    return out


def slot_of(dll: str, key: str, labels: Optional[Sequence[str]] = None) -> int:
    """The slot index of one key (see `slots`)."""
    return next(iter(slots(dll, labels, **{key: 0.0})))
