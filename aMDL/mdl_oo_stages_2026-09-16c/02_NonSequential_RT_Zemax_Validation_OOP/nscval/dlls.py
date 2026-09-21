"""Parameter slots of the stock RCWA diffraction DLLs (srg family).

The ZOS-API addresses the Diffraction-tab parameters by INDEX, COUNTED
FROM 0; the DLL only decorates the indices with labels. History of this
file, because it matters: the transcription of 2026-09-02 put the period
first; a 1-based listing on 2026-09-16 seemed to show "Max Order" first
and the period absent, and the map was "corrected" the wrong way. The
diag of the same day (a Split-by-Table taking its values at indices 0
and 1; every DLL listing ending in a blank entry) established the
0-based counting: index 0 is the parameter the 1-based listing never
showed -- the PERIOD for the srg DLLs (to be confirmed by the next
probe, which prints index 0), "X Period" for Diff2DSample. Leaving it
unset (= 0) is why every DLL split delivered zero power.

A run resolves every canonical key against the labels the DLL reports
at run time (`resolve`, returning 0-based indices); the verbatim label
lists here serve the mock and the documentation. A key whose pattern
matches no label stops the run with the full label list. The object's
own Lines/um (Diffraction Grating Par 10) must AGREE with the DLL's
period: OpticStudio takes the order directions from the object, the DLL
takes the efficiencies from its own period.

    key               pattern (case-insensitive regex on the label)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

# --- verbatim label lists (slot 1 first) --------------------------------------
# srg_blaze_RCWA.dll — probe 2026-09-16, OpticStudio 2024 R1, 23 parameters,
# list index = parameter index (0-based). Index 0 was not in the 1-based
# listing; "Period (um)" per the 2026-09-02 transcription, confirmed by
# the next probe.
LABELS_BLAZE: List[str] = [
    "Period (um)", "Max Order", "Unused", "Fill factor", "Alpha (deg)", "Beta (deg)",
    "Coat Thick Top(um)", "Coat Thick Side(um)", "# Layer", "Use Coating File",
    "Index Grate (R)", "Index Grate (I)", "Index Env (R)", "Index Env (I)",
    "Index Coat (R)", "Index Coat (I)", "Rotate Grating", "Interpolation",
    "Test Mode", "Only these orders", "Stochastic mode", "Coat mode", "NIL Thick",
]
# srg_step_RCWA.dll — probe 2026-09-16, 22 slots (verbatim, incl. the DLL's
# own "Only theseorders" spelling)
LABELS_STEP: List[str] = [
    "Period (um)", "Max Order", "Depth (um)", "Number of Steps", "Layers per step", "Alpha (deg)",
    "Coat Thick Top(um)", "Coat Thick Side(um)", "Unused", "Use Coating File",
    "Index Grate (R)", "Index Grate (I)", "Index Env (R)", "Index Env (I)",
    "Index Coat (R)", "Index Coat (I)", "Rotate Grating", "Interpolation",
    "Test Mode", "Only theseorders", "Stochastic mode", "NIL Thick",
]
# srg_step3_RCWA.dll — probe 2026-09-16, 22 parameters: as step, index 4 = "Number of Layers"
LABELS_STEP3: List[str] = list(LABELS_STEP)
LABELS_STEP3[4] = "Number of Layers"
# srg_user_defined_RCWA.dll — probe 2026-09-16, 9 slots (profile from
# user_grating_data_<File number>.txt in DLL\Diffractive)
LABELS_USER: List[str] = [
    "Period (um)", "Max Order", "File number", "Use Coating File", "Rotate Grating", "Interpolation",
    "Test Mode", "Only these orders", "Stochastic mode",
]

LABELS: Dict[str, List[str]] = {
    "srg_blaze_RCWA.dll": LABELS_BLAZE,
    "srg_step_RCWA.dll": LABELS_STEP,
    "srg_step3_RCWA.dll": LABELS_STEP3,
    "srg_user_defined_RCWA.dll": LABELS_USER,
}

# --- canonical keys -> label patterns ------------------------------------------
PATTERNS: Dict[str, str] = {
    "period_um": r"period",
    "max_order": r"^max\s*order",
    "fill": r"^fill",
    "alpha_deg": r"^alpha",
    "beta_deg": r"^beta",
    "depth_um": r"depth",
    "n_steps": r"^(#|num|number\s+of)\s*steps?",
    "layers_per_step": r"layers?\s*(per|/)\s*step",
    "n_layers": r"^number\s+of\s+layers",
    "file_number": r"^file\s*number",
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
    "only_orders": r"^only\s*these\s*orders",     # also matches "Only theseorders"
    "stochastic": r"^stochastic",
    "coat_mode": r"^coat\s*mode",
    "nil_thick_um": r"^nil\s*thick",
}

# keys that are NOT DLL parameters (none at present; the period IS index 0
# of the srg DLLs and must also be set on the object as Lines/um)
OBJECT_KEYS: Dict[str, str] = {}


def resolve(labels: Sequence[str], dll: str = "") -> Dict[str, int]:
    """{key: index} (0-based) for every canonical key whose pattern matches
    one of `labels` (list index = parameter index). Ambiguous keys raise."""
    out: Dict[str, int] = {}
    for key, pat in PATTERNS.items():
        rx = re.compile(pat, re.IGNORECASE)
        hits = [i for i, lab in enumerate(labels) if rx.search(lab.strip())]
        if len(hits) > 1:
            raise SystemExit("DLL %s: key %r matches slots %s (labels %s)"
                             % (dll, key, hits, [labels[h] for h in hits]))
        if hits:
            out[key] = hits[0]
    return out


def slots(dll: str, labels: Optional[Sequence[str]] = None, **values: float) -> Dict[int, float]:
    """{index: value} (0-based) for the named parameters of `dll`, resolved
    against the live `labels` (from ``DiffractionTab.names()``) or, when
    None, against the verbatim list on file. Unmatched keys stop the run."""
    labs = list(labels) if labels else LABELS[dll]
    m = resolve(labs, dll)
    out: Dict[int, float] = {}
    for k, v in values.items():
        if k in OBJECT_KEYS:
            raise KeyError("%r is not a DLL slot: it is %s" % (k, OBJECT_KEYS[k]))
        if k not in m:
            raise SystemExit("DLL %s reports no parameter for %r; its labels are: %s"
                             % (dll, k, " | ".join("[%d]%s" % (i, s) for i, s in enumerate(labs))))
        out[m[k]] = float(v)
    return out


def slot_of(dll: str, key: str, labels: Optional[Sequence[str]] = None) -> int:
    """The 0-based parameter index of one key (see `slots`)."""
    return next(iter(slots(dll, labels, **{key: 0.0})))
