"""Parameter SLOT maps of the stock RCWA diffraction DLLs, transcribed
from the verbatim panel screenshots of 2026-09-02
(zemax_doe_primitives.md sec. 3): labels are DLL cosmetics, the slot
index is what the ZOS-API addresses. A run echoes the labels the DLL
reports next to these names; a mismatch means the map must be fixed."""
from __future__ import annotations

from typing import Dict

SRG_BLAZE: Dict[str, int] = {
    "period_um": 1, "max_order": 2, "unused": 3, "fill": 4, "alpha_deg": 5,
    "beta_deg": 6, "coat_top_um": 7, "coat_side_um": 8, "n_layer": 9,
    "use_coating_file": 10, "index_grate_r": 11, "index_grate_i": 12,
    "index_env_r": 13, "index_env_i": 14, "index_coat_r": 15, "index_coat_i": 16,
    "rotate_deg": 17, "interpolation": 18, "test_mode": 19, "only_orders": 20,
    "stochastic": 21, "coat_mode": 22, "nil_thick_um": 23,
}

SRG_STEP: Dict[str, int] = {
    "period_um": 1, "max_order": 2, "depth_um": 3, "n_steps": 4,
    "layers_per_step": 5, "alpha_deg": 6, "coat_top_um": 7, "coat_side_um": 8,
    "unused": 9, "use_coating_file": 10, "index_grate_r": 11, "index_grate_i": 12,
    "index_env_r": 13, "index_env_i": 14, "index_coat_r": 15, "index_coat_i": 16,
    "rotate_deg": 17, "interpolation": 18, "test_mode": 19, "only_orders": 20,
    "stochastic": 21, "nil_thick_um": 22,
}

SLOT_MAPS: Dict[str, Dict[str, int]] = {
    "srg_blaze_RCWA.dll": SRG_BLAZE,
    "srg_step_RCWA.dll": SRG_STEP,
    "srg_step3_RCWA.dll": SRG_STEP,          # identical panel, one label differs
}


def slots(dll: str, **values: float) -> Dict[int, float]:
    """{slot: value} for the named parameters of a DLL."""
    m = SLOT_MAPS[dll]
    out: Dict[int, float] = {}
    for k, v in values.items():
        if k not in m:
            raise KeyError("%s has no parameter %r (known: %s)" % (dll, k, sorted(m)))
        out[m[k]] = float(v)
    return out
