"""
nscval -- STAGE 2c: NON-SEQUENTIAL OpticStudio validation of a designed
MDL (order-resolved efficiency, stray orders / halo, power budget).

Companion of ``02_Sequential_RT_Zemax_Validation_OOP`` (zval: the
sequential-mode closures rz / huy on the focal FIELD). This stage uses
OpticStudio's non-sequential ray-splitting framework, which the Ansys
knowledge base prescribes for diffractive elements: every diffractive
object is a two-layer model -- the OBJECT sets the direction of each
order (grating / phase-gradient law) and the DIFFRACTION DLL on its
face sets the ENERGY split among the orders (the srg_*_RCWA DLLs solve
the local microstructure rigorously). That is exactly the incoherent,
energy-flow question of a power budget; the focal PSF stays with the
Huygens hybrid of the sequential stage.

Modes (CLI: ../mdl_nsc_validation.py)

    probe   opens a non-sequential system, inserts a Diffraction Grating
            object with a diffraction DLL and PRINTS the ZOS-API member
            names of the object, its diffraction data and the ray-trace
            tool, plus the DLL's parameter names in slot order. Run this
            FIRST on the OpticStudio machine: the adapter in nsc.py
            resolves those names at run time and this is how they are
            confirmed (the POP settings of 2026-09-07 were learned the
            same way).
    null    the RCWA null test: Source Ellipse (collimated, 1 W) ->
            Diffraction Grating object (Lines/um = 1/Period) with
            srg_blaze_RCWA -> Detector Rectangle. One trace per order
            (Start = Stop = m), eta_m = detector flux / source power,
            against the scalar prediction sinc^2(m - p), p = (n-1)d/lam.
            Closes the NSC plumbing on a stock DLL.
    ladder  the thin-element-validity ladder: srg_step_RCWA equal-step
            staircases at the design's fold depth over the local periods
            of <run>/zone_table.npz, at every design wavelength;
            eta_RCWA(m) vs eta_TEA(m) of the same staircase (nscval.tea)
            -> the ratio table that tells where the scalar model breaks.

Later rungs (NSC_TRACK.md): the real per-zone profiles through
srg_user_defined_RCWA (profile-file format to be pinned with the probe),
the custom chirped diffraction DLL on a custom User Defined Object, and
the detector-plane halo / power-budget maps of the whole lens.

Module map
----------
    settings.py   every knob (NULL_SETTINGS, LADDER_SETTINGS, TRACE_SETTINGS)
    tea.py        scalar references: blaze sinc^2, staircase TEA orders
    nsc.py        NscSystem (NCE builder), DiffractionTab (name adapter),
                  NscTrace (ray trace + detector readout)
    base.py       NscAnalysis: run folder, output subfolder
                  <run>/nsc/<stamp>_<mode>, run_info.json, ZOS session
    probe.py, nulltest.py, ladder.py   one class per mode
"""
SCRIPT_VERSION = "2026-09-16.04"

__all__ = ["SCRIPT_VERSION"]
