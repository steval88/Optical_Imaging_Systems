"""
04_lwir_thermal_sweep.py
------------------------
Bridge to PROJECTS 6-7 of your progression: build a simple LWIR germanium
singlet for an uncooled microbolometer and quantify its thermal defocus from
-40 C to +70 C. This demonstrates *why* athermalization is needed before you
attempt to fix it (with multi-config thermal pickups or material combos).

New API concepts:
  - IR system setup: wavelengths in um band 8-12, Ge material
  - System environment: temperature and pressure
  - Tools.MakeThermal-style manual sweep: change temperature, refocus, record
  - Quantifying focus shift vs temperature (the athermalization driver)

Physics you should see in the output:
  Germanium's dn/dT is ~ +4e-4 /K -- enormous. A Ge singlet in an aluminum
  housing defocuses by many depths-of-focus over the military temperature
  range. THAT number is the requirement your athermalization must fix.

Run:  python 04_lwir_thermal_sweep.py
"""

import numpy as np

from zos_connection import PythonStandaloneApplication

zos = PythonStandaloneApplication()
ZOSAPI = zos.ZOSAPI
TheSystem = zos.TheSystem
MOT = ZOSAPI.Editors.MFE.MeritOperandType

TheSystem.New(False)

# ---------------------------------------------------------------- system data
SysData = TheSystem.SystemData

# f/1.2, EFL 50 mm -> EPD ~ 41.7 mm : typical uncooled LWIR camera speed
SysData.Aperture.ApertureType = (
    ZOSAPI.SystemData.ZemaxApertureType.EntrancePupilDiameter
)
SysData.Aperture.ApertureValue = 41.7

# LWIR band: 8 / 10 / 12 um
waves = SysData.Wavelengths
waves.GetWavelength(1).Wavelength = 10.0
waves.AddWavelength(8.0, 1.0)
waves.AddWavelength(12.0, 1.0)
waves.GetWavelength(1).MakePrimary()

SysData.Fields.SetFieldType(ZOSAPI.SystemData.FieldType.Angle)
SysData.Fields.GetField(1).Y = 0.0
SysData.Fields.AddField(0.0, 4.4, 1.0)   # ~half-FOV for 640x512 @ 12um, 50mm

# Enable environment adjustments so temperature actually affects the model
SysData.Environment.AdjustIndexToEnvironment = True
SysData.Environment.Temperature = 20.0
SysData.Environment.Pressure = 1.0

# ------------------------------------------------------------------ Ge singlet
LDE = TheSystem.LDE
LDE.InsertNewSurfaceAt(2)
s1 = LDE.GetSurfaceAt(1)
s2 = LDE.GetSurfaceAt(2)

s1.Radius, s1.Thickness, s1.Material = 55.0, 6.0, "GERMANIUM"
s2.Radius = 75.0
s2.Thickness = 45.0
s1.RadiusCell.MakeSolveVariable()
s2.RadiusCell.MakeSolveVariable()
s2.ThicknessCell.MakeSolveVariable()

MFE = TheSystem.MFE
wizard = MFE.SEQOptimizationWizard
wizard.Data = 1
wizard.OverallWeight = 1.0
wizard.Ring = 3
wizard.Apply()
op = MFE.InsertNewOperandAt(1)
op.ChangeType(MOT.EFFL)
op.Target, op.Weight = 50.0, 1.0

LocalOpt = TheSystem.Tools.OpenLocalOptimization()
LocalOpt.RunAndWaitForCompletion()
LocalOpt.Close()

# Freeze the design: remove variables so the thermal sweep doesn't re-optimize
TheSystem.Tools.RemoveAllVariables()

efl = MFE.GetOperandValue(MOT.EFFL, 0, 2, 0, 0, 0, 0, 0, 0)
print(f"Nominal design @ 20 C: EFL = {efl:.3f} mm")

# --------------------------------------------------------------- depth of focus
# Diffraction DOF ~ +/- 2 * lambda * (f/#)^2
fnum = efl / SysData.Aperture.ApertureValue
dof = 2.0 * 10.0e-3 * fnum**2   # mm, at lambda = 10 um
print(f"f/{fnum:.2f}, diffraction depth of focus ~ +/-{dof*1000:.0f} um\n")

# ------------------------------------------------------------ temperature sweep
# At each temperature: (a) record on-axis RMS spot with FROZEN focus,
# (b) run QuickFocus to find where best focus moved to.
nominal_bfd = LDE.GetSurfaceAt(2).Thickness
temps = np.array([-40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 70.0])

print(f"{'T (C)':>7} | {'RMS spot frozen (um)':>21} | {'focus shift (um)':>17}")
print("-" * 55)

for T in temps:
    SysData.Environment.Temperature = float(T)

    # (a) image quality if focus is NOT adjusted
    LDE.GetSurfaceAt(2).Thickness = nominal_bfd
    rms_frozen = MFE.GetOperandValue(MOT.RSCE, 3, 0, 0, 0.0, 0, 0, 0, 0)

    # (b) where did best focus go?
    qf = TheSystem.Tools.OpenQuickFocus()
    qf.Criterion = ZOSAPI.Tools.General.QuickFocusCriterion.SpotSizeRadial
    qf.UseCentroid = True
    qf.RunAndWaitForCompletion()
    qf.Close()
    shift = (LDE.GetSurfaceAt(2).Thickness - nominal_bfd) * 1000.0  # um

    print(f"{T:>7.0f} | {rms_frozen*1000.0:>21.1f} | {shift:>+17.1f}")

# Restore nominal state
SysData.Environment.Temperature = 20.0
LDE.GetSurfaceAt(2).Thickness = nominal_bfd

print(
    "\nCompare the focus-shift column against the depth of focus above."
    "\nEverything beyond +/-DOF is the athermalization problem you solve in"
    "\nproject 7 (material combinations, housing CTE, or active focus)."
)

outfile = zos.samples_dir() + "\\API_lwir_ge_singlet.zos"
TheSystem.SaveAs(outfile)
print(f"\nSaved: {outfile}")

del zos
