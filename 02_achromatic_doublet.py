"""
02_achromatic_doublet.py
------------------------
PROJECT 2 of your progression: 100 mm EFL f/5 cemented achromat, designed and
optimized from code, including axial color check before/after.

New API concepts vs example 01:
  - Multiple glasses and a cemented interface
  - AXCL operand to quantify axial chromatic aberration
  - Hammer/global-ready structure: variables + wizard + local optimization
  - Substitute-glass optimization (lets the optimizer pick real catalog glasses)

Run:  python 02_achromatic_doublet.py
"""

from zos_connection import PythonStandaloneApplication

zos = PythonStandaloneApplication()
ZOSAPI = zos.ZOSAPI
TheSystem = zos.TheSystem
MOT = ZOSAPI.Editors.MFE.MeritOperandType

TheSystem.New(False)

# ---------------------------------------------------------------- system data
SysData = TheSystem.SystemData
SysData.Aperture.ApertureType = (
    ZOSAPI.SystemData.ZemaxApertureType.EntrancePupilDiameter
)
SysData.Aperture.ApertureValue = 20.0  # EFL 100 -> f/5
SysData.Wavelengths.SelectWavelengthPreset(
    ZOSAPI.SystemData.WavelengthPreset.FdC_Visible
)
SysData.Fields.SetFieldType(ZOSAPI.SystemData.FieldType.Angle)
SysData.Fields.GetField(1).Y = 0.0
SysData.Fields.AddField(0.0, 1.5, 1.0)  # small field: axial color dominates

# ------------------------------------------------------------------ surfaces
# Classic crown-in-front achromat starting point (Fraunhofer type):
#   surf 1: crown front (stop), surf 2: cemented interface, surf 3: flint rear
LDE = TheSystem.LDE
LDE.InsertNewSurfaceAt(2)
LDE.InsertNewSurfaceAt(3)

s1 = LDE.GetSurfaceAt(1)
s2 = LDE.GetSurfaceAt(2)
s3 = LDE.GetSurfaceAt(3)

s1.Radius, s1.Thickness, s1.Material = 62.0, 6.0, "N-BK7"
s2.Radius, s2.Thickness, s2.Material = -45.0, 3.0, "N-SF5"
s3.Radius, s3.Thickness = -130.0, 92.0

for s in (s1, s2, s3):
    s.RadiusCell.MakeSolveVariable()
s3.ThicknessCell.MakeSolveVariable()   # back focal distance

# ----------------------------------------------- baseline chromatic aberration
MFE = TheSystem.MFE
axcl_before = MFE.GetOperandValue(MOT.AXCL, 0, 0, 0, 0, 0, 0, 0, 0)

# ------------------------------------------------------------- merit function
wizard = MFE.SEQOptimizationWizard
wizard.Data = 1                      # RMS spot radius
wizard.OverallWeight = 1.0
wizard.Ring = 3
wizard.Apply()

op = MFE.InsertNewOperandAt(1)
op.ChangeType(MOT.EFFL)
op.Target, op.Weight = 100.0, 1.0

# ------------------------------------------------------- local optimization 1
LocalOpt = TheSystem.Tools.OpenLocalOptimization()
LocalOpt.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Automatic
LocalOpt.RunAndWaitForCompletion()
LocalOpt.Close()

# ----------------------------------- glass substitution (real catalog glasses)
# Flag both glasses as 'Substitute' then run the Hammer optimizer briefly.
# This is how you let Zemax explore the glass map from code.
sol1 = s1.MaterialCell.CreateSolveType(ZOSAPI.Editors.SolveType.MaterialSubstitute)
s1.MaterialCell.SetSolveData(sol1)
sol2 = s2.MaterialCell.CreateSolveType(ZOSAPI.Editors.SolveType.MaterialSubstitute)
s2.MaterialCell.SetSolveData(sol2)

Hammer = TheSystem.Tools.OpenHammerOptimization()
Hammer.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
import time
Hammer.Run()
time.sleep(30)          # let Hammer explore glass combinations for 30 s
Hammer.Cancel()
Hammer.WaitForCompletion()
Hammer.Close()

# ------------------------------------------------------- final local clean-up
LocalOpt = TheSystem.Tools.OpenLocalOptimization()
LocalOpt.RunAndWaitForCompletion()
mf_final = LocalOpt.CurrentMeritFunction
LocalOpt.Close()

# ----------------------------------------------------------------- reporting
axcl_after = MFE.GetOperandValue(MOT.AXCL, 0, 0, 0, 0, 0, 0, 0, 0)
efl = MFE.GetOperandValue(MOT.EFFL, 0, 2, 0, 0, 0, 0, 0, 0)
rms_axis = MFE.GetOperandValue(MOT.RSCE, 4, 0, 0, 0.0, 0, 0, 0, 0)

print(f"Final merit function : {mf_final:.6f}")
print(f"EFL                  : {efl:.3f} mm")
print(f"Axial color (AXCL)   : {axcl_before:.4f} mm  ->  {axcl_after:.4f} mm")
print(f"On-axis RMS spot     : {rms_axis*1000.0:.2f} um")
print("\nFinal prescription:")
for idx in range(1, LDE.NumberOfSurfaces - 1):
    s = LDE.GetSurfaceAt(idx)
    print(
        f"  surf {idx}: R = {s.Radius:10.3f}  t = {s.Thickness:8.3f}  "
        f"glass = {s.Material if s.Material else '(air)'}"
    )

outfile = zos.samples_dir() + "\\API_achromat_f5.zos"
TheSystem.SaveAs(outfile)
print(f"\nSaved: {outfile}")

del zos
