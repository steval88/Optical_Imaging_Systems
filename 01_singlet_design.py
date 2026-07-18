"""
01_singlet_design.py
--------------------
PROJECT 1 of your progression: build an f/4 N-BK7 singlet entirely from code.

What this teaches about the ZOS-API object model:
  - SystemData   : aperture, wavelengths, fields  (System Explorer in the GUI)
  - LDE          : the Lens Data Editor (surfaces)
  - Solves       : making a radius variable, setting a marginal-ray-height solve
  - Tools        : QuickFocus
  - MFE.GetOperandValue : the single most useful trick in the whole API --
                   evaluate ANY merit-function operand (EFFL, RSRE, ...) without
                   building a merit function.

Run:  python 01_singlet_design.py
"""

from zos_connection import PythonStandaloneApplication

zos = PythonStandaloneApplication()
ZOSAPI = zos.ZOSAPI
TheSystem = zos.TheSystem

# ---------------------------------------------------------------- new system
TheSystem.New(False)  # False = do not save current system first

# ------------------------------------------------------- system explorer data
# Aperture: entrance pupil diameter 25 mm  ->  with EFL 100 mm this is f/4
SysData = TheSystem.SystemData
SysData.Aperture.ApertureType = (
    ZOSAPI.SystemData.ZemaxApertureType.EntrancePupilDiameter
)
SysData.Aperture.ApertureValue = 25.0

# Wavelengths: visible d, F, C (use the built-in preset)
SysData.Wavelengths.SelectWavelengthPreset(
    ZOSAPI.SystemData.WavelengthPreset.FdC_Visible
)

# Fields: on-axis, 3.5 deg, 5 deg
SysData.Fields.SetFieldType(ZOSAPI.SystemData.FieldType.Angle)
SysData.Fields.GetField(1).Y = 0.0
SysData.Fields.AddField(0.0, 3.5, 1.0)
SysData.Fields.AddField(0.0, 5.0, 1.0)

# -------------------------------------------------------------- lens surfaces
# A new system has: OBJ (0), STO (1), IMA (2). Insert one surface after the
# stop so surfaces 1-2 form the singlet.
LDE = TheSystem.LDE
LDE.InsertNewSurfaceAt(2)

s1 = LDE.GetSurfaceAt(1)  # front surface (also the stop)
s2 = LDE.GetSurfaceAt(2)  # rear surface
im = LDE.GetSurfaceAt(3)

s1.Radius = 60.0          # rough starting point; optimization will fix it
s1.Thickness = 5.0
s1.Material = "N-BK7"
s1.Comment = "singlet front"

s2.Radius = -300.0
s2.Comment = "singlet rear"
s2.Thickness = 95.0       # placeholder; QuickFocus will adjust

# Make both radii variable for optimization
s1.RadiusCell.MakeSolveVariable()
s2.RadiusCell.MakeSolveVariable()

# ------------------------------------------------- merit function + optimize
# Build a default RMS-spot merit function with the wizard, add an EFL target.
MFE = TheSystem.MFE
wizard = MFE.SEQOptimizationWizard
wizard.Data = 1              # 1 = Spot Radius criterion
wizard.OverallWeight = 1.0
wizard.Ring = 3              # Gaussian quadrature rings
wizard.Apply()

# Insert EFFL = 100 mm as the first operand (row 1)
op = MFE.InsertNewOperandAt(1)
op.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.EFFL)
op.Target = 100.0
op.Weight = 1.0

# Damped least squares, automatic cycles
LocalOpt = TheSystem.Tools.OpenLocalOptimization()
LocalOpt.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Automatic
LocalOpt.NumberOfCores = 8
print(f"Initial merit function: {LocalOpt.InitialMeritFunction:.6f}")
LocalOpt.RunAndWaitForCompletion()
print(f"Final merit function:   {LocalOpt.CurrentMeritFunction:.6f}")
LocalOpt.Close()

# Refocus precisely
qf = TheSystem.Tools.OpenQuickFocus()
qf.Criterion = ZOSAPI.Tools.General.QuickFocusCriterion.SpotSizeRadial
qf.UseCentroid = True
qf.RunAndWaitForCompletion()
qf.Close()

# ----------------------------------------------------------------- reporting
# GetOperandValue(type, srf, wave, Hx, Hy, Px, Py, Ex, Ey) evaluates one
# operand on the fly. Field coords go in Hx/Hy for the RSRE-type operands.
MOT = ZOSAPI.Editors.MFE.MeritOperandType

efl = MFE.GetOperandValue(MOT.EFFL, 0, 2, 0, 0, 0, 0, 0, 0)
print(f"\nEffective focal length: {efl:.3f} mm  (f/{efl/25.0:.2f})")

print("\nRMS spot radius vs field (polychromatic, centroid ref):")
for i, hy in enumerate([0.0, 0.7, 1.0]):
    # RSCE: RMS spot radius, centroid reference; wave=0 -> polychromatic
    rms = MFE.GetOperandValue(MOT.RSCE, 3, 0, 0, hy, 0, 0, 0, 0)
    print(f"  field {hy:>4.1f} (norm): {rms*1000.0:8.2f} um")

print("\nSurface data after optimization:")
for idx in range(1, LDE.NumberOfSurfaces - 1):
    s = LDE.GetSurfaceAt(idx)
    print(
        f"  surf {idx}: R = {s.Radius:10.3f}  t = {s.Thickness:8.3f}  "
        f"glass = {s.Material if s.Material else '(air)'}"
    )

# ---------------------------------------------------------------------- save
outfile = zos.samples_dir() + "\\API_singlet_f4.zos"
TheSystem.SaveAs(outfile)
print(f"\nSaved: {outfile}")

del zos
