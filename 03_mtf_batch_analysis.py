"""
03_mtf_batch_analysis.py
------------------------
Extracting analysis DATA from OpticStudio into numpy/matplotlib -- the thing
the ZOS-API is genuinely best at, and the foundation of every trade study
you'll do later (project 9 in your progression).

This script:
  1. Opens the built-in Double Gauss sample file
  2. Runs an FFT MTF analysis, pulls the curves into numpy, plots them
  3. Sweeps the system through refocus positions and plots MTF @ 30 lp/mm
     vs defocus -- a mini through-focus MTF study done entirely from code

New API concepts:
  - TheSystem.LoadFile
  - Analyses.New_FftMtf, IAS_ settings, GetResults, DataSeries
  - Converting .NET arrays to numpy (the standard reshape trick)
  - Driving a system parameter in a loop and re-running an analysis

Run:  python 03_mtf_batch_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # avoids Qt's OleInitialize/COM clash with pythonnet
import matplotlib.pyplot as plt

from zos_connection import PythonStandaloneApplication

zos = PythonStandaloneApplication()
ZOSAPI = zos.ZOSAPI
TheSystem = zos.TheSystem

# ------------------------------------------------------------ load sample file
sample = (
    zos.samples_dir()
    + "\\Sequential\\Objectives\\Double Gauss 28 degree field.zos"
)
TheSystem.LoadFile(sample, False)
print(f"Loaded: {sample}")

nsurf = TheSystem.LDE.NumberOfSurfaces
image_surf = TheSystem.LDE.GetSurfaceAt(nsurf - 1)
focus_surf = TheSystem.LDE.GetSurfaceAt(nsurf - 2)  # last airspace = focus

# ================================================================ PART 1: MTF
mtf = TheSystem.Analyses.New_FftMtf()
settings = mtf.GetSettings()
settings.MaximumFrequency = 100.0     # lp/mm
settings.SampleSize = ZOSAPI.Analysis.SampleSizes.S_256x256
mtf.ApplyAndWaitForCompletion()
results = mtf.GetResults()

plt.figure(figsize=(9, 5.5))
colors = plt.cm.viridis(np.linspace(0, 0.85, results.NumberOfDataSeries))

for i in range(results.NumberOfDataSeries):
    series = results.GetDataSeries(i)
    # .NET 1-D array -> numpy
    x = np.array(list(series.XData.Data))
    # .NET 2-D array (npts x 2: tangential, sagittal) -> numpy
    y = np.array(list(series.YData.Data)).reshape(len(x), series.NumSeries)
    plt.plot(x, y[:, 0], color=colors[i], label=f"{series.Description} (T)")
    plt.plot(x, y[:, 1], color=colors[i], linestyle="--",
             label=f"{series.Description} (S)")

mtf.Close()

plt.xlabel("Spatial frequency (lp/mm)")
plt.ylabel("MTF")
plt.title("Double Gauss 28 deg -- FFT MTF")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("mtf_curves.png", dpi=150)
print("Saved mtf_curves.png")

# ================================================= PART 2: through-focus sweep
# Manually sweep the last airspace and record tangential MTF @ 30 lp/mm on
# axis. (OpticStudio has a native through-focus MTF analysis; doing it by hand
# teaches the loop pattern you will reuse for thermal sweeps, tolerance sweeps,
# zoom positions, etc.)
nominal_t = focus_surf.Thickness
defocus_range = np.linspace(-0.15, 0.15, 21)   # mm
mtf30 = []

MOT = ZOSAPI.Editors.MFE.MeritOperandType
MFE = TheSystem.MFE

for dz in defocus_range:
    focus_surf.Thickness = nominal_t + dz
    # MTFT operand: samp, wave, field, frequency  (see operand docs)
    val = MFE.GetOperandValue(MOT.MTFT, 2, 0, 1, 30.0, 0, 0, 0, 0)
    mtf30.append(val)

focus_surf.Thickness = nominal_t  # restore

plt.figure(figsize=(8, 5))
plt.plot(defocus_range * 1000.0, mtf30, "o-")
plt.xlabel("Defocus (um)")
plt.ylabel("Tangential MTF @ 30 lp/mm (on-axis)")
plt.title("Through-focus MTF sweep (driven from Python)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("through_focus_mtf.png", dpi=150)
print("Saved through_focus_mtf.png")

best = defocus_range[int(np.argmax(mtf30))]
print(f"Best focus offset: {best*1000.0:+.1f} um from nominal")

del zos
