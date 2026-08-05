"""
mdl_zemax_validation.py
=======================

Validate the designed MDL inside Ansys Zemax OpticStudio through the
ZOS-API, using the "MDL Rings" User-Defined Surface DLL
(us_mdl_rings.dll, zone-decomposition staircase model).

Prerequisites (on the Windows machine running OpticStudio)
----------------------------------------------------------
1. Copy  us_mdl_rings.dll  and  mdl_rings_1.txt  into
       {Documents}\\Zemax\\DLL\\Surfaces\\
   (mdl_rings_1.txt must sit NEXT TO the DLL -- the DLL loads
   "mdl_rings_<File #>.txt" from its own folder.)
2. Python 3.x + pythonnet, and zos_connection.py (the standalone
   connection helper already used by the other scripts in this project).

System layout (sequential)
--------------------------
    OBJ (inf)
    1   STOP  substrate front face, flat, material AZ4562-like model
              (n via Sellmeier-free 'MODEL' glass: nd/Vd fitted), 1.1 mm
    2   UDS   "MDL Rings"  rear face: staircase relief into air
    3   IMA   at the design BFD

The MDL was designed for a plane wave in AIR incident on the relief;
here the relief is traced resist->air, which is the same zone phase
(n-1)*h to first order. For exact correspondence with the design code
the substrate is included; the small plane-plate offset is absorbed by
refocusing (quick focus on the image plane).

Analyses performed
------------------
    * single-ray OPD sanity checks vs. the analytic staircase
    * Huygens PSF at the resonance wavelengths (0.60, 0.70, 1.05 um)
      and at two off-resonance wavelengths (0.50, 0.85 um)
    * through-focus Huygens PSF at 0.70 um
Results are written to text files next to this script.

NOTE ON SAMPLING: the lens has 7692 rings of 0.65 um over a 10 mm
aperture. Huygens/FFT PSF pupil sampling must resolve the rings --
use >= 8192x8192 ('S_8192x8192') where licensed; expect long runtimes.
Geometric analyses (spot diagrams) are meaningless for this surface;
only diffraction-based analyses are physical.
"""
import os
import sys

import clr  # pythonnet

from ZOS_API_Examples.zos_connection import PythonStandaloneApplication

# --- design selector: run as
#         python mdl_zemax_validation.py <design> [mode]
#     <design> in {na03, s3, s3comb};  [mode] in {zone (default), od}
#
# zone : us_mdl_rings.dll     (staircase / zone decomposition)
#        -> rays go STRAIGHT; only diffraction analyses (FFT/Huygens
#           PSF) are physical. Quantitative ground truth.
# od   : us_mdl_rings_od.dll  (order decomposition, Binary-2 style)
#        -> rays bend per the grating equation of ONE order; ray-based
#           chromatic analyses (longitudinal aberration, chromatic
#           focal shift, ray fans) are meaningful, one order per run.
#           fold_P / lam0_um below are the design fold parameters the
#           DLL needs to unfold the ring table; `orders` is the list
#           analyzed (each near-blazed at some band wavelength).
DESIGNS = {
    # NA-0.3 lens (m_final.npy / mdl_rings_1.txt): resonances 600/700/1050
    "na03": dict(epd_mm=10.0, bfd_mm=15.899, file_no=1,
                 wavelengths_um=[0.50, 0.60, 0.70, 0.85, 1.05],
                 primary_idx=3, fold_P=24, lam0_um=0.70,
                 orders=[16, 24, 28, 0]),
    # S3 reproduction (m_s3.npy / mdl_rings_2.txt): resonances 550/850
    "s3": dict(epd_mm=10.24, bfd_mm=50.94, file_no=2,
               wavelengths_um=[0.45, 0.55, 0.70, 0.85, 1.05],
               primary_idx=2, fold_P=17, lam0_um=0.55,
               orders=[11, 13, 17, 0]),
    # S3 comb-optimized (m_s3_comb.npy / mdl_rings_3.txt)
    "s3comb": dict(epd_mm=10.24, bfd_mm=50.94, file_no=3,
                   wavelengths_um=[0.45, 0.55, 0.70, 0.85, 1.05],
                   primary_idx=2, fold_P=15, lam0_um=0.60,
                   orders=[10, 12, 15, 0]),
}
import sys as _sys
DESIGN = DESIGNS[_sys.argv[1] if len(_sys.argv) > 1 else "s3"]
MODE = _sys.argv[2] if len(_sys.argv) > 2 else "zone"
assert MODE in ("zone", "od"), "mode must be 'zone' or 'od'"

WAVELENGTHS_UM = DESIGN["wavelengths_um"]
PRIMARY_IDX = DESIGN["primary_idx"]
EPD_MM = DESIGN["epd_mm"]
SUBSTRATE_MM = 1.1
BFD_GUESS_MM = DESIGN["bfd_mm"]     # design focal length (in air)
FILE_NO = DESIGN["file_no"]
FOLD_P = DESIGN["fold_P"]
LAM0_UM = DESIGN["lam0_um"]
ORDERS = DESIGN["orders"]
DLL_NAME = "us_mdl_rings.dll" if MODE == "zone" else "us_mdl_rings_od.dll"
OUT_DIR = os.path.join(os.path.expanduser("~"), "Documents",
                       "Zemax_MDL_Validation")


def main():
    zos = PythonStandaloneApplication()
    ZOSAPI = zos.ZOSAPI
    TheSystem = zos.TheSystem
    os.makedirs(OUT_DIR, exist_ok=True)

    TheSystem.New(False)
    TheSystem.MakeSequential()
    SysData = TheSystem.SystemData

    SysData.Aperture.ApertureType = \
        ZOSAPI.SystemData.ZemaxApertureType.EntrancePupilDiameter
    SysData.Aperture.ApertureValue = EPD_MM

    SysData.Wavelengths.RemoveWavelength(1)
    for i, w in enumerate(WAVELENGTHS_UM):
        SysData.Wavelengths.AddWavelength(w, 1.0)
    SysData.Wavelengths.GetWavelength(PRIMARY_IDX).MakePrimary()

    TheLDE = TheSystem.LDE

    # Surface 1: substrate front (STOP), model glass ~ AZ4562
    surf1 = TheLDE.GetSurfaceAt(1)
    surf1.Thickness = SUBSTRATE_MM
    # model glass: nd at 0.5876 um from the Cauchy fit n=1.594+0.01152/l^2
    #   nd = 1.6274, Abbe ~ (nd-1)/(nF-nC) with the same fit -> ~ 30.6
    surf1.MaterialCell.SetSolveData(
        TheLDE.GetSurfaceAt(1).MaterialCell.CreateSolveType(
            ZOSAPI.Editors.SolveType.MaterialModel))
    solve = surf1.MaterialCell.GetSolveData()._S_MaterialModel
    solve.IndexNd = 1.6274
    solve.AbbeVd = 30.6
    surf1.MaterialCell.SetSolveData(surf1.MaterialCell.GetSolveData())
    surf1.Comment = "AZ4562 substrate (model glass)"

    # Surface 2: the MDL relief -- User Defined Surface DLL
    surf2 = TheLDE.InsertNewSurfaceAt(2)
    uds_type = surf2.GetSurfaceTypeSettings(
        ZOSAPI.Editors.LDE.SurfaceType.UserDefined)
    uds_type.Filename = DLL_NAME
    surf2.ChangeType(uds_type)
    surf2.Thickness = BFD_GUESS_MM

    def set_par(num, value):
        col = getattr(ZOSAPI.Editors.LDE.SurfaceColumn, "Par%d" % num)
        surf2.GetSurfaceCell(col).DoubleValue = float(value)

    if MODE == "zone":
        surf2.Comment = "MDL Rings staircase (zone decomposition)"
        # Par 1..4: File #, Height scale, Z sign, Parax f
        set_par(1, FILE_NO)
        set_par(2, 1.0)
        # Z sign MUST be +1 in this trace direction (resist -> air): the
        # relief pokes into the exit space (z = +h), so taller rings add
        # resist path and the transmitted phase is +k(n-1)h as designed.
        # (-1 inverts the lens: diverging output, virtual focus at -F,
        # flat halo at the image plane.)
        set_par(3, 1.0)
        set_par(4, BFD_GUESS_MM)
    else:
        surf2.Comment = "MDL Rings order decomposition"
        # Par 1..6: File #, Order m, Design P, Lam0 um, Add OPL, Use eff
        set_par(1, FILE_NO)
        set_par(2, float(ORDERS[-1]))   # start at the design order
        set_par(3, float(FOLD_P))
        set_par(4, LAM0_UM)
        set_par(5, 1.0)
        set_par(6, 1.0)

    # ------------------------------------------------------------------
    # Single-ray trace sanity check at the primary wavelength
    # ------------------------------------------------------------------
    rt = TheSystem.Tools.OpenBatchRayTrace()
    norm = rt.CreateNormUnpol(8, ZOSAPI.Tools.RayTrace.RaysType.Real,
                              TheLDE.NumberOfSurfaces - 1)
    norm.ClearData()
    for py in (0.0, 0.2, 0.5, 0.9):
        norm.AddRay(PRIMARY_IDX, 0.0, 0.0, 0.0, py,
                    ZOSAPI.Tools.RayTrace.OPDMode.Current)
    rt.RunAndWaitForCompletion()
    ray_path = os.path.join(OUT_DIR, "ray_checks.txt")
    with open(ray_path, "w") as fh:
        norm.StartReadingResults()
        ok = True
        while ok:
            (ok, rn, err, vig, x, y, z, l, m, n,
             l2, m2, n2, opd, inten) = norm.ReadNextResult()
            if ok:
                fh.write("ray %d err=%d vig=%d  xyz=(%.6f, %.6f, %.6f)"
                         "  opd=%.6f\n" % (rn, err, vig, x, y, z, opd))
    rt.Close()
    print("ray checks -> %s" % ray_path)

    def typed_settings_g(analysis, probe_attr):
        st = analysis.GetSettings()
        if probe_attr and not hasattr(st, probe_attr):
            st = st.__implementation__
        return st

    if MODE == "od":
        # --------------------------------------------------------------
        # Ray-based chromatic analyses, one diffraction order per pass.
        # Longitudinal aberration / chromatic focal shift are now
        # meaningful because the OD surface bends rays.
        # --------------------------------------------------------------
        analyses = []
        for name, fname in (("FocalShiftDiagram", "chromatic_shift"),
                            ("LongitudinalAberration", "longitudinal"),
                            ("RayFan", "rayfan")):
            idm = getattr(ZOSAPI.Analysis.AnalysisIDM, name, None)
            if idm is not None:
                analyses.append((idm, fname))
            else:
                print("  (analysis %s not in this ZOS-API version)"
                      % name)

        for order in ORDERS:
            set_par(2, float(order))
            for idm, fname in analyses:
                an = TheSystem.Analyses.New_Analysis(idm)
                an.ApplyAndWaitForCompletion()
                res = an.GetResults()
                tag = "auto" if order == 0 else "m%d" % order
                out = os.path.join(OUT_DIR, "od_%s_%s.txt"
                                   % (fname, tag))
                res.GetTextFile(out)
                an.Close()
                print("order %s: %s -> %s" % ("auto" if order == 0 else str(order), fname, out))
        set_par(2, float(ORDERS[-1]))

        TheSystem.SaveAs(os.path.join(OUT_DIR,
                                      "mdl_validation_od.zos"))
        print("saved system -> mdl_validation_od.zos")
        del zos
        return

    # ------------------------------------------------------------------
    # PSF per wavelength (zone mode).
    #
    # FFT PSF (default): one 8192^2 pupil ray grid through the DLL + one
    # FFT -> minutes/wavelength, and fully adequate for the on-axis PSF
    # at NA ~ 0.1-0.3.
    # Huygens at high pupil sampling costs (pupil samples)x(image
    # samples) wavelet sums ~ HOURS per wavelength -- only use it (set
    # USE_HUYGENS = True) for a final spot check on one wavelength.
    # ------------------------------------------------------------------
    USE_HUYGENS = False

    # Optional through-focus scan: image-plane offsets in mm. For each
    # offset the UDS thickness is shifted and the PSFs recomputed --
    # this is the DIFFRACTION-analysis replacement for the (ray-based,
    # meaningless here) longitudinal aberration plot: rays exit the
    # zone-decomposition surface undeviated, so ray fans/longitudinal
    # color show nothing by construction; the chromatic focus ladder
    # only appears in through-focus PSF (or POP) data.
    THROUGH_FOCUS_MM = []              # e.g. [-3.0, -1.5, 0.0, 1.5, 3.0]

    def typed_settings(analysis, probe_attr):
        """GetSettings() may hand back the generic IAS_ base interface;
        the derived one is reached via __implementation__ on recent
        OpticStudio + pythonnet versions."""
        st = analysis.GetSettings()
        if not hasattr(st, probe_attr):
            st = st.__implementation__
        return st

    def set_best_sampling(st, attr, names):
        for size_name in names:
            try:
                setattr(st, attr, getattr(ZOSAPI.Analysis.SampleSizes,
                                          size_name))
                return size_name
            except Exception:
                continue
        return "(default)"

    for wi in range(1, len(WAVELENGTHS_UM) + 1):
        if USE_HUYGENS:
            an = TheSystem.Analyses.New_HuygensPsf()
            st = typed_settings(an, "Wavelength")
            st.Wavelength.SetWavelengthNumber(wi)
            set_best_sampling(st, "PupilSampleSize",
                              ("S_4096x4096", "S_2048x2048"))
            try:
                st.ImageSampleSize = ZOSAPI.Analysis.SampleSizes.S_256x256
                st.ImageDelta = 0.4       # um
            except Exception as exc:
                print("  sampling fallback: %s" % exc)
            tag = "huygens"
        else:
            an = TheSystem.Analyses.New_FftPsf()
            st = typed_settings(an, "Wavelength")
            st.Wavelength.SetWavelengthNumber(wi)
            used = set_best_sampling(st, "SampleSize",
                                     ("S_8192x8192", "S_4096x4096",
                                      "S_2048x2048"))
            print("  FFT PSF pupil sampling: %s" % used)
            try:
                st.Type = ZOSAPI.Analysis.Settings.Psf.FftPsfType.Linear
            except Exception:
                pass
            tag = "fft"
        for dz in (THROUGH_FOCUS_MM or [0.0]):
            surf2.Thickness = BFD_GUESS_MM + dz
            an.ApplyAndWaitForCompletion()
            res = an.GetResults()
            if dz == 0.0:
                out = os.path.join(OUT_DIR, "%s_psf_w%d.txt" % (tag, wi))
            else:
                out = os.path.join(OUT_DIR, "%s_psf_w%d_dz%+0.1fmm.txt"
                                   % (tag, wi, dz))
            res.GetTextFile(out)
            print("%s PSF wavelength %d dz=%+.1f -> %s"
                  % (tag, wi, dz, out))
        surf2.Thickness = BFD_GUESS_MM
        an.Close()

    TheSystem.SaveAs(os.path.join(OUT_DIR, "mdl_validation.zos"))
    print("saved system -> mdl_validation.zos")
    del zos


if __name__ == "__main__":
    main()