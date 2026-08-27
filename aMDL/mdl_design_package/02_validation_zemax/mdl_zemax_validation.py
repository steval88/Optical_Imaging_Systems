"""
mdl_zemax_validation.py -- STAGE 2b: Zemax OpticStudio validation
=================================================================

Validate a designed MDL inside Ansys Zemax OpticStudio through the
ZOS-API, using the "MDL Rings" User-Defined Surface DLLs:

    zone mode : us_mdl_rings.dll    (staircase / zone decomposition)
                rays go STRAIGHT; only diffraction analyses (FFT /
                Huygens PSF) are physical. Quantitative ground truth.
    od mode   : us_mdl_rings_od.dll (order decomposition, Binary-2
                style) rays bend per the grating equation of ONE
                order; ray-based chromatic analyses (longitudinal,
                chromatic focal shift, ray fans) are meaningful, one
                order per pass; Order m = 0 selects the dominant
                (blazed) order per wavelength (camera view).

Usage (run on the machine with OpticStudio + pythonnet):

    python mdl_zemax_validation.py <run-folder-or-design> [zone|od]

<run-folder-or-design> is preferably a RUN FOLDER produced by
run_MDL_design (e.g. runs\\20260825_165039_s3_comb_softmin_overlap):
every parameter -- aperture, focal, ring-table file number, test
wavelengths, OD fold P / lam0 and the order list -- is then derived
from the folder's config.json + design_metrics.json, and all analysis
output is written into <run-folder>\\zemax\\ so the Zemax results live
with the design they validate. The legacy names {na03, s3, s3comb}
still select the hand-kept registry entries below.

OD fold recovery from a run folder
----------------------------------
The OD DLL unfolds the ring table at h_fold = P*lam0/(n(lam0)-1).
 * harmonic-seeded runs store lam0 and p directly in the seed record;
 * echelle-seeded runs store h_fold_um; (P, lam0) are recovered as the
   target line with the smallest blaze detune -- for the
   s3_comb_softmin_overlap design that is 600 nm / P = 15, detune
   0.000, reproducing H_fold = 14.377 um exactly. Any (line, order)
   pair on the fold's congruence lattice would define the same h_fold;
   the smallest-detune line is the most faithful.
The default OD order list is {lowest ladder order, design P, highest
ladder order, 0=auto}; edit `orders` below or in the returned dict for
a denser scan (e.g. add 10 to look at the 850/950 nm lines of the
softmin design).

Prerequisites on the OpticStudio machine
----------------------------------------
1. Copy us_mdl_rings.dll, us_mdl_rings_od.dll and the design's ring
   table mdl_rings_<File#>.txt into {Documents}\\Zemax\\DLL\\Surfaces\\
   (the table must sit NEXT TO the DLLs -- they load it from their own
   folder).
2. pip install pythonnet; zos_connection.py importable (same folder or
   the ZOS_API_Examples package layout).

System layout (sequential)
--------------------------
    OBJ (inf)
    1   STOP  substrate front face, flat, MODEL glass fitted to the
              AZ4562 Cauchy (nd=1.6274, Vd~30.6), 1.1 mm
    2   UDS   MDL relief (zone or od DLL): resist -> air
    3   IMA   at the design BFD

NOTE ON SAMPLING: FFT/Huygens PSF pupil sampling must resolve the
rings -- use >= 8192x8192 where licensed; expect minutes/wavelength
(FFT) to hours (Huygens). Geometric analyses are meaningless in zone
mode; only diffraction-based analyses are physical there.
"""
import json
import os
import sys

# zos_connection may live next to this script or in the user's
# ZOS_API_Examples package; the import error is deferred to runtime so
# the config-parsing path stays testable on non-Zemax machines.
try:
    from zos_connection import PythonStandaloneApplication
except ImportError:
    try:
        from ZOS_API_Examples.zos_connection import (
            PythonStandaloneApplication)
    except ImportError:
        PythonStandaloneApplication = None

# same Cauchy as mdl_core.n_az4562 AND as hardcoded in both DLLs --
# keep the three in sync if the material ever changes
def n_resist(lam_um):
    return 1.594 + 0.01152 / lam_um ** 2


# --- LEGACY registry (pre-run-folder designs shipped as loose files) ---
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

SUBSTRATE_MM = 1.1


def design_from_run_folder(run_dir):
    """DESIGN dict derived from a run folder -- no registry entry.

    Geometry / file number / wavelengths come from config.json; the OD
    fold parameters from the seed record in design_metrics.json (see
    module docstring). Raises SystemExit with a actionable message if
    the folder lacks what is needed.
    """
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit("no config.json in %r -- pass a run folder "
                         "made by run_MDL_design, or one of the legacy "
                         "design names %s" % (run_dir,
                                              sorted(DESIGNS)))
    cfg = json.load(open(cfg_path))
    der = cfg["derived"]
    lams_all = list(cfg.get("target_wavelengths_um")
                    or cfg["verify_wavelengths_um"])
    # 5 representative wavelengths across the set (PSFs are expensive);
    # edit wavelengths_um in the returned dict for a different pick
    n = len(lams_all)
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    wavelengths = [float(lams_all[i]) for i in idxs]
    primary_idx = (len(wavelengths) + 1) // 2          # middle, 1-based

    dm_path = os.path.join(run_dir, "design_metrics.json")
    seed = {}
    if os.path.exists(dm_path):
        seed = json.load(open(dm_path)).get("seed") or {}

    if seed.get("mode") == "echelle" or "h_fold_um" in seed:
        h_fold = float(seed["h_fold_um"])
        best = None
        for lam in lams_all:
            alpha = (n_resist(lam) - 1.0) * h_fold / lam
            det = abs(alpha - round(alpha))
            if best is None or det < best[0]:
                best = (det, float(lam), int(round(alpha)))
        det0, lam0, P = best
        ladder = sorted({int(o) for o in seed.get("orders", [])})
        orders = sorted({ladder[0], P, ladder[-1]}) + [0] \
            if ladder else [P, 0]
        fold_note = ("echelle seed: h_fold=%.4f um -> P=%d @ %.2f um "
                     "(detune %.3f)" % (h_fold, P, lam0, det0))
    elif "lam0_um" in seed:                     # harmonic seed record
        lam0, P = float(seed["lam0_um"]), int(seed["p"])
        orders = sorted({max(1, P - 4), P, P + 4}) + [0]
        fold_note = "harmonic seed: P=%d @ %.2f um" % (P, lam0)
    else:
        raise SystemExit(
            "design_metrics.json in %r has no seed record (reused "
            "design?) -- od mode needs the fold: add fold_P/lam0_um "
            "by hand to a registry entry, or run zone mode only"
            % run_dir)

    return dict(epd_mm=cfg["diameter_um"] / 1000.0,
                bfd_mm=der["focal_um"] / 1000.0,
                file_no=int(cfg["dll_file_no"]),
                wavelengths_um=wavelengths,
                primary_idx=primary_idx,
                fold_P=P, lam0_um=lam0, orders=orders,
                run_dir=run_dir, fold_note=fold_note)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "s3"
    mode = sys.argv[2] if len(sys.argv) > 2 else "zone"
    assert mode in ("zone", "od", "huygens"), \
        "mode must be 'zone', 'od' or 'huygens <lam_um>'"
    # 'huygens' = fine-PSF SPOT CHECK on ONE wavelength (3rd argument,
    # um; nearest system wavelength is used). Needed because the FFT
    # PSF image spacing is fixed by the aperture geometry at ~2 lam F/#
    # (~4-11 um here) -- it cannot resolve this lens's 2-6 um PSF at
    # any display setting (measured on the 2026-08-26 run: OutputSize
    # only DECIMATES the full +/-16-44 mm window). Huygens with an
    # explicit ImageDelta is the fine instrument; budget HOURS per
    # wavelength, hence one at a time.
    # one wavelength or a comma-separated list, e.g.  huygens 0.95
    # or  huygens 0.55,0.95,1.10  (each costs HOURS -- queue with care)
    huygens_lams = ([float(v) for v in sys.argv[3].split(",")]
                    if mode == "huygens" and len(sys.argv) > 3
                    else [0.95])
    if mode == "huygens":
        raise SystemExit(
            "huygens mode is RETIRED: OpticStudio's Huygens PSF "
            "propagates each ray as a PLANE-WAVE wavelet along the ray "
            "direction, and the zone DLL's rays exit STRAIGHT -- every "
            "wavelet then has zero transverse phase variation across "
            "the image plane, so the coherent sum is CONSTANT (flat "
            "field, Strehl 0.000). Verified on every setting tried "
            "(2026-08-26: 4096^2 pupil, 32/128 image, delta "
            "0.4/0.8/8/auto) AND on the 2026-08-01 archives (8192^2, "
            "512^2, 0.2 um: 262144 samples, ONE distinct value). "
            "Fine PSFs in Zemax: run 'zone' and check whether the "
            "FFT ImageDelta experiment reports dx=0.4 in the grid "
            "echo; otherwise POP, or rely on RS+BPM (run_verify / "
            "bpm_validate), which own PSF morphology.")

    if os.path.isdir(arg):
        design = design_from_run_folder(arg)
        out_dir = os.path.join(arg, "zemax")
    elif arg in DESIGNS:
        design = DESIGNS[arg]
        out_dir = os.path.join(os.path.expanduser("~"), "Documents",
                               "Zemax_MDL_Validation")
    else:
        raise SystemExit("unknown design %r: pass a run folder or one "
                         "of %s" % (arg, sorted(DESIGNS)))
    # ABSOLUTE path is mandatory: SaveAs/GetTextFile/SaveTo are executed
    # by the OpticStudio PROCESS, which resolves relative paths against
    # ITS OWN working directory, not this script's -- with a relative
    # out_dir every OpticStudio-written file silently lands elsewhere
    # (only Python's own ray_checks.txt would appear here)
    out_dir = os.path.abspath(out_dir)

    wavelengths_um = design["wavelengths_um"]
    primary_idx = design["primary_idx"]
    epd_mm = design["epd_mm"]
    bfd_mm = design["bfd_mm"]
    file_no = design["file_no"]
    fold_p = design["fold_P"]
    lam0_um = design["lam0_um"]
    orders = design["orders"]
    dll_name = ("us_mdl_rings_od.dll" if mode == "od"
                else "us_mdl_rings.dll")   # zone + huygens: staircase

    print("design: %s" % (design.get("run_dir", arg)))
    print("  EPD %.2f mm | BFD %.2f mm | ring table mdl_rings_%d.txt"
          % (epd_mm, bfd_mm, file_no))
    print("  wavelengths (um): %s  (primary #%d = %.2f)"
          % (", ".join("%.2f" % w for w in wavelengths_um),
             primary_idx, wavelengths_um[primary_idx - 1]))
    if mode == "od":
        print("  od fold: P=%d @ lam0=%.2f um; orders %s"
              % (fold_p, lam0_um, orders))
        if "fold_note" in design:
            print("  (%s)" % design["fold_note"])
    print("  DLL: %s | output -> %s" % (dll_name, out_dir))
    print("  REMINDER: mdl_rings_%d.txt must sit next to the DLLs in "
          "{Documents}\\Zemax\\DLL\\Surfaces\\" % file_no)

    if PythonStandaloneApplication is None:
        raise SystemExit("zos_connection.py not importable -- run this "
                         "on the OpticStudio machine (pip install "
                         "pythonnet) with zos_connection.py next to "
                         "this script")

    zos = PythonStandaloneApplication()
    ZOSAPI = zos.ZOSAPI
    TheSystem = zos.TheSystem
    os.makedirs(out_dir, exist_ok=True)

    TheSystem.New(False)
    TheSystem.MakeSequential()
    SysData = TheSystem.SystemData

    SysData.Aperture.ApertureType = \
        ZOSAPI.SystemData.ZemaxApertureType.EntrancePupilDiameter
    SysData.Aperture.ApertureValue = epd_mm

    SysData.Wavelengths.RemoveWavelength(1)
    for w in wavelengths_um:
        SysData.Wavelengths.AddWavelength(w, 1.0)
    SysData.Wavelengths.GetWavelength(primary_idx).MakePrimary()

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
    uds_type.Filename = dll_name
    surf2.ChangeType(uds_type)
    surf2.Thickness = bfd_mm

    def set_par(num, value):
        col = getattr(ZOSAPI.Editors.LDE.SurfaceColumn, "Par%d" % num)
        surf2.GetSurfaceCell(col).DoubleValue = float(value)

    if mode != "od":                     # zone + huygens: staircase DLL
        surf2.Comment = "MDL Rings staircase (zone decomposition)"
        # Par 1..4: File #, Height scale, Z sign, Parax f
        set_par(1, file_no)
        set_par(2, 1.0)
        # Z sign MUST be +1 in this trace direction (resist -> air): the
        # relief pokes into the exit space (z = +h), so taller rings add
        # resist path and the transmitted phase is +k(n-1)h as designed.
        # (-1 inverts the lens: diverging output, virtual focus at -F,
        # flat halo at the image plane.)
        set_par(3, 1.0)
        set_par(4, bfd_mm)
    else:
        surf2.Comment = "MDL Rings order decomposition"
        # Par 1..6: File #, Order m, Design P, Lam0 um, Add OPL, Use eff
        set_par(1, file_no)
        set_par(2, float(orders[-1]))   # start at the last listed order
        set_par(3, float(fold_p))
        set_par(4, lam0_um)
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
        norm.AddRay(primary_idx, 0.0, 0.0, 0.0, py,
                    ZOSAPI.Tools.RayTrace.OPDMode.Current)
    rt.RunAndWaitForCompletion()
    ray_path = os.path.join(out_dir, "ray_checks.txt")
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

    # save the system FIRST, so a failure in any analysis below never
    # costs the .zos file (it is saved again, updated, at the end)
    zos_name = {"zone": "mdl_validation.zos",
                "od": "mdl_validation_od.zos",
                "huygens": "mdl_validation_huygens.zos"}[mode]
    TheSystem.SaveAs(os.path.join(out_dir, zos_name))
    print("saved system -> %s (will be re-saved after the analyses)"
          % zos_name)

    def save_psf_datagrid(an, out_base):
        """Extract the PSF straight from the analysis results DataGrid
        into numpy and save a compact compressed .npz -- BYPASSES
        GetTextFile, whose export dumps the full computation grid
        (observed: 6.8 GB of ASCII for an 8192^2 pupil, with the
        OutputSize cap ignored). Uses the pinned-GCHandle fast copy
        from the official ZOS-API Python examples. The stored array is
        the central 512x512 window (or the full grid if smaller);
        dx/dy carry the sample spacing so downstream scripts can
        rebuild physical coordinates. Returns the saved path or None.
        """
        try:
            import ctypes
            import numpy as np
            from System.Runtime.InteropServices import (GCHandle,
                                                        GCHandleType)
            res = an.GetResults()
            ng = int(res.NumberOfDataGrids)
            if ng < 1:
                print("  (no DataGrids in the results)")
                return None
            # save EVERY grid: which index is the PSF differs between
            # analyses (measured: FFT grid 0 = the PSF, but the Huygens
            # grid 0 came back as a CONSTANT field -- not the PSF).
            # Description/ValueLabel identify each grid; grid i is
            # stored as I<i> with its own meta, and I = grid 0 for
            # backward compatibility.
            payload = {}
            for gi in range(ng):
                dg = res.GetDataGrid(gi)
                nx, ny = int(dg.Nx), int(dg.Ny)
                hnd = GCHandle.Alloc(dg.Values, GCHandleType.Pinned)
                try:
                    ptr = hnd.AddrOfPinnedObject().ToInt64()
                    arr = np.empty((ny, nx), dtype=np.float64)
                    ctypes.memmove(arr.ctypes.data, ptr, arr.nbytes)
                finally:
                    hnd.Free()
                kx, ky = min(nx, 512), min(ny, 512)
                x0, y0 = (nx - kx) // 2, (ny - ky) // 2
                desc = "%s | %s" % (str(dg.Description),
                                    str(dg.ValueLabel))
                payload["I%d" % gi] = arr[y0:y0 + ky, x0:x0 + kx]
                payload["meta%d" % gi] = np.array(
                    [float(dg.Dx), float(dg.Dy), float(dg.MinX),
                     float(dg.MinY), nx, ny, x0, y0])
                payload["desc%d" % gi] = np.array(desc)
                print("  grid %d/%d: %dx%d, dx=%.4g, dy=%.4g, "
                      "min=%.3g, max=%.3g  [%s]"
                      % (gi, ng, ny, nx, float(dg.Dx), float(dg.Dy),
                         arr.min(), arr.max(), desc))
            payload["I"] = payload["I0"]
            m0 = payload["meta0"]
            payload.update(dx=m0[0], dy=m0[1], minx=m0[2], miny=m0[3],
                           nx_full=int(m0[4]), ny_full=int(m0[5]),
                           crop_x0=int(m0[6]), crop_y0=int(m0[7]))
            out = out_base + ".npz"
            np.savez_compressed(out, **payload)
            return out
        except Exception as exc:
            print("  (DataGrid extraction failed: %s)" % exc)
            return None

    def modify_settings_route(an, pairs, tag):
        """Version-proof analysis settings: SaveTo a .cfg, apply ZPL
        MODIFYSETTINGS codes, LoadFrom. Works on every ZOS-API build
        because SaveTo/ModifySettings/LoadFrom live on the BASE
        settings interface. Returns True on success."""
        try:
            st = an.GetSettings()
            cfg = os.path.join(out_dir, "_%s.cfg" % tag)
            st.SaveTo(cfg)
            for code, val in pairs:
                st.ModifySettings(cfg, code, str(val))
            st.LoadFrom(cfg)
            return True
        except Exception as exc:
            print("  (ModifySettings route failed: %s)" % exc)
            return False

    if mode == "od":
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

        for order in orders:
            set_par(2, float(order))
            for idm, fname in analyses:
                tag = "auto" if order == 0 else "m%d" % order
                try:
                    an = TheSystem.Analyses.New_Analysis(idm)
                    an.ApplyAndWaitForCompletion()
                    res = an.GetResults()
                    out = os.path.join(out_dir, "od_%s_%s.txt"
                                       % (fname, tag))
                    res.GetTextFile(out)
                    an.Close()
                    print("order %s: %s -> %s"
                          % ("auto" if order == 0 else str(order),
                             fname, out))
                except Exception as exc:
                    print("order %s: %s FAILED: %s (continuing)"
                          % (tag, fname, exc))
        set_par(2, float(orders[-1]))

        TheSystem.SaveAs(os.path.join(out_dir, "mdl_validation_od.zos"))
        print("re-saved system -> mdl_validation_od.zos")
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
    USE_HUYGENS = (mode == "huygens")

    # Optional through-focus scan: image-plane offsets in mm. For each
    # offset the UDS thickness is shifted and the PSFs recomputed --
    # this is the DIFFRACTION-analysis replacement for the (ray-based,
    # meaningless here) longitudinal aberration plot: rays exit the
    # zone-decomposition surface undeviated, so ray fans/longitudinal
    # color show nothing by construction; the chromatic focus ladder
    # only appears in through-focus PSF (or POP) data.
    THROUGH_FOCUS_MM = []              # e.g. [-3.0, -1.5, 0.0, 1.5, 3.0]

    def typed_settings(analysis, probe_attr):
        """Best-effort typed settings: GetSettings() may hand back the
        generic IAS_ base interface; on some OpticStudio + pythonnet
        combinations the derived interface is reachable via
        __implementation__, on others it is not reachable at all.
        NEVER raises -- when the probe attribute cannot be reached the
        base object is returned and the caller falls back to the
        modify_settings_route (ZPL MODIFYSETTINGS codes)."""
        st = analysis.GetSettings()
        if hasattr(st, probe_attr):
            return st
        impl = getattr(st, "__implementation__", None)
        if impl is not None and hasattr(impl, probe_attr):
            return impl
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

    # sampling index for the MODIFYSETTINGS route: 1 = 32x32, one step
    # per doubling -> 9 = 8192x8192 (OpticStudio clamps to the largest
    # grid the license/memory allows)
    SAMP_IDX_8192 = 9
    SAMP_IDX_4096 = 8

    if USE_HUYGENS:
        wi_list = []
        for hl in huygens_lams:
            wi_pick = min(range(len(wavelengths_um)),
                          key=lambda i: abs(wavelengths_um[i]
                                            - hl)) + 1
            if wi_pick not in wi_list:
                wi_list.append(wi_pick)
        print("Huygens spot check: %d wavelength(s): %s -- expect "
              "HOURS each" %
              (len(wi_list),
               ", ".join("#%d=%.2f um" % (w, wavelengths_um[w - 1])
                         for w in wi_list)))
    else:
        wi_list = list(range(1, len(wavelengths_um) + 1))
    for wi in wi_list:
        if USE_HUYGENS:
            an = TheSystem.Analyses.New_HuygensPsf()
            # probe for the attribute we need to SET: the base IAS_
            # interface already has .Wavelength, so probing for that
            # would never fetch the derived implementation
            st = typed_settings(an, "PupilSampleSize")
            used = set_best_sampling(st, "PupilSampleSize",
                                     ("S_4096x4096", "S_2048x2048"))
            # IMAGE settings: the typed assignments were observed to
            # "succeed" WITHOUT being applied (2026-08-26 run: log
            # clean, but the result grid came back 32x32 default,
            # 3 KB npz, 16-min runtime = 1/64 of the expected cost).
            # Therefore the image grid + wavelength are ALWAYS forced
            # through the ModifySettings file route (ZPL codes
            # HPS_IMAGESAMP: 1=32x32, +1 per doubling; HPS_IMAGEDELTA
            # in um; HPS_WAVE = wavelength number).
            # 128x128 @ 0.8 um = +/-51 um window, ~13 samples across
            # the 950 nm annulus radius, ~4-5 h at 4096^2 pupil
            # (256x256 @ 0.4 um would be ~17 h -- edit here if wanted).
            # HUY_DELTA_UM = None leaves ImageDelta at the analysis
            # default (0 = auto). EVIDENCE (2026-08-26): every run
            # that set an explicit delta (0.4 / 8 / 0.8) returned a
            # FLAT field with Strehl 0.000, while the only valid
            # Huygens PSFs ever produced (2026-08-03) ran with the
            # default -- possibly a lens-units (mm) interpretation of
            # the property. Leave None until a nonzero value is proven
            # against the per-grid echo.
            HUY_IMG_IDX, HUY_DELTA_UM = 3, None      # 3 = 128x128
            # INTEGER codes only through ModifySettings: decimal values
            # are parsed with the SYSTEM LOCALE (measured on an Italian
            # locale: "0.8" -> 8, the dot eaten as a group separator).
            pairs = [("HPS_IMAGESAMP", HUY_IMG_IDX),
                     ("HPS_WAVE", wi)]
            if used == "(default)":
                pairs.insert(0, ("HPS_PUPILSAMP", SAMP_IDX_4096))
            ok = modify_settings_route(an, pairs, "huygens")
            # the DOUBLE goes through the typed property (no string
            # parsing, locale-proof); set AFTER LoadFrom so the file
            # route cannot overwrite it
            if HUY_DELTA_UM is None:
                delta_note = "left at analysis default (auto)"
            else:
                delta_note = "typed FAILED"
                std = typed_settings(an, "ImageDelta")
                try:
                    std.ImageDelta = float(HUY_DELTA_UM)
                    delta_note = "typed to %g" % HUY_DELTA_UM
                except Exception as exc:
                    print("  (typed ImageDelta failed: %s)" % exc)
            print("  Huygens pupil %s; image 128x128 via ModifySettings"
                  ": %s; ImageDelta %s -- verify via the per-grid "
                  "echo below"
                  % (used, "ok" if ok else "FAILED", delta_note))
            tag = "huygens"
        else:
            an = TheSystem.Analyses.New_FftPsf()
            st = typed_settings(an, "SampleSize")     # see comment above

            def psf_enum(size_name):
                """FFT PSF settings take the PsfSampling enum (e.g.
                PsfS_8192x8192), NOT the generic SampleSizes enum --
                assigning the wrong type raises, which is why the
                original typed route failed. Fall back to SampleSizes
                for older API builds."""
                try:
                    return getattr(
                        ZOSAPI.Analysis.Settings.Psf.PsfSampling,
                        "PsfS_" + size_name)
                except AttributeError:
                    return getattr(ZOSAPI.Analysis.SampleSizes,
                                   "S_" + size_name)

            samp_ok = None
            for size in ("8192x8192", "4096x4096", "2048x2048"):
                try:
                    st.SampleSize = psf_enum(size)
                    samp_ok = size
                    break
                except Exception:
                    continue
            # OutputSize caps the DISPLAYED/EXPORTED grid independently
            # of the pupil sampling: without it the text export dumps
            # the full pupil grid (8192^2 points ~ 6.8 GB per file!).
            # 256x256 at the FFT's own image spacing covers roughly
            # +/-180 um around the chief ray -- ample for the PSF core,
            # annuli and near satellites -- at ~7 MB per file.
            out_ok = None
            for size in ("256x256", "512x512"):
                try:
                    st.OutputSize = psf_enum(size)
                    out_ok = size
                    break
                except Exception:
                    continue
            wave_ok = True
            try:
                st.Wavelength.SetWavelengthNumber(wi)
            except Exception:
                wave_ok = False
            try:
                st.Type = ZOSAPI.Analysis.Settings.Psf.FftPsfType.Linear
            except Exception:
                pass
            # EXPERIMENT (per the version's own ZOSAPI.chm):
            # IAS_FftPsf exposes ImageDelta [um]. If it is honored, the
            # exported grid spacing (echoed by the saver below) drops
            # from the geometry-pinned ~2 lam F/# to 0.4 um -- fine
            # PSFs for ALL wavelengths in minutes, making the huygens
            # mode optional. If dx stays coarse, the setting is
            # cosmetic on this build and huygens remains the fine
            # instrument.
            try:
                st.ImageDelta = 0.4                    # um
                print("  FFT ImageDelta requested: 0.4 um "
                      "(verify via the dx echoed below)")
            except Exception as exc:
                print("  (FFT ImageDelta not settable: %s)" % exc)
            if samp_ok is None or not wave_ok:
                pairs = []
                if samp_ok is None:
                    pairs += [("PSF_SAMP", SAMP_IDX_8192)]
                if not wave_ok:
                    pairs += [("PSF_WAVE", wi)]
                ok = modify_settings_route(an, pairs, "fftpsf")
                print("  FFT PSF sampling via ModifySettings: %s"
                      % ("ok" if ok else
                         "FAILED -- default sampling cannot resolve "
                         "the rings; treat these PSFs as INVALID"))
            else:
                print("  FFT PSF pupil sampling: S_%s" % samp_ok)
            if out_ok is None:
                print("  WARNING: could not cap OutputSize -- the text "
                      "export will contain the FULL pupil grid "
                      "(~GBytes per file); Ctrl+C and report if that "
                      "is not acceptable")
            else:
                print("  FFT PSF output grid capped at %s" % out_ok)
            tag = "fft"
        for dz in (THROUGH_FOCUS_MM or [0.0]):
            surf2.Thickness = bfd_mm + dz
            try:
                an.ApplyAndWaitForCompletion()
                base = ("%s_psf_w%d" % (tag, wi) if dz == 0.0 else
                        "%s_psf_w%d_dz%+0.1fmm" % (tag, wi, dz))
                out = save_psf_datagrid(an,
                                        os.path.join(out_dir, base))
                if tag == "huygens":
                    # 128^2 text export is small (~MBs) and is the
                    # route that produced valid PSFs in the 2026-08-03
                    # validation -- keep it as the cross-check while
                    # the DataGrid indexing of this analysis is being
                    # pinned down
                    txt = os.path.join(out_dir, base + ".txt")
                    try:
                        an.GetResults().GetTextFile(txt)
                        print("  text cross-check -> %s" % txt)
                    except Exception as exc:
                        print("  (text export failed: %s)" % exc)
                if out is None:
                    # last resort: the text export (can be HUGE -- the
                    # full computation grid; kept only so a run never
                    # ends empty-handed)
                    out = os.path.join(out_dir, base + ".txt")
                    print("  falling back to GetTextFile -- may be "
                          "gigabytes")
                    an.GetResults().GetTextFile(out)
                print("%s PSF wavelength %d dz=%+.1f -> %s"
                      % (tag, wi, dz, out))
            except Exception as exc:
                print("%s PSF wavelength %d dz=%+.1f FAILED: %s "
                      "(continuing)" % (tag, wi, dz, exc))
        surf2.Thickness = bfd_mm
        an.Close()

    TheSystem.SaveAs(os.path.join(out_dir, zos_name))
    print("re-saved system -> %s" % zos_name)
    print("next: run 'od' mode for the ray-based chromatic ladder:")
    print("  python %s %s od" % (os.path.basename(sys.argv[0]), arg))
    del zos


if __name__ == "__main__":
    main()