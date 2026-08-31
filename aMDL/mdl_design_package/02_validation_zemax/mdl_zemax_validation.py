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
                run_dir=run_dir, fold_note=fold_note,
                lams_all_um=[float(v) for v in
                             cfg["verify_wavelengths_um"]],
                rz_span_mm=float(cfg.get("rzmap_z_span_um",
                                         1000.0)) / 1000.0,
                rz_r_max_um=float(cfg.get("rzmap_r_max_um", 20.0)),
                rz_r_points=int(cfg.get("rzmap_r_points", 41)),
                rz_z_points=int(cfg.get("rzmap_z_points", 121)))


def load_ring_table(run_dir, file_no):
    """(rho_um, h_um, delta_um) from the run folder's ring table."""
    import numpy as np
    path = os.path.join(run_dir, "mdl_rings_%d.txt" % file_no)
    with open(path) as fh:
        N, delta_mm = fh.readline().split()
        N, delta = int(N), float(delta_mm) * 1000.0
        h = np.array([float(fh.readline()) for _ in range(N)]) * 1000.0
    rho = (np.arange(N) + 0.5) * delta
    return rho, h, delta


def rs_tiles(U, rho, delta, lam, zgrid, r0grid):
    """I(r0, z) of exit field U(rho_i) by the J0-reduced RS-I with the
    ring midpoint quadrature -- the IDENTICAL kernel and quadrature as
    run_verify.rs_psf (see that script's header STEPS 2-3 for the
    derivation and references), so Zemax-field tiles and RS tiles
    differ ONLY by whose exit field goes in."""
    import numpy as np
    from scipy.special import j0
    k = 2.0 * np.pi / lam
    out = np.empty((zgrid.size, r0grid.size))
    for iz, z in enumerate(zgrid):
        for ir, r0 in enumerate(r0grid):
            rb = np.sqrt(z * z + rho * rho + r0 * r0)
            s = np.sum(U * j0(k * rho * r0 / rb)
                       * np.exp(1j * k * rb) / rb ** 2 * rho)
            out[iz, ir] = np.abs((z / (1j * lam)) * 2.0 * np.pi
                                 * delta * s) ** 2
    return out


def main():
    # 'gui' anywhere on the command line switches the connection to
    # INTERACTIVE EXTENSION mode: the script drives the OPEN
    # OpticStudio GUI (Programming tab -> Interactive Extension must
    # be waiting), every analysis opens as a NATIVE OpticStudio window
    # with Zemax's own plots, and the windows STAY OPEN when the
    # script exits -- the route for assessing results with Zemax's own
    # plotting tools. Without 'gui': headless standalone (as before).
    argv = [a for a in sys.argv if a.strip().lower() != "gui"]
    GUI = len(argv) != len(sys.argv)
    arg = argv[1] if len(argv) > 1 else "s3"
    mode = argv[2] if len(argv) > 2 else "zone"
    assert mode in ("zone", "od", "huygens", "rz", "rzfft"), \
        "mode must be 'zone', 'od', 'rz [lams]', 'rzfft [lams] [nz]' " \
        "or 'huygens'"
    huygens_lams = ([float(v) for v in argv[3].split(",")]
                    if mode == "huygens" and len(argv) > 3
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
            "The Strehl zeros in old outputs are THIS instrument "
            "artifact, not a design property. For an intensity-based "
            "multiwavelength through-focus picture run the 'rz' mode "
            "(through-focus FFT PSF ladder); fine sub-Airy PSF "
            "morphology belongs to RS + BPM (run_verify / "
            "bpm_validate).")

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

    # ------------------------------------------------------------------
    # RING TABLE SYNC. The DLLs load mdl_rings_<n>.txt from THEIR OWN
    # folder ({Documents}\Zemax\DLL\Surfaces), not from the run folder
    # -- and file numbers get reused across design iterations, so a
    # stale copy silently validates the WRONG design (caught 2026-08-31
    # by the rz self-check: traced profile perfectly quantized and
    # dispersive but random vs the run table). For run-folder designs
    # the script now syncs the table itself: compare bytes, copy on
    # mismatch, warn if the Surfaces folder is missing.
    # ------------------------------------------------------------------
    if "run_dir" in design:
        _fno = int(design["file_no"])
        src_tab = os.path.join(design["run_dir"],
                               "mdl_rings_%d.txt" % _fno)
        dst_dir = os.path.join(os.path.expanduser("~"), "Documents",
                               "Zemax", "DLL", "Surfaces")
        dst_tab = os.path.join(dst_dir, "mdl_rings_%d.txt" % _fno)
        if not os.path.exists(src_tab):
            raise SystemExit("run folder has no %s -- re-run "
                             "run_verify.py to re-export it" % src_tab)
        if os.path.isdir(dst_dir):
            src_bytes = open(src_tab, "rb").read()
            dst_bytes = (open(dst_tab, "rb").read()
                         if os.path.exists(dst_tab) else None)
            if dst_bytes == src_bytes:
                print("  ring table in DLL folder matches the run "
                      "folder (byte-identical)")
            else:
                import shutil as _sh
                _sh.copy2(src_tab, dst_tab)
                print("  ring table %s: DLL-folder copy %s -- SYNCED "
                      "from the run folder"
                      % (os.path.basename(src_tab),
                         "was STALE (different design!)"
                         if dst_bytes is not None else "was missing"))
        else:
            print("  WARNING: %s not found -- copy %s there by hand "
                  "before trusting any Zemax result"
                  % (dst_dir, os.path.basename(src_tab)))

    wavelengths_um = design["wavelengths_um"]
    primary_idx = design["primary_idx"]
    epd_mm = design["epd_mm"]
    bfd_mm = design["bfd_mm"]
    file_no = design["file_no"]
    fold_p = design["fold_P"]
    lam0_um = design["lam0_um"]
    orders = design["orders"]
    dll_name = ("us_mdl_rings_od.dll" if mode == "od"
                else "us_mdl_rings.dll")   # zone/rz/rzfft: staircase

    # ------------------------------------------------------------------
    # rz / rzfft wavelength selection (argv[3] = 'all', a comma list in
    # um, or omitted). rz (batch-OPD route) defaults to ALL
    # verification lines -- it costs seconds per line; rzfft (FFT PSF
    # ladder, EXPERIMENTAL) defaults to the 5 representative lines and
    # takes argv[4] = number of z planes (default 21). The system
    # wavelength table is REPLACED by the selection.
    # ------------------------------------------------------------------
    if mode in ("rz", "rzfft"):
        if len(argv) > 3 and argv[3].strip().lower() == "all":
            rz_lams = design.get("lams_all_um", wavelengths_um)
        elif len(argv) > 3:
            rz_lams = [float(v) for v in argv[3].split(",")]
        elif mode == "rz":
            rz_lams = design.get("lams_all_um", wavelengths_um)
        else:
            rz_lams = list(wavelengths_um)
        rz_nz = int(argv[4]) if len(argv) > 4 else 21
        rz_span = design.get("rz_span_mm", 1.0)
        wavelengths_um = rz_lams
        primary_idx = (len(rz_lams) + 1) // 2

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
    if mode == "rz":
        print("  rz (batch-OPD route): Zemax traces one ray per ring "
              "through the DLL; the traced OPD is self-checked "
              "against the ring table, then the ZEMAX field is "
              "propagated to the full I(r,z) tile grid (RS-I kernel, "
              "same windows as rs/verify_rzmap) -- seconds per "
              "wavelength, %d wavelengths" % len(wavelengths_um))
    if mode == "rzfft":
        print("  rzfft ladder: %d planes over F +/- %.2f mm (%.0f um "
              "steps) x %d wavelengths = %d FFT PSF computations"
              % (rz_nz, rz_span, 2000.0 * rz_span / max(rz_nz - 1, 1),
                 len(wavelengths_um), rz_nz * len(wavelengths_um)))
        print("  EXPERIMENTAL -- measured 2026-08-31: the DataGrid "
              "returned the 256-pt DECIMATED display grid (dx 171.8 "
              "um) and all z-planes came back BIT-IDENTICAL; do not "
              "trust output unless the dx echo reads ~2 lam F/# AND "
              "the per-plane peaks differ. Use 'rz' instead.")
    print("  DLL: %s | output -> %s" % (dll_name, out_dir))
    print("  REMINDER: mdl_rings_%d.txt must sit next to the DLLs in "
          "{Documents}\\Zemax\\DLL\\Surfaces\\" % file_no)

    if PythonStandaloneApplication is None:
        raise SystemExit("zos_connection.py not importable -- run this "
                         "on the OpticStudio machine (pip install "
                         "pythonnet) with zos_connection.py next to "
                         "this script")

    if GUI:
        print("  GUI (interactive extension) mode: connecting to the "
              "OPEN OpticStudio -- Programming tab -> Interactive "
              "Extension must be waiting. Analyses will open as native "
              "windows and STAY OPEN after this script exits.")
        try:
            zos = PythonStandaloneApplication(mode="extension")
        except TypeError:
            raise SystemExit(
                "your zos_connection.py has no 'mode' parameter -- "
                "update it to the version with extension support")
    else:
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

    if mode != "od":                     # zone + rz: staircase DLL
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
                "rz": "mdl_validation_rz.zos",
                "rzfft": "mdl_validation_rzfft.zos",
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
            import numpy as np
            payload = {}
            grids = grab_all_grids(an)
            if grids is None:
                return None
            for gi, (arr, meta, desc) in enumerate(grids):
                ny, nx = arr.shape
                kx, ky = min(nx, 512), min(ny, 512)
                x0, y0 = (nx - kx) // 2, (ny - ky) // 2
                payload["I%d" % gi] = arr[y0:y0 + ky, x0:x0 + kx]
                payload["meta%d" % gi] = np.array(
                    list(meta) + [nx, ny, x0, y0])
                payload["desc%d" % gi] = np.array(desc)
                print("  grid %d/%d: %dx%d, dx=%.4g, dy=%.4g, "
                      "min=%.3g, max=%.3g  [%s]"
                      % (gi, len(grids), ny, nx, meta[0], meta[1],
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

    def grab_all_grids(an):
        """All results DataGrids as [(array, (dx, dy, minx, miny),
        desc)] via the pinned-GCHandle fast copy; None when the
        analysis reported invalid/empty results (aborted)."""
        import ctypes
        import numpy as np
        from System.Runtime.InteropServices import (GCHandle,
                                                    GCHandleType)
        res = an.GetResults()
        ng = int(res.NumberOfDataGrids)
        if ng < 1:
            print("  (no DataGrids in the results)")
            return None
        grids = []
        for gi in range(ng):
            dg = res.GetDataGrid(gi)
            nx, ny = int(dg.Nx), int(dg.Ny)
            if nx == 0 or ny == 0:
                print("  grid %d/%d is EMPTY -- the analysis reported "
                      "invalid results (aborted?)" % (gi, ng))
                return None
            hnd = GCHandle.Alloc(dg.Values, GCHandleType.Pinned)
            try:
                ptr = hnd.AddrOfPinnedObject().ToInt64()
                arr = np.empty((ny, nx), dtype=np.float64)
                ctypes.memmove(arr.ctypes.data, ptr, arr.nbytes)
            finally:
                hnd.Free()
            desc = "%s | %s" % (str(dg.Description),
                                str(dg.ValueLabel))
            grids.append((arr, (float(dg.Dx), float(dg.Dy),
                                float(dg.MinX), float(dg.MinY)), desc))
        return grids

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

    # sampling index for the MODIFYSETTINGS route: 1 = 32x32, one step
    # per doubling -> 9 = 8192x8192 (OpticStudio clamps to the largest
    # grid the license/memory allows)
    SAMP_IDX_8192 = 9

    def configure_fft_psf(an, wi, out_sizes=("256x256", "512x512")):
        """FFT PSF settings for one wavelength number: 8192^2 pupil
        (PsfSampling enum first -- the SampleSizes enum raises on this
        build), OutputSize from out_sizes, Linear, ImageDelta left at
        0 (any smaller value ABORTS the analysis on 2024 R1 --
        measured). zone mode caps the output small (display use); rz
        mode passes the FULL grid sizes because OutputSize DECIMATES
        the image window (measured: 256-pt output = 125-344 um pixels
        over +/-16-44 mm) and the tiles need the native ~2 lam F/#
        pitch -- the per-plane dx echo is the validity gate.
        Returns True when the sampling is trustworthy."""
        st = typed_settings(an, "SampleSize")

        def psf_enum(size_name):
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
        out_ok = None
        for size in out_sizes:
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
        if samp_ok is None or not wave_ok:
            pairs = []
            if samp_ok is None:
                pairs += [("PSF_SAMP", SAMP_IDX_8192)]
            if not wave_ok:
                pairs += [("PSF_WAVE", wi)]
            ok = modify_settings_route(an, pairs, "fftpsf")
            print("  FFT PSF sampling via ModifySettings: %s"
                  % ("ok" if ok else
                     "FAILED -- default sampling cannot resolve the "
                     "rings; treat these PSFs as INVALID"))
            return ok
        print("  FFT PSF pupil sampling: S_%s%s"
              % (samp_ok, "" if out_ok is None
                 else "; output capped at %s" % out_ok))
        return True

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
                    if not GUI:
                        an.Close()   # GUI mode: window stays open
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

    if mode == "rz":
        # --------------------------------------------------------------
        # BATCH-OPD ROUTE -> I(r, z) tiles from the ZEMAX-TRACED field.
        #
        # The batch ray trace is the one ZOS-API interface with zero
        # settings plumbing (no enums, no ModifySettings, no OutputSize
        # decimation) and it has worked flawlessly since day one
        # (ray_checks.txt). One subtlety, MEASURED 2026-08-31: the
        # batch OPD is referenced to the chief-ray REFERENCE SPHERE
        # (the exit-pupil convention of every OPD in OpticStudio), a
        # wavelength-independent sag of hundreds of waves across this
        # aperture -- the raw OPD wrapped against the ring table gave
        # RMS 0.283-0.296 waves at all 14 lines = 1/sqrt(12), i.e.
        # uniform-random, while the field still focused (+-1 mm off,
        # the reference sphere's power removed). Therefore a NULL
        # SUBTRACTION per wavelength: the zone DLL's Par 2 is a height
        # scale, so scale=0 makes the surface FLAT while leaving
        # pupils, chief ray and the reference-sphere convention
        # untouched -- trace flat (instrument terms only), trace the
        # design (instrument + surface), subtract: everything
        # instrumental cancels exactly. Steps per wavelength:
        #   1. two batch traces of one ray per ring (rho_i =
        #      (i+1/2)Delta, py = rho_i/R), height scale 0 then 1;
        #      d_opd = opd_design - opd_flat  [waves];
        #   2. SELF-CHECK: wrapped RMS of (d_opd - ring-table phase)
        #      in waves -- THE Zemax validation content: does
        #      OpticStudio's model of the surface reproduce the design
        #      phase?  (echoed per wavelength; expect ~0);
        #   3. propagate the ZEMAX field exp(i 2 pi OPD) to the same
        #      I(r,z) window as rs/verify_rzmap.npz with the identical
        #      RS-I kernel/quadrature (rs_tiles above) -- so any
        #      difference from the RS tiles is the surface model, not
        #      the propagator;
        #   4. if rs/verify_rzmap.npz exists, print the per-wavelength
        #      z-peak comparison and tile correlation.
        # Output: zemax/zemax_rzmap.npz (r0grid, zgrid [um], I_<nm>,
        # opd_waves_<nm>, phase_rms_waves) + fig_zemax_rz_tiles.png +
        # fig_zemax_onaxis_perlambda.png. Cost: seconds per wavelength.
        # --------------------------------------------------------------
        import time as _time
        import numpy as np
        rho_um, h_um, delta_um = load_ring_table(
            design["run_dir"], file_no)
        N_r = rho_um.size
        R_um = N_r * delta_um
        if abs(R_um / 1000.0 - epd_mm / 2.0) > 1e-6:
            print("  WARNING: ring table aperture %.4f mm != EPD/2 "
                  "%.4f mm" % (R_um / 1000.0, epd_mm / 2.0))
        F_um = bfd_mm * 1000.0
        span_um = design.get("rz_span_mm", 1.0) * 1000.0
        r0grid = np.linspace(0.0, design.get("rz_r_max_um", 20.0),
                             design.get("rz_r_points", 41))
        zgrid = np.linspace(F_um - span_um, F_um + span_um,
                            design.get("rz_z_points", 121))
        store = {"r0grid": r0grid, "zgrid": zgrid,
                 "lam_um": np.array(wavelengths_um)}
        npz_path = os.path.join(out_dir, "zemax_rzmap.npz")
        phase_rms = []
        zpk_zemax = {}
        def trace_opd(wi):
            """One batch trace, one ray per ring; OPD [waves] or NaN
            for missing/vignetted rays."""
            rt2 = TheSystem.Tools.OpenBatchRayTrace()
            norm2 = rt2.CreateNormUnpol(
                N_r + 8, ZOSAPI.Tools.RayTrace.RaysType.Real,
                TheLDE.NumberOfSurfaces - 1)
            norm2.ClearData()
            for rr in rho_um:
                norm2.AddRay(wi, 0.0, 0.0, 0.0, float(rr / R_um),
                             ZOSAPI.Tools.RayTrace.OPDMode.Current)
            rt2.RunAndWaitForCompletion()
            opd = np.full(N_r, np.nan)
            norm2.StartReadingResults()
            ok = True
            while ok:
                (ok, rn, err, vig, x, y, zz_, l, m, n,
                 l2, m2, n2, op, inten) = norm2.ReadNextResult()
                if ok and 1 <= rn <= N_r and err == 0 and vig == 0:
                    opd[rn - 1] = op
            rt2.Close()
            return opd

        for wi, lam in enumerate(wavelengths_um, start=1):
            t0 = _time.time()
            # --- 1. null-subtracted trace (see comment above) -------
            set_par(2, 0.0)               # height scale 0 = FLAT:
            opd_flat = trace_opd(wi)      # instrument terms only
            set_par(2, 1.0)               # the design surface
            opd_dsgn = trace_opd(wi)
            bad = int(np.isnan(opd_flat).sum() + np.isnan(opd_dsgn).sum())
            if bad:
                print("  lam=%.2f um: %d ray results missing/vignetted "
                      "-- SKIPPING" % (lam, bad))
                continue
            d_opd = opd_dsgn - opd_flat   # pure surface phase [waves]
            # --- 2. self-check vs the ring table --------------------
            phi_tab = (n_resist(lam) - 1.0) * h_um / lam
            d = (d_opd - d_opd[0]) - (phi_tab - phi_tab[0])
            d_wrap = d - np.round(d)          # wrapped to +/-0.5 wave
            rms_w = float(np.sqrt(np.mean(d_wrap ** 2)))
            phase_rms.append(rms_w)
            # --- 3. propagate the ZEMAX field to the tile grid ------
            U = np.exp(1j * 2.0 * np.pi * (d_opd - d_opd[0]))
            Mrz = rs_tiles(U, rho_um, delta_um, lam, zgrid, r0grid)
            key = int(round(lam * 1000))
            store["I_%d" % key] = Mrz
            store["opd_waves_%d" % key] = d_opd
            izm, irm = np.unravel_index(int(np.argmax(Mrz)), Mrz.shape)
            zpk_zemax[key] = float(zgrid[izm])
            print("  lam=%.2f um: OPD self-check wrapped RMS = %.4f "
                  "waves | tile peak r=%.1f um z=%.3f mm  (%.1fs)"
                  % (lam, rms_w, r0grid[irm], zgrid[izm] / 1000.0,
                     _time.time() - t0))
            store["phase_rms_waves"] = np.array(phase_rms)
            np.savez_compressed(npz_path, **store)
        print("saved %s" % npz_path)

        # --- 4. compare against the RS tiles, if present ------------
        rs_npz = None
        for cand in (os.path.join(design["run_dir"], "rs",
                                  "verify_rzmap.npz"),
                     os.path.join(design["run_dir"],
                                  "verify_rzmap.npz")):
            if os.path.exists(cand):
                rs_npz = cand
                break
        if rs_npz is not None:
            rs = np.load(rs_npz)
            print("Zemax-field vs RS tiles (%s):" % rs_npz)
            print("  lam(nm)  z_pk_zemax  z_pk_rs   dz(um)   corr")
            for lam in wavelengths_um:
                key = int(round(lam * 1000))
                if "I_%d" % key not in rs.files or key not in zpk_zemax:
                    continue
                Mr = rs["I_%d" % key]
                Mz = store["I_%d" % key]
                izr = int(np.argmax(np.max(Mr, axis=1)))
                z_rs = float(rs["zgrid"][izr])
                c = float(np.corrcoef(Mr.ravel(), Mz.ravel())[0, 1]) \
                    if Mr.shape == Mz.shape else float("nan")
                print("  %5d   %8.3f   %8.3f   %+6.0f   %.4f"
                      % (key, zpk_zemax[key] / 1000.0, z_rs / 1000.0,
                         zpk_zemax[key] - z_rs, c))
            print("  (corr = Pearson r over the full tile; 1.0000 = "
                  "identical structure. dz in um.)")

        # --- figures (same conventions as rs/bpm tiles) -------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            lams_nm = [int(round(l * 1000)) for l in wavelengths_um
                       if "I_%d" % int(round(l * 1000)) in store]
            n = len(lams_nm)
            if n:
                zz_mm = zgrid / 1000.0
                fig, axes = plt.subplots(1, n,
                                         figsize=(0.95 * n + 0.9, 3.0),
                                         sharey=True, squeeze=False)
                for j, lnm in enumerate(lams_nm):
                    ax = axes[0][j]
                    Mrz = store["I_%d" % lnm] / store["I_%d" % lnm].max()
                    sym = np.concatenate([Mrz[:, ::-1], Mrz[:, 1:]],
                                         axis=1)
                    im = ax.imshow(sym, aspect="auto", origin="lower",
                                   extent=[-r0grid[-1], r0grid[-1],
                                           zz_mm[0], zz_mm[-1]],
                                   cmap="OrRd", vmin=0, vmax=1)
                    ax.axhline(bfd_mm, color="black", lw=0.7, ls="--",
                               alpha=0.7)
                    ax.set_title("%d nm" % lnm, fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(False)
                    if j == 0:
                        ax.set_ylabel("z (mm)")
                        ax.set_xlabel("r (µm)")
                    else:
                        ax.set_xticklabels([])
                cb = fig.colorbar(im, ax=axes[0][-1], pad=0.04,
                                  fraction=0.15)
                cb.set_label("I / per-tile max", fontsize=8)
                cb.ax.tick_params(labelsize=7)
                fig.suptitle("I(r, z) around the focus -- ZEMAX-traced "
                             "field (batch OPD through the zone DLL) + "
                             "RS-I propagation", fontsize=10, y=1.02)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir,
                                         "fig_zemax_rz_tiles.png"),
                            dpi=150, bbox_inches="tight")
                # on-axis map, per-lambda normalized
                fig, ax = plt.subplots(figsize=(0.62 * n + 2.6, 3.2))
                Miz = np.array([store["I_%d" % l][:, 0]
                                for l in lams_nm])
                mx = Miz.max(axis=1, keepdims=True)
                mx[mx == 0] = 1.0
                ax.imshow((Miz / mx).T, aspect="auto", origin="lower",
                          extent=[0, n, zz_mm[0], zz_mm[-1]],
                          cmap="magma", vmin=0, vmax=1)
                ax.set_xticks(np.arange(n) + 0.5)
                ax.set_xticklabels(["%d" % l for l in lams_nm],
                                   fontsize=7)
                ax.axhline(bfd_mm, color="white", lw=0.8, ls="--",
                           alpha=0.8)
                ax.set_xlabel("wavelength (nm)")
                ax.set_ylabel("z (mm)")
                ax.set_title("On-axis intensity vs z -- Zemax-traced "
                             "field (per-λ norm)", fontsize=9)
                ax.grid(False)
                fig.tight_layout()
                fig.savefig(os.path.join(
                    out_dir, "fig_zemax_onaxis_perlambda.png"),
                    dpi=150, bbox_inches="tight")
                print("figures -> fig_zemax_rz_tiles.png, "
                      "fig_zemax_onaxis_perlambda.png (in zemax\\)")
        except Exception as exc:
            print("(figure rendering failed: %s -- the npz holds all "
                  "the data; re-plot offline)" % exc)

        TheSystem.SaveAs(os.path.join(out_dir, zos_name))
        print("re-saved system -> %s" % zos_name)
        print("read the tiles against rs\\fig_rz_tiles.png and "
              "bpm\\fig_rz_tiles_bpm.png: same peaks, satellites and "
              "ridge at F = Zemax's surface model agrees with the "
              "design at the intensity level.")
        del zos
        return

    if mode == "rzfft":
        # --------------------------------------------------------------
        # THROUGH-FOCUS FFT PSF LADDER -> I(r, z) per wavelength: the
        # Zemax-native analogue of the RS/BPM r-z tiles, i.e. the
        # intensity-based achromatic-focusing demonstration.
        #
        # For each wavelength, the FFT PSF (zone DLL: pupil-OPD-based,
        # immune to the Huygens straight-ray blindness) is recomputed
        # at rz_nz image planes z = F + dz, dz in +/- rz_span, by
        # shifting the UDS-to-image thickness. From each PSF the
        # central radial cut is extracted (the PSF peak pixel is
        # re-centered per plane, robust to half-pixel offsets).
        #
        # HONEST RESOLUTION LIMIT (measured, see VALIDATION_MODES.md):
        # the FFT image pitch is pinned at ~2 lam F/# = 3.9-10.7 um --
        # about half a FWHM -- so these tiles demonstrate achromatic
        # LOCALIZATION (bright band at F across the band, satellite
        # structure at the mm scale: the defocus cone at NA 0.1 spans
        # ~100 um at |dz| = 1 mm, i.e. 10-25 pixels) but NOT sub-Airy
        # PSF shape, which is owned by RS + BPM. The r axis is stored
        # per wavelength (rz_dx_<nm>) because the pitch is chromatic.
        #
        # Output: zemax_rzmap.npz in zemax/ with zgrid_mm,
        # Irz_<nm> (nz, nr), rz_dx_<nm> [um] and Iz_<nm> = on-axis
        # column -- deliberately shaped like verify_rzmap.npz /
        # bpm_psf.npz so the tile-plotting idiom carries over; the
        # figures are rendered directly below. The npz is re-saved
        # after EVERY completed plane, so Ctrl+C keeps partial data.
        # --------------------------------------------------------------
        import time as _time
        import numpy as np
        NR_KEEP = 64                       # radial pixels kept per cut
        dz_grid = np.linspace(-rz_span, rz_span, rz_nz)     # mm
        store = {"zgrid_mm": bfd_mm + dz_grid,
                 "span_mm": np.array([rz_span]),
                 "lam_um": np.array(wavelengths_um)}
        npz_path = os.path.join(out_dir, "zemax_rzmap_fft.npz")
        t_start = _time.time()
        n_total = len(wavelengths_um) * rz_nz
        n_done = 0
        for wi, lam in enumerate(wavelengths_um, start=1):
            an = TheSystem.Analyses.New_FftPsf()
            # FULL output grid: the tiles need the native ~2 lam F/#
            # image pitch; a capped OutputSize decimates to 100s of um
            trust = configure_fft_psf(
                an, wi, out_sizes=("8192x8192", "4096x4096",
                                   "2048x2048", "1024x1024"))
            if not trust:
                print("  wavelength %.2f um: sampling not trustworthy, "
                      "SKIPPING" % lam)
                an.Close()
                continue
            Mrz = np.full((rz_nz, NR_KEEP), np.nan)
            dx_um = None
            for iz, dz in enumerate(dz_grid):
                t0 = _time.time()
                surf2.Thickness = bfd_mm + float(dz)
                try:
                    an.ApplyAndWaitForCompletion()
                    grids = grab_all_grids(an)
                    if grids is None:
                        print("  lam=%.2f dz=%+.3f mm: EMPTY grid, "
                              "skipped" % (lam, dz))
                        continue
                    arr, meta, _desc = grids[0]   # FFT: grid 0 IS the
                    dx_um = meta[0]               # PSF (measured)
                    # VALIDITY GATE: expected pitch ~2 lam F/# =
                    # 2*lam*(F/D). A much larger dx means OutputSize
                    # decimation reached the DataGrid -- tiles at the
                    # +/-1 mm scale would be meaningless.
                    dx_exp = 2.0 * lam * bfd_mm / epd_mm
                    if dx_um > 3.0 * dx_exp:
                        print("  WARNING lam=%.2f: dx=%.3g um >> "
                              "expected %.2g um (2 lam F/#) -- the "
                              "DataGrid is DECIMATED; raise OutputSize "
                              "(see configure_fft_psf) before trusting "
                              "these tiles" % (lam, dx_um, dx_exp))
                    ny, nx = arr.shape
                    # re-center on the peak (on-axis system: the peak
                    # is the chief-ray pixel up to half-pixel parity)
                    py, px = np.unravel_index(int(np.argmax(arr)),
                                              arr.shape)
                    if abs(py - ny // 2) > 4 or abs(px - nx // 2) > 4:
                        py, px = ny // 2, nx // 2   # defocused: annulus
                    nr = min(NR_KEEP, nx - px)
                    Mrz[iz, :nr] = arr[py, px:px + nr]
                    n_done += 1
                    dt = _time.time() - t0
                    eta = (n_total - n_done) * dt / 60.0
                    print("  lam=%.2f um  z=F%+.3f mm: peak %.3g, "
                          "on-axis %.3g  (%.0fs; ~%.0f min left)"
                          % (lam, dz, arr.max(), Mrz[iz, 0], dt, eta))
                except Exception as exc:
                    print("  lam=%.2f dz=%+.3f mm FAILED: %s "
                          "(continuing)" % (lam, dz, exc))
                # abort-proof: persist after every plane
                key = int(round(lam * 1000))
                store["Irz_%d" % key] = Mrz
                store["Iz_%d" % key] = Mrz[:, 0]
                store["rz_dx_%d" % key] = np.array(
                    [dx_um if dx_um else np.nan])
                np.savez_compressed(npz_path, **store)
            surf2.Thickness = bfd_mm
            if not GUI:
                an.Close()               # GUI mode: window stays open
            print("  wavelength %.2f um done -> %s" % (lam, npz_path))

        # ---- figures (rendered here so zemax/ is self-contained) ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            zz = store["zgrid_mm"]
            lams_nm = [int(round(l * 1000)) for l in wavelengths_um
                       if "Irz_%d" % int(round(l * 1000)) in store]
            # tiles: per-wavelength I(r, z), mirrored, per-tile norm --
            # same convention as fig_rz_tiles / fig_rz_tiles_bpm; note
            # the per-tile r extent differs (chromatic FFT pitch)
            n = len(lams_nm)
            if n:
                fig, axes = plt.subplots(1, n,
                                         figsize=(1.1 * n + 0.9, 3.0),
                                         squeeze=False, sharey=True)
                for j, lnm in enumerate(lams_nm):
                    ax = axes[0][j]
                    Mrz = store["Irz_%d" % lnm].copy()
                    dxl = float(store["rz_dx_%d" % lnm][0])
                    Mrz[np.isnan(Mrz)] = 0.0
                    if Mrz.max() > 0:
                        Mrz = Mrz / Mrz.max()
                    r_max = dxl * (Mrz.shape[1] - 1)
                    sym = np.concatenate([Mrz[:, ::-1], Mrz[:, 1:]],
                                         axis=1)
                    ax.imshow(sym, aspect="auto", origin="lower",
                              extent=[-r_max, r_max, zz[0], zz[-1]],
                              cmap="OrRd", vmin=0, vmax=1)
                    ax.axhline(bfd_mm, color="black", lw=0.7, ls="--",
                               alpha=0.7)
                    ax.set_title("%d nm" % lnm, fontsize=8)
                    ax.tick_params(labelsize=7)
                    if j == 0:
                        ax.set_ylabel("z (mm)")
                        ax.set_xlabel("r (µm)")
                fig.suptitle("I(r, z) around the focus, Zemax zone DLL "
                             "+ through-focus FFT PSF (EXPERIMENTAL -- "
                             "verify dx echo before trusting)",
                             fontsize=9, y=1.04)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir,
                                         "fig_zemax_rz_tiles_fft.png"),
                            dpi=150, bbox_inches="tight")
                # on-axis map, per-lambda normalized (fig_onaxis_
                # perlambda twin: the achromatic ridge at F)
                fig, ax = plt.subplots(figsize=(0.62 * n + 2.6, 3.2))
                Miz = np.array([store["Iz_%d" % l] for l in lams_nm])
                mx = Miz.max(axis=1, keepdims=True)
                mx[mx == 0] = 1.0
                ax.imshow((Miz / mx).T, aspect="auto", origin="lower",
                          extent=[0, n, zz[0], zz[-1]], cmap="magma",
                          vmin=0, vmax=1)
                ax.set_xticks(np.arange(n) + 0.5)
                ax.set_xticklabels(["%d" % l for l in lams_nm],
                                   fontsize=7)
                ax.axhline(bfd_mm, color="white", lw=0.8, ls="--",
                           alpha=0.8)
                ax.set_xlabel("wavelength (nm)")
                ax.set_ylabel("z (mm)")
                ax.set_title("On-axis intensity vs z, Zemax "
                             "through-focus FFT PSF (per-λ norm)",
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(os.path.join(
                    out_dir, "fig_zemax_onaxis_perlambda_fft.png"),
                    dpi=150, bbox_inches="tight")
                print("figures -> fig_zemax_rz_tiles_fft.png, "
                      "fig_zemax_onaxis_perlambda_fft.png (in "
                      "zemax\\)")
        except Exception as exc:
            print("(figure rendering failed: %s -- the npz holds all "
                  "the data; re-plot offline)" % exc)

        TheSystem.SaveAs(os.path.join(out_dir, zos_name))
        print("re-saved system -> %s" % zos_name)
        print("read the tiles against rs\\fig_rz_tiles.png and "
              "bpm\\fig_rz_tiles_bpm.png: same bright band at F "
              "across the band = achromatic focusing, Zemax-native.")
        del zos
        return

    # ------------------------------------------------------------------
    # PSF per wavelength (zone mode): one FFT PSF at the design focus
    # per wavelength -- the coarse energy-localization check.
    # ------------------------------------------------------------------
    for wi in range(1, len(wavelengths_um) + 1):
        an = TheSystem.Analyses.New_FftPsf()
        configure_fft_psf(an, wi)
        try:
            an.ApplyAndWaitForCompletion()
            base = "fft_psf_w%d" % wi
            out = save_psf_datagrid(an, os.path.join(out_dir, base))
            if out is None:
                out = os.path.join(out_dir, base + ".txt")
                print("  falling back to GetTextFile -- may be "
                      "gigabytes")
                an.GetResults().GetTextFile(out)
            print("fft PSF wavelength %d -> %s" % (wi, out))
        except Exception as exc:
            print("fft PSF wavelength %d FAILED: %s (continuing)"
                  % (wi, exc))
        if not GUI:
            an.Close()                   # GUI mode: window stays open

    TheSystem.SaveAs(os.path.join(out_dir, zos_name))
    print("re-saved system -> %s" % zos_name)
    print("next: 'od' for the ray-based chromatic ladder, 'rz' for "
          "the intensity tiles from the Zemax-traced field:")
    print("  python %s %s od" % (os.path.basename(sys.argv[0]), arg))
    print("  python %s %s rz   (all lines, seconds per wavelength)"
          % (os.path.basename(sys.argv[0]), arg))
    del zos


if __name__ == "__main__":
    main()