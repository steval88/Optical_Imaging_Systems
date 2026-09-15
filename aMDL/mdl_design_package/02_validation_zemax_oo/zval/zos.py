"""ZOS-API plumbing: the connection, the sequential system for each
surface VARIANT, DLL parameters, batch ray traces, and the version-proof
analysis-settings / results readers. Nothing here knows about the
physics; every value written to OpticStudio is echoed by the caller."""
import ctypes
import locale
import math
import os

import numpy as np

from .settings import MODEL_GLASS, SUBSTRATE_MM

# zos_connection.py is looked up (in this order) next to the CLI, in the
# sibling folder 02_validation_zemax (the pre-refactor stage folder, when
# this package lives in 02_validation_zemax_oo next to it), then as the
# user's ZOS_API_Examples package. The resolved file is echoed at connect
# time. The import error is deferred to runtime so the config-parsing
# path stays testable on non-Zemax machines.
import sys as _sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _cand in (_HERE, os.path.join(os.path.dirname(_HERE), "02_validation_zemax")):
    if os.path.exists(os.path.join(_cand, "zos_connection.py")) \
            and _cand not in _sys.path:
        _sys.path.insert(0, _cand)
try:
    from zos_connection import PythonStandaloneApplication
    import zos_connection as _zc
    ZOS_CONNECTION_FILE = getattr(_zc, "__file__", "?")
except ImportError:
    try:
        from ZOS_API_Examples.zos_connection import (
            PythonStandaloneApplication)
        from ZOS_API_Examples import zos_connection as _zc
        ZOS_CONNECTION_FILE = getattr(_zc, "__file__", "?")
    except ImportError:
        PythonStandaloneApplication = None
        ZOS_CONNECTION_FILE = None


def zloc(v):
    """Format a number for ZPL MODIFYSETTINGS on THIS machine. MEASURED
    2026-09-04 (Italian Windows): the value is parsed with the SYSTEM
    locale -- '5.12' became 512. Floats carry the locale separator."""
    if isinstance(v, float):
        try:
            old = locale.setlocale(locale.LC_NUMERIC)
            try:
                locale.setlocale(locale.LC_NUMERIC, "")
                return locale.str(v)
            finally:
                locale.setlocale(locale.LC_NUMERIC, old)
        except Exception:
            return str(v)
    return str(v)


def samp_index(n):
    """MODIFYSETTINGS sampling index: 1 = 32x32, one step per doubling."""
    return int(round(math.log2(n))) - 4


# ---------------------------------------------------------------------------
class ZosSession:
    """One OpticStudio connection: headless standalone (default) or the
    OPEN GUI through the Interactive Extension ('gui'): every analysis
    then opens as a native window that STAYS OPEN when the script exits."""

    def __init__(self, gui=False, log=print, app=None):
        self.gui = bool(gui)
        self.log = log
        if app is not None:                       # injected (tests)
            self.app = app
        elif PythonStandaloneApplication is None:
            raise SystemExit("zos_connection.py not importable -- run this "
                             "on the OpticStudio machine (pip install "
                             "pythonnet) with zos_connection.py next to "
                             "mdl_zemax_validation.py")
        elif self.gui:
            log("  GUI (interactive extension) mode: connecting to the OPEN "
                "OpticStudio -- Programming tab -> Interactive Extension "
                "must be waiting. Analyses open as native windows and STAY "
                "OPEN after this script exits.")
            try:
                self.app = PythonStandaloneApplication(mode="extension")
            except TypeError:
                raise SystemExit("your zos_connection.py has no 'mode' "
                                 "parameter -- update it to the version "
                                 "with extension support")
            except Exception as exc:
                raise SystemExit(
                    "extension connection failed (%s: %s). Almost always: "
                    "1. the Interactive Extension tile is not WAITING at "
                    "the moment this script starts, or 2. an orphaned "
                    "headless OpticStudio from an earlier run holds the "
                    "license seat (Task Manager). Re-run with --gui, or "
                    "drop it to run headless." % (type(exc).__name__, exc))
        else:
            self.app = PythonStandaloneApplication()
        if app is None:
            log("  zos_connection: %s" % ZOS_CONNECTION_FILE)
        self.ZOSAPI = self.app.ZOSAPI
        self.TheSystem = self.app.TheSystem

    def close_analysis(self, an):
        """GUI mode keeps the window open."""
        if not self.gui:
            an.Close()

    def close(self):
        self.app = None


# ---------------------------------------------------------------------------
class ZosSystem:
    """The sequential system for one surface VARIANT:

        zone    1 STOP substrate (model glass, 1.1 mm) | 2 us_mdl_rings.dll
                | 3 IMAGE at BFD                       (zone, rz)
        od      same with us_mdl_rings_od.dll              (od)
        hybrid  1 STOP substrate | 2 Paraxial f=F (0 mm, air) | 3 zero-
                thickness glass plate | 4 us_mdl_rings.dll RESIDUAL
                (Sub ideal=1, OPD law 1, dz offset) | 5 IMAGE   (huy)
    """
    DLL = {"zone": "us_mdl_rings.dll", "od": "us_mdl_rings_od.dll",
           "hybrid": "us_mdl_rings.dll"}

    def __init__(self, session, design, variant, out_dir, log=print):
        self.s = session
        self.Z = session.ZOSAPI
        self.sys = session.TheSystem
        self.d = design
        self.variant = variant
        self.out_dir = out_dir
        self.log = log
        self.dll_name = self.DLL[variant]
        self.surf_par = self.surf_plate = self.uds = None
        self.par_f_mm = design.bfd_mm

    # --- build -----------------------------------------------------------
    def build(self, hybrid=None):
        Z, d, sysm = self.Z, self.d, self.sys
        sysm.New(False)
        sysm.MakeSequential()
        sd = sysm.SystemData
        sd.Aperture.ApertureType = \
            Z.SystemData.ZemaxApertureType.EntrancePupilDiameter
        sd.Aperture.ApertureValue = d.epd_mm
        sd.Wavelengths.RemoveWavelength(1)
        for w in d.wavelengths_um:
            sd.Wavelengths.AddWavelength(w, 1.0)
        sd.Wavelengths.GetWavelength(d.primary_idx).MakePrimary()
        self.lde = lde = sysm.LDE

        # surface 1: substrate front (STOP), model glass ~ AZ4562
        s1 = lde.GetSurfaceAt(1)
        s1.Thickness = SUBSTRATE_MM
        s1.MaterialCell.SetSolveData(
            s1.MaterialCell.CreateSolveType(Z.Editors.SolveType.MaterialModel))
        solve = s1.MaterialCell.GetSolveData()._S_MaterialModel
        solve.IndexNd = MODEL_GLASS["nd"]
        solve.AbbeVd = MODEL_GLASS["vd"]
        s1.MaterialCell.SetSolveData(s1.MaterialCell.GetSolveData())
        s1.Comment = "AZ4562 substrate (model glass)"
        self.surf1 = s1

        if self.variant == "hybrid":
            hy = hybrid
            par = lde.InsertNewSurfaceAt(2)
            par.ChangeType(par.GetSurfaceTypeSettings(
                Z.Editors.LDE.SurfaceType.Paraxial))
            par.Thickness = 0.0
            par.Comment = "IDEAL paraxial lens F=BFD (hybrid focusing)"
            par.GetSurfaceCell(Z.Editors.LDE.SurfaceColumn.Par1).DoubleValue \
                = float(d.bfd_mm)
            self.surf_par = par
            plate = lde.InsertNewSurfaceAt(3)
            plate.Thickness = 0.0
            try:
                plate.Material = str(hy["phase_glass"])
            except Exception as exc:
                self.log("  (could not set %s on the plate surface: %s -- "
                         "the wave engines will see NO residual phase)"
                         % (hy["phase_glass"], exc))
            plate.Comment = ("%s plate, zero thickness: index step for the "
                             "residual UDS (lens stays in air)"
                             % hy["phase_glass"])
            self.surf_plate = plate
            uds = lde.InsertNewSurfaceAt(4)
        else:
            uds = lde.InsertNewSurfaceAt(2)
        ut = uds.GetSurfaceTypeSettings(Z.Editors.LDE.SurfaceType.UserDefined)
        ut.Filename = self.dll_name
        uds.ChangeType(ut)
        uds.Thickness = d.bfd_mm
        self.uds = uds

        if self.variant == "zone":
            uds.Comment = "MDL Rings staircase (zone decomposition)"
            # Par 1..4: File #, Height scale, Z sign (+1: relief pokes
            # into the exit space, phase +k(n-1)h as designed), Parax f
            for p, v in ((1, d.file_no), (2, 1.0), (3, 1.0), (4, d.bfd_mm)):
                self.set_par(p, v)
        elif self.variant == "od":
            uds.Comment = "MDL Rings order decomposition"
            # Par 1..6: File #, Order m, Design P, Lam0 um, Add OPL, Use eff
            for p, v in ((1, d.file_no), (2, float(d.orders[-1])),
                         (3, float(d.fold_P)), (4, d.lam0_um), (5, 1.0),
                         (6, 1.0)):
                self.set_par(p, v)
        else:
            uds.Comment = "MDL Rings RESIDUAL phase (Sub ideal=1, hybrid)"
            for p, v in ((1, d.file_no), (2, 1.0), (3, 1.0), (4, d.bfd_mm),
                         (5, 1.0), (7, float(hy["opd_law"])), (9, 1.0),
                         (10, float(hy["dz_offset_mm"]))):
                self.set_par(p, v)
            self.echo_dll_headers()
            self.log("  residual UDS: OPD law %d, dz gain 1, dz offset %.4f "
                     "mm (all intercepts at z > 0)"
                     % (int(hy["opd_law"]), float(hy["dz_offset_mm"])))
        return self

    def set_par(self, num, value):
        col = getattr(self.Z.Editors.LDE.SurfaceColumn, "Par%d" % num)
        self.uds.GetSurfaceCell(col).DoubleValue = float(value)

    def echo_dll_headers(self):
        try:
            hdrs = [str(self.uds.GetSurfaceCell(getattr(
                self.Z.Editors.LDE.SurfaceColumn, "Par%d" % k)).Header)
                for k in range(1, 11)]
            self.log("  DLL parameter headers (from the loaded DLL): %s"
                     % ", ".join(hdrs))
            if "dz offset" not in hdrs:
                self.log("  WARNING: the installed us_mdl_rings.dll predates "
                         "Par 10 'dz offset' -- rebuild and copy it")
        except Exception as exc:
            self.log("  (DLL parameter headers not readable: %s)" % exc)

    @property
    def image_surface(self):
        return self.lde.NumberOfSurfaces - 1

    def save(self, name):
        # ABSOLUTE path: OpticStudio resolves relative paths against ITS
        # OWN working directory
        self.sys.SaveAs(os.path.join(os.path.abspath(self.out_dir), name))

    # --- batch traces ----------------------------------------------------
    def trace_rays(self, wave_idx, py_list, n_max=None):
        """Normalized-pupil rays (px=0) -> list of raw result tuples
        (ok, rn, err, vig, x, y, z, l, m, n, l2, m2, n2, opd, inten)."""
        Z = self.Z
        rt = self.sys.Tools.OpenBatchRayTrace()
        norm = rt.CreateNormUnpol(n_max or (len(py_list) + 8),
                                  Z.Tools.RayTrace.RaysType.Real,
                                  self.image_surface)
        norm.ClearData()
        for py in py_list:
            norm.AddRay(wave_idx, 0.0, 0.0, 0.0, float(py),
                        Z.Tools.RayTrace.OPDMode.Current)
        rt.RunAndWaitForCompletion()
        norm.StartReadingResults()
        out = []
        ok = True
        while ok:
            res = norm.ReadNextResult()
            ok = res[0]
            if ok:
                out.append(res)
        rt.Close()
        return out

    def ray_checks(self):
        """Four rays at the primary line -> ray_checks.txt (sanity)."""
        rows = self.trace_rays(self.d.primary_idx, (0.0, 0.2, 0.5, 0.9))
        path = os.path.join(self.out_dir, "ray_checks.txt")
        with open(path, "w") as fh:
            for (ok, rn, err, vig, x, y, z, l, m, n, l2, m2, n2, opd,
                 inten) in rows:
                fh.write("ray %d err=%d vig=%d  xyz=(%.6f, %.6f, %.6f)"
                         "  opd=%.6f\n" % (rn, err, vig, x, y, z, opd))
        self.log("ray checks -> %s" % path)

    def marginal_y(self, py=0.9):
        """Image-plane y of one marginal real ray at the primary line."""
        rows = self.trace_rays(self.d.primary_idx, (py,), n_max=2)
        if not rows or rows[0][2] != 0:
            return float("nan")
        return float(rows[0][5])

    def trace_opd(self, wave_idx, table):
        """One ray per ring (py = rho_i / R) -> OPD [waves], NaN for
        missing / vignetted rays."""
        rows = self.trace_rays(wave_idx, table.rho_um / table.R_um)
        opd = np.full(table.N, np.nan)
        for res in rows:
            rn, err, vig = res[1], res[2], res[3]
            if 1 <= rn <= table.N and err == 0 and vig == 0:
                opd[rn - 1] = res[13]
        return opd

    def hybrid_focus_check(self, store):
        """Does the Paraxial lens focus at F through the plate and the
        flat UDS (height scale 0)? If the marginal ray misses the axis
        the paraxial f is applied as a slope in glass: set n F. Echoed."""
        self.set_par(2, 0.0)
        y0 = 0.9 * self.d.epd_mm / 2.0
        y_img = self.marginal_y()
        self.log("  hybrid focus check: marginal ray (h=%.3f mm) at the "
                 "image with Paraxial f=%.4f: y=%.4f mm"
                 % (y0, self.d.bfd_mm, y_img))
        if np.isfinite(y_img) and abs(y_img) > 0.002:
            n_est = 1.0 - y_img / y0
            f_new = n_est * self.d.bfd_mm
            self.surf_par.GetSurfaceCell(
                self.Z.Editors.LDE.SurfaceColumn.Par1).DoubleValue = f_new
            y2 = self.marginal_y()
            self.log("  -> Paraxial f applied as a slope in the glass "
                     "(implied n=%.4f); set f=%.4f mm -> y=%.4f mm"
                     % (n_est, f_new, y2))
            self.par_f_mm = f_new
        store["par_f_mm"] = self.par_f_mm
        self.set_par(2, 1.0)


# ---------------------------------------------------------------------------
class AnalysisIO:
    """Version-proof analysis settings and results readers."""

    def __init__(self, session, out_dir, log=print):
        self.s = session
        self.Z = session.ZOSAPI
        self.out_dir = out_dir
        self.log = log

    def new(self, idm_name):
        idm = getattr(self.Z.Analysis.AnalysisIDM, idm_name)
        return self.s.TheSystem.Analyses.New_Analysis(idm)

    @staticmethod
    def typed_settings(an, probe_attr):
        """GetSettings() may hand back the generic IAS_ base interface;
        the derived one is sometimes reachable via __implementation__.
        Never raises: the caller falls back to MODIFYSETTINGS codes."""
        st = an.GetSettings()
        if hasattr(st, probe_attr):
            return st
        impl = getattr(st, "__implementation__", None)
        if impl is not None and hasattr(impl, probe_attr):
            return impl
        return st

    @staticmethod
    def try_set(obj, names, values, verify=True):
        """Set the first attribute of `names` that accepts one of
        `values`; (name, value) or (None, None). Read-back verified so
        a silently ignored assignment is not reported as a success."""
        for nm in names:
            if not hasattr(obj, nm):
                continue
            for val in values:
                try:
                    setattr(obj, nm, val)
                    if verify:
                        got = getattr(obj, nm)
                        if isinstance(val, bool) and bool(got) != val:
                            continue
                    return nm, val
                except Exception:
                    continue
        return None, None

    def modify_settings(self, an, pairs, tag):
        """SaveTo a .cfg, apply ZPL MODIFYSETTINGS codes (locale-safe),
        LoadFrom. Lives on the BASE settings interface of every build."""
        try:
            st = an.GetSettings()
            cfg = os.path.join(self.out_dir, "_%s.cfg" % tag)
            st.SaveTo(cfg)
            for code, val in pairs:
                st.ModifySettings(cfg, code, zloc(val))
            st.LoadFrom(cfg)
            return True
        except Exception as exc:
            self.log("  (ModifySettings route failed: %s)" % exc)
            return False

    def enum_candidates(self, n):
        """Enum values to try for a PSF/MTF sampling property."""
        out = []
        for path, pre in (("Analysis.Settings.Psf.PsfSampling", "PsfS_"),
                          ("Analysis.SampleSizes", "S_"),
                          ("Analysis.Settings.Mtf.HuygensMtfSampling", "S_"),
                          ("Analysis.Settings.HuygensPsfSampling", "S_")):
            try:
                ns = self.Z
                for part in path.split("."):
                    ns = getattr(ns, part)
                out.append(getattr(ns, "%s%dx%d" % (pre, n, n)))
            except Exception:
                continue
        return out

    def grab_grids(self, an):
        """All results DataGrids as [(array, (dx, dy, minx, miny), desc)]
        via the pinned-GCHandle fast copy (bypasses GetTextFile, whose
        export dumped 6.8 GB for an 8192^2 pupil); None when the
        analysis reported invalid / empty results."""
        from System.Runtime.InteropServices import GCHandle, GCHandleType
        res = an.GetResults()
        ng = int(res.NumberOfDataGrids)
        if ng < 1:
            self.log("  (no DataGrids in the results)")
            return None
        grids = []
        for gi in range(ng):
            dg = res.GetDataGrid(gi)
            nx, ny = int(dg.Nx), int(dg.Ny)
            if nx == 0 or ny == 0:
                self.log("  grid %d/%d is EMPTY -- the analysis reported "
                         "invalid results (aborted?)" % (gi, ng))
                return None
            hnd = GCHandle.Alloc(dg.Values, GCHandleType.Pinned)
            try:
                ptr = hnd.AddrOfPinnedObject().ToInt64()
                arr = np.empty((ny, nx), dtype=np.float64)
                ctypes.memmove(arr.ctypes.data, ptr, arr.nbytes)
            finally:
                hnd.Free()
            desc = "%s | %s" % (str(dg.Description), str(dg.ValueLabel))
            grids.append((arr, (float(dg.Dx), float(dg.Dy),
                                float(dg.MinX), float(dg.MinY)), desc))
        return grids

    def grab_series(self, an):
        """All DataSeries as [(x, Y[n, nseries], labels, description)]."""
        res = an.GetResults()
        out = []
        try:
            ns = int(res.NumberOfDataSeries)
        except Exception:
            return out
        for si in range(ns):
            ds = res.GetDataSeries(si)
            try:
                x = np.array(list(ds.XData.Data), dtype=float)
                yd = ds.YData.Data
                nrow, ncol = int(yd.GetLength(0)), int(yd.GetLength(1))
                Y = np.array([[float(yd[i, j]) for j in range(ncol)]
                              for i in range(nrow)])
                labels = [str(v) for v in ds.SeriesLabels] \
                    if hasattr(ds, "SeriesLabels") else []
                out.append((x, Y, labels, str(ds.Description)))
            except Exception as exc:
                self.log("  (DataSeries %d unreadable: %s)" % (si, exc))
        return out

    def save_grids_npz(self, an, out_base):
        """Central 512x512 window of every DataGrid -> compressed npz
        (dx/dy carry the sample spacing). Returns the path or None."""
        try:
            grids = self.grab_grids(an)
            if grids is None:
                return None
            payload = {}
            for gi, (arr, meta, desc) in enumerate(grids):
                ny, nx = arr.shape
                kx, ky = min(nx, 512), min(ny, 512)
                x0, y0 = (nx - kx) // 2, (ny - ky) // 2
                payload["I%d" % gi] = arr[y0:y0 + ky, x0:x0 + kx]
                payload["meta%d" % gi] = np.array(list(meta) + [nx, ny, x0, y0])
                payload["desc%d" % gi] = np.array(desc)
                self.log("  grid %d/%d: %dx%d, dx=%.4g, dy=%.4g, min=%.3g, "
                         "max=%.3g  [%s]" % (gi, len(grids), ny, nx, meta[0],
                                             meta[1], arr.min(), arr.max(),
                                             desc))
            payload["I"] = payload["I0"]
            m0 = payload["meta0"]
            payload.update(dx=m0[0], dy=m0[1], minx=m0[2], miny=m0[3],
                           nx_full=int(m0[4]), ny_full=int(m0[5]),
                           crop_x0=int(m0[6]), crop_y0=int(m0[7]))
            out = out_base + ".npz"
            np.savez_compressed(out, **payload)
            return out
        except Exception as exc:
            self.log("  (DataGrid extraction failed: %s)" % exc)
            return None
