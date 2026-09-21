"""ZOS-API plumbing for the non-sequential component editor (NCE):
building the three-object test systems, the Diffraction tab of an
object, the ray trace with splitting, and the detector readout.

Every ZOS-API member name used here is looked up through a small
adapter (``Members``) that tries the candidates in order and reports
which one it found, so a name that differs on the installed build shows
up as ONE clear line in the log and the ``probe`` mode prints the real
names to fix the candidate lists. The names below are the ones of the
ZOS-API syntax help of OpticStudio 2021-2024 (NCE examples e02/e04 of
the Python samples); the Diffraction-tab members are the least certain
and are the first thing the probe verifies.

Coordinates: NSC lens units are mm; the beam travels +z; objects are
placed by absolute z (RefObject 0).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

LogFn = Callable[[str], None]


# ---------------------------------------------------------------------------
class Members:
    """Resolve a member of a .NET object from a candidate list."""

    def __init__(self, log: LogFn) -> None:
        self.log = log
        self.found: Dict[str, str] = {}

    def get(self, obj: Any, role: str, candidates: Sequence[str]) -> Any:
        for name in candidates:
            if hasattr(obj, name):
                if role not in self.found:
                    self.found[role] = name
                    if name != candidates[0]:
                        self.log("  [members] %s -> %s" % (role, name))
                return getattr(obj, name)
        avail = [n for n in dir(obj) if not n.startswith("_")]
        raise SystemExit("ZOS-API member for %r not found among %s; the object "
                         "offers: %s  -- run the 'probe' mode and update the "
                         "candidate list in nscval/nsc.py"
                         % (role, list(candidates), ", ".join(avail)))

    def set(self, obj: Any, role: str, candidates: Sequence[str], value: Any) -> str:
        for name in candidates:
            if hasattr(obj, name):
                setattr(obj, name, value)
                self.found.setdefault(role, name)
                return name
        raise SystemExit("cannot set %r on %s (tried %s)" % (role, obj, list(candidates)))


# ---------------------------------------------------------------------------
class NscSystem:
    """A non-sequential OpticStudio system built from Python.

    Attributes
    ----------
    app, ZOSAPI, TheSystem   the session (zval.zos.ZosSession members)
    NCE                      the non-sequential component editor
    members                  the name adapter (its findings go to run_info)
    objects                  {label: object number}
    """

    def __init__(self, session: Any, log: LogFn) -> None:
        self.app = session.app
        self.ZOSAPI = session.ZOSAPI
        self.TheSystem = session.TheSystem
        self.log = log
        self.members = Members(log)
        self.objects: Dict[str, int] = {}
        self.TheSystem.New(False)
        self.TheSystem.MakeNonSequential()
        self.NCE = self.TheSystem.NCE
        self._col = self.ZOSAPI.Editors.NCE.ObjectColumn
        self._otype = self.ZOSAPI.Editors.NCE.ObjectType

    # -- system data -------------------------------------------------------------
    def set_wavelength(self, lam_um: float) -> None:
        """One system wavelength (every trace is monochromatic)."""
        wl = self.TheSystem.SystemData.Wavelengths
        while wl.NumberOfWavelengths > 1:
            wl.RemoveWavelength(wl.NumberOfWavelengths)
        w1 = wl.GetWavelength(1)
        w1.Wavelength = float(lam_um)
        w1.Weight = 1.0

    # -- objects -----------------------------------------------------------------
    def _new_object(self, label: str, type_name: str, z_mm: float,
                    comment: str = "") -> Any:
        n = self.NCE.NumberOfObjects
        # the editor always holds one (empty) object; fill it first
        if n == 1 and self.NCE.GetObjectAt(1).TypeName in ("", "Null Object") \
                and not self.objects:
            obj = self.NCE.GetObjectAt(1)
            num = 1
        else:
            self.NCE.InsertNewObjectAt(n + 1)
            num = n + 1
            obj = self.NCE.GetObjectAt(num)
        settings = obj.GetObjectTypeSettings(getattr(self._otype, type_name))
        obj.ChangeType(settings)
        obj.ZPosition = float(z_mm)
        obj.Comment = comment or label
        self.objects[label] = num
        return obj

    def par(self, obj: Any, k: int, value: float, integer: bool = False) -> None:
        """Object parameter k (1-based ObjectColumn.Par<k>).

        The editor cell carries its own storage type (``cell.DataType``:
        Integer, Double, String, ...) and refuses the other setter with
        'Expected Integer, got Double' (probe of 2026-09-16: the Diffraction
        Grating's Diffract Order, Par 11, is a Double cell although it holds
        an integer). So the cell type wins; `integer` is only the fallback
        when the cell does not report a type (the mock), and the other
        setter is tried when the first one is refused."""
        cell = obj.GetObjectCell(getattr(self._col, "Par%d" % k))
        dtype = str(getattr(cell, "DataType", "") or "")
        if "Integer" in dtype:
            cell.IntegerValue = int(round(value))
            return
        if "Double" in dtype:
            cell.DoubleValue = float(value)
            return
        setters: List[Tuple[str, Union[int, float]]] = [("IntegerValue", int(round(value))),
                                                         ("DoubleValue", float(value))]
        if not integer:
            setters.reverse()
        first, second = setters
        try:
            setattr(cell, first[0], first[1])
        except Exception:                                  # wrong setter for this cell
            setattr(cell, second[0], second[1])

    def add_source_ellipse(self, z_mm: float, half_x_mm: float, half_y_mm: float,
                           rays: int, layout_rays: int, power_w: float) -> Any:
        """Collimated uniform beam along +z: Source Ellipse with Source
        Distance 0 (Par 8). Par 1 layout rays, 2 analysis rays, 3 power
        [W], 4 wavenumber (0 = all), 5 colour, 6/7 half widths [mm]."""
        obj = self._new_object("source", "SourceEllipse", z_mm, "collimated source")
        self.par(obj, 1, layout_rays, integer=True)
        self.par(obj, 2, rays, integer=True)
        self.par(obj, 3, power_w)
        self.par(obj, 4, 0, integer=True)
        self.par(obj, 6, half_x_mm)
        self.par(obj, 7, half_y_mm)
        self.par(obj, 8, 0.0)                 # source distance 0 = collimated
        return obj

    def headers(self, obj: Any, n: int = 12) -> List[str]:
        """The editor's own column headers of Par 1..n for this object
        type (``cell.Header``), '' where the API gives none: the record of
        what each parameter slot means, echoed by the builders."""
        out: List[str] = []
        for k in range(1, n + 1):
            try:
                cell = obj.GetObjectCell(getattr(self._col, "Par%d" % k))
                out.append(str(getattr(cell, "Header", "") or ""))
            except Exception:
                out.append("")
        return out

    def add_diffraction_grating(self, z_mm: float, clear_mm: float,
                                lines_per_um: float, diff_order: int = 1,
                                thickness_mm: float = 1.0, material: str = "") -> Any:
        """Flat Diffraction Grating object -- a LENS-type object (two faces
        and an edge) with the grating on the front face. Editor columns
        (verbatim, zemax_doe_primitives.md 1.1): Par 1 Radius 1, 2 Conic 1,
        3 Clear 1, 4 Edge 1, 5 Thickness, 6 Radius 2, 7 Conic 2, 8 Clear 2,
        9 Edge 2, 10 Lines/um, 11 Diff Order, 12 Formula.

        thickness_mm  the object is a volume: a ZERO thickness makes it a
                      degenerate slab whose rays are lost (null test of
                      2026-09-16: every order read 0.0000). Default 1 mm.
        material      '' = air on both sides, so the flat faces do not
                      refract; the DLL's own Index Grate / Index Env
                      describe the microstructure. A real glass would add
                      Fresnel losses and a refracted order direction."""
        if thickness_mm <= 0.0:
            raise ValueError("Diffraction Grating thickness must be > 0 mm (a zero-thickness "
                             "volume traps every ray)")
        obj = self._new_object("grating", "DiffractionGrating", z_mm,
                               "grating, DLL on face 0")
        self.par(obj, 1, 0.0)
        self.par(obj, 3, clear_mm)
        self.par(obj, 4, clear_mm)
        self.par(obj, 5, thickness_mm)
        self.par(obj, 6, 0.0)
        self.par(obj, 8, clear_mm)
        self.par(obj, 9, clear_mm)
        self.par(obj, 10, lines_per_um)
        self.par(obj, 11, diff_order, integer=True)
        obj.Material = material
        hdr = self.headers(obj, 12)
        if any(hdr):
            self.log("  Diffraction Grating columns: " + " | ".join(
                "%d:%s" % (k + 1, h) for k, h in enumerate(hdr) if h))
        self.log("  Diffraction Grating: clear %.2f mm, thickness %.2f mm, material %r, "
                 "Lines/um %g, Diffract Order %d" % (clear_mm, thickness_mm, material or "",
                                                    lines_per_um, diff_order))
        return obj

    def add_detector_rect(self, z_mm: float, half_x_mm: float, half_y_mm: float,
                          pixels: int, label: str = "detector") -> Any:
        """Detector Rectangle: Par 1/2 half widths [mm], 3/4 pixels."""
        obj = self._new_object(label, "DetectorRectangle", z_mm, label)
        self.par(obj, 1, half_x_mm)
        self.par(obj, 2, half_y_mm)
        self.par(obj, 3, pixels, integer=True)
        self.par(obj, 4, pixels, integer=True)
        return obj

    # -- files -----------------------------------------------------------------
    def save(self, path: str) -> str:
        self.TheSystem.SaveAs(os.path.abspath(path))
        return path


# ---------------------------------------------------------------------------
class DiffractionTab:
    """The Diffraction tab of one object face through the ZOS-API.

    Parameter values are addressed BY INDEX, COUNTED FROM 0 (the DLL's
    UserParamNames labels are cosmetics over numbered slots, see
    zemax_doe_primitives.md 3.4; the 0-based counting was established on
    2026-09-16) and written to BOTH the Reflect and the Transmit column,
    as the srg DLLs require. ``names()`` returns the labels with list
    index = parameter index so a run can echo them next to the values.
    """
    SPLIT_BY_DLL = ("SplitByDLL", "SplitByDLLFunction")   # real name first (probe 2026-09-16)
    # member names as the probe of 2026-09-16 (OpticStudio 2024 R1) lists
    # them on IDiffractionData, first; the manual's spellings follow as
    # fallbacks. Note the API's own typo "GetTransmitParamaterName".
    SPLIT_ATTR = ("Split", "SplitType")
    DLL_ATTR = ("DLL", "DLLName", "DllName")
    NAME_GETTERS = ("GetTransmitParamaterName", "GetTransmitParameterName",
                    "GetReflectParameterName", "GetParameterName", "GetParamName")
    T_SETTERS = ("SetTransmitParameterValue", "SetTransmitValue", "SetTransmitParameter")
    R_SETTERS = ("SetReflectParameterValue", "SetReflectValue", "SetReflectParameter")
    T_GETTERS = ("GetTransmitParameterValue", "GetTransmitValue")
    R_GETTERS = ("GetReflectParameterValue", "GetReflectValue")

    def __init__(self, sysm: NscSystem, obj: Any, face: int = 0) -> None:
        self.sysm = sysm
        self.obj = obj
        self.face = face
        m = sysm.members
        self.data = m.get(obj, "diffraction data", ("DiffractionData", "GetDiffractionData"))
        if callable(self.data):
            self.data = self.data()
        try:
            m.set(self.data, "diffraction face", ("Face",), int(face))
        except SystemExit:
            pass                                    # no Face member (2024 R1): one tab per object

    # -- discovery ---------------------------------------------------------------
    def available_dlls(self) -> List[str]:
        """The DLL names OpticStudio itself offers in the Diffraction tab
        (``GetAvailableDLLs``), or [] when the API has no such list."""
        d = self.data
        if not hasattr(d, "GetAvailableDLLs"):
            return []
        try:
            return [str(x) for x in d.GetAvailableDLLs()]
        except Exception:
            return []

    def resolve_dll(self, dll: str) -> str:
        """The exact string OpticStudio expects for `dll`: matched case-
        insensitively, with or without the .dll extension, against the
        available list; `dll` itself when there is no list to check."""
        avail = self.available_dlls()
        if not avail:
            return dll
        want = dll.lower()
        stem = want[:-4] if want.endswith(".dll") else want
        for name in avail:
            low = name.lower()
            if low == want or low == stem or low == stem + ".dll":
                return name
        raise SystemExit("DLL %r is not among the Diffraction-tab DLLs OpticStudio lists: %s"
                         % (dll, ", ".join(avail)))

    def flags(self) -> Dict[str, Any]:
        """Read-only tab state for the log: DLL, split, orders, parameter
        count and the availability flags, whatever the API exposes."""
        d = self.data
        out: Dict[str, Any] = {}
        for key, cands in (("dll", self.DLL_ATTR), ("split", self.SPLIT_ATTR),
                           ("start_order", ("StartOrder",)), ("stop_order", ("StopOrder",)),
                           ("n_params", ("NumberOfParameters", "NumberOfParams")),
                           ("is_dll_required", ("IsDLLRequired",)),
                           ("is_diffraction_available", ("IsDiffractionAvailable",))):
            for name in cands:
                if hasattr(d, name):
                    try:
                        out[key] = str(getattr(d, name))
                    except Exception as exc:
                        out[key] = "<%s>" % exc
                    break
        return out

    # -- configuration -----------------------------------------------------------
    def use_dll(self, dll: str, start: int, stop: int) -> None:
        m, ZOSAPI, d = self.sysm.members, self.sysm.ZOSAPI, self.data
        enum = getattr(ZOSAPI.Editors.NCE, "DiffractionSplitType", None)
        split = None
        if enum is not None:
            for name in self.SPLIT_BY_DLL:
                split = getattr(enum, name, None)
                if split is not None:
                    break
        if split is None:
            split = 2                               # DontSplit 0, Table 1, DLL 2
        m.set(d, "split type", self.SPLIT_ATTR, split)
        name = self.resolve_dll(dll)
        m.set(d, "dll name", self.DLL_ATTR, name)
        m.set(d, "start order", ("StartOrder",), int(start))
        m.set(d, "stop order", ("StopOrder",), int(stop))
        got = self.flags().get("dll", "")
        if got and got.lower() != name.lower():
            raise SystemExit("Diffraction DLL did not take: set %r, tab reads %r" % (name, got))

    def set_split_none(self) -> None:
        """Split = DontSplitByOrder: the object's own Diffract Order sets
        the single ray direction, no DLL, no energy split (geometric check)."""
        m, ZOSAPI, d = self.sysm.members, self.sysm.ZOSAPI, self.data
        enum = getattr(ZOSAPI.Editors.NCE, "DiffractionSplitType", None)
        none = getattr(enum, "DontSplitByOrder", 0) if enum is not None else 0
        m.set(d, "split type", self.SPLIT_ATTR, none)

    def set_orders(self, start: int, stop: int) -> None:
        self.sysm.members.set(self.data, "start order", ("StartOrder",), int(start))
        self.sysm.members.set(self.data, "stop order", ("StopOrder",), int(stop))

    def n_params(self) -> int:
        d = self.data
        for name in ("NumberOfParameters", "NumberOfParams", "ParameterCount"):
            if hasattr(d, name):
                return int(getattr(d, name))
        return 40                                   # probe by name errors

    def names(self) -> List[str]:
        """The DLL's parameter labels, list index = PARAMETER INDEX, which the
        ZOS-API counts FROM 0 (diag of 2026-09-16: a Split-by-Table with
        orders 0..1 took its values at indices 0 and 1, and the 1-based
        listing of every DLL showed a blank last entry -- index 0 had never
        been read or written; for the srg DLLs it is the period)."""
        d = self.data
        getter = None
        for name in self.NAME_GETTERS:
            if hasattr(d, name):
                getter = getattr(d, name)
                break
        if getter is None:
            return []
        out: List[str] = []
        for i in range(0, self.n_params()):
            try:
                out.append(str(getter(i)))
            except Exception:
                break
        return out

    def _call_first(self, cands: Sequence[str], *args: Any) -> Optional[Any]:
        d = self.data
        for name in cands:
            if hasattr(d, name):
                return getattr(d, name)(*args)
        return None

    def get_slot(self, slot: int) -> Tuple[Optional[float], Optional[float]]:
        """(transmit, reflect) value of one parameter (0-based index), None if unreadable."""
        t = self._call_first(self.T_GETTERS, int(slot))
        r = self._call_first(self.R_GETTERS, int(slot))
        return (None if t is None else float(t), None if r is None else float(r))

    def set_slot(self, slot: int, value: float) -> None:
        """Write one parameter (0-based index) to Transmit AND Reflect."""
        d = self.data
        done = 0
        for cands in (self.T_SETTERS, self.R_SETTERS):
            for setter in cands:
                if hasattr(d, setter):
                    getattr(d, setter)(int(slot), float(value))
                    done += 1
                    break
        if done == 0:
            self.sysm.members.get(d, "diffraction parameter setter",
                                  (self.T_SETTERS[0],))   # raises with dir()

    def set_slot_one(self, slot: int, value: float, which: str = "T") -> bool:
        """Write one parameter (0-based index) to ONE side only: which = "T"
        (SetTransmitParameterValue) or "R" (SetReflectParameterValue).
        Returns False when that setter does not exist. Used by the diag to
        tell a normalised T/R double write (0.5) from a real efficiency."""
        d = self.data
        for setter in (self.T_SETTERS if which.upper() == "T" else self.R_SETTERS):
            if hasattr(d, setter):
                getattr(d, setter)(int(slot), float(value))
                return True
        return False

    def set_slots(self, values: Dict[int, float]) -> None:
        for k in sorted(values):
            self.set_slot(k, values[k])


# ---------------------------------------------------------------------------
class NscTrace:
    """One non-sequential ray trace and the detector readout."""

    def __init__(self, sysm: NscSystem, split: bool, scatter: bool,
                 polarization: bool, ignore_errors: bool,
                 save_rays_file: Optional[str] = None) -> None:
        self.sysm = sysm
        self.split, self.scatter = split, scatter
        self.polarization, self.ignore_errors = polarization, ignore_errors
        #: ZRD file name (relative to the saved .zos) when the ray database is wanted
        self.save_rays_file: Optional[str] = save_rays_file

    last: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """One trace. Returns (and keeps in ``last``) the tool's own
        report: succeeded, error message, total ray energy [W] launched
        (``GetTotalRayEnergy``), when the build exposes them."""
        S = self.sysm
        tool = S.TheSystem.Tools.OpenNSCRayTrace()
        tool.SplitNSCRays = bool(self.split)
        tool.ScatterNSCRays = bool(self.scatter)
        tool.UsePolarization = bool(self.polarization)
        tool.IgnoreErrors = bool(self.ignore_errors)
        if self.save_rays_file:
            tool.SaveRays = True
            tool.SaveRaysFile = self.save_rays_file
        else:
            tool.SaveRays = False
        tool.ClearDetectors(0)
        run_tool_interruptible(tool)
        rep: Dict[str, Any] = {}
        for key, name in (("succeeded", "Succeeded"), ("error", "ErrorMessage")):
            if hasattr(tool, name):
                try:
                    rep[key] = getattr(tool, name)
                except Exception as exc:
                    rep[key] = "<%s>" % exc
        if hasattr(tool, "GetTotalRayEnergy"):
            try:
                rep["total_ray_energy_w"] = float(tool.GetTotalRayEnergy())
            except Exception as exc:
                rep["total_ray_energy_w"] = "<%s>" % exc
        tool.Close()
        self.last = rep
        return rep

    def report(self) -> str:
        """One-line summary of the last trace for the log."""
        r = self.last
        return "trace: succeeded %s, error %r, launched %s W" % (
            r.get("succeeded", "?"), r.get("error", "") or "", r.get("total_ray_energy_w", "?"))

    def detector_total(self, obj_number: int) -> float:
        """Total flux [W] on a detector object (NSDD pixel 0 = sum of all
        pixels, data 0 = flux)."""
        NCE = self.sysm.NCE
        res = NCE.GetDetectorData(int(obj_number), 0, 0, 0.0)
        # pythonnet returns (ok, value) for an 'out' argument
        if isinstance(res, tuple):
            ok, value = res[0], res[-1]
            if not ok:
                raise SystemExit("GetDetectorData failed on object %d" % obj_number)
            return float(value)
        return float(res)

    def detector_total_pixels(self, obj_number: int) -> Optional[float]:
        """The same total from the per-pixel bulk reader, as a cross-check
        of the pixel-0 convention (None when the reader is absent)."""
        NCE = self.sysm.NCE
        for name in ("GetAllDetectorDataSafe", "GetAllDetectorData"):
            if hasattr(NCE, name):
                try:
                    return float(np.sum(np.asarray(list(getattr(NCE, name)(int(obj_number), 0)),
                                                   dtype=float)))
                except Exception:
                    continue
        return None

    def detector_map(self, obj_number: int, pixels: int) -> Optional[np.ndarray]:
        """Flux per pixel as a (pixels, pixels) array, or None if the
        bulk reader is absent on this build."""
        NCE = self.sysm.NCE
        for name in ("GetAllDetectorDataSafe", "GetAllDetectorData"):
            if hasattr(NCE, name):
                try:
                    arr = np.asarray(list(getattr(NCE, name)(int(obj_number), 0)),
                                     dtype=float)
                    return arr.reshape(pixels, pixels)
                except Exception:
                    continue
        return None


def dll_diffractive_dir() -> str:
    """{Documents}\\Zemax\\DLL\\Diffractive -- the diffraction DLLs and the
    user_grating_data_xx.txt profile files."""
    return os.path.join(os.path.expanduser("~"), "Documents", "Zemax",
                        "DLL", "Diffractive")


class DllLog:
    """The srg DLLs' own log (Test Mode = 1): DLL\\Diffractive\\<dll>_log.txt,
    one block per RCWA call with the input echo and, when the energy
    balance misses the DLL's tolerance, "Error: Power conservation.(error
    = -0.20 %)" -- the ray is then refused (2026-09-18). `mark()` before a
    trace, `since()` after: the new lines and a one-line summary.

    Attributes
    ----------
    path     the log file for this DLL ('srg_step_RCWA.dll' -> srg_step_RCWA_log.txt)
    start    byte offset recorded by mark()
    """

    def __init__(self, dll: str) -> None:
        stem = os.path.splitext(os.path.basename(dll))[0]
        self.path = os.path.join(dll_diffractive_dir(), stem + "_log.txt")
        self.start = 0

    def mark(self) -> int:
        try:
            self.start = os.path.getsize(self.path)
        except OSError:
            self.start = 0
        return self.start

    def since(self) -> Tuple[str, List[str]]:
        try:
            with open(self.path, errors="replace") as fh:
                fh.seek(self.start)
                new = fh.read()
        except OSError:
            return "log not readable", []
        lines = [ln.rstrip() for ln in new.splitlines() if ln.strip()]
        cons = [ln for ln in lines if "Power conservation" in ln]
        other = [ln for ln in lines if "Error" in ln and "Power conservation" not in ln]
        if cons:
            summary = "%d conservation errors, first: %s" % (len(cons), cons[0].split("]")[-1].strip())
        else:
            summary = "no conservation error; %d new lines" % len(lines)
        if other:
            summary += "; other: " + other[0].strip()
        return summary, lines


def order_geometry(lam_um: float, period_um: float, orders: Sequence[int],
                   z_mm: float) -> List[Tuple[int, float, float]]:
    """(m, sin theta_m, x_mm at the detector) for a normal-incidence
    transmission grating in air: sin theta_m = m lam / P."""
    out = []
    for m in orders:
        s = m * lam_um / period_um
        x = z_mm * s / np.sqrt(max(1.0 - s * s, 1e-12)) if abs(s) < 1 else float("nan")
        out.append((int(m), float(s), float(x)))
    return out


def nsc_settings(TheSystem: Any) -> Dict[str, Any]:
    """The system's non-sequential trace settings (SystemData.NonSequentialData):
    ray-intensity cut-offs and splitting flags that can silently drop split
    children. Whatever members the build exposes."""
    out: Dict[str, Any] = {}
    try:
        nsd = TheSystem.SystemData.NonSequentialData
    except Exception as exc:
        return {"error": str(exc)}
    for name in ("MaximumIntersectionsPerRay", "MaximumSegmentsPerRay", "MaximumNestedObjects",
                 "MinimumRelativeRayIntensity", "MinimumAbsoluteRayIntensity", "GlueDistance",
                 "MissRayDrawDistance", "MaximumSourceFileRays", "SimpleRaySplitting",
                 "RetraceSourceRays", "SplitRays", "ScatterRays"):
        if hasattr(nsd, name):
            try:
                out[name] = getattr(nsd, name)
            except Exception as exc:
                out[name] = "<%s>" % exc
    return out


def _enum_placeholder(ZOSAPI: Any) -> Any:
    """A RaysSegmentStatus value to pass as the dummy for the enum out-parameter."""
    for path in (("Tools", "RayTrace", "RaysSegmentStatus"), ("Tools", "RayTrace", "RaySegmentStatus"),
                 ("Tools", "RayTrace", "RayStatus")):
        obj: Any = ZOSAPI
        try:
            for a in path:
                obj = getattr(obj, a)
            for member in ("Terminated", "Reflected", "Transmitted"):
                if hasattr(obj, member):
                    return getattr(obj, member)
        except Exception:
            continue
    return 0


def run_tool_interruptible(tool: Any, poll_s: float = 0.25) -> None:
    """Run a ZOS-API tool so that Ctrl+C works.

    RunAndWaitForCompletion blocks inside .NET and pythonnet delivers the
    KeyboardInterrupt only when the call returns -- with the srg RCWA at
    17 ms per ray a trace of 20 000 rays ignores Ctrl+C for six minutes
    (2026-09-16). So: Run() (asynchronous), poll IsRunning from Python
    where the interrupt is delivered, and on Ctrl+C Cancel() the tool,
    wait for it to stop and re-raise. Tools without Run() (the mock) fall
    back to the blocking call."""
    if not (hasattr(tool, "Run") and hasattr(tool, "IsRunning")):
        tool.RunAndWaitForCompletion()
        return
    import time as _time
    tool.Run()
    try:
        while bool(tool.IsRunning):
            _time.sleep(poll_s)
    except KeyboardInterrupt:
        try:
            tool.Cancel()
            if hasattr(tool, "WaitForCompletion"):
                tool.WaitForCompletion()
        except Exception:
            pass
        raise
    if hasattr(tool, "WaitForCompletion"):
        tool.WaitForCompletion()


def zrd_raw(TheSystem: Any, zrd_path: str, ZOSAPI: Any = None, max_rays: int = 1000
            ) -> List[Tuple[int, float, List[Tuple[Any, ...]]]]:
    """[(ray_no, wavelength_um, [segment tuples])] of a ray database, the
    segments as the raw out-parameter tuples of ReadNextSegmentFull (29
    fields on 2024 R1: level, parent, hit_obj, hit_face, inside_of, status,
    x, y, z, l, m, n, exr, exi, eyr, eyi, ezr, ezi, intensity, path_length,
    xybin, lmbin, xNorm, yNorm, zNorm, index, phase0, phase_of, phase_at).
    Empty list when the reader is unavailable."""
    out: List[Tuple[int, float, List[Tuple[Any, ...]]]] = []
    try:
        rd = TheSystem.Tools.OpenRayDatabaseReader()
        rd.ZRDFile = zrd_path
        rd.RunAndWaitForCompletion()
        if hasattr(rd, "Succeeded") and not rd.Succeeded:
            return out
        res_obj = rd.GetResults() if hasattr(rd, "GetResults") else rd
        enum_dummy = _enum_placeholder(ZOSAPI) if ZOSAPI is not None else 0
        for _ in range(max_rays):
            try:
                res = res_obj.ReadNextResult()
            except TypeError:
                res = res_obj.ReadNextResult(0, 0, 0.0, 0)
            if not isinstance(res, tuple) or not res[0]:
                break
            segs: List[Tuple[Any, ...]] = []
            for _s in range(int(res[4])):
                try:
                    sr = res_obj.ReadNextSegmentFull()
                except TypeError:
                    sr = res_obj.ReadNextSegmentFull(*([0] * 5 + [enum_dummy] + [0.0] * 14))
                if isinstance(sr, tuple) and sr and not (isinstance(sr[0], str) and sr[0] == "FAILED"):
                    segs.append(tuple(sr[1:]))          # drop the leading success flag
            out.append((int(res[1]), float(res[3]), segs))
        try:
            rd.Close()
        except Exception:
            pass
    except Exception:
        return out
    return out


def order_histogram(rays: Sequence[Tuple[int, float, List[Tuple[Any, ...]]]], period_um: float,
                    detector_obj: int) -> Tuple[Dict[int, float], float, int, int]:
    """From `zrd_raw` output: {order m: summed intensity} of the segments
    that hit `detector_obj`, m = round(l * P / lam) (sin theta = m lam / P
    in air), plus the summed launched intensity, the number of parents and
    the number of child segments (the split multiplicity)."""
    hist: Dict[int, float] = {}
    launched = 0.0
    n_parent = n_child = 0
    for _ray, lam, segs in rays:
        for sg in segs:
            if len(sg) < 20:
                continue
            level, hit_obj, l_cos, inten = int(sg[0]), int(sg[2]), float(sg[9]), float(sg[18])
            if level == 0:
                launched += inten
                n_parent += 1
            else:
                n_child += 1
            if hit_obj == detector_obj:
                m = int(round(l_cos * period_um / lam))
                hist[m] = hist.get(m, 0.0) + inten
    return hist, launched, n_parent, n_child


def read_zrd(TheSystem: Any, zrd_path: str, max_rays: int = 5, max_segments: int = 12,
             ZOSAPI: Any = None) -> List[str]:
    """Read a ray database with the ZOS-API ZRD reader and return one text
    line per segment of the first `max_rays` rays: level, parent, object
    hit, face, inside-of, status flags, position, direction, intensity.

    IZRDReader.GetResults() -> IZRDReaderResults with ReadNextResult /
    ReadNextSegmentFull (probe 2026-09-16). The out-parameters come back
    from pythonnet as a tuple; ReadNextSegmentFull's sixth is an ENUM, so
    the call is tried argument-free first (pythonnet allows omitting
    out-parameters), then with an enum placeholder, then the shorter
    ReadNextSegment. Fields are printed by name when the count matches the
    documented signature and raw otherwise."""
    lines: List[str] = []
    try:
        rd = TheSystem.Tools.OpenRayDatabaseReader()
    except Exception as exc:
        return ["ZRD reader not available: %s" % exc]
    try:
        rd.ZRDFile = zrd_path
        rd.RunAndWaitForCompletion()
        if hasattr(rd, "Succeeded") and not rd.Succeeded:
            return ["ZRD reader failed on %s: %s" % (zrd_path, getattr(rd, "ErrorMessage", ""))]
        res_obj = rd.GetResults() if hasattr(rd, "GetResults") else rd
        seg_names = ["level", "parent", "hit_obj", "hit_face", "inside_of", "status",
                     "x", "y", "z", "l", "m", "n", "exr", "exi", "eyr", "eyi", "ezr", "ezi",
                     "intensity", "path_length"]
        enum_dummy = _enum_placeholder(ZOSAPI) if ZOSAPI is not None else 0

        def read_segment() -> Any:
            attempts = [
                ("ReadNextSegmentFull", ()),
                ("ReadNextSegmentFull", tuple([0] * 5 + [enum_dummy] + [0.0] * 14)),
                ("ReadNextSegment", ()),
                ("ReadNextSegment", tuple([0] * 5 + [enum_dummy] + [0.0] * 7)),
            ]
            last: Any = None
            for name, args in attempts:
                if not hasattr(res_obj, name):
                    continue
                try:
                    return getattr(res_obj, name)(*args)
                except TypeError as exc:
                    last = "%s%s: %s" % (name, "()" if not args else "(dummies)", exc)
            return ("FAILED", last)

        for iray in range(max_rays):
            try:
                res = res_obj.ReadNextResult()
            except TypeError:
                res = res_obj.ReadNextResult(0, 0, 0.0, 0)
            if not isinstance(res, tuple) or not res[0]:
                lines.append("ray %d: no more results (%r)" % (iray + 1, res))
                break
            ray_no, wave_idx, wl_um, n_seg = res[1], res[2], res[3], res[4]
            lines.append("ray %d (wave %s, %.4f um): %d segments" % (ray_no, wave_idx, wl_um, n_seg))
            for iseg in range(int(n_seg)):
                sr = read_segment()
                if isinstance(sr, tuple) and sr and sr[0] == "FAILED":
                    lines.append("   segment read failed: %s" % sr[1])
                    return lines
                if not isinstance(sr, tuple) or not sr[0]:
                    lines.append("   segment %d: read returned %r" % (iseg, sr))
                    break
                if iseg >= max_segments:
                    continue
                vals = list(sr[1:])
                if len(vals) == len(seg_names):
                    d = dict(zip(seg_names, vals))
                    lines.append("   seg %2d: level %s parent %s obj %s face %s inside %s status %s | "
                                 "xyz (%.3f, %.3f, %.3f) lmn (%.4f, %.4f, %.4f) | I %.4g"
                                 % (iseg, d["level"], d["parent"], d["hit_obj"], d["hit_face"],
                                    d["inside_of"], d["status"], d["x"], d["y"], d["z"],
                                    d["l"], d["m"], d["n"], d["intensity"]))
                else:
                    lines.append("   seg %2d raw (%d fields): %s" % (iseg, len(vals), vals))
    except Exception as exc:
        lines.append("ZRD read stopped: %s: %s" % (type(exc).__name__, exc))
    finally:
        try:
            rd.Close()
        except Exception:
            pass
    return lines


def source_polarization(obj: Any) -> Dict[str, Any]:
    """The Sources-tab polarization data of a source object (SourcesData):
    whatever members the build exposes (IsPolarized / Jx / Jy / XPhase /
    YPhase / RandomPolarization ...)."""
    out: Dict[str, Any] = {}
    try:
        sd = obj.SourcesData
    except Exception as exc:
        return {"error": str(exc)}
    for name in dir(sd):
        if name.startswith("_") or name.startswith(("get_", "set_")) or name in ("Equals", "GetHashCode",
                                                                                "GetType", "ToString"):
            continue
        try:
            v = getattr(sd, name)
            if callable(v):
                continue
            out[name] = v
        except Exception as exc:
            out[name] = "<%s>" % exc
    return out


def describe_object(NCE: Any, ZOSAPI: Any, num: int, n_par: int = 12) -> List[str]:
    """One text block on object `num` of an open NSC system: type, comment,
    position, material, Par 1..n with headers, the Diffraction tab (split,
    DLL, orders, parameter values) and the Coat/Scatter face data when the
    build exposes them. Used to compare a working sample file with ours."""
    out: List[str] = []
    try:
        o = NCE.GetObjectAt(num)
    except Exception as exc:
        return ["object %d: %s" % (num, exc)]
    col = ZOSAPI.Editors.NCE.ObjectColumn
    out.append("object %d: %s '%s' at z %.3f mm, material %r" % (
        num, getattr(o, "TypeName", "?"), getattr(o, "Comment", ""), float(getattr(o, "ZPosition", 0.0)),
        getattr(o, "Material", "")))
    pars = []
    for k in range(1, n_par + 1):
        try:
            c = o.GetObjectCell(getattr(col, "Par%d" % k))
            hdr = str(getattr(c, "Header", "") or "")
            val = c.IntegerValue if "Integer" in str(getattr(c, "DataType", "")) else c.DoubleValue
            if hdr:
                pars.append("%d:%s=%g" % (k, hdr, val))
        except Exception:
            break
    out.append("   params: " + " | ".join(pars))
    try:
        d = o.DiffractionData
        st = {}
        for name in ("Split", "DLL", "StartOrder", "StopOrder", "NumberOfParameters", "IsDLLRequired"):
            if hasattr(d, name):
                st[name] = str(getattr(d, name))
        out.append("   diffraction: %s" % st)
        n = int(getattr(d, "NumberOfParameters", 0) or 0)
        vals = []
        for i in range(0, n):
            try:
                lab = d.GetTransmitParamaterName(i) if hasattr(d, "GetTransmitParamaterName") else str(i)
                vals.append("[%d]%s=%g/%g" % (i, lab, d.GetTransmitParameterValue(i), d.GetReflectParameterValue(i)))
            except Exception:
                break
        if vals:
            out.append("   diffraction params (T/R): " + " | ".join(vals))
    except Exception as exc:
        out.append("   diffraction: <%s>" % exc)
    try:
        cs = o.CoatScatterData
        face_lines = []
        for f in range(0, 3):
            try:
                fd = cs.GetFaceData(f)
                face_lines.append("face %d: coating %r, scatter %s" % (
                    f, getattr(fd, "Coating", ""), getattr(fd, "ScatterModel", "?")))
            except Exception:
                break
        if face_lines:
            out.append("   coat/scatter: " + " | ".join(face_lines))
        for name in ("NumberOfFaces", "FaceNames"):
            if hasattr(cs, name):
                out.append("   %s: %s" % (name, getattr(cs, name)))
    except Exception as exc:
        out.append("   coat/scatter: <%s>" % exc)
    return out

