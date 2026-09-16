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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
        """Object parameter k (1-based ObjectColumn.Par<k>)."""
        cell = obj.GetObjectCell(getattr(self._col, "Par%d" % k))
        if integer:
            cell.IntegerValue = int(value)
        else:
            cell.DoubleValue = float(value)

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

    def add_diffraction_grating(self, z_mm: float, clear_mm: float,
                                lines_per_um: float, diff_order: int = 1,
                                thickness_mm: float = 0.0, material: str = "") -> Any:
        """Flat Diffraction Grating object. Editor columns (verbatim,
        zemax_doe_primitives.md 1.1): Par 1 Radius 1, 2 Conic 1, 3 Clear 1,
        4 Edge 1, 5 Thickness, 6 Radius 2, 7 Conic 2, 8 Clear 2, 9 Edge 2,
        10 Lines/um, 11 Diff Order, 12 Formula. Material '' = air: the
        DLL's own Index Grate / Index Env describe the microstructure."""
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

    Parameter values are addressed BY SLOT INDEX (the DLL's
    UserParamNames labels are cosmetics over numbered slots, see
    zemax_doe_primitives.md 3.4) and written to BOTH the Reflect and the
    Transmit column, as the srg DLLs require. ``names()`` returns the
    labels in slot order so a run can echo them next to the values.
    """
    SPLIT_BY_DLL = ("SplitByDLLFunction", "SplitByDLL")

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
            pass                                    # single-face objects

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
        m.set(d, "split type", ("SplitType", "Split"), split)
        m.set(d, "dll name", ("DLL", "DLLName", "DllName"), dll)
        m.set(d, "start order", ("StartOrder",), int(start))
        m.set(d, "stop order", ("StopOrder",), int(stop))

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
        d = self.data
        getter = None
        for name in ("GetParameterName", "GetParamName", "ParameterName"):
            if hasattr(d, name):
                getter = getattr(d, name)
                break
        if getter is None:
            return []
        out: List[str] = []
        for i in range(1, self.n_params() + 1):
            try:
                out.append(str(getter(i)))
            except Exception:
                break
        return out

    def set_slot(self, slot: int, value: float) -> None:
        """Write one parameter slot (1-based) to Reflect AND Transmit."""
        d = self.data
        done = 0
        for setter in ("SetTransmitValue", "SetTransmitParameter", "SetTransmitParam"):
            if hasattr(d, setter):
                getattr(d, setter)(int(slot), float(value))
                done += 1
                break
        for setter in ("SetReflectValue", "SetReflectParameter", "SetReflectParam"):
            if hasattr(d, setter):
                getattr(d, setter)(int(slot), float(value))
                done += 1
                break
        if done == 0:
            # generic accessor pattern: SetParameterValue(index, isTransmit, value)?
            for setter in ("SetParameterValue", "SetValue"):
                if hasattr(d, setter):
                    getattr(d, setter)(int(slot), float(value))
                    done = 1
                    break
        if done == 0:
            self.sysm.members.get(d, "diffraction parameter setter",
                                  ("SetTransmitValue",))   # raises with dir()

    def set_slots(self, values: Dict[int, float]) -> None:
        for k in sorted(values):
            self.set_slot(k, values[k])


# ---------------------------------------------------------------------------
class NscTrace:
    """One non-sequential ray trace and the detector readout."""

    def __init__(self, sysm: NscSystem, split: bool, scatter: bool,
                 polarization: bool, ignore_errors: bool) -> None:
        self.sysm = sysm
        self.split, self.scatter = split, scatter
        self.polarization, self.ignore_errors = polarization, ignore_errors

    def run(self) -> None:
        S = self.sysm
        tool = S.TheSystem.Tools.OpenNSCRayTrace()
        tool.SplitNSCRays = bool(self.split)
        tool.ScatterNSCRays = bool(self.scatter)
        tool.UsePolarization = bool(self.polarization)
        tool.IgnoreErrors = bool(self.ignore_errors)
        tool.SaveRays = False
        tool.ClearDetectors(0)
        tool.RunAndWaitForCompletion()
        tool.Close()

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
