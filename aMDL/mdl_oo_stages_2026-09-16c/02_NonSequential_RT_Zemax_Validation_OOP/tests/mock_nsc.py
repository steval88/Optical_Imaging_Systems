"""A stand-in for the ZOS-API non-sequential surface, enough to run the
three modes end to end without OpticStudio. The 'ray trace' returns
the SCALAR efficiency of the configured DLL (blaze sinc^2 or staircase
formula, nscval.tea) for the traced order range, so the null test
passes and the ladder reads ratio 1.000 -- a plumbing test, not physics.
Member names are the ones nscval/nsc.py tries FIRST; the probe on the
real build says whether they are right."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nscval import tea                                  # noqa: E402
from nscval.dlls import LABELS, resolve                 # noqa: E402


class Cell:
    def __init__(self) -> None:
        self.DoubleValue = 0.0
        self.IntegerValue = 0


class DiffractionData:
    def __init__(self) -> None:
        self.Split = 0
        self.DLL = ""
        self.StartOrder = 0
        self.StopOrder = 0
        self.NumberOfParameters = 23
        self.t: Dict[int, float] = {}
        self.r: Dict[int, float] = {}

    def GetAvailableDLLs(self):
        return list(LABELS)

    def GetTransmitParamaterName(self, i: int) -> str:   # sic: the API's spelling
        labs = LABELS.get(self.DLL, [])
        return labs[i - 1] if 0 < i <= len(labs) else ""

    def SetTransmitParameterValue(self, i: int, v: float) -> None:
        self.t[i] = v

    def SetReflectParameterValue(self, i: int, v: float) -> None:
        self.r[i] = v

    def GetReflectParameterValue(self, i: int) -> float:
        return self.r[i]

    def GetTransmitParameterValue(self, i: int) -> float:
        return self.t.get(i, 0.0)


class NceObject:
    def __init__(self) -> None:
        self.TypeName = "Null Object"
        self.ZPosition = 0.0
        self.Comment = ""
        self.Material = ""
        self.cells: Dict[str, Cell] = {}
        self.DiffractionData = DiffractionData()

    def GetObjectTypeSettings(self, t: str) -> str:
        return t

    def ChangeType(self, t: str) -> None:
        self.TypeName = t

    def GetObjectCell(self, col: str) -> Cell:
        return self.cells.setdefault(col, Cell())


class NCE:
    def __init__(self, system: "System") -> None:
        self.objs: List[NceObject] = [NceObject()]
        self.system = system

    @property
    def NumberOfObjects(self) -> int:
        return len(self.objs)

    def GetObjectAt(self, i: int) -> NceObject:
        return self.objs[i - 1]

    def InsertNewObjectAt(self, i: int) -> None:
        self.objs.insert(i - 1, NceObject())

    # detector flux from the scalar efficiency of the grating's DLL setup
    def GetDetectorData(self, obj: int, pixel: int, data: int, value: float):
        return True, self.system.scalar_flux()

    def GetAllDetectorDataSafe(self, obj: int, data: int):
        det = self.objs[obj - 1]
        n = det.GetObjectCell("Par3").IntegerValue
        return [self.system.scalar_flux() / (n * n)] * (n * n)


class Wavelengths:
    def __init__(self) -> None:
        self.w = [SimpleNamespace(Wavelength=0.55, Weight=1.0)]

    @property
    def NumberOfWavelengths(self) -> int:
        return len(self.w)

    def RemoveWavelength(self, i: int) -> None:
        self.w.pop(i - 1)

    def GetWavelength(self, i: int):
        return self.w[i - 1]


class Tool:
    def __init__(self) -> None:
        self.SplitNSCRays = self.ScatterNSCRays = self.UsePolarization = False
        self.IgnoreErrors = self.SaveRays = False

    def ClearDetectors(self, n: int) -> None: ...
    def RunAndWaitForCompletion(self) -> None: ...
    def Close(self) -> None: ...


class System:
    def __init__(self) -> None:
        self.NCE = NCE(self)
        self.SystemData = SimpleNamespace(Wavelengths=Wavelengths())
        self.Tools = SimpleNamespace(OpenNSCRayTrace=lambda: Tool())
        self.saved: List[str] = []

    def New(self, b: bool) -> None: ...
    def MakeNonSequential(self) -> None: ...

    def SaveAs(self, p: str) -> None:
        self.saved.append(p)
        open(p, "wb").close()

    def scalar_flux(self) -> float:
        grating = next(o for o in self.NCE.objs if o.TypeName == "DiffractionGrating")
        src = next(o for o in self.NCE.objs if o.TypeName == "SourceEllipse")
        power = src.GetObjectCell("Par3").DoubleValue
        d = grating.DiffractionData
        lam = self.SystemData.Wavelengths.GetWavelength(1).Wavelength
        m = resolve(LABELS[d.DLL], d.DLL)
        P = 1.0 / grating.GetObjectCell("Par10").DoubleValue      # the object's Lines/um
        n = d.t[m["index_grate_r"]]
        n_env = d.t[m["index_env_r"]]
        orders = list(range(d.StartOrder, d.StopOrder + 1))
        if "alpha_deg" in m and "depth_um" not in m:           # blaze
            import math
            depth = P * math.tan(math.radians(d.t[m["alpha_deg"]]))
            p = tea.waves_of_depth(depth, lam, n, n_env)
            eta = tea.blaze_orders(orders, p)
        else:                                                   # staircase
            depth = d.t[m["depth_um"]]
            N = int(round(d.t[m["n_steps"]]))
            p = tea.waves_of_depth(depth, lam, n, n_env)
            eta = tea.staircase_orders(N, p, [abs(o) for o in orders])
        return float(power * eta.sum())


class MockSession:
    """Quacks like zval.zos.ZosSession (app / ZOSAPI / TheSystem / close)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        nce = SimpleNamespace(
            ObjectType=SimpleNamespace(SourceEllipse="SourceEllipse",
                                       DiffractionGrating="DiffractionGrating",
                                       DetectorRectangle="DetectorRectangle"),
            ObjectColumn=SimpleNamespace(**{"Par%d" % i: "Par%d" % i for i in range(1, 31)}),
            DiffractionSplitType=SimpleNamespace(DontSplitByOrder=0, SplitByTable=1,
                                                 SplitByDLL=2))   # real name (probe 2026-09-16)
        self.ZOSAPI = SimpleNamespace(Editors=SimpleNamespace(NCE=nce))
        self.TheSystem = System()
        self.app = SimpleNamespace(ZOSAPI=self.ZOSAPI, TheSystem=self.TheSystem)

    def close(self) -> None: ...
