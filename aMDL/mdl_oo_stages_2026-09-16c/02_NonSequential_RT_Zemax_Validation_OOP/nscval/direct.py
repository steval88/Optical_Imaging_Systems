"""Call a diffraction DLL directly (ctypes), without OpticStudio.

The srg RCWA DLLs are native Win32 libraries exporting two C functions
(OpticStudio help, "Creating a New Diffraction DLL"; diff_samp_1.c):

    int UserDiffraction(double *data)      -> 0 ok, -1 refused
    int UserParamNames(char *data)         data[0] = parameter number (1-based),
                                           data[1] / data[2] = start / stop order;
                                           the name is written back into data

and the newer 2-D flavour (Diff2DSample.cpp):

    int UserDiffraction2D(double *data)
    int UserParamNames2D(char *paramData, double *valData)

The data[] array (help "Data[] values for Bulk Scatter, Diffraction,
Surface Scatter DLLs", Diffraction column; diff_samp_1.c):

    [0]  total number of doubles      [1..3]  x y z of the ray (lens units)
    [4..6]  ray direction cosines     [7..9]  surface normal cosines
    [10] wavelength um                [11] 0 refractive / 1 reflective request
    [12] approach-side index          [13] exit-side index
    [14] order to be traced now       [15] start order   [16] stop order
    [17] mm per lens unit             [18] random seed
    [20..25] incident Ex Ey Ez (re, im)
    [30] relative energy (OUT)        [31] flag (OUT): 1 phase + derivatives,
                                           2 complete output ray data
    [32] phase (OUT)  [33] dphase/dx  [34] dphase/dy   (dimensionless: m lam / P)
    [35..37] output cosines (flag 2)  [40..45] output E field (flag 2)
    [50] number of parameters, [51 + i] parameter i+1         (single value)
    [100..129] data-string block, [130..199] suggested data path (wide chars)
    [200] number of parameters, [201 + 2i] parameter i+1 REFLECT,
                                [202 + 2i] parameter i+1 TRANSMIT

Which of the two parameter blocks a given DLL reads is not documented per
DLL, so the harness fills BOTH with the same values. Everything the DLL
returns is printed raw: return code, energy, flag, phase derivative (its
sign is the DLL's order convention: dphase/dx = m lam / P for lines along
y), output cosines when flag = 2. A DLL that returns -1 for every call
either refuses the configuration or wants a running OpticStudio (license):
the harness says which by also calling Diff2DSample.dll, which has no
such check.

Requires 64-bit Python on Windows (the DLLs are x64). No ZOS-API.
"""
from __future__ import annotations

import ctypes
import math
import os
import platform
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

N_DATA = 300


def dll_dir() -> str:
    """{Documents}\\Zemax\\DLL\\Diffractive (where OpticStudio keeps them)."""
    return os.path.join(os.path.expanduser("~"), "Documents", "Zemax", "DLL", "Diffractive")


class DiffractionDll:
    """One diffraction DLL loaded with ctypes.

    Attributes
    ----------
    path        the file loaded
    lib         the ctypes handle (WinDLL: APIENTRY = __stdcall; on x64 the
                convention is unified so CDLL would do too)
    exports     {name: bool} for the four documented entry points
    fn_diff     UserDiffraction or UserDiffraction2D, whichever exists
    fn_names    UserParamNames / UserParamNames2D, whichever exists
    two_d       True when only the 2-D entry points exist
    """

    ENTRY_POINTS = ("UserDiffraction", "UserParamNames", "UserDiffraction2D", "UserParamNames2D")

    def __init__(self, path: str) -> None:
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if platform.architecture()[0] != "64bit":
            raise SystemExit("this Python is %s; the OpticStudio DLLs are x64 -- use the 64-bit "
                             "conda environment" % platform.architecture()[0])
        loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
        self.lib = loader(path)
        self.exports: Dict[str, bool] = {n: hasattr(self.lib, n) for n in self.ENTRY_POINTS}
        self.two_d = not self.exports["UserDiffraction"] and self.exports["UserDiffraction2D"]
        if self.two_d:
            self.fn_diff = self.lib.UserDiffraction2D
            self.fn_names = self.lib.UserParamNames2D
        elif self.exports["UserDiffraction"]:
            self.fn_diff = self.lib.UserDiffraction
            self.fn_names = self.lib.UserParamNames if self.exports["UserParamNames"] else None
        else:
            raise SystemExit("%s exports none of %s" % (path, self.ENTRY_POINTS))
        self.fn_diff.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.fn_diff.restype = ctypes.c_int
        if self.fn_names is not None:
            if self.two_d:
                self.fn_names.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
            else:
                self.fn_names.argtypes = [ctypes.c_char_p]
            self.fn_names.restype = ctypes.c_int

    # -- parameter names ---------------------------------------------------------------
    def param_names(self, start: int, stop: int, n_max: int = 40,
                    values: Optional[Sequence[float]] = None) -> List[str]:
        """The names the DLL reports for parameters 1..n (stops at the first
        empty name, like OpticStudio). 1-D signature: a char buffer whose
        first three bytes carry (number, start, stop). 2-D signature: the
        same buffer plus a data[] array carrying the orders and values."""
        out: List[str] = []
        if self.fn_names is None:
            return out
        for i in range(1, n_max + 1):
            buf = ctypes.create_string_buffer(512)
            buf[0] = bytes([i & 0xFF])
            buf[1] = bytes([start & 0xFF])
            buf[2] = bytes([stop & 0xFF])
            if self.two_d:
                arr = self._data(0.6, start, stop, 0, values or [], 1.0, 1.0, 0)
                self.fn_names(buf, arr)
            else:
                self.fn_names(buf)
            name = buf.value.decode("latin-1", errors="replace")
            if not name:
                break
            out.append(name)
        return out

    # -- one call -------------------------------------------------------------------------
    def _data(self, lam_um: float, start: int, stop: int, order: int, values: Sequence[float],
              n_in: float, n_out: float, reflect: int, lmn: Tuple[float, float, float] = (0.0, 0.0, 1.0),
              e_field: Tuple[float, float] = (1.0, 0.0), path: str = "") -> Any:
        """The data[] array for one call. 1-D layout: [14] order, [15] start,
        [16] stop, [17] mm/unit, [18] seed. 2-D layout (Diff2DSample.cpp):
        [14] X order, [15] Y order, [16]/[17] X start/stop, [18]/[19] Y
        start/stop, [27] mm/unit, [28] seed, [29] transmit 0 / reflect 1."""
        arr = (ctypes.c_double * N_DATA)()
        arr[0] = float(N_DATA)
        arr[4], arr[5], arr[6] = lmn
        arr[7], arr[8], arr[9] = 0.0, 0.0, 1.0
        arr[10] = float(lam_um)
        arr[11] = float(reflect)
        arr[12], arr[13] = float(n_in), float(n_out)
        if self.two_d:
            arr[14], arr[15] = float(order), 0.0
            arr[16], arr[17], arr[18], arr[19] = float(start), float(stop), 0.0, 0.0
            arr[27], arr[28] = 1.0, 0.5
        else:
            arr[14], arr[15], arr[16] = float(order), float(start), float(stop)
            arr[17], arr[18] = 1.0, 0.5
        arr[20], arr[22] = float(e_field[0]), float(e_field[1])     # Ex re, Ey re
        arr[29] = float(reflect)
        n = len(values)
        arr[50] = float(n)
        arr[200] = float(n)
        for i, v in enumerate(values):
            arr[51 + i] = float(v)
            arr[201 + 2 * i] = float(v)
            arr[202 + 2 * i] = float(v)
        if path:
            # wide-char path in the data[130..199] block (560 bytes), as Diff2DSample reads it
            wide = path.encode("utf-16-le")[:558] + b"\x00\x00"
            ctypes.memmove(ctypes.addressof(arr) + 130 * 8, wide, len(wide))
        return arr

    def call(self, lam_um: float, start: int, stop: int, order: int, values: Sequence[float],
             n_in: float = 1.0, n_out: float = 1.0, reflect: int = 0,
             lmn: Tuple[float, float, float] = (0.0, 0.0, 1.0),
             e_field: Tuple[float, float] = (1.0, 0.0), path: str = "") -> Dict[str, Any]:
        """UserDiffraction for ONE order. Returns the raw answer: rc, energy
        (data[30]), flag (data[31]), phase derivatives (33, 34), output
        cosines (35..37) and the wall time in ms."""
        arr = self._data(lam_um, start, stop, order, values, n_in, n_out, reflect, lmn, e_field, path)
        t0 = time.perf_counter()
        rc = int(self.fn_diff(arr))
        dt = 1e3 * (time.perf_counter() - t0)
        return {"rc": rc, "energy": float(arr[30]), "flag": int(arr[31]), "dpdx": float(arr[33]),
                "dpdy": float(arr[34]), "lmn_out": (float(arr[35]), float(arr[36]), float(arr[37])),
                "ms": dt}

    def efficiencies(self, lam_um: float, orders: Sequence[int], values: Sequence[float],
                     n_in: float = 1.0, n_out: float = 1.0, reflect: int = 0, path: str = "",
                     unpolarized: bool = True) -> Dict[int, Dict[str, Any]]:
        """{order: answer} over `orders` (start/stop = min/max of the list);
        unpolarized = mean of the Ex and Ey answers (the RCWA DLLs are
        polarization-resolved)."""
        out: Dict[int, Dict[str, Any]] = {}
        lo, hi = min(orders), max(orders)
        for m in orders:
            a = self.call(lam_um, lo, hi, m, values, n_in, n_out, reflect, path=path, e_field=(1.0, 0.0))
            if unpolarized:
                b = self.call(lam_um, lo, hi, m, values, n_in, n_out, reflect, path=path, e_field=(0.0, 1.0))
                a = dict(a, energy=0.5 * (a["energy"] + b["energy"]), energy_te=a["energy"],
                         energy_tm=b["energy"], rc=max(a["rc"], b["rc"]) if a["rc"] < 0 or b["rc"] < 0
                         else 0, ms=a["ms"] + b["ms"])
            out[m] = a
        return out


def sinc2(m: float, p: float) -> float:
    """Scalar blaze reference sinc^2(m - p), p = depth (n - 1) / lam in waves."""
    x = m - p
    return 1.0 if abs(x) < 1e-12 else (math.sin(math.pi * x) / (math.pi * x)) ** 2
