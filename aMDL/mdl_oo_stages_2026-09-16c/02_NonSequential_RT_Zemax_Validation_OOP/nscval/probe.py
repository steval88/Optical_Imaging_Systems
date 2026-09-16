"""probe -- print what THIS OpticStudio build exposes for the NSC work:
the members of an NCE object, of its diffraction data, of the ray-trace
tool and of the NCE itself; the parameter labels of the diffraction DLL
in slot order; the ObjectType / ObjectColumn enums; and any sample
user_grating_data_xx.txt in the Diffractive folder (the profile-file
format needed for srg_user_defined_RCWA). Nothing is traced."""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .base import NscAnalysis
from .nsc import NscSystem, dll_diffractive_dir
from .settings import PROBE_SETTINGS


def public(obj: Any) -> List[str]:
    return sorted(n for n in dir(obj) if not n.startswith("_"))


class Probe(NscAnalysis):
    MODE = "probe"
    DESCRIPTION = ("ZOS-API member discovery for the NSC stage; run first, paste "
                   "the log")

    def settings_record(self) -> Dict[str, Any]:
        return dict(PROBE_SETTINGS)

    def run(self) -> None:
        log, S = self.log, NscSystem(self.session, self.log)
        Z = S.ZOSAPI
        log.section("NCE object types and columns")
        log("ObjectType: " + ", ".join(public(Z.Editors.NCE.ObjectType)))
        log("ObjectColumn: " + ", ".join(public(Z.Editors.NCE.ObjectColumn)))
        enum = getattr(Z.Editors.NCE, "DiffractionSplitType", None)
        log("DiffractionSplitType: %s" % (", ".join(public(enum)) if enum else
                                          "NOT FOUND under ZOSAPI.Editors.NCE"))

        log.section("a Diffraction Grating object and its diffraction data")
        S.add_source_ellipse(-10.0, 1.0, 1.0, 1000, 20, 1.0)
        obj = S.add_diffraction_grating(0.0, 3.0, 0.02, 1)
        S.add_detector_rect(50.0, 5.0, 5.0, 50)
        log("object members: " + ", ".join(public(obj)))
        for name in ("DiffractionData", "GetDiffractionData", "CoatScatterData",
                     "DrawData", "ObjectData"):
            log("  has %-20s %s" % (name, hasattr(obj, name)))
        d = None
        if hasattr(obj, "DiffractionData"):
            d = obj.DiffractionData
        elif hasattr(obj, "GetDiffractionData"):
            d = obj.GetDiffractionData()
        if d is None:
            log("NO diffraction data member on the object -- paste this log")
        else:
            log("diffraction data members: " + ", ".join(public(d)))
            try:
                from .nsc import DiffractionTab
                tab = DiffractionTab(S, obj)
                log("tab state before: %s" % tab.flags())
                avail = tab.available_dlls()
                log("GetAvailableDLLs (%d): %s" % (len(avail), ", ".join(avail) or "none / no such member"))
                tab.use_dll(PROBE_SETTINGS["dll"], -1, 1)
                log("tab state after use_dll: %s" % tab.flags())
                names = tab.names()
                log("DLL %s: %d parameter labels in slot order:" % (PROBE_SETTINGS["dll"],
                                                                    len(names)))
                for i, n in enumerate(names, 1):
                    log("  slot %2d  %s" % (i, n))
                # write / read back one slot each way (slot 1 = Max Order on the srg DLLs)
                tab.set_slot(1, 12.0)
                log("slot 1 written 12 -> read back (transmit, reflect) = %s" % (tab.get_slot(1),))
                # the labels of every other srg DLL, verbatim, for nscval/dlls.py
                for other in avail:
                    if other.lower().startswith("srg_") and other != PROBE_SETTINGS["dll"]:
                        tab.use_dll(other, 0, 0)
                        labs = tab.names()
                        log("labels of %s (%d): %s" % (other, len(labs),
                                                        " | ".join("%d:%s" % (i + 1, x) for i, x in enumerate(labs))))
            except SystemExit as exc:
                log("DiffractionTab adapter stopped: %s" % exc)
        # editor cell storage types, for the record (integer vs double setters)
        src, det = S.NCE.GetObjectAt(S.objects["source"]), S.NCE.GetObjectAt(S.objects["detector"])
        for label, o, k in (("source Par1 (layout rays)", src, 1), ("source Par2 (analysis rays)", src, 2),
                            ("source Par8 (source distance)", src, 8), ("grating Par10 (lines/um)", obj, 10),
                            ("grating Par11 (diffract order)", obj, 11), ("detector Par3 (# x pixels)", det, 3)):
            cell = o.GetObjectCell(getattr(Z.Editors.NCE.ObjectColumn, "Par%d" % k))
            log("  cell type %-32s %s" % (label, getattr(cell, "DataType", "n/a")))

        log.section("ray trace tool and detector readers")
        tool = S.TheSystem.Tools.OpenNSCRayTrace()
        log("NSC ray trace members: " + ", ".join(public(tool)))
        tool.Close()
        log("NCE members: " + ", ".join(public(S.NCE)))
        for name in ("GetDetectorData", "GetAllDetectorData", "GetAllDetectorDataSafe",
                     "GetDetectorDimensions", "GetDetectorSize"):
            log("  NCE has %-24s %s" % (name, hasattr(S.NCE, name)))
        log("source object %d, grating %d, detector %d" % (S.objects["source"],
                                                           S.objects["grating"],
                                                           S.objects["detector"]))

        log.section("Diffractive DLL folder")
        ddir = dll_diffractive_dir()
        if os.path.isdir(ddir):
            files = sorted(os.listdir(ddir))
            log("%s: %d entries" % (ddir, len(files)))
            log("  DLLs: " + ", ".join(f for f in files if f.lower().endswith(".dll")))
            samples = [f for f in files if f.lower().startswith("user_grating_data")]
            log("  profile files: %s" % (", ".join(samples) if samples else "none"))
            for f in samples[:2]:
                log("  --- %s (first 20 lines) ---" % f)
                with open(os.path.join(ddir, f), errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if i >= 20:
                            break
                        log("  | " + line.rstrip())
        else:
            log("%s not found" % ddir)
        S.save(os.path.join(self.out_dir, "nsc_probe.zos"))
        self.members_found = dict(S.members.found)
        log("saved nsc_probe.zos -- paste this whole log back")
