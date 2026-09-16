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
                tab.use_dll(PROBE_SETTINGS["dll"], -1, 1)
                names = tab.names()
                log("DLL %s: %d parameter labels in slot order:" % (PROBE_SETTINGS["dll"],
                                                                    len(names)))
                for i, n in enumerate(names, 1):
                    log("  slot %2d  %s" % (i, n))
            except SystemExit as exc:
                log("DiffractionTab adapter stopped: %s" % exc)

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
