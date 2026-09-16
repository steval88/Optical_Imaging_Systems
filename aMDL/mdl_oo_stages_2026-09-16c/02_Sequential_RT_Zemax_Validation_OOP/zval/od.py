"""od -- ray-based chromatic analyses on the ORDER-DECOMPOSITION DLL,
one diffraction order per pass (Order m = 0 selects the dominant blazed
order per wavelength). Longitudinal aberration / chromatic focal shift
are meaningful here because the od surface bends the rays."""
import os

from .base import Analysis


class OdAnalysis(Analysis):
    MODE = "od"
    VARIANT = "od"
    ZOS_NAME = "mdl_validation_od.zos"
    ANALYSES = (("FocalShiftDiagram", "chromatic_shift"),
                ("LongitudinalAberration", "longitudinal"),
                ("RayFan", "rayfan"))

    def __init__(self, design, orders=None, **kw):
        super().__init__(design, **kw)
        if orders:
            design.orders = [int(o) for o in orders]
        if design.fold_P is None or not design.orders:
            raise SystemExit("od mode needs the fold (P, lam0) and an order "
                             "list: %s" % design.fold_note)

    def settings_record(self):
        return {"orders": list(self.d.orders), "fold_P": self.d.fold_P,
                "lam0_um": self.d.lam0_um}

    def describe(self):
        d = self.d
        self.log("  od fold: P=%d @ lam0=%.2f um; orders %s"
                 % (d.fold_P, d.lam0_um, d.orders))
        if d.fold_note:
            self.log("  (%s)" % d.fold_note)

    def run(self):
        Z, log, zs = self.session.ZOSAPI, self.log, self.zs
        log.section("od: chromatic analyses per order",
                    "FocalShiftDiagram, LongitudinalAberration, RayFan text "
                    "exports per listed order (0 = auto)")
        analyses = []
        for name, fname in self.ANALYSES:
            idm = getattr(Z.Analysis.AnalysisIDM, name, None)
            if idm is None:
                log("  (analysis %s not in this ZOS-API version)" % name)
            else:
                analyses.append((idm, fname))
        for order in self.d.orders:
            zs.set_par(2, float(order))
            tag = "auto" if order == 0 else "m%d" % order
            for idm, fname in analyses:
                try:
                    an = self.session.TheSystem.Analyses.New_Analysis(idm)
                    an.ApplyAndWaitForCompletion()
                    out = os.path.join(self.out_dir, "od_%s_%s.txt" % (fname, tag))
                    an.GetResults().GetTextFile(out)
                    self.session.close_analysis(an)
                    log("order %s: %s -> %s" % (tag, fname, out))
                except Exception as exc:
                    log("order %s: %s FAILED: %s (continuing)" % (tag, fname, exc))
        zs.set_par(2, float(self.d.orders[-1]))
