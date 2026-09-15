"""zone -- one FFT PSF at the design focus per wavelength on the
staircase surface (rays straight): the coarse energy-localization
check. Pupil sampling must resolve the rings (8192^2 where licensed);
OutputSize decimates the image window, so it is capped small for
display use. Fine PSF shape belongs to rz / huy / RS."""
import os

from .base import Analysis
from .settings import ZONE_SETTINGS
from .zos import samp_index


class ZoneAnalysis(Analysis):
    MODE = "zone"
    VARIANT = "zone"
    ZOS_NAME = "mdl_validation.zos"

    def settings_record(self):
        from .settings import ZONE_SETTINGS
        return {"ZONE_SETTINGS": {k: list(v) if isinstance(v, tuple) else v
                                  for k, v in ZONE_SETTINGS.items()}}

    def describe(self):
        self.log("  zone: FFT PSF per wavelength at the design focus (coarse "
                 "localization; rays exit straight, so ray-based image "
                 "analyses are meaningless on this surface)")

    def configure_fft_psf(self, an, wi):
        """8192^2 pupil (PsfSampling enum first), OutputSize from
        ZONE_SETTINGS, Linear, ImageDelta left at 0 (any smaller value
        ABORTS the analysis on 2024 R1). Returns True when the sampling
        is trustworthy."""
        Z, io = self.session.ZOSAPI, self.io
        st = io.typed_settings(an, "SampleSize")

        def psf_enum(size):
            try:
                return getattr(Z.Analysis.Settings.Psf.PsfSampling, "PsfS_" + size)
            except AttributeError:
                return getattr(Z.Analysis.SampleSizes, "S_" + size)

        samp_ok = out_ok = None
        for size in ("8192x8192", "4096x4096", "2048x2048"):
            try:
                st.SampleSize = psf_enum(size)
                samp_ok = size
                break
            except Exception:
                continue
        for size in ZONE_SETTINGS["out_sizes"]:
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
            st.Type = Z.Analysis.Settings.Psf.FftPsfType.Linear
        except Exception:
            pass
        if samp_ok is None or not wave_ok:
            pairs = []
            if samp_ok is None:
                pairs.append(("PSF_SAMP", samp_index(8192)))
            if not wave_ok:
                pairs.append(("PSF_WAVE", wi))
            ok = io.modify_settings(an, pairs, "fftpsf")
            self.log("  FFT PSF sampling via ModifySettings: %s"
                     % ("ok" if ok else "FAILED -- default sampling cannot "
                        "resolve the rings; treat these PSFs as INVALID"))
            return ok
        self.log("  FFT PSF pupil sampling: S_%s%s"
                 % (samp_ok, "" if out_ok is None
                    else "; output capped at %s" % out_ok))
        return True

    def run(self):
        log = self.log
        log.section("zone: FFT PSF per wavelength",
                    "8192^2 pupil where licensed; central 512^2 window of "
                    "the DataGrid saved per line")
        for wi in range(1, len(self.d.wavelengths_um) + 1):
            an = self.session.TheSystem.Analyses.New_FftPsf()
            self.configure_fft_psf(an, wi)
            try:
                an.ApplyAndWaitForCompletion()
                base = os.path.join(self.out_dir, "fft_psf_w%d" % wi)
                out = self.io.save_grids_npz(an, base)
                if out is None:
                    out = base + ".txt"
                    log("  falling back to GetTextFile -- may be gigabytes")
                    an.GetResults().GetTextFile(out)
                log("fft PSF wavelength %d -> %s" % (wi, out))
            except Exception as exc:
                log("fft PSF wavelength %d FAILED: %s (continuing)" % (wi, exc))
            self.session.close_analysis(an)
        log("next: 'od' for the ray-based chromatic ladder, 'rz' for the "
            "intensity tiles from the Zemax-traced field, 'huy' for the "
            "Huygens PSF/MTF on the hybrid.")
