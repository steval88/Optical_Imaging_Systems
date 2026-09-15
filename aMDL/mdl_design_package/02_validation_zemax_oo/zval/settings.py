"""Every knob of the validation INSTRUMENT in one place (the design
geometry comes from the run folder, never from here). Each dict is
echoed by the analysis that uses it; the CLI overrides are applied to
copies, so a default is never silently different from what is printed.
"""

# same Cauchy as mdl_core.n_az4562 AND as hardcoded in both DLLs --
# keep the three in sync if the material ever changes
def n_resist(lam_um):
    return 1.594 + 0.01152 / lam_um ** 2


SUBSTRATE_MM = 1.1          # surface 1 thickness (AZ4562 model glass)
MODEL_GLASS = dict(nd=1.6274, vd=30.6)   # Cauchy fit at 0.5876 um

# Huygens PSF / MTF on the HYBRID system (mode 'huy', 2026-09-07/08).
# MEASURED 2026-09-07: 1024^2 pupil x 256^2 image took 450-520 s per
# Huygens PSF and ~200 s per Huygens MTF (cost ~ pupil^2 x image^2);
# with UsePolarization=True every PSF grid came back ALL ZERO although
# the settings were accepted. 512^2 pupil = 20 um ray pitch (the cell
# mean holds to corr 1.0000 even at 142 um), 128^2 image at 0.4 um
# (+-26 um), polarization OFF (retried the other way once if a grid is
# all zero). The Huygens engine applies rel_surf_tran even with
# polarization off (2026-09-08), so the reference is phase+transmission.
HUY_SETTINGS = {
    "pupil_samp": 512,        # rays across the pupil: pitch EPD/N -> Avg cell
    "image_samp": 128,        # image grid per side: 128 x 0.4 um = +-26 um
    "image_delta_um": 0.4,    # image pitch [um] (Huygens 'Image Delta')
    "max_freq_lpmm": 600.0,   # Huygens MTF frequency range [cycles/mm]
    "use_polarization": False,
    "lams": "verify",         # 'verify' = the 5 representative lines,
                              # 'all' = every verification line,
                              # 'primary', or a comma list in um
}

# The HYBRID system hosting the residual UDS (huy mode).
HYBRID_SETTINGS = {
    # Medium of the zero-thickness plate between the Paraxial lens
    # (kept in AIR) and the residual UDS. MEASURED 2026-09-07/08: the
    # wave engines take a UDS phase from the PHYSICAL optical path
    # (n1-n2) dz of the displaced intercept, so the DLL needs an index
    # step; glass directly behind the Paraxial surface adds spherical
    # aberration rho^4/(8F^3)(1-1/n'^2) = 0.37 um at the rim, a
    # zero-thickness plate does not. Any catalog glass works.
    "phase_glass": "N-BK7",
    # Constant added to the residual displacement (DLL Par 10) so that
    # no intercept lies behind the vertex: exceeds lam/(2(n1-n2)) =
    # 1.08 um at 1.1 um for N-BK7. Constant phase only.
    "dz_offset_mm": 0.0015,
    "opd_law": 1,             # DLL Par 7: 1 = (n1-n2) dz physical path
    "r_max_um": 20.0,         # radial comparison window (= rs tiles)
    "rsf_step_um": 0.125,     # sub-ring step of the fine RS reference
}

# rz mode (batch-OPD route)
RZ_SETTINGS = {
    "lams": "all",            # 'all' verification lines (seconds each)
}

# zone mode (FFT PSF per wavelength, display use)
ZONE_SETTINGS = {
    "out_sizes": ("256x256", "512x512"),
}

# --- LEGACY registry (pre-run-folder designs shipped as loose files) ---
DESIGNS = {
    "na03": dict(epd_mm=10.0, bfd_mm=15.899, file_no=1,
                 wavelengths_um=[0.50, 0.60, 0.70, 0.85, 1.05],
                 primary_idx=3, fold_P=24, lam0_um=0.70,
                 orders=[16, 24, 28, 0]),
    "s3": dict(epd_mm=10.24, bfd_mm=50.94, file_no=2,
               wavelengths_um=[0.45, 0.55, 0.70, 0.85, 1.05],
               primary_idx=2, fold_P=17, lam0_um=0.55,
               orders=[11, 13, 17, 0]),
    "s3comb": dict(epd_mm=10.24, bfd_mm=50.94, file_no=3,
                   wavelengths_um=[0.45, 0.55, 0.70, 0.85, 1.05],
                   primary_idx=2, fold_P=15, lam0_um=0.60,
                   orders=[10, 12, 15, 0]),
}
