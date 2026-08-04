"""
run_verify.py -- physical verification of a designed MDL by
Rayleigh-Sommerfeld (RS1) diffraction of the actual staircase profile
(the same validation approach as the reference paper, ref. 46 / Fig. 2e).

Usage:
    python run_verify.py runs/<run_folder> [optional/other_m.npy]

All geometry, wavelengths and grids are read from the run folder's
config.json (written by run_MDL_design.py) -- nothing is hardcoded here.
The optional second argument verifies a different design vector against
that same configuration.

Computes, per verification wavelength:
  * on-axis intensity vs z (focal shift / achromaticity check)
  * PSF on the design focal plane, FWHM
  * focus efficiency (power within 1.5x FWHM disc / total incident)
  * Strehl-like normalized peak vs ideal lens of same aperture
Also evaluates J on the design objective, on the continuous band and on
the verification comb, and re-exports the DLL ring table.

Outputs (into the run folder):
    verify_metrics.json, verify_onaxis.npz, mdl_rings_<n>.txt
    scripts/run_verify.py (snapshot of this script)

Axisymmetric RS1 with the standard J0 far-kernel expansion:
  E(r0, z) = (z / i lambda) * SUM_i  E_i * J0(k rho_i r0 / rbar_i)
             * exp(i k rbar_i) / rbar_i^2 * (2 pi rho_i drho)
  rbar_i = sqrt(z^2 + rho_i^2 + r0^2)   (valid near axis, r0 small)
"""
import json
import os
import shutil
import sys
import time

import numpy as np
from numpy import pi
from scipy.special import j0

from aMDL.old.mdl_design_package.mdl_core import MDLProblem

T0 = time.time()


def log(s):
    print("[%6.1fs] %s" % (time.time() - T0, s), flush=True)


# ---- load run configuration ----------------------------------------------
if len(sys.argv) < 2:
    raise SystemExit("usage: python run_verify.py runs/<run_folder> "
                     "[optional/other_m.npy]")
run_dir = sys.argv[1].rstrip("/\\")
cfg_path = os.path.join(run_dir, "config.json")
if not os.path.exists(cfg_path):
    raise SystemExit("no config.json in %r -- is this a run folder made "
                     "by run_MDL_design.py?" % run_dir)
cfg = json.load(open(cfg_path))
der = cfg["derived"]

D, na, F = cfg["diameter_um"], der["na"], der["focal_um"]
lmin, lmax = cfg["lam_min_um"], cfg["lam_max_um"]
lams_obj = cfg["target_wavelengths_um"]        # None = continuous objective
lam_list = np.asarray(cfg["verify_wavelengths_um"], dtype=float)

m_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir,
                                                            "m_final.npy")
m = np.load(m_file)

# continuous-band problem (geometry + alias-safe J_continuous)
prob = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"],
                  cfg["h_max_um"], cfg["dh_um"],
                  n_wavelengths=cfg["n_wavelengths"])
if m.size != prob.N:
    raise SystemExit("%s has %d rings; config expects %d"
                     % (m_file, m.size, prob.N))
h = m * prob.dh                      # [um]
rho = prob.rho                       # [um]
drho = prob.delta
R = prob.R

# snapshot this script into the run folder
os.makedirs(os.path.join(run_dir, "scripts"), exist_ok=True)
shutil.copy2(__file__, os.path.join(run_dir, "scripts",
                                    os.path.basename(__file__)))

log("run: %s" % run_dir)
log("profile %s: J_continuous = %.4f" % (m_file, prob.fom(m)))


def exit_field(lam):
    """Complex field just after the lens for unit-amplitude plane wave."""
    n = float(prob.n_func(lam))
    k = 2 * pi / lam
    # transmission phase of the staircase (thin-element for the relief,
    # exactly what the staircase adds in OPL for a normally incident ray)
    return np.exp(1j * k * (n - 1.0) * h)


def rs_onaxis(lam, zgrid):
    E0 = exit_field(lam)
    k = 2 * pi / lam
    out = np.empty(zgrid.size, dtype=complex)
    for iz, z in enumerate(zgrid):
        rb = np.sqrt(z * z + rho * rho)
        integ = E0 * np.exp(1j * k * rb) / rb ** 2 * rho
        out[iz] = (z / (1j * lam)) * 2 * pi * drho * np.sum(integ)
    return out


def rs_psf(lam, z, r0grid):
    E0 = exit_field(lam)
    k = 2 * pi / lam
    out = np.empty(r0grid.size, dtype=complex)
    for ir, r0 in enumerate(r0grid):
        rb = np.sqrt(z * z + rho * rho + r0 * r0)
        integ = E0 * j0(k * rho * r0 / rb) * np.exp(1j * k * rb) / rb ** 2 \
            * rho
        out[ir] = (z / (1j * lam)) * 2 * pi * drho * np.sum(integ)
    return out


# ideal reference: peak of a perfect lens of same R, F at each lam
def ideal_peak(lam):
    k = 2 * pi / lam
    rb = np.sqrt(F * F + rho * rho)
    integ = np.exp(-1j * k * (rb - F)) * np.exp(1j * k * rb) / rb ** 2 * rho
    return abs((F / (1j * lam)) * 2 * pi * drho * np.sum(integ))


results = {"run_dir": run_dir, "m_file": m_file,
           "lam_um": lam_list.tolist(), "F_um": F}
zgrid = np.linspace(F - cfg["verify_z_span_um"],
                    F + cfg["verify_z_span_um"], cfg["verify_z_points"])
onax = {}
log("on-axis scans...")
for lam in lam_list:
    I = np.abs(rs_onaxis(lam, zgrid)) ** 2
    onax[lam] = I
    zpk = float(zgrid[np.argmax(I)])
    results.setdefault("z_peak_um", []).append(zpk)
log("on-axis done")

# PSF at the design focal plane
r0grid = np.linspace(0.0, cfg["verify_r_max_um"], cfg["verify_r_points"])
psf_metrics = {"fwhm_um": [], "eff_3fwhm": [], "strehl_like": [],
               "onax_I_at_F": []}
log("PSFs at z=F...")
for lam in lam_list:
    E = rs_psf(lam, F, r0grid)
    I = np.abs(E) ** 2
    Ipk = I[0] if I[0] == I.max() else I.max()
    # FWHM (first crossing)
    half = Ipk / 2
    idx = np.where(I < half)[0]
    fwhm = 2 * r0grid[idx[0]] if idx.size else np.nan
    # efficiency: power within 1.5*FWHM radius over total incident
    r_int = 1.5 * fwhm
    sel = r0grid <= r_int
    p_in = np.trapezoid(I[sel] * 2 * pi * r0grid[sel], r0grid[sel])
    p_tot = pi * R * R          # unit-amplitude over aperture
    ideal = ideal_peak(lam)
    psf_metrics["fwhm_um"].append(float(fwhm))
    psf_metrics["eff_3fwhm"].append(float(p_in / p_tot))
    psf_metrics["strehl_like"].append(float(Ipk / ideal ** 2))
    psf_metrics["onax_I_at_F"].append(float(I[0]))
    log("  lam=%.2f: FWHM=%.2f um (dl %.2f), eff=%.3f, S=%.3f"
        % (lam, fwhm, lam / (2 * na), psf_metrics["eff_3fwhm"][-1],
           psf_metrics["strehl_like"][-1]))
results.update(psf_metrics)

# ---- J metrics: objective / continuous / verification comb ---------------
results["J_continuous"] = prob.fom(m)
comb = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"], cfg["h_max_um"],
                  cfg["dh_um"],
                  n_wavelengths=lam_list.size).set_wavelengths(lam_list)
results["J_verify_comb"] = comb.fom(m)
if lams_obj is not None:
    obj = MDLProblem(D, na, lmin, lmax, cfg["ring_width_um"],
                     cfg["h_max_um"], cfg["dh_um"],
                     n_wavelengths=len(lams_obj)).set_wavelengths(
                         np.asarray(lams_obj, dtype=float))
    results["J_objective"] = obj.fom(m)
else:
    results["J_objective"] = results["J_continuous"]
log("J objective=%.4f   continuous=%.4f   verify-comb=%.4f"
    % (results["J_objective"], results["J_continuous"],
       results["J_verify_comb"]))

# ---- save into the run folder --------------------------------------------
np.savez(os.path.join(run_dir, "verify_onaxis.npz"), zgrid=zgrid,
         **{"I_%d" % int(l * 1000): onax[l] for l in lam_list})
json.dump(results, open(os.path.join(run_dir, "verify_metrics.json"), "w"),
          indent=1)

# re-export ring table for the Zemax DLL (heights in mm!) so the table
# always matches the vector that was actually verified
ring_path = os.path.join(run_dir, "mdl_rings_%d.txt" % cfg["dll_file_no"])
with open(ring_path, "w") as f:
    f.write("%d %.9f\n" % (prob.N, prob.delta / 1000.0))
    f.writelines("%.9f\n" % (v / 1000.0) for v in h)
log("saved: verify_metrics.json, verify_onaxis.npz, %s"
    % os.path.basename(ring_path))
log("next:  python make_plots.py %s" % run_dir)