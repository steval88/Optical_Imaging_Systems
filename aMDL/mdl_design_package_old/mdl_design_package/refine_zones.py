"""Zone-piston refinement: HJA over per-Fresnel-zone gray-level offsets."""
import json, os
import time

import numpy as np

from aMDL.old.mdl_design_package.mdl_core import MDLProblem, hooke_jeeves

T0 = time.time()
os.makedirs("out", exist_ok=True)

prob = MDLProblem(10000.0, 0.3, 0.40, 1.10, 0.65, 28.0, 0.078,
                  n_wavelengths=2001)
prob.G = prob.G.astype(np.complex64)
prob.L = prob.L.astype(np.complex64)

m = np.load("out/m_final.npy")
f0 = prob.fom(m)
print("start J=%.4f  (%.0fs setup)" % (f0, time.time() - T0), flush=True)

# detect zone boundaries: big downward resets of the seed profile
dh_steps = np.diff(m.astype(int))
reset = np.where(dh_steps < -int(0.5 * prob.M))[0] + 1
zones = np.split(np.arange(prob.N), reset)
print("zones: %d" % len(zones), flush=True)

zone_of = np.empty(prob.N, dtype=np.int32)
for zi, idx in enumerate(zones):
    zone_of[idx] = zi
nz = len(zones)

off = np.zeros(nz, dtype=np.int32)


def apply(off_vec):
    mm = m.astype(np.int64) + off_vec[zone_of]
    return np.clip(mm, 0, prob.M).astype(np.int32)


# integer HJA over zone offsets (full FOM evals; nz is small)
best_f = f0
d = 32
sweeps = 0
while d >= 1 and sweeps < 60:
    improved = False
    for zi in range(nz):
        for step in (d, -d):
            trial = off.copy()
            trial[zi] += step
            f = prob.fom(apply(trial))
            if f > best_f + 1e-7:
                off, best_f, improved = trial, f, True
                break
    if not improved:
        d //= 2
    sweeps += 1
    print("sweep %d (d=%d): J=%.4f  [%.0fs]" % (sweeps, d, best_f,
                                                time.time() - T0),
          flush=True)

m_best = apply(off)
# final per-ring polish at d0=1
m_best, f_pol = hooke_jeeves(prob, m_best, d0=1, max_sweeps=25)
print("after zone HJA + ring polish: J=%.4f" % f_pol, flush=True)

np.save("out/m_final2.npy", m_best)
json.dump({"J_zone_refined": f_pol, "n_zones": nz,
           "offsets": off.tolist()},
          open("out/zone_refine.json", "w"), indent=1)
