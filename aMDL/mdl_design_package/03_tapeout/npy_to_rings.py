"""
npy_to_rings.py -- convert a saved gray-level vector (.npy) into the
ring-table text format used by both the Zemax UDS DLL (us_mdl_rings.dll)
and export_gds.py.

    line 1:  N  delta_mm
    then N lines: h_mm  (= gray_level * dh)

Usage:
    python npy_to_rings.py --npy out\m_s3.npy --out out\mdl_rings_2.txt \
        --delta-um 2.0 --dh-um 0.078

Ring width / height quantum per design:
    S3 designs (m_s3.npy, m_s3_comb.npy) : --delta-um 2.0  --dh-um 0.078
    NA-0.3 design (m_final.npy)          : --delta-um 0.65 --dh-um 0.078
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npy", required=True, help="gray-level vector .npy")
ap.add_argument("--out", required=True, help="ring table .txt to write")
ap.add_argument("--delta-um", type=float, required=True,
                help="ring width in um")
ap.add_argument("--dh-um", type=float, default=0.078,
                help="height per gray level in um")
args = ap.parse_args()

m = np.load(args.npy).astype(int)
h_mm = m * args.dh_um / 1000.0
delta_mm = args.delta_um / 1000.0

out_dir = os.path.dirname(os.path.abspath(args.out))
os.makedirs(out_dir, exist_ok=True)
with open(args.out, "w") as fh:
    fh.write("%d %.9f\n" % (m.size, delta_mm))
    fh.writelines("%.9f\n" % v for v in h_mm)

print("wrote %s: %d rings, width %.4f um, levels 0..%d "
      "(max height %.3f um)"
      % (args.out, m.size, args.delta_um, m.max(), m.max() * args.dh_um))