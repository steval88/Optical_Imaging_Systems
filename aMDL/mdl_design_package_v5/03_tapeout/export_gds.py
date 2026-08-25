"""
export_gds.py
=============

Export an MDL ring-height table to GDSII for maskless (grayscale)
lithography, e.g. Heidelberg Instruments DWL systems.

Encoding
--------
The lens is a staircase of concentric annuli with quantized heights
h_i = m_i * dh (gray levels m_i in 0..M). Two GDS encodings are
provided (choose with --mode):

  index   (default)  layer L contains the annuli whose gray level == L.
                     This matches the common Heidelberg grayscale
                     conversion convention "one GDS layer = one gray /
                     dose value". Level 0 (no resist left / full
                     exposure reference) is NOT drawn by default --
                     enable with --draw-zero if your dose table needs
                     an explicit polygon for it.

  terrace            layer L contains the region where gray level >= L
                     (nested "topographic contour" masks, L = 1..M).
                     Useful for etch-back / multi-mask flows and for
                     conversion tools that build height by stacking.

Geometry
--------
* Adjacent rings with equal gray level are merged into a single
  annulus (fewer, cleaner polygons).
* Annuli are drawn with gdstk.ellipse(inner_radius=...) using a chord
  tolerance (--tol, default 0.02 um) so circle facets stay well below
  the writer's address grid; polygons auto-fractured to <= 8190 pts.
* Units: user unit 1 um, database unit 1 nm (--precision to change).

Usage
-----
    python export_gds.py --rings mdl_rings_2.txt --out mdl_s3.gds \
        [--mode index|terrace] [--dh-um 0.078] [--cell MDL] \
        [--tol 0.02] [--draw-zero]

The ring table is the same file the Zemax UDS DLL reads:
    line 1: N delta_mm
    then N lines: h_mm

A sidecar CSV "<out>.layers.csv" maps layer -> level -> height (um) ->
suggested normalized gray value, for the lithography tool's dose table.
"""
import argparse
import csv
import os

import numpy as np
import gdstk


def read_rings(path):
    with open(path) as fh:
        first = fh.readline().split()
        n, delta_mm = int(first[0]), float(first[1])
        h_mm = np.array([float(fh.readline()) for _ in range(n)])
    return h_mm * 1000.0, delta_mm * 1000.0          # -> um


def merge_runs(levels):
    """Group consecutive equal levels: [(level, i_start, i_end_excl)]."""
    runs = []
    start = 0
    for i in range(1, levels.size + 1):
        if i == levels.size or levels[i] != levels[start]:
            runs.append((int(levels[start]), start, i))
            start = i
    return runs


def annulus(r_in_um, r_out_um, layer, tol):
    """Annulus polygon(s) on the given layer, fractured to <=8190 pts."""
    if r_in_um <= 0.0:
        p = gdstk.ellipse((0, 0), r_out_um, layer=layer, tolerance=tol)
    else:
        p = gdstk.ellipse((0, 0), r_out_um, inner_radius=r_in_um,
                          layer=layer, tolerance=tol)
    if p.size > 8190:
        return p.fracture(max_points=8190)
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("index", "terrace"),
                    default="index")
    ap.add_argument("--dh-um", type=float, default=0.078,
                    help="gray-level height quantum used at design time")
    ap.add_argument("--cell", default="MDL")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="circle chord tolerance in um")
    ap.add_argument("--precision", type=float, default=1e-9,
                    help="GDS database unit in meters (default 1 nm)")
    ap.add_argument("--draw-zero", action="store_true",
                    help="also draw level-0 annuli (index mode)")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    h_um, delta_um = read_rings(args.rings)
    n = h_um.size
    levels = np.rint(h_um / args.dh_um).astype(int)
    m_max = int(levels.max())
    print("rings: %d  ring width: %.4f um  levels: 0..%d  R = %.3f mm"
          % (n, delta_um, m_max, n * delta_um / 1000.0))

    lib = gdstk.Library(name="MDL", unit=1e-6, precision=args.precision)
    cell = lib.new_cell(args.cell)

    n_poly = 0
    if args.mode == "index":
        for lev, i0, i1 in merge_runs(levels):
            if lev == 0 and not args.draw_zero:
                continue
            polys = annulus(i0 * delta_um, i1 * delta_um, lev, args.tol)
            cell.add(*polys)
            n_poly += len(polys)
    else:  # terrace: layer L = {rho : level >= L}
        for L in range(1, m_max + 1):
            mask = levels >= L
            # group contiguous True rings into annuli
            i = 0
            while i < n:
                if mask[i]:
                    j = i
                    while j < n and mask[j]:
                        j += 1
                    polys = annulus(i * delta_um, j * delta_um, L,
                                    args.tol)
                    cell.add(*polys)
                    n_poly += len(polys)
                    i = j
                else:
                    i += 1

    lib.write_gds(args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print("wrote %s  (%.1f MB, %d polygons, mode=%s)"
          % (args.out, size_mb, n_poly, args.mode))

    # layer map sidecar for the dose table
    side = args.out + ".layers.csv"
    with open(side, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gds_layer", "gray_level", "height_um",
                    "height_frac_of_max"])
        for L in range(0 if args.draw_zero else 1, m_max + 1):
            w.writerow([L, L, "%.4f" % (L * args.dh_um),
                        "%.5f" % (L / m_max)])
    print("layer map -> %s" % side)


if __name__ == "__main__":
    main()