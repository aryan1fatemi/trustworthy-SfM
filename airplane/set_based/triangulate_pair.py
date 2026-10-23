"""Step 1: set-based triangulation from the initial pair.

    python triangulate_pair.py [--eps 3] [--pair 162,15]

Cameras 1 and 2 (COLMAP's initial pair, #162 = r_82.png and #15 = r_15.png) are held at COLMAP's
poses, in the gauge R_1 = I, C_1 = 0, x of C_2 = 1. For every landmark seen in both images, the
set of 3D positions consistent with both detections within eps pixels is the polytope

    F_k = { X : G_k X <= h_k },

four window faces plus depth per observation (duplicate detections in one image add their own rows).
It is kept as its inequalities; its box, Chebyshev centre, vertices and long-axis length are
reported for convenience.

Writes results/pair_sets.npz   (G, h stacked, with landmark index; box; centre; COLMAP point)
       results/pair_sets.csv   (one row per landmark)
       results/pair_summary.json
"""
import os
import csv
import json
import argparse
import numpy as np
from scipy.spatial import HalfspaceIntersection

from setlib import Scene, window_rows, chebyshev, box_of, results_dir


def build_sets(sc, eps, scheme="fixed", kappa=1.5, margin=1.0, cap=3.0):
    i1, i2 = sc.i1, sc.i2
    common = sorted(set(sc.det[i1]) & set(sc.det[i2]))
    depths = [sc.pose(i)[0][2] @ (sc.point(k) - sc.pose(i)[1]) for k in common for i in (i1, i2)]
    zmin = 1e-3 * float(np.median(depths))
    sets = {}
    for k in common:
        G, H = [], []
        for i in (i1, i2):
            for xn, e in zip(sc.det[i][k], sc.eps_list(i, k, eps, scheme, kappa, margin, cap)):
                g, h = window_rows(*sc.pose(i), xn, sc.f[i], e, zmin)
                G.append(g)
                H.append(h)
        G, H = np.vstack(G), np.concatenate(H)
        c, r = chebyshev(G, H)
        if c is None:
            sets[k] = {"empty": True}
            continue
        bx = box_of(G, H)
        if bx is None:
            sets[k] = {"empty": False, "bounded": False, "G": G, "h": H, "centre": c}
            continue
        V = HalfspaceIntersection(np.c_[G, -H], c).intersections
        axis = np.linalg.svd(V - V.mean(0), full_matrices=False)[2][0]
        sets[k] = {"empty": False, "bounded": True, "G": G, "h": H, "centre": c, "radius": r,
                   "lo": bx[0], "hi": bx[1], "vertices": V, "axis": axis,
                   "length": float(np.ptp(V @ axis))}
    return sets, zmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--pair", default="162,15")
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"],
                    help="per-detection tolerance: fixed eps, max(eps, kappa r_hat), max(eps, r_hat + margin)")
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--cap", type=float, default=3.0, help="tolerance <= cap * eps")
    a = ap.parse_args()
    RESULTS = results_dir(a.scheme)
    pair = tuple(int(x) for x in a.pair.split(","))
    sc = Scene(pair=pair, extra=())
    sets, zmin = build_sets(sc, a.eps, a.scheme, a.kappa, a.margin, a.cap)

    ok = [k for k, s in sets.items() if s.get("bounded")]
    rows_G, rows_h, rows_k = [], [], []
    for k in ok:
        rows_G.append(sets[k]["G"])
        rows_h.append(sets[k]["h"])
        rows_k += [k] * len(sets[k]["h"])
    np.savez_compressed(os.path.join(RESULTS, "pair_sets.npz"),
                        landmarks=np.array(ok), G=np.vstack(rows_G), h=np.concatenate(rows_h),
                        row_landmark=np.array(rows_k), lo=np.array([sets[k]["lo"] for k in ok]),
                        hi=np.array([sets[k]["hi"] for k in ok]),
                        centre=np.array([sets[k]["centre"] for k in ok]),
                        colmap_point=np.array([sc.point(k) for k in ok]), eps=a.eps, z_min=zmin)
    with open(os.path.join(RESULTS, "pair_sets.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point3D_id", "box_x", "box_y", "box_z", "box_diag", "long_axis_length",
                    "chebyshev_radius", "colmap_point_inside"])
        for k in ok:
            s = sets[k]
            size = s["hi"] - s["lo"]
            inside = bool(np.all(s["G"] @ sc.point(k) <= s["h"] + 1e-9))
            w.writerow([k, *np.round(size, 6), round(float(np.linalg.norm(size)), 6),
                        round(s["length"], 6), round(float(s["radius"]), 6), int(inside)])
    diag = np.array([np.linalg.norm(sets[k]["hi"] - sets[k]["lo"]) for k in ok])
    length = np.array([sets[k]["length"] for k in ok])
    inside = np.array([np.all(sets[k]["G"] @ sc.point(k) <= sets[k]["h"] + 1e-9) for k in ok])
    rep = {"eps_px": a.eps, "scheme": a.scheme, "kappa": a.kappa, "margin": a.margin, "cap": a.cap, "pair": [sc.imgs[i].name for i in pair], "z_min": zmin,
           "landmarks_common": len(sets), "empty": sum(1 for s in sets.values() if s.get("empty")),
           "unbounded": sum(1 for s in sets.values() if not s.get("empty") and not s.get("bounded")),
           "bounded": len(ok), "box_diag_median": float(np.median(diag)),
           "long_axis_length_median": float(np.median(length)),
           "colmap_point_inside_fraction": float(inside.mean()),
           "units": "gauge: camera 1 at the origin, x of camera 2's centre = 1"}
    json.dump(rep, open(os.path.join(RESULTS, "pair_summary.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
