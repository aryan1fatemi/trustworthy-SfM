"""How much do the image-3 slices of one landmark overlap across feasible poses?

    python slice_overlap.py [--eps 3] [--third 17]

Needs results/pair_sets.npz and results/image3_poses.npz (backproject_image3.py).
Pose order in image3_poses.npz: COLMAP's pose, then per direction the rotation at the boundary angle
t_b (6 extreme centres + their mean) and at t_b / 2 (6 extreme centres + their mean).
Pose classes:
    interior   rotation t_b / 2, centre = mean of the 6 extremes
    centre-extreme / rotation-boundary   the others
Per landmark:
    exact 3D intersection of the slices (LP feasibility) for each pose class, and for all poses;
    along the set's long axis: the best-covered point and the fraction of poses whose slice contains it,
    and the fraction of pose pairs whose slices overlap.
Writes results/slice_overlap.csv and results/slice_overlap.json.
"""
import os
import csv
import json
import argparse
import numpy as np
from scipy.optimize import linprog

from setlib import Scene, box_normals, feasible, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--sub", type=int, default=600, help="poses used for the 1-D coverage statistics")
    a = ap.parse_args()
    P = np.load(os.path.join(RESULTS, "pair_sets.npz"))
    Q = np.load(os.path.join(RESULTS, "image3_poses.npz"))
    Rs, Cs = Q["R"], Q["C"]
    sc = Scene(pair=(162, 15), extra=(a.third,))
    f3, zmin = sc.f[a.third], float(P["z_min"])
    L = [int(k) for k in P["landmarks"] if int(k) in sc.det[a.third]]

    # pose classes (block of 14 per direction after COLMAP's pose)
    n = len(Rs)
    cls = np.array(["colmap"] + [""] * (n - 1), dtype=object)
    for b in range((n - 1) // 14):
        base = 1 + 14 * b
        cls[base:base + 6] = "rot-boundary, centre-extreme"
        cls[base + 6] = "rot-boundary, centre-mean"
        cls[base + 7:base + 13] = "rot-half, centre-extreme"
        cls[base + 13] = "rot-half, centre-mean (interior)"
    classes = ["rot-half, centre-mean (interior)", "rot-half, centre-extreme",
               "rot-boundary, centre-mean", "rot-boundary, centre-extreme"]

    def cam_rows(k, R, C):
        A, b = [], []
        for xn in sc.det[a.third][k]:
            for g in box_normals(xn, f3, a.eps) @ R:
                A.append(g)
                b.append(g @ C)
            A.append(-R[2])
            b.append(-zmin - R[2] @ C)
        return np.array(A), np.array(b)

    rng = np.random.default_rng(0)
    sub = np.r_[0, rng.choice(np.arange(1, n), size=min(a.sub, n - 1), replace=False)]
    out = []
    for k in L:
        G = P["G"][P["row_landmark"] == k]
        h = P["h"][P["row_landmark"] == k]
        rows = {c: [] for c in ["all"] + classes}
        for i in range(n):
            A, b = cam_rows(k, Rs[i], Cs[i])
            rows["all"].append((A, b))
            if cls[i] in rows:
                rows[cls[i]].append((A, b))
        res = {"point3D_id": k}
        for c, lst in rows.items():
            A = np.vstack([G] + [x[0] for x in lst])
            b = np.concatenate([h] + [x[1] for x in lst])
            res["common_point_" + c] = int(feasible(A, b))
        # long axis of F_k from its 6 extreme points
        E = []
        for d in np.eye(3):
            for sg in (1, -1):
                E.append(linprog(sg * d, A_ub=G, b_ub=h, bounds=[(None, None)] * 3, method="highs").x)
        E = np.array(E)
        ax = np.linalg.svd(E - E.mean(0), full_matrices=False)[2][0]
        iv = []
        for i in sub:
            A, b = cam_rows(k, Rs[i], Cs[i])
            A, b = np.vstack([G, A]), np.r_[h, b]
            r1 = linprog(ax, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
            r2 = linprog(-ax, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
            if r1.status == 0 and r2.status == 0:
                iv.append((r1.fun, -r2.fun))
        iv = np.array(iv)
        grid = np.linspace(iv[:, 0].min(), iv[:, 1].max(), 400)
        cover = ((iv[:, 0][None] <= grid[:, None]) & (iv[:, 1][None] >= grid[:, None])).mean(1)
        lo, hi = iv[:, 0], iv[:, 1]
        overlap = (np.maximum(lo[:, None], lo[None]) <= np.minimum(hi[:, None], hi[None])).mean()
        res.update(best_point_coverage=float(cover.max()), pair_overlap_fraction=float(overlap))
        out.append(res)
    with open(os.path.join(RESULTS, "slice_overlap.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    col = lambda key: np.array([r[key] for r in out])
    counts = {c: int(np.sum(cls == c)) for c in classes}
    rep = {"landmarks": len(L), "poses": n, "poses_per_class": counts,
           "landmarks_with_a_common_point": {c: int(col("common_point_" + c).sum()) for c in ["all"] + classes},
           "best_point_coverage_median": float(np.median(col("best_point_coverage"))),
           "best_point_coverage_p10": float(np.quantile(col("best_point_coverage"), 0.1)),
           "pair_overlap_fraction_median": float(np.median(col("pair_overlap_fraction"))),
           "note": "coverage and pair overlap along the set's long axis (1-D); common points exact (3-D LP)"}
    json.dump(rep, open(os.path.join(RESULTS, "slice_overlap.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
