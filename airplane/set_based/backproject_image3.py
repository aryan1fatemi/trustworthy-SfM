"""Step 2: set-based back-projection of image 3 onto the pair's landmark sets.

    python backproject_image3.py [--eps 3] [--third 17] [--directions 300]

Needs results/pair_sets.npz from triangulate_pair.py.

Problem (unknown: camera-3 pose theta = (R, C); constraint: every back-projected window hits its set)
    P_3 = { theta : for every landmark k seen by image 3, the window cone of theta meets F_k }.
For a FIXED rotation R the conditions are linear in (C, X_k), so one LP decides whether R is feasible
and gives the exact feasible centres (the intersection of the landmarks' back-shadows).

1. Reference rotation: P3P + RANSAC + refinement on the Chebyshev centres of the sets (search order
   only). The search starts from the first of {P3P rotation, COLMAP rotation} that is exactly feasible.
2. Feasible poses (inner approximation, every pose exactly verified): for D directions u, bisection on
   t until exp([t u]x) R_start is infeasible; at t_b and t_b / 2 the six extreme centres and their mean.
3. Slices: for a pose theta, S_k(theta) = F_k intersected with theta's window cone (a polytope).
   Reported per landmark: slice at COLMAP's pose, slice lengths along the set's long axis, the union of
   slices over the sampled poses, and whether all slices share a point (sampled poses: an empty
   intersection is certain, a non-empty one is an upper bound).

Writes results/image3_poses.npz, results/image3_slices.csv, results/image3_summary.json,
       results/image3_slices.png, results/image3_slice_sizes.png
"""
import os
import csv
import json
import math
import time
import argparse
import numpy as np
import cv2
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setlib import Scene, box_normals, feasible, expR, results_dir

BLUE, GREEN = "#2a78d6", "#008300"


class System:
    """unknowns: C (3), X_k (3 each); exact constraints for a fixed camera-3 rotation."""

    def __init__(self, L, G, h, rowk, normals, zmin):
        self.L, self.normals, self.zmin = L, normals, zmin
        self.idx = {k: 3 + 3 * n for n, k in enumerate(L)}
        self.n = 3 + 3 * len(L)
        rows, b = [], []
        for k in L:
            sel = rowk == k
            for g, hh in zip(G[sel], h[sel]):
                r = np.zeros(self.n)
                r[self.idx[k]:self.idx[k] + 3] = g
                rows.append(r)
                b.append(hh)
        self.AF, self.bF = np.array(rows), np.array(b)

    def rows(self, R):
        rows, b = [], []
        for k in self.L:
            iv = self.idx[k]
            for N in self.normals[k]:
                for g in list(N @ R) + [-R[2]]:
                    r = np.zeros(self.n)
                    r[iv:iv + 3] = g
                    r[0:3] = -g
                    rows.append(r)
                    b.append(0.0)
                b[-1] = -self.zmin                      # last row of the block is the depth row
        return np.vstack([self.AF, rows]), np.concatenate([self.bF, b])

    def feasible(self, R):
        return feasible(*self.rows(R))

    def centres(self, R):
        """the six extreme feasible centres for rotation R (and their mean)."""
        A, b = self.rows(R)
        out = []
        for j in range(3):
            for sg in (1.0, -1.0):
                c = np.zeros(self.n)
                c[j] = sg
                r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * self.n, method="highs")
                if r.status == 0:
                    out.append(r.x[:3])
        if out:
            out.append(np.mean(out, 0))
        return out


_S = {}


def _init(system, R0):
    _S.update(system=system, R0=R0)


def _ray(u):
    S, R0 = _S["system"], _S["R0"]
    f = lambda t: S.feasible(expR(t * u) @ R0)
    lo, hi = 0.0, math.radians(10.0)
    if f(hi):
        lo = hi
    else:
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
    poses = []
    for t in (lo, 0.5 * lo):
        R = expR(t * u) @ R0
        poses += [(R, C) for C in S.centres(R)]
    return lo, poses


def fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--directions", type=int, default=300)
    ap.add_argument("--slice-poses", type=int, default=400, help="poses used for slice lengths")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"])
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--cap", type=float, default=3.0)
    a = ap.parse_args()
    RESULTS = results_dir(a.scheme)
    T0 = time.time()
    P = np.load(os.path.join(RESULTS, "pair_sets.npz"))
    sc = Scene(pair=(162, 15), extra=(a.third,))
    i3 = a.third
    f3 = sc.f[i3]
    zmin = float(P["z_min"])
    lm_all = [int(k) for k in P["landmarks"]]
    L = [k for k in lm_all if k in sc.det[i3]]
    pos = {k: n for n, k in enumerate(lm_all)}
    eps3 = {k: sc.eps_list(i3, k, a.eps, a.scheme, a.kappa, a.margin, a.cap) for k in L}
    normals = {k: [box_normals(xn, f3, e) for xn, e in zip(sc.det[i3][k], eps3[k])] for k in L}
    S = System(L, P["G"], P["h"], P["row_landmark"], normals, zmin)
    R_col, C_col = sc.pose(i3)
    n_obs = sum(len(sc.det[i3][k]) for k in L)
    print("image 3 = %s: %d landmarks with pair sets, %d observations, eps = %g px"
          % (sc.imgs[i3].name, len(L), n_obs, a.eps), flush=True)

    # ---- 1. reference rotation ---------------------------------------------------------------
    P3 = np.array([P["centre"][pos[k]] for k in L])
    p2 = np.array([sc.det[i3][k][0] for k in L])
    _, rvec, tvec, inl = cv2.solvePnPRansac(P3, p2, np.eye(3), None, flags=cv2.SOLVEPNP_P3P,
                                            reprojectionError=a.eps / f3[0], iterationsCount=2000)
    rvec, tvec = cv2.solvePnPRefineLM(P3[inl[:, 0]], p2[inl[:, 0]], np.eye(3), None, rvec, tvec)
    R_p3p = cv2.Rodrigues(rvec)[0]
    starts = [("P3P on set centres", R_p3p), ("COLMAP", R_col)]
    start = next(((nm, R) for nm, R in starts if S.feasible(R)), None)
    if start is None:
        print("no feasible start rotation: the set of poses may be empty at eps = %g" % a.eps)
        return
    print("1. reference: P3P (%d/%d inliers, %.3f deg from COLMAP); search starts from: %s"
          % (len(inl), len(L), math.degrees(Rotation.from_matrix(R_p3p @ R_col.T).magnitude()), start[0]),
          flush=True)

    # ---- 2. feasible poses ----------------------------------------------------------------------
    U = fibonacci(a.directions)
    with Pool(a.procs, initializer=_init, initargs=(S, start[1])) as pool:
        res = pool.map(_ray, list(U), chunksize=4)
    radius = np.array([r[0] for r in res])
    poses = [(R_col, C_col)] + [p for r in res for p in r[1]]
    Rs = np.array([p[0] for p in poses])
    Cs = np.array([p[1] for p in poses])
    W = np.array([Rotation.from_matrix(R @ R_col.T).as_rotvec() for R in Rs])
    ang = np.degrees(np.linalg.norm(W, axis=1))
    np.savez_compressed(os.path.join(RESULTS, "image3_poses.npz"), R=Rs, C=Cs, w_rel_colmap=W, angle_deg=ang)
    R_gt, C_gt = sc.gt_pose(i3)
    gt_ok = S.feasible(R_gt)
    print("2. %d exactly feasible poses: rotation up to %.2f deg from COLMAP's (boundary radius "
          "min/median/max %.2f/%.2f/%.2f deg); centre extent %s; GT rotation feasible: %s"
          % (len(poses), ang.max(), *np.degrees([radius.min(), np.median(radius), radius.max()]),
             np.round(np.ptp(Cs, 0), 4), gt_ok), flush=True)

    # ---- 3. slices -------------------------------------------------------------------------------
    def slice_rows(k, R, C):
        G = P["G"][P["row_landmark"] == k]
        h = P["h"][P["row_landmark"] == k]
        A, b = [G], [h]
        for N in normals[k]:
            for g in N @ R:
                A.append(g[None])
                b.append(np.array([g @ C]))
            A.append(-R[2][None])
            b.append(np.array([-zmin - R[2] @ C]))
        return np.vstack(A), np.concatenate(b)

    def extent(A, b, d):
        r1 = linprog(d, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        r2 = linprog(-d, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        return (r1.fun, -r2.fun) if r1.status == 0 and r2.status == 0 else None

    sub = np.r_[0, np.linspace(1, len(poses) - 1, a.slice_poses).astype(int)]
    rows_out, per_lm = [], {}
    for k in L:
        G = P["G"][P["row_landmark"] == k]
        h = P["h"][P["row_landmark"] == k]
        lo0, hi0 = P["lo"][pos[k]], P["hi"][pos[k]]
        # long axis of F_k from its box-extremal points (via its box diagonal direction is crude, so
        # use the principal axis of the extreme points along x, y, z)
        ext_pts = []
        for d in np.eye(3):
            for sg in (1, -1):
                r = linprog(sg * d, A_ub=G, b_ub=h, bounds=[(None, None)] * 3, method="highs")
                ext_pts.append(r.x)
        ext_pts = np.array(ext_pts)
        axis = np.linalg.svd(ext_pts - ext_pts.mean(0), full_matrices=False)[2][0]
        e0 = extent(G, h, axis)
        L0 = e0[1] - e0[0]
        segs = []
        for n in sub:
            e = extent(*slice_rows(k, Rs[n], Cs[n]), axis)
            if e is not None:
                segs.append(e)
        segs = np.array(segs)
        A_all = [G]
        b_all = [h]
        for R, C in poses:
            A_s, b_s = slice_rows(k, R, C)
            A_all.append(A_s[len(h):])
            b_all.append(b_s[len(h):])
        common = feasible(np.vstack(A_all), np.concatenate(b_all))
        col = slice_rows(k, R_col, C_col)
        from setlib import box_of
        bx = box_of(*col)
        rows_out.append({"point3D_id": k, "Fk_box_diag": float(np.linalg.norm(hi0 - lo0)),
                         "Fk_length": float(L0),
                         "slice_colmap_box_diag": float(np.linalg.norm(bx[1] - bx[0])) if bx else np.nan,
                         "slice_colmap_length": float(segs[0, 1] - segs[0, 0]),
                         "slice_length_median": float(np.median(segs[:, 1] - segs[:, 0])),
                         "union_length": float(segs[:, 1].max() - segs[:, 0].min()),
                         "slices_share_a_point": int(common)})
        per_lm[k] = (segs, e0, axis)
    with open(os.path.join(RESULTS, "image3_slices.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    col = lambda key: np.array([r[key] for r in rows_out])
    rep = {
        "eps_px": a.eps, "scheme": a.scheme, "kappa": a.kappa, "margin": a.margin, "cap": a.cap,
        "image3_eps_median_max": [float(np.median([e for v in eps3.values() for e in v])),
                                  float(max(e for v in eps3.values() for e in v))],
        "image3": sc.imgs[i3].name, "landmarks": len(L), "observations": n_obs,
        "reference": {"p3p_inliers": int(len(inl)),
                      "p3p_deg_from_colmap": math.degrees(Rotation.from_matrix(R_p3p @ R_col.T).magnitude()),
                      "search_start": start[0]},
        "pose_set_inner": {"poses": len(poses), "directions": a.directions,
                           "max_angle_from_colmap_deg": float(ang.max()),
                           "w_rel_colmap_lo_deg": np.degrees(W.min(0)).tolist(),
                           "w_rel_colmap_hi_deg": np.degrees(W.max(0)).tolist(),
                           "centre_lo": Cs.min(0).tolist(), "centre_hi": Cs.max(0).tolist(),
                           "centre_size": np.ptp(Cs, 0).tolist(),
                           "note": "inner approximation: every pose listed is exactly feasible"},
        "gt": {"rotation_offset_from_colmap_deg": np.degrees(Rotation.from_matrix(R_gt @ R_col.T).as_rotvec()).tolist(),
               "centre": C_gt.tolist(), "colmap_centre": C_col.tolist(), "gt_rotation_exactly_feasible": bool(gt_ok)},
        "slices": {"Fk_length_median": float(np.median(col("Fk_length"))),
                   "slice_at_colmap_length_median": float(np.median(col("slice_colmap_length"))),
                   "slice_length_over_Fk_median": float(np.median(col("slice_length_median") / col("Fk_length"))),
                   "union_over_Fk_median": float(np.median(col("union_length") / col("Fk_length"))),
                   "landmarks_whose_slices_share_a_point": int(col("slices_share_a_point").sum())},
        "units": "gauge: camera 1 at the origin with R = I, x of camera 2's centre = 1",
        "seconds": round(time.time() - T0)}
    json.dump(rep, open(os.path.join(RESULTS, "image3_summary.json"), "w"), indent=2)
    print(json.dumps(rep["slices"], indent=1))

    # ---- figures ------------------------------------------------------------------------------------
    k = sorted(L, key=lambda k: per_lm[k][1][1] - per_lm[k][1][0])[len(L) // 2]
    segs, e0, axis = per_lm[k]
    angs = ang[sub]
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(0, e0[1] - e0[0], color=BLUE, alpha=0.10, label="F_k along its long axis")
    for (lo, hi), an in zip(segs, angs):
        ax.plot([lo - e0[0], hi - e0[0]], [an, an], color=cmap(an / max(angs.max(), 1e-9)), lw=1.2)
    ax.axvline(sc.point(k) @ axis - e0[0], color="#0b0b0b", lw=1.2, label="COLMAP point")
    ax.set_xlabel("position along the set's long axis (baseline units)")
    ax.set_ylabel("pose: rotation offset from COLMAP's (deg)")
    ax.set_title("Landmark #%d: slice of F_k for each feasible camera-3 pose (eps = %g px)" % (k, a.eps), fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "image3_slices.png"), dpi=120)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    ratios = np.concatenate([(per_lm[k][0][:, 1] - per_lm[k][0][:, 0]) / (per_lm[k][1][1] - per_lm[k][1][0]) for k in L])
    axs[0].hist(ratios, bins=40, color=BLUE, alpha=0.85)
    axs[0].set_xlabel("slice length / F_k length")
    axs[0].set_ylabel("landmark-pose pairs")
    axs[0].set_title("one pose: part of the set its window keeps", fontsize=10)
    axs[1].hist(col("union_length") / col("Fk_length"), bins=25, color="#1baf7a", alpha=0.85)
    axs[1].set_xlabel("union of slices over feasible poses / F_k length")
    axs[1].set_ylabel("landmarks")
    axs[1].set_title("all feasible poses: part of the set used", fontsize=10)
    for x in axs:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "image3_slice_sizes.png"), dpi=120)
    print("-> %s  (%ds)" % (RESULTS, time.time() - T0))


if __name__ == "__main__":
    main()
