"""Incremental set-based registration of images 3, 4, 5 (COLMAP's order after the initial pair).

    python incremental_sets.py [--eps 3] [--images 17,73,5] [--scheme fixed]

Starts from the pair's landmark sets F_k (results*/pair_sets.npz). For each new image i, in order:
 1. Pose set by radial search (exact fixed-rotation LP): verified feasible poses (R, C) around COLMAP's
    rotation; each is reported as w (rotation offset from COLMAP's, deg) and translation t = -R C.
 2. INTERSECTION ("for every pose, every point"): feasible poses are added in order of distance from
    COLMAP's pose (rotation angle + centre shift / median depth); a pose is kept only if every landmark
    still has a point that fits ALL kept poses. Result: pose set P_i^all and shrunken sets
        F_k <- F_k  intersected with  S_k(theta) for every kept theta,
    used by the next image.
 3. UNIONS: tuples (pose, slice volume). For a pose theta, S_k(theta) = F_k  intersected with  its
    back-projected window; each tuple is itself a "for every pose, every point" pair (one pose, its slices);
    their union over theta is the union of slices.
Writes results_incremental[_scheme]/: summary.json, image<i>_intersection.csv (kept poses),
image<i>_unions.csv (tuples), sets_after_image<i>.npz (shrunken F_k).
"""
import os
import csv
import json
import math
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation

from setlib import Scene, box_normals, feasible, chebyshev, expR, results_dir, HERE
import backproject_image3 as BP


def polytope_volume(A, b):
    c, r = chebyshev(A, b)
    if c is None or r < 1e-10:
        return 0.0
    try:
        return float(ConvexHull(HalfspaceIntersection(np.c_[A, -b], c).intersections).volume)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--images", default="17,73,5")
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"])
    ap.add_argument("--directions", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=600, help="feasible poses tried for the intersection")
    ap.add_argument("--union-poses", type=int, default=200, help="poses reported as (pose, volume) tuples")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    T0 = time.time()
    SRC = results_dir(a.scheme)
    OUT = os.path.join(HERE, "results_incremental" + ("" if a.scheme == "fixed" else "_" + a.scheme))
    os.makedirs(OUT, exist_ok=True)
    P = np.load(os.path.join(SRC, "pair_sets.npz"))
    zmin = float(P["z_min"])
    order = [int(x) for x in a.images.split(",")]
    sc = Scene(pair=(162, 15), extra=tuple(order))
    F = {int(k): (P["G"][P["row_landmark"] == k], P["h"][P["row_landmark"] == k]) for k in P["landmarks"]}
    F0 = dict(F)
    vol0 = {k: polytope_volume(*F[k]) for k in F}
    report = {"eps_px": a.eps, "scheme": a.scheme, "images": [], "units":
              "w: rotation offset from COLMAP's (deg); t = -R C and C in the gauge (camera 1 at the origin, x of camera 2's centre = 1)"}

    for step, i in enumerate(order, start=3):
        t_img = time.time()
        f = sc.f[i]
        L = [k for k in F if k in sc.det[i]]
        eps_i = {k: sc.eps_list(i, k, a.eps, a.scheme) for k in L}
        normals = {k: [box_normals(xn, f, e) for xn, e in zip(sc.det[i][k], eps_i[k])] for k in L}
        G = np.vstack([F[k][0] for k in L])
        h = np.concatenate([F[k][1] for k in L])
        rowk = np.concatenate([[k] * len(F[k][1]) for k in L])
        S = BP.System(L, G, h, rowk, normals, zmin)
        Rc, Cc = sc.pose(i)
        R_gt, C_gt = sc.gt_pose(i)
        info = {"image": sc.imgs[i].name, "image_number": step, "landmarks_with_sets": len(L)}
        print("\n== image %d = %s: %d landmarks with sets" % (step, sc.imgs[i].name, len(L)), flush=True)
        if len(L) < 4 or not S.feasible(Rc):
            info["status"] = "COLMAP rotation infeasible with the current sets (or < 4 landmarks): pose set empty"
            print("   " + info["status"], flush=True)
            report["images"].append(info)
            break

        # ---- 1. pose set by radial search ---------------------------------------------------------
        U = BP.fibonacci(a.directions)
        with Pool(a.procs, initializer=BP._init, initargs=(S, Rc)) as pool:
            res = pool.map(BP._ray, list(U), chunksize=4)
        poses = [(Rc, Cc)] + [p for r in res for p in r[1]]
        Rs = np.array([p[0] for p in poses])
        Cs = np.array([p[1] for p in poses])
        W = np.degrees([Rotation.from_matrix(R @ Rc.T).as_rotvec() for R in Rs])
        T = np.array([-R @ C for R, C in zip(Rs, Cs)])
        zbar = float(np.median([np.linalg.norm(S_pt - Cc) for S_pt in [sc.point(k) for k in L]]))
        dist = np.linalg.norm(W, axis=1) + np.degrees(np.linalg.norm(Cs - Cc, axis=1) / zbar)
        info["pose_set_radial"] = {"poses": len(poses), "w_lo_deg": W.min(0).tolist(), "w_hi_deg": W.max(0).tolist(),
                                   "t_lo": T.min(0).tolist(), "t_hi": T.max(0).tolist(),
                                   "centre_size": np.ptp(Cs, 0).tolist()}
        print("   radial search: %d feasible poses; w in %s .. %s deg; centre extent %s"
              % (len(poses), np.round(W.min(0), 2), np.round(W.max(0), 2), np.round(np.ptp(Cs, 0), 3)), flush=True)

        # ---- 2. intersection: grow the pose set while every landmark keeps a common point -----------
        def cam_rows(k, R, C):
            A, b = [], []
            for N in normals[k]:
                for g in N @ R:
                    A.append(g)
                    b.append(g @ C)
                A.append(-R[2])
                b.append(-zmin - R[2] @ C)
            return np.array(A), np.array(b)

        order_idx = np.argsort(dist)
        cand = order_idx[np.unique(np.linspace(0, len(order_idx) - 1, min(a.candidates, len(order_idx))).astype(int))]
        cand = np.r_[0, cand[cand != 0]]
        acc_A = {k: [F[k][0]] for k in L}
        acc_b = {k: [F[k][1]] for k in L}
        kept = []
        for n in cand:
            add = {k: cam_rows(k, Rs[n], Cs[n]) for k in L}
            if all(feasible(np.vstack(acc_A[k] + [add[k][0]]), np.concatenate(acc_b[k] + [add[k][1]])) for k in L):
                for k in L:
                    acc_A[k].append(add[k][0])
                    acc_b[k].append(add[k][1])
                kept.append(n)
        newF = {k: (np.vstack(acc_A[k]), np.concatenate(acc_b[k])) for k in L}
        vol_new = np.array([polytope_volume(*newF[k]) for k in L])
        vol_old = np.array([polytope_volume(*F[k]) for k in L])
        vol_pair = np.array([vol0[k] for k in L])
        Wk, Tk, Ck = W[kept], T[kept], Cs[kept]
        with open(os.path.join(OUT, "image%d_intersection.csv" % step), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["pose", "w_x_deg", "w_y_deg", "w_z_deg", "t_1", "t_2", "t_3", "C_x", "C_y", "C_z"])
            for n in kept:
                wr.writerow([n, *np.round(W[n], 5), *np.round(T[n], 6), *np.round(Cs[n], 6)])
        # GT checks
        w_gt = np.degrees(Rotation.from_matrix(R_gt @ Rc.T).as_rotvec())
        gt_feasible_now = all(feasible(*[np.vstack([F[k][0], cam_rows(k, R_gt, C_gt)[0]]),
                                         np.r_[F[k][1], cam_rows(k, R_gt, C_gt)[1]]]) for k in L)
        info["intersection"] = {
            "candidates_tried": int(len(cand)), "poses_kept": len(kept),
            "w_lo_deg": Wk.min(0).tolist(), "w_hi_deg": Wk.max(0).tolist(),
            "t_lo": Tk.min(0).tolist(), "t_hi": Tk.max(0).tolist(),
            "C_lo": Ck.min(0).tolist(), "C_hi": Ck.max(0).tolist(),
            "max_rotation_from_colmap_deg": float(np.linalg.norm(Wk, axis=1).max()),
            "volume_ratio_vs_previous_median": float(np.median(vol_new / np.maximum(vol_old, 1e-30))),
            "volume_ratio_vs_pair_median": float(np.median(vol_new / np.maximum(vol_pair, 1e-30)))}
        info["gt"] = {"w_rel_colmap_deg": w_gt.tolist(), "t": (-R_gt @ C_gt).tolist(),
                      "gt_pose_feasible_with_sets_before_this_image": bool(gt_feasible_now),
                      "gt_rotation_within_kept_range": bool(np.all((w_gt >= Wk.min(0) - 1e-9) & (w_gt <= Wk.max(0) + 1e-9)))}
        print("   intersection: %d of %d candidate poses kept; w in %s .. %s deg (max %.3f deg); t in %s .. %s"
              % (len(kept), len(cand), np.round(Wk.min(0), 3), np.round(Wk.max(0), 3),
                 info["intersection"]["max_rotation_from_colmap_deg"], np.round(Tk.min(0), 4), np.round(Tk.max(0), 4)), flush=True)
        print("   shrunken F_k: volume / previous median %.3f, / pair median %.3f; GT pose feasible before: %s; "
              "GT rotation offset %s deg" % (info["intersection"]["volume_ratio_vs_previous_median"],
                                             info["intersection"]["volume_ratio_vs_pair_median"], gt_feasible_now,
                                             np.round(w_gt, 3)), flush=True)

        # ---- 3. unions: (pose, slice volume) tuples --------------------------------------------------
        pick = order_idx[np.unique(np.linspace(0, len(order_idx) - 1, min(a.union_poses, len(order_idx))).astype(int))]
        with open(os.path.join(OUT, "image%d_unions.csv" % step), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["pose", "distance_from_colmap", "w_x_deg", "w_y_deg", "w_z_deg", "t_1", "t_2", "t_3",
                         "slice_volume_total", "slice_volume_ratio_median"])
            tot, med = [], []
            for n in pick:
                vols = np.array([polytope_volume(np.vstack([F[k][0], cam_rows(k, Rs[n], Cs[n])[0]]),
                                                 np.r_[F[k][1], cam_rows(k, Rs[n], Cs[n])[1]]) for k in L])
                tot.append(vols.sum())
                med.append(float(np.median(vols / np.maximum(vol_old, 1e-30))))
                wr.writerow([n, round(float(dist[n]), 5), *np.round(W[n], 5), *np.round(T[n], 6),
                             "%.6e" % vols.sum(), round(med[-1], 5)])
        info["unions"] = {"tuples": len(pick), "slice_volume_ratio_median_range": [float(min(med)), float(max(med))]}
        print("   unions: %d (pose, volume) tuples; slice volume / set volume median from %.3f to %.3f"
              % (len(pick), min(med), max(med)), flush=True)
        np.savez_compressed(os.path.join(OUT, "sets_after_image%d.npz" % step),
                            landmarks=np.array(L), **{"G_%d" % k: newF[k][0] for k in L}, **{"h_%d" % k: newF[k][1] for k in L})
        info["seconds"] = round(time.time() - t_img)
        report["images"].append(info)
        F.update(newF)                            # the next image starts from the shrunken sets

    report["seconds"] = round(time.time() - T0)
    json.dump(report, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\n-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
