"""Incremental registration of images 3 and 4 with the UNION of slices (truth-preserving version).

    python union_incremental.py [--eps 3] [--scheme fixed] [--images 17,73]

Cameras 1, 2 (#162, #15) are held at COLMAP's poses; their landmark sets F_k come from
results*/pair_sets.npz. For each new image i:
 1. Pose set P_i by radial search (exact fixed-rotation LP): verified feasible poses (R, C), reported
    with the (pose, slice volume) tuples.
 2. Existing landmarks seen by i:   F_k <- F_k  intersected with  box( union over theta of S_k(theta) ),
    S_k(theta) = F_k  intersected with  K(theta, W_ik), theta over the sampled poses of P_i.
 3. New landmarks seen by i and by exactly one fixed camera (1 or 2): the union over theta of
    K(fixed camera, W) intersected with K(theta, W_ik), stored as those rows plus its box.
 4. (image 4) Landmarks seen by images 3 and 4 only: union over pose pairs (theta_3, theta_4).
The union is taken over sampled feasible poses and stored as its convex outer box (intersected with the
landmark's own inequalities), so every set stays a polytope for the next image.
GT checks after each image: GT pose feasible with the sets used, GT rotation / centre inside the pose
ranges, GT point inside the updated set (per group of landmarks).
Writes results_union[_scheme]/: summary.json, image<n>_poses.csv, image<n>_tuples.csv, sets_after_image<n>.npz
"""
import os
import sys
import csv
import json
import math
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation

from setlib import Scene, box_normals, window_rows, feasible, chebyshev, results_dir, HERE, SCENE
import backproject_image3 as BP

_W = {}


def _init(data):
    _W.update(data)


def cam_rows(normals, R, C, zmin):
    A, b = [], []
    for N in normals:
        for g in N @ R:
            A.append(g)
            b.append(g @ C)
        A.append(-R[2])
        b.append(-zmin - R[2] @ C)
    return np.array(A), np.array(b)


def lp_box(A, b):
    lo, hi = np.zeros(3), np.zeros(3)
    for d in range(3):
        e = np.zeros(3)
        e[d] = 1
        r1 = linprog(e, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        r2 = linprog(-e, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        if r1.status == 2 or r2.status == 2:
            return None                          # empty
        lo[d] = r1.fun if r1.status == 0 else -np.inf
        hi[d] = -r2.fun if r2.status == 0 else np.inf
    return lo, hi


def _union_box(task):
    """box of the union over poses of (base rows) intersected with each pose's (or pose pair's) window rows."""
    k, A0, b0, cams = task                      # cams: list over poses of [(normals, R, C), ...]
    zmin = _W["zmin"]
    lo, hi, n_ok = np.full(3, np.inf), np.full(3, -np.inf), 0
    for views in cams:
        A, b = [A0], [b0]
        for normals, R, C in views:
            Ak, bk = cam_rows(normals, R, C, zmin)
            A.append(Ak)
            b.append(bk)
        bx = lp_box(np.vstack(A), np.concatenate(b))
        if bx is None:
            continue
        n_ok += 1
        lo, hi = np.minimum(lo, bx[0]), np.maximum(hi, bx[1])
    return k, lo, hi, n_ok


def box_rows(lo, hi):
    A = np.vstack([np.eye(3), -np.eye(3)])
    return A, np.r_[hi, -lo]


def volume(A, b):
    c, r = chebyshev(A, b)
    if c is None or r < 1e-10:
        return 0.0
    try:
        return float(ConvexHull(HalfspaceIntersection(np.c_[A, -b], c).intersections).volume)
    except Exception:
        return 0.0


def gt_points(sc):
    sys.path.insert(0, os.path.join(os.path.dirname(SCENE), "chair", "COLMAP results", "python-codes"))
    import cp_conformal as CC
    from cp_step0_align import umeyama
    gt = CC.load_gt(SCENE)
    ids = [i for i in sc.imgs if sc.imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[sc.imgs[i].name]["c"] for i in ids]), np.array([sc.imgs[i].center for i in ids]))
    lifted = {}
    for rw in CC.observations(sc.cams, sc.imgs, sc.pts, gt, s_, Ra, ta):
        if rw["valid"]:
            lifted.setdefault(rw["pid"], []).append(rw["Xgt"])
    g1 = gt[sc.imgs[sc.i1].name]
    scl = (g1["R"] @ (gt[sc.imgs[sc.i2].name]["c"] - g1["c"]))[0]
    return {k: g1["R"] @ (np.median(v, 0) - g1["c"]) / scl for k, v in lifted.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"])
    ap.add_argument("--images", default="17,73")
    ap.add_argument("--directions", type=int, default=200)
    ap.add_argument("--union-poses", type=int, default=300, help="sampled feasible poses used for the unions")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    T0 = time.time()
    SRC = results_dir(a.scheme)
    OUT = os.path.join(HERE, "results_union" + ("" if a.scheme == "fixed" else "_" + a.scheme))
    os.makedirs(OUT, exist_ok=True)
    P = np.load(os.path.join(SRC, "pair_sets.npz"))
    zmin = float(P["z_min"])
    order = [int(x) for x in a.images.split(",")]
    sc = Scene(pair=(162, 15), extra=tuple(order))
    i1, i2 = sc.i1, sc.i2
    fixed = {i1: sc.pose(i1), i2: sc.pose(i2)}
    Xgt = gt_points(sc)
    sets = {int(k): (P["G"][P["row_landmark"] == k], P["h"][P["row_landmark"] == k]) for k in P["landmarks"]}
    origin = {k: "pair" for k in sets}
    posesets = {}                                 # image -> list of (R, C) sampled feasible poses
    report = {"eps_px": a.eps, "scheme": a.scheme, "images": [],
              "units": "w: rotation offset from COLMAP's (deg); t = -R C and C in the gauge (camera 1 at the origin, x of camera 2's centre = 1)"}

    def normals_for(i, k):
        return [box_normals(xn, sc.f[i], e) for xn, e in zip(sc.det[i][k], sc.eps_list(i, k, a.eps, a.scheme))]

    def fixed_rows_for(k, cams):
        A, b = [], []
        for c in cams:
            for xn, e in zip(sc.det[c][k], sc.eps_list(c, k, a.eps, a.scheme)):
                g, h = window_rows(*fixed[c], xn, sc.f[c], e, zmin)
                A.append(g)
                b.append(h)
        return np.vstack(A), np.concatenate(b)

    def gt_in(k, A, b):
        return bool(np.all(A @ Xgt[k] <= b + 1e-9)) if k in Xgt else None

    for step, i in enumerate(order, start=3):
        t_img = time.time()
        L = [k for k in sets if k in sc.det[i]]
        normals = {k: normals_for(i, k) for k in L}
        Gs = np.vstack([sets[k][0] for k in L])
        hs = np.concatenate([sets[k][1] for k in L])
        rowk = np.concatenate([[k] * len(sets[k][1]) for k in L])
        S = BP.System(L, Gs, hs, rowk, normals, zmin)
        Rc, Cc = sc.pose(i)
        R_gt, C_gt = sc.gt_pose(i)
        info = {"image": sc.imgs[i].name, "image_number": step, "landmarks_with_sets": len(L)}
        print("\n== image %d = %s: %d landmarks with sets" % (step, sc.imgs[i].name, len(L)), flush=True)
        start = Rc if S.feasible(Rc) else None
        gt_feas = bool(S.feasible(R_gt)) and all(
            feasible(np.vstack([sets[k][0], cam_rows(normals[k], R_gt, C_gt, zmin)[0]]),
                     np.r_[sets[k][1], cam_rows(normals[k], R_gt, C_gt, zmin)[1]]) for k in L)
        if start is None:
            info["status"] = "COLMAP rotation infeasible with the current sets"
            print("   " + info["status"])
            report["images"].append(info)
            break

        # ---- 1. pose set by radial search -------------------------------------------------------
        with Pool(a.procs, initializer=BP._init, initargs=(S, start)) as pool:
            res = pool.map(BP._ray, list(BP.fibonacci(a.directions)), chunksize=4)
        poses = [(Rc, Cc)] + [p for r in res for p in r[1]]
        Rs = np.array([p[0] for p in poses])
        Cs = np.array([p[1] for p in poses])
        W = np.degrees([Rotation.from_matrix(R @ Rc.T).as_rotvec() for R in Rs])
        T = np.array([-R @ C for R, C in zip(Rs, Cs)])
        w_gt = np.degrees(Rotation.from_matrix(R_gt @ Rc.T).as_rotvec())
        inside = lambda x, lo, hi: bool(np.all((x >= lo - 1e-9) & (x <= hi + 1e-9)))
        info["pose_set"] = {"poses": len(poses), "w_lo_deg": W.min(0).tolist(), "w_hi_deg": W.max(0).tolist(),
                            "t_lo": T.min(0).tolist(), "t_hi": T.max(0).tolist(),
                            "C_lo": Cs.min(0).tolist(), "C_hi": Cs.max(0).tolist(),
                            "gt_pose_exactly_feasible": gt_feas,
                            "gt_rotation_in_range": inside(w_gt, W.min(0), W.max(0)),
                            "gt_centre_in_range": inside(C_gt, Cs.min(0), Cs.max(0)),
                            "gt_w_rel_colmap_deg": w_gt.tolist(), "gt_t": (-R_gt @ C_gt).tolist()}
        print("   pose set: %d feasible poses; w %s .. %s deg; C %s .. %s; GT pose feasible: %s, GT rotation in "
              "range: %s, GT centre in range: %s" % (len(poses), np.round(W.min(0), 2), np.round(W.max(0), 2),
                                                     np.round(Cs.min(0), 3), np.round(Cs.max(0), 3), gt_feas,
                                                     info["pose_set"]["gt_rotation_in_range"],
                                                     info["pose_set"]["gt_centre_in_range"]), flush=True)
        with open(os.path.join(OUT, "image%d_poses.csv" % step), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["w_x_deg", "w_y_deg", "w_z_deg", "t_1", "t_2", "t_3", "C_x", "C_y", "C_z"])
            for n in range(len(poses)):
                wr.writerow([*np.round(W[n], 5), *np.round(T[n], 6), *np.round(Cs[n], 6)])
        dist = np.linalg.norm(W, axis=1)
        pick = np.argsort(dist)[np.unique(np.linspace(0, len(poses) - 1, min(a.union_poses, len(poses))).astype(int))]
        pick = np.r_[0, pick[pick != 0]]
        sample = [(Rs[n], Cs[n]) for n in pick]
        posesets[i] = sample

        # ---- 2. existing landmarks: union of slices ---------------------------------------------
        data = {"zmin": zmin}
        tasks = [(k, sets[k][0], sets[k][1], [[(normals[k], R, C)] for R, C in sample]) for k in L]
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            out = pool.map(_union_box, tasks, chunksize=2)
        new_sets, tuples_vol = {}, np.zeros(len(sample))
        for k, lo, hi, n_ok in out:
            if n_ok == 0 or not np.all(np.isfinite(np.r_[lo, hi])):
                new_sets[k] = sets[k]
                continue
            A_b, b_b = box_rows(lo, hi)
            new_sets[k] = (np.vstack([sets[k][0], A_b]), np.r_[sets[k][1], b_b])
        vol_before = np.array([volume(*sets[k]) for k in L])
        vol_after = np.array([volume(*new_sets[k]) for k in L])
        gt_before = [gt_in(k, *sets[k]) for k in L]
        gt_after = [gt_in(k, *new_sets[k]) for k in L]
        # (pose, volume) tuples
        for m, (R, C) in enumerate(sample):
            tuples_vol[m] = sum(volume(np.vstack([sets[k][0], cam_rows(normals[k], R, C, zmin)[0]]),
                                       np.r_[sets[k][1], cam_rows(normals[k], R, C, zmin)[1]])
                                for k in L[::4])
        with open(os.path.join(OUT, "image%d_tuples.csv" % step), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["w_x_deg", "w_y_deg", "w_z_deg", "t_1", "t_2", "t_3", "slice_volume_total_every_4th_landmark"])
            for m, n in enumerate(pick):
                wr.writerow([*np.round(W[n], 5), *np.round(T[n], 6), "%.6e" % tuples_vol[m]])
        valid = lambda xs: [x for x in xs if x is not None]
        info["existing_landmarks"] = {
            "count": len(L), "volume_after_over_before_median": float(np.median(vol_after / np.maximum(vol_before, 1e-30))),
            "gt_in_set_before": int(sum(valid(gt_before))), "gt_in_set_after": int(sum(valid(gt_after))),
            "with_gt": len(valid(gt_before))}
        print("   existing landmarks: volume after / before median %.3f; GT point inside before %d -> after %d (of %d with GT)"
              % (info["existing_landmarks"]["volume_after_over_before_median"], sum(valid(gt_before)),
                 sum(valid(gt_after)), len(valid(gt_before))), flush=True)
        sets.update(new_sets)

        # ---- 3. new landmarks: seen by i and exactly one fixed camera --------------------------------
        cand = [k for k in sc.det[i] if k not in sets and ((k in sc.det[i1]) != (k in sc.det[i2]))]
        tasks = []
        for k in cand:
            cams = [c for c in (i1, i2) if k in sc.det[c]]
            A0, b0 = fixed_rows_for(k, cams)
            tasks.append((k, A0, b0, [[(normals_for(i, k), R, C)] for R, C in sample]))
        # ---- 4. landmarks seen by this image and an earlier free image only ---------------------------
        earlier = [j for j in order[:step - 3]]
        pair_tasks = []
        for j in earlier:
            for k in sc.det[i]:
                if k in sets or k in sc.det[i1] or k in sc.det[i2] or k not in sc.det[j] or k in cand:
                    continue
                rng = np.random.default_rng(k)
                pairs = [(posesets[j][0], sample[0])] + [
                    (posesets[j][rng.integers(len(posesets[j]))], sample[rng.integers(len(sample))])
                    for _ in range(len(sample) - 1)]
                pair_tasks.append((k, np.zeros((0, 3)), np.zeros(0),
                                   [[(normals_for(j, k), Rj, Cj), (normals_for(i, k), Ri, Ci)]
                                    for (Rj, Cj), (Ri, Ci) in pairs]))
        added = {"one_fixed_camera": [], "two_free_cameras": []}
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            for group, tk in (("one_fixed_camera", tasks), ("two_free_cameras", pair_tasks)):
                if not tk:
                    continue
                for (k, lo, hi, n_ok), task in zip(pool.map(_union_box, tk, chunksize=2), tk):
                    if n_ok == 0 or not np.all(np.isfinite(np.r_[lo, hi])):
                        continue
                    A_b, b_b = box_rows(lo, hi)
                    sets[k] = (np.vstack([task[1], A_b]) if len(task[1]) else A_b,
                               np.r_[task[2], b_b] if len(task[2]) else b_b)
                    origin[k] = "image%d_%s" % (step, group)
                    added[group].append(k)
        for group, ks in added.items():
            g = valid([gt_in(k, *sets[k]) for k in ks])
            diag = [float(np.linalg.norm(-sets[k][1][3:6] - sets[k][1][0:3])) if False else
                    float(np.linalg.norm(lp_box(*sets[k])[1] - lp_box(*sets[k])[0])) for k in ks[:200]]
            info["new_landmarks_" + group] = {"count": len(ks), "with_gt": len(g), "gt_in_set": int(sum(g)),
                                              "box_diag_median": float(np.median(diag)) if diag else None}
            if ks:
                print("   new landmarks (%s): %d, box diagonal median %.3f; GT point inside %d of %d with GT"
                      % (group, len(ks), np.median(diag), sum(g), len(g)), flush=True)
        np.savez_compressed(os.path.join(OUT, "sets_after_image%d.npz" % step), landmarks=np.array(list(sets)),
                            origin=np.array([origin[k] for k in sets]),
                            **{"G_%d" % k: sets[k][0] for k in sets}, **{"h_%d" % k: sets[k][1] for k in sets})
        info["seconds"] = round(time.time() - t_img)
        report["images"].append(info)

    report["seconds"] = round(time.time() - T0)
    json.dump(report, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\n-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
