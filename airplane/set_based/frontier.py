"""Trade-off frontier between a ball of camera-3 poses and the points that fit every pose in it.

    python frontier.py [--eps 3] [--samples 200]

Pose ball of radius r around COLMAP's pose:
    P(r) = feasible poses (R, C) with angle(R, R_colmap) <= r and ||C - C_colmap|| <= r * z_bar,
    z_bar = median landmark depth (a rotation r and a shift r * z_bar move the image by about the same).
Point set that fits every pose in the ball:
    S_k(r) = F_k  intersected with  S_k(theta) over theta in P(r)     (exact polytope intersection)
P(r) is sampled from one pool of exactly feasible poses (every landmark has a non-empty slice):
COLMAP's pose, random samples at every radius (80 % on a ball surface, 20 % inside) and the radial-search
poses of backproject_image3.py; ball r uses every pooled pose inside it, so the balls are nested. A sampled ball gives an
UPPER bound on the intersection over the continuous ball.
Reported per radius: feasible poses, landmarks with non-empty S_k(r), and vol(S_k(r)) / vol(F_k).
Knee: the radius where the normalised curve (r, mean volume ratio) is farthest from its chord.
Writes results/frontier.json, results/frontier.csv and results/frontier.png.
"""
import os
import csv
import json
import math
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setlib import Scene, box_normals, chebyshev, expR, results_dir

_D = {}


def _init(data):
    _D.update(data)


def cam_rows(k, R, C):
    A, b = [], []
    for N in _D["normals"][k]:
        for g in N @ R:
            A.append(g)
            b.append(g @ C)
        A.append(-R[2])
        b.append(-_D["zmin"] - R[2] @ C)
    return np.array(A), np.array(b)


def _pose_feasible(pose):
    R, C = pose
    for k in _D["L"]:
        G, h = _D["F"][k]
        A, b = cam_rows(k, R, C)
        r = linprog(np.zeros(3), A_ub=np.vstack([G, A]), b_ub=np.r_[h, b], bounds=[(None, None)] * 3,
                    method="highs")
        if r.status != 0:
            return False
    return True


def volume(A, b):
    c, rad = chebyshev(A, b)
    if c is None or rad < 1e-9:
        return 0.0
    try:
        V = HalfspaceIntersection(np.c_[A, -b], c).intersections
        return float(ConvexHull(V).volume)
    except Exception:
        return 0.0


def _intersection(args):
    k, poses = args
    G, h = _D["F"][k]
    A, b = [G], [h]
    for R, C in poses:
        Ak, bk = cam_rows(k, R, C)
        A.append(Ak)
        b.append(bk)
    return k, volume(np.vstack(A), np.concatenate(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--samples", type=int, default=200)
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
    f3, zmin = sc.f[a.third], float(P["z_min"])
    L = [int(k) for k in P["landmarks"] if int(k) in sc.det[a.third]]
    F = {k: (P["G"][P["row_landmark"] == k], P["h"][P["row_landmark"] == k]) for k in L}
    normals = {k: [box_normals(xn, f3, e) for xn, e in
                   zip(sc.det[a.third][k], sc.eps_list(a.third, k, a.eps, a.scheme, a.kappa, a.margin, a.cap))]
               for k in L}
    R0, C0 = sc.pose(a.third)
    zbar = float(np.median([np.linalg.norm(P["centre"][list(P["landmarks"]).index(k)] - C0) for k in L]))
    data = {"L": L, "F": F, "normals": normals, "zmin": zmin}
    _init(data)
    volF = {k: volume(*F[k]) for k in L}
    radii = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    rng = np.random.default_rng(0)
    rows = []
    print("image 3 = %s, %d landmarks; ball: rotation <= r, centre <= r * %.2f; %d samples per radius"
          % (sc.imgs[a.third].name, len(L), zbar, a.samples), flush=True)
    from scipy.spatial.transform import Rotation
    Q = np.load(os.path.join(RESULTS, "image3_poses.npz"))
    with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
        # pool of exactly feasible poses: random samples at every radius + the radial-search poses
        pool_R, pool_C = [R0], [C0]
        n_rand = 0
        for r_deg in radii[1:]:
            r = math.radians(r_deg)
            cand = []
            for s in range(a.samples):
                u = rng.normal(size=3)
                v = rng.normal(size=3)
                u /= np.linalg.norm(u)
                v /= np.linalg.norm(v)
                sr, sc_ = (1.0, 1.0) if s < 0.8 * a.samples else (rng.uniform() ** (1 / 3), rng.uniform() ** (1 / 3))
                cand.append((expR(sr * r * u) @ R0, C0 + sc_ * r * zbar * v))
            ok = pool.map(_pose_feasible, cand, chunksize=4)
            for (R, C), o in zip(cand, ok):
                if o:
                    pool_R.append(R)
                    pool_C.append(C)
                    n_rand += 1
        pool_R += list(Q["R"][1:])
        pool_C += list(Q["C"][1:])
        pool_R, pool_C = np.array(pool_R), np.array(pool_C)
        ang = np.degrees([Rotation.from_matrix(R @ R0.T).magnitude() for R in pool_R])
        dc = np.linalg.norm(pool_C - C0, axis=1)
        print("  pool: %d feasible random samples + %d radial-search poses" % (n_rand, len(Q["R"]) - 1), flush=True)
        for r_deg in radii:
            t = time.time()
            r = math.radians(r_deg)
            inside = np.flatnonzero((ang <= r_deg + 1e-9) & (dc <= r * zbar + 1e-12))
            poses = [(pool_R[i], pool_C[i]) for i in inside]
            res = dict(pool.map(_intersection, [(k, poses) for k in L], chunksize=2))
            ratio = np.array([res[k] / volF[k] if volF[k] > 0 else 0.0 for k in L])
            row = {"radius_deg": r_deg, "centre_radius": r * zbar, "poses_in_ball": len(poses),
                   "landmarks_nonempty": int((ratio > 0).sum()), "volume_ratio_mean": float(ratio.mean()),
                   "volume_ratio_median": float(np.median(ratio))}
            rows.append(row)
            print("  r = %4.2f deg (centre %.3f): %4d feasible poses in the ball; non-empty %3d / %d; "
                  "vol(S)/vol(F) mean %.3f median %.3f  (%.0fs)"
                  % (r_deg, r * zbar, len(poses), row["landmarks_nonempty"], len(L),
                     row["volume_ratio_mean"], row["volume_ratio_median"], time.time() - t), flush=True)
    rr = np.array([x["radius_deg"] for x in rows])
    vv = np.array([x["volume_ratio_mean"] for x in rows])
    x = (rr - rr.min()) / (rr.max() - rr.min())
    y = (vv - vv.min()) / max(vv.max() - vv.min(), 1e-12)
    dist = np.abs(x + y - 1) / math.sqrt(2)        # distance to the chord from (0, 1) to (1, 0)
    knee = int(np.argmax(dist))
    rep = {"eps_px": a.eps, "scheme": a.scheme, "image3": sc.imgs[a.third].name, "z_bar": zbar, "samples_per_radius": a.samples,
           "frontier": rows, "knee": rows[knee],
           "note": "sampled balls: intersections are upper bounds on the intersection over the continuous ball",
           "seconds": round(time.time() - T0)}
    json.dump(rep, open(os.path.join(RESULTS, "frontier.json"), "w"), indent=2)
    with open(os.path.join(RESULTS, "frontier.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(rr, vv, "-o", color="#2a78d6", lw=2, ms=5, label="mean vol(S) / vol(F_k)")
    ax[0].plot(rr, [x["volume_ratio_median"] for x in rows], "-s", color="#eb6834", lw=2, ms=4,
               label="median vol(S) / vol(F_k)")
    ax[0].axvline(rr[knee], color="#52514e", ls="--", lw=1)
    ax[0].text(rr[knee], vv.max() * 0.95, "  knee r = %.2f deg" % rr[knee], color="#52514e", fontsize=9)
    ax[0].set_xlabel("pose-ball radius r (deg; centre radius r x %.2f)" % zbar)
    ax[0].set_ylabel("points that fit every pose in the ball")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(rr, [x["landmarks_nonempty"] for x in rows], "-o", color="#1baf7a", lw=2, ms=5)
    ax[1].set_xlabel("pose-ball radius r (deg)")
    ax[1].set_ylabel("landmarks with a non-empty point set (of %d)" % len(L))
    for x_ in ax:
        x_.spines[["top", "right"]].set_visible(False)
        x_.grid(color="#e5e4df", lw=0.8)
    fig.suptitle("Image 3 (%s): pose ball vs points that fit every pose in it (eps = %g px)"
                 % (sc.imgs[a.third].name, a.eps), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "frontier.png"), dpi=120)
    print("knee: r = %.2f deg -> %s  (%ds)" % (rr[knee], RESULTS, time.time() - T0))


if __name__ == "__main__":
    main()
