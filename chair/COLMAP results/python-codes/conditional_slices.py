"""Conditional landmark slices for camera 3: S_k(theta) = F_k  intersected with  cone_k(theta).

    python conditional_slices.py --scene airplane --eps 3

For a pose theta = (R, C) of camera 3, the slice of landmark k is the part of its set F_k (from
cameras 1, 2) that projects within eps of camera 3's detection(s): every point of the slice is
consistent with that pose ("for every pose, for every point of its slice").

Feasible poses are sampled EXACTLY (fixed-rotation LP, no linearisation):
    rotations: from the P3P reference R_hat, along 300 directions, the boundary angle t_b found by
               bisection, and t_b / 2 (interior);
    centres:   for each such rotation, the 6 extreme centres (min / max of each coordinate) and their
               mean (feasible by convexity for a fixed rotation);
    plus COLMAP's pose.
Reported:
  1. slice size at COLMAP's pose vs the size of F_k (is the slice smaller?);
  2. intersection of the slices over feasible poses within rotation radius r of COLMAP's rotation,
     for growing r: is it empty? Over SAMPLED poses the intersection is a superset of the intersection
     over all feasible poses, so "empty" is certain, "non-empty" is an upper bound.
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
from stage_a import undistort
from pnp_set import box_normals, fixed_rows, expR
import linpnp_set as LS

_D = {}


def _init(data, Rhat):
    _D.update(data=data, Rhat=Rhat)


def solve(A, b, bnds, c):
    r = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method="highs")
    return r.x if r.status == 0 else None


def _direction(u):
    """exact boundary along u, then feasible centres at t_b and t_b / 2."""
    D, Rh = _D["data"], _D["Rhat"]
    feas = lambda t: LS.lp_feasible(*D.rows_exact(expR(t * u) @ Rh))
    lo, hi = 0.0, math.radians(10.0)
    if feas(hi):
        lo = hi
    else:
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if feas(mid) else (lo, mid)
    out = []
    for t in (lo, 0.5 * lo):
        R = expR(t * u) @ Rh
        A, b, bnds = D.rows_exact(R)
        cs = []
        for j in range(3):
            for sg in (1.0, -1.0):
                c = np.zeros(D.n)
                c[3 + j] = sg
                x = solve(A, b, bnds, c)
                if x is not None:
                    cs.append(x[3:6])
        if cs:
            cs.append(np.mean(cs, 0))
            out += [(R, C) for C in cs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0, help="window in image 3")
    ap.add_argument("--eps-f", type=float, default=None, help="window in images 1, 2 (size of F_k); default = eps")
    ap.add_argument("--pair", default="162,15")
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--directions", type=int, default=300)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    eps_f = a.eps if a.eps_f is None else a.eps_f
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "pnp_set", "conditional_slices_eps%g_epsf%g_img%d" % (a.eps, eps_f, a.third))
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()

    # ---- setup (as linpnp_set.py) -----------------------------------------------------------
    cams, imgs, pts = read_model(MODEL)
    i1, i2 = [int(x) for x in a.pair.split(",")]
    i3 = a.third
    intr = {i: cams[imgs[i].camera_id].unpack() for i in (i1, i2, i3)}
    f = {i: (intr[i][0], intr[i][1]) for i in intr}
    R1, c1 = imgs[i1].R, imgs[i1].center
    scale = (R1 @ (imgs[i2].center - c1))[0]
    pose = {i1: (np.eye(3), np.zeros(3)), i2: (imgs[i2].R @ R1.T, R1 @ (imgs[i2].center - c1) / scale)}
    R_col = imgs[i3].R @ R1.T
    c_col = R1 @ (imgs[i3].center - c1) / scale
    Xg = {k: R1 @ (p.xyz - c1) / scale for k, p in pts.items()}
    det = {}
    for i in (i1, i2, i3):
        d = {}
        for j in np.flatnonzero(imgs[i].p3d_ids >= 0):
            d.setdefault(int(imgs[i].p3d_ids[j]), []).append(undistort(*imgs[i].xys[j], intr[i]))
        det[i] = d
    common = sorted(set(det[i1]) & set(det[i2]))
    zmin = 1e-3 * float(np.median([pose[i][0][2] @ (Xg[k] - pose[i][1]) for k in common for i in (i1, i2)]))
    Fk, Fbox, X0 = {}, {}, {}
    for k in common:
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i][k]:
                g_, h_ = fixed_rows(*pose[i], xn, f[i], eps_f, zmin)
                G.append(g_)
                H.append(h_)
        G, H = np.vstack(G), np.concatenate(H)
        res = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[G, np.linalg.norm(G, axis=1)], b_ub=H,
                      bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if res.status != 0 or res.x[3] <= 0:
            continue
        bb = []
        for dd in range(3):
            e = np.zeros(3)
            e[dd] = 1
            r1 = linprog(e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
            r2 = linprog(-e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
            bb.append([r1.fun, -r2.fun] if r1.status == 0 and r2.status == 0 else [-np.inf, np.inf])
        bb = np.array(bb)
        if np.all(np.isfinite(bb)):
            Fk[k], Fbox[k], X0[k] = (G, H), (bb[:, 0], bb[:, 1]), res.x[:3]
    L3 = sorted(k for k in Fk if k in det[i3])
    obs3 = [(k, box_normals(xn, f[i3], a.eps)[0]) for k in L3 for xn in det[i3][k]]
    data = LS.Data(L3, Fk, Fbox, X0, obs3, zmin)
    prev = os.path.join(MODEL, "pnp_set", "eps%g_img%d_d05_h0.1" % (a.eps, i3))
    Rhat = np.array(json.load(open(os.path.join(prev, "sets.json")))["R_hat3"])
    normals = {k: [box_normals(xn, f[i3], a.eps)[0] for xn in det[i3][k]] for k in L3}

    def slice_rows(k, R, C):
        """camera-3 rows of landmark k for a fixed pose: A x <= b."""
        A, b = [], []
        for N in normals[k]:
            for nrm in N:
                g = nrm @ R
                A.append(g)
                b.append(g @ C)
            A.append(-R[2])
            b.append(-zmin - R[2] @ C)
        return np.array(A), np.array(b)

    def box_of(A, b):
        lo, hi = np.zeros(3), np.zeros(3)
        for d in range(3):
            e = np.zeros(3)
            e[d] = 1
            r1 = linprog(e, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
            r2 = linprog(-e, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
            if r1.status != 0 or r2.status != 0:
                return None
            lo[d], hi[d] = r1.fun, -r2.fun
        return lo, hi

    # ---- 1. slice at COLMAP's pose vs F_k ----------------------------------------------------------
    col_ok = LS.lp_feasible(*data.rows_exact(R_col)) and all(
        linprog(np.zeros(3), A_ub=np.vstack([Fk[k][0], slice_rows(k, R_col, c_col)[0]]),
                b_ub=np.r_[Fk[k][1], slice_rows(k, R_col, c_col)[1]],
                bounds=[(None, None)] * 3, method="highs").status == 0 for k in L3)
    sizes = []
    for k in L3:
        A_s, b_s = slice_rows(k, R_col, c_col)
        bx = box_of(np.vstack([Fk[k][0], A_s]), np.r_[Fk[k][1], b_s])
        lo0, hi0 = Fbox[k]
        if bx is not None:
            sizes.append((np.linalg.norm(hi0 - lo0), np.linalg.norm(bx[1] - bx[0]),
                          (hi0 - lo0).max(), (bx[1] - bx[0]).max()))
    sizes = np.array(sizes)
    print("1. COLMAP pose feasible with every landmark: %s" % col_ok)
    print("   F_k box diagonal median %.4f -> slice at COLMAP's pose %.4f (ratio median %.3f); "
          "longest side %.4f -> %.4f (%d landmarks)"
          % (np.median(sizes[:, 0]), np.median(sizes[:, 1]), np.median(sizes[:, 1] / sizes[:, 0]),
             np.median(sizes[:, 2]), np.median(sizes[:, 3]), len(sizes)), flush=True)

    # ---- sample feasible poses (exact) ----------------------------------------------------------------
    t = time.time()
    U = LS.fibonacci(a.directions)
    with Pool(a.procs, initializer=_init, initargs=(data, Rhat)) as pool:
        res = pool.map(_direction, list(U), chunksize=4)
    poses = [(R_col, c_col)] + [p for r in res for p in r]
    ang = np.array([math.degrees(Rotation.from_matrix(R @ R_col.T).magnitude()) for R, _ in poses])
    np.savez_compressed(os.path.join(OUT, "poses.npz"), R=np.array([p[0] for p in poses]),
                        C=np.array([p[1] for p in poses]), angle_deg=ang)
    print("   sampled %d exactly feasible poses (rotation up to %.2f deg from COLMAP's) in %.0fs"
          % (len(poses), ang.max(), time.time() - t), flush=True)

    # ---- 2. intersection of the slices over poses within radius r ------------------------------------
    radii = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, float(ang.max()) + 1e-6]
    table = []
    for r in radii:
        sel = [p for p, an in zip(poses, ang) if an <= r]
        nonempty, diag = 0, []
        for k in L3:
            rows = [Fk[k][0]]
            rhs = [Fk[k][1]]
            for R, C in sel:
                A_s, b_s = slice_rows(k, R, C)
                rows.append(A_s)
                rhs.append(b_s)
            A, b = np.vstack(rows), np.concatenate(rhs)
            ok = linprog(np.zeros(3), A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs").status == 0
            if ok:
                nonempty += 1
                bx = box_of(A, b)
                if bx is not None:
                    diag.append(np.linalg.norm(bx[1] - bx[0]))
        row = {"rotation_radius_deg": r, "poses": len(sel), "landmarks_nonempty": nonempty,
               "landmarks": len(L3), "intersection_diag_median": float(np.median(diag)) if diag else None}
        table.append(row)
        print("2. poses within %5.2f deg of COLMAP (%4d poses): intersection non-empty for %3d / %d landmarks; "
              "median diagonal %s" % (r, len(sel), nonempty, len(L3),
                                      ("%.4f" % row["intersection_diag_median"]) if diag else "-"), flush=True)

    rep = {"eps": a.eps, "eps_f": eps_f, "colmap_pose_feasible": bool(col_ok),
           "Fk_diag_median": float(np.median(sizes[:, 0])), "slice_at_colmap_diag_median": float(np.median(sizes[:, 1])),
           "slice_over_Fk_diag_ratio_median": float(np.median(sizes[:, 1] / sizes[:, 0])),
           "Fk_longest_side_median": float(np.median(sizes[:, 2])),
           "slice_longest_side_median": float(np.median(sizes[:, 3])),
           "sampled_poses": len(poses), "intersection_by_radius": table,
           "note": "intersection over sampled poses is a superset of the intersection over all feasible poses",
           "seconds": round(time.time() - T0)}
    json.dump(rep, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
