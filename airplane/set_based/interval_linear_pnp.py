"""Step 3: sound pose intervals for image 3 (linearisation + interval remainder, SIVIA).

    python interval_linear_pnp.py [--eps 3] [--w0-deg 8] [--leaf-deg 1] [--c0 0.8]

Unknowns: camera-3 pose beta = (w, C), with R = exp([w]x) R_colmap; landmarks X_k in F_k.
Constraint (every window, multiplied by the depth, so EXACTLY linear in X and C):
    n_j^T R (X_k - C) <= 0  (4 faces),   r_3^T (X_k - C) >= z_min.
Only the rotation is nonlinear. In a rotation box with centre w0 (R0 = exp([w0]x) R_colmap) every
rotation is R = exp([w']x) R0 with ||w'|| <= delta (Hartley-Kahl, delta = norm of the half-widths), and
    R (X - C) = q + w' x q0 + w' x (q - q0) + r,      q = R0 (X - C),  q0 = R0 (X0 - C0),
    ||r|| <= g(delta) ||q||,  g(t) = t - sin t + 1 - cos t      (Rodrigues; Lagrange-type remainder).
The first two terms are linear (the Jacobian part, as in Eq. 15 of the RAIM paper); the others are
bounded by an interval over the box:
    n^T q + (q0 x n)^T w'  <=  delta (Delta_X + Delta_C) + g(delta) Q
    Delta_X >= ||X - X0|| on the box of F_k,  Delta_C >= ||C - C0|| on the centre box,  Q >= ||X - C||.
Per box (one LP system over w', C, X):
    feasible?  no  -> discard (no pose in the box satisfies every window, for any point in F_k)
               yes -> contract the centre box to its LP range (sound), repeat with the smaller slack.
SIVIA: split the widest rotation interval; keep boxes whose rotation intervals are <= leaf width.
Output: the hull of the kept boxes -> one interval per pose parameter; every pose consistent with all
windows (for some points in the sets) lies inside, provided the true pose is inside the initial box.
Writes results/interval_pose.json and results/interval_leaves.npz.
"""
import os
import json
import math
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog

from setlib import Scene, box_normals, expR, RESULTS
from scipy.spatial.transform import Rotation

E3 = np.array([0, 0, 1.0])


def g_rem(t):
    return t - math.sin(t) + 1 - math.cos(t)


def maxdist(lo, hi, p):
    return float(np.linalg.norm(np.maximum(np.abs(lo - p), np.abs(hi - p))))


class Box:
    """per-box LP over (w' (3), C (3), X_k (3 each))."""

    def __init__(self, L, G, h, rowk, Flo, Fhi, X0, normals, zmin, Rc):
        self.L, self.normals, self.zmin, self.Rc = L, normals, zmin, Rc
        self.Flo, self.Fhi, self.X0 = Flo, Fhi, X0
        self.idx = {k: 6 + 3 * n for n, k in enumerate(L)}
        self.n = 6 + 3 * len(L)
        rows, b = [], []
        for k in L:
            sel = rowk == k
            for gg, hh in zip(G[sel], h[sel]):
                r = np.zeros(self.n)
                r[self.idx[k]:self.idx[k] + 3] = gg
                s = np.linalg.norm(gg)
                rows.append(r / s)
                b.append(hh / s)
        self.AF, self.bF = np.array(rows), np.array(b)

    def system(self, w0, delta, clo, chi):
        R0 = expR(w0) @ self.Rc
        c0 = 0.5 * (clo + chi)
        dC = maxdist(clo, chi, c0)
        gd = g_rem(delta)
        rows, b = [], []
        for k in self.L:
            iv = self.idx[k]
            q0 = R0 @ (self.X0[k] - c0)
            dX = maxdist(self.Flo[k], self.Fhi[k], self.X0[k])
            Q = float(np.linalg.norm(np.maximum(np.abs(self.Fhi[k] - clo), np.abs(self.Flo[k] - chi))))
            s = delta * (dX + dC) + gd * Q
            for N in self.normals[k]:
                for nrm in N:
                    r = np.zeros(self.n)
                    r[0:3] = np.cross(q0, nrm)
                    r[3:6] = -(nrm @ R0)
                    r[iv:iv + 3] = nrm @ R0
                    rows.append(r)
                    b.append(s)
                r = np.zeros(self.n)                      # depth: q3 >= z_min (with slack)
                r[0:3] = -np.cross(q0, E3)
                r[3:6] = R0[2]
                r[iv:iv + 3] = -R0[2]
                rows.append(r)
                b.append(-self.zmin + s)
        A = np.vstack([self.AF, np.array(rows)])
        bb = np.concatenate([self.bF, np.array(b)])
        bounds = ([(-delta, delta)] * 3 + list(zip(clo, chi)) + [(None, None)] * (self.n - 6))
        return A, bb, bounds


def lp_feasible(A, b, bounds):
    A2 = np.hstack([A, -np.ones((len(A), 1))])
    c = np.zeros(A.shape[1] + 1)
    c[-1] = 1
    r = linprog(c, A_ub=A2, b_ub=b, bounds=bounds + [(0, None)], method="highs")
    return bool(r.status == 0 and r.fun <= 1e-9)


_B = {}


def _init(box):
    _B["box"] = box


def _test(item):
    """returns None (discarded) or the contracted centre box."""
    wlo, whi, clo, chi = item
    B = _B["box"]
    w0 = 0.5 * (wlo + whi)
    delta = float(np.linalg.norm(0.5 * (whi - wlo)))
    for it in range(2):
        A, b, bounds = B.system(w0, delta, clo, chi)
        if not lp_feasible(A, b, bounds):
            return None
        nlo, nhi = clo.copy(), chi.copy()
        for j in range(3):
            for sg in (1.0, -1.0):
                c = np.zeros(B.n)
                c[3 + j] = sg
                r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
                if r.status != 0:
                    return None
                if sg > 0:
                    nlo[j] = max(nlo[j], r.fun - 1e-9)
                else:
                    nhi[j] = min(nhi[j], -r.fun + 1e-9)
        shrink = float(np.max((chi - clo) - (nhi - nlo)))
        clo, chi = nlo, nhi
        if shrink < 1e-4:
            break
    return clo, chi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--w0-deg", type=float, default=8.0, help="initial rotation half-width around COLMAP's")
    ap.add_argument("--c0", type=float, default=0.8, help="initial centre half-width around COLMAP's")
    ap.add_argument("--leaf-deg", type=float, default=1.0, help="leaf width of each rotation interval")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    T0 = time.time()
    P = np.load(os.path.join(RESULTS, "pair_sets.npz"))
    sc = Scene(pair=(162, 15), extra=(a.third,))
    f3, zmin = sc.f[a.third], float(P["z_min"])
    lm = [int(k) for k in P["landmarks"]]
    pos = {k: n for n, k in enumerate(lm)}
    L = [k for k in lm if k in sc.det[a.third]]
    normals = {k: [box_normals(xn, f3, a.eps) for xn in sc.det[a.third][k]] for k in L}
    Rc, Cc = sc.pose(a.third)
    box = Box(L, P["G"], P["h"], P["row_landmark"],
              {k: P["lo"][pos[k]] for k in L}, {k: P["hi"][pos[k]] for k in L},
              {k: P["centre"][pos[k]] for k in L}, normals, zmin, Rc)
    w0 = math.radians(a.w0_deg)
    leaf = math.radians(a.leaf_deg)
    level = [(-w0 * np.ones(3), w0 * np.ones(3), Cc - a.c0, Cc + a.c0)]
    kept, log, tested = [], [], 0
    print("image 3 = %s, %d landmarks; initial box: rotation +-%g deg around COLMAP's, centre COLMAP +-%g; "
          "leaf %g deg" % (sc.imgs[a.third].name, len(L), a.w0_deg, a.c0, a.leaf_deg), flush=True)
    with Pool(a.procs, initializer=_init, initargs=(box,)) as pool:
        while level:
            t = time.time()
            res = pool.map(_test, level, chunksize=1)
            tested += len(level)
            survive = [(it[0], it[1], *r) for it, r in zip(level, res) if r is not None]
            width = math.degrees(float(np.max(level[0][1] - level[0][0])))
            delta = math.degrees(float(np.linalg.norm(0.5 * (level[0][1] - level[0][0]))))
            log.append({"rotation_width_deg": width, "tested": len(level), "kept": len(survive)})
            print("  rotation width %6.3f deg (delta %.3f): %5d tested, %5d kept  (%.0fs)"
                  % (width, delta, len(level), len(survive), time.time() - t), flush=True)
            nxt = []
            for wlo, whi, clo, chi in survive:
                wd = whi - wlo
                if wd.max() <= leaf + 1e-12:
                    kept.append((wlo, whi, clo, chi))
                    continue
                j = int(np.argmax(wd))
                mid = 0.5 * (wlo[j] + whi[j])
                for lo_j, hi_j in ((wlo[j], mid), (mid, whi[j])):
                    nlo, nhi = wlo.copy(), whi.copy()
                    nlo[j], nhi[j] = lo_j, hi_j
                    nxt.append((nlo, nhi, clo, chi))
            level = nxt
    if not kept:
        print("empty: no pose in the initial box satisfies every window at eps = %g" % a.eps)
        return
    W = np.array([[k[0], k[1]] for k in kept])            # (n, 2, 3)
    C = np.array([[k[2], k[3]] for k in kept])
    w_lo, w_hi = np.degrees(W[:, 0].min(0)), np.degrees(W[:, 1].max(0))
    c_lo, c_hi = C[:, 0].min(0), C[:, 1].max(0)
    delta_max = math.degrees(max(float(np.linalg.norm(np.maximum(np.abs(k[0]), np.abs(k[1])))) for k in kept))
    touches = bool(np.any(np.isclose(np.abs(W), w0)) or np.any(np.isclose(C[:, 0], Cc - a.c0))
                   or np.any(np.isclose(C[:, 1], Cc + a.c0)))
    R_gt, C_gt = sc.gt_pose(a.third)
    w_gt = Rotation.from_matrix(R_gt @ Rc.T).as_rotvec()
    in_leaf = any(np.all((w_gt >= k[0] - 1e-12) & (w_gt <= k[1] + 1e-12)) and
                  np.all((C_gt >= k[2] - 1e-9) & (C_gt <= k[3] + 1e-9)) for k in kept)
    names = ["w_x (deg)", "w_y (deg)", "w_z (deg)", "c_x", "c_y", "c_z"]
    lo_all, hi_all = np.r_[w_lo, c_lo], np.r_[w_hi, c_hi]
    gt_all = np.r_[np.degrees(w_gt), C_gt]
    col_all = np.r_[np.zeros(3), Cc]
    print("\nsound intervals (rotation relative to COLMAP's; centre in the gauge):")
    for n_, lo_, hi_, g_, c_ in zip(names, lo_all, hi_all, gt_all, col_all):
        print("  %-10s [%+8.3f, %+8.3f]   COLMAP %+8.3f   GT %+8.3f   GT inside: %s"
              % (n_, lo_, hi_, c_, g_, lo_ <= g_ <= hi_))
    print("rotation within %.2f deg of COLMAP's; GT pose inside a kept box: %s; set touches the initial "
          "box boundary: %s" % (delta_max, in_leaf, touches))
    rep = {"eps_px": a.eps, "image3": sc.imgs[a.third].name, "initial_box": {"w_half_deg": a.w0_deg, "c_half": a.c0},
           "leaf_deg": a.leaf_deg, "boxes_tested": tested, "boxes_kept": len(kept), "levels": log,
           "intervals": {n_: [float(lo_), float(hi_)] for n_, lo_, hi_ in zip(names, lo_all, hi_all)},
           "rotation_max_from_colmap_deg": delta_max, "colmap_centre": Cc.tolist(),
           "gt": {"w_rel_colmap_deg": np.degrees(w_gt).tolist(), "centre": C_gt.tolist(), "inside_a_kept_box": bool(in_leaf)},
           "touches_initial_box": touches, "seconds": round(time.time() - T0),
           "guarantee": "every pose that reprojects every landmark within eps, for some point of its set F_k, "
                        "lies in the union of kept boxes (hence in these intervals), if it lies in the initial box"}
    json.dump(rep, open(os.path.join(RESULTS, "interval_pose.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, "interval_leaves.npz"), w=W, c=C)
    print("-> %s  (%ds)" % (RESULTS, time.time() - T0))


if __name__ == "__main__":
    main()
