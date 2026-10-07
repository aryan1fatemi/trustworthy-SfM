"""Set-valued PnP for the first image after the initial pair (rotation branch-and-bound).

    python pnp_set.py --scene airplane --eps 3 --h-min-deg 0.01

Setup   cameras 1, 2 = COLMAP's initial pair (#162, #15), poses held at their estimates;
        gauge R_1 = I, c_1 = 0, x-coordinate of c_2 = 1. For every landmark seen in both:
        F_k = {X : G_k X <= h_k} (four planes + depth per observation), kept as inequalities.
Step 1  reference rotation R_hat3: P3P + RANSAC + LM refinement on the Chebyshev centres of
        the F_k (search order only; COLMAP's estimate is reported alongside).
Step 2  rotations R(w) = exp([w]x) R_hat3, w in [-delta0, delta0]^3. delta0 = pi covers every
        rotation but does not prune (camera 3 can recede until all landmarks share one direction);
        a user-chosen delta0 makes the result conditional on the true rotation being within it.
Step 3  a cell (centre w0, half-width h) is tested with delta = sqrt(3) h. Each camera-3 face
        n^T q <= 0 (n unit) is widened to n^T q_hat <= sin(delta) ||q_hat||, and the cone is
        linearised conservatively: the true q lies inside its detection pyramid, whose corner
        angle from the optical axis is rho_ik, so angle(q_hat, axis) <= rho_ik + delta and
        ||q_hat|| <= q_hat_3 / cos(rho_ik + delta). Sound LP:
            n^T q_hat <= sin(delta) / cos(rho + delta) * q_hat_3,
            q_hat_3   >= z_min cos(rho + delta).
        Unknowns c and X_k (each X_k restricted to its own F_k). If rho + delta >= 90 deg the
        cell cannot be tested and is split.
Step 4  branch and bound: split feasible cells into 8, closest to w = 0 first, keep at h_min.
Step 5  per surviving cell, min/max of each coordinate of c; the box of all results.
Step 6  delta_3 = max (||w0|| + sqrt(3) h); clusters of surviving cells.
Step 7  empty result reported (no automatic outlier removal).
Step 8  camera 3 added with its centre free and rotation bound delta_3 around R_hat3; boxes of
        the common landmarks (shrink) and of new landmarks seen by camera 3 and camera 1 or 2.
Evaluation (GT, same gauge rule) is reported last and never used above. Note that cameras
1 and 2 are held at COLMAP's estimates, so the GT camera 3 is only approximately consistent
with this frame.
"""
import os
import sys
import json
import math
import time
import heapq
import argparse
import numpy as np
import cv2
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
import cp_conformal as CC
from stage_a import undistort

TOL = 1e-7           # phase-1 slack above which a cell is declared infeasible


# --------------------------------------------------------------------------
def box_normals(xn, f, eps):
    """camera-frame unit normals of the four faces, and the box-corner angle rho."""
    a_p, a_m = xn[0] + eps / f[0], xn[0] - eps / f[0]
    b_p, b_m = xn[1] + eps / f[1], xn[1] - eps / f[1]
    N = np.array([[1, 0, -a_p], [-1, 0, a_m], [0, 1, -b_p], [0, -1, b_m]], float)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    rho = max(math.atan(math.hypot(a, b)) for a in (a_p, a_m) for b in (b_p, b_m))
    return N, rho


def fixed_rows(R, c, xn, f, eps, zmin):
    """rows (g, h) with g^T X <= h for a camera with known pose."""
    N, _ = box_normals(xn, f, eps)
    G = [n @ R for n in N] + [-R[2]]
    H = [g @ c for g in G[:4]] + [-zmin - R[2] @ c]
    return np.array(G), np.array(H)


class Problem:
    """variables: c (3), then X_k for k in L (3 each)."""

    def __init__(self, L, Fk, obs3, zmin):
        self.L = L
        self.idx = {k: 3 + 3 * n for n, k in enumerate(L)}
        self.n = 3 + 3 * len(L)
        self.zmin = zmin
        rows, b = [], []
        for k in L:
            G, H = Fk[k]
            for g, h in zip(G, H):
                r = np.zeros(self.n)
                r[self.idx[k]:self.idx[k] + 3] = g
                s = np.linalg.norm(g)
                rows.append(r / s)
                b.append(h / s)
        self.A_fixed, self.b_fixed = np.array(rows), np.array(b)
        self.obs3 = obs3                       # (k, N (4,3), rho)

    def cam3_rows(self, R0, delta):
        rows, b = [], []
        for k, N, rho in self.obs3:
            ang = rho + delta
            if ang >= math.pi / 2 - 1e-9:
                return None                    # cell too large to test
            kap = math.sin(delta) / math.cos(ang)
            for n in N:
                g = R0.T @ (n - kap * np.array([0, 0, 1.0]))   # g^T (X - c) <= 0
                s = np.linalg.norm(g)
                r = np.zeros(self.n)
                r[self.idx[k]:self.idx[k] + 3] = g / s
                r[0:3] = -g / s
                rows.append(r)
                b.append(0.0)
            g = -R0[2]                                          # -q3 <= -zmin cos(ang)
            r = np.zeros(self.n)
            r[self.idx[k]:self.idx[k] + 3] = g
            r[0:3] = -g
            rows.append(r)
            b.append(-self.zmin * math.cos(ang))
        return np.array(rows), np.array(b)

    def system(self, R0, delta):
        c3 = self.cam3_rows(R0, delta)
        if c3 is None:
            return None
        return np.vstack([self.A_fixed, c3[0]]), np.concatenate([self.b_fixed, c3[1]])

    def feasible(self, R0, delta):
        S = self.system(R0, delta)
        if S is None:
            return True, None
        A, b = S
        A2 = np.hstack([A, -np.ones((len(A), 1))])
        c = np.zeros(self.n + 1)
        c[-1] = 1
        res = linprog(c, A_ub=A2, b_ub=b, bounds=[(None, None)] * self.n + [(0, None)],
                      method="highs")
        return bool(res.status == 0 and res.fun <= TOL), (res.fun if res.status == 0 else None)

    def extent(self, R0, delta, var):
        A, b = self.system(R0, delta)
        out = []
        for sg in (1.0, -1.0):
            cvec = np.zeros(self.n)
            cvec[var] = sg
            res = linprog(cvec, A_ub=A, b_ub=b, bounds=[(None, None)] * self.n, method="highs")
            out.append(sg * res.fun if res.status == 0 else (-sg * np.inf if res.status == 3 else np.nan))
        return out


def expR(w):
    return Rotation.from_rotvec(w).as_matrix()


_P = {}


def _init(prob, Rhat):
    _P["prob"], _P["Rhat"] = prob, Rhat


def _test(cell):
    w0, h = cell
    ok, t = _P["prob"].feasible(expR(w0) @ _P["Rhat"], math.sqrt(3) * h)
    return cell, ok


def _centre_box(cell):
    w0, h = cell
    R0, d = expR(w0) @ _P["Rhat"], math.sqrt(3) * h
    return [_P["prob"].extent(R0, d, j) for j in range(3)]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--pair", default="162,15")
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--h-min-deg", type=float, default=0.01)
    ap.add_argument("--delta0-deg", type=float, default=180.0,
                    help="search half-width; < 180 makes the result conditional on |w_true| <= delta0")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "pnp_set", "eps%g_img%d_d0%g_h%g" % (a.eps, a.third, a.delta0_deg, a.h_min_deg))
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    rep = {"eps": a.eps}

    cams, imgs, pts = read_model(MODEL)
    i1, i2 = [int(x) for x in a.pair.split(",")]
    i3 = a.third
    intr = {i: cams[imgs[i].camera_id].unpack() for i in (i1, i2, i3)}
    f = {i: (intr[i][0], intr[i][1]) for i in intr}

    # ---- gauge ----------------------------------------------------------------
    R1, c1 = imgs[i1].R, imgs[i1].center
    c2g = R1 @ (imgs[i2].center - c1)
    scale = c2g[0]
    pose = {i1: (np.eye(3), np.zeros(3)), i2: (imgs[i2].R @ R1.T, c2g / scale)}
    R3_colmap = imgs[i3].R @ R1.T
    c3_colmap = R1 @ (imgs[i3].center - c1) / scale
    Xg = {k: R1 @ (p.xyz - c1) / scale for k, p in pts.items()}
    rep["images"] = {"camera1": imgs[i1].name, "camera2": imgs[i2].name, "camera3": imgs[i3].name}

    def detections(i):
        im = imgs[i]
        out = {}
        for j in np.flatnonzero(im.p3d_ids >= 0):
            out.setdefault(int(im.p3d_ids[j]), []).append(undistort(*im.xys[j], intr[i]))
        return out

    det = {i: detections(i) for i in (i1, i2, i3)}
    common = sorted(set(det[i1]) & set(det[i2]))
    depths = [pose[i][0][2] @ (Xg[k] - pose[i][1]) for k in common for i in (i1, i2)]
    zmin = 1e-3 * float(np.median(depths))

    # ---- setup: F_k from cameras 1, 2 ------------------------------------------
    Fk, empty, cheb = {}, [], {}
    for k in common:
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i][k]:
                g, h = fixed_rows(*pose[i], xn, f[i], a.eps, zmin)
                G.append(g)
                H.append(h)
        G, H = np.vstack(G), np.concatenate(H)
        nrm = np.linalg.norm(G, axis=1)
        res = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[G, nrm], b_ub=H,
                      bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if res.status == 0 and res.x[3] > 0:
            Fk[k] = (G, H)
            cheb[k] = res.x[:3]
        elif res.status == 3:                         # unbounded set: centre undefined
            Fk[k] = (G, H)
        else:
            empty.append(k)
    L3 = sorted(k for k in Fk if k in det[i3])
    print("setup: %d landmarks common to %s and %s; %d have non-empty F_k at eps=%g (%d empty); "
          "%d of them observed by %s"
          % (len(common), imgs[i1].name, imgs[i2].name, len(Fk), a.eps, len(empty), len(L3), imgs[i3].name),
          flush=True)
    rep["setup"] = {"common_landmarks": len(common), "nonempty_sets": len(Fk), "empty_sets": len(empty),
                    "L3": len(L3), "z_min": zmin}

    obs3 = []
    for k in L3:
        for xn in det[i3][k]:
            N, rho = box_normals(xn, f[i3], a.eps)
            obs3.append((k, N, rho))
    prob = Problem(L3, Fk, obs3, zmin)

    # ---- Step 1: reference rotation (P3P + RANSAC + refinement) -----------------
    ks = [k for k in L3 if k in cheb]
    P3 = np.array([cheb[k] for k in ks], np.float64)
    p2 = np.array([det[i3][k][0] for k in ks], np.float64)
    okp, rvec, tvec, inl = cv2.solvePnPRansac(P3, p2, np.eye(3), None, flags=cv2.SOLVEPNP_P3P,
                                              reprojectionError=a.eps / f[i3][0], iterationsCount=2000)
    rvec, tvec = cv2.solvePnPRefineLM(P3[inl[:, 0]], p2[inl[:, 0]], np.eye(3), None, rvec, tvec)
    Rhat = cv2.Rodrigues(rvec)[0]
    c_pnp = -Rhat.T @ tvec.ravel()
    ang_pnp_colmap = float(np.degrees(Rotation.from_matrix(Rhat @ R3_colmap.T).magnitude()))
    print("Step 1: P3P+RANSAC on %d set centres: %d inliers; rotation differs from COLMAP's by %.4f deg"
          % (len(ks), len(inl), ang_pnp_colmap), flush=True)
    rep["step1"] = {"points": len(ks), "inliers": int(len(inl)), "deg_from_colmap": ang_pnp_colmap,
                    "centre_pnp": c_pnp.tolist(), "centre_colmap": c3_colmap.tolist()}

    # ---- Steps 2-4: branch and bound --------------------------------------------
    h_min = math.radians(a.h_min_deg)
    level = [(np.zeros(3), math.radians(a.delta0_deg))]
    rep["delta0_deg"] = a.delta0_deg
    survivors, tested, t = [], 0, time.time()
    with Pool(a.procs, initializer=_init, initargs=(prob, Rhat)) as pool:
        while level:
            level.sort(key=lambda cell: np.linalg.norm(cell[0]))     # closest to w = 0 first
            res = pool.map(_test, level, chunksize=max(1, len(level) // (4 * a.procs)))
            tested += len(level)
            keep = [cell for cell, ok in res if ok]
            h = level[0][1]
            print("  h = %8.4f deg: %6d cells tested, %6d feasible" % (math.degrees(h), len(level), len(keep)),
                  flush=True)
            if not keep:
                break
            if h / 2 < h_min:
                survivors = keep
                break
            level = [(w0 + (h / 2) * np.array(s), h / 2) for w0, _ in keep
                     for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]
        rep["step4"] = {"cells_tested": tested, "survivors": len(survivors),
                        "h_final_deg": math.degrees(survivors[0][1]) if survivors else None,
                        "seconds": round(time.time() - t)}
        if not survivors:
            print("Step 7: every cell discarded -> camera 3 has no feasible pose at eps=%g. "
                  "Exclude flagged tracks or allow m violated landmarks." % a.eps)
            json.dump(rep, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
            return

        # ---- Step 6: rotation bound and clusters --------------------------------
        hs = survivors[0][1]
        W = np.array([w for w, _ in survivors])
        delta3 = float(max(np.linalg.norm(W, axis=1)) + math.sqrt(3) * hs)
        # clusters: survivors are on a grid of spacing 2h; 26-neighbourhood
        lab = -np.ones(len(W), int)
        nlab = 0
        from scipy.spatial import cKDTree
        tree = cKDTree(W)
        for s0 in range(len(W)):
            if lab[s0] >= 0:
                continue
            stack = [s0]
            lab[s0] = nlab
            while stack:
                u = stack.pop()
                for v in tree.query_ball_point(W[u], 2 * hs * math.sqrt(3) + 1e-12):
                    if lab[v] < 0:
                        lab[v] = nlab
                        stack.append(v)
            nlab += 1
        print("Step 6: %d surviving cells (h = %.4f deg) in %d cluster(s); delta_3 = %.4f deg"
              % (len(W), math.degrees(hs), nlab, math.degrees(delta3)), flush=True)

        # ---- Step 5: centre box over surviving cells -----------------------------
        t = time.time()
        boxes = pool.map(_centre_box, survivors, chunksize=max(1, len(survivors) // (4 * a.procs)))
        B = np.array(boxes)                                         # (cells, 3, 2)
        c_lo, c_hi = B[:, :, 0].min(0), B[:, :, 1].max(0)
        print("Step 5: centre box lo %s hi %s size %s  (%ds)"
              % (np.round(c_lo, 4), np.round(c_hi, 4), np.round(c_hi - c_lo, 4), time.time() - t), flush=True)
    rot_extent = np.degrees(np.ptp(W, 0) + 2 * hs)
    rep["step6"] = {"delta3_deg": math.degrees(delta3), "clusters": nlab,
                    "rotation_cells_extent_deg_xyz": rot_extent.tolist(),
                    "surviving_cell_centres_deg_mean": np.degrees(W.mean(0)).tolist()}
    rep["step5"] = {"centre_lo": c_lo.tolist(), "centre_hi": c_hi.tolist(), "centre_size": (c_hi - c_lo).tolist(),
                    "centre_colmap": c3_colmap.tolist()}

    # ---- Step 8: joint update with camera 3 (rotation bound delta3 around R_hat3) ----
    new = sorted(k for k in set(det[i3]) & (set(det[i1]) | set(det[i2])) if k not in Fk and k not in empty)
    Fk8 = dict(Fk)
    for k in new:
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i].get(k, []):
                g, h = fixed_rows(*pose[i], xn, f[i], a.eps, zmin)
                G.append(g)
                H.append(h)
        Fk8[k] = (np.vstack(G), np.concatenate(H))
    L8 = sorted(set(L3) | set(new))
    obs8 = [(k, *box_normals(xn, f[i3], a.eps)) for k in L8 for xn in det[i3][k]]
    prob8 = Problem(L8, Fk8, obs8, zmin)
    ok8, _ = prob8.feasible(Rhat, delta3)
    box_before, box_after = {}, {}
    if ok8:
        for k in L8:
            v = prob8.idx[k]
            box_after[k] = np.array([prob8.extent(Rhat, delta3, v + d) for d in range(3)])
        for k in L3:
            G, H = Fk[k]
            bb = []
            for d in range(3):
                e = np.zeros(3)
                e[d] = 1
                r1 = linprog(e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
                r2 = linprog(-e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
                bb.append([r1.fun if r1.status == 0 else -np.inf, -r2.fun if r2.status == 0 else np.inf])
            box_before[k] = np.array(bb)
    print("Step 8: joint system with camera 3 feasible: %s; %d common + %d new landmarks"
          % (ok8, len(L3), len(new)), flush=True)

    # ---- evaluation against GT (same gauge rule) ------------------------------------
    gt = CC.load_gt(SCENE)
    g1 = gt[imgs[i1].name]
    Rg = lambda i: gt[imgs[i].name]["R"] @ g1["R"].T
    cgt = lambda i: g1["R"] @ (gt[imgs[i].name]["c"] - g1["c"])
    sc_gt = cgt(i2)[0]
    R3_gt, c3_gt = Rg(i3), cgt(i3) / sc_gt
    w_gt = Rotation.from_matrix(R3_gt @ Rhat.T).as_rotvec()
    in_cell = bool(np.any(np.all(np.abs(W - w_gt) <= hs + 1e-12, axis=1)))
    from cp_step0_align import umeyama
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]), np.array([imgs[i].center for i in ids]))
    lifted = {}
    for rw in CC.observations(cams, imgs, pts, gt, s_, Ra, ta):
        if rw["valid"]:
            lifted.setdefault(rw["pid"], []).append(rw["Xgt"])
    Xgt = {k: g1["R"] @ (np.median(v, 0) - g1["c"]) / sc_gt for k, v in lifted.items()}
    inside = lambda bx, x: bool(np.all((x >= bx[:, 0] - 1e-9) & (x <= bx[:, 1] + 1e-9)))
    ev = {"camera2_rotation_error_deg": float(np.degrees(Rotation.from_matrix(Rg(i2) @ pose[i2][0].T).magnitude())),
          "camera3_gt_rotation_from_Rhat_deg": float(np.degrees(np.linalg.norm(w_gt))),
          "camera3_gt_rotation_in_surviving_cells": in_cell,
          "camera3_gt_rotation_within_delta3": bool(np.linalg.norm(w_gt) <= delta3),
          "camera3_gt_centre_in_box": bool(np.all((c3_gt >= c_lo - 1e-9) & (c3_gt <= c_hi + 1e-9))),
          "camera3_gt_centre": c3_gt.tolist(),
          "colmap_rotation_within_delta3": bool(Rotation.from_matrix(R3_colmap @ Rhat.T).magnitude() <= delta3),
          "colmap_centre_in_box": bool(np.all((c3_colmap >= c_lo) & (c3_colmap <= c_hi)))}
    lm_rows = []
    for k in L8:
        r = {"point3D_id": k, "group": "common" if k in Fk else "new",
             "colmap": Xg[k].tolist(), "gt": Xgt[k].tolist() if k in Xgt else None}
        if k in box_before:
            r["box_pair"] = box_before[k].tolist()
            r["diag_pair"] = float(np.linalg.norm(box_before[k][:, 1] - box_before[k][:, 0]))
            r["gt_in_pair_box"] = inside(box_before[k], Xgt[k]) if k in Xgt else None
        if k in box_after:
            r["box_with_cam3"] = box_after[k].tolist()
            r["diag_with_cam3"] = float(np.linalg.norm(box_after[k][:, 1] - box_after[k][:, 0]))
            r["gt_in_box_with_cam3"] = inside(box_after[k], Xgt[k]) if k in Xgt else None
        lm_rows.append(r)
    rc = [r for r in lm_rows if r["group"] == "common" and "diag_with_cam3" in r]
    rn = [r for r in lm_rows if r["group"] == "new" and "diag_with_cam3" in r]
    fin = lambda v: np.array(v)[np.isfinite(v)]
    ev["landmarks"] = {
        "common_median_diag_pair": float(np.median(fin([r["diag_pair"] for r in rc]))) if rc else None,
        "common_median_diag_with_cam3": float(np.median(fin([r["diag_with_cam3"] for r in rc]))) if rc else None,
        "common_median_shrink_ratio": float(np.median(fin([r["diag_with_cam3"] / r["diag_pair"] for r in rc]))) if rc else None,
        "common_gt_in_pair_box": float(np.mean([r["gt_in_pair_box"] for r in rc if r["gt_in_pair_box"] is not None])) if rc else None,
        "common_gt_in_box_with_cam3": float(np.mean([r["gt_in_box_with_cam3"] for r in rc if r["gt_in_box_with_cam3"] is not None])) if rc else None,
        "new_count": len(rn),
        "new_median_diag": float(np.median(fin([r["diag_with_cam3"] for r in rn]))) if rn else None,
        "new_bounded_fraction": float(np.mean([np.isfinite(r["diag_with_cam3"]) for r in rn])) if rn else None,
        "new_gt_in_box": float(np.mean([r["gt_in_box_with_cam3"] for r in rn if r["gt_in_box_with_cam3"] is not None])) if rn else None}
    rep["evaluation_gt"] = ev
    rep["seconds"] = round(time.time() - T0)
    json.dump(rep, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    json.dump({"surviving_cells_w_rad": W.tolist(), "h_rad": hs, "R_hat3": Rhat.tolist(),
               "cell_centre_boxes": B.tolist(), "landmarks": lm_rows},
              open(os.path.join(OUT, "sets.json"), "w"))
    print(json.dumps(ev, indent=1))
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
