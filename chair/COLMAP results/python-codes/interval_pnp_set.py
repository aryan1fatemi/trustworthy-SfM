"""Camera-3 pose set by interval analysis: SIVIA with two contractions.

    python interval_pnp_set.py --scene airplane --eps 3

Setup as before: cameras 1, 2 (#162, #15) held at COLMAP's poses; gauge R_1 = I, c_1 = 0,
x of c_2 = 1; detections undistorted; eps in the undistorted image. Camera 3 = #17.

Pose box, centred on COLMAP's estimate:
    R = exp([w]x) R_colmap,  w in [w_lo, w_hi] (3 intervals),   centre C in [C_lo, C_hi] (3).
Landmark boxes [X_k] start as the boxes of F_k (from cameras 1 and 2).

Interval enclosure of the rotation over a w-box (entrywise, sound):
    exp(W) = I + W + E,  ||E|| <= g(theta) = theta - sin(theta) + 1 - cos(theta),
    [R] = R_colmap + [W] R_colmap +- g(theta_max)
Camera-3 constraints for observation (k, window [wa, wb] x [va, vb] in normalised coordinates),
with D = X_k - C and q = R D (all intervals):
    q1 - wb q3 <= 0,  wa q3 - q1 <= 0,  q2 - vb q3 <= 0,  va q3 - q2 <= 0,  -q3 <= -z_min
each written as  sum_j [a_j] D_j <= b  with interval coefficients [a_j] (rows of [R] combined
first, which reduces the dependency problem). Cameras 1, 2 add the rows of F_k (constant
coefficients, D = X_k).

Contraction (HC4-style, for  sum_j [a_j] D_j <= b):
    infeasible if  sum_j min([a_j][D_j]) > b;
    otherwise for each j:  [a_j] D_j <= s_j = b - sum_{i != j} min([a_i][D_i]),
    which bounds D_j above (a_j > 0) or below (a_j < 0); then D = X - C narrows X_k (contraction 1)
    and C (contraction 2). Iterated to a fixed point. Only values that cannot belong to any
    consistent (pose, points) are removed, so the truth is never lost (premise + truth in the
    initial box). Floating-point rounding is not directed (tolerance 1e-12).
SIVIA: contract the box; discard if empty; otherwise split the widest rotation interval in two
and repeat on each half (children inherit the contracted C and X boxes); keep boxes whose
rotation intervals are all <= the leaf width.
Output: hull of surviving rotation and centre boxes, and the hull of each landmark's contracted
boxes (a subset of its original box). GT (same gauge rule) only in the evaluation.
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
import cp_conformal as CC
from stage_a import undistort
from pnp_set import fixed_rows

TOL = 1e-12


def g_rem(t):
    return t - math.sin(t) + 1 - math.cos(t)


def imul(alo, ahi, blo, bhi):
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return p.min(0), p.max(0)


def rot_box(wlo, whi, Rc):
    """entrywise interval enclosure of exp([w]x) Rc over the box [wlo, whi]."""
    W = [[(0.0, 0.0), (-whi[2], -wlo[2]), (wlo[1], whi[1])],
         [(wlo[2], whi[2]), (0.0, 0.0), (-whi[0], -wlo[0])],
         [(-whi[1], -wlo[1]), (wlo[0], whi[0]), (0.0, 0.0)]]
    lo, hi = Rc.copy(), Rc.copy()
    for i in range(3):
        for j in range(3):
            for m in range(3):
                a, b = W[i][m]
                v = np.array([a * Rc[m, j], b * Rc[m, j]])
                lo[i, j] += v.min()
                hi[i, j] += v.max()
    th = float(np.linalg.norm(np.maximum(np.abs(wlo), np.abs(whi))))
    g = g_rem(th)
    return lo - g, hi + g


def bound_D(alo, ahi, s, Dlo, Dhi):
    """narrow D given exists a in [alo, ahi] with a * D <= s (vectorised)."""
    Dlo, Dhi = Dlo.copy(), Dhi.copy()
    pos, neg = alo > 0, ahi < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(s >= 0, s / alo, s / ahi)          # a > 0: D <= up
        dn = np.where(s >= 0, s / ahi, s / alo)          # a < 0: D >= dn
    Dhi = np.where(pos, np.minimum(Dhi, up), Dhi)
    Dlo = np.where(neg, np.maximum(Dlo, dn), Dlo)
    return Dlo, Dhi


class Problem:
    def __init__(self, Rc, L, Fk, Fbox, obs, zmin):
        self.Rc, self.L, self.zmin = Rc, L, zmin
        self.K = len(L)
        kid = {k: n for n, k in enumerate(L)}
        # F_k rows (constant coefficients): G x <= h, landmark index
        G, H, idx = [], [], []
        for k in L:
            g, h = Fk[k]
            G.append(g)
            H.append(h)
            idx += [kid[k]] * len(h)
        self.FG, self.FH, self.Fi = np.vstack(G), np.concatenate(H), np.array(idx)
        self.X0lo = np.array([Fbox[k][0] for k in L])
        self.X0hi = np.array([Fbox[k][1] for k in L])
        # camera-3 observations: landmark index and window in normalised coordinates
        self.ok = np.array([kid[k] for k, _ in obs])
        self.win = np.array([w for _, w in obs])         # (N, 4): wa, wb, va, vb

    def cam3_coeffs(self, Rlo, Rhi):
        """interval coefficients (N*5, 3) and rhs for the five camera-3 constraints per obs."""
        wa, wb, va, vb = self.win.T
        rows_lo, rows_hi, b = [], [], []
        for r_main, w, sgn in ((0, wb, 1), (0, wa, -1), (1, vb, 1), (1, va, -1)):
            # sgn=+1: q_r - w q3 <= 0 ; sgn=-1: w q3 - q_r <= 0
            m_lo, m_hi = (Rlo[r_main], Rhi[r_main]) if sgn > 0 else (-Rhi[r_main], -Rlo[r_main])
            c = -sgn * w[:, None]                          # coefficient on q3
            t1, t2 = c * Rlo[2][None], c * Rhi[2][None]
            rows_lo.append(m_lo[None] + np.minimum(t1, t2))
            rows_hi.append(m_hi[None] + np.maximum(t1, t2))
            b.append(np.zeros(len(w)))
        rows_lo.append(np.repeat(-Rhi[2][None], len(wa), 0))
        rows_hi.append(np.repeat(-Rlo[2][None], len(wa), 0))
        b.append(np.full(len(wa), -self.zmin))
        return (np.concatenate(rows_lo), np.concatenate(rows_hi), np.concatenate(b),
                np.tile(self.ok, 5))

    def contract(self, wlo, whi, Clo, Chi, Xlo, Xhi, max_iter=30):
        """returns None if the box is empty, else contracted (Clo, Chi, Xlo, Xhi)."""
        Rlo, Rhi = rot_box(wlo, whi, self.Rc)
        Alo, Ahi, b, ki = self.cam3_coeffs(Rlo, Rhi)
        Clo, Chi, Xlo, Xhi = Clo.copy(), Chi.copy(), Xlo.copy(), Xhi.copy()
        for it in range(max_iter):
            old = np.r_[Clo, Chi, Xlo.ravel(), Xhi.ravel()]
            # --- F_k rows (cameras 1, 2): constant coefficients on X ---------------
            g = self.FG
            Pl, Ph = imul(g, g, Xlo[self.Fi], Xhi[self.Fi])
            sl = Pl.sum(1)
            if np.any(sl > self.FH + 1e-9):
                return None
            for j in range(3):
                s = self.FH - (sl - Pl[:, j])
                dlo, dhi = bound_D(g[:, j], g[:, j], s, Xlo[self.Fi, j], Xhi[self.Fi, j])
                np.maximum.at(Xlo[:, j], self.Fi, dlo)
                np.minimum.at(Xhi[:, j], self.Fi, dhi)
            if np.any(Xlo > Xhi + TOL):
                return None
            # --- camera-3 rows: D = X_k - C -------------------------------------------
            Dlo, Dhi = Xlo[ki] - Chi[None], Xhi[ki] - Clo[None]
            Pl, Ph = imul(Alo, Ahi, Dlo, Dhi)
            sl = Pl.sum(1)
            if np.any(sl > b + 1e-9):
                return None
            for j in range(3):
                s = b - (sl - Pl[:, j])
                nlo, nhi = bound_D(Alo[:, j], Ahi[:, j], s, Dlo[:, j], Dhi[:, j])
                # contraction 1: landmark boxes;  contraction 2: camera centre
                np.maximum.at(Xlo[:, j], ki, nlo + Clo[j])
                np.minimum.at(Xhi[:, j], ki, nhi + Chi[j])
                Clo[j] = max(Clo[j], float(np.max(Xlo[ki, j] - nhi)))
                Chi[j] = min(Chi[j], float(np.min(Xhi[ki, j] - nlo)))
            if np.any(Xlo > Xhi + TOL) or np.any(Clo > Chi + TOL):
                return None
            new = np.r_[Clo, Chi, Xlo.ravel(), Xhi.ravel()]
            if np.max(np.abs(new - old)) < 1e-7:
                break
        return Clo, Chi, Xlo, Xhi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--pair", default="162,15")
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--w0-deg", type=float, default=8.0, help="initial rotation half-width around COLMAP")
    ap.add_argument("--c0", type=float, default=1.0, help="initial centre half-width around COLMAP")
    ap.add_argument("--leaf-deg", type=float, default=0.5)
    ap.add_argument("--leaf-c", type=float, default=0.05, help="leaf width of the centre box")
    ap.add_argument("--rot-only", action="store_true", help="split only the rotation (no centre splitting)")
    ap.add_argument("--c-box", default="sound", choices=["sound", "colmap"],
                    help="initial centre box: hull of linpnp_set.py's sound bound (+5%%), or COLMAP +- c0")
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "pnp_set", "interval_eps%g_img%d%s" % (a.eps, a.third, "_rotonly" if a.rot_only else ""))
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()

    # ---- setup ----------------------------------------------------------------------
    cams, imgs, pts = read_model(MODEL)
    i1, i2 = [int(x) for x in a.pair.split(",")]
    i3 = a.third
    intr = {i: cams[imgs[i].camera_id].unpack() for i in (i1, i2, i3)}
    f = {i: (intr[i][0], intr[i][1]) for i in intr}
    R1, c1 = imgs[i1].R, imgs[i1].center
    scale = (R1 @ (imgs[i2].center - c1))[0]
    pose = {i1: (np.eye(3), np.zeros(3)), i2: (imgs[i2].R @ R1.T, R1 @ (imgs[i2].center - c1) / scale)}
    Rc = imgs[i3].R @ R1.T
    cc = R1 @ (imgs[i3].center - c1) / scale
    Xg = {k: R1 @ (p.xyz - c1) / scale for k, p in pts.items()}
    det = {}
    for i in (i1, i2, i3):
        d = {}
        for j in np.flatnonzero(imgs[i].p3d_ids >= 0):
            d.setdefault(int(imgs[i].p3d_ids[j]), []).append(undistort(*imgs[i].xys[j], intr[i]))
        det[i] = d
    common = sorted(set(det[i1]) & set(det[i2]))
    zmin = 1e-3 * float(np.median([pose[i][0][2] @ (Xg[k] - pose[i][1]) for k in common for i in (i1, i2)]))
    Fk, Fbox = {}, {}
    for k in common:
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i][k]:
                g_, h_ = fixed_rows(*pose[i], xn, f[i], a.eps, zmin)
                G.append(g_)
                H.append(h_)
        G, H = np.vstack(G), np.concatenate(H)
        bb = []
        for dd in range(3):
            e = np.zeros(3)
            e[dd] = 1
            r1 = linprog(e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
            r2 = linprog(-e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
            bb.append([r1.fun, -r2.fun] if r1.status == 0 and r2.status == 0 else [np.nan, np.nan])
        bb = np.array(bb)
        if np.all(np.isfinite(bb)):
            Fk[k], Fbox[k] = (G, H), (bb[:, 0], bb[:, 1])
    L3 = sorted(k for k in Fk if k in det[i3])
    ex, ey = a.eps / f[i3][0], a.eps / f[i3][1]
    obs = [(k, (xn[0] - ex, xn[0] + ex, xn[1] - ey, xn[1] + ey)) for k in L3 for xn in det[i3][k]]
    P = Problem(Rc, L3, Fk, Fbox, obs, zmin)
    print("setup: %d landmark boxes, %d camera-3 observations; initial box: w +-%g deg around COLMAP, "
          "centre COLMAP +-%g" % (len(L3), len(obs), a.w0_deg, a.c0), flush=True)

    # ---- root box: contraction alone ------------------------------------------------------
    w0 = math.radians(a.w0_deg)
    if a.c_box == "sound":
        sd = json.load(open(os.path.join(MODEL, "pnp_set", "linearised_eps%g_img%d" % (a.eps, i3), "summary.json")))["sound"]
        slo, shi = np.array(sd["centre_lo"]), np.array(sd["centre_hi"])
        pad = 0.05 * (shi - slo)
        c_lo0, c_hi0 = slo - pad, shi + pad
    else:
        c_lo0, c_hi0 = cc - a.c0, cc + a.c0
    root = (-w0 * np.ones(3), w0 * np.ones(3), c_lo0, c_hi0, P.X0lo, P.X0hi)
    depth = float(np.median([np.linalg.norm(Xg[k] - cc) for k in L3]))
    t = time.time()
    r = P.contract(*root)
    if r is None:
        print("root box empty after contraction")
        return
    vol = lambda lo, hi: np.prod(hi - lo, axis=-1)
    print("root contraction (%.2fs): centre box size %s -> %s; landmark box volume median ratio %.3f"
          % (time.time() - t, np.round(c_hi0 - c_lo0, 3), np.round(r[1] - r[0], 4),
             np.median(vol(r[2], r[3]) / vol(P.X0lo, P.X0hi))), flush=True)

    # ---- SIVIA --------------------------------------------------------------------------------
    leaf = math.radians(a.leaf_deg)
    stack = [(root[0], root[1]) + tuple(r)]
    kept, tested, levels = [], 0, {}
    t = time.time()
    while stack:
        wlo, whi, Clo, Chi, Xlo, Xhi = stack.pop()
        res = P.contract(wlo, whi, Clo, Chi, Xlo, Xhi)
        tested += 1
        width = whi - wlo
        lv = int(round(math.log2((2 * w0) / width.max())))
        cw = Chi - Clo
        levels.setdefault(lv, [0, 0])
        levels[lv][0] += 1
        if res is None:
            continue
        levels[lv][1] += 1
        Clo, Chi, Xlo, Xhi = res
        cw = Chi - Clo
        rot_done = width.max() <= leaf + 1e-12
        c_done = a.rot_only or cw.max() <= a.leaf_c + 1e-12
        if rot_done and c_done:
            kept.append((wlo, whi, Clo, Chi, Xlo, Xhi))
            continue
        # widest dimension, rotation measured as image-plane displacement (width * depth)
        sw = np.r_[np.where(rot_done, 0, width * depth), np.where(c_done, 0, cw)]
        j = int(np.argmax(sw))
        if j < 3:
            mid = 0.5 * (wlo[j] + whi[j])
            for lo_j, hi_j in ((wlo[j], mid), (mid, whi[j])):
                nlo, nhi = wlo.copy(), whi.copy()
                nlo[j], nhi[j] = lo_j, hi_j
                stack.append((nlo, nhi, Clo, Chi, Xlo, Xhi))
        else:
            j -= 3
            mid = 0.5 * (Clo[j] + Chi[j])
            for lo_j, hi_j in ((Clo[j], mid), (mid, Chi[j])):
                nlo, nhi = Clo.copy(), Chi.copy()
                nlo[j], nhi[j] = lo_j, hi_j
                stack.append((wlo, whi, nlo, nhi, Xlo, Xhi))
        if tested % 2000 == 0:
            print("  %d boxes tested, %d leaves kept, %d pending (%.0fs)"
                  % (tested, len(kept), len(stack), time.time() - t), flush=True)
    print("SIVIA: %d boxes tested, %d leaves kept (%.0fs)" % (tested, len(kept), time.time() - t), flush=True)
    rep = {"eps": a.eps, "w0_deg": a.w0_deg, "c_box": a.c_box, "c_box_initial": [c_lo0.tolist(), c_hi0.tolist()],
           "leaf_deg": a.leaf_deg, "leaf_c": a.leaf_c, "rot_only": a.rot_only,
           "boxes_tested": tested, "leaves": len(kept),
           "levels": {str(k): {"tested": v[0], "kept": v[1]} for k, v in sorted(levels.items())}}
    if not kept:
        print("empty: no pose consistent with all observations at eps = %g" % a.eps)
        json.dump(rep, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
        return
    Wlo = np.min([k[0] for k in kept], 0)
    Whi = np.max([k[1] for k in kept], 0)
    Clo = np.min([k[2] for k in kept], 0)
    Chi = np.max([k[3] for k in kept], 0)
    Xlo = np.min([k[4] for k in kept], 0)
    Xhi = np.max([k[5] for k in kept], 0)
    delta = max(float(np.linalg.norm(np.maximum(np.abs(k[0]), np.abs(k[1])))) for k in kept)
    shrink = vol(Xlo, Xhi) / vol(P.X0lo, P.X0hi)
    diag0 = np.linalg.norm(P.X0hi - P.X0lo, axis=1)
    diag1 = np.linalg.norm(Xhi - Xlo, axis=1)

    # ---- GT -------------------------------------------------------------------------------------
    gt = CC.load_gt(SCENE)
    g1 = gt[imgs[i1].name]
    sc = (g1["R"] @ (gt[imgs[i2].name]["c"] - g1["c"]))[0]
    R_gt = gt[imgs[i3].name]["R"] @ g1["R"].T
    c_gt = g1["R"] @ (gt[imgs[i3].name]["c"] - g1["c"]) / sc
    w_gt = Rotation.from_matrix(R_gt @ Rc.T).as_rotvec()
    inb = lambda x, lo, hi: bool(np.all((x >= lo - 1e-9) & (x <= hi + 1e-9)))
    gt_leaf = any(inb(w_gt, k[0], k[1]) and inb(c_gt, k[2], k[3]) for k in kept)
    from cp_step0_align import umeyama
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]), np.array([imgs[i].center for i in ids]))
    lifted = {}
    for rw in CC.observations(cams, imgs, pts, gt, s_, Ra, ta):
        if rw["valid"]:
            lifted.setdefault(rw["pid"], []).append(rw["Xgt"])
    Xgt = {k: g1["R"] @ (np.median(v, 0) - g1["c"]) / sc for k, v in lifted.items()}
    in_old = [inb(Xgt[k], P.X0lo[n], P.X0hi[n]) for n, k in enumerate(L3) if k in Xgt]
    in_new = [inb(Xgt[k], Xlo[n], Xhi[n]) for n, k in enumerate(L3) if k in Xgt]

    rep.update({
        "rotation_hull_deg_rel_colmap": {"lo": np.degrees(Wlo).tolist(), "hi": np.degrees(Whi).tolist()},
        "delta_from_colmap_deg_max": math.degrees(delta),
        "centre_hull": {"lo": Clo.tolist(), "hi": Chi.tolist(), "size": (Chi - Clo).tolist()},
        "landmarks": {"count": len(L3), "box_volume_ratio_new_over_old_median": float(np.median(shrink)),
                      "box_diag_old_median": float(np.median(diag0)), "box_diag_new_median": float(np.median(diag1)),
                      "gt_in_old_box": float(np.mean(in_old)), "gt_in_new_box": float(np.mean(in_new))},
        "gt": {"w_rel_colmap_deg": np.degrees(w_gt).tolist(), "centre": c_gt.tolist(),
               "colmap_centre": cc.tolist(), "gt_pose_in_a_kept_leaf": gt_leaf,
               "gt_rotation_in_hull": inb(np.degrees(w_gt), np.degrees(Wlo), np.degrees(Whi)),
               "gt_centre_in_hull": inb(c_gt, Clo, Chi)},
        "seconds": round(time.time() - T0)})
    json.dump(rep, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, "leaves.npz"), w_lo=np.array([k[0] for k in kept]),
                        w_hi=np.array([k[1] for k in kept]), c_lo=np.array([k[2] for k in kept]),
                        c_hi=np.array([k[3] for k in kept]), X_lo=Xlo, X_hi=Xhi,
                        X0_lo=P.X0lo, X0_hi=P.X0hi, landmarks=np.array(L3))
    print(json.dumps({k: rep[k] for k in ("rotation_hull_deg_rel_colmap", "delta_from_colmap_deg_max",
                                          "centre_hull", "landmarks", "gt")}, indent=1))
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
