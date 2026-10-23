"""Joint alternation of camera pose sets and landmark sets (set-based SfM uncertainty).

    python joint_sets.py --scene /path/to/dataset --window 0 --n-cams 6

The dataset folder must contain the COLMAP text model in "COLMAP results/" (cameras.txt, images.txt,
points3D.txt) and, optionally, colmap_log.txt (registration order) and NeRF-style transforms*.json with
depth maps (for the ground-truth check). Cameras are taken in COLMAP's registration order; --window w uses
cameras [w * (n_cams - 2), w * (n_cams - 2) + n_cams) of that order, so consecutive windows overlap by 2.

Gauge: the first camera of the window is the origin (R = I, C = 0), the second camera's centre has x = 1;
the ground truth is expressed by the same rule. Each camera's pose set is a list of rotation cells (rotation
offset w from COLMAP's, half-width h) with a centre box; each landmark's set is a box.
 A. points from poses: each landmark is cut by every camera of its track (its pyramid through the +-eps
    window, with that camera's rotation and centre ranging over its pose set); the box is the intersection.
 B. poses from points: each rotation cell is tested with a linearised LP (rotation Jacobian plus a bound on
    the remainder) over all landmarks the camera sees; infeasible cells are discarded, the others split and
    their centre boxes contracted.
Repeated --iters times. Results are conditional on the initial boxes (--w0-deg, --c0, --x0) and on every
detection being within eps of the true projection.
"""
import os
import sys
import json
import math
import time
import argparse
import itertools
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from geom import Scene, box_normals, registration_order, run_name


def expR(w):
    return Rotation.from_rotvec(w).as_matrix()


E3 = np.array([0, 0, 1.0])
_G = {}


def _init(data):
    _G.update(data)


def g_rem(t):
    return t - math.sin(t) + 1 - math.cos(t)


def maxdist(lo, hi, p):
    return float(np.linalg.norm(np.maximum(np.abs(lo - p), np.abs(hi - p))))


# ---------------------------------------------------------------------------- step A
def _point_update(k):
    """new box of landmark k from every camera of its track (each: union over its rotation cells).
    Per cell the rotation offset w' (|w'| <= delta) is an LP unknown in the linearised faces
        n^T R0 (X - C) + (q0 x n)^T w' <= delta (Delta_X + Delta_C) + g(delta) Q,
    so only the remainder is slack (as in step B)."""
    D = _G
    xlo, xhi = D["Xbox"][k]
    for i in D["track"][k]:
        normals = D["normals"][(i, k)]
        if i == D["cam1"]:
            cells = [(np.zeros(3), 0.0, np.zeros(3), np.zeros(3))]
        else:
            cells = D["cells"][i]
        blo, bhi, any_ok = np.full(3, np.inf), np.full(3, -np.inf), False
        for w0, h, clo, chi in cells:
            R0 = expR(w0) @ D["Rc"][i] if i != D["cam1"] else np.eye(3)
            delta = math.sqrt(3) * h
            X0, C0 = 0.5 * (xlo + xhi), 0.5 * (clo + chi)
            q0 = R0 @ (X0 - C0)
            Q = float(np.linalg.norm(np.maximum(np.abs(xhi - clo), np.abs(xlo - chi))))
            s = delta * (maxdist(xlo, xhi, X0) + maxdist(clo, chi, C0)) + g_rem(delta) * Q
            rows, b = [], []
            for N in normals:
                for n in N:
                    g = n @ R0
                    rows.append(np.r_[g, -g, np.cross(q0, n)])
                    b.append(s)
                rows.append(np.r_[-R0[2], R0[2], -np.cross(q0, E3)])
                b.append(-D["zmin"] + s)
            A, bb = np.array(rows), np.array(b)
            bounds = list(zip(xlo, xhi)) + list(zip(clo, chi)) + [(-delta, delta)] * 3
            ok_cell, cl, ch = True, np.zeros(3), np.zeros(3)
            for d in range(3):
                c = np.zeros(9)
                c[d] = 1
                r1 = linprog(c, A_ub=A, b_ub=bb, bounds=bounds, method="highs")
                if r1.status != 0:
                    ok_cell = False
                    break
                r2 = linprog(-c, A_ub=A, b_ub=bb, bounds=bounds, method="highs")
                cl[d], ch[d] = r1.fun, -r2.fun
            if ok_cell:
                any_ok = True
                blo, bhi = np.minimum(blo, cl), np.maximum(bhi, ch)
        if not any_ok:
            return k, None
        xlo, xhi = np.maximum(xlo, blo), np.minimum(xhi, bhi)
        if np.any(xlo > xhi + 1e-12):
            return k, None
    return k, (xlo, xhi)


# ---------------------------------------------------------------------------- step B
def _cell_test(task):
    """linearised LP with remainder for one rotation cell of camera i; None or contracted centre box."""
    i, w0, h, clo, chi = task
    D = _G
    L = D["seen"][i]
    R0 = expR(w0) @ D["Rc"][i]
    delta = math.sqrt(3) * h
    n = 6 + 3 * len(L)
    c0 = 0.5 * (clo + chi)
    dC = maxdist(clo, chi, c0)
    gd = g_rem(delta)
    rows, b = [], []
    for m, k in enumerate(L):
        xlo, xhi = D["Xbox"][k]
        X0 = 0.5 * (xlo + xhi)
        q0 = R0 @ (X0 - c0)
        s = delta * (maxdist(xlo, xhi, X0) + dC) + gd * float(np.linalg.norm(np.maximum(np.abs(xhi - clo), np.abs(xlo - chi))))
        iv = 6 + 3 * m
        for N in D["normals"][(i, k)]:
            for nr in N:
                r = np.zeros(n)
                r[0:3] = np.cross(q0, nr)
                r[3:6] = -(nr @ R0)
                r[iv:iv + 3] = nr @ R0
                rows.append(r)
                b.append(s)
            r = np.zeros(n)
            r[0:3] = -np.cross(q0, E3)
            r[3:6] = R0[2]
            r[iv:iv + 3] = -R0[2]
            rows.append(r)
            b.append(-D["zmin"] + s)
    A, bb = np.array(rows), np.array(b)
    cb = list(zip(clo, chi))
    if i == D["cam2"]:
        cb[0] = (1.0, 1.0)                           # scale gauge: x of camera 2's centre = 1
    bounds = [(-delta, delta)] * 3 + cb + [tuple(x) for k in L for x in zip(*D["Xbox"][k])]
    A2 = np.hstack([A, -np.ones((len(A), 1))])
    c = np.zeros(n + 1)
    c[-1] = 1
    r = linprog(c, A_ub=A2, b_ub=bb, bounds=bounds + [(0, None)], method="highs")
    if r.status != 0 or r.fun > 1e-9:
        return None
    nlo, nhi = np.array([x[0] for x in cb]), np.array([x[1] for x in cb])
    for d in range(3):
        if cb[d][0] == cb[d][1]:
            continue
        e = np.zeros(n)
        e[3 + d] = 1
        r1 = linprog(e, A_ub=A, b_ub=bb, bounds=bounds, method="highs")
        r2 = linprog(-e, A_ub=A, b_ub=bb, bounds=bounds, method="highs")
        if r1.status != 0 or r2.status != 0:
            return None
        nlo[d], nhi[d] = max(nlo[d], r1.fun - 1e-9), min(nhi[d], -r2.fun + 1e-9)
    return nlo, nhi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="dataset folder (contains 'COLMAP results/')")
    ap.add_argument("--model", default=None, help="COLMAP text model folder (default: <scene>/COLMAP results)")
    ap.add_argument("--n-cams", type=int, default=6, help="cameras per window")
    ap.add_argument("--window", type=int, default=0, help="window index in COLMAP's registration order")
    ap.add_argument("--eps", type=float, default=3.0, help="pixel tolerance")
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"],
                    help="per-detection tolerance: eps, max(eps, 1.5 r), max(eps, r + 1); r = COLMAP residual")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--w0-deg", type=float, default=2.0, help="initial rotation half-width around COLMAP's")
    ap.add_argument("--c0", type=float, default=0.1, help="initial centre half-width (gauge units)")
    ap.add_argument("--x0", type=float, default=0.5, help="initial landmark half-width (gauge units)")
    ap.add_argument("--h-start-deg", type=float, default=0.5)
    ap.add_argument("--h-min-deg", type=float, default=0.125)
    ap.add_argument("--max-cells", type=int, default=400)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    T0 = time.time()
    model = a.model or os.path.join(a.scene, "COLMAP results")
    from colmap_io import read_model
    _, imgs_all, _ = read_model(model)
    start = a.window * (a.n_cams - 2)
    order = registration_order(a.scene, imgs_all, start + a.n_cams, model)
    cams = order[start:start + a.n_cams]
    if len(cams) < 3:
        print("window %d is past the end of the registration order" % a.window)
        return
    OUT = os.path.join(a.out, "%s_window%d" % (run_name(a.scene, a.model), a.window))
    os.makedirs(OUT, exist_ok=True)
    sc = Scene(a.scene, model, cams)
    cam1, cam2 = cams[0], cams[1]
    zmin = 1e-3 * 3.0
    track = {}
    for i in cams:
        for k in sc.det[i]:
            track.setdefault(k, []).append(i)
    track = {k: v for k, v in track.items() if len(v) >= 2}
    L = sorted(track)
    seen = {i: [k for k in L if i in track[k]] for i in cams}
    normals = {(i, k): [box_normals(xn, sc.f[i], e) for xn, e in zip(sc.det[i][k], sc.eps_list(i, k, a.eps, a.scheme))]
               for k in L for i in track[k]}
    Rc = {i: sc.pose(i)[0] for i in cams}
    Cc = {i: sc.pose(i)[1] for i in cams}
    Xc = {k: sc.point(k) for k in L}
    Xbox = {k: (Xc[k] - a.x0, Xc[k] + a.x0) for k in L}
    h0 = math.radians(a.h_start_deg)
    nper = max(1, int(round(a.w0_deg / a.h_start_deg)))
    grid = (np.arange(nper) * 2 + 1 - nper) * h0
    cells = {}
    for i in cams[1:]:
        clo, chi = Cc[i] - a.c0, Cc[i] + a.c0
        if i == cam2:
            clo[0] = chi[0] = 1.0
        cells[i] = [(np.array(w), h0, clo.copy(), chi.copy()) for w in itertools.product(grid, grid, grid)]
    gt = sc.load_gt()
    gt_pose, Xgt = {}, {}
    if gt is not None and all(sc.imgs[i].name in gt for i in cams):
        gt_pose, Xgt = sc.gt_in_gauge(gt, cams, L)
    print("window %d: cameras %s; %d landmarks; initial rotation +-%g deg (%d cells), centre +-%g, points +-%g; GT %s"
          % (a.window, [sc.imgs[i].name for i in cams], len(L), a.w0_deg, nper ** 3, a.c0, a.x0,
             "available" if gt_pose else "not available"), flush=True)

    def report(tag):
        rows = {}
        for i in cams[1:]:
            W = np.array([c[0] for c in cells[i]])
            h = np.array([c[1] for c in cells[i]])
            clo = np.min([c[2] for c in cells[i]], 0)
            chi = np.max([c[3] for c in cells[i]], 0)
            r = {"cells": len(cells[i]), "w_lo_deg": np.degrees((W - h[:, None]).min(0)).tolist(),
                 "w_hi_deg": np.degrees((W + h[:, None]).max(0)).tolist(), "C_lo": clo.tolist(), "C_hi": chi.tolist()}
            if i in gt_pose:
                Rg, Cg = gt_pose[i]
                wg = Rotation.from_matrix(Rg @ Rc[i].T).as_rotvec()
                r["gt_in_a_cell"] = bool(any(np.all(np.abs(wg - c[0]) <= c[1] + 1e-12) and
                                             np.all((Cg >= c[2] - 1e-9) & (Cg <= c[3] + 1e-9)) for c in cells[i]))
            rows[sc.imgs[i].name] = r
            print("  [%s] %-12s %4d cells; w %s .. %s deg; C size %s; GT pose in a cell: %s"
                  % (tag, sc.imgs[i].name, r["cells"], np.round(r["w_lo_deg"], 3), np.round(r["w_hi_deg"], 3),
                     np.round(chi - clo, 4), r.get("gt_in_a_cell", "n/a")), flush=True)
        diag = np.array([np.linalg.norm(Xbox[k][1] - Xbox[k][0]) for k in L])
        pts = {"landmarks": len(L), "box_diag_median": float(np.median(diag))}
        if Xgt:
            g = [bool(np.all((Xgt[k] >= Xbox[k][0] - 1e-9) & (Xgt[k] <= Xbox[k][1] + 1e-9))) for k in L if k in Xgt]
            pts.update(gt_inside=int(sum(g)), with_gt=len(g))
        print("  [%s] points: box diagonal median %.4f; GT inside %s"
              % (tag, pts["box_diag_median"], "%d / %d" % (pts["gt_inside"], pts["with_gt"]) if Xgt else "n/a"), flush=True)
        return {"points": pts, "cameras": rows}

    history = []
    hmin = math.radians(a.h_min_deg)
    for it in range(1, a.iters + 1):
        t = time.time()
        data = {"X": Xc, "Xbox": Xbox, "track": track, "normals": normals, "cells": cells, "cam1": cam1,
                "Rc": Rc, "zmin": zmin}
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            res = pool.map(_point_update, L, chunksize=8)
        bad = 0
        for k, bx in res:
            if bx is None:
                bad += 1
            else:
                Xbox[k] = bx
        print("iteration %d, A (points): %d inconsistent landmarks kept unchanged (%.0fs)" % (it, bad, time.time() - t), flush=True)
        t = time.time()
        data = {"Xbox": Xbox, "seen": seen, "normals": normals, "Rc": Rc, "zmin": zmin, "cam2": cam2}
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            for i in cams[1:]:
                out = pool.map(_cell_test, [(i, w0, h, clo, chi) for w0, h, clo, chi in cells[i]], chunksize=1)
                keep = [(w0, h, r[0], r[1]) for (w0, h, _, _), r in zip(cells[i], out) if r is not None]
                if not keep:
                    print("   %s: every cell discarded (inconsistent at eps); previous cells kept" % sc.imgs[i].name)
                    continue
                if keep[0][1] / 2 >= hmin - 1e-12 and 8 * len(keep) <= a.max_cells:
                    keep = [(w0 + 0.5 * h * np.array(s), 0.5 * h, clo.copy(), chi.copy())
                            for w0, h, clo, chi in keep for s in itertools.product((-1, 1), repeat=3)]
                cells[i] = keep
        print("iteration %d, B (poses) (%.0fs)" % (it, time.time() - t), flush=True)
        history.append(report("iteration %d" % it))
    json.dump({"scene": a.scene, "window": a.window, "cameras": [sc.imgs[i].name for i in cams], "eps_px": a.eps,
               "scheme": a.scheme, "initial": {"w_deg": a.w0_deg, "c": a.c0, "x": a.x0}, "iterations": history,
               "seconds": round(time.time() - T0)}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, "landmark_boxes.npz"), landmarks=np.array(L),
                        lo=np.array([Xbox[k][0] for k in L]), hi=np.array([Xbox[k][1] for k in L]))
    for i in cams[1:]:
        np.savez_compressed(os.path.join(OUT, "pose_cells_%s.npz" % os.path.splitext(sc.imgs[i].name)[0]),
                            w_rad=np.array([c[0] for c in cells[i]]), h_rad=np.array([c[1] for c in cells[i]]),
                            centre_lo=np.array([c[2] for c in cells[i]]), centre_hi=np.array([c[3] for c in cells[i]]))
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
