"""Joint alternation on a subset of cameras: pose sets from point sets, point sets from pose sets.

    python joint_alternation.py [--eps 3] [--cams 162,15,17,73,5,152] [--iters 3]

Gauge: camera 1 is exactly the origin (R = I, C = 0) and camera 2's centre has x = 1; GT is expressed
by the same rule, so both conditions hold for the truth. Every other pose parameter is free.

Pose set of camera i: a list of rotation cells (centre w0, half-width h, rotation R = exp([w]x) R_colmap)
each with a centre box. Point set of landmark k: a box. Initial (generous) sets:
    rotation within +-w0_deg of COLMAP's, centre within +-c0, point within +-x0 of COLMAP's.
Alternation (each step removes only what is impossible, so the truth survives if the premise holds and it
is inside the initial sets):
 A. points from poses. For landmark k and a camera i of its track, with a cell (R0, delta, C box):
        n^T R0 (X - C) <= delta * Q   (4 faces),   -r3_0^T (X - C) <= -z_min + delta * Q,   C in C box,
    where ||(R - R0)(X - C)|| <= delta ||X - C|| <= delta * Q bounds the rotation error. B_ik = box of the
    union over the camera's cells; X box <- X box intersected with every B_ik of its track.
 B. poses from points. For camera i, each rotation cell is tested with the linearised LP with remainder
    (as interval_linear_pnp.py) over all landmarks it sees, with X_k in its box: infeasible cells are
    discarded, surviving cells are split (down to --h-min-deg) and their centre boxes are contracted.
Writes results_joint/: summary.json, cells_<camera>.npz, landmark_boxes.npz.
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

from setlib import Scene, box_normals, expR, HERE, SCENE
from union_incremental import gt_points

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
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"])
    ap.add_argument("--cams", default="162,15,17,73,5,152")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--w0-deg", type=float, default=2.0)
    ap.add_argument("--c0", type=float, default=0.1)
    ap.add_argument("--x0", type=float, default=0.5)
    ap.add_argument("--h-start-deg", type=float, default=0.5, help="initial cell half-width")
    ap.add_argument("--h-min-deg", type=float, default=0.125)
    ap.add_argument("--max-cells", type=int, default=400)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    T0 = time.time()
    OUT = os.path.join(HERE, "results_joint" + ("" if a.scheme == "fixed" else "_" + a.scheme))
    os.makedirs(OUT, exist_ok=True)
    cams = [int(x) for x in a.cams.split(",")]
    sc = Scene(pair=(cams[0], cams[1]), extra=tuple(cams[2:]))
    cam1, cam2 = cams[0], cams[1]
    zmin = 1e-3 * 3.0
    # landmarks with >= 2 cameras of the subset
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
    # initial cells: cube of half-width w0 around COLMAP's rotation split into cells of half-width h_start
    h0 = math.radians(a.h_start_deg)
    nper = int(round(a.w0_deg / a.h_start_deg))
    grid = (np.arange(nper) * 2 + 1 - nper) * h0
    cells = {}
    for i in cams[1:]:
        clo, chi = Cc[i] - a.c0, Cc[i] + a.c0
        if i == cam2:
            clo[0] = chi[0] = 1.0
        cells[i] = [(np.array(w), h0, clo.copy(), chi.copy()) for w in itertools.product(grid, grid, grid)]
    Xgt = gt_points(sc)
    gt_pose = {i: sc.gt_pose(i) for i in cams}
    print("cameras %s; %d landmarks (>= 2 views in the subset); initial: rotation +-%g deg (%d cells of half-width %g "
          "deg), centre +-%g, points +-%g" % ([sc.imgs[i].name for i in cams], len(L), a.w0_deg, nper ** 3,
                                               a.h_start_deg, a.c0, a.x0), flush=True)

    def report(tag):
        rows = {}
        for i in cams[1:]:
            W = np.array([c[0] for c in cells[i]])
            h = np.array([c[1] for c in cells[i]])
            lo = np.degrees((W - h[:, None]).min(0))
            hi = np.degrees((W + h[:, None]).max(0))
            clo = np.min([c[2] for c in cells[i]], 0)
            chi = np.max([c[3] for c in cells[i]], 0)
            Rg, Cg = gt_pose[i]
            wg = Rotation.from_matrix(Rg @ Rc[i].T).as_rotvec()
            in_cell = any(np.all(np.abs(wg - c[0]) <= c[1] + 1e-12) and np.all((Cg >= c[2] - 1e-9) & (Cg <= c[3] + 1e-9))
                          for c in cells[i])
            rows[sc.imgs[i].name] = {"cells": len(cells[i]), "w_lo_deg": lo.tolist(), "w_hi_deg": hi.tolist(),
                                     "C_lo": clo.tolist(), "C_hi": chi.tolist(), "C_size": (chi - clo).tolist(),
                                     "gt_in_a_cell": bool(in_cell), "gt_w_deg": np.degrees(wg).tolist()}
        diag = np.array([np.linalg.norm(Xbox[k][1] - Xbox[k][0]) for k in L])
        gt_in = [bool(np.all((Xgt[k] >= Xbox[k][0] - 1e-9) & (Xgt[k] <= Xbox[k][1] + 1e-9))) for k in L if k in Xgt]
        pts = {"landmarks": len(L), "box_diag_median": float(np.median(diag)), "box_diag_p90": float(np.quantile(diag, 0.9)),
               "gt_inside": int(sum(gt_in)), "with_gt": len(gt_in)}
        print("  [%s] points: box diagonal median %.4f (p90 %.4f); GT inside %d / %d" %
              (tag, pts["box_diag_median"], pts["box_diag_p90"], pts["gt_inside"], pts["with_gt"]), flush=True)
        for nm, r in rows.items():
            print("  [%s] %-12s %4d cells; w %s .. %s deg; C size %s; GT pose in a cell: %s"
                  % (tag, nm, r["cells"], np.round(r["w_lo_deg"], 3), np.round(r["w_hi_deg"], 3),
                     np.round(r["C_size"], 4), r["gt_in_a_cell"]), flush=True)
        return {"points": pts, "cameras": rows}

    history = []
    hmin = math.radians(a.h_min_deg)
    for it in range(1, a.iters + 1):
        t = time.time()
        # ---- A: points from poses -----------------------------------------------------------------
        data = {"X": Xc, "Xbox": Xbox, "track": track, "normals": normals, "cells": cells, "cam1": cam1,
                "Rc": Rc, "zmin": zmin}
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            res = pool.map(_point_update, L, chunksize=8)
        dropped = 0
        for k, bx in res:
            if bx is None:
                dropped += 1                          # inconsistent at eps: keep the previous box
            else:
                Xbox[k] = bx
        print("iteration %d, A (points): %d landmarks updated, %d inconsistent (kept)  (%.0fs)"
              % (it, len(L) - dropped, dropped, time.time() - t), flush=True)
        # ---- B: poses from points ------------------------------------------------------------------
        t = time.time()
        data = {"Xbox": Xbox, "seen": seen, "normals": normals, "Rc": Rc, "zmin": zmin, "cam2": cam2}
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            for i in cams[1:]:
                tasks = [(i, w0, h, clo, chi) for w0, h, clo, chi in cells[i]]
                out = pool.map(_cell_test, tasks, chunksize=1)
                keep = [(w0, h, r[0], r[1]) for (w0, h, _, _), r in zip(cells[i], out) if r is not None]
                if not keep:
                    print("   %s: every cell discarded (inconsistent at eps); keeping the previous cells" % sc.imgs[i].name)
                    continue
                # split surviving cells while they are large and the budget allows
                if keep[0][1] / 2 >= hmin - 1e-12 and 8 * len(keep) <= a.max_cells:
                    split = []
                    for w0, h, clo, chi in keep:
                        for s in itertools.product((-1, 1), repeat=3):
                            split.append((w0 + 0.5 * h * np.array(s), 0.5 * h, clo.copy(), chi.copy()))
                    keep = split
                cells[i] = keep
        print("iteration %d, B (poses) done (%.0fs)" % (it, time.time() - t), flush=True)
        history.append(report("iteration %d" % it))
    json.dump({"eps_px": a.eps, "scheme": a.scheme, "cameras": [sc.imgs[i].name for i in cams],
               "initial": {"w_deg": a.w0_deg, "c": a.c0, "x": a.x0}, "iterations": history,
               "seconds": round(time.time() - T0),
               "note": "sets are conditional on the initial boxes; cells are tested after splitting in the next iteration"},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, "landmark_boxes.npz"), landmarks=np.array(L),
                        lo=np.array([Xbox[k][0] for k in L]), hi=np.array([Xbox[k][1] for k in L]))
    for i in cams[1:]:
        np.savez_compressed(os.path.join(OUT, "cells_%d.npz" % i), w=np.array([c[0] for c in cells[i]]),
                            h=np.array([c[1] for c in cells[i]]), c_lo=np.array([c[2] for c in cells[i]]),
                            c_hi=np.array([c[3] for c in cells[i]]))
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
