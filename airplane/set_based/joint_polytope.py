"""Joint alternation, polytope version: landmark sets are polytopes, each camera is solved jointly with them.

    python joint_polytope.py [--eps 3] [--cams 162,15,17,73,5,152] [--iters 3]

Same gauge, initial sets and pose-set representation as joint_alternation.py (rotation cells around COLMAP's
rotation, each with a centre box). What changes is the landmark set: P_k = {X : A_k X <= b_k} instead of a box.
    init. P_k = (COLMAP point +- x0 box) cut by the exact pyramid of camera 1 (fixed pose: 5 rows per detection).
 A. points from poses. For each other camera i of the track and each direction a (the 4 window faces and the
    depth axis of camera i at COLMAP's rotation, a = R_colmap^T n), the support
        b = max over cells of  max a^T X  s.t.  X in P_k,  linearised window rows of the cell (X, C, w' unknown)
    is a valid face for the union over the camera's cells; a^T X <= b is added to P_k (tightened camera by camera).
 B. poses from points. One LP per rotation cell over (w', C, X_1, ..., X_m): the linearised window rows of every
    landmark the camera sees, with X_k in P_k (its rows, not a box). All landmarks share the one centre C, so they
    pin the camera together. Infeasible cells are discarded; the centre box of the others is contracted.
Writes results_joint_poly/: summary.json, cells_<camera>.npz, landmark_polytopes.npz.
"""
import os
import json
import math
import time
import argparse
import itertools
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.spatial.transform import Rotation

from setlib import Scene, box_normals, expR, HERE, chebyshev
from union_incremental import gt_points
from joint_alternation import g_rem, maxdist

E3 = np.array([0, 0, 1.0])
_G = {}


def _init(data):
    _G.update(data)


def bbox(A, b):
    lo, hi = np.zeros(3), np.zeros(3)
    for d in range(3):
        c = np.zeros(3)
        c[d] = 1
        r1 = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        r2 = linprog(-c, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        if r1.status != 0 or r2.status != 0:
            return None
        lo[d], hi[d] = r1.fun, -r2.fun
    return lo, hi


def lin_rows(normals, R0, q0):
    """linearised window rows in (X, C, w'): n^T R0 (X - C) + (q0 x n)^T w' <= 0 (+ slack), depth row last."""
    G, rhs = [], []
    for N in normals:
        for n in N:
            g = n @ R0
            G.append(np.r_[g, -g, np.cross(q0, n)])
            rhs.append(0.0)
        G.append(np.r_[-R0[2], R0[2], -np.cross(q0, E3)])
        rhs.append(-1.0)                                 # times z_min
    return np.array(G), np.array(rhs)


# ---------------------------------------------------------------------------- step A
def _point_update(k):
    """new rows {camera: (dirs, b)} of landmark k; P_k is tightened camera by camera. None: inconsistent."""
    D = _G
    A, b = D["poly"][k]
    lo, hi = D["box"][k]
    new = {}
    for i in D["track"][k]:
        if i == D["cam1"]:
            continue
        normals = D["normals"][(i, k)]
        dirs = D["dirs"][(i, k)]
        best = np.full(len(dirs), -np.inf)
        X0 = 0.5 * (lo + hi)
        dX = maxdist(lo, hi, X0)
        for w0, h, clo, chi in D["cells"][i]:
            R0 = expR(w0) @ D["Rc"][i]
            delta = math.sqrt(3) * h
            C0 = 0.5 * (clo + chi)
            q0 = R0 @ (X0 - C0)
            Q = float(np.linalg.norm(np.maximum(np.abs(hi - clo), np.abs(lo - chi))))
            s = delta * (dX + maxdist(clo, chi, C0)) + g_rem(delta) * Q
            G, rr = lin_rows(normals, R0, q0)
            AA = np.vstack([G, np.hstack([A, np.zeros((len(A), 6))])])
            bb = np.r_[rr * D["zmin"] + s, b]
            bounds = list(zip(lo, hi)) + list(zip(clo, chi)) + [(-delta, delta)] * 3
            for j, a_ in enumerate(dirs):
                r = linprog(-np.r_[a_, np.zeros(6)], A_ub=AA, b_ub=bb, bounds=bounds, method="highs")
                if r.status != 0:
                    break                                # cell incompatible with this landmark
                best[j] = max(best[j], -r.fun)
        if not np.all(np.isfinite(best)):
            return k, None
        old = D["camrows"].get((k, i))
        if old is not None:
            best = np.minimum(best, old)
        new[i] = best
        A, b = np.vstack([A, dirs]), np.r_[b, best + 1e-10]
        bx = bbox(A, b)
        if bx is None:
            return k, None
        lo, hi = bx
    return k, (new, lo, hi)


# ---------------------------------------------------------------------------- step B
def _cell_test(task):
    """joint LP over (w', C, X_1..X_m) for one rotation cell of camera i; None or the contracted centre box."""
    i, w0, h, clo, chi = task
    D = _G
    L = D["seen"][i]
    R0 = expR(w0) @ D["Rc"][i]
    delta = math.sqrt(3) * h
    n = 6 + 3 * len(L)
    c0 = 0.5 * (clo + chi)
    dC = maxdist(clo, chi, c0)
    gd = g_rem(delta)
    data, ri, ci, b = [], [], [], []
    row = 0

    def put(cols, vals, rhs):
        nonlocal row
        for c, v in zip(cols, vals):
            if v != 0.0:
                ri.append(row)
                ci.append(c)
                data.append(v)
        b.append(rhs)
        row += 1

    for m, k in enumerate(L):
        lo, hi = D["box"][k]
        X0 = 0.5 * (lo + hi)
        q0 = R0 @ (X0 - c0)
        s = delta * (maxdist(lo, hi, X0) + dC) + gd * float(np.linalg.norm(np.maximum(np.abs(hi - clo), np.abs(lo - chi))))
        iv = 6 + 3 * m
        cols = [iv, iv + 1, iv + 2, 3, 4, 5, 0, 1, 2]
        G, rr = lin_rows(D["normals"][(i, k)], R0, q0)
        for g, r_ in zip(G, rr):
            put(cols, g, r_ * D["zmin"] + s)
        A, bk = D["poly"][k]
        for a_, b_ in zip(A, bk):
            put([iv, iv + 1, iv + 2], a_, b_)
    A = csr_matrix((data, (ri, ci)), shape=(row, n))
    bb = np.array(b)
    cb = list(zip(clo, chi))
    if i == D["cam2"]:
        cb[0] = (1.0, 1.0)
    bounds = [(-delta, delta)] * 3 + cb + [x for k in L for x in zip(*D["box"][k])]
    from scipy.sparse import hstack
    A2 = hstack([A, csr_matrix(-np.ones((row, 1)))]).tocsr()
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


def volume(A, b):
    try:
        x, rad = chebyshev(A, b)
        if rad <= 1e-9:
            return 0.0
        hs = HalfspaceIntersection(np.hstack([A, -b[:, None]]), x)
        return float(ConvexHull(hs.intersections).volume)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"])
    ap.add_argument("--cams", default="162,15,17,73,5,152")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--w0-deg", type=float, default=2.0)
    ap.add_argument("--c0", type=float, default=0.1)
    ap.add_argument("--x0", type=float, default=0.5)
    ap.add_argument("--h-start-deg", type=float, default=0.5)
    ap.add_argument("--h-min-deg", type=float, default=0.125)
    ap.add_argument("--max-cells", type=int, default=600)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    T0 = time.time()
    OUT = os.path.join(HERE, "results_joint_poly" + ("" if a.scheme == "fixed" else "_" + a.scheme) + a.tag)
    os.makedirs(OUT, exist_ok=True)
    cams = [int(x) for x in a.cams.split(",")]
    sc = Scene(pair=(cams[0], cams[1]), extra=tuple(cams[2:]))
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
    # directions of the rows camera i adds to P_k: its window faces and depth axis at COLMAP's rotation
    dirs = {(i, k): np.array([n @ Rc[i] for N in normals[(i, k)] for n in N] + [-Rc[i][2]])
            for k in L for i in track[k] if i != cam1}
    poly, box = {}, {}
    for k in L:
        X = sc.point(k)
        A = [np.eye(3), -np.eye(3)]
        b = [X + a.x0, -(X - a.x0)]
        if cam1 in track[k]:
            for N in normals[(cam1, k)]:
                A.append(N)
                b.append(np.zeros(4))
                A.append(-E3[None])
                b.append([-zmin])
        poly[k] = (np.vstack(A), np.concatenate(b))
        box[k] = bbox(*poly[k])
    base = dict(poly)
    camrows = {}
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
    old_box = None
    ob = os.path.join(HERE, "results_joint", "landmark_boxes.npz")
    if os.path.exists(ob):
        z = np.load(ob)
        old_box = {int(k): (lo_, hi_) for k, lo_, hi_ in zip(z["landmarks"], z["lo"], z["hi"])}
    print("cameras %s; %d landmarks; initial: rotation +-%g deg (%d cells of half-width %g deg), centre +-%g, "
          "points +-%g (cut by camera 1's pyramid)" % ([sc.imgs[i].name for i in cams], len(L), a.w0_deg, nper ** 3,
                                                     a.h_start_deg, a.c0, a.x0), flush=True)

    def report(tag, vol=False):
        rows = {}
        for i in cams[1:]:
            W = np.array([c[0] for c in cells[i]])
            h = np.array([c[1] for c in cells[i]])
            clo = np.min([c[2] for c in cells[i]], 0)
            chi = np.max([c[3] for c in cells[i]], 0)
            Rg, Cg = gt_pose[i]
            wg = Rotation.from_matrix(Rg @ Rc[i].T).as_rotvec()
            in_cell = any(np.all(np.abs(wg - c[0]) <= c[1] + 1e-12) and np.all((Cg >= c[2] - 1e-9) & (Cg <= c[3] + 1e-9))
                          for c in cells[i])
            rows[sc.imgs[i].name] = {"cells": len(cells[i]), "h_deg": float(np.degrees(h.max())),
                                     "w_lo_deg": np.degrees((W - h[:, None]).min(0)).tolist(),
                                     "w_hi_deg": np.degrees((W + h[:, None]).max(0)).tolist(),
                                     "C_lo": clo.tolist(), "C_hi": chi.tolist(), "C_size": (chi - clo).tolist(),
                                     "gt_in_a_cell": bool(in_cell), "gt_w_deg": np.degrees(wg).tolist()}
        diag = np.array([np.linalg.norm(box[k][1] - box[k][0]) for k in L])
        gt_in = [bool(np.all(poly[k][0] @ Xgt[k] <= poly[k][1] + 1e-9)) for k in L if k in Xgt]
        pts = {"landmarks": len(L), "bbox_diag_median": float(np.median(diag)), "bbox_diag_p90": float(np.quantile(diag, 0.9)),
               "rows_median": int(np.median([len(poly[k][1]) for k in L])),
               "gt_inside": int(sum(gt_in)), "with_gt": len(gt_in)}
        if vol:
            v = np.array([volume(*poly[k]) for k in L])
            pts["volume_median"] = float(np.nanmedian(v))
            if old_box is not None:
                vb = np.array([np.prod(old_box[k][1] - old_box[k][0]) if k in old_box else np.nan for k in L])
                pts["box_version_volume_median"] = float(np.nanmedian(vb))
                pts["volume_ratio_poly_over_box_median"] = float(np.nanmedian(v / vb))
        print("  [%s] points: bbox diagonal median %.4f (p90 %.4f); %d rows; GT inside %d / %d%s" %
              (tag, pts["bbox_diag_median"], pts["bbox_diag_p90"], pts["rows_median"], pts["gt_inside"], pts["with_gt"],
               "; volume median %.2e (box version %.2e, ratio %.3f)" % (pts["volume_median"], pts.get("box_version_volume_median", np.nan),
                                                                     pts.get("volume_ratio_poly_over_box_median", np.nan)) if vol else ""),
              flush=True)
        for nm, r in rows.items():
            print("  [%s] %-12s %4d cells (h %.3f deg); w %s .. %s deg; C size %s; GT pose in a cell: %s"
                  % (tag, nm, r["cells"], r["h_deg"], np.round(r["w_lo_deg"], 3), np.round(r["w_hi_deg"], 3),
                     np.round(r["C_size"], 4), r["gt_in_a_cell"]), flush=True)
        return {"points": pts, "cameras": rows}

    history = [report("initial")]
    hmin = math.radians(a.h_min_deg)
    for it in range(1, a.iters + 1):
        t = time.time()
        data = {"poly": poly, "box": box, "track": track, "normals": normals, "dirs": dirs, "cells": cells,
                "cam1": cam1, "Rc": Rc, "zmin": zmin, "camrows": camrows}
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            res = pool.map(_point_update, L, chunksize=4)
        bad = 0
        for k, out in res:
            if out is None:
                bad += 1
                continue
            new, lo, hi = out
            camrows.update({(k, i): bi for i, bi in new.items()})
            A, b = base[k]
            for i in track[k]:
                if (k, i) in camrows:
                    A, b = np.vstack([A, dirs[(i, k)]]), np.r_[b, camrows[(k, i)] + 1e-10]
            poly[k], box[k] = (A, b), (lo, hi)
        print("iteration %d, A (points): %d inconsistent landmarks kept unchanged (%.0fs)" % (it, bad, time.time() - t), flush=True)
        t = time.time()
        data = {"poly": poly, "box": box, "seen": seen, "normals": normals, "Rc": Rc, "zmin": zmin, "cam2": cam2}
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            for i in cams[1:]:
                out = pool.map(_cell_test, [(i, w0, h, clo, chi) for w0, h, clo, chi in cells[i]], chunksize=1)
                keep = [(w0, h, r[0], r[1]) for (w0, h, _, _), r in zip(cells[i], out) if r is not None]
                print("   %s: %d / %d cells kept" % (sc.imgs[i].name, len(keep), len(cells[i])), flush=True)
                if not keep:
                    print("   %s: every cell discarded (inconsistent at eps); previous cells kept" % sc.imgs[i].name)
                    continue
                if keep[0][1] / 2 >= hmin - 1e-12 and 8 * len(keep) <= a.max_cells:
                    keep = [(w0 + 0.5 * h * np.array(s), 0.5 * h, clo.copy(), chi.copy())
                            for w0, h, clo, chi in keep for s in itertools.product((-1, 1), repeat=3)]
                cells[i] = keep
        print("iteration %d, B (poses) (%.0fs)" % (it, time.time() - t), flush=True)
        history.append(report("iteration %d" % it, vol=(it == a.iters)))
    json.dump({"eps_px": a.eps, "scheme": a.scheme, "cameras": [sc.imgs[i].name for i in cams],
               "initial": {"w_deg": a.w0_deg, "c": a.c0, "x": a.x0, "h_start_deg": a.h_start_deg},
               "iterations": history, "seconds": round(time.time() - T0)},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, "landmark_polytopes.npz"), landmarks=np.array(L),
                        A=np.array([poly[k][0] for k in L], dtype=object), b=np.array([poly[k][1] for k in L], dtype=object),
                        bbox_lo=np.array([box[k][0] for k in L]), bbox_hi=np.array([box[k][1] for k in L]))
    for i in cams[1:]:
        np.savez_compressed(os.path.join(OUT, "cells_%d.npz" % i), w=np.array([c[0] for c in cells[i]]),
                            h=np.array([c[1] for c in cells[i]]), c_lo=np.array([c[2] for c in cells[i]]),
                            c_hi=np.array([c[3] for c in cells[i]]))
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
