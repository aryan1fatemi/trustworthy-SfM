"""Guaranteed rotation bounds by search over rotation space (Hartley & Kahl 2009, IJCV), with the LP's l-inf windows.

    python rotation_bnb.py --scene /path/to/chair_spiral30 --model "/path/.../COLMAP results exhaustive" \
        --n-cams 30 --scheme mult --procs 32 --out results_bnb

Pair (i, j), camera i at the origin with R = I. The relative rotation is Q = Q_hat Exp(r) (Q_hat = COLMAP's), so the
angle between Q and Q_hat is |r|. The matches of the pair are every track seen by both cameras.
For fixed Q the pair is feasible iff there is a centre C_j and points X_m with every detection inside its window:
    camera i:  n' X_m <= 0 (4 window faces),  z_i(X_m) >= zmin
    camera j:  n' Q (X_m - C_j) <= 0,         z_j >= zmin
which is an LP. Scale is fixed by |C_j|_inf = 1 (6 LPs, one per face of the cube; the pair is feasible iff one is).
Cube test: for every r in a cube of half-width h around r_c, angle(Q(r), Q(r_c)) <= sqrt(3) h = s, so the true
camera-j ray u = Q (X - C) is within angle s of u_c = Q(r_c) (X - C). If u is in the window cone K, then u_c is within
s of K, hence n' u_c <= sin(s) |u_c| <= sin(s) / cos(a) d' u_c for every face n (d = the detection's unit ray,
a = s + the window's half-diagonal angle). These rows are linear in (X, C): if the relaxed LP is infeasible, NO
rotation in the cube is feasible and the cube is discarded (no linearisation, nothing assumed about the rotation).
Search: best-first on the farthest corner from r = 0 (= COLMAP); the first cube that is feasible and smaller than
--h-min gives the bound  angle(Q_true, Q_hat) <= its farthest corner distance.
Absolute bounds: camera 1 (gauge) has R = R_hat = I; along a path of pairs the angles add (triangle inequality of
the rotation angle metric), so theta_j <= the shortest path from camera 1. Valid if every detection of the used
pairs is inside its window at the true geometry (for some 3D point) and the depth of every point exceeds zmin.
Writes <out>/<run>_rotation_bnb.json in the format of rotation_bounds.py (cameras: lo, hi, theta) plus the pairs."""
import os
import json
import time
import heapq
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from geom import Scene, registration_order, run_name, box_normals
from colmap_io import read_model

_W = {}


def pair_data(sc, i, j, eps, scheme):
    """per match: camera-i faces (4,3), camera-j faces (4,3), camera-j unit ray d (3), half-diagonal angle beta."""
    Ni, Nj, D, B = [], [], [], []
    for k in set(sc.det[i]) & set(sc.det[j]):
        for xi, ei in zip(sc.det[i][k], sc.eps_list(i, k, eps, scheme)):
            for xj, ej in zip(sc.det[j][k], sc.eps_list(j, k, eps, scheme)):
                Ni.append(box_normals(xi, sc.f[i], ei))
                Nj.append(box_normals(xj, sc.f[j], ej))
                d = np.array([xj[0], xj[1], 1.0])
                d /= np.linalg.norm(d)
                cs = [np.array([xj[0] + a * ej / sc.f[j][0], xj[1] + b * ej / sc.f[j][1], 1.0]) for a in (-1, 1) for b in (-1, 1)]
                B.append(max(np.arccos(np.clip(d @ c / np.linalg.norm(c), -1, 1)) for c in cs))
                D.append(d)
    return np.array(Ni), np.array(Nj), np.array(D), np.array(B)


def feasible(Qc, s, P, zmin, face_order):
    """relaxed LP for the cube with centre rotation Qc and angle radius s. True if feasible (or undecidable)."""
    Ni, Nj, D, B = P
    M = len(Ni)
    alpha = B + s
    if alpha.max() >= np.radians(85):
        return True                                       # the relaxation is vacuous: cannot discard
    kap = np.sin(s) / np.cos(alpha)
    Gj = (Nj - kap[:, None, None] * D[:, None, :]) @ Qc   # (M, 4, 3): rows on v = X - C
    dz = D @ Qc                                           # rows d' Qc: d' Qc v >= zmin cos(alpha)
    # variables: C (3), X_m (3 each). Per match 10 rows: 4 cam i, 1 depth i, 4 cam j, 1 depth j
    rows, cols, vals, b = [], [], [], []
    r0 = 0
    for m in range(M):
        xc = 3 + 3 * m
        for f in range(4):                                # camera i: n' X <= 0
            rows += [r0] * 3; cols += [xc, xc + 1, xc + 2]; vals += list(Ni[m, f]); b.append(0.0); r0 += 1
        rows.append(r0); cols.append(xc + 2); vals.append(-1.0); b.append(-zmin); r0 += 1
        for f in range(4):                                # camera j: g' (X - C) <= 0
            g = Gj[m, f]
            rows += [r0] * 6; cols += [xc, xc + 1, xc + 2, 0, 1, 2]; vals += list(g) + list(-g); b.append(0.0); r0 += 1
        g = dz[m]
        rows += [r0] * 6; cols += [xc, xc + 1, xc + 2, 0, 1, 2]; vals += list(-g) + list(g)
        b.append(-zmin * np.cos(alpha[m])); r0 += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r0, 3 + 3 * M)).tocsr()
    b = np.array(b)
    for ax, sg in face_order:
        bnds = [(-1.0, 1.0)] * 3 + [(None, None)] * (3 * M)
        bnds[ax] = (sg, sg)
        r = linprog(np.zeros(3 + 3 * M), A_ub=A, b_ub=b, bounds=bnds, method="highs")
        if r.status != 2:                                 # feasible, or not proven infeasible: keep the cube
            return True
    return False


def search(Qhat, P, zmin, h0, hmin, face_order, max_nodes):
    """upper bound on |r| over feasible r (angle to COLMAP's relative rotation), or None if no rotation is feasible."""
    far = lambda rc, h: float(np.linalg.norm(np.abs(rc) + h))
    heap = [(-far(np.zeros(3), h0), h0, (0.0, 0.0, 0.0))]
    nodes = lps = 0
    while heap:
        nf, h, rc = heapq.heappop(heap)
        nodes += 1
        if nodes > max_nodes:
            return -nf, nodes, "max_nodes"                # still a valid bound: nothing farther survived
        Qc = Qhat @ Rotation.from_rotvec(rc).as_matrix()
        if not feasible(Qc, np.sqrt(3) * h, P, zmin, face_order):
            continue
        if h <= hmin:
            return -nf, nodes, "ok"
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    c = np.array(rc) + 0.5 * h * np.array([dx, dy, dz])
                    if np.linalg.norm(c) - np.sqrt(3) * h / 2 > np.pi:
                        continue                          # entirely outside the rotation ball
                    heapq.heappush(heap, (-far(c, h / 2), h / 2, tuple(c)))
    return None, nodes, "infeasible"


def _init(d):
    _W.update(d)


def _pair(task):
    i, j = task
    t = time.time()
    sc = _W["sc"]
    P = pair_data(sc, i, j, _W["eps"], _W["scheme"])
    Ri, Rj = sc.pose(i)[0], sc.pose(j)[0]
    Qhat = Rj @ Ri.T
    Ci, Cj = sc.pose(i)[1], sc.pose(j)[1]
    base = Ri @ (Cj - Ci)                                 # COLMAP's baseline in camera i: try its face first
    ax = int(np.argmax(np.abs(base)))
    first = (ax, float(np.sign(base[ax])))
    order = [first] + [(a, s) for a in range(3) for s in (1.0, -1.0) if (a, s) != first]
    bound, nodes, status = search(Qhat, P, _W["zmin"], _W["h0"], _W["hmin"], order, _W["max_nodes"])
    out = dict(i=i, j=j, matches=len(P[0]), bound=bound, nodes=nodes, status=status, seconds=round(time.time() - t, 1))
    if _W["gt"]:
        gi, gj = _W["gt"][i][0], _W["gt"][j][0]
        out["gt_angle"] = float(np.linalg.norm(Rotation.from_matrix(Qhat.T @ (gj @ gi.T)).as_rotvec()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-cams", type=int, default=30)
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--scheme", default="mult", choices=["fixed", "mult", "add"])
    ap.add_argument("--min-matches", type=int, default=30, help="pairs with fewer matches are not used")
    ap.add_argument("--max-pairs-per-camera", type=int, default=6, help="each camera's pairs with the most matches")
    ap.add_argument("--h0", type=float, default=180.0, help="half-width of the first cube (deg); 180 = all rotations")
    ap.add_argument("--h-min", type=float, default=0.01, help="smallest cube half-width (deg)")
    ap.add_argument("--zmin", type=float, default=1e-3, help="depth lower bound, in units of |C_j|_inf")
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--pairs", default=None, help="only these pairs, e.g. '22.png:21.png,22.png:23.png' (testing)")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    T0 = time.time()
    model = a.model or os.path.join(a.scene, "COLMAP results")
    _, imgs_all, _ = read_model(model)
    cams = registration_order(a.scene, imgs_all, a.n_cams, model)
    sc = Scene(a.scene, model, cams)
    gt = sc.load_gt()
    gp = sc.gt_in_gauge(gt, cams, [])[0] if gt is not None and all(sc.imgs[i].name in gt for i in cams) else {}
    name = {i: sc.imgs[i].name for i in cams}
    by_name = {v: k for k, v in name.items()}
    if a.pairs:
        pairs = [tuple(by_name[n] for n in p.split(":")) for p in a.pairs.split(",")]
    else:
        cnt = {(i, j): len(set(sc.det[i]) & set(sc.det[j])) for x, i in enumerate(cams) for j in cams[x + 1:]}
        keep = set()
        for i in cams:
            mine = sorted(((c, p) for p, c in cnt.items() if i in p and c >= a.min_matches), reverse=True)
            keep |= {p for _, p in mine[:a.max_pairs_per_camera]}
        pairs = sorted(keep, key=lambda p: -cnt[p])
    print("%d cameras, %d pairs; first cube +-%.0f deg, h_min %.3f deg (%.0fs)"
          % (len(cams), len(pairs), a.h0, a.h_min, time.time() - T0), flush=True)
    data = dict(sc=sc, eps=a.eps, scheme=a.scheme, zmin=a.zmin, h0=np.radians(a.h0), hmin=np.radians(a.h_min),
                max_nodes=a.max_nodes, gt=gp)
    res = []
    with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
        for r in pool.imap_unordered(_pair, pairs):
            res.append(r)
            print("  %-8s %-8s matches %4d  bound %s  true %s  nodes %5d  %s  %.0fs"
                  % (name[r["i"]], name[r["j"]], r["matches"],
                     "%.3f deg" % np.degrees(r["bound"]) if r["bound"] is not None else "EMPTY",
                     "%.3f deg" % np.degrees(r["gt_angle"]) if "gt_angle" in r else "n/a",
                     r["nodes"], r["status"], r["seconds"]), flush=True)
    ok = [r for r in res if r["bound"] is not None]
    if gp:
        ins = sum(r["gt_angle"] <= r["bound"] + 1e-9 for r in ok)
        print("pairs: true relative rotation inside the bound for %d / %d; bound median %.3f deg, true median %.3f deg"
              % (ins, len(ok), np.degrees(np.median([r["bound"] for r in ok])), np.degrees(np.median([r["gt_angle"] for r in ok]))))
    # absolute bounds: shortest path from camera 1 over the pair graph (Dijkstra)
    adj = {i: [] for i in cams}
    for r in ok:
        adj[r["i"]].append((r["j"], r["bound"]))
        adj[r["j"]].append((r["i"], r["bound"]))
    theta = {sc.i1: 0.0}
    heap = [(0.0, sc.i1)]
    while heap:
        t, i = heapq.heappop(heap)
        if t > theta.get(i, np.inf):
            continue
        for j, w in adj[i]:
            if t + w < theta.get(j, np.inf):
                theta[j] = t + w
                heapq.heappush(heap, (t + w, j))
    out = {"note": "rotation_bnb.py: per-camera |delta| <= theta (guaranteed under the window assumption); "
                   "box lo/hi = +-theta", "scene": a.scene, "model": model, "eps": a.eps, "scheme": a.scheme,
           "h0_deg": a.h0, "h_min_deg": a.h_min, "seconds": round(time.time() - T0),
           "pairs": [dict(r, i=name[r["i"]], j=name[r["j"]]) for r in res], "cameras": {}}
    errs = []
    for i in cams[1:]:
        if i not in theta:
            print("  %-8s no path from camera 1: unbounded" % name[i])
            continue
        th = theta[i]
        c = {"lo": [-th] * 3, "hi": [th] * 3, "theta": th}
        if gp:
            e = float(np.linalg.norm(Rotation.from_matrix(sc.pose(i)[0].T @ gp[i][0]).as_rotvec()))
            c["gt"] = Rotation.from_matrix(sc.pose(i)[0].T @ gp[i][0]).as_rotvec().tolist()
            c["gt_inside"] = bool(e <= th + 1e-9)
            errs.append((th, e))
        out["cameras"][name[i]] = c
    th = np.degrees([t for t, _ in errs] or [v["theta"] for v in out["cameras"].values()])
    print("cameras: theta median %.3f deg, max %.3f deg" % (np.median(th), th.max())
          + ("; true rotation inside %d / %d (true error median %.3f deg)"
             % (sum(e <= t + 1e-9 for t, e in errs), len(errs), np.degrees(np.median([e for _, e in errs]))) if errs else ""))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "%s_rotation_bnb.json" % run_name(a.scene, a.model))
    json.dump(out, open(path, "w"), indent=1)
    print("-> %s (%ds)" % (path, out["seconds"]))


if __name__ == "__main__":
    main()
