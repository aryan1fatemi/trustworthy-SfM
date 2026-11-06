"""Fast per-camera rotation bounds for known_rotation.py --rot (decoupled, outer, not optimal).

    python rotation_bounds.py --scene /path/to/airplane --model "/path/to/airplane/COLMAP results fixedK" \
        --centres results/<run>_known_rotation_198cams.json --procs 8

Rotations are written R_i = R_hat_i Exp(delta_i) (R_hat_i = COLMAP's). Instead of the joint LP, two small decoupled
LP stages are alternated, both with the rotation linearised in delta and the slack of known_rotation.rot_slack:
  1. landmark boxes: for every track, min / max of X_k over its own rows with every viewing camera's centre in its
     box (the centre ranges of a known-rotation run, widened by --margin) and delta_j in its current bounds;
  2. rotation bounds: for every camera, min / max of each delta_i component over its own rows with its centre in its
     box and every landmark it sees in its box from step 1.
Start: |delta_i| <= --rho0 degrees for every camera; each round shrinks the bounds. The result contains the true
rotation if (a) the true rotation errors are within --rho0, (b) the true centres are inside the widened boxes and
(c) every detection's window holds at the true geometry. (b) is an assumption: the boxes come from the fixed-rotation
run. Decoupling drops all correlation between cameras and points, so the bounds are loose but each LP is small.
Writes <out>/<run>_rotation_bounds.json, the input of known_rotation.py --rot.
"""
import os
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from geom import Scene, registration_order, run_name
from colmap_io import read_model
from known_rotation import detection_rows, rot_slack

INF = float("inf")


def corner_dist(lo, hi, c):
    """farthest box corner from c (inf if the box is unbounded)."""
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return INF
    return float(np.linalg.norm(np.maximum(np.abs(hi - c), np.abs(c - lo))))


_W = {}


def _init(data):
    _W.update(data)


def _box_lp(c, A, b, bnds):
    """min and max of each of the first three variables."""
    lo, hi = np.full(3, -INF), np.full(3, INF)
    for d in range(3):
        for sg in (1, -1):
            cc = np.zeros(A.shape[1])
            cc[d] = sg
            r = linprog(cc, A_ub=A, b_ub=b, bounds=bnds, method="highs")
            if r.status == 0:
                (lo if sg == 1 else hi)[d] = sg * r.fun
            elif r.status == 2:
                return None, None                                   # infeasible under the current assumptions
    return lo, hi


def _track(k):
    """stage 1: box of X_k. Variables (X_k, then C_j and delta_j of every viewing camera j except camera 1)."""
    W = _W
    views = [i for i in W["views"][k]]
    blocks, A, b, bnds = {}, [], [], [(None, None)] * 3
    for i in views:
        if i != W["i1"]:
            blocks[i] = len(bnds)
            bnds += [(lo, hi) for lo, hi in zip(*W["cbox"][i])]
            bnds += [(lo, hi) for lo, hi in zip(*W["rbox"][i])]
    n = len(bnds)
    for i in views:
        G, h = W["rows"][(i, k)]
        row = np.zeros((len(G), n))
        row[:, :3] = G
        if i != W["i1"]:
            D = W["rX"][k] + W["rC"][i]
            if not np.isfinite(D):
                continue
            dhat = W["Xhat"][k] - W["Chat"][i]
            h = h + rot_slack(W["theta"][i], D, np.linalg.norm(dhat))
            m = blocks[i]
            row[:, m:m + 3] = -G
            row[:, m + 3:m + 6] = np.cross(dhat, G)
        A.append(row)
        b.append(h)
    if not A:
        return k, np.full(3, -INF), np.full(3, INF)
    fin = lambda v: None if v is None or not np.isfinite(v) else float(v)
    lo, hi = _box_lp(None, np.vstack(A), np.concatenate(b), [(fin(l), fin(u)) for l, u in bnds])
    return k, lo, hi


def _camera(i):
    """stage 2: bounds of delta_i. Variables (delta_i, C_i, X_k of every landmark camera i sees with a finite box)."""
    W = _W
    tracks = [k for k in W["tracks_of"][i] if np.isfinite(W["rX"][k])]
    fin = lambda v: None if not np.isfinite(v) else float(v)
    bnds = [(fin(l), fin(u)) for l, u in zip(*W["rbox"][i])] + [(fin(l), fin(u)) for l, u in zip(*W["cbox"][i])]
    off = {}
    for k in tracks:
        off[k] = len(bnds)
        bnds += [(fin(l), fin(u)) for l, u in zip(*W["xbox"][k])]
    n = len(bnds)
    A, b = [], []
    for k in tracks:
        D = W["rX"][k] + W["rC"][i]
        if not np.isfinite(D):
            continue
        G, h = W["rows"][(i, k)]
        dhat = W["Xhat"][k] - W["Chat"][i]
        row = np.zeros((len(G), n))
        row[:, :3] = np.cross(dhat, G)
        row[:, 3:6] = -G
        row[:, off[k]:off[k] + 3] = G
        A.append(row)
        b.append(h + rot_slack(W["theta"][i], D, np.linalg.norm(dhat)))
    if not A:
        return i, None, None, 0
    lo, hi = _box_lp(None, np.vstack(A), np.concatenate(b), bnds)
    return i, lo, hi, len(tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--centres", required=True, help="json of a known_rotation.py run (centre_ranges)")
    ap.add_argument("--n-cams", type=int, default=200)
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--scheme", default="mult", choices=["fixed", "mult", "add"])
    ap.add_argument("--rho0", type=float, default=2.0, help="starting bound on every rotation error (degrees)")
    ap.add_argument("--margin", type=float, default=0.02, help="widening of the centre boxes (gauge units)")
    ap.add_argument("--D-unsolved", type=float, default=0.5,
                    help="|C_i - C_hat_i| bound for cameras with an unsolved centre axis (gauge units)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    T0 = time.time()
    model = a.model or os.path.join(a.scene, "COLMAP results")
    _, imgs_all, _ = read_model(model)
    cams = registration_order(a.scene, imgs_all, a.n_cams, model)
    sc = Scene(a.scene, model, cams)
    zmin = 1e-3
    cnt = {}
    for i in cams:
        for k in sc.det[i]:
            cnt[k] = cnt.get(k, 0) + 1
    L = sorted(k for k, c in cnt.items() if c >= 2)
    views = {k: [i for i in cams if k in sc.det[i]] for k in L}
    tracks_of = {i: [k for k in sc.det[i] if k in views] for i in cams}
    rows = {(i, k): detection_rows(sc, i, k, a.eps, a.scheme, zmin) for k in L for i in views[k]}
    Chat = {i: sc.pose(i)[1] for i in cams}
    Xhat = {k: sc.point(k) for k in L}

    # centre boxes from the fixed-rotation run, widened; unsolved axes are free
    J = json.load(open(a.centres))
    cbox, rC = {}, {}
    for i in cams:
        if i == sc.i1:
            cbox[i], rC[i] = (np.zeros(3), np.zeros(3)), 0.0
            continue
        r = J["centre_ranges"].get(sc.imgs[i].name)
        lo = np.array(r["lo"], float) - a.margin if r else np.full(3, -INF)
        hi = np.array(r["hi"], float) + a.margin if r else np.full(3, INF)
        lo[~np.isfinite(lo)], hi[~np.isfinite(hi)] = -INF, INF
        if i == sc.i2:
            lo[0] = hi[0] = sc.x2
        cbox[i] = (lo, hi)
        rC[i] = corner_dist(lo, hi, Chat[i])
        if not np.isfinite(rC[i]):
            rC[i] = a.D_unsolved                    # assumption for cameras whose box is incomplete
    rho0 = np.radians(a.rho0)
    rbox = {i: (np.full(3, -rho0), np.full(3, rho0)) for i in cams if i != sc.i1}
    theta = {i: rho0 for i in rbox}
    rX = {k: a.D_unsolved for k in L}                # first round: assumed bound on |X_k - X_hat_k|
    xbox = {}
    gt = sc.load_gt()
    gtd = {}
    if gt is not None:
        gp, _ = sc.gt_in_gauge(gt, cams, [])
        gtd = {i: Rotation.from_matrix(sc.pose(i)[0].T @ gp[i][0]).as_rotvec() for i in rbox}
    print("%d cameras, %d landmarks, %d (camera, landmark) pairs; start |delta| <= %.2f deg"
          % (len(cams), len(L), len(rows), a.rho0), flush=True)
    hist = []
    for rnd in range(1, a.rounds + 1):
        data = dict(views=views, tracks_of=tracks_of, rows=rows, Chat=Chat, Xhat=Xhat, cbox=cbox, rC=rC,
                    rbox=rbox, theta=theta, rX=rX, xbox=xbox, i1=sc.i1)
        t = time.time()
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            res = pool.map(_track, L, chunksize=32)
        bad = 0
        for k, lo, hi in res:
            if lo is None:
                bad += 1
                lo, hi = np.full(3, -INF), np.full(3, INF)
            xbox[k] = (lo, hi)
            rX[k] = corner_dist(lo, hi, Xhat[k])
        fin = np.array([v for v in rX.values() if np.isfinite(v)])
        print("round %d stage 1: %d landmark boxes in %.0fs; bounded %d (median |X - X_hat| bound %.3f), infeasible %d"
              % (rnd, len(L), time.time() - t, len(fin), np.median(fin), bad), flush=True)
        data.update(rX=rX, xbox=xbox)
        t = time.time()
        with Pool(a.procs, initializer=_init, initargs=(data,)) as pool:
            res = pool.map(_camera, list(rbox), chunksize=4)
        new = {}
        for i, lo, hi, nt in res:
            if lo is None:
                continue
            lo, hi = np.maximum(lo, rbox[i][0]), np.minimum(hi, rbox[i][1])
            new[i] = (lo, hi)
        for i, (lo, hi) in new.items():
            rbox[i] = (lo, hi)
            theta[i] = float(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))))
        th = np.degrees(np.array(list(theta.values())))
        inside = [bool(np.all((gtd[i] >= rbox[i][0] - 1e-9) & (gtd[i] <= rbox[i][1] + 1e-9))) for i in gtd]
        err = np.degrees([np.linalg.norm(gtd[i]) for i in gtd]) if gtd else []
        print("round %d stage 2: %d cameras in %.0fs; |delta| bound median %.3f, 90%% %.3f, max %.3f deg%s"
              % (rnd, len(new), time.time() - t, np.median(th), np.percentile(th, 90), th.max(),
                 ("; true delta inside %d / %d (true error median %.3f deg)" % (sum(inside), len(inside), np.median(err)))
                 if gtd else ""), flush=True)
        hist.append(dict(round=rnd, theta_deg_median=float(np.median(th)), theta_deg_max=float(th.max()),
                         gt_inside=int(sum(inside)) if gtd else None))
    out = {"scene": a.scene, "model": model, "centres": a.centres, "rho0_deg": a.rho0, "margin": a.margin,
           "rounds": hist, "seconds": round(time.time() - T0),
           "cameras": {sc.imgs[i].name: {"lo": rbox[i][0].tolist(), "hi": rbox[i][1].tolist(), "theta": theta[i],
                                          **({"gt": gtd[i].tolist()} if i in gtd else {})} for i in rbox},
           "D_cam": {sc.imgs[i].name: rC[i] for i in cams},
           "D_track": {int(k): (rX[k] if np.isfinite(rX[k]) else None) for k in L}}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "%s_rotation_bounds.json" % run_name(a.scene, a.model))
    json.dump(out, open(path, "w"), indent=1)
    print("-> %s (%ds)" % (path, out["seconds"]))


if __name__ == "__main__":
    main()
