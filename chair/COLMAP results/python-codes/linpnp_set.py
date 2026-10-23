"""Pose set of image 3 by linearised LPs (Jacobian of the rotation), with an inner search.

    python linpnp_set.py --scene airplane --eps 3 --delta0-deg 5

Same setup as pnp_set.py: cameras 1, 2 (#162, #15) held at COLMAP's poses, gauge R_1 = I,
c_1 = 0, x of c_2 = 1; F_k = {X : G_k X <= h_k} from cameras 1 and 2; camera 3 = #17.
Rotation of camera 3: R = exp([w]x) R0. With q_hat = R0 (X - c) (linear in X, c) and a
reference q0_k = R0 (X0_k - c0):

    exp([w]x) q_hat = q_hat + w x q0 + w x (q_hat - q0) + r,   ||r|| <= g(|w|) ||q_hat||,
    g(t) = t - sin t + 1 - cos t   (exact bound from Rodrigues' formula; ~ t^2 / 2)

(1) PLAIN linearised LP: keep q_hat + w x q0 (linear in w, c, X), drop the rest. Each face
    n^T q <= 0 becomes  n^T q_hat + (q0 x n)^T w <= 0. Min/max of every component of w and c.
    Fast, approximate: not a guarantee.
(2) SOUND linearised branch and bound over w in [-delta0, delta0]^3: in a cell (centre w0,
    half-width h, delta = sqrt(3) h) linearise at R0 = exp([w0]x) R_hat and add the slack
        s_k = delta * Delta_k + g(delta) * Q_k,
    Delta_k >= ||q_hat - q0|| and Q_k >= ||q_hat|| from boxes known to contain the truth
    (landmark boxes of F_k; centre box, starting from pnp_set.py's sound box and tightened
    by iterating: each LP's centre box contains the truth, so it can replace the previous one).
    Infeasible cell -> discarded. Survivors give rotation boxes w0 + [w'_lo, w'_hi] and
    centre boxes; their union is the outer bound (conditional on |w_true| <= delta0).
(3) INNER radial search: from a feasible rotation, for many directions u, bisect on t until
    exp([t u]x) R is infeasible, using the EXACT fixed-rotation LP (no linearisation). Every
    accepted rotation is proven feasible; its exact centre range is computed too.
GT (same gauge rule) is used only in the final comparison.
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
import cp_conformal as CC
from stage_a import undistort
from pnp_set import box_normals, fixed_rows, expR

E3 = np.array([0, 0, 1.0])


def g_rem(t):
    return t - math.sin(t) + 1 - math.cos(t)


def maxdist(lo, hi, p):
    """max ||x - p|| over the box [lo, hi]."""
    return float(np.linalg.norm(np.maximum(np.abs(lo - p), np.abs(hi - p))))


class Data:
    """variables: w (3), c (3), X_k (3 each, k in L)."""

    def __init__(self, L, Fk, Fbox, X0, obs3, zmin):
        self.L, self.Fk, self.Fbox, self.X0, self.obs3, self.zmin = L, Fk, Fbox, X0, obs3, zmin
        self.idx = {k: 6 + 3 * n for n, k in enumerate(L)}
        self.n = 6 + 3 * len(L)
        rows, b = [], []
        for k in L:
            for gr, h in zip(*Fk[k]):
                r = np.zeros(self.n)
                r[self.idx[k]:self.idx[k] + 3] = gr
                s = np.linalg.norm(gr)
                rows.append(r / s)
                b.append(h / s)
        self.A_F, self.b_F = np.array(rows), np.array(b)

    def rows_linear(self, R0, c0, cbox, delta, wbound):
        """linearised camera-3 rows around (R0, c0, X0); slack from boxes if delta > 0."""
        rows, b = [], []
        for k, N in self.obs3:
            q0 = R0 @ (self.X0[k] - c0)
            if delta > 0:
                lo, hi = self.Fbox[k]
                D = maxdist(lo, hi, self.X0[k]) + maxdist(cbox[0], cbox[1], c0)
                Q = float(np.linalg.norm(np.maximum(np.abs(hi - cbox[0]), np.abs(lo - cbox[1]))))
                s = delta * D + g_rem(delta) * Q
            else:
                s = 0.0
            iv = self.idx[k]
            for nrm in N:
                r = np.zeros(self.n)
                r[0:3] = np.cross(q0, nrm)
                r[3:6] = -(nrm @ R0)
                r[iv:iv + 3] = nrm @ R0
                rows.append(r)
                b.append(s)
            r = np.zeros(self.n)                      # depth: q3 >= zmin
            r[0:3] = -np.cross(q0, E3)
            r[3:6] = R0[2]
            r[iv:iv + 3] = -R0[2]
            rows.append(r)
            b.append(-self.zmin + s)
        A = np.vstack([self.A_F, np.array(rows)])
        bb = np.concatenate([self.b_F, np.array(b)])
        bounds = [(-wbound, wbound)] * 3 + [(None, None)] * (self.n - 3)
        return A, bb, bounds

    def rows_exact(self, R):
        """exact camera-3 rows for a FIXED rotation R (w fixed to 0)."""
        rows, b = [], []
        for k, N in self.obs3:
            iv = self.idx[k]
            for nrm in list(N) + [-E3]:
                r = np.zeros(self.n)
                r[3:6] = -(nrm @ R)
                r[iv:iv + 3] = nrm @ R
                rows.append(r)
                b.append(0.0 if nrm[2] != -1 or nrm[0] != 0 else -self.zmin)
        A = np.vstack([self.A_F, np.array(rows)])
        bb = np.concatenate([self.b_F, np.array(b)])
        bounds = [(0, 0)] * 3 + [(None, None)] * (self.n - 3)
        return A, bb, bounds


def lp_feasible(A, b, bounds):
    A2 = np.hstack([A, -np.ones((len(A), 1))])
    c = np.zeros(A.shape[1] + 1)
    c[-1] = 1
    res = linprog(c, A_ub=A2, b_ub=b, bounds=bounds + [(0, None)], method="highs")
    return bool(res.status == 0 and res.fun <= 1e-8)


def lp_range(A, b, bounds, var):
    out = []
    for sg in (1.0, -1.0):
        cv = np.zeros(A.shape[1])
        cv[var] = sg
        res = linprog(cv, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        out.append(sg * res.fun if res.status == 0 else (-sg * np.inf if res.status == 3 else np.nan))
    return out


# ---- workers -------------------------------------------------------------------
_W = {}


def _init(data, Rhat, cbox):
    _W.update(data=data, Rhat=Rhat, cbox=cbox)


def _cell(cell, iters=4):
    """sound test of one rotation cell; returns None (discarded) or (w_box, c_box)."""
    w0, h = cell
    D = _W["data"]
    R0 = expR(w0) @ _W["Rhat"]
    delta = math.sqrt(3) * h
    cbox = (_W["cbox"][0].copy(), _W["cbox"][1].copy())
    for it in range(iters):
        c0 = 0.5 * (cbox[0] + cbox[1])
        # every rotation of the cell is within delta of R0 (Hartley-Kahl), so its offset w' from
        # R0 satisfies ||w'|| <= delta, but not necessarily |w'|_inf <= h: bound w' by delta
        A, b, bnds = D.rows_linear(R0, c0, cbox, delta, delta)
        if not lp_feasible(A, b, bnds):
            return None
        rng = np.array([lp_range(A, b, bnds, 3 + d) for d in range(3)])
        new = (np.maximum(cbox[0], rng[:, 0]), np.minimum(cbox[1], rng[:, 1]))
        shrink = np.max((cbox[1] - cbox[0]) - (new[1] - new[0]))
        cbox = new
        if shrink < 1e-4:
            break
    wr = np.array([lp_range(A, b, bnds, d) for d in range(3)])
    return (wr[:, 0], wr[:, 1]), cbox          # w' range relative to R0 = exp([w0]x) R_hat


def _ray(u, t_max, iters):
    """bisection along direction u from the start rotation, exact LPs."""
    D, R = _W["data"], _W["Rhat"]
    feas = lambda t: lp_feasible(*D.rows_exact(expR(t * u) @ R))
    if feas(t_max):
        t_lo = t_max
    else:
        lo, hi = 0.0, t_max
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if feas(mid) else (lo, mid)
        t_lo = lo
    A, b, bnds = D.rows_exact(expR(t_lo * u) @ R)
    crng = np.array([lp_range(A, b, bnds, 3 + d) for d in range(3)])
    return t_lo, crng


def fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


# --------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--pair", default="162,15")
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--delta0-deg", type=float, default=5.0)
    ap.add_argument("--h-min-deg", type=float, default=0.6)
    ap.add_argument("--directions", type=int, default=300)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "pnp_set", "linearised_eps%g_img%d" % (a.eps, a.third))
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    rep = {"eps": a.eps, "delta0_deg": a.delta0_deg}

    # ---- setup (identical to pnp_set.py) -------------------------------------------
    cams, imgs, pts = read_model(MODEL)
    i1, i2 = [int(x) for x in a.pair.split(",")]
    i3 = a.third
    intr = {i: cams[imgs[i].camera_id].unpack() for i in (i1, i2, i3)}
    f = {i: (intr[i][0], intr[i][1]) for i in intr}
    R1, c1 = imgs[i1].R, imgs[i1].center
    c2g = R1 @ (imgs[i2].center - c1)
    scale = c2g[0]
    pose = {i1: (np.eye(3), np.zeros(3)), i2: (imgs[i2].R @ R1.T, c2g / scale)}
    R3_colmap = imgs[i3].R @ R1.T
    c3_colmap = R1 @ (imgs[i3].center - c1) / scale
    Xg = {k: R1 @ (p.xyz - c1) / scale for k, p in pts.items()}
    det = {}
    for i in (i1, i2, i3):
        im = imgs[i]
        d = {}
        for j in np.flatnonzero(im.p3d_ids >= 0):
            d.setdefault(int(im.p3d_ids[j]), []).append(undistort(*im.xys[j], intr[i]))
        det[i] = d
    common = sorted(set(det[i1]) & set(det[i2]))
    zmin = 1e-3 * float(np.median([pose[i][0][2] @ (Xg[k] - pose[i][1]) for k in common for i in (i1, i2)]))
    Fk, Fbox, X0 = {}, {}, {}
    for k in common:
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i][k]:
                g_, h_ = fixed_rows(*pose[i], xn, f[i], a.eps, zmin)
                G.append(g_)
                H.append(h_)
        G, H = np.vstack(G), np.concatenate(H)
        nrm = np.linalg.norm(G, axis=1)
        res = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[G, nrm], b_ub=H,
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
        if not np.all(np.isfinite(bb)):
            continue
        Fk[k], Fbox[k], X0[k] = (G, H), (bb[:, 0], bb[:, 1]), res.x[:3]
    L3 = sorted(k for k in Fk if k in det[i3])
    obs3 = [(k, box_normals(xn, f[i3], a.eps)[0]) for k in L3 for xn in det[i3][k]]
    data = Data(L3, Fk, Fbox, X0, obs3, zmin)
    print("setup: %d bounded landmark sets seen by %s, %d camera-3 observations"
          % (len(L3), imgs[i3].name, len(obs3)), flush=True)

    # reference rotation and sound starting centre box (from pnp_set.py)
    prev = os.path.join(MODEL, "pnp_set", "eps%g_img%d_d05_h0.1" % (a.eps, i3))
    sp = json.load(open(os.path.join(prev, "sets.json")))
    sm = json.load(open(os.path.join(prev, "summary.json")))
    Rhat = np.array(sp["R_hat3"])
    cbox0 = (np.array(sm["step5"]["centre_lo"]), np.array(sm["step5"]["centre_hi"]))
    rep["reference"] = {"R_hat3_from": "pnp_set.py Step 1 (P3P+RANSAC on set centres)",
                        "start_centre_box": [cbox0[0].tolist(), cbox0[1].tolist()]}

    # ---- (1) plain linearised LP ----------------------------------------------------
    t = time.time()
    d0 = math.radians(a.delta0_deg)
    c0 = 0.5 * (cbox0[0] + cbox0[1])
    A, b, bnds = data.rows_linear(Rhat, c0, cbox0, 0.0, d0)
    ok = lp_feasible(A, b, bnds)
    plain = None
    if ok:
        w_r = np.array([lp_range(A, b, bnds, d) for d in range(3)])
        c_r = np.array([lp_range(A, b, bnds, 3 + d) for d in range(3)])
        plain = {"w_lo_deg": np.degrees(w_r[:, 0]).tolist(), "w_hi_deg": np.degrees(w_r[:, 1]).tolist(),
                 "centre_lo": c_r[:, 0].tolist(), "centre_hi": c_r[:, 1].tolist(),
                 "centre_size": (c_r[:, 1] - c_r[:, 0]).tolist(), "seconds": round(time.time() - t, 1)}
        print("(1) plain linearised LP (%.1fs): w in [%s] .. [%s] deg; centre size %s"
              % (time.time() - t, np.round(np.degrees(w_r[:, 0]), 3), np.round(np.degrees(w_r[:, 1]), 3),
                 np.round(c_r[:, 1] - c_r[:, 0], 4)), flush=True)
    else:
        print("(1) plain linearised LP infeasible", flush=True)
    rep["plain"] = plain

    # ---- (2) sound linearised branch and bound --------------------------------------
    t = time.time()
    level, cbox, survivors, tested = [(np.zeros(3), d0)], cbox0, [], 0
    h_min = math.radians(a.h_min_deg)
    rep["sound_levels"] = []
    while level:
        with Pool(a.procs, initializer=_init, initargs=(data, Rhat, cbox)) as pool:
            res = pool.map(_cell, level, chunksize=1)
        tested += len(level)
        keep = [(cell, r) for cell, r in zip(level, res) if r is not None]
        h = level[0][1]
        if keep:
            cbox = (np.min([r[1][0] for _, r in keep], 0), np.max([r[1][1] for _, r in keep], 0))
            # approximate w-box relative to R_hat (w0 + w'); the sound angle bound is delta_3 below
            wl = np.min([cell[0] + r[0][0] for cell, r in keep], 0)
            wh = np.max([cell[0] + r[0][1] for cell, r in keep], 0)
        print("(2) h = %.4f deg (slack at delta: %.2f px second-order): %d tested, %d kept; "
              "w box %s .. %s deg; centre size %s"
              % (math.degrees(h), f[i3][0] * g_rem(math.sqrt(3) * h), len(level), len(keep),
                 np.round(np.degrees(wl), 3) if keep else "-", np.round(np.degrees(wh), 3) if keep else "-",
                 np.round(cbox[1] - cbox[0], 4) if keep else "-"), flush=True)
        rep["sound_levels"].append({"h_deg": math.degrees(h), "tested": len(level), "kept": len(keep),
                                    "w_lo_deg": np.degrees(wl).tolist() if keep else None,
                                    "w_hi_deg": np.degrees(wh).tolist() if keep else None,
                                    "centre_lo": cbox[0].tolist(), "centre_hi": cbox[1].tolist()})
        if not keep:
            survivors = []
            break
        if h / 2 < h_min:
            survivors = keep
            break
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        level = [(w0 + (h / 2) * s, h / 2) for (w0, _), _ in keep for s in signs]
    sound = None
    if survivors:
        # angle(R, R_hat) <= angle(R, R0) + angle(R0, R_hat) <= max||w'|| + ||w0||
        delta3 = max(np.linalg.norm(cell[0]) +
                     np.linalg.norm(np.maximum(np.abs(r[0][0]), np.abs(r[0][1])))
                     for cell, r in survivors)
        sound = {"cells_tested": tested, "survivors": len(survivors), "delta3_deg": math.degrees(delta3),
                 "w_lo_deg": np.degrees(wl).tolist(), "w_hi_deg": np.degrees(wh).tolist(),
                 "centre_lo": cbox[0].tolist(), "centre_hi": cbox[1].tolist(),
                 "centre_size": (cbox[1] - cbox[0]).tolist(), "seconds": round(time.time() - t)}
        print("(2) sound: %d survivors of %d cells, delta_3 = %.3f deg, centre size %s (%ds)"
              % (len(survivors), tested, math.degrees(delta3), np.round(cbox[1] - cbox[0], 4), time.time() - t),
              flush=True)
    rep["sound"] = sound

    # ---- (3) inner radial search -------------------------------------------------------
    t = time.time()
    starts = [("R_hat3", Rhat), ("COLMAP", R3_colmap)]
    if plain is not None:
        wm = np.radians(0.5 * (np.array(plain["w_lo_deg"]) + np.array(plain["w_hi_deg"])))
        starts.append(("plain-LP centre", expR(wm) @ Rhat))
    R_start, start_name = None, None
    for nm, R in starts:
        if lp_feasible(*data.rows_exact(R)):
            R_start, start_name = R, nm
            break
    inner = None
    if R_start is None:
        print("(3) no feasible start rotation among %s" % [s[0] for s in starts], flush=True)
    else:
        U = fibonacci(a.directions)
        with Pool(a.procs, initializer=_init, initargs=(data, R_start, cbox0)) as pool:
            res = pool.starmap(_ray, [(u, math.radians(10.0), 12) for u in U])
        T = np.array([r[0] for r in res])
        Wb = T[:, None] * U
        w_off = Rotation.from_matrix(R_start @ Rhat.T).as_rotvec()
        Wb_hat = np.array([Rotation.from_matrix(expR(w) @ R_start @ Rhat.T).as_rotvec() for w in Wb])
        C = np.array([r[1] for r in res])
        clo, chi = C[:, :, 0].min(0), C[:, :, 1].max(0)
        inner = {"start": start_name, "start_offset_from_R_hat_deg": math.degrees(np.linalg.norm(w_off)),
                 "directions": a.directions, "radius_deg_min_median_max":
                     np.degrees([T.min(), np.median(T), T.max()]).tolist(),
                 "w_lo_deg_rel_R_hat": np.degrees(Wb_hat.min(0)).tolist(),
                 "w_hi_deg_rel_R_hat": np.degrees(Wb_hat.max(0)).tolist(),
                 "max_angle_from_R_hat_deg": math.degrees(np.linalg.norm(Wb_hat, axis=1).max()),
                 "centre_lo": clo.tolist(), "centre_hi": chi.tolist(), "centre_size": (chi - clo).tolist(),
                 "seconds": round(time.time() - t)}
        print("(3) inner (start %s): radius min/median/max %s deg; w %s .. %s deg (rel. R_hat); "
              "centre size %s (%ds)" % (start_name, np.round(np.degrees([T.min(), np.median(T), T.max()]), 3),
                                        np.round(np.degrees(Wb_hat.min(0)), 3), np.round(np.degrees(Wb_hat.max(0)), 3),
                                        np.round(chi - clo, 4), time.time() - t), flush=True)
    rep["inner"] = inner

    # ---- GT comparison ---------------------------------------------------------------------
    gt = CC.load_gt(SCENE)
    g1 = gt[imgs[i1].name]
    cgt = lambda i: g1["R"] @ (gt[imgs[i].name]["c"] - g1["c"])
    sc = cgt(i2)[0]
    R3_gt, c3_gt = gt[imgs[i3].name]["R"] @ g1["R"].T, cgt(i3) / sc
    w_gt = Rotation.from_matrix(R3_gt @ Rhat.T).as_rotvec()
    w_col = Rotation.from_matrix(R3_colmap @ Rhat.T).as_rotvec()
    inbox = lambda x, lo, hi: bool(np.all((np.asarray(x) >= np.asarray(lo) - 1e-9) & (np.asarray(x) <= np.asarray(hi) + 1e-9)))
    ev = {"gt_w_deg": np.degrees(w_gt).tolist(), "colmap_w_deg": np.degrees(w_col).tolist(),
          "gt_centre": c3_gt.tolist(), "colmap_centre": c3_colmap.tolist(),
          "gt_rotation_exactly_feasible": lp_feasible(*data.rows_exact(R3_gt))}
    for nm, s in (("plain", plain), ("sound", sound)):
        if s:
            ev[nm + "_contains_gt_rotation_box"] = inbox(np.degrees(w_gt), s["w_lo_deg"], s["w_hi_deg"])
            ev[nm + "_contains_gt_centre"] = inbox(c3_gt, s["centre_lo"], s["centre_hi"])
    if sound:
        ev["sound_gt_in_a_surviving_cell"] = any(
            inbox(Rotation.from_matrix(R3_gt @ (expR(cell[0]) @ Rhat).T).as_rotvec(), r[0][0], r[0][1])
            and inbox(c3_gt, r[1][0], r[1][1]) for cell, r in survivors)
        ev["sound_gt_rotation_within_delta3"] = bool(np.linalg.norm(w_gt) <= sound["delta3_deg"] * math.pi / 180)
    rep["gt"] = ev
    rep["seconds"] = round(time.time() - T0)
    json.dump(rep, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(ev, indent=1))
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
