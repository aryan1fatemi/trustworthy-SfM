"""Pose box around COLMAP's pose and the points that fit EVERY pose in it (Jacobian linearisation).

    python linearized_forall.py [--eps 3]

Pose box: theta = theta_hat + delta, |delta_j| <= d_j, delta = (w_x, w_y, w_z, c_x, c_y, c_z)
with R = exp([w]x) R_hat, C = C_hat + c. Linearising the projection in the pose (Jacobian H_k, 2x6,
at COLMAP's pose and point, as in Eq. 15 of the RAIM paper):
    pi(theta, X) ~ pi(theta_hat, X) + H_k delta
"Within eps for every pose in the box" becomes "within the SHRUNKEN window at COLMAP's pose":
    |pi(theta_hat, X) - y_k| <= eps - s_k,     s_k = sum_j |H_k[., j]| d_j     (per pixel axis)
so the point set is a polytope,
    S_k(d) = F_k  intersected with  the shrunken window back-projected at COLMAP's pose.
The pair (box, {S_k}) is a "for every pose, for every point" pair (up to the linearisation).
Growth:
    together: d_j = t * w_j, w_j = 1 / median_k |H_k[., j]|_inf  (each parameter moves the image by t px)
    one parameter at a time: largest d_j (others 0) with every S_k non-empty
Checks: GT pose inside the box, GT point inside S_k; exact reprojection of the vertices of S_k
through the 64 corner poses of the box (linearisation error).
Writes results/linearized_forall.json, .csv, .png.
"""
import os
import csv
import json
import math
import argparse
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setlib import Scene, chebyshev, expR, results_dir, SCENE

NAMES = ["w_x", "w_y", "w_z", "c_x", "c_y", "c_z"]


def window_rows_uv(R, C, xn, f, eu, ev, zmin):
    """rows G X <= h for a window of half-widths (eu, ev) px around normalised point xn."""
    a_p, a_m = xn[0] + eu / f[0], xn[0] - eu / f[0]
    b_p, b_m = xn[1] + ev / f[1], xn[1] - ev / f[1]
    N = np.array([[1, 0, -a_p], [-1, 0, a_m], [0, 1, -b_p], [0, -1, b_m]], float)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    G = [n @ R for n in N] + [-R[2]]
    H = [g @ C for g in G[:4]] + [-zmin - R[2] @ C]
    return np.array(G), np.array(H)


def polytope(A, b):
    c, r = chebyshev(A, b)
    if c is None or r < 1e-10:
        return None
    try:
        return HalfspaceIntersection(np.c_[A, -b], c).intersections
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"])
    a = ap.parse_args()
    RES = results_dir(a.scheme)
    P = np.load(os.path.join(RES, "pair_sets.npz"))
    sc = Scene(pair=(162, 15), extra=(a.third,))
    i3, f3, zmin = a.third, sc.f[a.third], float(P["z_min"])
    L = [int(k) for k in P["landmarks"] if int(k) in sc.det[i3]]
    Rh, Ch = sc.pose(i3)
    F = {k: (P["G"][P["row_landmark"] == k], P["h"][P["row_landmark"] == k]) for k in L}
    epsk = {k: sc.eps_list(i3, k, a.eps, a.scheme) for k in L}

    # Jacobian of the pinhole projection (undistorted pixels) at COLMAP's pose and point
    def proj(th, X):
        R = Rotation.from_rotvec(th[:3]).as_matrix() @ Rh
        q = R @ (X - (Ch + th[3:]))
        return np.array([f3[0] * q[0] / q[2], f3[1] * q[1] / q[2]])

    H = {}
    for k in L:
        Xk = sc.point(k)
        J = np.zeros((2, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1e-6
            J[:, j] = (proj(e, Xk) - proj(-e, Xk)) / 2e-6
        H[k] = J
    Hall = np.array([H[k] for k in L])
    w = 1.0 / np.median(np.abs(Hall).max(1), 0)              # parameter step that moves the image ~1 px
    conv = np.array([180 / math.pi] * 3 + [1.0] * 3)

    def sets_for(d):
        """S_k(d) as (A, b) per landmark, or None if the shrunken window closes."""
        out = {}
        for k in L:
            s = np.abs(H[k]) @ d                              # (s_u, s_v)
            A, b = [F[k][0]], [F[k][1]]
            ok = True
            for xn, e in zip(sc.det[i3][k], epsk[k]):
                eu, ev = e - s[0], e - s[1]
                if eu <= 0 or ev <= 0:
                    ok = False
                    break
                g, h = window_rows_uv(Rh, Ch, xn, f3, eu, ev, zmin)
                A.append(g)
                b.append(h)
            out[k] = (np.vstack(A), np.concatenate(b)) if ok else None
        return out

    volF = {}
    for k in L:
        V = polytope(*F[k])
        volF[k] = ConvexHull(V).volume if V is not None else np.nan

    def evaluate(d):
        S = sets_for(d)
        vols, nonempty = [], 0
        for k in L:
            V = polytope(*S[k]) if S[k] is not None else None
            if V is not None:
                nonempty += 1
                vols.append(ConvexHull(V).volume / volF[k])
            else:
                vols.append(0.0)
        return S, nonempty, np.array(vols)

    # GT
    R_gt, C_gt = sc.gt_pose(i3)
    d_gt = np.r_[Rotation.from_matrix(R_gt @ Rh.T).as_rotvec(), C_gt - Ch]
    from colmap_io import read_model  # noqa: F401  (Scene already loaded the model)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(SCENE), "chair", "COLMAP results", "python-codes"))
    try:
        import cp_conformal as CC
        from cp_step0_align import umeyama
        gt = CC.load_gt(SCENE)
        ids = [i for i in sc.imgs if sc.imgs[i].name in gt]
        s_, Ra, ta = umeyama(np.array([gt[sc.imgs[i].name]["c"] for i in ids]),
                             np.array([sc.imgs[i].center for i in ids]))
        lifted = {}
        for rw in CC.observations(sc.cams, sc.imgs, sc.pts, gt, s_, Ra, ta):
            if rw["valid"]:
                lifted.setdefault(rw["pid"], []).append(rw["Xgt"])
        g1 = gt[sc.imgs[162].name]
        scl = (g1["R"] @ (gt[sc.imgs[15].name]["c"] - g1["c"]))[0]
        Xgt = {k: g1["R"] @ (np.median(v, 0) - g1["c"]) / scl for k, v in lifted.items()}
    except Exception as ex:
        print("GT points unavailable:", ex)
        Xgt = {}

    # ---- growth together --------------------------------------------------------------------
    ts = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    rows = []
    print("image 3 = %s, %d landmarks, scheme %s; unit step w = %s (deg / baseline units)"
          % (sc.imgs[i3].name, len(L), a.scheme, np.round(w * conv, 4)))
    for t in ts:
        d = t * w
        S, ne, vol = evaluate(d)
        gt_pose_in = bool(np.all(np.abs(d_gt) <= d + 1e-12))
        gt_pts = [bool(np.all(S[k][0] @ Xgt[k] <= S[k][1] + 1e-9)) for k in L if S[k] is not None and k in Xgt]
        row = {"t_px": t, **{"d_" + n: float(v) for n, v in zip(NAMES, d * conv)},
               "landmarks_nonempty": ne, "volume_ratio_mean": float(vol.mean()), "volume_ratio_median": float(np.median(vol)),
               "gt_pose_in_box": gt_pose_in, "gt_point_in_S_fraction": float(np.mean(gt_pts)) if gt_pts else float("nan")}
        rows.append(row)
        print("  t = %4.2f px: box +-%s deg, +-%s; non-empty %3d / %d; vol(S)/vol(F) mean %.3f; GT pose in box %s; "
              "GT point in S %.2f" % (t, np.round(d[:3] * conv[:3], 3), np.round(d[3:], 4), ne, len(L),
                                      row["volume_ratio_mean"], gt_pose_in, row["gt_point_in_S_fraction"]))
    # largest t with every landmark non-empty (bisection)
    lo, hi = 0.0, 3.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if evaluate(mid * w)[1] == len(L) else (lo, mid)
    t_all = lo
    d_all = t_all * w
    print("largest box with all %d landmarks non-empty: t = %.3f px -> +-%s deg, +-%s"
          % (len(L), t_all, np.round(d_all[:3] * conv[:3], 4), np.round(d_all[3:], 5)))

    # ---- one parameter at a time ----------------------------------------------------------------
    single = {}
    for j in range(6):
        lo, hi = 0.0, 10.0 * w[j]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            d = np.zeros(6)
            d[j] = mid
            lo, hi = (mid, hi) if evaluate(d)[1] == len(L) else (lo, mid)
        single[NAMES[j]] = float(lo * conv[j])
    print("one parameter at a time (all landmarks non-empty): %s" % {k: round(v, 4) for k, v in single.items()})

    # ---- exact check of the linearisation at the largest all-landmark box -----------------------
    S, _, _ = evaluate(d_all)
    worst = 0.0
    corners = [np.array(c) * d_all for c in itertools.product((-1, 1), repeat=6)]
    for k in L[::4]:
        V = polytope(*S[k])
        if V is None:
            continue
        for dc in corners:
            R = Rotation.from_rotvec(dc[:3]).as_matrix() @ Rh
            C = Ch + dc[3:]
            for xn, e in zip(sc.det[i3][k], epsk[k]):
                q = (V - C) @ R.T
                err = np.abs(np.c_[f3[0] * (q[:, 0] / q[:, 2] - xn[0]), f3[1] * (q[:, 1] / q[:, 2] - xn[1])]).max()
                worst = max(worst, err - e)
    print("exact check at the largest box: worst reprojection beyond the window over 64 corner poses x "
          "vertices of S_k: %+.3f px" % worst)

    rep = {"eps_px": a.eps, "scheme": a.scheme, "image3": sc.imgs[i3].name, "landmarks": len(L),
           "unit_step": {n: float(v) for n, v in zip(NAMES, w * conv)}, "growth_together": rows,
           "largest_box_all_landmarks": {"t_px": t_all, **{n: float(v) for n, v in zip(NAMES, d_all * conv)}},
           "one_parameter_at_a_time": single, "gt_offset": {n: float(v) for n, v in zip(NAMES, d_gt * conv)},
           "linearisation_check_px_beyond_window": worst,
           "units": "rotation deg relative to COLMAP's, centre in baseline units"}
    json.dump(rep, open(os.path.join(RES, "linearized_forall.json"), "w"), indent=2)
    with open(os.path.join(RES, "linearized_forall.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    tt = [r["t_px"] for r in rows]
    ax[0].plot(tt, [r["volume_ratio_mean"] for r in rows], "-o", color="#2a78d6", lw=2, ms=5, label="mean")
    ax[0].plot(tt, [r["volume_ratio_median"] for r in rows], "-s", color="#eb6834", lw=2, ms=4, label="median")
    ax[0].set_xlabel("pose box size t (image motion per parameter, px)")
    ax[0].set_ylabel("vol(S_k) / vol(F_k)")
    ax[0].legend(frameon=False)
    ax[1].plot(tt, [r["landmarks_nonempty"] for r in rows], "-o", color="#1baf7a", lw=2, ms=5)
    ax[1].axvline(t_all, color="#52514e", ls="--", lw=1)
    ax[1].text(t_all, len(L) * 0.5, "  all non-empty up to t = %.2f" % t_all, fontsize=9, color="#52514e")
    ax[1].set_xlabel("pose box size t (px)")
    ax[1].set_ylabel("landmarks with non-empty S_k (of %d)" % len(L))
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
        x.grid(color="#e5e4df", lw=0.8)
    fig.suptitle("Image 3: pose box around COLMAP's pose vs points that fit every pose in it "
                 "(linearised, eps = %g px, %s)" % (a.eps, a.scheme), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "linearized_forall.png"), dpi=120)
    print("->", RES)


if __name__ == "__main__":
    main()
