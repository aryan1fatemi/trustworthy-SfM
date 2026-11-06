"""Baseline 1 for image 3 with a per-observation tolerance eps_ik.

    python adaptive_eps_baseline1.py [--eps 3] [--margin 1] [--kappa 1.5] [--cap 3]

COLMAP's pose and COLMAP's points are held fixed (as in Baseline 1); the pose is perturbed,
R = exp([w]x) R_hat, C = C_hat + dc. Residual of observation k: r_k = pi(theta, X_hat_k) - y_k
(COLMAP camera model, pixels); r_hat_k = |r_k|_inf at COLMAP's pose.
Schemes for eps_k (all capped at cap * eps):
    fixed           eps
    additive        max(eps, r_hat_k + margin)
    multiplicative  max(eps, kappa * r_hat_k)
For each scheme:
    one parameter at a time, exact: largest step with |r_k(theta)|_inf <= eps_k for all k
    all six together, Jacobian LP:   {d : |r_k + H_k d| <= eps_k}, min / max of each component
GT (same gauge rule) only for evaluation:
    premise rate: fraction of observations whose GT detection error s_det (leave-one-image-out GT
    reference, from cp_scores.csv) is <= eps_k;
    whether the GT pose satisfies every window with COLMAP's points.
Writes results/adaptive_eps_baseline1.json.
"""
import os
import csv
import json
import math
import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from setlib import Scene, RESULTS, SCENE

NAMES = ["w_x", "w_y", "w_z", "c_x", "c_y", "c_z"]


def project(R, C, intr, X):
    fx, fy, cx, cy, k1, k2 = intr
    q = (X - C) @ R.T
    x, y = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.c_[fx * x * d + cx, fy * y * d + cy]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--cap", type=float, default=3.0, help="eps_k <= cap * eps")
    ap.add_argument("--third", type=int, default=17)
    a = ap.parse_args()
    sc = Scene(pair=(162, 15), extra=(a.third,))
    im = sc.imgs[a.third]
    intr = sc.intr[a.third]
    Rh, Ch = sc.pose(a.third)
    m = np.flatnonzero(im.p3d_ids >= 0)
    X = np.array([sc.point(int(k)) for k in im.p3d_ids[m]])
    Y = im.xys[m]

    def resid(th):
        R = Rotation.from_rotvec(th[:3]).as_matrix() @ Rh
        return project(R, Ch + th[3:], intr, X) - Y

    r0 = resid(np.zeros(6))
    rhat = np.abs(r0).max(1)
    H = np.zeros((len(m), 2, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = 1e-6
        H[:, :, j] = (resid(d) - resid(-d)) / 2e-6

    # GT detection error per observation (s_det) and GT pose in the gauge
    sdet = {}
    for row in csv.DictReader(open(os.path.join(SCENE, "COLMAP results", "conformal", "cp_scores.csv"))):
        if int(row["image_id"]) == a.third and row["valid"] == "1" and row["s_det"] not in ("", "nan"):
            sdet[int(row["point2D_idx"])] = float(row["s_det"])
    s_obs = np.array([sdet.get(int(j), np.nan) for j in m])
    R_gt, C_gt = sc.gt_pose(a.third)
    th_gt = np.r_[Rotation.from_matrix(R_gt @ Rh.T).as_rotvec(), C_gt - Ch]
    r_gt = np.abs(resid(th_gt)).max(1)

    cap = a.cap * a.eps
    schemes = {"fixed": np.full(len(m), a.eps),
               "additive (r_hat + %g)" % a.margin: np.minimum(np.maximum(a.eps, rhat + a.margin), cap),
               "multiplicative (%g r_hat)" % a.kappa: np.minimum(np.maximum(a.eps, a.kappa * rhat), cap)}
    conv = np.array([180 / math.pi] * 3 + [1.0] * 3)
    out = {}
    print("image 3 = %s: %d observations; COLMAP residual median %.2f, max %.2f px"
          % (im.name, len(m), np.median(rhat), rhat.max()))
    for name, E in schemes.items():
        okE = rhat <= E
        res = {"eps_median": float(np.median(E)), "eps_max": float(E.max()),
               "obs_over_eps_at_colmap": int((~okE).sum()),
               "premise_rate_gt": float(np.nanmean(s_obs <= E)),
               "gt_pose_fits_all_windows_with_colmap_points": bool(np.all(r_gt <= E))}
        # exact one-parameter limits (needs a feasible start: drop observations already outside)
        idx = okE
        lim = []
        for j in range(6):
            pair = []
            for s in (-1, 1):
                e = np.zeros(6)
                e[j] = s
                f = lambda t: np.all(np.abs(resid(t * e)[idx]).max(1) <= E[idx])
                lo, hi = 0.0, (math.radians(20) if j < 3 else 2.0)
                if f(hi):
                    lo = hi
                else:
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if f(mid) else (lo, mid)
                pair.append(s * lo * conv[j])
            lim.append(pair)
        res["one_param_exact"] = {n: p for n, p in zip(NAMES, lim)}
        # Jacobian LP, all six together (all observations)
        A = np.vstack([H.reshape(-1, 6), -H.reshape(-1, 6)])
        b = np.r_[np.repeat(E, 2) - r0.ravel(), np.repeat(E, 2) + r0.ravel()]
        feas = linprog(np.zeros(6), A_ub=A, b_ub=b, bounds=[(None, None)] * 6, method="highs")
        if feas.status != 0:
            res["combined_lp"] = "infeasible"
        else:
            box = {}
            for j in range(6):
                c = np.zeros(6)
                c[j] = 1
                lo = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * 6, method="highs").x[j]
                hi = linprog(-c, A_ub=A, b_ub=b, bounds=[(None, None)] * 6, method="highs").x[j]
                box[NAMES[j]] = [lo * conv[j], hi * conv[j]]
            res["combined_lp"] = box
            res["gt_inside_combined_box"] = bool(all(box[n][0] <= v * cv <= box[n][1]
                                                     for n, v, cv in zip(NAMES, th_gt, conv)))
        out[name] = res
        print("\n== %s: eps median %.2f max %.2f px; obs over eps at COLMAP %d; GT premise rate %.3f; "
              "GT pose fits every window: %s" % (name, res["eps_median"], res["eps_max"],
                                                 res["obs_over_eps_at_colmap"], res["premise_rate_gt"],
                                                 res["gt_pose_fits_all_windows_with_colmap_points"]))
        print("   %-4s %22s %24s %9s" % ("", "one parameter (exact)", "all six (Jacobian LP)", "GT"))
        for j, n in enumerate(NAMES):
            lp = res["combined_lp"]
            lp_s = ("%+8.3f .. %+8.3f" % tuple(lp[n])) if isinstance(lp, dict) else "%19s" % lp
            print("   %-4s %+8.3f .. %+8.3f   %s   %+8.3f" % (n, *res["one_param_exact"][n], lp_s, th_gt[j] * conv[j]))
    json.dump({"image3": im.name, "eps": a.eps, "margin": a.margin, "kappa": a.kappa, "cap": cap,
               "units": "rotation deg relative to COLMAP's, centre in baseline units", "schemes": out},
              open(os.path.join(RESULTS, "adaptive_eps_baseline1.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
