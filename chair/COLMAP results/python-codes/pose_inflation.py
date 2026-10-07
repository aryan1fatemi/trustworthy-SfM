"""Section 3.3 of the draft: widen each tolerance by the pixel effect of pose error.

    python pose_inflation.py --scene airplane

    Y_ik = X_ik (+) (-H_ik Theta_i),       H_ik = d pi / d theta  at (theta_hat_i, X_hat_k)

Pose perturbation in the tangent space at theta_hat_i, centre form (as the draft):
    R = exp([w]x) R_hat,   c = c_hat + dc,   dtheta = (w, dc) in R^6.

Theta_i^lin = { dtheta : |H_ik' dtheta| <= b  for every landmark k' seen by image i }
(the draft's centred form, no exclusion of k), and its per-axis extremes from 12
LPs give the box that is propagated.

Test design -- separate the two error sources:
    X_ik  = square of half-width eps_det, conformal on s_det   (GT poses: detection only)
    Y_ik  = X_ik inflated by the pose box                      (detection + pose model)
    C_ik  = square of half-width eps_tot, conformal on s_tot   (detection + true pose error)
If the inflation is right, Y_ik should behave like C_ik.

Also checked directly: does Theta_i contain camera i's TRUE pose error?
"""
import os
import sys
import csv
import json
import math
import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project
from cp_step0_align import umeyama
import cp_conformal as CC
import exact_polytope as ex

ALPHAS = [0.05, 0.10]
H_STEP = 1e-6


def perturbed(R, c, d):
    Rp = Rotation.from_rotvec(d[:3]).as_matrix() @ R
    cp = c + d[3:]
    return Rp, -Rp @ cp


def jacobians(R, c, intr, X):
    """(N,2,6) d pi / d theta at theta_hat for points X (N,3)."""
    J = np.zeros((len(X), 2, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = H_STEP
        up = project(*perturbed(R, c, d), intr, X)[0]
        um = project(*perturbed(R, c, -d), intr, X)[0]
        J[:, :, j] = (up - um) / (2 * H_STEP)
    return J


def pose_box(H, b):
    """Per-axis extremes of {d : |H d| <= b} (H stacked (2N,6)).  12 LPs."""
    A = np.vstack([H, -H])
    ub = np.full(len(A), b)
    lo, hi = np.zeros(6), np.zeros(6)
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        r1 = linprog(e, A_ub=A, b_ub=ub, bounds=[(None, None)] * 6, method="highs")
        r2 = linprog(-e, A_ub=A, b_ub=ub, bounds=[(None, None)] * 6, method="highs")
        lo[j] = r1.x[j] if r1.success else -np.inf
        hi[j] = r2.x[j] if r2.success else np.inf
    return lo, hi


def interval_image(Hk, lo, hi):
    """Pixel interval of -Hk d over the box [lo,hi]  ->  (2,) lo, (2,) hi."""
    a = -Hk * lo[None, :]
    b = -Hk * hi[None, :]
    return np.minimum(a, b).sum(1), np.maximum(a, b).sum(1)


def halfspaces_rect(views, lo_list, hi_list, z_min=1e-3):
    """F_k with axis-aligned, possibly off-centre pixel tolerances
       lo <= pi(X) - y <= hi   (same linearisation as exact_polytope)."""
    H = []
    for (R, t, intr, y), lo, hi in zip(views, lo_list, hi_list):
        fx, fy = intr[0], intr[1]
        n_star, gain = ex.undistort(np.asarray(y, float), intr)
        for a, f in ((0, fx), (1, fy)):
            k = f * gain                       # pixel residual ~ k (n_a - n*_a)
            for lim, sg in ((n_star[a] + hi[a] / k, 1.0), (n_star[a] + lo[a] / k, -1.0)):
                row = sg * (R[a] - lim * R[2])
                off = sg * (t[a] - lim * t[2])
                H.append(np.append(row, off))
        H.append(np.append(-R[2], -t[2] + z_min))
    return np.asarray(H)


def diameter(H):
    c, r = ex.chebyshev_center(H)
    if c is None:
        f = linprog(np.zeros(3), A_ub=H[:, :3], b_ub=-H[:, 3],
                    bounds=[(None, None)] * 3, method="highs")
        return float("inf") if f.status == 0 else float("nan")
    if r <= 0:
        return float("nan")
    V, _, _ = ex.vertices_and_volume(H, c)
    return float(np.linalg.norm(V[:, None] - V[None], axis=-1).max())


def conformal(sorted_cal, alpha):
    N = len(sorted_cal)
    i = math.ceil((N + 1) * (1 - alpha))
    return float(sorted_cal[i - 1]) if i <= N else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--exact-images", type=int, default=12,
                    help="images for the exact (non-box) inflation check")
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "conformal", "pose_inflation")
    os.makedirs(OUT, exist_ok=True)

    cams, imgs, pts = read_model(MODEL)
    gt = CC.load_gt(SCENE)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))
    print("GT references + scores ...", flush=True)
    rows = CC.observations(cams, imgs, pts, gt, s_, Ra, ta)
    lm = CC.references_and_scores(rows, cams, imgs, pts, gt)

    # ---- split (seed 0, by landmark, same as cp_conformal) ----------------
    pids_all = np.array(sorted(pts))
    perm = np.random.default_rng(0).permutation(pids_all)
    cal = set(perm[: len(perm) // 2].tolist())
    scor = [r for r in rows if np.isfinite(r["s_tot"])]
    eps = {}
    for sc in ("det", "tot"):
        srt = np.sort([r["s_" + sc] for r in scor if r["pid"] in cal])
        for al in ALPHAS:
            eps[(sc, al)] = conformal(srt, al)
    print("  eps_det: %s   eps_tot: %s" % ({al: round(eps[("det", al)], 3) for al in ALPHAS},
                                           {al: round(eps[("tot", al)], 3) for al in ALPHAS}))

    # ---- true pose error of each camera, in the reconstruction frame ------
    true_err = {}
    for i in ids:
        g = gt[imgs[i].name]
        R_true = g["R"] @ Ra.T                              # world(recon) -> cam
        c_true = s_ * Ra @ g["c"] + ta
        w = Rotation.from_matrix(R_true @ imgs[i].R.T).as_rotvec()
        true_err[i] = np.concatenate([w, c_true - imgs[i].center])

    # ---- Theta_i per image, per alpha --------------------------------------
    print("Jacobians + pose boxes ...", flush=True)
    Hs, box = {}, {}
    img_rows = []
    for i, im in imgs.items():
        R, t, intr = pose[i]
        m = im.p3d_ids >= 0
        X = np.array([pts[int(p)].xyz for p in im.p3d_ids[m]])
        J = jacobians(R, im.center, intr, X)
        Hs[i] = dict(zip(np.flatnonzero(m).tolist(), J))
        Hstack = J.reshape(-1, 6)
        d_true = true_err[i]
        for al in ALPHAS:
            b = eps[("det", al)]
            lo, hi = pose_box(Hstack, b)
            box[(i, al)] = (lo, hi)
            in_poly = bool((np.abs(Hstack @ d_true) <= b + 1e-9).all())
            in_box = bool(((d_true >= lo) & (d_true <= hi)).all())
            img_rows.append([i, im.name, al, int(m.sum()), in_poly, in_box]
                            + list(0.5 * (hi - lo)) + list(d_true))
    with open(os.path.join(OUT, "pose_boxes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "name", "alpha", "n_landmarks", "true_err_in_polytope",
                    "true_err_in_box", "hw_wx", "hw_wy", "hw_wz", "hw_cx", "hw_cy", "hw_cz",
                    "true_wx", "true_wy", "true_wz", "true_cx", "true_cy", "true_cz"])
        w.writerows(img_rows)

    # ---- observation-level: regions, coverage, width ----------------------
    print("evaluating regions ...", flush=True)
    res = {}
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["pid"], []).append(r)
    test_lm = [p for p in pids_all if p not in cal and lm[p]["Xstar"] is not None]
    stats = {}
    for al in ALPHAS:
        ed, et = eps[("det", al)], eps[("tot", al)]
        cover = {"det": [], "inflated": [], "tot": []}
        width = {"det": [], "inflated": [], "tot": []}
        lm_ok = {"det": [], "inflated": [], "tot": []}
        infl_lo, infl_hi = {}, {}
        for p in test_lm:
            X = lm[p]["Xstar"]
            ok = {"det": True, "inflated": True, "tot": True}
            for r in by_pid[p]:
                i, j = r["iid"], r["p2d"]
                R, t, intr = pose[i]
                res_px = project(R, t, intr, X)[0][0] - imgs[i].xys[j]   # pi(theta_hat,X*) - y
                plo, phi = interval_image(Hs[i][j], *box[(i, al)])
                infl_lo[(p, i, j)], infl_hi[(p, i, j)] = plo, phi
                c_det = bool((np.abs(res_px) <= ed).all())
                c_inf = bool(((res_px >= -ed + plo) & (res_px <= ed + phi)).all())
                c_tot = bool((np.abs(res_px) <= et).all())
                for key, c in (("det", c_det), ("inflated", c_inf), ("tot", c_tot)):
                    cover[key].append(c)
                    ok[key] &= c
                width["det"].append(ed)
                width["tot"].append(et)
                width["inflated"].append(ed + 0.5 * float(np.mean(phi - plo)))
            for key in ok:
                lm_ok[key].append(ok[key])
        stats[al] = dict(
            eps_det=ed, eps_tot=et,
            obs_coverage={k: float(np.mean(v)) for k, v in cover.items()},
            landmark_coverage={k: float(np.mean(v)) for k, v in lm_ok.items()},
            median_half_width_px={k: float(np.median(v)) for k, v in width.items()},
            p90_half_width_px={k: float(np.percentile(v, 90)) for k, v in width.items()},
            images_true_err_in_polytope=float(np.mean([r[4] for r in img_rows if r[2] == al])),
            images_true_err_in_box=float(np.mean([r[5] for r in img_rows if r[2] == al])))
        res[al] = (infl_lo, infl_hi)
        print("  alpha %.2f  half-width det %.2f  inflated %.2f  tot %.2f   "
              "landmark cov det %.3f  inflated %.3f  tot %.3f"
              % (al, stats[al]["median_half_width_px"]["det"],
                 stats[al]["median_half_width_px"]["inflated"],
                 stats[al]["median_half_width_px"]["tot"],
                 stats[al]["landmark_coverage"]["det"],
                 stats[al]["landmark_coverage"]["inflated"],
                 stats[al]["landmark_coverage"]["tot"]), flush=True)

    # ---- set size of F_k for each region (alpha = 0.10) -------------------
    print("set sizes ...", flush=True)
    al = 0.10
    infl_lo, infl_hi = res[al]
    ed, et = eps[("det", al)], eps[("tot", al)]
    dia = {"det": [], "inflated": [], "tot": []}
    for p in test_lm:
        P = pts[p]
        views = [(*pose[i], imgs[i].xys[j]) for i, j in zip(P.image_ids, P.point2D_idxs)]
        keys = [(p, int(i), int(j)) for i, j in zip(P.image_ids, P.point2D_idxs)]
        for kind in dia:
            if kind == "det":
                lo_l = [np.full(2, -ed)] * len(views); hi_l = [np.full(2, ed)] * len(views)
            elif kind == "tot":
                lo_l = [np.full(2, -et)] * len(views); hi_l = [np.full(2, et)] * len(views)
            else:
                lo_l = [-ed + infl_lo[k] for k in keys]
                hi_l = [ed + infl_hi[k] for k in keys]
            d = diameter(halfspaces_rect(views, lo_l, hi_l))
            dia[kind].append(d / lm[p]["d_k"] if np.isfinite(d) else d)
    for kind, v in dia.items():
        v = np.array(v)
        f = v[np.isfinite(v)]
        stats[al].setdefault("set_size", {})[kind] = {
            "median_diam_over_dk": float(np.median(f)),
            "unbounded_rate": float(np.isinf(v).mean()),
            "empty_rate": float(np.isnan(v).mean())}

    # ---- how much does the box over-approximate H*Theta? ------------------
    print("box vs exact inflation ...", flush=True)
    ratio = []
    for i in list(imgs)[: a.exact_images]:
        Hstack = np.array(list(Hs[i].values())).reshape(-1, 6)
        A = np.vstack([Hstack, -Hstack])
        ub = np.full(len(A), eps[("det", 0.10)])
        lo, hi = box[(i, 0.10)]
        for j, Hk in list(Hs[i].items())[::10]:
            for ax_ in (0, 1):
                ex_hi = -linprog(Hk[ax_], A_ub=A, b_ub=ub, bounds=[(None, None)] * 6,
                                 method="highs").fun                  # max of -H d
                plo, phi = interval_image(Hk, lo, hi)
                if ex_hi > 1e-9:
                    ratio.append(phi[ax_] / ex_hi)
    stats["box_overapproximation"] = {
        "samples": len(ratio), "median": float(np.median(ratio)),
        "p10": float(np.percentile(ratio, 10)), "p90": float(np.percentile(ratio, 90))}

    json.dump({str(k): v for k, v in stats.items()},
              open(os.path.join(OUT, "pose_inflation.json"), "w"), indent=2)
    print(json.dumps({str(k): v for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
