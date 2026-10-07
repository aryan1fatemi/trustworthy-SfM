"""White-box (GT-free) tolerance model vs. CP-with-GT.

    python whitebox.py --scene airplane

Error decomposition for observation (i,k), w.r.t. the ESTIMATED camera:

    y_ik - pi(theta_hat_i, X*_k)  =  eta_ik                (detection / matching)
                                   + H_ik dtheta*_i         (pose, first order)

GT-free estimators (reconstruction internals only):
  detection : standardized leave-one-IMAGE-out residuals
                  z_jk = (I - L_jk)^{-1/2} (y_jk - pi(theta_hat_j, X_hat_k)),
                  L_jk = J_jk A_k^{-1} J_jk^T,  A_k = sum_i J_ik^T J_ik
              -> eps_hat(q) = q-quantile of ||z||_inf,   sigma_hat = 1.4826 MAD(z)
  pose      : (a) bounded-error  Theta_i = {d : |H_ik' d| <= eps_hat}  with the EXACT
                  pixel extent of -H_ik Theta_i by LP  (fix 1: no interval box)
              (b) Gauss-Markov   Sigma_i = sigma_hat^2 (H_i^T H_i)^{-1},
                  bound +- z_q sqrt(h^T Sigma_i h) per pixel axis
  landmark  : per-view level q_k = (1 - alpha0)^{1/m_k}   (Sidak; independence assumed)

GT enters ONLY the evaluation: coverage of X*_k and set size, against the
landmark-level CP-with-GT benchmark from cp_normalized.py.
"""
import os
import sys
import csv
import json
import math
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.stats import norm, chi2
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project
from cp_step0_align import umeyama
import cp_conformal as CC
from pose_inflation import jacobians, pose_box, interval_image, halfspaces_rect, diameter

ALPHA0 = 0.10


# --------------------------------------------------------------------------
def exact_extents(args):
    """Worker: exact [min,max] of -H_k[a] . d over {d : |H d| <= b}, per target obs."""
    Hstack, b, targets = args
    A = np.vstack([Hstack, -Hstack])
    ub = np.full(len(A), b)
    out = []
    for Hk in targets:
        lo, hi = np.zeros(2), np.zeros(2)
        for a in range(2):
            r1 = linprog(Hk[a], A_ub=A, b_ub=ub, bounds=[(None, None)] * 6, method="highs")
            r2 = linprog(-Hk[a], A_ub=A, b_ub=ub, bounds=[(None, None)] * 6, method="highs")
            hi[a] = -r1.fun if r1.success else np.inf     # max of -H d
            lo[a] = r2.fun if r2.success else -np.inf     # min of -H d
        out.append((lo, hi))
    return out


def landmark_jacobians(views, X, h=1e-6):
    J = np.zeros((len(views), 2, 3))
    for a in range(3):
        d = np.zeros(3)
        d[a] = h
        for n, (R, t, intr, _) in enumerate(views):
            J[n, :, a] = (project(R, t, intr, X + d)[0][0] -
                          project(R, t, intr, X - d)[0][0]) / (2 * h)
    return J


def standardized_loio(p, imgs, pose):
    """z (m,2) for every observation of landmark p, or None if < 3 images."""
    views = [(*pose[i], imgs[i].xys[j]) for i, j in zip(p.image_ids, p.point2D_idxs)]
    img = np.array(p.image_ids)
    if len(set(img.tolist())) < 3:
        return None
    J = landmark_jacobians(views, p.xyz)
    e = np.array([v[3] - project(*v[:3], p.xyz)[0][0] for v in views])     # y - pi
    Ainv = np.linalg.inv(np.einsum("mai,maj->ij", J, J))
    z = np.zeros_like(e)
    for im in set(img.tolist()):
        S = np.flatnonzero(img == im)
        JS = J[S].reshape(-1, 3)
        L = JS @ Ainv @ JS.T
        w, V = np.linalg.eigh(np.eye(len(L)) - L)
        if w.min() < 1e-6:
            return None
        z[S] = (V @ np.diag(w ** -0.5) @ V.T @ e[S].reshape(-1)).reshape(-1, 2)
    return z


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "conformal", "whitebox")
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()

    cams, imgs, pts = read_model(MODEL)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}

    # ================= GT-FREE MODEL =======================================
    print("detection model: standardized leave-one-image-out residuals ...", flush=True)
    Z = []
    for p in pts.values():
        z = standardized_loio(p, imgs, pose)
        if z is not None:
            Z.append(z)
    Z = np.concatenate(Z)
    zinf = np.abs(Z).max(1)
    sig = 1.4826 * float(np.median(np.abs(Z - np.median(Z, 0))))
    eps_hat = lambda q: float(np.quantile(zinf, q))
    print("  %d observations  sigma_hat %.3f px  eps_hat(0.90) %.3f  (0.99) %.3f  (0.999) %.3f"
          % (len(Z), sig, eps_hat(0.90), eps_hat(0.99), eps_hat(0.999)), flush=True)

    print("pose model: Jacobians + Gauss-Markov covariance ...", flush=True)
    Hs, Sig = {}, {}
    for i, im in imgs.items():
        m = im.p3d_ids >= 0
        X = np.array([pts[int(q)].xyz for q in im.p3d_ids[m]])
        J = jacobians(pose[i][0], im.center, pose[i][2], X)
        Hs[i] = dict(zip(np.flatnonzero(m).tolist(), J))
        Hst = J.reshape(-1, 6)
        Sig[i] = sig ** 2 * np.linalg.inv(Hst.T @ Hst)

    # ================= GT: evaluation only =================================
    gt = CC.load_gt(SCENE)
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))
    rows = CC.observations(cams, imgs, pts, gt, s_, Ra, ta)
    lm = CC.references_and_scores(rows, cams, imgs, pts, gt)
    pids_all = np.array(sorted(pts))
    perm = np.random.default_rng(0).permutation(pids_all)
    cal = set(perm[: len(perm) // 2].tolist())
    test = [p for p in pids_all if p not in cal and lm[p]["Xstar"] is not None]

    # validation of the pose covariance against the TRUE pose error
    mah, pix_ratio = [], []
    for i in ids:
        g = gt[imgs[i].name]
        R_true, c_true = g["R"] @ Ra.T, s_ * Ra @ g["c"] + ta
        d = np.concatenate([Rotation.from_matrix(R_true @ imgs[i].R.T).as_rotvec(),
                            c_true - imgs[i].center])
        mah.append(float(d @ np.linalg.solve(Sig[i], d)))
        for Hk in list(Hs[i].values())[::5]:
            sd = np.sqrt(np.einsum("aj,jk,ak->a", Hk, Sig[i], Hk))
            pix_ratio.append(np.abs(Hk @ d) / sd)
    pix_ratio = np.concatenate(pix_ratio)
    pose_check = {"median_mahalanobis2": float(np.median(mah)),
                  "chi2_6_median_expected": float(chi2.median(6)),
                  "median_|true pixel shift| / predicted std": float(np.median(pix_ratio)),
                  "expected_if_calibrated (|N(0,1)| median)": 0.674}
    print("pose covariance check:", pose_check, flush=True)

    # ---- per-observation residual of the TRUE point under theta_hat ------
    obs = []                                          # (pid, i, j, res(2), m_k)
    for p in test:
        P = pts[p]
        for i, j in zip(P.image_ids, P.point2D_idxs):
            r = project(*pose[i], lm[p]["Xstar"])[0][0] - imgs[i].xys[j]   # pi - y
            obs.append((p, int(i), int(j), r, P.track_length))

    # ---- exact bounded-error pose extents (fix 1), per b -----------------
    def exact_for(b):
        tasks, keys = [], []
        by_img = {}
        for p, i, j, r, m in obs:
            by_img.setdefault(i, []).append(j)
        for i, js in by_img.items():
            Hst = np.array(list(Hs[i].values())).reshape(-1, 6)
            tasks.append((Hst, b, [Hs[i][j] for j in js]))
            keys.append([(i, j) for j in js])
        with Pool(a.procs) as pool:
            res = pool.map(exact_extents, tasks)
        out = {}
        for ks, rr in zip(keys, res):
            out.update(dict(zip(ks, rr)))
        return out

    # ---- configurations: per-observation pixel interval for (pi - y) -----
    lam_gt = None
    with open(os.path.join(MODEL, "conformal", "normalized", "nm_efficiency.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["variant"] == "plain" and float(r["alpha"]) == ALPHA0:
                lam_gt = float(r["lambda_seed0"])
    eps_det_gt = None
    srt = np.sort([r["s_det"] for r in rows if np.isfinite(r["s_det"]) and r["pid"] in cal])
    eps_det_gt = CC.conformal_eps(srt, 0.10)

    configs = {}
    configs["CP-GT (landmark-level, benchmark)"] = lambda o: (np.full(2, -lam_gt), np.full(2, lam_gt))

    print("draft box inflation, b = eps_det(GT) ...", flush=True)
    boxes = {}
    for i in imgs:
        Hst = np.array(list(Hs[i].values())).reshape(-1, 6)
        boxes[i] = pose_box(Hst, eps_det_gt)
    configs["draft: eps_det(GT) + box"] = lambda o: (
        -eps_det_gt + interval_image(Hs[o[1]][o[2]], *boxes[o[1]])[0],
        eps_det_gt + interval_image(Hs[o[1]][o[2]], *boxes[o[1]])[1])

    print("fix 1: exact inflation, b = eps_det(GT) ...", flush=True)
    ex_gt = exact_for(eps_det_gt)
    configs["fix1: eps_det(GT) + exact"] = lambda o: (-eps_det_gt + ex_gt[(o[1], o[2])][0],
                                                     eps_det_gt + ex_gt[(o[1], o[2])][1])

    q90 = eps_hat(0.90)
    print("white-box bounded-error, b = eps_hat(0.90) ...", flush=True)
    ex_hat = exact_for(q90)
    configs["WB bounded: eps_hat(.90) + exact"] = lambda o: (-q90 + ex_hat[(o[1], o[2])][0],
                                                            q90 + ex_hat[(o[1], o[2])][1])

    def gm(o, q):
        sd = np.sqrt(np.einsum("aj,jk,ak->a", Hs[o[1]][o[2]], Sig[o[1]], Hs[o[1]][o[2]]))
        e = eps_hat(q)
        zq = norm.ppf(0.5 + 0.5 * q)
        return -e - zq * sd, e + zq * sd

    for q in (0.90, 0.99, 0.999):
        configs["WB statistical: q=%.3f" % q] = (lambda qq: (lambda o: gm(o, qq)))(q)
    configs["WB statistical: Sidak q_k"] = lambda o: gm(o, (1 - ALPHA0) ** (1.0 / o[4]))

    # ---- evaluate ---------------------------------------------------------
    print("evaluating %d configurations on %d test landmarks ..." % (len(configs), len(test)),
          flush=True)
    by_p = {}
    for o in obs:
        by_p.setdefault(o[0], []).append(o)
    results = []
    for name, fn in configs.items():
        ok_l, width, dia, cover_o = [], [], [], []
        for p in test:
            los, his, allok = [], [], True
            for o in by_p[p]:
                lo, hi = fn(o)
                inside = bool(((o[3] >= lo) & (o[3] <= hi)).all())
                cover_o.append(inside)
                allok &= inside
                los.append(lo)
                his.append(hi)
                width.append(0.5 * float(np.mean(hi - lo)))
            ok_l.append(allok)
            P = pts[p]
            views = [(*pose[i], imgs[i].xys[j]) for i, j in zip(P.image_ids, P.point2D_idxs)]
            d = diameter(halfspaces_rect(views, los, his))
            dia.append(d / lm[p]["d_k"] if np.isfinite(d) else d)
        dia = np.array(dia)
        fin = dia[np.isfinite(dia)]
        results.append({"config": name,
                        "landmark_coverage": float(np.mean(ok_l)),
                        "observation_coverage": float(np.mean(cover_o)),
                        "median_half_width_px": float(np.median(width)),
                        "median_diam_over_dk": float(np.median(fin)),
                        "unbounded": float(np.isinf(dia).mean()),
                        "empty": float(np.isnan(dia).mean())})
        r = results[-1]
        print("  %-36s landmark cov %.3f   half-width %7.2f px   diam/d_k %.4f   unbounded %.1f%%"
              % (name, r["landmark_coverage"], r["median_half_width_px"],
                 r["median_diam_over_dk"], 100 * r["unbounded"]), flush=True)

    rep = {"scene": a.scene, "alpha0": ALPHA0, "test_landmarks": len(test),
           "gt_free_model": {"sigma_hat_px": sig, "eps_hat": {q: eps_hat(q) for q in
                                                              (0.5, 0.9, 0.99, 0.999)},
                             "observations_used": len(Z)},
           "gt_benchmarks": {"lambda_CP_landmark": lam_gt, "eps_det_CP_obs_0.10": eps_det_gt},
           "pose_covariance_check": pose_check, "results": results,
           "seconds": round(time.time() - T0)}
    json.dump(rep, open(os.path.join(OUT, "whitebox.json"), "w"), indent=2)
    with open(os.path.join(OUT, "whitebox.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
