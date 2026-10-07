"""RAIM-style pose inflation (Bruvik, Valentin et al., protection levels for PnP).

    python raim.py --scene airplane

For each image i, the pose error dtheta_i is bounded the way residual-based RAIM
bounds a navigation error, but for the quantity we actually need: the pixel shift
of observation (i,k) caused by pose error, h^T dtheta with h = H_ik[a] (row a of
the 2x6 Jacobian).  No per-axis box, so no box looseness.

    whitened model   z = H dtheta + v + A_S delta,   v ~ N(0, I),  H = H_raw / sigma
    residual         r = (I - H H^+) z               (only this part is visible)
    test             ||r||^2 ~ chi2_dof under H0,    tau from the false-alarm budget
    PL(h)            max_S g_S(h) * lam_bar  +  k_H0 * sigma_h
                     g_S(h)^2 = u_S^T (A_S^T P A_S)^{-1} u_S,  u = A_S^T H^+T h
                     sigma_h  = sigma * sqrt(h^T (H_raw^T H_raw)^{-1} h)
    region           Y_ik = y_ik (+) [-w, w]^2,  w_a = k_det * sigma + PL(H_ik[a])

Fault hypotheses S are single observations of image i (s_max = 1), prior
P(H_S) = P_kp (1 - P_kp)^(m-1).  Risk per (observation, axis) is Sidak-split from
the landmark level alpha over 2 m_k axis constraints, half to detection noise,
half to pose.

GT-free inputs: sigma (standardized leave-one-image-out residuals, whitebox.py),
H from theta_hat and X_hat, residual dof corrected for point leverage.
Near-duplicate keypoints (same landmark twice in one image) are merged into one
row before building H, the fault set and the dof ("dedup"); "dup" keeps them.

GT enters only the evaluation (coverage of X*_k, true pose error) and the
"shape + CP scale" configuration, which calibrates one scalar on the calibration
half (a within-scene proxy for cross-scene calibration, V6).
"""
import os
import sys
import csv
import json
import math
import time
import argparse
import numpy as np
from scipy.stats import norm, chi2, ncx2
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project
from cp_step0_align import umeyama
import cp_conformal as CC
from pose_inflation import jacobians, halfspaces_rect, diameter
import whitebox as WB

ALPHA = 0.10          # landmark-level risk
C_REQ = 0.01          # false-alarm budget of the per-image residual test
P_KPS = [1e-4, 1e-3, 1e-2]


# --------------------------------------------------------------------------
def sidak(m):
    """per-(observation, axis) risk for landmark risk ALPHA over 2m constraints"""
    return 1.0 - (1.0 - ALPHA) ** (1.0 / (2 * m))


_lam_cache = {}


def lam_bar(df, tau2, p_md):
    """non-centrality at which the fault is missed with probability p_md"""
    key = (round(df, 1), round(tau2, 2), round(math.log(p_md), 3))
    if key not in _lam_cache:
        f = lambda nc: ncx2.cdf(tau2, df, nc) - p_md
        if f(1e-9) <= 0:                # budget exceeds the fault prior: no fault term
            _lam_cache[key] = 0.0
            return 0.0
        hi = 10.0
        while f(hi) > 0:
            hi *= 2
        _lam_cache[key] = math.sqrt(brentq(f, 1e-9, hi, xtol=1e-6))
    return _lam_cache[key]


class ImageRAIM:
    """Per-image quantities that do not depend on the functional h."""

    def __init__(self, Hraw, r, dof, sigma, p_kp):
        # Hraw (n,2,6) pixels per tangent unit; r (n,2) residual y - pi(theta_hat, X_hat)
        self.H = Hraw
        self.n = len(Hraw)
        Hs = Hraw.reshape(-1, 6)
        self.Ninv = np.linalg.inv(Hs.T @ Hs)
        self.sigma = sigma
        # P_SS = I - H_S N^-1 H_S^T  (sigma-free), inverted per hypothesis
        L = np.einsum("nai,ij,nbj->nab", Hraw, self.Ninv, Hraw)
        P = np.eye(2)[None] - L
        self.Q = np.linalg.pinv(P)
        self.dof = dof
        self.p0 = (1 - p_kp) ** self.n
        self.p_single = self.n * p_kp * (1 - p_kp) ** (self.n - 1)
        self.p_nm = max(0.0, 1.0 - self.p0 - self.p_single)
        self.tau2 = float(chi2.ppf(1 - C_REQ / self.p0, dof))
        self.stat = float((r.reshape(-1) ** 2).sum() / sigma ** 2)
        self.passed = self.stat <= self.tau2

    def pl(self, h, I_req, fault=True):
        """protection level on h^T dtheta at integrity risk I_req; h (K,6), I_req (K,)"""
        s_h = self.sigma * np.sqrt(np.einsum("ki,ij,kj->k", h, self.Ninv, h))
        I_h0 = I_req / 2
        k_h0 = norm.ppf(1 - I_h0 / (2 * self.p0))
        if not fault:
            return norm.ppf(1 - I_req / 2) * s_h
        # u_S = sigma * H_S N^-1 h       -> (n,2,K)
        U = self.sigma * np.einsum("nai,ij,kj->nak", self.H, self.Ninv, h)
        g2 = np.einsum("nak,nab,nbk->nk", U, self.Q, U)
        g = np.sqrt(np.maximum(g2, 0).max(0))
        p_md = (I_req / 2) / self.p_single
        lb = np.array([lam_bar(self.dof, self.tau2, float(min(pm, 0.999))) for pm in p_md])
        return g * lb + k_h0 * s_h


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--set", default="all", choices=["all", "perimg"],
                    help="perimg: only the per-image variance-factor variant")
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "conformal", "raim")
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()

    cams, imgs, pts = read_model(MODEL)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}

    # ---- GT-free noise scale ---------------------------------------------
    wbj = os.path.join(MODEL, "conformal", "whitebox", "whitebox.json")
    if os.path.exists(wbj):
        sigma = json.load(open(wbj))["gt_free_model"]["sigma_hat_px"]
    else:
        Z = [z for z in (WB.standardized_loio(p, imgs, pose) for p in pts.values())
             if z is not None]
        Z = np.concatenate(Z)
        sigma = 1.4826 * float(np.median(np.abs(Z - np.median(Z, 0))))
    print("sigma_hat (GT-free) %.3f px" % sigma, flush=True)

    # ---- unique (landmark, image) rows, multiplicity, point leverage -------
    # lev[(pid, iid)] = (leverage dedup, leverage with duplicates), trace of J A^-1 J^T
    print("point leverage ...", flush=True)
    lev = {}
    n_dup_obs = 0
    for pid, P in pts.items():
        iids, cnt = np.unique(P.image_ids, return_counts=True)
        n_dup_obs += int((cnt - 1).sum())
        views = [pose[i] + (None,) for i in iids]
        J = WB.landmark_jacobians(views, P.xyz)                  # (u,2,3)
        A1 = np.einsum("uai,uaj->ij", J, J)
        A2 = np.einsum("u,uai,uaj->ij", cnt, J, J)
        t1 = np.einsum("uai,ij,uaj->u", J, np.linalg.pinv(A1), J)
        t2 = np.einsum("uai,ij,uaj->u", J, np.linalg.pinv(A2), J)
        for i, c, a1, a2 in zip(iids, cnt, t1, t2):
            lev[(pid, int(i))] = (float(a1), float(a2), int(c))
    n_obs = sum(len(P.image_ids) for P in pts.values())
    print("  %d observations, %d are near-duplicates (%.1f%%)"
          % (n_obs, n_dup_obs, 100 * n_dup_obs / n_obs), flush=True)

    # ---- per-image Jacobians and residuals --------------------------------
    print("per-image Jacobians ...", flush=True)
    img_rows = {}          # iid -> dict(pids, H (u,2,6), r_dedup (u,2), rows_all, r_all)
    for i, im in imgs.items():
        m = np.flatnonzero(im.p3d_ids >= 0)
        pid_all = im.p3d_ids[m].astype(int)
        upid, inv = np.unique(pid_all, return_inverse=True)
        X = np.array([pts[p].xyz for p in upid])
        H = jacobians(pose[i][0], im.center, pose[i][2], X)
        proj = project(*pose[i], X)[0]
        r_all = im.xys[m] - proj[inv]
        r_ded = np.array([r_all[inv == u].mean(0) for u in range(len(upid))])
        img_rows[i] = dict(pids=upid, H=H, inv=inv, r_all=r_all, r_ded=r_ded,
                           row_of={int(p): u for u, p in enumerate(upid)})

    # ---- GT (evaluation only) ---------------------------------------------
    gt = CC.load_gt(SCENE)
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))
    rows = CC.observations(cams, imgs, pts, gt, s_, Ra, ta)
    lm = CC.references_and_scores(rows, cams, imgs, pts, gt)
    pids_all = np.array(sorted(pts))
    perm = np.random.default_rng(0).permutation(pids_all)
    cal = set(perm[: len(perm) // 2].tolist())
    have = [p for p in pids_all if lm[p]["Xstar"] is not None]
    test = [p for p in have if p not in cal]
    calL = [p for p in have if p in cal]

    dtheta = {}
    for i in ids:
        g = gt[imgs[i].name]
        R_true, c_true = g["R"] @ Ra.T, s_ * Ra @ g["c"] + ta
        dtheta[i] = np.concatenate([Rotation.from_matrix(R_true @ imgs[i].R.T).as_rotvec(),
                                    c_true - imgs[i].center])

    # observations of GT landmarks: (pid, iid, j, residual pi(theta_hat, X*) - y, m_k)
    obs = []
    for p in have:
        P = pts[p]
        for i, j in zip(P.image_ids, P.point2D_idxs):
            r = project(*pose[i], lm[p]["Xstar"])[0][0] - imgs[i].xys[j]
            obs.append((p, int(i), int(j), r, len(set(P.image_ids.tolist()))))
    by_img = {}
    for n, o in enumerate(obs):
        by_img.setdefault(o[1], []).append(n)

    # ---- configurations ----------------------------------------------------
    variants = [("RAIM dedup, P_kp=%g" % pk, True, pk, True, False) for pk in P_KPS]
    variants += [("RAIM dup, P_kp=1e-3", False, 1e-3, True, False),
                 ("nominal only (no fault term), dedup", True, 1e-3, False, False)]
    shape_of = "RAIM dedup, P_kp=%g" % 1e-3
    if a.set == "perimg":
        variants = [("RAIM dedup, per-image sigma_i, P_kp=1e-3", True, 1e-3, True, True)]
        shape_of = variants[0][0]

    results, pose_rep, img_rep, W1 = [], {}, {}, None
    for name, dedup, pk, fault, perimg in variants:
        print("[%s]" % name, flush=True)
        w = np.zeros((len(obs), 2))           # half-width per obs and axis (sigma as given)
        pose_cov, pose_ratio, passed, pnm, dofs, sig_used = [], [], [], [], [], []
        for i, R in img_rows.items():
            if dedup:
                H, r = R["H"], R["r_ded"]
                dof = sum(2 - lev[(int(p), i)][0] for p in R["pids"]) - 6
            else:
                H, r = R["H"][R["inv"]], R["r_all"]
                dof = sum(lev[(int(p), i)][2] * (2 - lev[(int(p), i)][1])
                          for p in R["pids"]) - 6
            ir = ImageRAIM(H, r, dof, sigma, pk)
            s_i = sigma
            if perimg:                    # a-posteriori variance factor, GT-free
                s_i = sigma * math.sqrt(ir.stat / dof)
                ir = ImageRAIM(H, r, dof, s_i, pk)
            sig_used.append(s_i)
            passed.append(ir.passed)
            pnm.append(ir.p_nm)
            dofs.append(dof)
            # pose-axis protection levels vs true pose error (Bonferroni over 6 axes)
            if i in dtheta:
                pl6 = ir.pl(np.eye(6), np.full(6, ALPHA / 6), fault)
                pose_cov.append(bool((np.abs(dtheta[i]) <= pl6).all()))
                pose_ratio.append(np.abs(dtheta[i]) / pl6)
            # pixel-shift protection levels for every GT observation in this image
            ns = by_img.get(i, [])
            if not ns:
                continue
            hs, Ireq = [], []
            for n in ns:
                o = obs[n]
                Hk = R["H"][R["row_of"][o[0]]]
                q = sidak(o[4])
                hs += [Hk[0], Hk[1]]
                Ireq += [q / 2, q / 2]
            pl = ir.pl(np.array(hs), np.array(Ireq), fault).reshape(-1, 2)
            for n, plk in zip(ns, pl):
                k_det = norm.ppf(1 - sidak(obs[n][4]) / 4)
                w[n] = k_det * s_i + plk
        pose_ratio = np.array(pose_ratio)
        pose_rep[name] = {"images_all6_axes_covered": float(np.mean(pose_cov)),
                          "per_axis_covered": (pose_ratio <= 1).mean(0).round(3).tolist(),
                          "median_|true err|/PL_per_axis": np.median(pose_ratio, 0).round(2).tolist()}
        img_rep[name] = {"images_pass_residual_test": float(np.mean(passed)),
                         "median_unmonitored_prior_P_NM": float(np.median(pnm)),
                         "median_dof": float(np.median(dofs)),
                         "sigma_used_px_p10_p50_p90": np.percentile(sig_used, [10, 50, 90]).round(3).tolist()}
        print("  images pass test %.3f   median P_NM %.2e   pose: all 6 axes covered %.3f"
              % (np.mean(passed), np.median(pnm), np.mean(pose_cov)), flush=True)
        results.append((name, w.copy()))
        if name == shape_of:
            W1 = w / sigma                       # shape at unit scale (PL is linear in sigma)

    # ---- shape + CP scale (GT on the calibration half) ---------------------
    obs_of = {}
    for n, o in enumerate(obs):
        obs_of.setdefault(o[0], []).append(n)
    score = {p: max(float((np.abs(obs[n][3]) / W1[n]).max()) for n in obs_of[p]) for p in have}
    lam_hat = CC.conformal_eps(np.sort([score[p] for p in calL]), ALPHA)
    results.append(("%s shape + CP scale (GT cal half)" % shape_of, W1 * lam_hat))
    print("CP scale on RAIM shape: lambda %.3f  (GT-free sigma %.3f, ratio %.1f)"
          % (lam_hat, sigma, lam_hat / sigma), flush=True)

    # ---- evaluate on test landmarks ----------------------------------------
    print("evaluating on %d test landmarks ..." % len(test), flush=True)
    table = []
    for name, w in results:
        ok, dia, width, cov_o = [], [], [], []
        for p in test:
            ns = obs_of[p]
            inside = [bool((np.abs(obs[n][3]) <= w[n]).all()) for n in ns]
            cov_o += inside
            ok.append(all(inside))
            width += [float(w[n].mean()) for n in ns]
            P = pts[p]
            views = [(*pose[i], imgs[i].xys[j]) for i, j in zip(P.image_ids, P.point2D_idxs)]
            d = diameter(halfspaces_rect(views, [-w[n] for n in ns], [w[n] for n in ns]))
            dia.append(d / lm[p]["d_k"] if np.isfinite(d) else d)
        dia = np.array(dia)
        fin = dia[np.isfinite(dia)]
        row = {"config": name, "landmark_coverage": float(np.mean(ok)),
               "observation_coverage": float(np.mean(cov_o)),
               "median_half_width_px": float(np.median(width)),
               "median_diam_over_dk": float(np.median(fin)) if len(fin) else float("nan"),
               "unbounded": float(np.isinf(dia).mean()), "empty": float(np.isnan(dia).mean())}
        table.append(row)
        print("  %-40s cov %.3f  half-width %7.2f px  diam/d_k %.4f"
              % (name, row["landmark_coverage"], row["median_half_width_px"],
                 row["median_diam_over_dk"]), flush=True)

    rep = {"scene": a.scene, "alpha": ALPHA, "C_REQ": C_REQ, "sigma_hat_px": sigma,
           "observations": n_obs, "near_duplicate_observations": n_dup_obs,
           "test_landmarks": len(test), "cp_scale_lambda": lam_hat,
           "image_tests": img_rep, "pose_check": pose_rep, "results": table,
           "seconds": round(time.time() - T0)}
    tag = "" if a.set == "all" else "_" + a.set
    json.dump(rep, open(os.path.join(OUT, "raim%s.json" % tag), "w"), indent=2)
    with open(os.path.join(OUT, "raim%s.csv" % tag), "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(table[0].keys()))
        wr.writeheader()
        wr.writerows(table)
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
