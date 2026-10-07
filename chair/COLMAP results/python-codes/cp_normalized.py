"""Normalized (locally adaptive) conformal calibration at the LANDMARK level.

    python cp_normalized.py --scene airplane

Tolerance region per observation, one global scale lambda:

        X_ik = y_ik (+) lambda * A_ik [-1, 1]^2          (a parallelogram)

and F_k is the intersection of the back-projected regions, still a polyhedron.
X*_k lies in F_k  <=>  max_i || A_ik^{-1} (pi(theta_hat_i, X*_k) - y_ik) ||_inf <= lambda,
so the landmark-level conformity score is

        s_k = max_{i in views(k)}  || A_ik^{-1} r_ik ||_inf ,   r_ik = y_ik - pi(theta_hat_i, X*_k)

and split conformal on s_k gives  P(X*_k in F_k(lambda)) >= 1 - alpha  directly.

Four choices of A (the shape), compared on coverage AND set size:
    plain     A = I                         (the uniform square used so far)
    scale     A = s_ik I                    (SIFT detection scale)
    tensor    A = M_ik^{-1/2}               (structure tensor: Forstner covariance shape)
    scale+aniso  A = s_ik Mhat_ik^{-1/2}    (scale for size, tensor only for anisotropy;
                                             Mhat = M / sqrt(det M))

M is the structure tensor at the keypoint's own scale: Gaussian derivatives at
sigma_D = clip(0.5 s, 0.7, 12) px, integration sigma_I = 2 sigma_D.
"""
import os
import sys
import csv
import json
import math
import sqlite3
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project
from cp_step0_align import umeyama
import cp_conformal as CC
import exact_polytope as ex

ALPHAS = [0.05, 0.10, 0.20]
SEEDS = list(range(20))
VARIANTS = ["plain", "scale", "tensor", "scale+aniso"]
SIG_LEVELS = np.geomspace(0.7, 12.0, 14)
NK_BINS = [(2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10 ** 6, ">20")]


# --------------------------------------------------------------------------
def keypoint_shapes(db_path, imgs):
    """image_id -> (N,) detection scale in px, read from database.db."""
    db = sqlite3.connect(db_path)
    out = {}
    for iid, im in imgs.items():
        r, c, blob = db.execute("select rows, cols, data from keypoints where image_id=?",
                                (iid,)).fetchone()
        k = np.frombuffer(blob, np.float32).reshape(r, c)
        assert np.abs(k[:, :2] - im.xys).max() < 1e-3, "keypoints do not match model"
        a = k[:, 2:6]
        out[iid] = 0.5 * (np.hypot(a[:, 0], a[:, 2]) + np.hypot(a[:, 1], a[:, 3]))
    return out


def noise_sigma(gray):
    """Immerkaer (1996) fast noise std estimate, same intensity units as gray."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    from scipy.signal import convolve2d
    h, w = gray.shape
    return math.sqrt(math.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(
        convolve2d(gray, k, mode="valid")).sum()


def structure_tensors(img_path, uv, scales):
    """M (N,2,2) at each keypoint, at its own scale.  Intensities in [0,255]."""
    I = np.asarray(Image.open(img_path).convert("L"), float)
    sd = np.clip(0.5 * scales, SIG_LEVELS[0], SIG_LEVELS[-1])
    lvl = np.abs(np.log(sd)[:, None] - np.log(SIG_LEVELS)[None]).argmin(1)
    M = np.zeros((len(uv), 2, 2))
    coords = np.stack([uv[:, 1] - 0.5, uv[:, 0] - 0.5])          # (row, col)
    for L in np.unique(lvl):
        s_d = SIG_LEVELS[L]
        Iu = gaussian_filter(I, s_d, order=(0, 1))
        Iv = gaussian_filter(I, s_d, order=(1, 0))
        J = [gaussian_filter(a, 2 * s_d) for a in (Iu * Iu, Iu * Iv, Iv * Iv)]
        m = lvl == L
        vals = [map_coordinates(j, coords[:, m], order=1, mode="nearest") for j in J]
        M[m, 0, 0], M[m, 0, 1], M[m, 1, 1] = vals
        M[m, 1, 0] = M[m, 0, 1]
    return M, noise_sigma(I), SIG_LEVELS[lvl]


def inv_sqrt(M, reg):
    w, V = np.linalg.eigh(M + reg * np.eye(2))
    return V @ np.diag(w ** -0.5) @ V.T, w


# --------------------------------------------------------------------------
def shape_matrix(variant, s, M, reg):
    """A such that X_ik = y + lambda * A [-1,1]^2."""
    if variant == "plain":
        return np.eye(2)
    if variant == "scale":
        return s * np.eye(2)
    Mi, w = inv_sqrt(M, reg)
    if variant == "tensor":
        return Mi
    return s * Mi * math.sqrt(math.sqrt(np.prod(w)))          # unit-determinant shape


def halfspaces_general(views, Ainv_list, lam, z_min=1e-3):
    """F_k with parallelogram tolerances; same linearisation as exact_polytope."""
    H = []
    for (R, t, intr, y), Ainv in zip(views, Ainv_list):
        fx, fy = intr[0], intr[1]
        n_star, gain = ex.undistort(np.asarray(y, float), intr)
        for a in Ainv:                                        # rows of A^{-1}
            b = gain * np.array([a[0] * fx, a[1] * fy])      # pixel residual ~ b . (n - n*)
            bn = b @ n_star
            for sg in (1.0, -1.0):
                lim = bn + sg * lam
                row = sg * (b[0] * R[0] + b[1] * R[1] - lim * R[2])
                off = sg * (b[0] * t[0] + b[1] * t[1] - lim * t[2])
                H.append(np.append(row, off))
        H.append(np.append(-R[2], -t[2] + z_min))
    return np.asarray(H)


def set_size(H):
    c, r = ex.chebyshev_center(H)
    if c is None:
        f = linprog(np.zeros(3), A_ub=H[:, :3], b_ub=-H[:, 3],
                    bounds=[(None, None)] * 3, method="highs")
        return (float("inf"), "unbounded") if f.status == 0 else (float("nan"), "empty")
    if r <= 0:
        return float("nan"), "empty"
    V, _, _ = ex.vertices_and_volume(H, c)
    return float(np.linalg.norm(V[:, None] - V[None], axis=-1).max()), "ok"


def conformal(sorted_cal, alpha):
    N = len(sorted_cal)
    i = math.ceil((N + 1) * (1 - alpha))
    return float(sorted_cal[i - 1]) if i <= N else float("inf")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "conformal", "normalized")
    os.makedirs(OUT, exist_ok=True)

    cams, imgs, pts = read_model(MODEL)
    gt = CC.load_gt(SCENE)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))

    print("GT references ...", flush=True)
    rows = CC.observations(cams, imgs, pts, gt, s_, Ra, ta)
    lm = CC.references_and_scores(rows, cams, imgs, pts, gt)

    print("keypoint scales + structure tensors ...", flush=True)
    scale = keypoint_shapes(os.path.join(SCENE, "database.db"), imgs)
    Mt, sd_used, sigma_img = {}, {}, []
    for iid, im in imgs.items():
        M, sn, sdl = structure_tensors(os.path.join(SCENE, "images", im.name), im.xys, scale[iid])
        Mt[iid], sd_used[iid] = M, sdl
        sigma_img.append(sn)
    sigma_n = float(np.median(sigma_img))
    allM = np.concatenate([Mt[i] for i in imgs])
    lam2 = np.linalg.eigvalsh(allM)[:, 0]
    reg = 1e-3 * float(np.median(np.trace(allM, axis1=1, axis2=2)))
    print("  image noise sigma (Immerkaer) median %.3f intensity levels" % sigma_n)
    print("  tensor regulariser %.3g  (flat keypoints, lambda2 < reg: %.1f%%)"
          % (reg, 100 * (lam2 < reg).mean()))

    # ---- per-landmark scores for every variant ---------------------------
    pids = [p for p in sorted(pts) if lm[p]["Xstar"] is not None]
    S = {v: np.zeros(len(pids)) for v in VARIANTS}
    nk = np.zeros(len(pids), int)
    dk = np.zeros(len(pids))
    obs_rows = []
    views_of, Ainv_of = {}, {}
    for n, pid in enumerate(pids):
        p = pts[pid]
        X = lm[pid]["Xstar"]
        nk[n] = len(set(p.image_ids.tolist()))
        dk[n] = lm[pid]["d_k"]
        views_of[pid] = [(*pose[i], imgs[i].xys[j]) for i, j in zip(p.image_ids, p.point2D_idxs)]
        Ainv_of[pid] = {v: [] for v in VARIANTS}
        per = {v: [] for v in VARIANTS}
        for i, j in zip(p.image_ids, p.point2D_idxs):
            r = imgs[i].xys[j] - project(*pose[i], X)[0][0]
            M = Mt[i][j]
            s = float(scale[i][j])
            w, V = np.linalg.eigh(M + reg * np.eye(2))
            for v in VARIANTS:
                A = shape_matrix(v, s, M, reg)
                Ai = np.linalg.inv(A)
                Ainv_of[pid][v].append(Ai)
                per[v].append(np.abs(Ai @ r).max())
            e2 = V[:, 0]
            pred = sigma_n / math.sqrt(4 * math.pi * (2 * sd_used[i][j]) ** 2 * w[0])
            obs_rows.append([pid, int(i), int(j), s, w[1], w[0], r[0], r[1],
                             abs(r @ e2), pred] + [per[v][-1] for v in VARIANTS])
        for v in VARIANTS:
            S[v][n] = max(per[v])

    with open(os.path.join(OUT, "nm_observations.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["point3D_id", "image_id", "point2D_idx", "scale_px", "lambda1", "lambda2",
                      "r_u", "r_v", "abs_r_along_weak_axis", "noise_model_std_weak_axis"]
                     + ["score_" + v for v in VARIANTS])
        wtr.writerows(obs_rows)

    # ---- split conformal: pooled and Mondrian-by-n_k ---------------------
    cov_rows = []
    lam_seed0 = {}
    for v in VARIANTS:
        for al in ALPHAS:
            covp, covm, lams, bin_p, bin_m = [], [], [], {b[2]: [] for b in NK_BINS}, \
                {b[2]: [] for b in NK_BINS}
            for seed in SEEDS:
                perm = np.random.default_rng(seed).permutation(len(pids))
                cal, te = perm[: len(pids) // 2], perm[len(pids) // 2:]
                lam = conformal(np.sort(S[v][cal]), al)
                lams.append(lam)
                covp.append(float((S[v][te] <= lam).mean()))
                hit_m = np.zeros(len(te), bool)
                for lo, hi, lab in NK_BINS:
                    mc = (nk[cal] >= lo) & (nk[cal] <= hi)
                    mt = (nk[te] >= lo) & (nk[te] <= hi)
                    if not mt.any():
                        continue
                    lb = conformal(np.sort(S[v][cal][mc]), al) if mc.sum() else float("inf")
                    hit_m[mt] = S[v][te][mt] <= lb
                    bin_p[lab].append(float((S[v][te][mt] <= lam).mean()))
                    bin_m[lab].append(float(hit_m[mt].mean()))
                covm.append(float(hit_m.mean()))
                if seed == 0:
                    lam_seed0[(v, al)] = (lam, te)
            cov_rows.append([v, al, np.mean(lams), np.min(lams), np.max(lams),
                             np.mean(covp), np.min(covp), np.max(covp), np.mean(covm)]
                            + [np.mean(bin_p[b[2]]) for b in NK_BINS]
                            + [np.mean(bin_m[b[2]]) for b in NK_BINS])
    with open(os.path.join(OUT, "nm_coverage.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["variant", "alpha", "lambda_mean", "lambda_min", "lambda_max",
                      "coverage_mean", "coverage_min", "coverage_max", "coverage_mondrian"]
                     + ["pooled_nk_" + b[2] for b in NK_BINS]
                     + ["mondrian_nk_" + b[2] for b in NK_BINS])
        wtr.writerows(cov_rows)

    # ---- set size at the calibrated lambda (seed 0, test landmarks) ------
    print("set sizes ...", flush=True)
    eff_rows = []
    for v in VARIANTS:
        for al in (0.05, 0.10):
            lam, te = lam_seed0[(v, al)]
            dia, stat = [], []
            for n in te:
                pid = pids[n]
                H = halfspaces_general(views_of[pid], Ainv_of[pid][v], lam)
                d, st = set_size(H)
                dia.append(d / dk[n] if st == "ok" else (np.inf if st == "unbounded" else np.nan))
                stat.append(st)
            dia = np.array(dia)
            stat = np.array(stat)
            fin = dia[np.isfinite(dia)]
            eff_rows.append([v, al, lam, len(te), float(np.median(fin)),
                             float(np.percentile(fin, 90)), float((stat == "unbounded").mean()),
                             float((stat == "empty").mean())]
                            + [float(np.median(dia[(nk[te] >= lo) & (nk[te] <= hi) &
                                                   np.isfinite(dia)]))
                               if ((nk[te] >= lo) & (nk[te] <= hi) & np.isfinite(dia)).any()
                               else float("nan") for lo, hi, _ in NK_BINS])
            print("  %-12s a=%.2f  lambda %.4g  median diam/d_k %.4f  unbounded %.1f%%"
                  % (v, al, lam, eff_rows[-1][4], 100 * eff_rows[-1][6]), flush=True)
    with open(os.path.join(OUT, "nm_efficiency.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["variant", "alpha", "lambda_seed0", "n_test", "median_diam_over_dk",
                      "p90_diam_over_dk", "unbounded_rate", "empty_rate"]
                     + ["median_diam_nk_" + b[2] for b in NK_BINS])
        wtr.writerows(eff_rows)

    # ---- noise-model diagnostic -----------------------------------------
    o = np.array([[r[8], r[9]] for r in obs_rows])
    ratio = o[:, 0] / o[:, 1]
    rep = {"scene": a.scene, "landmarks_with_GT": len(pids), "observations": len(obs_rows),
           "image_noise_sigma": sigma_n, "tensor_regulariser": reg,
           "flat_keypoint_fraction": float((lam2 < reg).mean()),
           "noise_model_weak_axis": {
               "note": "predicted std = sigma / sqrt(4 pi sigma_I^2 lambda_2) (Forstner, "
                       "Gaussian window); measured = |r . e_2|",
               "median_predicted_std_px": float(np.median(o[:, 1])),
               "median_measured_abs_error_px": float(np.median(o[:, 0])),
               "ratio_measured_over_predicted": {
                   "p25": float(np.percentile(ratio, 25)),
                   "median": float(np.median(ratio)),
                   "p75": float(np.percentile(ratio, 75))}}}
    json.dump(rep, open(os.path.join(OUT, "nm_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
