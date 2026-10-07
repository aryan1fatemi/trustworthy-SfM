"""Can a GT-free quantity recover the ground-truth epsilon?

    python loo_epsilon.py --scene airplane

For every observation j of landmark k, re-triangulate k from its OTHER images
(estimated poses, no GT), then measure where detection j falls:

    e_jk = || y_jk - pi(theta_hat_j, X_hat_{k,-j}) ||_inf        (no ground truth)

In-sample reprojection residuals are biased low, because bundle adjustment
fitted X_hat_k TO those detections.  Leaving view j out removes that fit.

The GT-free epsilon is the (1-alpha) quantile of the landmark-level LOO score
over ALL landmarks.  It is then scored against ground truth: what fraction of
landmarks actually contain X*_k at that epsilon?
"""
import os
import sys
import csv
import json
import math
import argparse
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project


def triangulate(views, x0):
    def res(X):
        return np.concatenate([project(R, t, intr, X)[0][0] - y for R, t, intr, y in views])
    return least_squares(res, x0, method="lm", max_nfev=200).x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    a = ap.parse_args()
    MODEL = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene, "COLMAP results"))
    OUT = os.path.join(MODEL, "conformal", "normalized")
    cams, imgs, pts = read_model(MODEL)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}

    loo_k, insample_k, nk_k = {}, {}, {}
    loo_obs, ins_obs = [], []
    for n, (pid, p) in enumerate(pts.items()):
        obs = [(int(i), (*pose[i], imgs[i].xys[j])) for i, j in zip(p.image_ids, p.point2D_idxs)]
        images = sorted(set(i for i, _ in obs))
        nk_k[pid] = len(images)
        ins = [np.abs(project(*v[:3], p.xyz)[0][0] - v[3]).max() for _, v in obs]
        insample_k[pid] = max(ins)
        ins_obs += ins
        if len(images) < 3:                  # leaving one out leaves a single ray
            continue
        worst = 0.0
        for img in images:
            rest = [v for i, v in obs if i != img]
            Xm = triangulate(rest, p.xyz)
            for i, v in obs:
                if i == img:
                    e = float(np.abs(project(*v[:3], Xm)[0][0] - v[3]).max())
                    loo_obs.append(e)
                    worst = max(worst, e)
        loo_k[pid] = worst
        if (n + 1) % 1000 == 0:
            print("  %d / %d" % (n + 1, len(pts)), flush=True)

    # ground-truth landmark score (plain square) from the normalized run
    gt_k = {}
    with open(os.path.join(OUT, "nm_observations.csv")) as fh:
        for r in csv.DictReader(fh):
            k = int(r["point3D_id"])
            gt_k[k] = max(gt_k.get(k, 0.0), float(r["score_plain"]))
    cov = {}
    with open(os.path.join(OUT, "nm_coverage.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["variant"] == "plain":
                cov[float(r["alpha"])] = float(r["lambda_mean"])

    both = [k for k in gt_k if k in loo_k]
    G = np.array([gt_k[k] for k in both])
    L = np.array([loo_k[k] for k in both])
    I = np.array([insample_k[k] for k in both])
    Gall = np.array(list(gt_k.values()))
    rep = {"scene": a.scene, "landmarks_with_GT": len(gt_k),
           "landmarks_with_LOO(n_k>=3)": len(loo_k), "compared": len(both),
           "rank_corr_LOO_vs_GT": float(np.corrcoef(np.argsort(np.argsort(L)),
                                                    np.argsort(np.argsort(G)))[0, 1]),
           "per_alpha": {}}
    print("\n%-6s | %10s | %12s %8s | %12s %8s" % ("alpha", "eps GT-CP", "eps LOO", "GT cov",
                                                   "eps in-samp", "GT cov"))
    for al in (0.05, 0.10, 0.20):
        eL = float(np.quantile(L, 1 - al))
        eI = float(np.quantile(I, 1 - al))
        cL = float((Gall <= eL).mean())
        cI = float((Gall <= eI).mean())
        rep["per_alpha"][str(al)] = {"eps_GT_conformal": cov[al], "eps_LOO": eL,
                                     "coverage_with_eps_LOO": cL, "eps_insample": eI,
                                     "coverage_with_eps_insample": cI}
        print("%-6.2f | %8.2f px | %9.2f px %8.3f | %9.2f px %8.3f"
              % (al, cov[al], eL, cL, eI, cI))
    q = [50, 90, 95, 99]
    rep["observation_quantiles"] = {
        "LOO": dict(zip(map(str, q), np.percentile(loo_obs, q).tolist())),
        "in_sample": dict(zip(map(str, q), np.percentile(ins_obs, q).tolist()))}
    print("\nobservation level  LOO p50/p90/p99: %s   in-sample: %s"
          % (np.round(np.percentile(loo_obs, [50, 90, 99]), 2),
             np.round(np.percentile(ins_obs, [50, 90, 99]), 2)))
    print("rank correlation, landmark LOO score vs GT score: %.2f" % rep["rank_corr_LOO_vs_GT"])
    json.dump(rep, open(os.path.join(OUT, "loo_epsilon.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
