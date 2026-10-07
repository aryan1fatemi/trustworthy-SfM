"""Stage A grown image by image, in COLMAP's registration order.

    python stage_a_incremental.py --scene airplane --eps 3 --order 162,15,17,73

Step n uses the first n images of --order (default: COLMAP's initial pair #162/#15, then the
3rd and 4th registered images, parsed from colmap_log.txt). Landmarks: COLMAP's final tracks
restricted to those images, kept if seen in >= 2 of them. Gauge fixed across steps:
R_r = I, c_r = 0 for r = order[0]; e_j^T c_s = 1 for s = order[1]. Rotations held at COLMAP's
(Stage A). For each step: feasibility at eps, eps*, box of every landmark and camera centre,
and GT coverage in the same gauge (GT enters only the evaluation).
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
import cp_conformal as CC
import stage_a as SA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--order", default="162,15,17,73")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "stage_a", "incremental_eps%g" % a.eps)
    os.makedirs(OUT, exist_ok=True)
    order = [int(x) for x in a.order.split(",")]

    cams, imgs, pts = read_model(MODEL)
    r, s = order[0], order[1]
    Rr, cr = imgs[r].R, imgs[r].center
    c_s = Rr @ (imgs[s].center - cr)
    jj = int(np.argmax(np.abs(c_s)))
    scale = c_s[jj]
    Rg = {i: imgs[i].R @ Rr.T for i in order}
    cg = {i: Rr @ (imgs[i].center - cr) / scale for i in order}
    Xg = {k: Rr @ (p.xyz - cr) / scale for k, p in pts.items()}

    # GT in the same gauge (rule, not fit)
    gt = CC.load_gt(SCENE)
    from cp_step0_align import umeyama
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))
    lifted = {}
    for rw in CC.observations(cams, imgs, pts, gt, s_, Ra, ta):
        if rw["valid"]:
            lifted.setdefault(rw["pid"], []).append(rw["Xgt"])
    gr = gt[imgs[r].name]
    Rs_ = {i: gt[imgs[i].name]["R"] @ gr["R"].T for i in order}
    cs_ = {i: gr["R"] @ (gt[imgs[i].name]["c"] - gr["c"]) for i in order}
    sc = cs_[s][jj]
    cs_ = {i: cs_[i] / sc for i in order}
    Xs_ = {k: gr["R"] @ (np.median(v, 0) - gr["c"]) / sc for k, v in lifted.items()}
    rot_err = {i: float(np.degrees(Rotation.from_matrix(Rs_[i] @ Rg[i].T).magnitude())) for i in order}

    summary, lm_track, cam_track = [], {}, {}
    for n in range(2, len(order) + 1):
        S = order[:n]
        Sset = set(S)
        intr = {i: cams[imgs[i].camera_id].unpack() for i in S}
        obs_raw = [(i, int(imgs[i].p3d_ids[j]), *imgs[i].xys[j]) for i in S
                   for j in np.flatnonzero(imgs[i].p3d_ids >= 0)]
        by_k = {}
        for o in obs_raw:
            by_k.setdefault(o[1], set()).add(o[0])
        L = sorted(k for k, v in by_k.items() if len(v) >= 2)
        Lset = set(L)
        obs_raw = [o for o in obs_raw if o[1] in Lset]
        obs = [(i, k, SA.undistort(u, v, intr[i]), (intr[i][0], intr[i][1])) for i, k, u, v in obs_raw]
        depths = [Rg[i][2] @ (Xg[k] - cg[i]) for i, k, _, _ in obs]
        zmin = 1e-3 * float(np.median(depths))
        res_c = np.array([max(abs(f[0] * (q[0] / q[2] - xn[0])), abs(f[1] * (q[1] / q[2] - xn[1])))
                          for (i, k, xn, f) in obs for q in [Rg[i] @ (Xg[k] - cg[i])]])
        sysA = SA.System(Rg, obs, {i: m for m, i in enumerate(i for i in S if i != r)},
                         {k: m for m, k in enumerate(L)}, r, s, jj, zmin)
        ok, _ = sysA.feasible(a.eps)
        lo, hi = 0.0, float(res_c.max()) * 1.0001
        while hi - lo > 1e-3:
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if sysA.feasible(mid)[0] else (mid, hi)
        eps_star = hi
        # premise for the Stage A constraints (fixed R_hat, COLMAP intrinsics)
        prem = []
        for (i, k, xn, f) in obs:
            if k in Xs_:
                q = Rg[i] @ (Xs_[k] - cs_[i])
                prem.append(max(abs(f[0] * (q[0] / q[2] - xn[0])), abs(f[1] * (q[1] / q[2] - xn[1]))))
        prem = np.array(prem)
        row = {"images": n, "added": imgs[S[-1]].name, "landmarks": len(L), "observations": len(obs),
               "colmap_max_residual_px": float(res_c.max()), "feasible_at_eps": ok, "eps_star_px": eps_star,
               "gt_obs_within_eps_stageA": float((prem <= a.eps).mean()),
               "rotation_error_added_cam_deg": rot_err[S[-1]]}
        print("\n== %d images (+%s): %d landmarks, %d obs | COLMAP max res %.2f px | feasible at %g: %s"
              " | eps* %.3f px | GT obs within eps (fixed R) %.1f%% | rot err of added cam %.3f deg"
              % (n, imgs[S[-1]].name, len(L), len(obs), res_c.max(), a.eps, ok, eps_star,
                 100 * (prem <= a.eps).mean(), rot_err[S[-1]]), flush=True)
        if ok:
            geo = {"step": n, "eps": a.eps, "landmarks": {}, "cameras": {}}
            A, b, Aeq, beq = sysA.build(a.eps)
            with Pool(a.procs, initializer=SA._init, initargs=(A, b, Aeq, beq, sysA.n)) as pool:
                ext = {v: (l_, h_) for v, l_, h_ in pool.imap_unordered(SA._extent, range(sysA.n), chunksize=8)}
            box = lambda v0: (np.array([ext[v0 + d][0] for d in range(3)]),
                              np.array([ext[v0 + d][1] for d in range(3)]))
            diag, cov = [], []
            for k in L:
                lo_, hi_ = box(sysA.lvar(k))
                dk = min(Rg[i][2] @ (Xg[k] - cg[i]) for i in by_k[k] if i in Sset)
                d = float(np.linalg.norm(hi_ - lo_)) / dk
                inn = int(bool(((Xs_[k] >= lo_ - 1e-9) & (Xs_[k] <= hi_ + 1e-9)).all())) if k in Xs_ else -1
                lm_track.setdefault(k, {})[n] = (d, inn)
                geo["landmarks"][int(k)] = {"lo": lo_.tolist(), "hi": hi_.tolist(), "colmap": Xg[k].tolist(),
                                            "gt": Xs_[k].tolist() if k in Xs_ else None, "gt_in_box": inn,
                                            "views": sorted(int(i) for i in by_k[k] if i in Sset)}
                diag.append(d)
                if inn >= 0:
                    cov.append(inn)
            diag = np.array(diag)
            row.update(box_diag_over_dk_median=float(np.median(diag[np.isfinite(diag)])),
                       bounded=float(np.isfinite(diag).mean()), gt_in_box=float(np.mean(cov)))
            print("   landmark boxes: bounded %.0f%%, diag/depth median %.4f, GT inside %.1f%%"
                  % (100 * np.isfinite(diag).mean(), np.median(diag[np.isfinite(diag)]), 100 * np.mean(cov)))
            for i in S:
                if i == r:
                    continue
                lo_, hi_ = box(sysA.cvar(i))
                inn = bool(((cs_[i] >= lo_ - 1e-9) & (cs_[i] <= hi_ + 1e-9)).all())
                geo["cameras"][imgs[i].name] = {"id": int(i), "lo": lo_.tolist(), "hi": hi_.tolist(),
                                                "R": Rg[i].tolist(), "colmap": cg[i].tolist(),
                                                "R_gt": Rs_[i].tolist(), "gt": cs_[i].tolist()}
                cam_track.setdefault(imgs[i].name, {})[n] = {
                    "lo": lo_.tolist(), "hi": hi_.tolist(), "size": (hi_ - lo_).tolist(),
                    "colmap": cg[i].tolist(), "gt": cs_[i].tolist(), "gt_inside": inn}
                print("   camera %-9s centre box size %s  (COLMAP %s, GT %s) GT inside: %s"
                      % (imgs[i].name, np.round(hi_ - lo_, 4), np.round(cg[i], 4), np.round(cs_[i], 4), inn))
        if ok:
            geo["cameras"][imgs[r].name] = {"id": int(r), "lo": [0, 0, 0], "hi": [0, 0, 0], "R": Rg[r].tolist(),
                                            "colmap": [0, 0, 0], "R_gt": Rs_[r].tolist(), "gt": cs_[r].tolist(),
                                            "reference": True}
            geo["intrinsics"] = {imgs[i].name: list(map(float, cams[imgs[i].camera_id].unpack())) for i in S}
            json.dump(geo, open(os.path.join(OUT, "geometry_%dimg.json" % n), "w"))
        summary.append(row)

    # landmarks of the initial pair, followed through the steps
    pair = [k for k, v in lm_track.items() if 2 in v]
    with open(os.path.join(OUT, "landmark_steps.csv"), "w", newline="") as fh:
        steps = sorted({n for v in lm_track.values() for n in v})
        w = csv.writer(fh)
        w.writerow(["point3D_id"] + ["diag_over_dk_%dimg" % n for n in steps] + ["gt_in_box_%dimg" % n for n in steps])
        for k in sorted(lm_track):
            v = lm_track[k]
            w.writerow([k] + [v[n][0] if n in v else "" for n in steps] + [v[n][1] if n in v else "" for n in steps])
    if pair:
        print("\nInitial-pair landmarks followed through the steps (%d):" % len(pair))
        for n in sorted({n for v in lm_track.values() for n in v}):
            d = np.array([lm_track[k][n][0] for k in pair if n in lm_track[k]])
            g = [lm_track[k][n][1] for k in pair if n in lm_track[k] and lm_track[k][n][1] >= 0]
            print("   %d images: diag/depth median %.4f  (p90 %.4f)   GT inside %.1f%%"
                  % (n, np.median(d[np.isfinite(d)]), np.quantile(d[np.isfinite(d)], 0.9), 100 * np.mean(g)))
    json.dump({"eps": a.eps, "order": [imgs[i].name for i in order], "gauge": {
        "reference": imgs[r].name, "scale_camera": imgs[s].name, "scale_axis": jj},
        "steps": summary, "cameras": cam_track}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
