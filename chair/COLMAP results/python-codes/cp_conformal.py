"""Split-conformal calibration of eps and containment validation (Steps 0-9).

    python cp_conformal.py --scene airplane

F_k is computed with exact_polytope.halfspaces / chebyshev_center unchanged.
For speed the sweep over (seed, alpha, score, scheme) uses two per-landmark
thresholds that are exact consequences of that construction:

    eps_star_k : smallest eps with F_k(eps) non-empty  (bisection on the LP)
    cont_k     : smallest eps whose half-spaces all hold at X*_k
                 = max_i f*gain_i*|n_i(X*_k) - n*_i|_inf

so "empty" <=> eps < eps_star_k and "contains" <=> eps >= cont_k.  Both are
cross-checked against the literal LP + half-space test for seed 0 (reported).

Definitions:
    n_k = number of DISTINCT images observing k       (strata, protocol's n_k)
    m_k = number of observations = constraint blocks  (Bonferroni, theory)
Leave-one-out drops EVERY observation of the held-out image (near-duplicate
SIFT keypoints in one image would otherwise leak into the reference).
"""
import os
import sys
import csv
import json
import math
import time
import argparse
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project
from cp_step0_align import umeyama
import exact_polytope as ex

FLIP = np.diag([1.0, -1.0, -1.0])
ALPHAS = [0.01, 0.05, 0.10, 0.20]
SEEDS = list(range(20))
DISC = 0.02
SPREAD_FLAG = 0.05
W = H = 800


# ======================================================================
# data
# ======================================================================
def load_gt(scene):
    """name-as-registered-in-COLMAP -> GT pose, intrinsics, depth path.

    Handles both dataset layouts:
      * chair / airplane : transforms_train.json + transforms_val.json, the val
        split renamed with a '_val' suffix by prepare_images.py
      * office           : a single transforms.json, images used as-is
    """
    global W, H
    gt = {}
    single = os.path.join(scene, "transforms.json")
    layouts = ([(single, "")] if os.path.exists(single)
               else [(os.path.join(scene, "transforms_%s.json" % sp), suf)
                     for sp, suf in (("train", ""), ("val", "_val"))])
    for path, suf in layouts:
        J = json.load(open(path))
        W = H = J.get("w", W)
        f = J.get("fl_x", 0.5 * W / math.tan(0.5 * J["camera_angle_x"]))
        cx, cy = J.get("cx", W / 2), J.get("cy", H / 2)
        for fr in J["frames"]:
            M = np.array(fr["transform_matrix"], float)
            name = os.path.basename(fr["file_path"]) + suf + ".png"
            gt[name] = dict(R=(M[:3, :3] @ FLIP).T, c=M[:3, 3].copy(),
                            K=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]),
                            depth=os.path.join(scene, fr["depth_path"][2:]),
                            R_noflip=M[:3, :3].T)
    return gt


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def conformal_eps(sorted_cal, alpha):
    N = len(sorted_cal)
    idx = math.ceil((N + 1) * (1 - alpha))
    return float(sorted_cal[idx - 1]) if idx <= N else float("inf")


# ======================================================================
# Step 0
# ======================================================================
def step0(cams, imgs, pts, gt, s, Ra, ta):
    rep = {}
    names = [im.name for im in imgs.values()]
    rep["name_mapping"] = {"colmap_images": len(names),
                           "mapped_by_filename": sum(n in gt for n in names)}
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}
    worst = 0.0
    for p in pts.values():
        r = [np.linalg.norm(project(*pose[i][:3], p.xyz)[0][0] - imgs[i].xys[j])
             for i, j in zip(p.image_ids, p.point2D_idxs)]
        worst = max(worst, abs(np.mean(r) - p.error))
    rep["error_column_regression_px"] = worst

    ang = lambda Q: math.degrees(math.acos(np.clip((np.trace(Q) - 1) / 2, -1, 1)))
    for lab, key in (("with_flip", "R"), ("without_flip", "R_noflip")):
        e = [ang(imgs[i].R @ (gt[imgs[i].name][key] @ Ra.T).T) for i in imgs]
        rep["rotation_err_deg_" + lab] = [float(np.median(e)), float(np.max(e))]

    # depth convention: back-project view A as z-depth vs ray distance, read in view B
    names = sorted(gt)[:3] + sorted(gt)[100:103]
    rng = np.random.default_rng(0)
    res = {"z": [], "ray": []}
    for a, b in zip(names[:-1], names[1:]):
        A, B = gt[a], gt[b]
        Da, Db = np.load(A["depth"])["depth"], np.load(B["depth"])["depth"]
        r, c = np.nonzero(Da > 0)
        k = rng.choice(r.size, 5000, replace=False)
        uv1 = np.stack([c[k] + 0.5, r[k] + 0.5, np.ones(5000)], 1)
        ray = uv1 @ np.linalg.inv(A["K"]).T
        z = Da[r[k], c[k]]
        for mode in ("z", "ray"):
            Xc = ray * z[:, None] if mode == "z" else \
                ray / np.linalg.norm(ray, axis=1, keepdims=True) * z[:, None]
            Xw = A["c"] + Xc @ A["R"]
            Xb = (Xw - B["c"]) @ B["R"].T
            p = Xb @ B["K"].T
            ub, vb = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
            ok = (ub > 1) & (ub < W - 1) & (vb > 1) & (vb < H - 1)
            db = Db[(vb[ok] - 0.5).round().astype(int), (ub[ok] - 0.5).round().astype(int)]
            vis = (db > 0) & (np.abs(Xb[ok, 2] - db) < 0.05 * db)
            res[mode].append(np.abs(Xb[ok, 2][vis] - db[vis]))
    rep["depth_convention_check"] = {
        m: {"median_abs": float(np.median(np.concatenate(v)))} for m, v in res.items()}
    rep["depth_convention"] = "z-depth" if (
        rep["depth_convention_check"]["z"]["median_abs"] <
        rep["depth_convention_check"]["ray"]["median_abs"]) else "ray"
    rep["depth_units"] = "GT scene units (depth ray-cast from the mesh in the pose frame)"
    rep["depth_background"] = "0.0 marks background / ray miss; treated as invalid"
    return rep


# ======================================================================
# Steps 2-4
# ======================================================================
def observations(cams, imgs, pts, gt, s, Ra, ta):
    rows = []
    for iid, im in imgs.items():
        g = gt[im.name]
        D = np.load(g["depth"])["depth"]
        K = g["K"]
        for j in np.flatnonzero(im.p3d_ids >= 0):
            pid = int(im.p3d_ids[j])
            u, v = im.xys[j]
            x, y = u - 0.5, v - 0.5
            x0, y0 = int(math.floor(x)), int(math.floor(y))
            row = dict(pid=pid, iid=iid, p2d=int(j), u=u, v=v, valid=0, reason="",
                       Xgt=None, Xs=None)
            if x0 < 0 or y0 < 0 or x0 + 1 >= W or y0 + 1 >= H:
                row["reason"] = "out_of_bounds"
                rows.append(row)
                continue
            q = np.array([D[y0, x0], D[y0, x0 + 1], D[y0 + 1, x0], D[y0 + 1, x0 + 1]])
            if (q <= 0).any():
                row["reason"] = "background"
                rows.append(row)
                continue
            ax, ay = x - x0, y - y0
            d = (1 - ay) * ((1 - ax) * q[0] + ax * q[1]) + ay * ((1 - ax) * q[2] + ax * q[3])
            if (q.max() - q.min()) / d > DISC:
                row["reason"] = "discontinuity"
                rows.append(row)
                continue
            ray = np.linalg.solve(K, [u, v, 1.0])
            Xgt = g["c"] + g["R"].T @ (d * ray)
            row.update(valid=1, Xgt=Xgt, Xs=s * Ra @ Xgt + ta)
            rows.append(row)
    return rows


def references_and_scores(rows, cams, imgs, pts, gt):
    by = {}
    for n, r in enumerate(rows):
        by.setdefault(r["pid"], []).append(n)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}
    lm = {}
    for pid, idx in by.items():
        V = [n for n in idx if rows[n]["valid"]]
        Xk = np.median([rows[n]["Xs"] for n in V], 0) if V else None
        info = dict(Xstar=Xk, n_valid=len(V))
        if Xk is not None:
            spread = max(np.linalg.norm(rows[n]["Xs"] - Xk) for n in V)
            depths = [(imgs[i].R @ Xk + imgs[i].tvec)[2] for i in set(rows[n]["iid"] for n in idx)]
            dk = min(d for d in depths if d > 0) if any(d > 0 for d in depths) else float("nan")
            info.update(spread=spread, d_k=dk, flag=int(spread / dk > SPREAD_FLAG))
        else:
            info.update(spread=float("nan"), d_k=float("nan"), flag=0)
        lm[pid] = info
        for n in idx:
            r = rows[n]
            r["s_tot"] = r["s_det"] = float("nan")
            if not r["valid"]:
                continue
            others = [m for m in V if rows[m]["iid"] != r["iid"]]
            if not others:
                r["reason"] = "no_reference"
                continue
            ref_s = np.median([rows[m]["Xs"] for m in others], 0)
            ref_g = np.median([rows[m]["Xgt"] for m in others], 0)
            R, t, intr = pose[r["iid"]]
            uv = project(R, t, intr, ref_s)[0][0]
            r["s_tot"] = float(np.abs(np.array([r["u"], r["v"]]) - uv).max())
            g = gt[imgs[r["iid"]].name]
            pc = g["K"] @ (g["R"] @ (ref_g - g["c"]))
            r["s_det"] = float(np.abs(np.array([r["u"], r["v"]]) - pc[:2] / pc[2]).max())
    return lm


# ======================================================================
# per-landmark thresholds (exact consequences of exact_polytope)
# ======================================================================
class Landmark:
    def __init__(self, p, imgs, pose):
        self.views = [(*pose[i], imgs[i].xys[j]) for i, j in zip(p.image_ids, p.point2D_idxs)]
        m = len(self.views)
        self.R = np.array([v[0] for v in self.views])
        self.t = np.array([v[1] for v in self.views])
        self.f = np.array([[v[2][0], v[2][1]] for v in self.views])
        ns = [ex.undistort(np.asarray(v[3], float), v[2]) for v in self.views]
        self.nstar = np.array([a for a, _ in ns])
        self.gain = np.array([b for _, b in ns])
        self.m = m

    def H(self, eps):
        """Same rows, same order as exact_polytope.halfspaces(views, eps)."""
        e = eps / (self.f * self.gain[:, None])
        blocks = []
        for a in (0, 1):
            for sg in (1.0, -1.0):
                lim = self.nstar[:, a] + sg * e[:, a]
                row = sg * (self.R[:, a, :] - lim[:, None] * self.R[:, 2, :])
                off = sg * (self.t[:, a] - lim * self.t[:, 2])
                blocks.append(np.hstack([row, off[:, None]]))
        blocks.append(np.hstack([-self.R[:, 2, :], (-self.t[:, 2] + 1e-3)[:, None]]))
        return np.stack(blocks, 1).reshape(-1, 4)

    def cont(self, X):
        """Smallest eps for which every half-space holds at X."""
        Xc = np.einsum("mij,j->mi", self.R, X) + self.t
        if (Xc[:, 2] <= 1e-3).any():
            return float("inf")
        n = Xc[:, :2] / Xc[:, 2:3]
        return float((np.abs(n - self.nstar) * self.f * self.gain[:, None]).max())

    def nonempty(self, eps):
        c, r = ex.chebyshev_center(self.H(eps))
        return c is not None and r > 0

    def eps_star(self, X_hat, iters=28):
        hi = self.cont(X_hat) * (1 + 1e-9) + 1e-12
        k = 0
        while not self.nonempty(hi) and k < 6:
            hi *= 2
            k += 1
        lo = 0.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.nonempty(mid):
                hi = mid
            else:
                lo = mid
        return hi


def diameter(L, eps):
    """(diameter, empty, H).  chebyshev_center returns None for ANY LP failure,
    which conflates 'infeasible' (empty) with 'unbounded' (nearly parallel rays
    whose cones never close at large eps).  Separate the two here; F_k itself
    is unchanged."""
    H_ = ex.halfspaces(L.views, eps)
    c, r = ex.chebyshev_center(H_)
    if c is None:
        feas = linprog(np.zeros(3), A_ub=H_[:, :3], b_ub=-H_[:, 3],
                       bounds=[(None, None)] * 3, method="highs")
        if feas.status == 0:
            return float("inf"), False, H_          # non-empty, unbounded
        return float("nan"), True, H_
    if r <= 0:
        return float("nan"), True, H_
    V, _, _ = ex.vertices_and_volume(H_, c)
    return float(np.linalg.norm(V[:, None] - V[None], axis=-1).max()), False, H_


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "conformal")
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()

    cams, imgs, pts = read_model(MODEL)
    gt = load_gt(SCENE)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}

    # ---------------- Step 1 ----------------
    ids = [i for i in imgs if imgs[i].name in gt]
    ch = np.array([imgs[i].center for i in ids])
    cg = np.array([gt[imgs[i].name]["c"] for i in ids])
    s, Ra, ta = umeyama(cg, ch)
    e = np.linalg.norm(ch - (s * cg @ Ra.T + ta), axis=1)
    rad = float(np.median(np.linalg.norm(ch - ch.mean(0), axis=1)))
    align = dict(n=len(ids), scale=float(s), rmse=float(np.sqrt((e ** 2).mean())),
                 median_radius=rad, max=float(e.max()))
    align["rmse_normalised"] = align["rmse"] / rad
    rep = {"step1_alignment": align}
    print("alignment RMSE %.4f  normalised %.3f%%" % (align["rmse"], 100 * align["rmse_normalised"]))
    if align["rmse_normalised"] > 0.01:
        json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=2)
        sys.exit("STOP: normalised alignment RMSE exceeds 1%")

    rep["step0"] = step0(cams, imgs, pts, gt, s, Ra, ta)
    print("step0:", json.dumps(rep["step0"]))

    # ---------------- Steps 2-4 ----------------
    rows = observations(cams, imgs, pts, gt, s, Ra, ta)
    lm = references_and_scores(rows, cams, imgs, pts, gt)
    reasons = {}
    for r in rows:
        key = "valid_scored" if (r["valid"] and not r["reason"]) else (r["reason"] or "valid")
        reasons[key] = reasons.get(key, 0) + 1
    rep["observations"] = {"total": len(rows), **reasons}
    print("observations:", rep["observations"], " %.0fs" % (time.time() - T0))

    # ---------------- per-landmark thresholds ----------------
    L = {}
    thr = {}
    t1 = time.time()
    for n, (pid, p) in enumerate(pts.items()):
        Lk = Landmark(p, imgs, pose)
        L[pid] = Lk
        Xs = lm[pid]["Xstar"]
        rmax = float(max(np.abs(project(*pose[i][:3], p.xyz)[0][0] - imgs[i].xys[j]).max()
                         for i, j in zip(p.image_ids, p.point2D_idxs)))
        thr[pid] = dict(n_k=len(set(p.image_ids.tolist())), m_k=Lk.m, r_max=rmax,
                        eps_star=Lk.eps_star(p.xyz),
                        cont=Lk.cont(Xs) if Xs is not None else float("nan"))
        if (n + 1) % 500 == 0:
            print("  thresholds %d/%d  %.0fs" % (n + 1, len(pts), time.time() - t1), flush=True)
    # regression: vectorised H must equal the existing implementation
    chk = [pid for pid in list(pts)[:50]]
    rep["H_matches_exact_polytope"] = bool(all(
        np.allclose(L[pid].H(1.0), ex.halfspaces(L[pid].views, 1.0), atol=1e-12) for pid in chk))

    # ---------------- cp_scores.csv (split column = seed 0) ----------------
    pids = np.array(sorted(pts))
    split0 = {}
    perm = np.random.default_rng(0).permutation(pids)
    for q, pid in enumerate(perm):
        split0[int(pid)] = "cal" if q < len(perm) // 2 else "test"
    with open(os.path.join(OUT, "cp_scores.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point3D_id", "image_id", "point2D_idx", "n_k", "m_k", "valid",
                    "reason", "s_tot", "s_det", "split_seed0", "track_flag"])
        for r in rows:
            w.writerow([r["pid"], r["iid"], r["p2d"], thr[r["pid"]]["n_k"], thr[r["pid"]]["m_k"],
                        r["valid"], r["reason"] or ("ok" if r["valid"] else ""),
                        r["s_tot"], r["s_det"], split0[r["pid"]], lm[r["pid"]]["flag"]])

    # ---------------- Steps 5-8 ----------------
    obs_pid = np.array([r["pid"] for r in rows])
    scorable = np.array([np.isfinite(r["s_tot"]) for r in rows])
    S = {"tot": np.array([r["s_tot"] for r in rows]), "det": np.array([r["s_det"] for r in rows])}
    m_obs = np.array([thr[r["pid"]]["m_k"] for r in rows])
    flag_obs = np.array([lm[r["pid"]]["flag"] for r in rows])
    has_ref = {pid for pid in pids if lm[pid]["Xstar"] is not None}
    flag = {pid: lm[pid]["flag"] for pid in pids}
    rep["landmarks"] = {"total": len(pids), "with_GT_reference": len(has_ref),
                        "no_valid_observation": len(pids) - len(has_ref),
                        "flagged_inconsistent": int(sum(flag[p] for p in has_ref))}

    cal_rows, cov_rows = [], []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(pids)
        cal = set(perm[: len(perm) // 2].tolist())
        for subset in ("all", "unflagged"):
            keep = (lambda p: True) if subset == "all" else (lambda p: not flag[p])
            is_cal = np.array([p in cal and keep(p) for p in obs_pid]) & scorable
            is_test = np.array([(p not in cal) and keep(p) for p in obs_pid]) & scorable
            test_lm = [p for p in pids if p not in cal and p in has_ref and keep(p)]
            for sc in ("tot", "det"):
                srt = np.sort(S[sc][is_cal])
                N = len(srt)
                for al in ALPHAS:
                    e_naive = conformal_eps(srt, al)
                    cal_rows.append([sc, al, "naive", "", e_naive, N, seed, subset])
                    mvals = sorted(set(thr[p]["m_k"] for p in pids))
                    e_bonf = {mv: conformal_eps(srt, al / mv) for mv in mvals}
                    for mv in mvals:
                        cal_rows.append([sc, al, "bonferroni", mv, e_bonf[mv], N, seed, subset])
                    # observation level
                    st = S[sc][is_test]
                    for scheme in ("naive", "bonferroni"):
                        if scheme == "naive":
                            ok = st <= e_naive
                            nominal = 1 - al
                        else:
                            ev = np.array([e_bonf[mv] for mv in m_obs[is_test]])
                            ok = st <= ev
                            nominal = float(np.mean(1 - al / m_obs[is_test]))
                        p_, lo, hi = wilson(int(ok.sum()), len(ok))
                        cov_rows.append([sc, al, scheme, "observation", p_, lo, hi, nominal,
                                         "", "", "", "", seed, subset, len(ok), N])
                        # landmark level
                        eps_k = np.array([e_naive if scheme == "naive" else e_bonf[thr[p]["m_k"]]
                                          for p in test_lm])
                        es = np.array([thr[p]["eps_star"] for p in test_lm])
                        cn = np.array([thr[p]["cont"] for p in test_lm])
                        mk = np.array([thr[p]["m_k"] for p in test_lm])
                        empty = eps_k < es
                        contains = eps_k >= cn
                        p_, lo, hi = wilson(int(contains.sum()), len(contains))
                        if scheme == "naive":
                            th_lo = float(np.mean(np.maximum(0, 1 - mk * al)))
                            th_ind = float(np.mean((1 - al) ** mk))
                            th_hi = 1 - al
                        else:
                            th_lo, th_ind, th_hi = 1 - al, float(np.mean((1 - al / mk) ** mk)), 1.0
                        cov_rows.append([sc, al, scheme, "landmark", p_, lo, hi,
                                         1 - al,
                                         th_lo, th_ind, th_hi, float(empty.mean()),
                                         seed, subset, len(contains), N])
    with open(os.path.join(OUT, "cp_calibration.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "alpha", "scheme", "m_k", "eps", "N", "seed", "subset"])
        w.writerows(cal_rows)
    with open(os.path.join(OUT, "cp_coverage.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "alpha", "scheme", "level", "coverage", "wilson_lo", "wilson_hi",
                    "nominal", "theory_lower", "theory_independent", "theory_upper",
                    "empty_rate", "seed", "subset", "n_test", "N_cal"])
        w.writerows(cov_rows)
    print("sweep done  %.0fs" % (time.time() - T0), flush=True)

    # ---------------- seed 0: literal LP cross-check, diameters, cp_landmarks.csv ----
    perm = np.random.default_rng(0).permutation(pids)
    cal = set(perm[: len(perm) // 2].tolist())
    test_lm = [p for p in pids if p not in cal and p in has_ref]
    is_cal = np.array([p in cal for p in obs_pid]) & scorable
    mis, tried = 0, 0
    dcache = {}
    lm_rows = []
    for sc in ("tot", "det"):
        srt = np.sort(S[sc][is_cal])
        for al in ALPHAS:
            for scheme in ("naive", "bonferroni"):
                for p in test_lm:
                    th = thr[p]
                    eps = conformal_eps(srt, al if scheme == "naive" else al / th["m_k"])
                    key = (p, round(eps, 12))
                    if key not in dcache:
                        if math.isinf(eps):
                            dcache[key] = (float("nan"), False, None)
                        else:
                            dcache[key] = diameter(L[p], eps)
                    dia, emp, H_ = dcache[key]
                    if H_ is not None:
                        Xs = lm[p]["Xstar"]
                        lit_c = bool((H_[:, :3] @ Xs + H_[:, 3] <= 0).all())
                        tried += 1
                        mis += (emp != (eps < th["eps_star"])) + (lit_c != (eps >= th["cont"]))
                    lm_rows.append([sc, al, scheme, p, th["n_k"], th["m_k"], th["r_max"], eps,
                                    (dia / lm[p]["d_k"] if np.isfinite(dia) else
                                     ("inf" if dia == float("inf") else "")),
                                    int(eps < th["eps_star"]), int(eps >= th["cont"]),
                                    lm[p]["flag"], lm[p]["spread"] / lm[p]["d_k"]])
    n_unb = sum(1 for v in dcache.values() if v[0] == float("inf"))
    rep["literal_LP_crosscheck_seed0"] = {"cases": tried, "disagreements": int(mis),
                                          "unbounded_sets": n_unb}
    with open(os.path.join(OUT, "cp_landmarks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "alpha", "scheme", "point3D_id", "n_k", "m_k", "r_max_px", "eps",
                    "diam_over_dk", "empty", "contains_gt", "track_flag", "spread_over_dk"])
        w.writerows(lm_rows)

    np.savez(os.path.join(OUT, "thresholds.npz"),
             pid=pids, n_k=[thr[p]["n_k"] for p in pids], m_k=[thr[p]["m_k"] for p in pids],
             r_max=[thr[p]["r_max"] for p in pids], eps_star=[thr[p]["eps_star"] for p in pids],
             cont=[thr[p]["cont"] for p in pids],
             flag=[lm[p]["flag"] for p in pids], has_ref=[p in has_ref for p in pids])
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=float)
    print(json.dumps(rep, indent=1, default=float))
    print("total %.0fs" % (time.time() - T0))


if __name__ == "__main__":
    main()
