"""Stage A of the joint feasible set: rotations fixed, centres and landmarks jointly unknown.

    python stage_a.py --scene airplane --eps 3 --n-images 30

Steps (protocol numbering):
  1-2  read the COLMAP model; observations T restricted to a connected subset of images,
       landmarks seen in >= 2 distinct subset images
  3    undistort detections (SIMPLE_RADIAL, fixed-point iteration); eps lives in the
       undistorted image
  4    COLMAP residuals in the undistorted image (is COLMAP a feasible witness?)
  5    gauge: reference camera r (R_r = I, c_r = 0), scale e_j^T c_s = 1
  6    four plane constraints + depth margin per observation:  A w <= b,  e_j^T c_s = 1
  7    feasibility at eps, and eps* (smallest feasible eps) by bisection
  8    per-coordinate min/max of every landmark and camera centre (6 LPs each), always on
       the full joint system
  12   certificate: ||X_hat_k - X*_k|| <= max over box corners of ||X_hat_k - v||
  13   GT in the same gauge by the gauge rule (no best-fit alignment): premise, coverage,
       tightness
Ground truth enters only in Step 13. X*_k is the median of the track's detections lifted
by the GT depth maps (same reference as the earlier experiments).
For comparison, the fixed-pose box (centres held at COLMAP's, i.e. V1) is computed too.
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
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project
import cp_conformal as CC


# --------------------------------------------------------------------------
def undistort(u, v, intr, iters=30):
    fx, fy, cx, cy, k1 = intr[0], intr[1], intr[2], intr[3], intr[4]
    xd = np.array([(u - cx) / fx, (v - cy) / fy])
    x = xd.copy()
    for _ in range(iters):
        x = xd / (1 + k1 * (x @ x))
    return x


def select_subset(imgs, n):
    """reference = most covisible image; greedily add the image sharing most landmarks."""
    sets = {i: set(int(p) for p in im.p3d_ids[im.p3d_ids >= 0]) for i, im in imgs.items()}
    cov = {i: sum(len(sets[i] & sets[j]) for j in imgs if j != i) for i in imgs}
    r = max(cov, key=cov.get)
    S, seen = [r], set(sets[r])
    while len(S) < n:
        j = max((j for j in imgs if j not in S), key=lambda j: len(sets[j] & seen))
        S.append(j)
        seen |= sets[j]
    return r, S


class System:
    """A w <= b, e_j^T c_s = 1 for given eps; variables: centres (cams != r), then landmarks."""

    def __init__(self, cams_R, obs, cam_index, lm_index, r, s, j, zmin):
        self.R, self.obs, self.ci, self.li = cams_R, obs, cam_index, lm_index
        self.r, self.s, self.j, self.zmin = r, s, j, zmin
        self.n = 3 * len(cam_index) + 3 * len(lm_index)

    def cvar(self, i):
        return None if i == self.r else 3 * self.ci[i]

    def lvar(self, k):
        return 3 * len(self.ci) + 3 * self.li[k]

    def build(self, eps):
        rows, cols, vals, b = [], [], [], []
        m = 0
        for (i, k, xn, f) in self.obs:
            Rm = self.R[i]
            r1, r2, r3 = Rm
            a_p, a_m = xn[0] + eps / f[0], xn[0] - eps / f[0]
            b_p, b_m = xn[1] + eps / f[1], xn[1] - eps / f[1]
            G = [r1 - a_p * r3, a_m * r3 - r1, r2 - b_p * r3, b_m * r3 - r2, -r3]
            rhs = [0, 0, 0, 0, -self.zmin]
            for g, h in zip(G, rhs):
                lv = self.lvar(k)
                rows += [m] * 3
                cols += [lv, lv + 1, lv + 2]
                vals += list(g)
                cv = self.cvar(i)
                if cv is not None:
                    rows += [m] * 3
                    cols += [cv, cv + 1, cv + 2]
                    vals += list(-g)
                b.append(h)
                m += 1
        A = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(m, self.n)))
        Aeq = np.zeros((1, self.n))
        Aeq[0, self.cvar(self.s) + self.j] = 1.0
        return A, np.array(b, float), csr_matrix(Aeq), np.array([1.0])

    def feasible(self, eps, tol=1e-8):
        """phase-1 LP: min t s.t. A w - t <= b, t >= 0; feasible iff t* <= tol.
        (dual simplex stalls on infeasible instances; IPM on this form is fast and robust)"""
        A, b, Aeq, beq = self.build(eps)
        A2 = hstack([A, -np.ones((A.shape[0], 1))]).tocsr()
        Aeq2 = hstack([Aeq, csr_matrix((1, 1))]).tocsr()
        c = np.zeros(self.n + 1)
        c[-1] = 1.0
        res = linprog(c, A_ub=A2, b_ub=b, A_eq=Aeq2, b_eq=beq,
                      bounds=[(None, None)] * self.n + [(0, None)], method="highs-ipm")
        return bool(res.status == 0 and res.fun <= tol), res


_G = {}


def _init(A, b, Aeq, beq, n):
    _G.update(A=A, b=b, Aeq=Aeq, beq=beq, n=n)


def _extent(var):
    """(min, max) of variable index var over the joint system; inf if unbounded."""
    out = []
    for sg in (1.0, -1.0):
        c = np.zeros(_G["n"])
        c[var] = sg
        res = linprog(c, A_ub=_G["A"], b_ub=_G["b"], A_eq=_G["Aeq"], b_eq=_G["beq"],
                      bounds=[(None, None)] * _G["n"], method="highs-ipm")
        if res.status == 0:
            out.append(sg * res.fun)
        elif res.status == 3:
            out.append(-sg * np.inf)
        else:
            out.append(np.nan)
    return var, out[0], out[1]


def fixed_pose_box(R_list, c_list, xns, fs, eps, zmin):
    """V1: centres and rotations fixed; box of the landmark's own polyhedron (6 LPs)."""
    A, b = [], []
    for Rm, c, xn, f in zip(R_list, c_list, xns, fs):
        r1, r2, r3 = Rm
        a_p, a_m = xn[0] + eps / f[0], xn[0] - eps / f[0]
        b_p, b_m = xn[1] + eps / f[1], xn[1] - eps / f[1]
        for g in (r1 - a_p * r3, a_m * r3 - r1, r2 - b_p * r3, b_m * r3 - r2):
            A.append(g)
            b.append(g @ c)
        A.append(-r3)
        b.append(-zmin - r3 @ c)
    A, b = np.array(A), np.array(b)
    lo, hi = np.zeros(3), np.zeros(3)
    for d in range(3):
        e = np.zeros(3)
        e[d] = 1
        r1 = linprog(e, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        r2 = linprog(-e, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
        lo[d] = r1.fun if r1.status == 0 else (-np.inf if r1.status == 3 else np.nan)
        hi[d] = -r2.fun if r2.status == 0 else (np.inf if r2.status == 3 else np.nan)
    return lo, hi


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "stage_a", "eps%g_n%d" % (a.eps, a.n_images))
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    log = {}

    # ---- Steps 1-3 ----------------------------------------------------------
    cams, imgs, pts = read_model(MODEL)
    r, S = select_subset(imgs, a.n_images)
    Sset = set(S)
    intr = {i: cams[imgs[i].camera_id].unpack() for i in S}
    obs_raw = []                                   # (i, k, u, v, j2d)
    for i in S:
        im = imgs[i]
        for j in np.flatnonzero(im.p3d_ids >= 0):
            obs_raw.append((i, int(im.p3d_ids[j]), float(im.xys[j][0]), float(im.xys[j][1]), int(j)))
    by_k = {}
    for o in obs_raw:
        by_k.setdefault(o[1], set()).add(o[0])
    L = sorted(k for k, v in by_k.items() if len(v) >= 2)
    Lset = set(L)
    obs_raw = [o for o in obs_raw if o[1] in Lset]
    print("subset: %d images (reference %s), %d landmarks, %d observations"
          % (len(S), imgs[r].name, len(L), len(obs_raw)), flush=True)

    # ---- Step 5: gauge ------------------------------------------------------
    Rr, cr = imgs[r].R, imgs[r].center
    Rg = {i: imgs[i].R @ Rr.T for i in S}
    cg = {i: Rr @ (imgs[i].center - cr) for i in S}
    s = max((i for i in S if i != r), key=lambda i: np.abs(cg[i]).max())
    jj = int(np.argmax(np.abs(cg[s])))
    scale = cg[s][jj]
    cg = {i: cg[i] / scale for i in S}
    Xg = {k: Rr @ (pts[k].xyz - cr) / scale for k in L}
    depths = [Rg[o[0]][2] @ (Xg[o[1]] - cg[o[0]]) for o in obs_raw]
    zmin = 1e-3 * float(np.median(depths))
    log["gauge"] = {"reference": imgs[r].name, "scale_camera": imgs[s].name, "scale_axis": jj,
                    "z_min": zmin, "min_colmap_depth": float(min(depths))}

    # ---- Step 3 + 4: undistorted coordinates, COLMAP residuals ---------------
    obs, res_colmap = [], []
    for (i, k, u, v, j2d) in obs_raw:
        xn = undistort(u, v, intr[i])
        f = (intr[i][0], intr[i][1])
        obs.append((i, k, xn, f))
        q = Rg[i] @ (Xg[k] - cg[i])
        res_colmap.append(max(abs(f[0] * (q[0] / q[2] - xn[0])), abs(f[1] * (q[1] / q[2] - xn[1]))))
    res_colmap = np.array(res_colmap)
    log["step4_colmap_residual_undistorted_px"] = {
        "max": float(res_colmap.max()), "p99": float(np.quantile(res_colmap, 0.99)),
        "n_over_eps": int((res_colmap > a.eps).sum()), "witness_at_eps": bool(res_colmap.max() <= a.eps)}
    print("Step 4: COLMAP max residual %.3f px, %d obs > eps=%g -> witness: %s"
          % (res_colmap.max(), (res_colmap > a.eps).sum(), a.eps, res_colmap.max() <= a.eps), flush=True)

    cam_index = {i: n for n, i in enumerate(i for i in S if i != r)}
    lm_index = {k: n for n, k in enumerate(L)}
    sysA = System(Rg, obs, cam_index, lm_index, r, s, jj, zmin)

    # ---- Step 7: feasibility and eps* -----------------------------------------
    t = time.time()
    ok, _ = sysA.feasible(a.eps)
    print("Step 7: feasible at eps=%g: %s  (phase-1 t* = %.3g, %.1fs per LP)"
          % (a.eps, ok, _.fun, time.time() - t), flush=True)
    lo, hi = 0.0, float(res_colmap.max()) * 1.0001
    assert sysA.feasible(hi)[0], "COLMAP solution should be feasible at its own max residual"
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        if sysA.feasible(mid)[0]:
            hi = mid
        else:
            lo = mid
    eps_star = hi
    print("        eps* = %.4f px (L_inf-optimal with rotations fixed)" % eps_star, flush=True)
    log["step7"] = {"feasible_at_eps": ok, "eps_star_px": eps_star}

    # ---- Step 13 setup: GT in the same gauge (rule, not fit) ----------------
    gt = CC.load_gt(SCENE)
    from cp_step0_align import umeyama
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))
    rows = CC.observations(cams, imgs, pts, gt, s_, Ra, ta)       # only Xgt (GT frame) is used
    lifted = {}
    for rw in rows:
        if rw["valid"]:
            lifted.setdefault(rw["pid"], []).append(rw["Xgt"])
    gr = gt[imgs[r].name]
    Rr_s, cr_s = gr["R"], gr["c"]
    Rs_ = {i: gt[imgs[i].name]["R"] @ Rr_s.T for i in S}
    cs_ = {i: Rr_s @ (gt[imgs[i].name]["c"] - cr_s) for i in S}
    sc_s = cs_[s][jj]
    cs_ = {i: cs_[i] / sc_s for i in S}
    Xs_ = {k: Rr_s @ (np.median(lifted[k], 0) - cr_s) / sc_s for k in L if k in lifted}
    rot_err = {i: float(np.degrees(Rotation.from_matrix(Rs_[i] @ Rg[i].T).magnitude())) for i in S}

    # premise: (a) with true rotations and GT intrinsics vs raw detections (protocol 13.2)
    #          (b) the Stage A constraints themselves (fixed R_hat, COLMAP intrinsics)
    prem_true, prem_A = {}, {}
    for (i, k, u, v, j2d), (_, _, xn, f) in zip(obs_raw, obs):
        if k not in Xs_:
            continue
        g = gt[imgs[i].name]
        pc = g["K"] @ (g["R"] @ (np.median(lifted[k], 0) - g["c"]))
        e_true = float(np.abs(pc[:2] / pc[2] - [u, v]).max())
        q = Rg[i] @ (Xs_[k] - cs_[i])
        e_A = max(abs(f[0] * (q[0] / q[2] - xn[0])), abs(f[1] * (q[1] / q[2] - xn[1]))) if q[2] > 0 else np.inf
        prem_true.setdefault(k, []).append(e_true)
        prem_A.setdefault(k, []).append(e_A)
    allt = np.concatenate([v for v in prem_true.values()])
    allA = np.concatenate([v for v in prem_A.values()])
    log["step13_premise"] = {
        "observations_with_GT": int(len(allt)),
        "true_rotations_frac_obs_within_eps": float((allt <= a.eps).mean()),
        "true_rotations_max_px": float(allt.max()),
        "stageA_constraints_frac_obs_within_eps": float((allA <= a.eps).mean()),
        "stageA_constraints_max_px": float(allA.max()),
        "premise_holds_for_whole_subset": bool(allA.max() <= a.eps),
        "rotation_error_deg_median_max": [float(np.median(list(rot_err.values()))),
                                          float(max(rot_err.values()))]}
    print("Step 13 premise: %.1f%% of obs within eps with true rotations (max %.1f px); "
          "%.1f%% satisfy the Stage A constraints (max %.1f px); rotation error median %.3f deg"
          % (100 * (allt <= a.eps).mean(), allt.max(), 100 * (allA <= a.eps).mean(), allA.max(),
             np.median(list(rot_err.values()))), flush=True)

    # ---- Step 8: boxes (only if feasible) -------------------------------------
    if not ok:
        print("infeasible at eps=%g: no sets. Rerun with eps >= eps* = %.3f" % (a.eps, eps_star))
    else:
        A, b, Aeq, beq = sysA.build(a.eps)
        variables = list(range(sysA.n))
        print("Step 8: %d LPs on %d workers ..." % (2 * len(variables), a.procs), flush=True)
        t = time.time()
        with Pool(a.procs, initializer=_init, initargs=(A, b, Aeq, beq, sysA.n)) as pool:
            ext = {}
            for n_done, (var, l_, h_) in enumerate(pool.imap_unordered(_extent, variables, chunksize=4), 1):
                ext[var] = (l_, h_)
                if n_done % 500 == 0:
                    el = time.time() - t
                    print("  %d / %d variables  (%.0fs, eta %.0fs)"
                          % (n_done, len(variables), el, el / n_done * (len(variables) - n_done)), flush=True)
        log["step8_seconds"] = round(time.time() - t)

        def box(v0):
            return (np.array([ext[v0 + d][0] for d in range(3)]),
                    np.array([ext[v0 + d][1] for d in range(3)]))

        # landmarks
        lm_rows = []
        for k in L:
            lo_, hi_ = box(sysA.lvar(k))
            Xh = Xg[k]
            dk = min(Rg[i][2] @ (Xh - cg[i]) for i in by_k[k] if i in Sset)
            diag = float(np.linalg.norm(hi_ - lo_))
            cert = float(np.linalg.norm(np.maximum(np.abs(Xh - lo_), np.abs(hi_ - Xh))))
            # V1 comparison: own constraints, centres fixed at COLMAP's
            mine = [o for o in obs if o[1] == k]
            vlo, vhi = fixed_pose_box([Rg[o[0]] for o in mine], [cg[o[0]] for o in mine],
                                      [o[2] for o in mine], [o[3] for o in mine], a.eps, zmin)
            row = {"point3D_id": k, "n_views_subset": len(by_k[k] & Sset), "d_k": dk,
                   "box_diag": diag, "box_diag_over_dk": diag / dk,
                   "fixedpose_box_diag_over_dk": float(np.linalg.norm(vhi - vlo)) / dk,
                   "certificate": cert, "certificate_over_dk": cert / dk}
            if k in Xs_:
                Xt = Xs_[k]
                row.update(has_gt=1, gt_in_box=int(bool(((Xt >= lo_ - 1e-9) & (Xt <= hi_ + 1e-9)).all())),
                           gt_in_fixedpose_box=int(bool(((Xt >= vlo - 1e-9) & (Xt <= vhi + 1e-9)).all())),
                           own_obs_premise_true=int(max(prem_true[k]) <= a.eps),
                           own_obs_satisfy_stageA=int(max(prem_A[k]) <= a.eps),
                           colmap_error=float(np.linalg.norm(Xh - Xt)),
                           colmap_error_over_certificate=float(np.linalg.norm(Xh - Xt)) / cert)
            else:
                row.update(has_gt=0)
            lm_rows.append(row)
        cam_rows = []
        for i in S:
            if i == r:
                lo_ = hi_ = np.zeros(3)
            else:
                lo_, hi_ = box(sysA.cvar(i))
            ct = cs_[i]
            cam_rows.append({"image": imgs[i].name, "reference": int(i == r), "scale_cam": int(i == s),
                             "box_diag": float(np.linalg.norm(hi_ - lo_)),
                             "gt_in_box": int(bool(((ct >= lo_ - 1e-9) & (ct <= hi_ + 1e-9)).all())),
                             "colmap_centre_error": float(np.linalg.norm(cg[i] - ct)),
                             "certificate": float(np.linalg.norm(np.maximum(np.abs(cg[i] - lo_),
                                                                            np.abs(hi_ - cg[i])))),
                             "rotation_error_deg": rot_err[i]})
        for name, rr in (("landmarks.csv", lm_rows), ("cameras.csv", cam_rows)):
            keys = sorted({k for x in rr for k in x}, key=lambda k: list(rr[0]).index(k) if k in rr[0] else 99)
            with open(os.path.join(OUT, name), "w", newline="") as fh:
                w = csv.DictWriter(fh, keys)
                w.writeheader()
                w.writerows(rr)

        G = [x for x in lm_rows if x["has_gt"]]
        fin = np.array([x["box_diag_over_dk"] for x in lm_rows])
        finite = fin[np.isfinite(fin)]
        grp = lambda key, val: [x for x in G if x[key] == val]
        cov = lambda xs, key="gt_in_box": (float(np.mean([x[key] for x in xs])) if xs else float("nan"), len(xs))
        log["step11_sets"] = {
            "landmarks": len(L), "bounded_fraction": float(np.isfinite(fin).mean()),
            "box_diag_over_dk_median": float(np.median(finite)),
            "fixedpose_box_diag_over_dk_median": float(np.nanmedian(
                [x["fixedpose_box_diag_over_dk"] for x in lm_rows if np.isfinite(x["fixedpose_box_diag_over_dk"])])),
            "camera_box_diag_median": float(np.median([c["box_diag"] for c in cam_rows if not c["reference"]]))}
        log["step13_coverage"] = {
            "landmarks_all": cov(G),
            "landmarks_own_obs_satisfy_stageA": cov(grp("own_obs_satisfy_stageA", 1)),
            "landmarks_own_obs_violate_stageA": cov(grp("own_obs_satisfy_stageA", 0)),
            "landmarks_fixedpose_box_all": cov(G, "gt_in_fixedpose_box"),
            "cameras": (float(np.mean([c["gt_in_box"] for c in cam_rows if not c["reference"]])),
                        len(cam_rows) - 1)}
        ratio = np.array([x["colmap_error_over_certificate"] for x in G if np.isfinite(x["colmap_error_over_certificate"])])
        log["step13_tightness"] = {"colmap_error_over_certificate_median": float(np.median(ratio)),
                                   "certificate_over_dk_median": float(np.median(
                                       [x["certificate_over_dk"] for x in lm_rows if np.isfinite(x["certificate_over_dk"])]))}
        for k_, v_ in log.items():
            if k_.startswith("step1") and k_ != "step13_premise":
                print(k_, json.dumps(v_), flush=True)

    log["eps"] = a.eps
    log["subset_images"] = [imgs[i].name for i in S]
    log["seconds"] = round(time.time() - T0)
    json.dump(log, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("-> %s  (%ds)" % (OUT, time.time() - T0))


if __name__ == "__main__":
    main()
