"""Baseline 1: how far can camera 3 move before a reprojection leaves its eps box?

    python baseline1_pose.py --scene airplane --eps 3 --image 17

COLMAP's pose theta_hat = (R_hat, c_hat) and COLMAP's 3D points X_hat_k are held fixed (no point
uncertainty). Pose perturbation: R = exp([w]x) R_hat, c = c_hat + dc, theta = (w, dc) in R^6.
Residual of observation k: r_k(theta) = pi(theta, X_hat_k) - y_k (COLMAP camera model, with
distortion; pixels). Gauge as before: camera #162 at the origin with R = I, x of #15's centre = 1.

(a) EXACT, one parameter at a time: for each parameter j and sign s, the largest t with
    max_k |r_k(theta_hat + s t e_j)|_inf <= eps (bisection on real reprojections). Needs a feasible
    start, so it uses the observations that already satisfy |r_k(theta_hat)|_inf <= eps.
(b) JACOBIAN H_k = d pi / d theta at theta_hat (2x6 per observation, as in Eq. 7 of the RAIM paper):
    r_k(theta_hat + d) ~ r_k + H_k d, linear set {d : |r_k + H_k d|_inf <= eps for all k}.
      (b1) one parameter at a time, closed form
      (b2) all six together: min/max of each component by LP (12 LPs); each extreme pose is then
           re-checked with exact reprojection (shows the linearisation error)
    (b2) is also tried with ALL observations (including those already outside at theta_hat).
Reference: Gauss-Markov +-2 sigma, Sigma = s^2 (H^T H)^-1, s^2 = sum r^2 / (2m - 6).
GT (same gauge rule) is reported last, for comparison only.
"""
import os
import sys
import json
import math
import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
import cp_conformal as CC

NAMES = ["w_x", "w_y", "w_z", "c_x", "c_y", "c_z"]
COL = {"exact": "#2a78d6", "lin1": "#eb6834", "lp": "#1baf7a", "gm": "#52514e"}


def project(R, c, intr, X):
    """COLMAP SIMPLE_RADIAL projection of points X (N,3) by world-to-camera R and centre c."""
    fx, fy, cx, cy, k1 = intr[:5]
    q = (X - c) @ R.T
    x, y = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
    d = 1 + k1 * (x * x + y * y)
    return np.c_[fx * x * d + cx, fy * y * d + cy], q[:, 2]


def pose_of(theta, Rhat, chat):
    return Rotation.from_rotvec(theta[:3]).as_matrix() @ Rhat, chat + theta[3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--image", type=int, default=17)
    ap.add_argument("--pair", default="162,15")
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(MODEL, "pnp_set", "baseline1_eps%g_img%d" % (a.eps, a.image))
    os.makedirs(OUT, exist_ok=True)

    cams, imgs, pts = read_model(MODEL)
    i1, i2 = [int(x) for x in a.pair.split(",")]
    i3 = a.image
    R1, c1 = imgs[i1].R, imgs[i1].center
    scale = (R1 @ (imgs[i2].center - c1))[0]
    Rhat = imgs[i3].R @ R1.T
    chat = R1 @ (imgs[i3].center - c1) / scale
    intr = list(cams[imgs[i3].camera_id].unpack())
    im = imgs[i3]
    m = np.flatnonzero(im.p3d_ids >= 0)
    pid = im.p3d_ids[m].astype(int)
    X = np.array([R1 @ (pts[k].xyz - c1) / scale for k in pid])
    Y = im.xys[m]
    common = set(imgs[i1].p3d_ids) & set(imgs[i2].p3d_ids)
    sel_common = np.array([k in common for k in pid])

    def resid(theta, idx=slice(None)):
        R, c = pose_of(theta, Rhat, chat)
        uv, z = project(R, c, intr, X[idx])
        r = uv - Y[idx]
        r[z <= 0] = np.inf
        return r

    r0 = resid(np.zeros(6))
    rinf0 = np.abs(r0).max(1)
    ok0 = rinf0 <= a.eps
    rep = {"eps": a.eps, "image": im.name, "observations": int(len(m)),
           "residual_at_colmap_pose_px": {"median": float(np.median(rinf0)), "max": float(rinf0.max()),
                                          "n_over_eps": int((~ok0).sum())},
           "units": "rotation in degrees; centre in gauge units (x of camera 2's centre = 1)"}
    print("%s: %d observations; at COLMAP's pose: median %.2f px, max %.2f px, %d over eps = %g"
          % (im.name, len(m), np.median(rinf0), rinf0.max(), (~ok0).sum(), a.eps))

    # ---- Jacobian (numeric, central differences) ---------------------------------------
    H = np.zeros((len(m), 2, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = 1e-6
        H[:, :, j] = (resid(d) - resid(-d)) / 2e-6

    def exact_limit(j, s, idx, t_max):
        e = np.zeros(6)
        e[j] = s
        f = lambda t: np.abs(resid(t * e, idx)).max() <= a.eps
        if f(t_max):
            return t_max
        lo, hi = 0.0, t_max
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return lo

    def lin_limit(j, s, idx):
        h = s * H[idx][:, :, j].ravel()
        r = r0[idx].ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(h > 0, (a.eps - r) / h, np.where(h < 0, (-a.eps - r) / h, np.inf))
        return float(t.min())

    def lp_box(idx):
        A = np.vstack([H[idx].reshape(-1, 6), -H[idx].reshape(-1, 6)])
        b = np.concatenate([a.eps - r0[idx].ravel(), a.eps + r0[idx].ravel()])
        feas = linprog(np.zeros(6), A_ub=A, b_ub=b, bounds=[(None, None)] * 6, method="highs")
        if feas.status != 0:
            return None
        lo, hi, xl, xh = np.zeros(6), np.zeros(6), [], []
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1
            r1 = linprog(e, A_ub=A, b_ub=b, bounds=[(None, None)] * 6, method="highs")
            r2 = linprog(-e, A_ub=A, b_ub=b, bounds=[(None, None)] * 6, method="highs")
            lo[j], hi[j] = r1.x[j], r2.x[j]
            xl.append(r1.x)
            xh.append(r2.x)
        return lo, hi, xl, xh

    T_MAX = np.array([math.radians(20)] * 3 + [2.0] * 3)
    sets = {"all": np.ones(len(m), bool), "fit at COLMAP pose": ok0,
            "fit at COLMAP pose, landmarks common with the pair": ok0 & sel_common}
    results = {}
    for name, idx in sets.items():
        res = {"n_obs": int(idx.sum())}
        if ok0[idx].all():
            ex = np.array([[exact_limit(j, s, idx, T_MAX[j]) for s in (-1, 1)] for j in range(6)])
            l1 = np.array([[lin_limit(j, s, idx) for s in (-1, 1)] for j in range(6)])
            res["exact_one_param"] = {"minus": (-ex[:, 0]).tolist(), "plus": ex[:, 1].tolist()}
            res["linear_one_param"] = {"minus": (-l1[:, 0]).tolist(), "plus": l1[:, 1].tolist()}
        box = lp_box(idx)
        if box is None:
            res["lp_combined"] = "infeasible"
        else:
            lo, hi, xl, xh = box
            ver = [float(np.abs(resid(x, idx)).max()) for x in xl + xh]
            res["lp_combined"] = {"lo": lo.tolist(), "hi": hi.tolist(),
                                  "exact_max_residual_at_extremes_px": ver}
            res["_extremes"] = (xl, xh)
        # Gauss-Markov
        Hs = H[idx].reshape(-1, 6)
        s2 = float((r0[idx] ** 2).sum() / (Hs.shape[0] - 6))
        sd = np.sqrt(np.diag(s2 * np.linalg.inv(Hs.T @ Hs)))
        res["gauss_markov_2sigma"] = (2 * sd).tolist()
        results[name] = res

    # ---- GT in the same gauge ---------------------------------------------------------------
    gt = CC.load_gt(SCENE)
    g1 = gt[imgs[i1].name]
    sc = (g1["R"] @ (gt[imgs[i2].name]["c"] - g1["c"]))[0]
    R_gt = gt[im.name]["R"] @ g1["R"].T
    c_gt = g1["R"] @ (gt[im.name]["c"] - g1["c"]) / sc
    th_gt = np.r_[Rotation.from_matrix(R_gt @ Rhat.T).as_rotvec(), c_gt - chat]
    rep["gt_offset"] = th_gt.tolist()

    # ---- report -------------------------------------------------------------------------------
    conv = np.array([180 / math.pi] * 3 + [1.0] * 3)
    fmt = lambda v: "%+8.3f" % v
    print("\nparameter ranges around COLMAP's pose (rotation deg, centre in baseline units); GT offset last")
    for name, res in results.items():
        print("\n== observations: %s (%d)" % (name, res["n_obs"]))
        print("   %-5s %21s %21s %21s %10s %9s" % ("", "(a) exact, 1 param", "(b1) linear, 1 param",
                                                 "(b2) LP, combined", "GM +-2sd", "GT"))
        for j in range(6):
            cols = []
            for key in ("exact_one_param", "linear_one_param"):
                if key in res:
                    cols.append("%s .. %s" % (fmt(res[key]["minus"][j] * conv[j]), fmt(res[key]["plus"][j] * conv[j])))
                else:
                    cols.append("%21s" % "start infeasible")
            lp = res["lp_combined"]
            cols.append("%s .. %s" % (fmt(lp["lo"][j] * conv[j]), fmt(lp["hi"][j] * conv[j])) if isinstance(lp, dict)
                        else "%21s" % lp)
            print("   %-5s %21s %21s %21s %10.3f %+9.3f" % (NAMES[j], *cols, res["gauss_markov_2sigma"][j] * conv[j],
                                                         th_gt[j] * conv[j]))
        if isinstance(res["lp_combined"], dict):
            v = res["lp_combined"]["exact_max_residual_at_extremes_px"]
            print("   exact max residual at the 12 LP extreme poses: %.2f .. %.2f px (eps %g)" % (min(v), max(v), a.eps))
    rep["results"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    json.dump(rep, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    # ---- figure 1: residuals at COLMAP's pose and at each one-parameter limit (exact) ---------
    idx = sets["fit at COLMAP pose"]
    res = results["fit at COLMAP pose"]
    fig, axs = plt.subplots(3, 5, figsize=(17, 10.5))
    axs = axs.ravel()
    panels = [("COLMAP pose", np.zeros(6), None)]
    for j in range(6):
        for s, key in ((-1, "minus"), (1, "plus")):
            th = np.zeros(6)
            th[j] = res["exact_one_param"][key][j]
            panels.append(("%s %s: %s" % (NAMES[j], "-" if s < 0 else "+",
                                          ("%.3f deg" % (th[j] * 180 / math.pi)) if j < 3 else ("%.4f" % th[j])), th, j))
    for ax, (title, th, j) in zip(axs, panels):
        r = resid(th, idx)
        rr = np.abs(r).max(1)
        hit = np.argmax(rr)
        ax.add_patch(plt.Rectangle((-a.eps, -a.eps), 2 * a.eps, 2 * a.eps, fill=False, ec="#0b0b0b", lw=1.2))
        ax.scatter(r[:, 0], r[:, 1], s=6, color=COL["exact"], alpha=0.6)
        if j is not None:
            ax.scatter(r[hit, 0], r[hit, 1], s=40, color="#e34948", zorder=3, label="first point at the box edge")
        ax.set_xlim(-1.6 * a.eps, 1.6 * a.eps)
        ax.set_ylim(-1.6 * a.eps, 1.6 * a.eps)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[len(panels):]:
        ax.axis("off")
    axs[0].set_xlabel("residual u (px)")
    axs[0].set_ylabel("residual v (px)")
    axs[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("%s: reprojection residuals of %d COLMAP points (those within the box at COLMAP's pose); "
                 "square = +-%g px. Each panel moves one pose parameter to its exact limit."
                 % (im.name, idx.sum(), a.eps), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "residuals_at_limits.png"), dpi=110)

    # ---- figure 2: ranges per parameter ---------------------------------------------------------------
    fig, axs = plt.subplots(1, 6, figsize=(17, 3.8))
    for j, ax in enumerate(axs):
        rows = [("(a) exact", res["exact_one_param"]["minus"][j], res["exact_one_param"]["plus"][j], COL["exact"]),
                ("(b1) linear", res["linear_one_param"]["minus"][j], res["linear_one_param"]["plus"][j], COL["lin1"])]
        lp = res["lp_combined"]
        if isinstance(lp, dict):
            rows.append(("(b2) LP combined", lp["lo"][j], lp["hi"][j], COL["lp"]))
        g = res["gauss_markov_2sigma"][j]
        rows.append(("GM +-2 sd", -g, g, COL["gm"]))
        for y, (lab, lo, hi, col) in enumerate(rows[::-1]):
            ax.plot([lo * conv[j], hi * conv[j]], [y, y], color=col, lw=6, solid_capstyle="round")
        ax.axvline(th_gt[j] * conv[j], color="#008300", lw=1.5, ls="--", label="GT")
        ax.axvline(0, color="#0b0b0b", lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows[::-1]] if j == 0 else [], fontsize=9)
        ax.set_title("%s (%s)" % (NAMES[j], "deg" if j < 3 else "baseline units"), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e4df", lw=0.8)
    axs[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("%s: pose range around COLMAP's pose with COLMAP's points fixed, eps = %g px" % (im.name, a.eps), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ranges.png"), dpi=120)
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
