"""Figures for the plain linearised LP (image 3), in the style of baseline1_pose.py.

    python viz_plain_lp.py --scene airplane --eps 3

1. ranges.png             per-parameter ranges: plain LP vs inner radial search (exact), sound
                          linearised B&B, and Baseline 1 (COLMAP points fixed, same 115 observations)
2. residuals_at_limits.png  for each of the 12 LP extremes, the LP's own (w, c, X_k) reprojected
                          EXACTLY (no linearisation) into image 3: residual of every observation,
                          with the +-eps box. Points outside the box show the linearisation error.
Rotations are relative to R_hat3 (P3P reference), centres absolute, in the gauge of linpnp_set.py.
"""
import os
import sys
import json
import math
import argparse
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
from stage_a import undistort
from pnp_set import box_normals, fixed_rows, expR
import linpnp_set as LS

NAMES = ["w_x", "w_y", "w_z", "c_x", "c_y", "c_z"]
COL = {"plain": "#eb6834", "inner": "#2a78d6", "sound": "#52514e", "base": "#1baf7a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--pair", default="162,15")
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--delta0-deg", type=float, default=5.0)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    D = os.path.join(MODEL, "pnp_set", "linearised_eps%g_img%d" % (a.eps, a.third))
    OUT = os.path.join(D, "plain_lp_figures")
    os.makedirs(OUT, exist_ok=True)

    # ---- rebuild the plain LP (same setup as linpnp_set.py) ----------------------------
    cams, imgs, pts = read_model(MODEL)
    i1, i2 = [int(x) for x in a.pair.split(",")]
    i3 = a.third
    intr = {i: cams[imgs[i].camera_id].unpack() for i in (i1, i2, i3)}
    f = {i: (intr[i][0], intr[i][1]) for i in intr}
    R1, c1 = imgs[i1].R, imgs[i1].center
    scale = (R1 @ (imgs[i2].center - c1))[0]
    pose = {i1: (np.eye(3), np.zeros(3)), i2: (imgs[i2].R @ R1.T, R1 @ (imgs[i2].center - c1) / scale)}
    Xg = {k: R1 @ (p.xyz - c1) / scale for k, p in pts.items()}
    det = {}
    for i in (i1, i2, i3):
        d = {}
        for j in np.flatnonzero(imgs[i].p3d_ids >= 0):
            d.setdefault(int(imgs[i].p3d_ids[j]), []).append(undistort(*imgs[i].xys[j], intr[i]))
        det[i] = d
    common = sorted(set(det[i1]) & set(det[i2]))
    zmin = 1e-3 * float(np.median([pose[i][0][2] @ (Xg[k] - pose[i][1]) for k in common for i in (i1, i2)]))
    Fk, Fbox, X0 = {}, {}, {}
    for k in common:
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i][k]:
                g_, h_ = fixed_rows(*pose[i], xn, f[i], a.eps, zmin)
                G.append(g_)
                H.append(h_)
        G, H = np.vstack(G), np.concatenate(H)
        res = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[G, np.linalg.norm(G, axis=1)], b_ub=H,
                      bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if res.status != 0 or res.x[3] <= 0:
            continue
        bb = []
        for dd in range(3):
            e = np.zeros(3)
            e[dd] = 1
            r1 = linprog(e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
            r2 = linprog(-e, A_ub=G, b_ub=H, bounds=[(None, None)] * 3, method="highs")
            bb.append([r1.fun, -r2.fun] if r1.status == 0 and r2.status == 0 else [-np.inf, np.inf])
        bb = np.array(bb)
        if np.all(np.isfinite(bb)):
            Fk[k], Fbox[k], X0[k] = (G, H), (bb[:, 0], bb[:, 1]), res.x[:3]
    L3 = sorted(k for k in Fk if k in det[i3])
    obs3 = [(k, box_normals(xn, f[i3], a.eps)[0]) for k in L3 for xn in det[i3][k]]
    xn3 = [xn for k in L3 for xn in det[i3][k]]
    k3 = [k for k in L3 for xn in det[i3][k]]
    data = LS.Data(L3, Fk, Fbox, X0, obs3, zmin)
    prev = os.path.join(MODEL, "pnp_set", "eps%g_img%d_d05_h0.1" % (a.eps, i3))
    Rhat = np.array(json.load(open(os.path.join(prev, "sets.json")))["R_hat3"])
    sm = json.load(open(os.path.join(prev, "summary.json")))
    cbox0 = (np.array(sm["step5"]["centre_lo"]), np.array(sm["step5"]["centre_hi"]))
    c0 = 0.5 * (cbox0[0] + cbox0[1])
    A, b, bnds = data.rows_linear(Rhat, c0, cbox0, 0.0, math.radians(a.delta0_deg))

    extremes = []
    for j in range(6):
        for sg in (1.0, -1.0):
            cv = np.zeros(data.n)
            cv[j] = sg
            r = linprog(cv, A_ub=A, b_ub=b, bounds=bnds, method="highs")
            extremes.append((j, -sg, r.x))            # -sg: +1 = maximum, -1 = minimum

    def exact_residuals(x):
        """reproject the LP's own X_k through the exact pose exp([w]x) R_hat, c (undistorted px)."""
        R = expR(x[:3]) @ Rhat
        c = x[3:6]
        out = []
        for k, xn in zip(k3, xn3):
            X = x[data.idx[k]:data.idx[k] + 3]
            q = R @ (X - c)
            out.append([f[i3][0] * (q[0] / q[2] - xn[0]), f[i3][1] * (q[1] / q[2] - xn[1])])
        return np.array(out)

    def ref_residuals():
        """reference: P3P rotation, PnP centre, set centres."""
        sm1 = sm["step1"]
        c = np.array(sm1["centre_pnp"])
        out = []
        for k, xn in zip(k3, xn3):
            q = Rhat @ (X0[k] - c)
            out.append([f[i3][0] * (q[0] / q[2] - xn[0]), f[i3][1] * (q[1] / q[2] - xn[1])])
        return np.array(out)

    # ---- figure: residuals at the 12 LP extremes -------------------------------------------
    fig, axs = plt.subplots(3, 5, figsize=(17, 10.5))
    axs = axs.ravel()
    panels = [("reference: P3P pose, set centres", ref_residuals(), None)]
    for j, s, x in extremes:
        lab = ("%.3f deg" % math.degrees(x[j])) if j < 3 else ("%.4f" % x[j])
        panels.append(("%s %s: %s" % (NAMES[j], "max" if s > 0 else "min", lab), exact_residuals(x), x))
    worst = []
    for ax, (title, r, x) in zip(axs, panels):
        rr = np.abs(r).max(1)
        out = rr > a.eps + 1e-6
        worst.append(rr.max())
        ax.add_patch(plt.Rectangle((-a.eps, -a.eps), 2 * a.eps, 2 * a.eps, fill=False, ec="#0b0b0b", lw=1.2))
        ax.scatter(r[~out, 0], r[~out, 1], s=7, color=COL["inner"], alpha=0.7)
        ax.scatter(r[out, 0], r[out, 1], s=18, color="#e34948", zorder=3,
                   label="outside the box (linearisation error)")
        lim = max(1.6 * a.eps, 1.1 * np.abs(r).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title("%s\nmax %.2f px, %d outside" % (title, rr.max(), out.sum()), fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[len(panels):]:
        ax.axis("off")
    axs[0].set_xlabel("residual u (px)")
    axs[0].set_ylabel("residual v (px)")
    for ax in axs[1:len(panels)]:
        if ax.collections and len(ax.collections) > 1 and len(ax.collections[1].get_offsets()):
            ax.legend(fontsize=8, frameon=False, loc="lower left")
            break
    fig.suptitle("%s, plain linearised LP: each panel is one LP extreme (min/max of one pose parameter). "
                 "The LP's own centre and landmark positions are reprojected exactly; square = +-%g px."
                 % (imgs[i3].name, a.eps), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "residuals_at_limits.png"), dpi=110)

    # ---- figure: ranges per parameter -----------------------------------------------------------
    S = json.load(open(os.path.join(D, "summary.json")))
    B1 = json.load(open(os.path.join(MODEL, "pnp_set", "baseline1_eps%g_img%d" % (a.eps, i3), "summary.json")))
    b1 = B1["results"]["fit at COLMAP pose, landmarks common with the pair"]["lp_combined"]
    w_col = np.array(S["gt"]["colmap_w_deg"])
    c_col = np.array(S["gt"]["colmap_centre"])
    rows = {
        "plain LP": (S["plain"]["w_lo_deg"] + S["plain"]["centre_lo"], S["plain"]["w_hi_deg"] + S["plain"]["centre_hi"], COL["plain"]),
        "inner (exact)": (S["inner"]["w_lo_deg_rel_R_hat"] + S["inner"]["centre_lo"],
                          S["inner"]["w_hi_deg_rel_R_hat"] + S["inner"]["centre_hi"], COL["inner"]),
        "sound B&B": (S["sound"]["w_lo_deg"] + S["sound"]["centre_lo"], S["sound"]["w_hi_deg"] + S["sound"]["centre_hi"], COL["sound"]),
        "Baseline 1\n(points fixed)": (list(w_col + np.degrees(b1["lo"][:3])) + list(c_col + np.array(b1["lo"][3:])),
                                       list(w_col + np.degrees(b1["hi"][:3])) + list(c_col + np.array(b1["hi"][3:])), COL["base"]),
    }
    gtv = list(S["gt"]["gt_w_deg"]) + list(S["gt"]["gt_centre"])
    colv = list(w_col) + list(c_col)
    fig, axs = plt.subplots(1, 6, figsize=(17, 4.2))
    labels = list(rows)
    for j, ax in enumerate(axs):
        for y, lab in enumerate(labels[::-1]):
            lo, hi, col = rows[lab]
            ax.plot([lo[j], hi[j]], [y, y], color=col, lw=7, solid_capstyle="round")
        ax.axvline(gtv[j], color="#008300", lw=1.6, ls="--", label="GT")
        ax.axvline(colv[j], color="#0b0b0b", lw=1.0, label="COLMAP")
        if j < 3:
            ax.axvspan(-a.delta0_deg, a.delta0_deg, color="#e5e4df", alpha=0.35, lw=0, label="search region")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1] if j == 0 else [], fontsize=9)
        ax.set_title("%s (%s)" % (NAMES[j], "deg, rel. R_hat3" if j < 3 else "baseline units"), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e4df", lw=0.8)
    axs[-1].legend(fontsize=8, frameon=False)
    axs[0].legend(*axs[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("%s: camera-3 pose ranges, eps = %g px (115 observations of 107 landmarks shared with the pair)"
                 % (imgs[i3].name, a.eps), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ranges.png"), dpi=120)
    print("worst exact residual at the 12 LP extremes: %s px" % np.round(worst[1:], 2))
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
