"""What the slices look like: S_k(theta) = F_k  intersected with  the back-projected eps window of camera 3.

    python viz_slices.py --scene airplane --eps 3 [--landmark ID]

Uses the exactly feasible poses saved by conditional_slices.py (poses.npz).
  slices_3d.png / slices_3d.html   one landmark: F_k (needle) and the slices for poses at increasing
                                   rotation offset from COLMAP's (colour = offset), COLMAP and GT point
  slices_along_needle.png          the same landmark, every slice reduced to its extent along the
                                   needle's long axis, for poses sorted by rotation offset
  slice_sizes.png                  all landmarks: slice length / F_k length per pose, and the union
                                   of the slices / F_k length
"""
import os
import sys
import json
import math
import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
import cp_conformal as CC
from stage_a import undistort
from pnp_set import fixed_rows, box_normals

BLUE, GREEN = "#2a78d6", "#008300"


def cheb(A, b):
    nrm = np.linalg.norm(A, axis=1)
    r = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[A, nrm], b_ub=b, bounds=[(None, None)] * 3 + [(0, None)],
                method="highs")
    return r.x[:3] if r.status == 0 and r.x[3] > 1e-12 else None


def vertices(A, b):
    c = cheb(A, b)
    return None if c is None else HalfspaceIntersection(np.c_[A, -b], c).intersections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--pair", default="162,15")
    ap.add_argument("--third", type=int, default=17)
    ap.add_argument("--landmark", type=int, default=None)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    D = os.path.join(MODEL, "pnp_set", "conditional_slices_eps%g_epsf%g_img%d" % (a.eps, a.eps, a.third))
    P = np.load(os.path.join(D, "poses.npz"))
    Rs, Cs, ang = P["R"], P["C"], P["angle_deg"]

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
    F = {}
    for k in sorted(set(common) & set(det[i3])):
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i][k]:
                g_, h_ = fixed_rows(*pose[i], xn, f[i], a.eps, zmin)
                G.append(g_)
                H.append(h_)
        G, H = np.vstack(G), np.concatenate(H)
        V = vertices(G, H)
        if V is not None and np.ptp(V, 0).max() < 5:
            F[k] = (G, H, V)
    normals = {k: [box_normals(xn, f[i3], a.eps)[0] for xn in det[i3][k]] for k in F}

    def slice_poly(k, R, C):
        A, b = [F[k][0]], [F[k][1]]
        for N in normals[k]:
            for nrm in N:
                g = nrm @ R
                A.append(g[None])
                b.append([g @ C])
            A.append(-R[2][None])
            b.append([-zmin - R[2] @ C])
        return np.vstack(A), np.concatenate([np.atleast_1d(x) for x in b])

    def axis_of(V):
        Vc = V - V.mean(0)
        return np.linalg.svd(Vc, full_matrices=False)[2][0]

    # GT point (same gauge rule)
    gt = CC.load_gt(SCENE)
    g1 = gt[imgs[i1].name]
    sc = (g1["R"] @ (gt[imgs[i2].name]["c"] - g1["c"]))[0]
    from cp_step0_align import umeyama
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]), np.array([imgs[i].center for i in ids]))
    lifted = {}
    for rw in CC.observations(cams, imgs, pts, gt, s_, Ra, ta):
        if rw["valid"]:
            lifted.setdefault(rw["pid"], []).append(rw["Xgt"])
    Xgt = {k: g1["R"] @ (np.median(v, 0) - g1["c"]) / sc for k, v in lifted.items()}

    # ---- statistics over all landmarks (subsample of poses for speed) -----------------------------
    sub = np.r_[0, np.linspace(1, len(Rs) - 1, 400).astype(int)]
    ratio_slice, ratio_union, per_lm = [], [], {}
    for k in F:
        ax_ = axis_of(F[k][2])
        s0 = F[k][2] @ ax_
        L0 = np.ptp(s0)
        ext = []
        for n in sub:
            A, b = slice_poly(k, Rs[n], Cs[n])
            lo = linprog(ax_, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
            hi = linprog(-ax_, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
            if lo.status == 0 and hi.status == 0:
                ext.append((lo.fun, -hi.fun))
        ext = np.array(ext)
        ratio_slice += list((ext[:, 1] - ext[:, 0]) / L0)
        ratio_union.append((ext[:, 1].max() - ext[:, 0].min()) / L0)
        per_lm[k] = (ext, s0.min(), s0.max(), ax_)
    ratio_slice, ratio_union = np.array(ratio_slice), np.array(ratio_union)
    print("all %d landmarks, %d poses each: slice length / F_k length median %.2f (p10 %.2f, p90 %.2f); "
          "union of slices / F_k length median %.2f"
          % (len(F), len(sub), np.median(ratio_slice), np.quantile(ratio_slice, 0.1), np.quantile(ratio_slice, 0.9),
             np.median(ratio_union)))

    # ---- one landmark ----------------------------------------------------------------------------------
    if a.landmark is None:
        k = sorted(F, key=lambda k: np.ptp(F[k][2], 0).max())[len(F) // 2]
    else:
        k = a.landmark
    V = F[k][2]
    ax_ = axis_of(V)
    cmap = plt.get_cmap("viridis")
    amax = ang.max()
    pick = [0] + [int(np.argmin(np.abs(ang - q))) for q in np.linspace(0.5, amax, 9)]
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    hull = ConvexHull(V)
    ax.add_collection3d(Poly3DCollection([V[s] for s in hull.simplices], facecolor=BLUE, alpha=0.08,
                                         edgecolor=BLUE, linewidth=0.2))
    traces = [go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=hull.simplices[:, 0], j=hull.simplices[:, 1],
                        k=hull.simplices[:, 2], color=BLUE, opacity=0.12, name="F_k (cameras 1, 2)")]
    for n in pick:
        A, b = slice_poly(k, Rs[n], Cs[n])
        Vs = vertices(A, b)
        if Vs is None:
            continue
        h = ConvexHull(Vs)
        col = "#0b0b0b" if n == 0 else matplotlib.colors.to_hex(cmap(ang[n] / amax))
        ax.add_collection3d(Poly3DCollection([Vs[s] for s in h.simplices], facecolor=col, alpha=0.55,
                                             edgecolor=col, linewidth=0.3))
        traces.append(go.Mesh3d(x=Vs[:, 0], y=Vs[:, 1], z=Vs[:, 2], i=h.simplices[:, 0], j=h.simplices[:, 1],
                                k=h.simplices[:, 2], color=col, opacity=0.6,
                                name=("slice, COLMAP pose" if n == 0 else "slice, pose %.2f deg from COLMAP" % ang[n])))
    ax.scatter(*Xg[k], color="#0b0b0b", s=25, label="COLMAP point")
    if k in Xgt:
        ax.scatter(*Xgt[k], color=GREEN, marker="D", s=30, label="GT point")
    ax.set_box_aspect(np.ptp(V, 0))
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(a) landmark #%d: F_k (blue) and slices for 10 feasible poses\n"
                 "(black = COLMAP's pose; colour = rotation offset from COLMAP's)" % k, fontsize=10)
    fig.add_trace if False else None
    figp = go.Figure(traces)
    figp.add_trace(go.Scatter3d(x=[Xg[k][0]], y=[Xg[k][1]], z=[Xg[k][2]], mode="markers",
                                marker=dict(size=4, color="#0b0b0b"), name="COLMAP point"))
    if k in Xgt:
        figp.add_trace(go.Scatter3d(x=[Xgt[k][0]], y=[Xgt[k][1]], z=[Xgt[k][2]], mode="markers",
                                    marker=dict(size=4, color=GREEN, symbol="diamond"), name="GT point"))
    figp.update_layout(scene=dict(aspectmode="data"), title="Landmark #%d: F_k and slices" % k,
                       margin=dict(l=0, r=0, t=40, b=0))
    figp.write_html(os.path.join(D, "slices_3d.html"), include_plotlyjs=True)

    ext, s_lo, s_hi, ax_ = per_lm[k]
    angs = ang[sub]
    order = np.argsort(angs)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axvspan(0, s_hi - s_lo, color=BLUE, alpha=0.10, label="F_k along its long axis")
    for y, n in enumerate(order):
        lo, hi = ext[n] - s_lo
        ax2.plot([lo, hi], [angs[n], angs[n]], color=cmap(angs[n] / amax), lw=1.2)
    ax2.axvline(Xg[k] @ ax_ - s_lo, color="#0b0b0b", lw=1.2, label="COLMAP point")
    if k in Xgt:
        ax2.axvline(Xgt[k] @ ax_ - s_lo, color=GREEN, lw=1.5, ls="--", label="GT point")
    ax2.set_xlabel("position along the needle (baseline units)")
    ax2.set_ylabel("pose: rotation offset from COLMAP's (deg)")
    ax2.legend(fontsize=8, frameon=False, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("(b) each line = one feasible pose's slice, measured along the needle", fontsize=10)
    fig.suptitle("Slices S_k(theta) = F_k  intersected with  camera 3's back-projected +-%g px window (%s)"
                 % (a.eps, imgs[i3].name), fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "slices.png"), dpi=120)

    fig3, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    axs[0].hist(ratio_slice, bins=40, color=BLUE, alpha=0.85)
    axs[0].set_xlabel("slice length / F_k length (along the needle)")
    axs[0].set_ylabel("landmark-pose pairs")
    axs[0].set_title("one pose: how much of the needle the window keeps", fontsize=10)
    axs[1].hist(ratio_union, bins=25, color="#1baf7a", alpha=0.85)
    axs[1].set_xlabel("union of slices over feasible poses / F_k length")
    axs[1].set_ylabel("landmarks")
    axs[1].set_title("all feasible poses: how much of the needle is used", fontsize=10)
    for x in axs:
        x.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    fig3.savefig(os.path.join(D, "slice_sizes.png"), dpi=120)
    json.dump({"landmark_shown": int(k), "slice_over_Fk_length_median": float(np.median(ratio_slice)),
               "slice_over_Fk_length_p10_p90": [float(np.quantile(ratio_slice, 0.1)), float(np.quantile(ratio_slice, 0.9))],
               "union_over_Fk_length_median": float(np.median(ratio_union))},
              open(os.path.join(D, "slices_summary.json"), "w"), indent=2)
    print("landmark shown: #%d -> %s" % (k, D))


if __name__ == "__main__":
    main()
