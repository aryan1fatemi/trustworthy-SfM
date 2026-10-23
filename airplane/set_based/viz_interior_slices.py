"""Slices of image 3 for the INTERIOR feasible poses only.

    python viz_interior_slices.py [--eps 3] [--third 17]

Interior poses (from image3_poses.npz): rotation at half the boundary angle, centre = mean of the six
extreme centres for that rotation (300 poses). For each pose the WHOLE +-eps window is back-projected:
    S_k(R, C) = F_k  intersected with  { X : n_j^T R (X - C) <= 0 (4 faces),  r_3^T (X - C) >= z_min }.
Two landmarks are shown: one whose interior slices share a point and one whose do not
(from slice_overlap.csv). Per landmark:
    left   3D: F_k (light), 12 slices (colour = rotation offset from COLMAP), the exact common part
    right  every interior slice along the set's long axis, with the common part shaded
Writes results/interior_slices.png and results/interior_slices.html.
"""
import os
import csv
import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from setlib import Scene, box_normals, chebyshev, RESULTS

BLUE = "#2a78d6"
COMMON = "#e34948"


def vertices(A, b):
    c, r = chebyshev(A, b)
    return None if c is None else HalfspaceIntersection(np.c_[A, -b], c).intersections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--third", type=int, default=17)
    a = ap.parse_args()
    P = np.load(os.path.join(RESULTS, "pair_sets.npz"))
    Q = np.load(os.path.join(RESULTS, "image3_poses.npz"))
    Rs, Cs, ang = Q["R"], Q["C"], Q["angle_deg"]
    interior = np.array([1 + 14 * b + 13 for b in range((len(Rs) - 1) // 14)])
    sc = Scene(pair=(162, 15), extra=(a.third,))
    f3, zmin = sc.f[a.third], float(P["z_min"])

    ov = {int(r["point3D_id"]): r for r in csv.DictReader(open(os.path.join(RESULTS, "slice_overlap.csv")))}
    key = "common_point_rot-half, centre-mean (interior)"
    yes = [k for k, r in ov.items() if r[key] == "1"]
    no = [k for k, r in ov.items() if r[key] == "0"]
    # typical cases: median best-point coverage within each group
    pick = lambda ks: sorted(ks, key=lambda k: float(ov[k]["best_point_coverage"]))[len(ks) // 2]
    shown = [(pick(yes), "interior slices share a point"), (pick(no), "interior slices do not share a point")]

    def Fk(k):
        sel = P["row_landmark"] == k
        return P["G"][sel], P["h"][sel]

    def window(k, R, C):
        A, b = [], []
        for xn in sc.det[a.third][k]:
            for g in box_normals(xn, f3, a.eps) @ R:
                A.append(g)
                b.append(g @ C)
            A.append(-R[2])
            b.append(-zmin - R[2] @ C)
        return np.array(A), np.array(b)

    cmap = plt.get_cmap("viridis")
    amax = ang[interior].max()
    fig = plt.figure(figsize=(16, 11))
    pfig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                         subplot_titles=["#%d: %s" % s for s in shown])
    for row, (k, label) in enumerate(shown):
        G, h = Fk(k)
        V0 = vertices(G, h)
        E = np.array([linprog(sg * d, A_ub=G, b_ub=h, bounds=[(None, None)] * 3, method="highs").x
                      for d in np.eye(3) for sg in (1, -1)])
        ax_ = np.linalg.svd(E - E.mean(0), full_matrices=False)[2][0]
        s0 = V0 @ ax_
        rows_all, iv, polys = [G], [], []
        b_all = [h]
        order = interior[np.argsort(ang[interior])]
        show = set(order[np.linspace(0, len(order) - 1, 12).astype(int)].tolist())
        for i in order:
            A, b = window(k, Rs[i], Cs[i])
            rows_all.append(A)
            b_all.append(b)
            AA, bb = np.vstack([G, A]), np.r_[h, b]
            r1 = linprog(ax_, A_ub=AA, b_ub=bb, bounds=[(None, None)] * 3, method="highs")
            r2 = linprog(-ax_, A_ub=AA, b_ub=bb, bounds=[(None, None)] * 3, method="highs")
            iv.append((r1.fun, -r2.fun, ang[i]))
            if i in show:
                polys.append((vertices(AA, bb), ang[i]))
        Vc = vertices(np.vstack(rows_all), np.concatenate(b_all))
        iv = np.array(iv)

        # 3D
        ax = fig.add_subplot(2, 2, 2 * row + 1, projection="3d")
        hull = ConvexHull(V0)
        ax.add_collection3d(Poly3DCollection([V0[s] for s in hull.simplices], facecolor=BLUE, alpha=0.06,
                                             edgecolor=BLUE, linewidth=0.15))
        pfig.add_trace(go.Mesh3d(x=V0[:, 0], y=V0[:, 1], z=V0[:, 2], i=hull.simplices[:, 0],
                                 j=hull.simplices[:, 1], k=hull.simplices[:, 2], color=BLUE, opacity=0.08,
                                 name="F_k #%d" % k), row=1, col=row + 1)
        for Vs, an in polys:
            if Vs is None:
                continue
            hs = ConvexHull(Vs)
            col = matplotlib.colors.to_hex(cmap(an / amax))
            ax.add_collection3d(Poly3DCollection([Vs[s] for s in hs.simplices], facecolor=col, alpha=0.25,
                                                 edgecolor=col, linewidth=0.2))
            pfig.add_trace(go.Mesh3d(x=Vs[:, 0], y=Vs[:, 1], z=Vs[:, 2], i=hs.simplices[:, 0],
                                     j=hs.simplices[:, 1], k=hs.simplices[:, 2], color=col, opacity=0.25,
                                     name="slice, %.2f deg" % an, showlegend=False), row=1, col=row + 1)
        if Vc is not None:
            hc = ConvexHull(Vc)
            ax.add_collection3d(Poly3DCollection([Vc[s] for s in hc.simplices], facecolor=COMMON, alpha=0.9,
                                                 edgecolor=COMMON, linewidth=0.3))
            pfig.add_trace(go.Mesh3d(x=Vc[:, 0], y=Vc[:, 1], z=Vc[:, 2], i=hc.simplices[:, 0],
                                     j=hc.simplices[:, 1], k=hc.simplices[:, 2], color=COMMON, opacity=0.95,
                                     name="common to all 300 interior slices"), row=1, col=row + 1)
        ax.set_box_aspect(np.ptp(V0, 0))
        ax.set_title("#%d (%s)\nlight blue F_k, coloured: 12 of the 300 interior slices%s"
                     % (k, label, ", red: common part" if Vc is not None else ""), fontsize=10)

        # along the needle
        ax2 = fig.add_subplot(2, 2, 2 * row + 2)
        ax2.axvspan(0, np.ptp(s0), color=BLUE, alpha=0.08, label="F_k")
        for lo, hi, an in iv:
            ax2.plot([lo - s0.min(), hi - s0.min()], [an, an], color=cmap(an / amax), lw=1.3)
        clo, chi = iv[:, 0].max() - s0.min(), iv[:, 1].min() - s0.min()
        if Vc is not None:
            ax2.axvspan(clo, chi, color=COMMON, alpha=0.25, label="common part (exact, 3D)")
        else:
            ax2.plot([], [], color="none", label="no common point (exact, 3D)")
        ax2.set_xlabel("position along the set's long axis (baseline units)")
        ax2.set_ylabel("rotation offset from COLMAP's (deg)")
        ax2.set_title("#%d: the 300 interior slices along the set" % k, fontsize=10)
        ax2.legend(fontsize=8, frameon=False, loc="upper right")
        ax2.spines[["top", "right"]].set_visible(False)
    pfig.update_scenes(aspectmode="data")
    pfig.update_layout(title="Image 3 slices for the 300 interior feasible poses (eps = %g px)" % a.eps,
                       margin=dict(l=0, r=0, t=60, b=0))
    pfig.write_html(os.path.join(RESULTS, "interior_slices.html"), include_plotlyjs=True)
    fig.suptitle("Image 3 (%s): slices of F_k for the 300 interior feasible poses, whole +-%g px window "
                 "back-projected each time" % (sc.imgs[a.third].name, a.eps), fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "interior_slices.png"), dpi=115)
    print("shown:", shown, "->", RESULTS)


if __name__ == "__main__":
    main()
