"""The 'laser' picture for camera 3: beam through the eps window, landmark set, back-shadow.

    python viz_back_shadow.py --scene airplane --eps 3 [--landmark ID]

For a FIXED orientation R of camera 3:
    F_k              landmark k's set from cameras 1, 2 (polytope)
    K_k(R)           the beam: world directions through the +-eps window of the detection
    back-shadow      B_k = F_k (+) (-K_k(R)) = { C : some point of F_k is seen inside the window from C }
                     (an unbounded polyhedron; drawn truncated far behind the camera)
    feasible centres  intersection of B_k over all landmarks seen by camera 3
                     (= the exact fixed-rotation LP of the inner search, as a polytope)
Gauge as in pnp_set.py / linpnp_set.py: camera #162 at the origin, x of #15's centre = 1.
Writes back_shadow.html (interactive) and back_shadow.png.
"""
import os
import sys
import json
import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
import cp_conformal as CC
from stage_a import undistort
from pnp_set import fixed_rows

BLUE, ORANGE, AQUA, GRAY, RED, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e34948", "#008300"


def cheb(A, b):
    """Chebyshev centre of {x : A x <= b}; None if empty."""
    nrm = np.linalg.norm(A, axis=1)
    r = linprog(np.r_[np.zeros(A.shape[1]), -1], A_ub=np.c_[A, nrm], b_ub=b,
                bounds=[(None, None)] * A.shape[1] + [(0, None)], method="highs")
    return (r.x[:-1], r.x[-1]) if r.status == 0 and r.x[-1] > 1e-12 else (None, 0)


def poly_vertices(A, b):
    c, r = cheb(A, b)
    if c is None:
        return None
    hs = HalfspaceIntersection(np.c_[A, -b], c)
    return hs.intersections


def mesh(ax_or_fig, V, color, alpha, name=None, plotly=False):
    hull = ConvexHull(V)
    if plotly:
        i, j, k = hull.simplices.T
        ax_or_fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=i, j=j, k=k, color=color,
                                      opacity=alpha, name=name, showlegend=True, flatshading=True))
    else:
        ax_or_fig.add_collection3d(Poly3DCollection([V[s] for s in hull.simplices], facecolor=color,
                                                    alpha=alpha, edgecolor=color, linewidth=0.2))


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
    OUT = os.path.join(MODEL, "pnp_set", "back_shadow")
    os.makedirs(OUT, exist_ok=True)

    cams, imgs, pts = read_model(MODEL)
    i1, i2 = [int(x) for x in a.pair.split(",")]
    i3 = a.third
    intr = {i: cams[imgs[i].camera_id].unpack() for i in (i1, i2, i3)}
    f = {i: (intr[i][0], intr[i][1]) for i in intr}
    R1, c1 = imgs[i1].R, imgs[i1].center
    scale = (R1 @ (imgs[i2].center - c1))[0]
    pose = {i1: (np.eye(3), np.zeros(3)), i2: (imgs[i2].R @ R1.T, R1 @ (imgs[i2].center - c1) / scale)}
    R_col = imgs[i3].R @ R1.T
    c_col = R1 @ (imgs[i3].center - c1) / scale
    det = {}
    for i in (i1, i2, i3):
        d = {}
        for j in np.flatnonzero(imgs[i].p3d_ids >= 0):
            d.setdefault(int(imgs[i].p3d_ids[j]), []).append(undistort(*imgs[i].xys[j], intr[i]))
        det[i] = d
    common = sorted(set(det[i1]) & set(det[i2]) & set(det[i3]))
    zmin = 1e-3

    # ---- landmark polytopes F_k ---------------------------------------------------------
    F = {}
    for k in common:
        G, H = [], []
        for i in (i1, i2):
            for xn in det[i][k]:
                g_, h_ = fixed_rows(*pose[i], xn, f[i], a.eps, zmin)
                G.append(g_)
                H.append(h_)
        G, H = np.vstack(G), np.concatenate(H)
        V = poly_vertices(G, H)
        if V is not None and np.all(np.isfinite(V)) and np.ptp(V, 0).max() < 5:
            F[k] = (G, H, V)

    # ---- choice of the fixed rotation: COLMAP's if some centre is feasible ----------------
    def beam_dirs(R, xn):
        ex, ey = a.eps / f[i3][0], a.eps / f[i3][1]
        D = np.array([[xn[0] + sx * ex, xn[1] + sy * ey, 1.0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        D = D @ R                                     # rows: R^T [x, y, 1]
        return D / np.linalg.norm(D, axis=1, keepdims=True)

    def shadow_halfspaces(R, k, T):
        """truncated back-shadow of landmark k (all its camera-3 observations): A c <= b."""
        V = F[k][2]
        A, b = [], []
        for xn in det[i3][k]:
            D = beam_dirs(R, xn)
            P = np.vstack([V] + [V - T * d for d in D])
            hull = ConvexHull(P)
            A.append(hull.equations[:, :3])
            b.append(-hull.equations[:, 3])
        return np.vstack(A), np.concatenate(b)

    T = 6.0
    choice = None
    prev = os.path.join(MODEL, "pnp_set", "eps%g_img%d_d05_h0.1" % (a.eps, i3), "sets.json")
    R_hat = np.array(json.load(open(prev))["R_hat3"]) if os.path.exists(prev) else None
    for nm, R in (("COLMAP rotation", R_col), ("P3P reference rotation", R_hat)):
        if R is None:
            continue
        AB = [shadow_halfspaces(R, k, T) for k in F]
        A = np.vstack([x[0] for x in AB])
        b = np.concatenate([x[1] for x in AB])
        c, r = cheb(A, b)
        if c is not None:
            choice = (nm, R, A, b)
            break
    if choice is None:
        print("no feasible centre for the candidate rotations")
        return
    nm, R, A_all, b_all = choice
    Vfeas = HalfspaceIntersection(np.c_[A_all, -b_all], cheb(A_all, b_all)[0]).intersections
    print("fixed orientation: %s; feasible-centre polytope from %d landmarks: size %s"
          % (nm, len(F), np.round(np.ptp(Vfeas, 0), 4)))

    # ---- the landmark to show: median-size box among those seen by camera 3 -------------------
    if a.landmark is None:
        sizes = {k: np.ptp(F[k][2], 0).max() for k in F}
        k = sorted(sizes, key=sizes.get)[len(sizes) // 2]
    else:
        k = a.landmark
    V = F[k][2]
    xn = det[i3][k][0]
    D = beam_dirs(R, xn)
    A_k, b_k = shadow_halfspaces(R, k, 1.2 * np.linalg.norm(V.mean(0) - c_col))
    V_shadow = HalfspaceIntersection(np.c_[A_k, -b_k], cheb(A_k, b_k)[0]).intersections
    dist = np.linalg.norm(V.mean(0) - c_col)

    # ground truth in the same gauge
    gt = CC.load_gt(SCENE)
    g1 = gt[imgs[i1].name]
    sc = (g1["R"] @ (gt[imgs[i2].name]["c"] - g1["c"]))[0]
    c_gt = g1["R"] @ (gt[imgs[i3].name]["c"] - g1["c"]) / sc
    inside = lambda P, Ah, bh: bool(np.all(Ah @ P <= bh + 1e-9))
    info = {"fixed_orientation": nm, "landmark": int(k), "landmark_box_size": np.ptp(V, 0).tolist(),
            "distance_camera_to_landmark": float(dist),
            "feasible_centre_size": np.ptp(Vfeas, 0).tolist(),
            "colmap_centre_in_this_shadow": inside(c_col, A_k, b_k),
            "colmap_centre_in_feasible_region": inside(c_col, A_all, b_all),
            "gt_centre_in_this_shadow": inside(c_gt, A_k, b_k),
            "gt_centre_in_feasible_region": inside(c_gt, A_all, b_all),
            "note": "the GT centre need not be feasible here: camera 3's orientation is fixed, not the true one"}
    json.dump(info, open(os.path.join(OUT, "back_shadow.json"), "w"), indent=2)
    print(json.dumps(info, indent=1))

    # ---- interactive 3D -------------------------------------------------------------------------
    fig = go.Figure()
    mesh(fig, V, BLUE, 0.85, "landmark set F_k (#%d)" % k, plotly=True)
    for d in D:
        p = c_col + d * dist * 1.1
        fig.add_trace(go.Scatter3d(x=[c_col[0], p[0]], y=[c_col[1], p[1]], z=[c_col[2], p[2]], mode="lines",
                                   line=dict(color=ORANGE, width=4), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color=ORANGE, width=4),
                               name="beam through the +-%g px window (from COLMAP's centre)" % a.eps))
    mesh(fig, V_shadow, AQUA, 0.18, "back-shadow: centres whose beam hits F_k", plotly=True)
    mesh(fig, Vfeas, GRAY, 0.9, "feasible centres (all %d back-shadows)" % len(F), plotly=True)
    for p, col, lab, sym in ((c_col, "#0b0b0b", "COLMAP centre", "circle"), (c_gt, GREEN, "GT centre", "diamond")):
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                   marker=dict(size=5, color=col, symbol=sym), name=lab))
    fig.update_layout(title="Camera 3 (%s), fixed orientation (%s), eps = %g px: landmark #%d"
                            % (imgs[i3].name, nm, a.eps, k),
                      scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")
    fig.write_html(os.path.join(OUT, "back_shadow.html"), include_plotlyjs=True)

    # ---- static figure: overview + zoom near the camera ---------------------------------------
    fig2 = plt.figure(figsize=(16, 7.5))
    ax = fig2.add_subplot(1, 2, 1, projection="3d")
    mesh(ax, V, BLUE, 0.9)
    mesh(ax, V_shadow, AQUA, 0.10)
    for d in D:
        p = c_col + d * dist * 1.1
        ax.plot(*zip(c_col, p), color=ORANGE, lw=1.2)
    ax.scatter(*c_col, color="#0b0b0b", s=30)
    ax.scatter(*c_gt, color=GREEN, marker="D", s=30)
    ax.text(*V.mean(0), "  F_k", color=BLUE)
    ax.text(*c_col, "  camera 3", color="#0b0b0b")
    allp = np.vstack([V, V_shadow, c_col[None]])
    ax.set_box_aspect(np.ptp(allp, 0))
    ax.set_title("(a) beam from COLMAP's centre through the window hits F_k (blue);\n"
                 "aqua = back-shadow: every centre from which the beam hits F_k", fontsize=10)
    ax2 = fig2.add_subplot(1, 2, 2, projection="3d")
    zoom = 0.25
    sel = np.all(np.abs(V_shadow - c_col) < 3 * zoom, axis=1)
    lo, hi = c_col - zoom, c_col + zoom
    mesh(ax2, V_shadow, AQUA, 0.12)
    mesh(ax2, Vfeas, GRAY, 0.9)
    ax2.scatter(*c_col, color="#0b0b0b", s=40, label="COLMAP centre")
    ax2.scatter(*c_gt, color=GREEN, marker="D", s=40, label="GT centre")
    ax2.set_xlim(lo[0], hi[0])
    ax2.set_ylim(lo[1], hi[1])
    ax2.set_zlim(lo[2], hi[2])
    ax2.set_box_aspect((1, 1, 1))
    ax2.plot([], [], color=AQUA, lw=6, alpha=0.4, label="back-shadow of #%d" % k)
    ax2.plot([], [], color=GRAY, lw=6, label="feasible centres: all %d back-shadows" % len(F))
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("(b) near camera 3: one back-shadow (aqua) vs their intersection (gray),\n"
                  "size %s (baseline units)" % np.round(np.ptp(Vfeas, 0), 3), fontsize=10)
    fig2.suptitle("Camera 3 (%s), orientation fixed at the %s, eps = %g px" % (imgs[i3].name, nm, a.eps), fontsize=12)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "back_shadow.png"), dpi=120)
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
