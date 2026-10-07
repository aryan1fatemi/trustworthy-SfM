"""Interactive 3D view of Stage A sets: landmark boxes, camera-centre boxes, frusta, GT.

    python viz_stage_a_sets.py --scene airplane --eps 3 --step 3

Reads stage_a/incremental_eps<eps>/geometry_<step>img.json (stage_a_incremental.py) and writes
stage_a_sets_<step>img.html next to it (self-contained, open in a browser) plus a static PNG.
All coordinates are in the Stage A gauge: reference camera at the origin with R = I,
x-coordinate of the second camera's centre = 1.
"""
import os
import sys
import json
import argparse
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INSIDE, OUTSIDE = "#2a78d6", "#e34948"
CAM, GT, EST = "#eb6834", "#008300", "#0b0b0b"
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    return np.array([[(hi if b & 4 else lo)[0], (hi if b & 2 else lo)[1], (hi if b & 1 else lo)[2]]
                     for b in range(8)])


def box_lines(boxes):
    xs, ys, zs = [], [], []
    for lo, hi in boxes:
        C = corners(lo, hi)
        for a, b in EDGES:
            xs += [C[a, 0], C[b, 0], None]
            ys += [C[a, 1], C[b, 1], None]
            zs += [C[a, 2], C[b, 2], None]
    return xs, ys, zs


def frustum(R, c, intr, depth):
    fx, fy, cx, cy = intr[:4]
    W, H = 2 * cx, 2 * cy
    R, c = np.asarray(R), np.asarray(c)
    pts = [c + R.T @ (depth * np.array([(u - cx) / fx, (v - cy) / fy, 1.0]))
           for u, v in ((0, 0), (W, 0), (W, H), (0, H))]
    xs, ys, zs = [], [], []
    for p in pts:
        xs += [c[0], p[0], None]; ys += [c[1], p[1], None]; zs += [c[2], p[2], None]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        xs += [pts[a][0], pts[b][0], None]; ys += [pts[a][1], pts[b][1], None]; zs += [pts[a][2], pts[b][2], None]
    return xs, ys, zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--step", type=int, default=3)
    a = ap.parse_args()
    D = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene, "COLMAP results", "stage_a",
                                     "incremental_eps%g" % a.eps))
    G = json.load(open(os.path.join(D, "geometry_%dimg.json" % a.step)))
    cams = G["cameras"]
    newest = max(cams, key=lambda n: (not cams[n].get("reference", False), cams[n]["id"] not in (162, 15)))
    newest_id = cams[newest]["id"]
    L = G["landmarks"]
    lms = list(L.values())
    span = np.ptp(np.array([l["colmap"] for l in lms]), 0).max()

    fig = go.Figure()
    groups = [("pair only", lambda l: newest_id not in l["views"]),
              ("seen by %s" % newest, lambda l: newest_id in l["views"])]
    for gname, sel in groups:
        for inside, col, lab in ((1, INSIDE, "GT inside"), (0, OUTSIDE, "GT outside")):
            B = [(l["lo"], l["hi"]) for l in lms if sel(l) and l["gt_in_box"] == inside]
            if not B:
                continue
            x, y, z = box_lines(B)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=col, width=2),
                                       name="landmark boxes, %s, %s (%d)" % (gname, lab, len(B)),
                                       legendgroup=gname, hoverinfo="skip",
                                       opacity=0.55 if inside else 0.9))
    X = np.array([l["colmap"] for l in lms])
    Xt = np.array([l["gt"] for l in lms if l["gt"] is not None])
    ids = list(L.keys())
    diag = [np.linalg.norm(np.subtract(l["hi"], l["lo"])) for l in lms]
    fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
                               marker=dict(size=2, color=EST), name="COLMAP landmark",
                               text=["#%s  box diag %.4f  views %s" % (k, d, l["views"])
                                     for k, d, l in zip(ids, diag, lms)], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=Xt[:, 0], y=Xt[:, 1], z=Xt[:, 2], mode="markers",
                               marker=dict(size=2.5, color=GT, symbol="diamond"), name="GT landmark"))
    for name, c in cams.items():
        intr = G["intrinsics"][name]
        x, y, z = frustum(c["R"], c["colmap"], intr, 0.25 * span)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=CAM, width=4),
                                   name="camera %s%s" % (name, " (reference)" if c.get("reference") else ""),
                                   legendgroup="cam" + name))
        if not c.get("reference"):
            C = corners(c["lo"], c["hi"])
            fig.add_trace(go.Mesh3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], alphahull=0, color=CAM, opacity=0.35,
                                    name="centre set %s  size %s" % (name, np.round(np.subtract(c["hi"], c["lo"]), 3)),
                                    legendgroup="cam" + name, showlegend=True))
        fig.add_trace(go.Scatter3d(x=[c["gt"][0]], y=[c["gt"][1]], z=[c["gt"][2]], mode="markers",
                                   marker=dict(size=5, color=GT, symbol="diamond"),
                                   name="GT centre %s" % name, legendgroup="cam" + name, showlegend=False))
        fig.add_trace(go.Scatter3d(x=[c["colmap"][0]], y=[c["colmap"][1]], z=[c["colmap"][2]],
                                   mode="markers", marker=dict(size=4, color=EST),
                                   name="COLMAP centre %s" % name, legendgroup="cam" + name, showlegend=False))
    fig.update_layout(
        title="Stage A, %d images, eps = %g px: landmark boxes and camera-centre sets "
              "(gauge: %s at origin, baseline x = 1)" % (a.step, a.eps, [n for n in cams if cams[n].get("reference")][0]),
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z (reference view direction)"),
        legend=dict(itemsizing="constant", font=dict(size=11)), margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white")
    out_html = os.path.join(D, "stage_a_sets_%dimg.html" % a.step)
    fig.write_html(out_html, include_plotlyjs=True)

    # ---- static PNG: overview + 2D projections (equal aspect) --------------------
    from matplotlib.patches import Rectangle
    AX = "xyz"
    fig2 = plt.figure(figsize=(18, 11))
    ax = fig2.add_subplot(2, 3, 1, projection="3d")
    for l in lms:
        C = corners(l["lo"], l["hi"])
        col = INSIDE if l["gt_in_box"] == 1 else OUTSIDE
        for e0, e1 in EDGES:
            ax.plot(*zip(C[e0], C[e1]), color=col, lw=0.4, alpha=0.6 if l["gt_in_box"] == 1 else 0.9)
    for name, c in cams.items():
        x, y, z = frustum(c["R"], c["colmap"], G["intrinsics"][name], 0.25 * span)
        seg = lambda v: [np.nan if t is None else t for t in v]
        ax.plot(seg(x), seg(y), seg(z), color=CAM, lw=1.2)
        ax.text(*c["colmap"], " " + name, fontsize=8)
    ax.set_box_aspect(np.ptp(np.vstack([X] + [np.array(c["colmap"])[None] for c in cams.values()]), 0))
    ax.set_title("(a) overview: cameras and landmark boxes", fontsize=11)

    def rects(axh, items, d0, d1, title):
        for lo, hi, col, lw, alpha in items:
            axh.add_patch(Rectangle((lo[d0], lo[d1]), hi[d0] - lo[d0], hi[d1] - lo[d1], fill=False,
                                    edgecolor=col, lw=lw, alpha=alpha))
        axh.set_xlabel(AX[d0]); axh.set_ylabel(AX[d1]); axh.set_aspect("equal", adjustable="datalim")
        axh.set_title(title, fontsize=11)
        axh.spines[["top", "right"]].set_visible(False)

    items = [(l["lo"], l["hi"], INSIDE if l["gt_in_box"] == 1 else OUTSIDE, 0.6,
              0.5 if l["gt_in_box"] == 1 else 0.9) for l in lms]
    for k, (d0, d1, lab) in enumerate(((0, 1, "front view (x-y)"), (0, 2, "top view (x-z, z = depth of reference)")), 2):
        axh = fig2.add_subplot(2, 3, k)
        rects(axh, items, d0, d1, "(%s) landmark boxes, %s" % ("bc"[k - 2], lab))
        axh.scatter(Xt[:, d0], Xt[:, d1], s=3, color=GT, zorder=3, label="GT landmark")
        axh.scatter(X[:, d0], X[:, d1], s=2, color=EST, zorder=3, label="COLMAP landmark")
        axh.plot([], [], color=INSIDE, label="box, GT inside"); axh.plot([], [], color=OUTSIDE, label="box, GT outside")
        axh.legend(fontsize=8, frameon=False, loc="best")
    panels = [(n, c, (1, 2)) for n, c in cams.items() if not c.get("reference") and n == "r_15.png"]
    panels += [(n, c, pr) for n, c in cams.items() if not c.get("reference") and n != "r_15.png"
               for pr in ((0, 2), (1, 2))]
    for k, (n, c, (d0, d1)) in enumerate(panels[:3], 4):
        axh = fig2.add_subplot(2, 3, k)
        rects(axh, [(c["lo"], c["hi"], CAM, 2, 1)], d0, d1,
              "(%s) centre set of %s, %s-%s\nsize (x,y,z) = %s" % ("def"[k - 4], n, AX[d0], AX[d1],
                                                              np.round(np.subtract(c["hi"], c["lo"]), 3)))
        axh.scatter(c["colmap"][d0], c["colmap"][d1], color=EST, s=30, zorder=3, label="COLMAP centre")
        axh.scatter(c["gt"][d0], c["gt"][d1], color=GT, marker="D", s=34, zorder=3, label="GT centre")
        axh.legend(fontsize=8, frameon=False)
        pad = 0.15 * max(np.subtract(c["hi"], c["lo"])[[d0, d1]])
        axh.set_xlim(c["lo"][d0] - pad, c["hi"][d0] + pad); axh.set_ylim(c["lo"][d1] - pad, c["hi"][d1] + pad)
    fig2.suptitle("Stage A, %d images, eps = %g px (gauge: reference camera at origin, baseline x = 1); "
                  "r_15 has x fixed by the scale constraint" % (a.step, a.eps), fontsize=12)
    fig2.tight_layout()
    out_png = os.path.join(D, "stage_a_sets_%dimg.png" % a.step)
    fig2.savefig(out_png, dpi=120)
    print(out_html)
    print(out_png)


if __name__ == "__main__":
    main()
