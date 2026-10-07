"""Draw the feasible set F_k on the actual images.

    python viz_feasible_on_image.py --scene airplane --eps 4.48 [--pid 1234]

For a landmark, F_k is computed exactly (exact_polytope), its vertices are
projected into several images with COLMAP's own camera model, and the 2D convex
hull of those projections is drawn on a zoomed crop.

Panels:
  * observing views -- the hull sits inside the +-eps tolerance box, because
    that is what defines it.  This shows the constraint being satisfied.
  * a NON-observing view -- nothing constrains the landmark there, so the set
    projects to a long streak along the weak direction.  This is the picture of
    3D uncertainty that a per-view pixel error cannot show.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, project
from feasible_set import build_views
import exact_polytope as ex
from cp_step0_align import umeyama
import cp_conformal as CC


def gt_landmark(scene, pid, pts, imgs, cams):
    """True X*_k for one landmark: lift each detection with GT depth, take the
    coordinate-wise median, map into the reconstruction frame (Umeyama)."""
    gt = CC.load_gt(scene)
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))
    p = pts[pid]
    per_view, X = {}, []
    for iid, j in zip(p.image_ids, p.point2D_idxs):
        g = gt[imgs[iid].name]
        D = np.load(g["depth"])["depth"]
        u, v = imgs[iid].xys[j]
        x, y = u - 0.5, v - 0.5
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        q = D[y0:y0 + 2, x0:x0 + 2]
        if (q <= 0).any() or (q.max() - q.min()) / q.mean() > 0.02:
            per_view[int(iid)] = None
            continue
        ax_, ay = x - x0, y - y0
        d = ((1 - ay) * ((1 - ax_) * q[0, 0] + ax_ * q[0, 1])
             + ay * ((1 - ax_) * q[1, 0] + ax_ * q[1, 1]))
        Xg = g["c"] + g["R"].T @ (d * np.linalg.solve(g["K"], [u, v, 1.0]))
        Xs = s_ * Ra @ Xg + ta
        per_view[int(iid)] = Xs
        X.append(Xs)
    return (np.median(X, 0) if X else None), per_view


def hull2d(P):
    if len(P) < 3:
        return P
    return P[ConvexHull(P).vertices]


def panel(ax, img, uv_hull, y_det, uv_hat, eps, title, gt_uv=None, pad=6.0):
    pts_ = uv_hull if gt_uv is None else np.vstack([uv_hull, gt_uv])
    lo, hi = pts_.min(0), pts_.max(0)
    ctr = 0.5 * (lo + hi)
    half = max((hi - lo).max() * 0.62, eps * 1.9, 3.0) + pad
    x0, x1 = ctr[0] - half, ctr[0] + half
    y0, y1 = ctr[1] - half, ctr[1] + half
    ax.imshow(np.asarray(img), extent=(0, img.size[0], img.size[1], 0),
              interpolation="nearest")
    ax.add_patch(Polygon(uv_hull, closed=True, facecolor="#2b6cb0", alpha=0.35,
                         edgecolor="#1a365d", lw=1.6, zorder=3))
    if y_det is not None:
        ax.add_patch(Rectangle((y_det[0] - eps, y_det[1] - eps), 2 * eps, 2 * eps,
                               fill=False, ec="#c05621", lw=1.4, ls="--", zorder=4))
        ax.plot(*y_det, "o", ms=4, color="#c05621", zorder=5)
    ax.plot(*uv_hat, "x", ms=6, mew=1.6, color="#1a202c", zorder=5)
    if gt_uv is not None:
        ax.plot(*gt_uv, "*", ms=11, color="#2f855a", mec="white", mew=0.6, zorder=6)
    ax.set(xlim=(x0, x1), ylim=(y1, y0))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--eps", type=float, default=4.48)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--n-obs", type=int, default=3, help="observing views to show")
    ap.add_argument("--gt", action="store_true", help="overlay the true X*_k")
    ap.add_argument("--no-other", action="store_true", help="only observing views")
    args = ap.parse_args()

    scene = os.path.abspath(os.path.join(HERE, "..", "..", "..", args.scene))
    model = os.path.join(scene, "COLMAP results")
    out = os.path.join(model, "figures")
    os.makedirs(out, exist_ok=True)
    cams, imgs, pts = read_model(model)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}

    if args.pid is not None:
        pids = [args.pid]
    else:                                   # one short, one medium, one long track
        by = {}
        for pid, p in pts.items():
            by.setdefault(min(p.track_length, 99), []).append(pid)
        pids = []
        for want in (2, 8, 40):
            cand = sorted(k for k in by if abs(k - want) <= 2)
            if cand:
                pids.append(sorted(by[cand[0]])[len(by[cand[0]]) // 2])

    for pid in pids:
        p = pts[pid]
        views = build_views(p, imgs, pose)
        H = ex.halfspaces(views, args.eps)
        c, r = ex.chebyshev_center(H)
        if c is None or r <= 0:
            print("pid %d: F_k empty at eps=%g - skipped" % (pid, args.eps))
            continue
        V, vol, _ = ex.vertices_and_volume(H, c)
        diam = float(np.linalg.norm(V[:, None] - V[None], axis=-1).max())

        obs_imgs = list(dict.fromkeys(p.image_ids.tolist()))[: args.n_obs]
        # a view that does NOT observe it: most different direction, point visible
        seen = set(p.image_ids.tolist())
        best, bestang = None, -1
        mean_dir = np.mean([imgs[i].R[2] for i in seen], 0)
        for i, im in imgs.items():
            if i in seen:
                continue
            uv, z = project(im.R, im.tvec, cams[im.camera_id].unpack(), p.xyz)
            if z[0] <= 0 or not (20 < uv[0][0] < 780 and 20 < uv[0][1] < 780):
                continue
            ang = 1 - im.R[2] @ mean_dir / np.linalg.norm(mean_dir)
            if ang > bestang:
                best, bestang = i, ang
        show = [(i, True) for i in obs_imgs]
        if not args.no_other and best:
            show.append((best, False))

        Xstar, per_view = (None, {})
        extra = ""
        if args.gt:
            Xstar, per_view = gt_landmark(scene, pid, pts, imgs, cams)
            if Xstar is not None:
                inside = bool((H[:, :3] @ Xstar + H[:, 3] <= 0).all())
                extra = ("\ngreen star = true $X^*_k$ (GT depth, projected with the "
                         "ESTIMATED pose)  |  dist to $\hat X_k$ = %.4f units  |  "
                         "inside $\mathcal{F}_k$: %s"
                         % (float(np.linalg.norm(Xstar - p.xyz)),
                            "YES" if inside else "NO"))

        fig, axes = plt.subplots(1, len(show), figsize=(3.5 * len(show), 4.1))
        axes = np.atleast_1d(axes)
        for ax, (iid, is_obs) in zip(axes, show):
            im = imgs[iid]
            R, t, intr = pose[iid]
            uvV = project(R, t, intr, V)[0]
            uvhat = project(R, t, intr, p.xyz)[0][0]
            y = None
            if is_obs:
                j = int(p.point2D_idxs[list(p.image_ids).index(iid)])
                y = im.xys[j]
            img = Image.open(os.path.join(scene, "images", im.name)).convert("RGB")
            ext = np.ptp(uvV, 0).max()
            gt_uv = (project(R, t, intr, Xstar)[0][0] if Xstar is not None else None)
            panel(ax, img, hull2d(uvV), y, uvhat, args.eps, gt_uv=gt_uv, title=
                  "%s\n%s  |  projected extent %.1f px"
                  % (im.name, "observes it" if is_obs else "does NOT observe it", ext))
        fig.suptitle("$\\mathcal{F}_k$ for landmark %d   ($\\epsilon$ = %g px, %d views, "
                     "diameter %.3f units)\nblue = projected $\\mathcal{F}_k$,  orange dashed = "
                     "$\\pm\\epsilon$ box at the detection,  x = COLMAP $\\hat X_k$"
                     % (pid, args.eps, p.track_length, diam) + extra, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        f = os.path.join(out, "feasible_on_image_pid%d_eps%g.png" % (pid, args.eps))
        fig.savefig(f, dpi=150)
        plt.close(fig)
        print("pid %d  views %d  diameter %.4f  ->  %s" % (pid, p.track_length, diam, f))


if __name__ == "__main__":
    main()
