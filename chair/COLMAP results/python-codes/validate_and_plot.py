"""Validate the brute-force F_k against the exact polytope, and draw figures.

Check 1 (soundness).  Every grid sample reported feasible is re-tested against
the exact non-linear projection.  Must be 0 violations.

Check 2 (completeness / resolution).  With fixed cameras F_k is exactly a
convex polyhedron (see exact_polytope.py), so its interval hull and volume are
available in closed form via LP.  The grid answer must be slightly SMALLER than
the exact one -- a grid can only under-cover -- and the gap is the
discretisation error.

Figures.  Feasible cloud projected onto the (v3, v2) plane of its own PCA
frame, one panel per landmark; plus normalised spread vs. track length.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_model
from feasible_set import build_views, max_inf_residual, MODEL, OUTDIR
import exact_polytope as ex


def main(tag="eps1"):
    with open(os.path.join(OUTDIR, "feasible_sets_%s.json" % tag)) as fh:
        res = json.load(fh)
    eps = res["eps"]
    clouds = np.load(os.path.join(OUTDIR, "clouds_%s.npz" % tag))

    cams, imgs, pts = read_model(MODEL)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack())
            for i, im in imgs.items()}

    good = [r for r in res["landmarks"] if not r["empty"]]
    print("%7s %6s | %11s %11s  %6s | %10s %10s %6s | %6s"
          % ("pid", "views", "ext grid", "ext exact", "ratio",
             "vol grid", "vol exact", "ratio", "bad"))
    print("-" * 92)

    report = []
    for r in good:
        pid = r["point3D_id"]
        p = pts[pid]
        views = build_views(p, imgs, pose)
        W = clouds[str(pid)]

        bad = int((max_inf_residual(views, W) > eps + 1e-12).sum())
        E = ex.analyse(views, eps, p.xyz)

        g_ext = r["max_interval_extent"]
        e_ext = E.get("max_interval_extent", np.nan)
        g_vol = r["volume"]
        e_vol = E.get("volume", np.nan)
        print("%7d %6d | %11.4e %11.4e  %6.3f | %10.3e %10.3e %6.3f | %6d"
              % (pid, r["n_views"], g_ext, e_ext, g_ext / e_ext,
                 g_vol, e_vol, g_vol / e_vol, bad))
        report.append({"point3D_id": pid, "n_views": r["n_views"],
                       "grid_max_extent": g_ext, "exact_max_extent": e_ext,
                       "extent_ratio": g_ext / e_ext,
                       "grid_volume": g_vol, "exact_volume": e_vol,
                       "volume_ratio": g_vol / e_vol,
                       "infeasible_grid_samples": bad,
                       "colmap_point_inside": E.get("colmap_inside"),
                       "exact": E})

    with open(os.path.join(OUTDIR, "validation_%s.json" % tag), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote validation_%s.json" % tag)

    # ---------------- figure: the sets themselves ------------------------
    n = len(good)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow))
    for ax, r in zip(np.ravel(axes), good):
        pid = r["point3D_id"]
        W = clouds[str(pid)]
        A = np.asarray(r["pca_axes_long_to_short"])   # row 0 = weak dir v3
        P = (W - W.mean(0)) @ A.T
        sub = P[::max(1, P.shape[0] // 6000)] * 1e3
        ax.scatter(sub[:, 0], sub[:, 2], s=0.7, lw=0, c="#2b6cb0", alpha=0.35)
        # true aspect ratio, shown in a square window so the sliver is visible
        h = 0.58 * (P[:, 0].max() - P[:, 0].min()) * 1e3
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("point %d   %d views\n$\\tilde s_k$ = %.2e"
                     % (pid, r["n_views"], r["s_tilde"]), fontsize=8)
        ax.set_xlabel("weak direction $v_3$  [$10^{-3}$ units]", fontsize=6.5)
        ax.set_ylabel("strong direction $v_1$", fontsize=6.5)
        ax.tick_params(labelsize=6)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Feasible sets $\\mathcal{F}_k=\\bigcap_i P_{ik}$, PCA frame, "
                 "equal aspect  ($\\epsilon$ = %g px)" % eps, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUTDIR, "feasible_sets_%s.png" % tag), dpi=150)

    # ---------------- figure: spread vs track length ---------------------
    fig2, ax = plt.subplots(figsize=(5.6, 4.0))
    for r in good:
        ax.loglog(r["n_views"], r["s_tilde"], "o", color="#c05621", ms=6)
        ax.annotate(str(r["point3D_id"]), (r["n_views"], r["s_tilde"]),
                    fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("track length  |views($k$)|")
    ax.set_ylabel("$\\tilde s_k$  =  max interval extent / $d_k$")
    ax.set_title("Normalised 3D uncertainty vs. number of views "
                 "($\\epsilon$ = %g px)" % eps, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "spread_vs_views_%s.png" % tag), dpi=150)
    print("wrote figures -> %s" % OUTDIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eps1")
