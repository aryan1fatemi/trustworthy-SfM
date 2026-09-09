"""Per-landmark size summary of the feasible set F_k -> CSV + plots.

Uses the exact polyhedral solver (exact_polytope.py), not the grid, so it runs
in ~20 ms per landmark and can cover thousands of points.

Size is reported three ways, because they answer different questions:

  volume        how much 3D space the landmark could occupy      (units^3)
  diameter      how far apart two consistent positions can be    (units)
  displacement  how far the truth can be from COLMAP's X_k       (units)

The oriented box (OBB) is built with its FIRST axis along the diameter
direction, then PCA in the orthogonal complement.  That makes obb_len1 exactly
the diameter and the box deterministic -- unlike a plain PCA box, whose longest
side can come out shorter than the true diameter.
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_model
from feasible_set import build_views, max_inf_residual, MODEL, OUTDIR
import exact_polytope as ex


def oriented_box(V):
    """OBB with axis 1 along the diameter. Returns (lengths long->short, axes)."""
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    a1 = V[j] - V[i]
    diameter = float(np.linalg.norm(a1))
    a1 = a1 / diameter

    C = V - V.mean(0)
    P = C - np.outer(C @ a1, a1)                 # project out the diameter axis
    # PCA of the residual gives the other two axes
    _, _, Wt = np.linalg.svd(P, full_matrices=False)
    a2 = Wt[0] - (Wt[0] @ a1) * a1
    a2 /= np.linalg.norm(a2)
    a3 = np.cross(a1, a2)

    A = np.vstack([a1, a2, a3])
    Q = C @ A.T
    lens = Q.max(0) - Q.min(0)
    lens[0] = diameter                            # exact, by construction
    return lens, A, diameter


def analyse_point(pid, pts, imgs, pose, eps):
    p = pts[pid]
    views = build_views(p, imgs, pose)
    centres = np.array([-R.T @ t for (R, t, _, _) in views])
    dist = np.linalg.norm(p.xyz - centres, axis=1)
    d_k = float(dist.mean())

    row = {
        "point3D_id": pid,
        "n_views": len(views),
        "eps_px": eps,
        "d_k": d_k,
        "d_k_min": float(dist.min()),
        "d_k_max": float(dist.max()),
        "colmap_error_px": float(p.error),
        "max_inf_residual_px": float(max_inf_residual(views, p.xyz)[0]),
        "X_k_x": p.xyz[0], "X_k_y": p.xyz[1], "X_k_z": p.xyz[2],
    }

    H = ex.halfspaces(views, eps)
    ctr, inradius = ex.chebyshev_center(H)
    if ctr is None or inradius <= 0:
        row["empty"] = 1
        return row
    row["empty"] = 0

    try:
        V, volume, area = ex.vertices_and_volume(H, ctr)
    except Exception:
        row["empty"] = 0
        row["degenerate"] = 1
        return row

    lens, A, diameter = oriented_box(V)
    lo, hi = V.min(0), V.max(0)
    disp = float(np.linalg.norm(V - p.xyz, axis=1).max())

    row.update({
        # --- size ---
        "volume": volume,
        "surface_area": area,
        "diameter": diameter,
        "max_displacement_from_X_k": disp,
        "equivalent_radius": float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)),
        "inradius": inradius,
        # --- oriented bounding box (axis 1 = diameter direction) ---
        "obb_len1": lens[0], "obb_len2": lens[1], "obb_len3": lens[2],
        "obb_diagonal": float(np.linalg.norm(lens)),
        "obb_volume": float(np.prod(lens)),
        "fill_ratio": float(volume / max(np.prod(lens), 1e-30)),
        "anisotropy": float(lens[0] / max(lens[2], 1e-15)),
        "flatness": float(lens[1] / max(lens[2], 1e-15)),
        "elongation": float(lens[0] / max(lens[1], 1e-15)),
        # --- weak direction (the long axis of the box, in world coords) ---
        "v3_x": A[0][0], "v3_y": A[0][1], "v3_z": A[0][2],
        # --- axis-aligned hull, for comparison with the earlier runs ---
        "aabb_ext_x": float(hi[0] - lo[0]),
        "aabb_ext_y": float(hi[1] - lo[1]),
        "aabb_ext_z": float(hi[2] - lo[2]),
        "aabb_max_ext": float((hi - lo).max()),
        # --- normalised, scale-free ---
        "s_diameter": diameter / d_k,
        "s_displacement": disp / d_k,
        "s_aabb": float((hi - lo).max()) / d_k,
        "n_vertices": int(V.shape[0]),
        "n_halfspaces": int(H.shape[0]),
    })
    return row


COLS = ["point3D_id", "n_views", "empty", "eps_px", "d_k", "d_k_min", "d_k_max",
        "colmap_error_px", "max_inf_residual_px",
        "volume", "diameter", "max_displacement_from_X_k", "equivalent_radius",
        "inradius", "surface_area",
        "obb_len1", "obb_len2", "obb_len3", "obb_diagonal", "obb_volume",
        "fill_ratio", "anisotropy", "flatness", "elongation",
        "v3_x", "v3_y", "v3_z",
        "aabb_ext_x", "aabb_ext_y", "aabb_ext_z", "aabb_max_ext",
        "s_diameter", "s_displacement", "s_aabb",
        "n_vertices", "n_halfspaces", "X_k_x", "X_k_y", "X_k_z"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=2000,
                    help="random sample size; 0 = every landmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    tag = args.tag or ("eps%g_n%s" % (args.eps, args.n or "all"))

    print("reading model ...", flush=True)
    cams, imgs, pts = read_model(MODEL)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack())
            for i, im in imgs.items()}

    ids = np.array(sorted(pts))
    if args.n and args.n < len(ids):
        ids = np.random.default_rng(args.seed).choice(ids, args.n, replace=False)
        ids.sort()
    print("analysing %d landmarks at eps = %g px ..." % (len(ids), args.eps),
          flush=True)

    rows, t0 = [], time.time()
    for n, pid in enumerate(ids):
        rows.append(analyse_point(int(pid), pts, imgs, pose, args.eps))
        if (n + 1) % 500 == 0:
            print("  %5d / %d   (%.1f s)" % (n + 1, len(ids), time.time() - t0),
                  flush=True)

    path = os.path.join(OUTDIR, "sets_%s.csv" % tag)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    print("\nwrote %s  (%d rows, %.1f s)" % (path, len(rows), time.time() - t0))

    ok = [r for r in rows if not r["empty"] and "diameter" in r]
    print("non-empty: %d / %d  (%.1f%%)" % (len(ok), len(rows),
                                            100 * len(ok) / len(rows)))
    if ok:
        d = np.array([r["diameter"] for r in ok])
        v = np.array([r["volume"] for r in ok])
        s = np.array([r["s_diameter"] for r in ok])
        for name, arr in (("diameter", d), ("volume", v), ("s_diameter", s)):
            q = np.percentile(arr, [5, 25, 50, 75, 95])
            print("  %-11s p5 %.3e  p25 %.3e  med %.3e  p75 %.3e  p95 %.3e"
                  % (name, *q))
    return path


if __name__ == "__main__":
    main()
