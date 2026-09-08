"""Brute-force computation of the SLUE feasible set

    F_k = { X in R^3 : for all i in views(k), ||pi(theta_i, X) - y_ik||_inf <= eps }

Cameras are FIXED at their COLMAP values (theta_i = theta_i_hat), so F_k is an
intersection of back-projection cones P_ik and is convex.

Method
------
The set is extremely anisotropic (thin across the rays, long along the weak
direction v3), so a naive axis-aligned cube of samples is hopeless.  The
membership test stays brute force -- every sample is tested against the exact
non-linear projection of every view -- but the SEARCH REGION is chosen with a
linearised outer estimate:

  1. Jacobian A = d pi / dX stacked over views at X_k, residual b = pi(X_k) - y.
     The linearised set {d : |A d + b|_inf <= eps} is a polytope; six LPs give
     its support along the right singular vectors v1,v2,v3 of A -> oriented box.
  2. Inflate that box, lay down an n^3 oriented grid, test exactly.
  3. If any feasible sample touches the shell of the box, the box was too small
     -> inflate and repeat.  That check is what makes the enumeration an outer
     bound rather than a guess.
  4. Refine: a second n^3 grid tight around the feasible bounding box.

Reported per landmark: axis-aligned interval hull [X_lo, X_hi], volume,
principal (PCA) extents, weak direction v3, and the normalised spread
s_k = max_a (X_hi - X_lo)_a / d_k with d_k the mean viewing distance.
"""
import os
import sys
import json
import csv
import argparse
import time
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_model, project

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(MODEL, "feasible_sets")


# --------------------------------------------------------------------------
# residuals / Jacobian
# --------------------------------------------------------------------------
def residuals(views, X):
    """pi_i(X) - y_ik for every view; X (M,3) -> (M, n_views, 2)."""
    X = np.atleast_2d(X)
    out = np.empty((X.shape[0], len(views), 2))
    for j, (R, t, intr, y) in enumerate(views):
        uv, z = project(R, t, intr, X)
        r = uv - y
        r[z <= 0] = np.inf                     # behind the camera: infeasible
        out[:, j, :] = r
    return out


def max_inf_residual(views, X):
    return np.abs(residuals(views, X)).max(axis=(1, 2))


def jacobian(views, X0, h=1e-5):
    """Central-difference d pi_i / dX, stacked -> (2*n_views, 3)."""
    A = np.empty((2 * len(views), 3))
    for a in range(3):
        d = np.zeros(3)
        d[a] = h
        rp = residuals(views, X0 + d)[0]
        rm = residuals(views, X0 - d)[0]
        A[:, a] = ((rp - rm) / (2 * h)).ravel()
    return A


# --------------------------------------------------------------------------
# step 1: linearised outer box, oriented along the singular vectors of A
# --------------------------------------------------------------------------
def linear_box(A, b, eps, Rmax):
    """Support of {d : |A d + b|_inf <= eps, |d|<=Rmax} along v1,v2,v3."""
    V = np.linalg.svd(A, full_matrices=True)[2]           # rows = v1,v2,v3
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([eps - b, eps + b])
    lo = np.empty(3)
    hi = np.empty(3)
    capped = False
    bounds = [(-Rmax, Rmax)] * 3
    for j in range(3):
        for sgn in (1.0, -1.0):
            r = linprog(-sgn * V[j], A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                        method="highs")
            if not r.success:                  # infeasible / numerical trouble
                val = sgn * Rmax
                capped = True
            else:
                val = float(V[j] @ r.x)
            if sgn > 0:
                hi[j] = val
            else:
                lo[j] = val
        if hi[j] - lo[j] >= 1.9 * Rmax:
            capped = True
    return V, lo, hi, capped


# --------------------------------------------------------------------------
# steps 2-4: brute-force enumeration on an oriented grid
# --------------------------------------------------------------------------
def grid_sweep(views, X0, V, ctr, half, eps, n):
    """Test an n^3 oriented grid.  Returns feasible points (world and V frame),
    whether any of them sits on the shell of the box, and the cell volume."""
    axes = [np.linspace(ctr[j] - half[j], ctr[j] + half[j], n) for j in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)   # V frame
    W = X0 + G @ V                                                       # world

    ok = np.ones(W.shape[0], bool)
    for (R, t, intr, y) in views:                    # per view, to bound memory
        uv, z = project(R, t, intr, W)
        ok &= (z > 0) & (np.abs(uv - y).max(axis=1) <= eps)
        if not ok.any():
            break

    cell = float(np.prod(2 * half / (n - 1)))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), False, cell

    ijk = np.stack(np.unravel_index(idx, (n, n, n)), -1)
    shell = bool((ijk == 0).any() or (ijk == n - 1).any())
    return W[idx], G[idx], shell, cell


def feasible_set(views, X0, eps, n_coarse=81, n_fine=101, max_inflate=5):
    """Full pipeline for one landmark.  Returns (feasible world points, info)."""
    centres = np.array([-R.T @ t for (R, t, _, _) in views])
    d_k = float(np.linalg.norm(X0 - centres, axis=1).mean())
    Rmax = 0.5 * d_k

    A = jacobian(views, X0)
    b = residuals(views, X0)[0].ravel()
    V, lo, hi, capped = linear_box(A, b, eps, Rmax)

    ctr = 0.5 * (lo + hi)
    half0 = np.maximum(0.5 * (hi - lo), 1e-9 * d_k)

    info = {"d_k": d_k,
            "lin_capped": bool(capped),
            "lin_extent": (hi - lo).tolist(),
            "singular_values": np.linalg.svd(A, compute_uv=False).tolist()}

    # ---- coarse pass; inflate until nothing feasible touches the shell ----
    infl = 1.6
    Wc = np.zeros((0, 3))
    Gc = np.zeros((0, 3))
    half = half0 * infl
    shell = False
    for attempt in range(max_inflate):
        half = np.minimum(half0 * infl, Rmax)
        Wc, Gc, shell, _ = grid_sweep(views, X0, V, ctr, half, eps, n_coarse)
        if Wc.shape[0] == 0:
            # either F_k is empty, or it slipped between grid nodes
            if max_inf_residual(views, X0)[0] <= eps:
                Wc = X0[None, :]
                Gc = np.zeros((1, 3))
                shell = False
                break
            infl *= 2.5
            continue
        if not shell:
            break
        infl *= 2.5
    info["coarse_inflate"] = infl
    info["coarse_hits"] = int(Wc.shape[0])
    info["coarse_touched_shell"] = bool(shell)
    if Wc.shape[0] == 0:
        info["empty"] = True
        return None, info
    info["empty"] = False

    # ---- fine pass, tight around the coarse feasible region ---------------
    g_lo, g_hi = Gc.min(0), Gc.max(0)
    ctr2 = 0.5 * (g_lo + g_hi)
    pad = 2 * half / (n_coarse - 1)                      # one coarse cell
    Wf = np.zeros((0, 3))
    shell2 = True
    cell = float(np.prod(2 * half / (n_coarse - 1)))
    for _ in range(4):                     # grow the pad until nothing clips
        half2 = np.maximum(0.5 * (g_hi - g_lo) + pad, pad)
        Wf, Gf, shell2, cell = grid_sweep(views, X0, V, ctr2, half2, eps, n_fine)
        if Wf.shape[0] and not shell2:
            g_lo, g_hi = Gf.min(0), Gf.max(0)
            ctr2 = 0.5 * (g_lo + g_hi)
            break
        if Wf.shape[0]:
            g_lo, g_hi = Gf.min(0), Gf.max(0)
            ctr2 = 0.5 * (g_lo + g_hi)
        pad = pad * 3.0
    if Wf.shape[0] == 0:
        Wf, cell, shell2 = Wc, float(np.prod(2 * half / (n_coarse - 1))), shell
    info["fine_touched_shell"] = bool(shell2)
    info["n_feasible"] = int(Wf.shape[0])
    info["cell_volume"] = float(cell)
    return Wf, info


def summarise(Wf, X0, views, eps, info):
    """Interval hull, volume, PCA extents, weak direction."""
    lo, hi = Wf.min(0), Wf.max(0)
    ext = hi - lo
    d_k = info["d_k"]

    C = Wf - Wf.mean(0)
    if Wf.shape[0] >= 4:
        Vt = np.linalg.svd(C, full_matrices=False)[2]
        P = C @ Vt.T
        pc_ext = P.max(0) - P.min(0)
        order = np.argsort(-pc_ext)                      # long -> short
        pc_ext, Vt = pc_ext[order], Vt[order]
    else:
        pc_ext = np.zeros(3)
        Vt = np.eye(3)

    return {
        "X_colmap": X0.tolist(),
        "centroid": Wf.mean(0).tolist(),
        "interval_hull_lo": lo.tolist(),
        "interval_hull_hi": hi.tolist(),
        "interval_extent": ext.tolist(),
        "max_interval_extent": float(ext.max()),
        "s_tilde": float(ext.max() / d_k),
        "volume": float(info["n_feasible"] * info["cell_volume"]),
        "pca_extent": pc_ext.tolist(),
        "pca_axes_long_to_short": Vt.tolist(),
        "v3_weak_direction": Vt[0].tolist(),      # axis of LARGEST extent
        "anisotropy": float(pc_ext[0] / max(pc_ext[2], 1e-12)),
        "d_k": d_k,
        "n_views": len(views),
        "eps": eps,
        "n_feasible_samples": info["n_feasible"],
        "reliable": bool((not info["fine_touched_shell"])
                         and info["n_feasible"] > 20),
    }


# --------------------------------------------------------------------------
def build_views(pt, imgs, pose):
    return [(*pose[iid], imgs[iid].xys[idx])
            for iid, idx in zip(pt.image_ids, pt.point2D_idxs)]


def select_landmarks(pts, imgs, pose, eps, n_per_bin=3, seed=0):
    """Stratify by track length; keep only tracks that are certainly non-empty."""
    rng = np.random.default_rng(seed)
    bins = [(2, 2), (3, 4), (5, 9), (10, 20), (21, 10 ** 6)]
    chosen = []
    ids = np.array(sorted(pts))
    tl = np.array([pts[i].track_length for i in ids])
    for lo, hi in bins:
        cand = ids[(tl >= lo) & (tl <= hi)]
        rng.shuffle(cand)
        got = 0
        for pid in cand:
            v = build_views(pts[pid], imgs, pose)
            if max_inf_residual(v, pts[pid].xyz)[0] <= eps:   # X_k itself feasible
                chosen.append(int(pid))
                got += 1
            if got == n_per_bin:
                break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1.0, help="pixel tolerance")
    ap.add_argument("--n-per-bin", type=int, default=3)
    ap.add_argument("--n-coarse", type=int, default=81)
    ap.add_argument("--n-fine", type=int, default=101)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    tag = args.tag or ("eps%g" % args.eps)

    print("reading model ...", flush=True)
    cams, imgs, pts = read_model(MODEL)
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack())
            for i, im in imgs.items()}

    ids = args.ids or select_landmarks(pts, imgs, pose, args.eps, args.n_per_bin)
    print("eps = %g px, %d landmarks: %s\n" % (args.eps, len(ids), ids), flush=True)

    rows, clouds = [], {}
    hdr = ("%7s %5s %7s %7s %10s %9s %10s %7s %8s  ok"
           % ("pid", "views", "d_k", "maxres", "extent_max", "s_tilde",
              "volume", "aniso", "nsamp"))
    print(hdr)
    print("-" * len(hdr))
    for pid in ids:
        p = pts[pid]
        views = build_views(p, imgs, pose)
        t0 = time.time()
        Wf, info = feasible_set(views, p.xyz, args.eps, args.n_coarse, args.n_fine)
        if Wf is None:
            print("%7d %5d   ---- F_k empty at eps=%g ----" % (pid, len(views), args.eps))
            rows.append({"point3D_id": int(pid), "empty": True,
                         "n_views": len(views), "eps": args.eps})
            continue
        s = summarise(Wf, p.xyz, views, args.eps, info)
        s["point3D_id"] = int(pid)
        s["colmap_error_px"] = float(p.error)
        s["max_inf_residual_at_Xk"] = float(max_inf_residual(views, p.xyz)[0])
        s["empty"] = False
        s["seconds"] = round(time.time() - t0, 2)
        s["diagnostics"] = info
        rows.append(s)
        clouds[pid] = Wf
        print("%7d %5d %7.3f %7.3f %10.3e %9.2e %10.2e %7.1f %8d  %s"
              % (pid, s["n_views"], s["d_k"], s["max_inf_residual_at_Xk"],
                 s["max_interval_extent"], s["s_tilde"], s["volume"],
                 s["anisotropy"], s["n_feasible_samples"],
                 "y" if s["reliable"] else "CHECK"), flush=True)

    with open(os.path.join(OUTDIR, "feasible_sets_%s.json" % tag), "w") as fh:
        json.dump({"eps": args.eps, "model": MODEL, "landmarks": rows}, fh, indent=2)

    cols = ["point3D_id", "empty", "n_views", "eps", "d_k", "colmap_error_px",
            "max_inf_residual_at_Xk", "max_interval_extent", "s_tilde", "volume",
            "anisotropy", "n_feasible_samples", "reliable", "seconds"]
    with open(os.path.join(OUTDIR, "feasible_sets_%s.csv" % tag), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    np.savez_compressed(os.path.join(OUTDIR, "clouds_%s.npz" % tag),
                        **{str(k): v for k, v in clouds.items()})
    print("\nwrote -> %s" % OUTDIR)


if __name__ == "__main__":
    main()
