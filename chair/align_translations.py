"""
Compare COLMAP camera translations (positions) against the NeRF-synthetic
ground truth, after a Sim(3) alignment.

Why Sim(3): a COLMAP reconstruction is only defined up to an arbitrary
similarity transform (scale + rotation + translation) relative to the true
world frame. We estimate that transform with the Umeyama algorithm (WITH
scaling) on the camera centers, apply it to the COLMAP centers, then measure
the residual ||gt - aligned_colmap|| per image.

Why we can ignore the OpenGL<->OpenCV convention flip here: it only rotates
each camera's *local* axes (where it looks), not the camera *center* in world
space. Camera position is convention-independent, so it does not affect a
translation-only comparison. (It WOULD matter for a rotation comparison.)

Camera center in world coordinates:
  - GT (Blender): transform_matrix is camera-to-world; center = its [:3, 3].
  - COLMAP: images.txt gives world-to-camera (R, t); center = -R^T t.
"""

import os
import json
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLMAP_DIR = os.path.join(HERE, "COLMAP results")
IMAGES_TXT = os.path.join(COLMAP_DIR, "images.txt")
OUT_CSV = os.path.join(HERE, "translation_comparison.csv")


def quat_to_rotmat(qw, qx, qy, qz):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix (world-to-camera)."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def load_colmap_centers(path):
    """Return {image_name: camera_center_world (3,)} from COLMAP images.txt."""
    centers = {}
    with open(path, "r") as f:
        lines = f.readlines()
    # Data starts after comment lines; every image is TWO lines (pose, then 2D pts).
    data = [ln for ln in lines if not ln.startswith("#")]
    for i in range(0, len(data), 2):
        parts = data[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        R = quat_to_rotmat(qw, qx, qy, qz)      # world-to-camera
        t = np.array([tx, ty, tz])
        center = -R.T @ t                        # camera center in world
        centers[name] = center
    return centers


def load_gt_centers():
    """Return {image_name: camera_center_world (3,)} from the transforms JSONs.

    Maps GT file_path to the merged images/ filenames used for COLMAP:
      ./train/r_N -> r_N.png      ./val/r_N -> r_N_val.png
    """
    centers = {}
    for split, suffix in [("train", ""), ("val", "_val")]:
        jpath = os.path.join(HERE, f"transforms_{split}.json")
        with open(jpath, "r") as f:
            meta = json.load(f)
        for frame in meta["frames"]:
            stem = os.path.basename(frame["file_path"])       # r_N
            name = f"{stem}{suffix}.png"                        # r_N.png / r_N_val.png
            M = np.array(frame["transform_matrix"], dtype=np.float64)  # camera-to-world
            centers[name] = M[:3, 3]
    return centers


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform mapping src -> dst.

    src, dst: (N, 3). Returns (s, R, t) s.t. dst ~= s * R @ src + t.
    """
    n, dim = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


def main():
    colmap = load_colmap_centers(IMAGES_TXT)
    gt = load_gt_centers()

    names = sorted(set(colmap) & set(gt))
    missing = sorted(set(gt) - set(colmap))
    print(f"GT images: {len(gt)} | COLMAP registered: {len(colmap)} | matched: {len(names)}")
    if missing:
        print(f"Not registered by COLMAP ({len(missing)}): {missing}")

    src = np.array([colmap[n] for n in names])   # COLMAP centers
    dst = np.array([gt[n] for n in names])        # GT centers

    s, R, t = umeyama(src, dst, with_scale=True)
    aligned = (s * (R @ src.T)).T + t
    residuals = np.linalg.norm(aligned - dst, axis=1)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "split",
            "colmap_x", "colmap_y", "colmap_z",           # raw COLMAP center (arbitrary frame)
            "gt_x", "gt_y", "gt_z",                        # GT center (Blender world)
            "aligned_x", "aligned_y", "aligned_z",         # COLMAP center after Sim(3)->GT
            "residual",                                    # ||aligned - gt||
        ])
        for i, n in enumerate(names):
            split = "val" if n.endswith("_val.png") else "train"
            w.writerow([
                n, split,
                *np.round(src[i], 6),
                *np.round(dst[i], 6),
                *np.round(aligned[i], 6),
                round(float(residuals[i]), 6),
            ])

    # GT scene scale, so the residual has a reference magnitude.
    gt_extent = np.linalg.norm(dst.max(axis=0) - dst.min(axis=0))
    print("\n=== Sim(3) alignment (COLMAP -> GT) ===")
    print(f"scale s      = {s:.6f}")
    print(f"translation  = {np.round(t, 4)}")
    print("\n=== Translation residual (world units) ===")
    print(f"mean   = {residuals.mean():.6f}")
    print(f"median = {np.median(residuals):.6f}")
    print(f"max    = {residuals.max():.6f}  ({names[int(residuals.argmax())]})")
    print(f"RMSE   = {np.sqrt((residuals**2).mean()):.6f}")
    print(f"GT scene extent (diag) = {gt_extent:.4f}  ->  mean residual = "
          f"{100*residuals.mean()/gt_extent:.3f}% of scene")
    print(f"\nCSV written: {OUT_CSV}")


if __name__ == "__main__":
    main()
