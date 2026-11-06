"""Shared pieces for the set-based scripts (pair triangulation, image-3 back-projection).

Conventions
    world -> camera:  q = R (X - C),  depth q_3 > 0
    window of a detection (undistorted, normalised coordinates x = (u - p_x) / f):
        [x - eps/f, x + eps/f] x [y - eps/f, y + eps/f]
    as four faces n^T q <= 0 (unit normals) plus depth q_3 >= z_min
    gauge: camera 1 at the origin with R = I, x-coordinate of camera 2's centre = 1
"""
import os
import json
import math
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from colmap_io import read_model

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.abspath(os.path.join(HERE, ".."))                 # nerf_synthetic/airplane
MODEL = os.path.join(SCENE, "COLMAP results")
RESULTS = os.path.join(HERE, "results")


def results_dir(scheme):
    """results/ for the fixed tolerance, results_<scheme>/ otherwise."""
    d = RESULTS if scheme == "fixed" else os.path.join(HERE, "results_%s" % scheme)
    os.makedirs(d, exist_ok=True)
    return d
FLIP = np.diag([1.0, -1.0, -1.0])                                  # OpenGL -> OpenCV camera axes


# ---- geometry -----------------------------------------------------------------------------
def expR(w):
    return Rotation.from_rotvec(w).as_matrix()


def undistort(u, v, intr, iters=30):
    """pixel -> undistorted normalised coordinates (SIMPLE_RADIAL / RADIAL, fixed-point iteration)."""
    fx, fy, cx, cy, k1, k2 = intr
    xd = np.array([(u - cx) / fx, (v - cy) / fy])
    x = xd.copy()
    for _ in range(iters):
        r2 = x @ x
        x = xd / (1 + k1 * r2 + k2 * r2 * r2)
    return x


def box_normals(xn, f, eps):
    """unit normals (4, 3) of the window's pyramid faces in camera coordinates: n^T q <= 0."""
    a_p, a_m = xn[0] + eps / f[0], xn[0] - eps / f[0]
    b_p, b_m = xn[1] + eps / f[1], xn[1] - eps / f[1]
    N = np.array([[1, 0, -a_p], [-1, 0, a_m], [0, 1, -b_p], [0, -1, b_m]], float)
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def window_rows(R, C, xn, f, eps, zmin):
    """rows (G, h) with G X <= h for a camera with KNOWN pose (R, C): 4 faces + depth."""
    N = box_normals(xn, f, eps)
    G = [n @ R for n in N] + [-R[2]]
    H = [g @ C for g in G[:4]] + [-zmin - R[2] @ C]
    return np.array(G), np.array(H)


def chebyshev(A, b):
    """centre and radius of the largest ball in {x : A x <= b}; (None, 0) if empty."""
    nrm = np.linalg.norm(A, axis=1)
    n = A.shape[1]
    r = linprog(np.r_[np.zeros(n), -1], A_ub=np.c_[A, nrm], b_ub=b,
                bounds=[(None, None)] * n + [(0, None)], method="highs")
    return (r.x[:n], r.x[n]) if r.status == 0 and r.x[n] > 1e-12 else (None, 0.0)


def box_of(A, b):
    """axis-aligned extent of {x : A x <= b} (exact, 2 LPs per coordinate); None if empty."""
    n = A.shape[1]
    lo, hi = np.zeros(n), np.zeros(n)
    for d in range(n):
        e = np.zeros(n)
        e[d] = 1
        r1 = linprog(e, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        r2 = linprog(-e, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        if r1.status != 0 or r2.status != 0:
            return None
        lo[d], hi[d] = r1.fun, -r2.fun
    return lo, hi


def feasible(A, b, bounds=None):
    """phase-1 LP: min t s.t. A x - t <= b; feasible iff t* ~ 0."""
    n = A.shape[1]
    bounds = bounds if bounds is not None else [(None, None)] * n
    A2 = np.hstack([A, -np.ones((len(A), 1))])
    c = np.zeros(n + 1)
    c[-1] = 1
    r = linprog(c, A_ub=A2, b_ub=b, bounds=bounds + [(0, None)], method="highs")
    return bool(r.status == 0 and r.fun <= 1e-8)


# ---- data ----------------------------------------------------------------------------------
class Scene:
    """COLMAP model in the gauge of the initial pair, with undistorted detections."""

    def __init__(self, pair=(162, 15), extra=(17,)):
        self.cams, self.imgs, self.pts = read_model(MODEL)
        self.i1, self.i2 = pair
        ids = list(pair) + list(extra)
        self.intr = {i: self.cams[self.imgs[i].camera_id].unpack() for i in ids}
        self.f = {i: (self.intr[i][0], self.intr[i][1]) for i in ids}
        R1, c1 = self.imgs[self.i1].R, self.imgs[self.i1].center
        self.scale = (R1 @ (self.imgs[self.i2].center - c1))[0]
        self.R1, self.c1 = R1, c1
        self.det, self.px = {}, {}
        for i in ids:
            d, q = {}, {}
            im = self.imgs[i]
            for j in np.flatnonzero(im.p3d_ids >= 0):
                d.setdefault(int(im.p3d_ids[j]), []).append(undistort(*im.xys[j], self.intr[i]))
                q.setdefault(int(im.p3d_ids[j]), []).append(im.xys[j].copy())
            self.det[i], self.px[i] = d, q

    def colmap_residual(self, i, k, uv):
        """|pi(theta_hat_i, X_hat_k) - uv|_inf in pixels, COLMAP camera model (with distortion)."""
        im = self.imgs[i]
        fx, fy, cx, cy, k1, k2 = self.intr[i]
        q = im.R @ self.pts[k].xyz + im.tvec
        x, y = q[0] / q[2], q[1] / q[2]
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        return float(max(abs(fx * x * d + cx - uv[0]), abs(fy * y * d + cy - uv[1])))

    def eps_list(self, i, k, eps, scheme="fixed", kappa=1.5, margin=1.0, cap=3.0):
        """per-detection tolerances for landmark k in image i."""
        out = []
        for uv in self.px[i][k]:
            r = self.colmap_residual(i, k, uv)
            if scheme == "mult":
                e = max(eps, kappa * r)
            elif scheme == "add":
                e = max(eps, r + margin)
            else:
                e = eps
            out.append(min(e, cap * eps))
        return out

    def pose(self, i):
        """COLMAP pose of image i in the gauge: (R, C)."""
        if i == self.i1:
            return np.eye(3), np.zeros(3)
        return self.imgs[i].R @ self.R1.T, self.R1 @ (self.imgs[i].center - self.c1) / self.scale

    def point(self, k):
        return self.R1 @ (self.pts[k].xyz - self.c1) / self.scale

    def gt_pose(self, i):
        """ground-truth pose of image i in the same gauge rule (transforms_train/val.json)."""
        gt = {}
        for split, suf in (("train", ""), ("val", "_val")):
            J = json.load(open(os.path.join(SCENE, "transforms_%s.json" % split)))
            for fr in J["frames"]:
                M = np.array(fr["transform_matrix"], float)
                gt[os.path.basename(fr["file_path"]) + suf + ".png"] = ((M[:3, :3] @ FLIP).T, M[:3, 3])
        Rg1, cg1 = gt[self.imgs[self.i1].name]
        sc = (Rg1 @ (gt[self.imgs[self.i2].name][1] - cg1))[0]
        Rg, cg = gt[self.imgs[i].name]
        return Rg @ Rg1.T, Rg1 @ (cg - cg1) / sc
