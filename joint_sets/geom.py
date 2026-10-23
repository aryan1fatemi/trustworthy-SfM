"""Geometry and data loading for joint_sets.py (self-contained).

Conventions
    world -> camera:  q = R (X - C), depth q_3 > 0; pinhole after undistortion
    window of a detection: +-eps pixels around it, written as 4 faces n^T q <= 0 plus depth
    gauge: camera 1 at the origin with R = I, x-coordinate of camera 2's centre = 1
"""
import os
import re
import json
import math
import numpy as np

from colmap_io import read_model

FLIP = np.diag([1.0, -1.0, -1.0])           # OpenGL (Blender / NeRF) -> OpenCV camera axes


def undistort(u, v, intr, iters=30):
    fx, fy, cx, cy, k1, k2 = intr
    xd = np.array([(u - cx) / fx, (v - cy) / fy])
    x = xd.copy()
    for _ in range(iters):
        r2 = x @ x
        x = xd / (1 + k1 * r2 + k2 * r2 * r2)
    return x


def box_normals(xn, f, eps):
    """unit normals (4, 3) of a window's pyramid faces in camera coordinates: n^T q <= 0."""
    a_p, a_m = xn[0] + eps / f[0], xn[0] - eps / f[0]
    b_p, b_m = xn[1] + eps / f[1], xn[1] - eps / f[1]
    N = np.array([[1, 0, -a_p], [-1, 0, a_m], [0, 1, -b_p], [0, -1, b_m]], float)
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def run_name(scene, model=None):
    """output name: dataset folder, plus the model folder when it is not the default "COLMAP results"."""
    name = os.path.basename(os.path.normpath(scene))
    if model and os.path.basename(os.path.normpath(model)) != "COLMAP results":
        name += "_" + os.path.basename(os.path.normpath(model)).replace(" ", "_")
    return name


def registration_order(scene, imgs, n, model=None):
    """COLMAP's order (initial pair, then registered images) from colmap_log.txt (next to the model if present,
    else in the scene folder), else by covisibility."""
    logs = ([os.path.join(model, "colmap_log.txt")] if model else []) + [os.path.join(scene, "colmap_log.txt")]
    log = next((l for l in logs if os.path.exists(l)), "")
    if log:
        lines = open(log).read().splitlines()
        inits = [(n_, m) for n_, l in enumerate(lines) for m in [re.search(r"Initializing with image pair #(\d+) and #(\d+)", l)] if m]
        if inits:
            start, m = inits[-1]
            order = [int(m.group(1)), int(m.group(2))]
            for j, l in enumerate(lines[start:], start):
                r = re.match(r"Registering image #(\d+) \(\d+\)", l)
                if r:
                    blk = []
                    for l2 in lines[j + 1:j + 60]:
                        if l2.startswith("Registering image #") or "Finding good initial" in l2:
                            break
                        blk.append(l2)
                    i = int(r.group(1))
                    if not any("Could not register" in b for b in blk) and i not in order and i in imgs:
                        order.append(i)
                if len(order) >= n:
                    break
            if all(i in imgs for i in order[:2]):
                return order[:n]
    sets = {i: set(int(p) for p in im.p3d_ids[im.p3d_ids >= 0]) for i, im in imgs.items()}
    ids = list(imgs)
    best = max(((a, b) for a in ids for b in ids if a < b), key=lambda p: len(sets[p[0]] & sets[p[1]]))
    order, seen = list(best), sets[best[0]] | sets[best[1]]
    while len(order) < n:
        j = max((j for j in ids if j not in order), key=lambda j: len(sets[j] & seen))
        order.append(j)
        seen |= sets[j]
    return order


class Scene:
    def __init__(self, scene, model, cams):
        self.scene = scene
        self.cams, self.imgs, self.pts = read_model(model)
        self.i1, self.i2 = cams[0], cams[1]
        self.intr = {i: self.cams[self.imgs[i].camera_id].unpack() for i in cams}
        self.f = {i: (self.intr[i][0], self.intr[i][1]) for i in cams}
        R1, c1 = self.imgs[self.i1].R, self.imgs[self.i1].center
        self.R1, self.c1 = R1, c1
        self.scale = (R1 @ (self.imgs[self.i2].center - c1))[0]
        self.det, self.px = {}, {}
        for i in cams:
            d, q = {}, {}
            im = self.imgs[i]
            for j in np.flatnonzero(im.p3d_ids >= 0):
                d.setdefault(int(im.p3d_ids[j]), []).append(undistort(*im.xys[j], self.intr[i]))
                q.setdefault(int(im.p3d_ids[j]), []).append(im.xys[j].copy())
            self.det[i], self.px[i] = d, q

    def pose(self, i):
        if i == self.i1:
            return np.eye(3), np.zeros(3)
        return self.imgs[i].R @ self.R1.T, self.R1 @ (self.imgs[i].center - self.c1) / self.scale

    def point(self, k):
        return self.R1 @ (self.pts[k].xyz - self.c1) / self.scale

    def eps_list(self, i, k, eps, scheme="fixed", kappa=1.5, margin=1.0, cap=3.0):
        """per-detection tolerance: fixed eps, max(eps, kappa r), or max(eps, r + margin); r = COLMAP residual."""
        out = []
        for uv in self.px[i][k]:
            im = self.imgs[i]
            fx, fy, cx, cy, k1, k2 = self.intr[i]
            q = im.R @ self.pts[k].xyz + im.tvec
            x, y = q[0] / q[2], q[1] / q[2]
            r2 = x * x + y * y
            dd = 1 + k1 * r2 + k2 * r2 * r2
            r = max(abs(fx * x * dd + cx - uv[0]), abs(fy * y * dd + cy - uv[1]))
            e = {"mult": max(eps, kappa * r), "add": max(eps, r + margin)}.get(scheme, eps)
            out.append(min(e, cap * eps))
        return out

    # ---- ground truth (optional: NeRF-style transforms*.json with depth maps) ----------------------
    def load_gt(self):
        single = os.path.join(self.scene, "transforms.json")
        layouts = ([(single, "")] if os.path.exists(single) else
                   [(os.path.join(self.scene, "transforms_%s.json" % sp), suf) for sp, suf in (("train", ""), ("val", "_val"))])
        gt = {}
        for path, suf in layouts:
            if not os.path.exists(path):
                return None
            J = json.load(open(path))
            W = J.get("w", 800)
            f = J.get("fl_x", 0.5 * W / math.tan(0.5 * J["camera_angle_x"]))
            cx, cy = J.get("cx", W / 2), J.get("cy", W / 2)
            for fr in J["frames"]:
                M = np.array(fr["transform_matrix"], float)
                name = os.path.basename(fr["file_path"]) + suf + ".png"
                dp = fr.get("depth_path")
                gt[name] = dict(R=(M[:3, :3] @ FLIP).T, c=M[:3, 3].copy(),
                                K=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]),
                                depth=os.path.join(self.scene, dp[2:]) if dp else None)
        return gt

    def gt_in_gauge(self, gt, cams, landmarks, disc=0.02):
        """GT poses {i: (R, C)} and GT points {k: X} in the gauge rule (camera 1 origin, x of camera 2 = 1)."""
        g1, g2 = gt[self.imgs[self.i1].name], gt[self.imgs[self.i2].name]
        sc = (g1["R"] @ (g2["c"] - g1["c"]))[0]
        poses = {i: (gt[self.imgs[i].name]["R"] @ g1["R"].T, g1["R"] @ (gt[self.imgs[i].name]["c"] - g1["c"]) / sc)
                 for i in cams}
        lifted = {}
        cache = {}
        for k in landmarks:
            P = self.pts[k]
            for i, j in zip(P.image_ids, P.point2D_idxs):
                name = self.imgs[i].name
                g = gt.get(name)
                if g is None or g["depth"] is None or not os.path.exists(g["depth"]):
                    continue
                if name not in cache:
                    cache[name] = np.load(g["depth"])["depth"]
                D = cache[name]
                u, v = self.imgs[i].xys[j]
                x, y = u - 0.5, v - 0.5
                x0, y0 = int(math.floor(x)), int(math.floor(y))
                if x0 < 0 or y0 < 0 or x0 + 1 >= D.shape[1] or y0 + 1 >= D.shape[0]:
                    continue
                q = np.array([D[y0, x0], D[y0, x0 + 1], D[y0 + 1, x0], D[y0 + 1, x0 + 1]])
                if (q <= 0).any():
                    continue
                ax, ay = x - x0, y - y0
                d = (1 - ay) * ((1 - ax) * q[0] + ax * q[1]) + ay * ((1 - ax) * q[2] + ax * q[3])
                if (q.max() - q.min()) / d > disc:
                    continue
                Xw = g["c"] + g["R"].T @ (d * np.linalg.solve(g["K"], [u, v, 1.0]))
                lifted.setdefault(k, []).append(Xw)
        points = {k: g1["R"] @ (np.median(v, 0) - g1["c"]) / sc for k, v in lifted.items()}
        return poses, points
