"""Draw random detections with their window, a SIFT-shaped region fitted into it, and the GT pixel.

    python viz_windows.py --scene ../../chair [--n 10] [--gamma 0.25] [--seed 0]

Per panel (image crop around one detection, pixel grid visible):
    +  detection (COLMAP keypoint)
    x  GT pixel: the true camera's projection of the track's point triangulated by least squares with the TRUE
       cameras (tracks with >= 5 views)
    square   the LP window: eps = min(max(3, 1.5 r), 9) px, r = COLMAP inf-norm reprojection residual
    ellipse  tempered structure-tensor shape Sigma^gamma (Sigma ~ M^-1 at the keypoint's scale), enlarged to touch
             the square; dotted: the full shape (gamma = 1)
Only detections inside their window (r <= eps) are drawn; with mult windows that is all of them."""
import os, sys, json, math, sqlite3, argparse
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares
from scipy import ndimage
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from geom import Scene
from colmap_io import read_model

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--model", default="COLMAP results fixedK")
ap.add_argument("--db", default="database_fixedK.db")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--gamma", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--misses", action="store_true", help="only detections whose GT pixel is outside the fitted gamma ellipse")
ap.add_argument("--out", default=None, help="png path (default <scene>/figures/windows_gamma<g>.png)")
a = ap.parse_args()
S = os.path.abspath(a.scene)
cams, imgs, pts = read_model(os.path.join(S, a.model))
gt = Scene.load_gt(SimpleNamespace(scene=S))
db = sqlite3.connect("file:" + os.path.join(S, a.db) + "?mode=ro", uri=True)
dbid = {n: i for i, n in db.execute("select image_id, name from images")}


def keypoint_scale(name, j):
    r, c, d = db.execute("select rows, cols, data from keypoints where image_id=?", (dbid[name],)).fetchone()
    k = np.frombuffer(d, np.float32).reshape(r, c)[j]
    return math.sqrt(abs(k[2] * k[5] - k[3] * k[4]))


def proj_gt(g, X):
    q = g["K"] @ (g["R"] @ (X - g["c"]))
    return q[:2] / q[2]


def proj_colmap(im, X):
    fx, fy, cx, cy = cams[im.camera_id].unpack()[:4]
    q = im.R @ X + im.tvec
    return np.array([fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy])


def structure_cov(I, u, v, s):
    """Sigma ~ M^-1, M = structure tensor (derivative sigma 1 px, integration sigma max(1.5, scale))."""
    si = max(1.5, s)
    h = int(4 * si + 4)
    y0, x0 = int(round(v - 0.5)), int(round(u - 0.5))
    P = np.pad(I, h + 4, mode="edge")[y0 + 4:y0 + 2 * h + 5, x0 + 4:x0 + 2 * h + 5]
    Ix, Iy = ndimage.gaussian_filter(P, 1, order=(0, 1)), ndimage.gaussian_filter(P, 1, order=(1, 0))
    J = [ndimage.gaussian_filter(t, si)[h, h] for t in (Ix * Ix, Ix * Iy, Iy * Iy)]
    return np.linalg.inv(np.array([[J[0], J[1]], [J[1], J[2]]]))


def fitted(Sig, eps, gamma):
    """(half-axes, angle deg, scale) of the ellipse Sigma^gamma enlarged until it touches the eps box."""
    w, V = np.linalg.eigh(Sig)
    Sg = V @ np.diag(w ** gamma) @ V.T
    s = eps / math.sqrt(max(Sg[0, 0], Sg[1, 1]))
    wg, Vg = np.linalg.eigh(Sg)
    return s * np.sqrt(wg[::-1]), math.degrees(math.atan2(Vg[1, 1], Vg[0, 1])), Sg, s


rng = np.random.default_rng(a.seed)
tracks = [k for k, P in pts.items() if sum(imgs[i].name in gt for i in P.image_ids) >= 5]
picked, tried, _grey = [], 0, {}
while len(picked) < a.n and tried < 20000:
    tried += 1
    k = tracks[rng.integers(len(tracks))]
    P = pts[k]
    obs = [(i, j) for i, j in zip(P.image_ids, P.point2D_idxs) if imgs[i].name in gt]
    f = lambda X: np.concatenate([proj_gt(gt[imgs[i].name], X) - imgs[i].xys[j] for i, j in obs])
    Xg = least_squares(f, P.xyz, method="lm").x
    i, j = obs[rng.integers(len(obs))]
    im = imgs[i]
    uv = im.xys[j]
    r = float(np.max(np.abs(uv - proj_colmap(im, P.xyz))))
    eps = min(max(3.0, 1.5 * r), 9.0)
    if r > eps or any(d["k"] == k and d["name"] == im.name for d in picked):
        continue
    if a.misses:
        name = im.name
        if name not in _grey:
            _grey[name] = np.asarray(Image.open(os.path.join(S, "images", name)).convert("RGB"), float).mean(-1) / 255.0
        Sig = structure_cov(_grey[name], *uv, keypoint_scale(name, j))
        _, _, Sg, sc = fitted(Sig, eps, a.gamma)
        e = proj_gt(gt[name], Xg) - uv
        if e @ np.linalg.solve(Sg, e) <= sc ** 2:
            continue
    picked.append(dict(k=k, name=im.name, uv=uv, gtuv=proj_gt(gt[im.name], Xg), r=r, eps=eps,
                       s=keypoint_scale(im.name, j), views=len(obs)))

cols = 5
rows = math.ceil(len(picked) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.7 * rows))
cache = {}
for ax, d in zip(np.ravel(axes), picked):
    if d["name"] not in cache:
        I = np.asarray(Image.open(os.path.join(S, "images", d["name"])).convert("RGB"), float) / 255.0
        cache[d["name"]] = (I, I.mean(-1))
    rgb, grey = cache[d["name"]]
    Sig = structure_cov(grey, *d["uv"], d["s"])
    (A1, A2), ang, Sg, sc = fitted(Sig, d["eps"], a.gamma)
    (B1, B2), angB, S1, sc1 = fitted(Sig, d["eps"], 1.0)
    e = d["gtuv"] - d["uv"]
    in_box = np.max(np.abs(e)) <= d["eps"]
    in_ell = e @ np.linalg.solve(Sg, e) <= sc ** 2
    in_full = e @ np.linalg.solve(S1, e) <= sc1 ** 2
    w = d["eps"] + 3
    u, v = d["uv"]
    c0, c1 = int(math.floor(u - w)), int(math.ceil(u + w))
    r0, r1 = int(math.floor(v - w)), int(math.ceil(v + w))
    H, W = grey.shape
    crop = np.ones((r1 - r0, c1 - c0, 3))
    rr0, cc0 = max(r0, 0), max(c0, 0)
    crop[rr0 - r0:min(r1, H) - r0, cc0 - c0:min(c1, W) - c0] = rgb[rr0:min(r1, H), cc0:min(c1, W)]
    ax.imshow(crop, extent=[c0, c1, r1, r0], interpolation="nearest")     # pixel (r, c) covers u in [c, c+1]
    ax.add_patch(Rectangle((u - d["eps"], v - d["eps"]), 2 * d["eps"], 2 * d["eps"], fill=False, ec="black", lw=1.5))
    ax.add_patch(Ellipse((u, v), 2 * A1, 2 * A2, angle=ang, fill=False, ec="tab:blue", lw=1.8))
    ax.add_patch(Ellipse((u, v), 2 * B1, 2 * B2, angle=angB, fill=False, ec="tab:blue", lw=1, ls=":"))
    ax.plot(u, v, "+", color="red", ms=12, mew=2)
    ax.plot(*d["gtuv"], "x", color="lime", ms=10, mew=2.5)
    ax.plot(*d["gtuv"], "x", color="black", ms=10, mew=0.8)
    ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  track %d (%d views)\nr %.2f px, eps %.1f px, |e| %.2f px\nGT in: box %s, ellipse %s, full %s"
                 % (d["name"], d["k"], d["views"], d["r"], d["eps"], np.hypot(*e),
                    "yes" if in_box else "NO", "yes" if in_ell else "NO", "yes" if in_full else "NO"), fontsize=8)
for ax in np.ravel(axes)[len(picked):]:
    ax.axis("off")
fig.suptitle(("ONLY GT OUTSIDE THE ELLIPSE -- " if a.misses else "") + "%s: + detection, x GT pixel, square = LP window, solid ellipse = SIFT shape (gamma %.2f) fitted to the window, "
             "dotted = full shape (gamma 1)" % (os.path.basename(S), a.gamma), fontsize=10)
fig.tight_layout()
out = a.out or os.path.join(S, "figures", "windows_gamma%.2f_seed%d%s.png" % (a.gamma, a.seed, "_misses" if a.misses else ""))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print("->", out)
