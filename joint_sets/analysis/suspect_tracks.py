"""Tracks whose detections may not be the same physical point (likely wrong matches).

    python suspect_tracks.py --scene ../../chair [--thresh 9] [--per-page 12]

Every detection is lifted to the surface point under it with ITS OWN view's exact depth and the true camera.
deviation = distance from its lifted point to the median of the track's OTHER lifted points, in pixels at its depth.
Same 600-track sample (seed 0, tracks with >= 5 views) as the match check; tracks with a detection > thresh px are drawn.

Per track, two panels:
  left   the worst detection's view:   + the detection,  x the consensus point (median of the other lifts) projected
  right  a reference view (the detection that agrees best with the consensus):
         + the reference detection,  o where the WORST detection's surface point lands in this view
If the match is wrong, the o sits on a different feature than the +."""
import os, sys, math, argparse
from types import SimpleNamespace
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from geom import Scene
from colmap_io import read_model

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--model", default="COLMAP results fixedK")
ap.add_argument("--thresh", type=float, default=9.0)
ap.add_argument("--sample", type=int, default=600)
ap.add_argument("--per-page", type=int, default=12)
a = ap.parse_args()
S = os.path.abspath(a.scene)
cams, imgs, pts = read_model(os.path.join(S, a.model))
gt = Scene.load_gt(SimpleNamespace(scene=S))
_D, _I = {}, {}


def depth(name):
    if name not in _D:
        _D[name] = np.load(gt[name]["depth"])["depth"]
    return _D[name]


def image(name):
    if name not in _I:
        _I[name] = np.asarray(Image.open(os.path.join(S, "images", name)).convert("RGB"), float) / 255.0
    return _I[name]


def proj(g, X):
    q = g["K"] @ (g["R"] @ (X - g["c"]))
    return q[:2] / q[2]


def lift(g, uv, Z):
    """bilinear z-depth at (u, v) (pixel centres at +0.5); None if any of the 4 pixels is background or depth jumps."""
    x, y = uv[0] - 0.5, uv[1] - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    ax_, ay = x - x0, y - y0
    q = Z[y0:y0 + 2, x0:x0 + 2]
    if q.shape != (2, 2) or (q <= 0).any() or q.max() - q.min() > 0.02 * q.mean():
        return None
    z = (1 - ay) * ((1 - ax_) * q[0, 0] + ax_ * q[0, 1]) + ay * ((1 - ax_) * q[1, 0] + ax_ * q[1, 1])
    d = np.linalg.solve(g["K"], [uv[0], uv[1], 1.0])
    return g["R"].T @ (z * d) + g["c"]


rng = np.random.default_rng(0)
ks = [k for k, P in pts.items() if len(P.image_ids) >= 5]
sus = []
for k in rng.choice(ks, a.sample, replace=False):
    P = pts[k]
    L = []
    for i, j in zip(P.image_ids, P.point2D_idxs):
        n = imgs[i].name
        x = lift(gt[n], imgs[i].xys[j], depth(n))
        if x is not None:
            L.append((n, imgs[i].xys[j], x))
    if len(L) < 4:
        continue
    P3 = np.array([x for *_, x in L])
    dev = []
    for m, (n, uv, x) in enumerate(L):
        g = gt[n]
        med = np.median(np.delete(P3, m, 0), 0)
        dev.append(np.linalg.norm(x - med) * g["K"][0, 0] / (g["R"] @ (x - g["c"]))[2])
    dev = np.array(dev)
    if dev.max() > a.thresh:
        w, r = int(np.argmax(dev)), int(np.argmin(dev))
        sus.append(dict(k=int(k), L=L, dev=dev, w=w, r=r, n_bad=int((dev > a.thresh).sum()),
                        cons=np.median(np.delete(P3, w, 0), 0)))
sus.sort(key=lambda d: -d["dev"].max())
print("%d suspect tracks (detection > %.0f px) in the %d-track sample" % (len(sus), a.thresh, a.sample))


def panel(ax, name, pts_, title):
    """crop around the given points (list of (uv, marker, colour, label)), at least 40 px wide."""
    U = np.array([p[0] for p in pts_])
    c = U.mean(0)
    h = max(20.0, 0.5 * np.ptp(U, 0).max() + 12)
    u0, u1, v0, v1 = int(c[0] - h), int(c[0] + h), int(c[1] - h), int(c[1] + h)
    I = image(name)
    H, W = I.shape[:2]
    crop = np.ones((v1 - v0, u1 - u0, 3))
    a0, b0 = max(v0, 0), max(u0, 0)
    crop[a0 - v0:min(v1, H) - v0, b0 - u0:min(u1, W) - u0] = I[a0:min(v1, H), b0:min(u1, W)]
    ax.imshow(crop, extent=[u0, u1, v1, v0], interpolation="nearest")
    for uv, mk, col, lab in pts_:
        ax.plot(*uv, mk, color=col, ms=11, mew=2.2, mfc="none", label=lab)
    ax.set_xlim(u0, u1); ax.set_ylim(v1, v0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7.5)


out_dir = os.path.join(S, "figures")
os.makedirs(out_dir, exist_ok=True)
pages = [sus[i:i + a.per_page] for i in range(0, len(sus), a.per_page)]
for p, page in enumerate(pages, 1):
    rows = math.ceil(len(page) / 2)
    fig, axes = plt.subplots(rows, 4, figsize=(13.5, 3.5 * rows), squeeze=False)
    for t, d in enumerate(page):
        aw, ar = axes[t // 2, 2 * (t % 2)], axes[t // 2, 2 * (t % 2) + 1]
        nw, uvw, xw = d["L"][d["w"]]
        nr, uvr, xr = d["L"][d["r"]]
        cpix = proj(gt[nw], d["cons"])
        land = proj(gt[nr], xw)
        panel(aw, nw, [(uvw, "+", "red", "detection"), (cpix, "x", "lime", "consensus")],
              "track %d (%d lifted views, %d > %.0f px)\nWORST %s: dev %.1f px, |det - consensus| %.1f px"
              % (d["k"], len(d["L"]), d["n_bad"], a.thresh, nw, d["dev"][d["w"]], np.linalg.norm(uvw - cpix)))
        panel(ar, nr, [(uvr, "+", "red", "reference detection"), (land, "o", "magenta", "worst's surface point")],
              "reference %s (dev %.1f px)\nworst's surface point lands %.1f px from it"
              % (nr, d["dev"][d["r"]], np.linalg.norm(uvr - land)))
    for ax in axes.ravel()[2 * len(page):]:
        ax.axis("off")
    fig.suptitle("Suspect tracks, page %d/%d. Left: worst view (+ detection, x consensus of the other views). "
                 "Right: reference view (+ its detection, o where the worst detection's surface point lands)"
                 % (p, len(pages)), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    f = os.path.join(out_dir, "suspect_tracks_p%d.png" % p)
    fig.savefig(f, dpi=120)
    plt.close(fig)
    print("->", f)
