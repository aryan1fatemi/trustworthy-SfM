"""Side-by-side correspondence view for one image pair of a reconstruction.

    python viz_matches.py --scene airplane [--pair NAME_A NAME_B] [--min-angle 15]

Lines join the two detections of every landmark both images observe (the
verified matches COLMAP kept in the model).  Red dots are all keypoints.

Second panel (scenes with GT depth): each line is coloured by whether the match
is geometrically correct -- detection A is lifted onto the true surface with
A's GT depth and projected into B with B's GT camera; the gap to detection B in
pixels decides the colour.
"""
import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model

FLIP = np.diag([1.0, -1.0, -1.0])


def load_gt(scene):
    cam = {}
    for sp, suf in (("train", ""), ("val", "_val")):
        p = os.path.join(scene, "transforms_%s.json" % sp)
        if not os.path.exists(p):
            return None
        J = json.load(open(p))
        for fr in J["frames"]:
            if "depth_path" not in fr:
                return None
            M = np.array(fr["transform_matrix"])
            cam[os.path.basename(fr["file_path"]) + suf + ".png"] = dict(
                C2W=M[:3, :3] @ FLIP, c=M[:3, 3], f=J["fl_x"], cx=J["cx"], cy=J["cy"],
                depth=os.path.join(scene, fr["depth_path"][2:]))
    return cam


def lift(g, D, u, v):
    x, y = u - 0.5, v - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= D.shape[1] or y0 + 1 >= D.shape[0]:
        return None
    q = D[y0:y0 + 2, x0:x0 + 2]
    if (q <= 0).any() or (q.max() - q.min()) / q.mean() > 0.02:
        return None
    ax, ay = x - x0, y - y0
    d = (1 - ay) * ((1 - ax) * q[0, 0] + ax * q[0, 1]) + ay * ((1 - ax) * q[1, 0] + ax * q[1, 1])
    return g["c"] + g["C2W"] @ (d * np.array([(u - g["cx"]) / g["f"], (v - g["cy"]) / g["f"], 1.0]))


def proj(g, X):
    Xc = g["C2W"].T @ (X - g["c"])
    return np.array([g["f"] * Xc[0] / Xc[2] + g["cx"], g["f"] * Xc[1] / Xc[2] + g["cy"]]), Xc[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--min-angle", type=float, default=15.0,
                    help="auto-pick: minimum angle between viewing directions (deg)")
    ap.add_argument("--good-px", type=float, default=2.0)
    a = ap.parse_args()
    scene = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    model = os.path.join(scene, "COLMAP results")
    out_dir = os.path.join(model, "figures")
    os.makedirs(out_dir, exist_ok=True)
    cams, imgs, pts = read_model(model)
    by_name = {im.name: im for im in imgs.values()}

    def shared(A, B):
        ma = {int(p): j for j, p in enumerate(A.p3d_ids) if p >= 0}
        return [(ma[int(p)], j) for j, p in enumerate(B.p3d_ids) if p >= 0 and int(p) in ma]

    if a.pair:
        A, B = by_name[a.pair[0]], by_name[a.pair[1]]
    else:                                   # most shared landmarks at a real baseline
        ims = list(imgs.values())
        best = (0, None, None)
        for i, A in enumerate(ims):
            for B in ims[i + 1:]:
                ang = np.degrees(np.arccos(np.clip(A.R[2] @ B.R[2], -1, 1)))
                if ang < a.min_angle:
                    continue
                n = len(set(A.p3d_ids[A.p3d_ids >= 0]) & set(B.p3d_ids[B.p3d_ids >= 0]))
                if n > best[0]:
                    best = (n, A, B)
        _, A, B = best
    pairs = shared(A, B)
    ang = np.degrees(np.arccos(np.clip(A.R[2] @ B.R[2], -1, 1)))

    ia = Image.open(os.path.join(scene, "images", A.name)).convert("RGB")
    ib = Image.open(os.path.join(scene, "images", B.name)).convert("RGB")
    W, H = ia.size

    gt = load_gt(scene)
    errs = None
    if gt:
        ga, gb = gt[A.name], gt[B.name]
        Da, Db = np.load(ga["depth"])["depth"], np.load(gb["depth"])["depth"]
        errs = []
        for ja, jb in pairs:
            X = lift(ga, Da, *A.xys[ja])
            if X is None:
                errs.append(np.nan)
                continue
            uv, z = proj(gb, X)
            vis = Db[int(np.clip(uv[1] - 0.5, 0, H - 1)), int(np.clip(uv[0] - 0.5, 0, W - 1))]
            occluded = vis > 0 and z > vis * 1.02
            errs.append(np.nan if occluded else float(np.abs(uv - B.xys[jb]).max()))
        errs = np.array(errs)

    def panel(colour_fn, title):
        canvas = Image.new("RGB", (2 * W, H + 34), "white")
        canvas.paste(ia, (0, 34))
        canvas.paste(ib, (W, 34))
        d = ImageDraw.Draw(canvas)
        for im, off in ((A, 0), (B, W)):
            for u, v in im.xys:
                d.ellipse([off + u - 1.6, 34 + v - 1.6, off + u + 1.6, 34 + v + 1.6],
                          outline=(220, 40, 40))
        for n, (ja, jb) in enumerate(pairs):
            (ua, va), (ub, vb) = A.xys[ja], B.xys[jb]
            d.line([ua, 34 + va, W + ub, 34 + vb], fill=colour_fn(n), width=1)
        d.text((8, 10), title, fill=(0, 0, 0))
        return canvas

    tag = "%s__%s" % (A.name[:-4], B.name[:-4])
    title = ("%s  <->  %s   |  %d matched landmarks  |  %.0f deg apart  |  keypoints %d / %d"
             % (A.name, B.name, len(pairs), ang, len(A.xys), len(B.xys)))
    panel(lambda n: (40, 200, 40), title).save(os.path.join(out_dir, "matches_%s.png" % tag))
    print("pair", A.name, B.name, "shared", len(pairs), "angle %.1f" % ang)

    if errs is not None:
        good = errs <= a.good_px
        bad = errs > a.good_px
        def col(n):
            if np.isnan(errs[n]):
                return (150, 150, 150)
            return (40, 200, 40) if good[n] else (230, 30, 30)
        t2 = ("GT check: green = correct (<= %.0f px, %d)   red = wrong (%d)   grey = no GT depth "
              "or occluded (%d)   |  median error %.2f px"
              % (a.good_px, good.sum(), bad.sum(), np.isnan(errs).sum(), np.nanmedian(errs)))
        panel(col, t2).save(os.path.join(out_dir, "matches_gtcheck_%s.png" % tag))
        print(t2)
    print("->", out_dir)


if __name__ == "__main__":
    main()
