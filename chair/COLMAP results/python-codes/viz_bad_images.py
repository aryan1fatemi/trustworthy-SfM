"""Draw detected keypoints on the images with the worst detection error.

    python viz_bad_images.py --scene airplane --n 10

Detection error per observation is s_det = || y_ik - pi(theta_GT_i, X_ref) ||_inf, with X_ref
the median of the OTHER images' GT-lifted detections (cp_conformal.py, leave-one-image-out).
Images are ranked by the 90th percentile of s_det over their valid observations.

Each output image (2x upsampled) shows
    grey dot      keypoint detected by SIFT but not part of any 3D point
    blue ring     observation of a 3D point with no GT score (background, depth edge, ...)
    coloured dot  observation with a GT score: green < 1 px, yellow 1-4, orange 4-10, red > 10
    line          from the detection to where the GT camera projects the reference point
                  (drawn for s_det > 1 px, length exaggerated x MAG)
"""
import os
import sys
import csv
import argparse
import sqlite3
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model
from cp_step0_align import umeyama
import cp_conformal as CC

UP = 2          # upsampling of the drawing
MAG = 3         # exaggeration of the error segments
BINS = [(1, (40, 170, 60)), (4, (230, 200, 20)), (10, (245, 130, 20)), (np.inf, (220, 30, 30))]


def colour(s):
    for hi, c in BINS:
        if s < hi:
            return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-obs", type=int, default=50)
    a = ap.parse_args()
    SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    MODEL = os.path.join(SCENE, "COLMAP results")
    OUT = os.path.join(SCENE, "figures", "bad_detection_images")
    os.makedirs(OUT, exist_ok=True)

    cams, imgs, pts = read_model(MODEL)
    gt = CC.load_gt(SCENE)
    ids = [i for i in imgs if imgs[i].name in gt]
    s_, Ra, ta = umeyama(np.array([gt[imgs[i].name]["c"] for i in ids]),
                         np.array([imgs[i].center for i in ids]))
    rows = CC.observations(cams, imgs, pts, gt, s_, Ra, ta)
    CC.references_and_scores(rows, cams, imgs, pts, gt)

    # GT pixel of the leave-one-image-out reference, for the error segments
    by = {}
    for n, r in enumerate(rows):
        by.setdefault(r["pid"], []).append(n)
    for idx in by.values():
        V = [n for n in idx if rows[n]["valid"]]
        for n in idx:
            r = rows[n]
            others = [m for m in V if rows[m]["iid"] != r["iid"]]
            r["gt_uv"] = None
            if r["valid"] and others:
                ref = np.median([rows[m]["Xgt"] for m in others], 0)
                g = gt[imgs[r["iid"]].name]
                pc = g["K"] @ (g["R"] @ (ref - g["c"]))
                r["gt_uv"] = pc[:2] / pc[2]

    per = {}
    for r in rows:
        per.setdefault(r["iid"], []).append(r)
    stats = []
    for iid, rs in per.items():
        s = np.array([r["s_det"] for r in rs if np.isfinite(r["s_det"])])
        if len(s) < a.min_obs:
            continue
        stats.append(dict(image_id=iid, name=imgs[iid].name, observations=len(rs), scored=len(s),
                          median_px=float(np.median(s)), p90_px=float(np.quantile(s, 0.9)),
                          frac_over_4px=float((s > 4).mean()), frac_over_10px=float((s > 10).mean()),
                          max_px=float(s.max())))
    stats.sort(key=lambda d: -d["p90_px"])
    pick = stats[: a.n]
    all_p90 = np.median([d["p90_px"] for d in stats])

    db = sqlite3.connect(os.path.join(SCENE, "database.db"))
    name2dbid = dict((n, i) for i, n in db.execute("select image_id, name from images"))
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = small = ImageFont.load_default()

    for rank, d in enumerate(pick, 1):
        iid, name = d["image_id"], d["name"]
        img = Image.open(os.path.join(SCENE, "images", name)).convert("RGB")
        W0, H0 = img.size
        img = img.resize((W0 * UP, H0 * UP), Image.LANCZOS)
        head = 70
        canvas = Image.new("RGB", (img.width, img.height + head), (255, 255, 255))
        canvas.paste(img, (0, head))
        dr = ImageDraw.Draw(canvas)
        P = lambda u, v: (u * UP, v * UP + head)

        # all SIFT keypoints of this image
        r_, c_, blob = db.execute("select rows, cols, data from keypoints where image_id=?",
                                  (name2dbid[name],)).fetchone()
        kp = np.frombuffer(blob, np.float32).reshape(r_, c_)[:, :2]
        used = set(int(j) for j in np.flatnonzero(imgs[iid].p3d_ids >= 0))
        for j, (u, v) in enumerate(kp):
            if j not in used:
                x, y = P(u, v)
                dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(150, 150, 150))
        # observations, worst drawn last
        rs = sorted(per[iid], key=lambda r: r["s_det"] if np.isfinite(r["s_det"]) else -1)
        for r in rs:
            x, y = P(r["u"], r["v"])
            if not np.isfinite(r["s_det"]):
                dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(40, 90, 230), width=2)
                continue
            c = colour(r["s_det"])
            if r["s_det"] > 1 and r["gt_uv"] is not None:
                gx, gy = r["u"] + MAG * (r["gt_uv"][0] - r["u"]), r["v"] + MAG * (r["gt_uv"][1] - r["v"])
                dr.line([P(r["u"], r["v"]), P(gx, gy)], fill=c, width=2)
            rad = 4 if r["s_det"] < 4 else 6
            dr.ellipse([x - rad, y - rad, x + rad, y + rad], fill=c, outline=(0, 0, 0))

        dr.text((10, 6), "#%d  %s   p90 s_det %.2f px (median image %.2f)   median %.2f   "
                "> 4 px %.0f%%   > 10 px %.0f%%   max %.0f px"
                % (rank, name, d["p90_px"], all_p90, d["median_px"], 100 * d["frac_over_4px"],
                   100 * d["frac_over_10px"], d["max_px"]), fill=(0, 0, 0), font=font)
        x0 = 10
        for lab, col in (("< 1 px", BINS[0][1]), ("1-4", BINS[1][1]), ("4-10", BINS[2][1]),
                         ("> 10", BINS[3][1])):
            dr.ellipse([x0, 42, x0 + 14, 56], fill=col, outline=(0, 0, 0))
            dr.text((x0 + 20, 38), lab, fill=(0, 0, 0), font=small)
            x0 += 100
        dr.ellipse([x0, 42, x0 + 14, 56], outline=(40, 90, 230), width=2)
        dr.text((x0 + 20, 38), "no GT score", fill=(0, 0, 0), font=small)
        x0 += 150
        dr.ellipse([x0 + 4, 46, x0 + 10, 52], fill=(150, 150, 150))
        dr.text((x0 + 20, 38), "keypoint, no 3D point", fill=(0, 0, 0), font=small)
        x0 += 250
        dr.text((x0, 38), "line: to GT projection, x%d" % MAG, fill=(0, 0, 0), font=small)
        # crop to the keypoints (+ margin), keeping the header
        m = 40
        u0, v0 = np.maximum(kp.min(0) - m, 0)
        u1, v1 = np.minimum(kp.max(0) + m, [W0, H0])
        body = canvas.crop((int(u0 * UP), head + int(v0 * UP), int(u1 * UP), head + int(v1 * UP)))
        out = Image.new("RGB", (max(body.width, 1300), body.height + head), (255, 255, 255))
        out.paste(canvas.crop((0, 0, out.width, head)), (0, 0))
        out.paste(body, (0, head))
        fn = "%02d_%s" % (rank, name)
        out.save(os.path.join(OUT, fn))
        d["file"] = fn
        print("%2d  %-14s p90 %.2f  >4px %.0f%%  max %.0f" % (rank, name, d["p90_px"],
                                                              100 * d["frac_over_4px"], d["max_px"]))

    with open(os.path.join(OUT, "selected_images.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(pick[0].keys()))
        w.writeheader()
        w.writerows(pick)
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
