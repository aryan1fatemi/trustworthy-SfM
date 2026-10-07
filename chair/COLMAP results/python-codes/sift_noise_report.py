"""How SIFT detection and matching respond to pixel noise (no reconstruction).

    python sift_noise_report.py

Reads <dataset>/database_features.db for the clean airplane and its two noisy copies
(run_features_only.sh), and writes to airplane/figures/sift_noise/:
    NN_<image>.png        the same 10 images, clean | low | high side by side, keypoints drawn
                          green if they have >= 1 geometrically verified match, red if not
    per_image.csv         keypoints (object / background), matched fraction, per dataset
    summary.json, report.md, summary.png
The object mask is the GT depth map (depth > 0); no GT poses are used.
Repeatability: fraction of the clean image's object keypoints with a keypoint of the noisy
image within 1.5 px, and the median shift of those.
"""
import os
import sys
import csv
import json
import sqlite3
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cp_conformal as CC

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCENE = os.path.join(ROOT, "airplane")
DATASETS = [("clean", "airplane"), ("low", "airplane_noise_low"), ("high", "airplane_noise_high")]
COL = {"clean": "#2a78d6", "low": "#eb6834", "high": "#1baf7a"}
OUT = os.path.join(SCENE, "figures", "sift_noise")
PAIR = 2147483647
REPEAT_PX = 1.5


def load_db(path):
    db = sqlite3.connect(path)
    name = dict(db.execute("select image_id, name from images"))
    kp = {}
    for iid, r, c, blob in db.execute("select image_id, rows, cols, data from keypoints"):
        a = np.frombuffer(blob, np.float32).reshape(r, c)
        kp[name[iid]] = a
    matched = {n: np.zeros(len(a), bool) for n, a in kp.items()}
    raw_pairs = db.execute("select count(*) from matches where rows > 0").fetchone()[0]
    inl = []
    for pid, r, blob in db.execute("select pair_id, rows, data from two_view_geometries where rows > 0"):
        i2 = pid % PAIR
        i1 = (pid - i2) // PAIR
        m = np.frombuffer(blob, np.uint32).reshape(r, 2)
        matched[name[i1]][m[:, 0]] = True
        matched[name[i2]][m[:, 1]] = True
        inl.append(r)
    return kp, matched, raw_pairs, np.array(inl)


def scale(a):
    """SIFT scale in px from the affine shape columns (a11, a12, a21, a22)."""
    return np.sqrt(np.abs(a[:, 2] * a[:, 5] - a[:, 3] * a[:, 4])) if a.shape[1] >= 6 else a[:, 2]


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = CC.load_gt(SCENE)
    D = {}
    for tag, folder in DATASETS:
        D[tag] = load_db(os.path.join(ROOT, folder, "database_features.db"))
    names = sorted(D["clean"][0])
    masks = {n: np.load(gt[n]["depth"])["depth"] > 0 for n in names}

    def on_object(n, xy):
        m = masks[n]
        u = np.clip(xy[:, 0].astype(int), 0, m.shape[1] - 1)
        v = np.clip(xy[:, 1].astype(int), 0, m.shape[0] - 1)
        return m[v, u]

    # ---- per-image table --------------------------------------------------
    rows, summ = [], {}
    for tag, _ in DATASETS:
        kp, matched, raw_pairs, inl = D[tag]
        per = []
        for n in names:
            a = kp[n]
            obj = on_object(n, a[:, :2])
            r = dict(dataset=tag, image=n, keypoints=len(a), on_object=int(obj.sum()),
                     on_background=int((~obj).sum()), matched=int(matched[n].sum()),
                     matched_object=int((matched[n] & obj).sum()),
                     matched_background=int((matched[n] & ~obj).sum()))
            if tag != "clean":
                c = D["clean"][0][n]
                c = c[on_object(n, c[:, :2])][:, :2]
                d, _ = cKDTree(a[:, :2]).query(c)
                r["repeatability_object"] = float((d <= REPEAT_PX).mean())
                r["median_shift_px"] = float(np.median(d[d <= REPEAT_PX])) if (d <= REPEAT_PX).any() else float("nan")
            per.append(r)
        rows += per
        allkp = np.concatenate([kp[n] for n in names])
        obj_all = np.concatenate([on_object(n, kp[n][:, :2]) for n in names])
        s = scale(allkp)
        med = lambda k: float(np.median([p[k] for p in per]))
        summ[tag] = {
            "keypoints_per_image_median": med("keypoints"),
            "keypoints_per_image_min_max": [min(p["keypoints"] for p in per), max(p["keypoints"] for p in per)],
            "object_keypoints_per_image_median": med("on_object"),
            "background_keypoints_per_image_median": med("on_background"),
            "background_fraction": float((~obj_all).mean()),
            "matched_fraction_object": float(sum(p["matched_object"] for p in per) / max(1, sum(p["on_object"] for p in per))),
            "matched_fraction_background": float(sum(p["matched_background"] for p in per) / max(1, sum(p["on_background"] for p in per))),
            "image_pairs_with_raw_matches": int(raw_pairs),
            "verified_pairs": int(len(inl)),
            "verified_inliers_per_pair_median": float(np.median(inl)) if len(inl) else 0.0,
            "verified_inliers_total": int(inl.sum()),
            "sift_scale_px_median_object": float(np.median(s[obj_all])),
            "sift_scale_px_median_background": float(np.median(s[~obj_all])) if (~obj_all).any() else float("nan"),
            "_scales_object": s[obj_all], "_scales_bg": s[~obj_all], "_per": per,
        }
        if tag != "clean":
            summ[tag]["repeatability_object_median"] = med("repeatability_object")
            summ[tag]["median_shift_px"] = float(np.nanmedian([p["median_shift_px"] for p in per]))

    keys = list(rows[0].keys()) + ["repeatability_object", "median_shift_px"]
    with open(os.path.join(OUT, "per_image.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    noise = {t: (json.load(open(os.path.join(ROOT, f, "noise.json"))) if t != "clean" else None)
             for t, f in DATASETS}
    json.dump({t: {k: v for k, v in s.items() if not k.startswith("_")} for t, s in summ.items()}
              | {"noise": noise}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    # ---- the same 10 images, side by side -----------------------------------
    sel_csv = os.path.join(SCENE, "figures", "bad_detection_images", "selected_images.csv")
    pick = [r["name"] for r in csv.DictReader(open(sel_csv))][:10]
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    UP = 2
    for k, n in enumerate(pick, 1):
        c = D["clean"][0][n][:, :2]
        m = masks[n]
        vv, uu = np.nonzero(m)
        u0, v0 = max(uu.min() - 30, 0), max(vv.min() - 30, 0)
        u1, v1 = min(uu.max() + 30, m.shape[1]), min(vv.max() + 30, m.shape[0])
        panels = []
        for tag, folder in DATASETS:
            kp, matched = D[tag][0][n], D[tag][1][n]
            im = Image.open(os.path.join(ROOT, folder, "images", n)).convert("RGB")
            im = im.crop((u0, v0, u1, v1)).resize(((u1 - u0) * UP, (v1 - v0) * UP), Image.NEAREST)
            head = 34
            P = Image.new("RGB", (im.width, im.height + head), (255, 255, 255))
            P.paste(im, (0, head))
            dr = ImageDraw.Draw(P)
            for (x, y), ok in sorted(zip(kp[:, :2], matched[: len(kp)]), key=lambda t: t[1]):
                if not (u0 <= x < u1 and v0 <= y < v1):
                    continue
                X, Y = (x - u0) * UP, (y - v0) * UP + head
                col = (30, 160, 60) if ok else (220, 40, 40)
                dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=col, outline=(0, 0, 0))
            nk, nm = len(kp), int(matched.sum())
            obj = on_object(n, kp[:, :2])
            dr.text((8, 6), "%s: %d keypoints (%d object, %d background), %d matched"
                    % (tag, nk, obj.sum(), (~obj).sum(), nm), fill=(0, 0, 0), font=font)
            panels.append(P)
        Wt = sum(p.width for p in panels) + 20 * (len(panels) - 1)
        S = Image.new("RGB", (Wt, panels[0].height), (255, 255, 255))
        x = 0
        for p in panels:
            S.paste(p, (x, 0))
            x += p.width + 20
        S.save(os.path.join(OUT, "%02d_%s" % (k, n)))

    # ---- summary chart --------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tags = [t for t, _ in DATASETS]
    for j, (key, lab) in enumerate((("on_object", "object"), ("on_background", "background"))):
        for i, t in enumerate(tags):
            v = np.median([p[key] for p in summ[t]["_per"]])
            ax[0].bar(j * 4 + i, v, color=COL[t], width=0.9, label=t if j == 0 else None)
            ax[0].text(j * 4 + i, v, "%d" % v, ha="center", va="bottom", fontsize=9, color="#52514e")
    ax[0].set_xticks([1, 5])
    ax[0].set_xticklabels(["on the object", "on the background"])
    ax[0].set_ylabel("keypoints per image (median)")
    ax[0].set_title("Where SIFT fires")
    ax[0].legend(frameon=False)
    bins = np.geomspace(1, 64, 40)
    for t in tags:
        ax[1].hist(summ[t]["_scales_object"], bins=bins, histtype="step", lw=2, color=COL[t], label=t)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("SIFT scale (px), object keypoints")
    ax[1].set_ylabel("keypoints (all images)")
    ax[1].set_title("Scale of object keypoints")
    ax[1].legend(frameon=False)
    for i, t in enumerate(tags):
        v = 100 * summ[t]["matched_fraction_object"]
        ax[2].bar(i, v, color=COL[t], width=0.8)
        ax[2].text(i, v, "%.0f%%" % v, ha="center", va="bottom", fontsize=9, color="#52514e")
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(tags)
    ax[2].set_ylabel("% of object keypoints with a verified match")
    ax[2].set_title("Matchability")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(axis="y", color="#e5e4df", lw=0.8)
        a_.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "summary.png"), dpi=130)

    # ---- report ---------------------------------------------------------------
    L = ["# SIFT under pixel noise: airplane, 200 images", "",
         "Noise: I' = clip(round(I + U(-a, a))), per pixel and channel, 0-255 scale. "
         "low: a = %g (std %.2f); high: a = %g (std %.2f). "
         "COLMAP 3.9.1 SIFT (CPU, max 8192 features) + exhaustive matching with two-view "
         "verification; no reconstruction. Object/background from the GT depth mask."
         % (noise["low"]["amp"], noise["low"]["std"], noise["high"]["amp"], noise["high"]["std"]), "",
         "| | clean | low | high |", "|---|---|---|---|"]
    fmt = [("keypoints per image, median", "keypoints_per_image_median", "%d"),
           ("  on the object", "object_keypoints_per_image_median", "%d"),
           ("  on the background", "background_keypoints_per_image_median", "%d"),
           ("background share of all keypoints", "background_fraction", "%.1f%%"),
           ("object keypoints with a verified match", "matched_fraction_object", "%.1f%%"),
           ("background keypoints with a verified match", "matched_fraction_background", "%.1f%%"),
           ("image pairs with raw matches", "image_pairs_with_raw_matches", "%d"),
           ("verified image pairs", "verified_pairs", "%d"),
           ("verified inliers per pair, median", "verified_inliers_per_pair_median", "%d"),
           ("verified inliers, total", "verified_inliers_total", "%d"),
           ("SIFT scale, object keypoints, median (px)", "sift_scale_px_median_object", "%.2f"),
           ("repeatability of clean object keypoints (<= 1.5 px)", "repeatability_object_median", "%.1f%%"),
           ("shift of repeated keypoints, median (px)", "median_shift_px", "%.2f")]
    for lab, k, f in fmt:
        cells = []
        for t in tags:
            v = summ[t].get(k)
            if v is None:
                cells.append("-")
            else:
                cells.append(f % (100 * v) if f.endswith("%%") else f % v)
        L.append("| %s | %s |" % (lab, " | ".join(cells)))
    L += ["", "Figures: `NN_<image>.png` (clean | low | high; green = has a verified match, "
          "red = none), `summary.png`. Per-image numbers: `per_image.csv`."]
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
