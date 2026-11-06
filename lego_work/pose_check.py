import sys, os, json, csv
import numpy as np
sys.dont_write_bytecode = True      # don't create __pycache__ inside ../chair
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chair"))
from align_translations import load_colmap_centers, umeyama

scene = sys.argv[1]                                  # lego_canon  or  ../chair
tag = os.path.basename(os.path.abspath(scene))
meta = json.load(open(os.path.join(scene, "transforms.json")))
colmap = load_colmap_centers(os.path.join(scene, "COLMAP results fixedK", "images.txt"))

gt, origin = {}, {}
for fr in meta["frames"]:
    name = os.path.basename(fr["file_path"]) + ".png"      # ./images/37 -> 37.png
    gt[name] = np.array(fr["transform_matrix"], float)[:3, 3]
    origin[name] = fr.get("same_view_as", "")

names = sorted(set(colmap) & set(gt))
print(f"{tag}: GT {len(gt)} | COLMAP {len(colmap)} | matched {len(names)}")
src = np.array([colmap[n] for n in names])
dst = np.array([gt[n] for n in names])
s, R, t = umeyama(src, dst, with_scale=True)
res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)

with open(f"translation_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "same_view_as", "residual"])
    for n, r in zip(names, res):
        w.writerow([n, origin[n], round(float(r), 6)])

extent = np.linalg.norm(dst.max(0) - dst.min(0))
print(f"scale s = {s:.6f}")
print(f"mean {res.mean():.6f} | median {np.median(res):.6f} | RMSE {np.sqrt((res**2).mean()):.6f}")
print(f"max {res.max():.6f} ({names[int(res.argmax())]}) | mean = {100*res.mean()/extent:.3f}% of GT extent {extent:.3f}")
