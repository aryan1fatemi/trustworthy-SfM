import json, os, shutil
import numpy as np
from PIL import Image

SRC, OUT = "lego_gt", "lego_canon"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/depth", exist_ok=True)

meta, frames, n = None, [], 0
for split in ("train", "val"):
    m = json.load(open(f"{SRC}/transforms_{split}.json"))
    if meta is None:
        meta = {k: v for k, v in m.items() if k != "frames"}
    for i, fr in enumerate(m["frames"]):
        n += 1
        a = np.asarray(Image.open(f"{SRC}/{split}/r_{i}.png").convert("RGBA"), np.float32) / 255
        rgb = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4])      # composite on white
        Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(f"{OUT}/images/{n}.png")
        shutil.copy(f"{SRC}/{split}/r_{i}_depth.npz", f"{OUT}/depth/{n}.npz")
        frames.append({"file_path": f"./images/{n}", "depth_path": f"./depth/{n}.npz",
                       "same_view_as": f"{split} r_{i}",
                       "transform_matrix": fr["transform_matrix"]})
meta["frames"] = frames
meta["background"] = "white, composited (rgb' = a*rgb + 1 - a)"
meta["views"] = "1..100 from seed 0, 101..200 from seed 1000"
json.dump(meta, open(f"{OUT}/transforms.json", "w"), indent=2)
print(n, "views written to", OUT)
