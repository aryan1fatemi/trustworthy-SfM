"""
Visualize NeRF-synthetic depth maps as heatmaps and dump per-pixel depth.

For each selected test image r_N:
  - reads test/r_N_depth_0000.png  (8-bit grayscale packed in RGBA; R=G=B,
    alpha = object mask; background = 0)
  - writes a heatmap PNG: object colored by depth (turbo colormap:
    blue->green->yellow->red), background white
  - writes a .txt with one line per pixel:  u  v  depth
        u = column (0..W-1), v = row (0..H-1), depth = raw 8-bit value
        (0 = background/no object; larger = brighter in the source map)

NOTE: these raw values are the depth encoded by Blender at 8-bit precision,
NOT metric distance. Recovering metric scale is a separate calibration step.

Outputs go to:  depth/output/
"""

import os
import numpy as np
from PIL import Image
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(HERE, "test")
OUT_DIR = os.path.join(HERE, "depth", "output")

IMAGES = ["r_0", "r_1"]          # which test images to process
COLORMAP = "turbo"               # blue -> green -> yellow -> red


def load_depth(stem):
    """Return HxW uint16-safe array of raw depth values (R channel)."""
    path = os.path.join(TEST_DIR, f"{stem}_depth_0000.png")
    arr = np.array(Image.open(path))     # (H, W, 4) RGBA
    return arr[:, :, 0].astype(np.float64)   # R channel = depth (R=G=B)


def make_heatmap(depth, stem):
    """Color object pixels by depth with `turbo`; background -> white."""
    mask = depth > 0
    rgb = np.ones((*depth.shape, 3), dtype=np.float64)  # start all-white
    if mask.any():
        vals = depth[mask]
        lo, hi = vals.min(), vals.max()
        norm = (depth - lo) / (hi - lo + 1e-9)          # object -> [0,1]
        colored = cm.get_cmap(COLORMAP)(norm)[..., :3]  # drop alpha
        rgb[mask] = colored[mask]                        # keep bg white
    out = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    out_path = os.path.join(OUT_DIR, f"{stem}_heatmap.png")
    out.save(out_path)
    return out_path


def dump_txt(depth, stem):
    """Write 'u v depth' for every pixel (row-major)."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    table = np.column_stack([
        us.ravel().astype(int),
        vs.ravel().astype(int),
        depth.ravel().astype(int),
    ])
    out_path = os.path.join(OUT_DIR, f"{stem}_depth.txt")
    header = f"# {stem}_depth_0000.png  size={W}x{H}  columns: u v depth(raw 8-bit, 0=background)"
    np.savetxt(out_path, table, fmt="%d", header=header, comments="")
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for stem in IMAGES:
        depth = load_depth(stem)
        obj = depth[depth > 0]
        hp = make_heatmap(depth, stem)
        tp = dump_txt(depth, stem)
        print(f"{stem}: object pixels={obj.size}  raw depth range=[{int(obj.min())},{int(obj.max())}]")
        print(f"   heatmap -> {hp}")
        print(f"   txt     -> {tp}")


if __name__ == "__main__":
    main()
