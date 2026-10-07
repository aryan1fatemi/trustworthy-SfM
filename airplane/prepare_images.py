"""
Prepare a COLMAP-ready images/ folder from a NeRF-synthetic-style dataset (chair, airplane, ...).

- Composites each RGBA render onto a solid WHITE background (COLMAP ignores
  alpha, and a transparent background otherwise becomes featureless black).
- Merges the train and val splits into one folder, renaming to avoid the
  filename collision (both splits use r_0.png, r_1.png, ...):
      train/r_N.png -> images/r_N.png
      val/r_N.png   -> images/r_N_val.png
  This keeps every output filename traceable back to its entry in
  transforms_train.json / transforms_val.json (file_path "./train/r_N" or
  "./val/r_N").
"""

import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))  # dataset folder this script sits in
OUT = os.path.join(HERE, "images")
BG = (255, 255, 255)  # white background

# (source split dir, output filename suffix)
SPLITS = [("train", ""), ("val", "_val")]


def flatten_to_white(src_path, dst_path):
    im = Image.open(src_path).convert("RGBA")
    bg = Image.new("RGBA", im.size, BG + (255,))
    composed = Image.alpha_composite(bg, im).convert("RGB")
    composed.save(dst_path)


def main():
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for split, suffix in SPLITS:
        src_dir = os.path.join(HERE, split)
        names = sorted(n for n in os.listdir(src_dir) if n.lower().endswith(".png"))
        for name in names:
            stem = os.path.splitext(name)[0]          # r_0
            dst_name = f"{stem}{suffix}.png"           # r_0 or r_0_val
            flatten_to_white(os.path.join(src_dir, name),
                             os.path.join(OUT, dst_name))
            total += 1
        print(f"{split}: {len(names)} images -> images/ (suffix '{suffix}')")
    print(f"Done. {total} images written to {OUT}")


if __name__ == "__main__":
    main()
