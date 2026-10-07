"""Copies of a scene's images with additive uniform pixel noise.

    python make_noisy.py --scene airplane --amp 4 --tag noise_low
    python make_noisy.py --scene airplane --amp 20 --tag noise_high

I' = clip(round(I + U(-amp, amp)), 0, 255), independent per pixel and channel, one seed per
image (reproducible). Output: nerf_synthetic/<scene>_<tag>/images/ with the same file names.
"""
import os
import sys
import json
import argparse
import zlib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    ap.add_argument("--amp", type=float, required=True, help="half-width of U(-amp, amp), 0-255 scale")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    src = os.path.join(ROOT, a.scene, "images")
    dst_root = os.path.join(ROOT, "%s_%s" % (a.scene, a.tag))
    dst = os.path.join(dst_root, "images")
    os.makedirs(dst, exist_ok=True)
    names = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    for n in names:
        I = np.asarray(Image.open(os.path.join(src, n)).convert("RGB"), np.float64)
        rng = np.random.default_rng(zlib.crc32(n.encode()))
        J = np.clip(np.rint(I + rng.uniform(-a.amp, a.amp, I.shape)), 0, 255).astype(np.uint8)
        Image.fromarray(J).save(os.path.join(dst, n))
    json.dump({"source": src, "noise": "uniform U(-amp, amp) per pixel and channel, then round+clip",
               "amp": a.amp, "std": a.amp / np.sqrt(3), "images": len(names)},
              open(os.path.join(dst_root, "noise.json"), "w"), indent=2)
    print("%d images -> %s  (amp %g, std %.2f)" % (len(names), dst, a.amp, a.amp / np.sqrt(3)))


if __name__ == "__main__":
    main()
