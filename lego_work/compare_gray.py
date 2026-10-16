import json, numpy as np
for split in ("train", "val"):
    A = json.load(open(f"lego_gt_gray/transforms_{split}.json"))["frames"]
    B = json.load(open(f"lego_gt/transforms_{split}.json"))["frames"]
    for n in range(2):
        pa, pb = np.array(A[n]["transform_matrix"]), np.array(B[n]["transform_matrix"])
        da = np.load(f"lego_gt_gray/{split}/r_{n}_depth.npz")["depth"]
        db = np.load(f"lego_gt/{split}/r_{n}_depth.npz")["depth"]
        print(split, n, "pose diff", abs(pa - pb).max(),
              "| depth diff", abs(da - db).max(),
              "| same mask", bool(((da > 0) == (db > 0)).all()))
