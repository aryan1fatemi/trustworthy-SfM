
import json, numpy as np, cv2

D = "lego_gt"
VIEWS = {"train": [0, 50, 99], "val": [0, 50, 99]}

for split, ns in VIEWS.items():
    meta = json.load(open(f"{D}/transforms_{split}.json"))
    W, H, F = meta["w"], meta["h"], meta["fl_x"]
    print(f"== {split}: {len(meta['frames'])} frames in json")
    for n in ns:
        M = np.array(meta["frames"][n]["transform_matrix"])
        depth = np.load(f"{D}/{split}/r_{n}_depth.npz")["depth"]
        img = cv2.imread(f"{D}/{split}/r_{n}.png", cv2.IMREAD_UNCHANGED)
        hit, alpha = depth > 0, img[..., 3] > 127
        iou = (hit & alpha).sum() / (hit | alpha).sum()

        rr, cc = np.nonzero(hit)
        d = depth[rr, cc]
        cam = np.stack([(cc + 0.5 - W / 2) / F * d, -(rr + 0.5 - H / 2) / F * d,
                        -d, np.ones_like(d)])
        radius = np.linalg.norm((M @ cam)[:3], axis=0).max()
        print(f"{split} r_{n}: object px {hit.sum()}, IoU {iou:.4f}, "
              f"depth {d.min():.3f}-{d.max():.3f}, max radius {radius:.4f}")
