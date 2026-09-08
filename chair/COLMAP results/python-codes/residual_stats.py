"""Sanity check + epsilon selection.

F_k is non-empty only if some X makes EVERY view's residual fit in the box, so
the relevant statistic is max_i ||res||_inf over the track -- not COLMAP's
reported mean L2 ERROR.  This prints the distribution of that max at X = X_k.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_model, project

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

cams, imgs, pts = read_model(MODEL)
print(f"{len(cams)} cameras, {len(imgs)} images, {len(pts)} points3D")

# cache per-image (R, t, intrinsics)
pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}

maxinf, meanl2, tlen, colmap_err, behind = [], [], [], [], []
for pid, p in pts.items():
    res = []
    nb = 0
    for iid, idx in zip(p.image_ids, p.point2D_idxs):
        R, t, intr = pose[iid]
        uv, z = project(R, t, intr, p.xyz)
        if z[0] <= 0:
            nb += 1
        res.append(uv[0] - imgs[iid].xys[idx])
    res = np.asarray(res)
    maxinf.append(np.abs(res).max())
    meanl2.append(np.linalg.norm(res, axis=1).mean())
    tlen.append(len(res)); colmap_err.append(p.error); behind.append(nb)

maxinf = np.asarray(maxinf); meanl2 = np.asarray(meanl2)
tlen = np.asarray(tlen); colmap_err = np.asarray(colmap_err)

print("\nreproduction check (our mean-L2 vs COLMAP ERROR): "
      f"max abs diff = {np.abs(meanl2 - colmap_err).max():.3e}")
print(f"points with a view behind the camera: {int((np.asarray(behind)>0).sum())}")

qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("\ntrack length      percentiles:", np.percentile(tlen, qs).round(1))
print("mean  L2 residual percentiles:", np.percentile(meanl2, qs).round(3))
print("MAX  Linf residual percentiles:", np.percentile(maxinf, qs).round(3))

print("\n  epsilon | frac. of landmarks that could have non-empty F_k")
for e in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    print(f"  {e:7.1f} | {(maxinf <= e).mean():6.1%}   "
          f"(track>=5: {(maxinf[tlen>=5] <= e).mean():6.1%})")
