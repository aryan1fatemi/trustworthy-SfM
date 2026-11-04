"""COLMAP model vs Blender GT: registration, points, track lengths, residuals, pose errors after a similarity
alignment of the camera centres (Umeyama).
    python model_vs_gt.py --scene ../../chair [--model "COLMAP results fixedK"]"""
import os, sys, argparse
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geom import Scene
from colmap_io import read_model
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--model", default="COLMAP results fixedK")
a = ap.parse_args()
S = os.path.abspath(a.scene)
cams, imgs, pts = read_model(os.path.join(S, a.model))
gt = Scene.load_gt(SimpleNamespace(scene=S))
names = [im.name for im in imgs.values()]
print("%s | %s: registered %d / %d images, %d points" % (os.path.basename(S), a.model, len(imgs), len(gt), len(pts)))
L = np.array([len(P.image_ids) for P in pts.values()])
print("track length: median %d, mean %.1f, p90 %d, max %d; detections %d" % (np.median(L), L.mean(), np.quantile(L, .9), L.max(), L.sum()))
res = []
for P in pts.values():
    for i, j in zip(P.image_ids, P.point2D_idxs):
        im = imgs[i]; fx, fy, cx, cy = cams[im.camera_id].unpack()[:4]
        q = im.R @ P.xyz + im.tvec
        res.append(np.max(np.abs([fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy] - im.xys[j])))
res = np.array(res)
print("COLMAP residual |r|_inf: median %.2f, p90 %.2f, max %.2f px" % (np.median(res), np.quantile(res, .9), res.max()))
ids = [i for i in imgs if imgs[i].name in gt]
A = np.array([imgs[i].center for i in ids]); B = np.array([gt[imgs[i].name]["c"] for i in ids])
ma, mb = A.mean(0), B.mean(0); U, s, Vt = np.linalg.svd((B - mb).T @ (A - ma))
D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Rs = U @ D @ Vt; sc = (s * np.diag(D)).sum() / ((A - ma) ** 2).sum()
Aa = sc * (A - ma) @ Rs.T + mb
ce = np.linalg.norm(Aa - B, axis=1)
re = np.array([np.degrees(np.linalg.norm(Rotation.from_matrix(gt[imgs[i].name]["R"] @ Rs @ imgs[i].R.T).as_rotvec())) for i in ids])
print("after similarity alignment: centre error median %.4f, p90 %.4f, max %.4f (scene units; camera distance 4.03)" % (np.median(ce), np.quantile(ce, .9), ce.max()))
print("rotation error: median %.3f, p90 %.3f, p99 %.3f, max %.3f deg" % (np.median(re), np.quantile(re, .9), np.quantile(re, .99), re.max()))
# gauge-free check: rotation error after aligning by rotations alone (best global rotation, chordal L2 mean),
# and relative rotation error over all camera pairs
Q = [gt[imgs[i].name]["R"].T @ imgs[i].R for i in ids]                  # = G^T for the global rotation G
U2, _, V2 = np.linalg.svd(sum(Q)); G = (U2 @ np.diag([1, 1, np.linalg.det(U2 @ V2)]) @ V2).T
re2 = np.array([np.degrees(np.linalg.norm(Rotation.from_matrix(gt[imgs[i].name]["R"] @ G.T @ imgs[i].R.T).as_rotvec())) for i in ids])
print("rotation error, rotations aligned directly: median %.3f, p90 %.3f, max %.3f deg" % (np.median(re2), np.quantile(re2, .9), re2.max()))
rel = []
for x in range(len(ids)):
    for y in range(x + 1, len(ids)):
        a1, b1 = imgs[ids[x]], imgs[ids[y]]
        Rc = b1.R @ a1.R.T; Rg = gt[b1.name]["R"] @ gt[a1.name]["R"].T
        rel.append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rg.T @ Rc).as_rotvec())))
print("relative rotation error over all pairs: median %.3f, p90 %.3f, max %.3f deg" % (np.median(rel), np.quantile(rel, .9), np.max(rel)))
