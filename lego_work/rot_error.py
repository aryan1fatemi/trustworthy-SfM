import sys, os
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "js_local"))
from geom import Scene, registration_order
from colmap_io import read_model
from scipy.spatial.transform import Rotation

SCENE = "lego30"
MODEL = os.path.join(SCENE, "COLMAP results fixedK")
_, imgs_all, _ = read_model(MODEL)
cams = registration_order(SCENE, imgs_all, 22, MODEL)
sc = Scene(SCENE, MODEL, cams)
gt = sc.load_gt()
gt_pose, _ = sc.gt_in_gauge(gt, cams, [])

errs = []
for i in cams:
    if i not in gt_pose:
        continue
    d = Rotation.from_matrix(sc.pose(i)[0].T @ gt_pose[i][0]).as_rotvec()
    errs.append((np.degrees(np.linalg.norm(d)), sc.imgs[i].name))

errs.sort()
for a, n in errs:
    print(f"{n:9} {a:.4f} deg")
e = np.array([a for a, _ in errs])
print(f"\n{len(e)} cameras | median {np.median(e):.4f} | mean {e.mean():.4f} | max {e.max():.4f} deg ({errs[-1][1]})")
