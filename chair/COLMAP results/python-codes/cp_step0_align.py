"""Conformal task, Step 0 (conventions) and Step 1 (similarity alignment).

Nothing here touches depth.  Outputs go to ../conformal/.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_model, project
from feasible_set import MODEL

SCENE = os.path.abspath(os.path.join(MODEL, ".."))
OUT = os.path.join(MODEL, "conformal")
FLIP = np.diag([1.0, -1.0, -1.0])          # OpenGL/Blender camera -> OpenCV camera


def load_gt():
    """name-in-COLMAP -> (R_w2c OpenCV, centre, f_gt). Mapping follows prepare_images.py."""
    gt = {}
    for split, suffix in (("train", ""), ("val", "_val"), ("test", "_test")):
        d = json.load(open(os.path.join(SCENE, "transforms_%s.json" % split)))
        f = 0.5 * 800 / np.tan(0.5 * d["camera_angle_x"])
        for fr in d["frames"]:
            stem = os.path.basename(fr["file_path"])            # r_N
            c2w = np.array(fr["transform_matrix"], float)
            R_c2w = c2w[:3, :3] @ FLIP                            # flip camera y, z
            gt[stem + suffix + ".png"] = (R_c2w.T, c2w[:3, 3].copy(), f)
    return gt


def umeyama(src, dst):
    """s, R, t minimising sum ||dst - (s R src + t)||^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def main():
    os.makedirs(OUT, exist_ok=True)
    cams, imgs, pts = read_model(MODEL)
    gt = load_gt()
    rep = {}

    # 0.4 name mapping ---------------------------------------------------
    names = [im.name for im in imgs.values()]
    missing = [n for n in names if n not in gt]
    rep["colmap_images"] = len(names)
    rep["mapped_by_filename"] = len(names) - len(missing)
    rep["unmapped"] = missing

    # 0.5 regression -----------------------------------------------------
    pose = {i: (im.R, im.tvec, cams[im.camera_id].unpack()) for i, im in imgs.items()}
    worst = 0.0
    for p in pts.values():
        r = [np.linalg.norm(project(*pose[i][:3], p.xyz)[0][0] - imgs[i].xys[j])
             for i, j in zip(p.image_ids, p.point2D_idxs)]
        worst = max(worst, abs(np.mean(r) - p.error))
    rep["error_column_max_abs_diff_px"] = worst

    # Step 1 alignment ---------------------------------------------------
    ids = [i for i in imgs if imgs[i].name in gt]
    c_hat = np.array([imgs[i].center for i in ids])
    c_gt = np.array([gt[imgs[i].name][1] for i in ids])
    s, Ra, ta = umeyama(c_gt, c_hat)
    res = np.linalg.norm(c_hat - (s * c_gt @ Ra.T + ta), axis=1)
    rmse = float(np.sqrt((res ** 2).mean()))
    spread = float(np.median(np.linalg.norm(c_hat - c_hat.mean(0), axis=1)))
    rep.update(n_align=len(ids), scale=s, rmse=rmse, median_radius=spread,
               rmse_normalised=rmse / spread, max_centre_err=float(res.max()))

    # 0.1 convention check: rotations must agree after alignment ---------
    #   R_colmap = R_gt_cv @ Ra^T  if the flip is right
    def ang(Rm):
        return np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    for label, F in (("with_flip", np.eye(3)), ("without_flip", FLIP)):
        e = [ang(imgs[i].R @ (F @ gt[imgs[i].name][0] @ Ra.T).T) for i in ids]
        rep["rot_err_deg_" + label] = [float(np.median(e)), float(np.max(e))]

    f_col = np.array([cams[imgs[i].camera_id].unpack()[0] for i in ids])
    rep["f_gt"] = gt[imgs[ids[0]].name][2]
    rep["f_colmap_range"] = [float(f_col.min()), float(np.median(f_col)), float(f_col.max())]

    np.savez(os.path.join(OUT, "alignment.npz"), s=s, R=Ra, t=ta)
    json.dump(rep, open(os.path.join(OUT, "step0_step1.json"), "w"), indent=2)
    for k, v in rep.items():
        print("%-32s %s" % (k, v))


if __name__ == "__main__":
    main()
