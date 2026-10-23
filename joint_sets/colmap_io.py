"""Minimal reader for a COLMAP TEXT reconstruction (cameras/images/points3D).

Everything SLUE needs lives in these three files:
  cameras.txt  -> intrinsics  (f, cx, cy, k)  per CAMERA_ID
  images.txt   -> pose (qvec, tvec) world->cam, plus every 2D observation y_ik
  points3D.txt -> the reconstructed X_k and its track

No dependency on database.db: images.txt already carries the keypoint pixel
coordinates together with the POINT3D_ID they were matched to.
"""
import numpy as np


class Camera:
    __slots__ = ("id", "model", "w", "h", "params")

    def __init__(self, cid, model, w, h, params):
        self.id, self.model, self.w, self.h = cid, model, w, h
        self.params = np.asarray(params, float)

    # ---- intrinsics in a common (fx, fy, cx, cy, k1, k2) form -------------
    def unpack(self):
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            f, cx, cy = p;              return f, f, cx, cy, 0.0, 0.0
        if self.model == "PINHOLE":
            fx, fy, cx, cy = p;         return fx, fy, cx, cy, 0.0, 0.0
        if self.model == "SIMPLE_RADIAL":
            f, cx, cy, k = p;           return f, f, cx, cy, k, 0.0
        if self.model == "RADIAL":
            f, cx, cy, k1, k2 = p;      return f, f, cx, cy, k1, k2
        raise NotImplementedError(f"camera model {self.model} not handled")


class Image:
    __slots__ = ("id", "qvec", "tvec", "camera_id", "name", "xys", "p3d_ids")

    def __init__(self, iid, qvec, tvec, camera_id, name, xys, p3d_ids):
        self.id, self.camera_id, self.name = iid, camera_id, name
        self.qvec = np.asarray(qvec, float)
        self.tvec = np.asarray(tvec, float)
        self.xys = np.asarray(xys, float).reshape(-1, 2)
        self.p3d_ids = np.asarray(p3d_ids, np.int64)

    @property
    def R(self):
        return qvec2rotmat(self.qvec)

    @property
    def center(self):
        """Camera centre c_i in world coordinates (X_cam = R (X - c))."""
        return -self.R.T @ self.tvec


class Point3D:
    __slots__ = ("id", "xyz", "rgb", "error", "image_ids", "point2D_idxs")

    def __init__(self, pid, xyz, rgb, error, image_ids, point2D_idxs):
        self.id, self.error = pid, error
        self.xyz = np.asarray(xyz, float)
        self.rgb = np.asarray(rgb, int)
        self.image_ids = np.asarray(image_ids, np.int64)
        self.point2D_idxs = np.asarray(point2D_idxs, np.int64)

    @property
    def track_length(self):
        return len(self.image_ids)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]])


def read_cameras(path):
    cams = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            cams[int(t[0])] = Camera(int(t[0]), t[1], int(t[2]), int(t[3]),
                                     [float(v) for v in t[4:]])
    return cams


def read_images(path):
    imgs = {}
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        h = lines[i].split()
        if not h:
            continue
        iid = int(h[0])
        qvec = [float(v) for v in h[1:5]]
        tvec = [float(v) for v in h[5:8]]
        cam_id, name = int(h[8]), h[9]
        d = lines[i + 1].split()
        arr = np.asarray(d, float).reshape(-1, 3) if d else np.zeros((0, 3))
        imgs[iid] = Image(iid, qvec, tvec, cam_id, name,
                          arr[:, :2], arr[:, 2].astype(np.int64))
    return imgs


def read_points3D(path):
    pts = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            pid = int(t[0])
            xyz = [float(v) for v in t[1:4]]
            rgb = [int(v) for v in t[4:7]]
            err = float(t[7])
            track = np.asarray(t[8:], np.int64).reshape(-1, 2)
            pts[pid] = Point3D(pid, xyz, rgb, err, track[:, 0], track[:, 1])
    return pts


def read_model(folder):
    import os
    j = os.path.join
    return (read_cameras(j(folder, "cameras.txt")),
            read_images(j(folder, "images.txt")),
            read_points3D(j(folder, "points3D.txt")))


# ---------------------------------------------------------------------------
# projection  pi(theta_i, X)
# ---------------------------------------------------------------------------
def project(R, t, intr, X):
    """X: (...,3) world points -> (...,2) pixels. COLMAP SIMPLE_RADIAL/RADIAL."""
    fx, fy, cx, cy, k1, k2 = intr
    X = np.atleast_2d(X)
    Xc = X @ R.T + t                      # camera frame
    z = Xc[:, 2]
    xn = Xc[:, 0] / z
    yn = Xc[:, 1] / z
    r2 = xn * xn + yn * yn
    d = 1.0 + k1 * r2 + k2 * r2 * r2
    return np.stack([fx * d * xn + cx, fy * d * yn + cy], axis=-1), z
