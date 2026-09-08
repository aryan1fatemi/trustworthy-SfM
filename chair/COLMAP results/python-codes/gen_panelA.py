"""Generate panel A of the SLUE diagram: three rectangular viewing pyramids
meeting at the landmark, in 3D perspective (matching the hand sketch).

The geometry is real: cameras are placed in 3D, each image plane is a rectangle
perpendicular to its viewing ray, the pixel tolerance box is a square on that
plane, and P_ik is the 4-sided pyramid through that square's corners.  Everything
is then projected with a single perspective view camera, so the pyramids really
do converge on X_hat.

Writes the panel-A fragment; assemble_diagram.py splices it into the full figure.
"""
import numpy as np

# --------------------------------------------------------------------------
# 3D scene
# --------------------------------------------------------------------------
XHAT = np.array([0.0, 0.0, 0.0])                 # the landmark

def _sph(R, az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([R * np.cos(az) * np.cos(el),
                     R * np.sin(az) * np.cos(el),
                     R * np.sin(el)])


R_CAM, EL_CAM = 4.20, -19.0          # cameras on a ring below the landmark
CAMS = [
    (1, _sph(R_CAM, 358.0, EL_CAM)),   # right
    (2, _sph(R_CAM, 262.0, EL_CAM)),   # front / bottom
    (3, _sph(R_CAM, 150.0, EL_CAM)),   # left
]

DP = 1.15          # image plane distance from the camera centre
PW, PH = 1.50, 1.05  # image plane rectangle size
BOX = 0.115        # half-size of the pixel tolerance box on that plane
EXT = 1.34         # pyramid length as a multiple of |X - c|

WORLD_UP = np.array([0.0, 0.0, 1.0])


def frame(c):
    """Right-handed camera frame looking from c at XHAT."""
    f = XHAT - c
    f = f / np.linalg.norm(f)
    r = np.cross(f, WORLD_UP)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


def camera_geometry(c):
    f, r, u = frame(c)
    ctr = c + f * DP
    plane = [ctr + a * (PW / 2) * r + b * (PH / 2) * u
             for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    box = [ctr + a * BOX * r + b * BOX * u
           for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    L = np.linalg.norm(XHAT - c) * EXT
    pyr = [c + (corner - c) / DP * L for corner in box]     # rays through the box
    return dict(c=c, f=f, r=r, u=u, ctr=ctr, plane=plane, box=box, pyr=pyr)


# --------------------------------------------------------------------------
# view camera -> 2D
# --------------------------------------------------------------------------
EYE = _sph(10.5, 272.0, 22.0)
LOOK = np.array([-0.20, -0.90, -1.00])
FOCAL = 1150.0


def view_frame():
    f = LOOK - EYE
    f = f / np.linalg.norm(f)
    r = np.cross(f, WORLD_UP)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


_VF = view_frame()


def proj(P):
    """World -> raw image coords (before fitting)."""
    P = np.atleast_2d(np.asarray(P, float))
    d = P - EYE
    f, r, u = _VF
    z = d @ f
    return np.stack([FOCAL * (d @ r) / z, -FOCAL * (d @ u) / z], -1)


# --------------------------------------------------------------------------
def build(box=(24, 74, 596, 548)):
    """Return (svg fragment, dict of 2D anchor points)."""
    geo = [(i, camera_geometry(c)) for i, c in CAMS]

    pts = [XHAT]
    for _, g in geo:
        pts += [g["c"], *g["plane"], *g["pyr"], g["ctr"]]
    P = proj(np.asarray(pts))

    x0, y0, x1, y1 = box
    lo, hi = P.min(0), P.max(0)
    s = min((x1 - x0) / (hi[0] - lo[0]), (y1 - y0) / (hi[1] - lo[1]))
    off = np.array([x0 + (x1 - x0 - s * (hi[0] - lo[0])) / 2 - s * lo[0],
                    y0 + (y1 - y0 - s * (hi[1] - lo[1])) / 2 - s * lo[1]])

    def T(Q):
        q = proj(Q) * s + off
        return q[0] if np.asarray(Q).ndim == 1 else q

    def path(ptsw, close=True):
        q = T(np.asarray(ptsw))
        d = "M " + " L ".join("%.1f,%.1f" % tuple(p) for p in q)
        return d + (" Z" if close else "")

    COL = {1: "#2b6cb0", 2: "#c05621", 3: "#2f855a"}
    DARK = {1: "#1a365d", 2: "#7b341e", 3: "#22543d"}

    out = []
    A = {}

    # ---- pyramids first (behind everything) --------------------------
    for i, g in geo:
        col = COL[i]
        c2 = T(g["c"])
        pyr = T(np.asarray(g["pyr"]))
        # two side faces give the pyramid its body
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            out.append('  <path d="M %.1f,%.1f L %.1f,%.1f L %.1f,%.1f Z" '
                       'fill="%s" fill-opacity="0.10" stroke="%s" '
                       'stroke-width="0.9" stroke-opacity="0.5"/>'
                       % (c2[0], c2[1], pyr[a][0], pyr[a][1],
                          pyr[b][0], pyr[b][1], col, col))
        out.append('  <path d="%s" fill="none" stroke="%s" stroke-width="1.1" '
                   'stroke-opacity="0.75"/>' % (path(g["pyr"]), col))

    # ---- image planes, tolerance boxes, keypoints --------------------
    for i, g in geo:
        col, dark = COL[i], DARK[i]
        out.append('  <path d="%s" fill="#ffffff" fill-opacity="0.72" '
                   'stroke="#4a5568" stroke-width="1.3"/>' % path(g["plane"]))
        out.append('  <path d="%s" fill="%s" fill-opacity="0.30" stroke="%s" '
                   'stroke-width="1.6" stroke-dasharray="3.5 2.5"/>'
                   % (path(g["box"]), col, col))
        k = T(g["ctr"])
        out.append('  <circle cx="%.1f" cy="%.1f" r="2.9" fill="%s"/>'
                   % (k[0], k[1], dark))
        A["y%d" % i] = k
        A["plane%d" % i] = T(np.asarray(g["plane"]))
        c2 = T(g["c"])
        out.append('  <circle cx="%.1f" cy="%.1f" r="4.2" fill="#1f2933"/>'
                   % (c2[0], c2[1]))
        A["c%d" % i] = c2
        A["box%d" % i] = T(np.asarray(g["box"]))

    # ---- the landmark ------------------------------------------------
    xq = T(XHAT)
    out.append('  <circle cx="%.1f" cy="%.1f" r="6.2" fill="#1a365d"/>' % tuple(xq))
    A["X"] = xq
    return "\n".join(out), A


if __name__ == "__main__":
    frag, anchors = build()
    for k, v in sorted(anchors.items()):
        print(k, np.round(np.asarray(v), 1).tolist())
