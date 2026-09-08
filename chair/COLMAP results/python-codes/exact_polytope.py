"""Exact H-representation of F_k -- the reference the brute force is checked against.

Key fact: with the cameras FIXED, the constraint is *linear in X*.  Writing
X_c = R X + t and n = (X_c1/X_c3, X_c2/X_c3), the pinhole constraint

        | f n_x + c_x - u |  <=  eps        (and the same in v)

multiplies out, using X_c3 > 0, to four half-spaces in X:

        ( R_1 - (n*_x + e) R_3 ) . X  <=  -( t_1 - (n*_x + e) t_3 )
       -( R_1 - (n*_x - e) R_3 ) . X  <=   ( t_1 - (n*_x - e) t_3 )      etc.

so F_k = intersection of 4|views| half-spaces (+ cheirality) -- a convex
polyhedron.  The interval hull is then six LPs and the volume is one
half-space intersection, both exact and ~1000x faster than any grid.

Radial distortion is folded in by undistorting the measurement y_ik to a
normalised point n* and scaling the tolerance by the local distortion gain;
for this dataset (|k1| <~ 0.06, r <~ 0.36) that correction is ~0.3%.
"""
import os
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def undistort(uv, intr, iters=20):
    """pixel -> normalised camera point n*, and the local distortion gain."""
    fx, fy, cx, cy, k1, k2 = intr
    n = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy])
    for _ in range(iters):
        r2 = n @ n
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        n = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy]) / d
    r2 = n @ n
    d = 1.0 + k1 * r2 + k2 * r2 * r2
    gain = d + 2.0 * r2 * (k1 + 2.0 * k2 * r2)      # d/dn of (d*n), radial part
    return n, gain


def halfspaces(views, eps, z_min=1e-3):
    """Rows [a1 a2 a3 b] meaning a.X + b <= 0 (scipy HalfspaceIntersection form)."""
    H = []
    for (R, t, intr, y) in views:
        fx, fy = intr[0], intr[1]
        n_star, gain = undistort(np.asarray(y, float), intr)
        e = np.array([eps / (fx * gain), eps / (fy * gain)])
        for a in (0, 1):                       # x then y normalised coordinate
            for s in (+1.0, -1.0):
                # s*(n_a - n*_a) <= e_a   ->   s*(R_a - (n*_a + s e_a) R_3).X + ... <= 0
                lim = n_star[a] + s * e[a]
                row = s * (R[a] - lim * R[2])
                off = s * (t[a] - lim * t[2])
                H.append(np.append(row, off))
        H.append(np.append(-R[2], -t[2] + z_min))          # cheirality X_c3 >= z_min
    return np.asarray(H)


def chebyshev_center(H):
    """Largest inscribed ball; also the feasibility test.  Returns (x, radius)."""
    A, b = H[:, :3], H[:, 3]
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    A_ub = np.hstack([A, norm])
    res = linprog(np.array([0.0, 0.0, 0.0, -1.0]), A_ub=A_ub, b_ub=-b,
                  bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if not res.success:
        return None, -1.0
    return res.x[:3], float(res.x[3])


def interval_hull(H):
    """Exact axis-aligned bounding box of the polyhedron (six LPs)."""
    A, b = H[:, :3], H[:, 3]
    lo = np.empty(3)
    hi = np.empty(3)
    for a in range(3):
        c = np.zeros(3)
        c[a] = 1.0
        r = linprog(c, A_ub=A, b_ub=-b, bounds=[(None, None)] * 3, method="highs")
        lo[a] = r.x[a] if r.success else -np.inf
        r = linprog(-c, A_ub=A, b_ub=-b, bounds=[(None, None)] * 3, method="highs")
        hi[a] = r.x[a] if r.success else np.inf
    return lo, hi


def vertices_and_volume(H, interior):
    """V-representation + exact volume of the (bounded) polytope."""
    hs = HalfspaceIntersection(H, np.asarray(interior, float))
    V = hs.intersections
    hull = ConvexHull(V)
    return V[hull.vertices], float(hull.volume), float(hull.area)


def analyse(views, eps, X0=None):
    """Everything SLUE wants, exactly."""
    H = halfspaces(views, eps)
    ctr, radius = chebyshev_center(H)
    if ctr is None or radius <= 0:
        return {"empty": True}

    lo, hi = interval_hull(H)
    out = {"empty": False,
           "chebyshev_center": ctr.tolist(),
           "inradius": radius,
           "interval_hull_lo": lo.tolist(),
           "interval_hull_hi": hi.tolist(),
           "interval_extent": (hi - lo).tolist(),
           "max_interval_extent": float((hi - lo).max()),
           "n_halfspaces": int(H.shape[0])}
    try:
        V, vol, area = vertices_and_volume(H, ctr)
        C = V - V.mean(0)
        Vt = np.linalg.svd(C, full_matrices=False)[2]
        P = C @ Vt.T
        ext = P.max(0) - P.min(0)
        order = np.argsort(-ext)
        out.update({"volume": vol, "surface_area": area,
                    "n_vertices": int(V.shape[0]),
                    "pca_extent": ext[order].tolist(),
                    "pca_axes_long_to_short": Vt[order].tolist(),
                    "v3_weak_direction": Vt[order][0].tolist(),
                    "anisotropy": float(ext[order][0] / max(ext[order][2], 1e-15))})
    except Exception as exc:                       # unbounded / degenerate
        out["volume_error"] = repr(exc)
    if X0 is not None:
        out["X_colmap"] = np.asarray(X0).tolist()
        out["colmap_inside"] = bool(np.all(H[:, :3] @ X0 + H[:, 3] <= 1e-12))
    return out
