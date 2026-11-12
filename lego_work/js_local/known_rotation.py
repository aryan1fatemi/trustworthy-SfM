"""Known-rotation joint set: every rotation fixed at COLMAP's, all camera centres and all points solved together.

    python known_rotation.py --scene /path/to/airplane --n-cams 30 --solver highs-ipm
    python known_rotation.py --scene /path/to/airplane --n-cams 200 --solver cuopt      (GPU, Linux / WSL2)

With R_i fixed, every window condition pi((R_i, C_i), X_k) in W_ik (with depth > z_min) is linear in (C_i, X_k):
    n^T R_i (X_k - C_i) <= 0   for the 4 faces n of the window's pyramid,     r_i3^T (X_k - C_i) >= z_min,
so the joint set F(eps | R) = {(C_1..C_N, X_1..X_M) : all rows hold} is one polytope (one LP, no linearisation).
Gauge: camera 1 at the origin, x of camera 2's centre = +-1, sign kept (rotations are fixed, so the rotation gauge is too).
Steps:
  1. feasibility: min s  s.t. every row holds up to s >= 0  (F(eps | R) non-empty iff s* = 0);
  2. ranges: min / max of each centre coordinate over F(eps | R) (the exact projection; other cameras and all
     points free), optionally the same for --point-ranges points.
The sets are conditional on the rotations being COLMAP's (the true rotations differ slightly).
Writes <out>/<dataset>_known_rotation_<n>cams.json.
--rot <json from rotation_bounds.py> makes the rotations unknowns too: R_i = R_hat_i Exp(delta_i) with delta_i in
per-camera bounds, linearised in delta with a slack that covers the linearisation error (see rot_slack), so the
set contains the truth whenever the true rotation errors are inside those bounds.
--save-duals (HiGHS) also writes each range LP's dual weights, summed per track, to <checkpoint>.duals.jsonl:
the rows with y_r > 0 are the ones whose weighted sum certifies that bound; a track with zero weight can be
removed without changing it.
"""
import os
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool, TimeoutError as PoolTimeout
from scipy.sparse import coo_matrix, csr_matrix, hstack

from geom import Scene, box_normals, registration_order, run_name
from colmap_io import read_model


# ---------------------------------------------------------------------------- the LP
def detection_rows(sc, i, k, eps, scheme, zmin):
    """(G, h): rows G (X_k - C_i) <= h of every detection of landmark k in camera i at rotation R_hat_i
    (4 window faces + depth per detection); the rows of G are unit vectors."""
    R = sc.pose(i)[0]
    G, h = [], []
    for xn, e in zip(sc.det[i][k], sc.eps_list(i, k, eps, scheme)):
        G += [n @ R for n in box_normals(xn, sc.f[i], e)] + [-R[2]]
        h += [0, 0, 0, 0, -zmin]
    return np.array(G), np.array(h, float)


def rot_slack(theta, D, dnorm):
    """Rotation R = R_hat Exp(delta), |delta| <= theta. A row g (X - C) <= h at the true rotation implies
        g (X - C) + (d_hat x g) . delta <= h + rot_slack,      d_hat = X_hat - C_hat (COLMAP),
    when |(X - C) - d_hat| <= D: the dropped terms are g [delta]x ((X - C) - d_hat) <= theta D and
    g (Exp(delta) - I - [delta]x)(X - C) <= (theta - sin theta + 1 - cos theta) |X - C|."""
    return theta * D + (theta - np.sin(theta) + 1 - np.cos(theta)) * (dnorm + D)


def build(sc, cams, eps, scheme, zmin, rot=None):
    """rows A x <= b over x = (C of cams[1:], X of every landmark seen >= 2 times[, delta of cams[1:]]);
    row_pt[r] = index into L of row r's landmark (the row's track).
    rot = (box, theta, D): camera rotations R_i = R_hat_i Exp(delta_i) with delta_i in box[i] = (lo, hi) (rad),
    |delta_i| <= theta[i], and D(i, k) >= |(X_k - C_i) - (X_hat_k - C_hat_i)|; each row gets the linearised
    rotation term and rot_slack (detections with D = inf are left out). Camera 1 keeps R = I (rotation gauge)."""
    cnt = {}
    for i in cams:
        for k in sc.det[i]:
            cnt[k] = cnt.get(k, 0) + 1
    L = sorted(k for k, c in cnt.items() if c >= 2)
    pi = {k: m for m, k in enumerate(L)}
    ci = {i: m for m, i in enumerate(cams[1:])}
    nC = 3 * len(ci)
    nP = nC + 3 * len(L)                                   # first rotation column
    ri, cj, val, b, row_pt = [], [], [], [], []
    row, dropped = 0, 0
    for i in cams:
        for k in sc.det[i]:
            if k not in pi:
                continue
            G, h = detection_rows(sc, i, k, eps, scheme, zmin)
            if rot is not None and i in ci:
                dhat = sc.point(k) - sc.pose(i)[1]
                D = rot[2](i, k)
                if not np.isfinite(D):
                    dropped += 1
                    continue
                h = h + rot_slack(rot[1][i], D, np.linalg.norm(dhat))
            for g, rhs in zip(G, h):
                for d in range(3):
                    ri.append(row); cj.append(nC + 3 * pi[k] + d); val.append(g[d])
                    if i in ci:
                        ri.append(row); cj.append(3 * ci[i] + d); val.append(-g[d])
                if rot is not None and i in ci:
                    for d, gd in enumerate(np.cross(dhat, g)):
                        ri.append(row); cj.append(nP + 3 * ci[i] + d); val.append(gd)
                b.append(rhs)
                row_pt.append(pi[k])
                row += 1
    if dropped:
        print("  rotation model: %d (camera, landmark) pairs left out (no bound D on X - C)" % dropped)
    nv = nP + (3 * len(ci) if rot is not None else 0)
    A = coo_matrix((val, (ri, cj)), shape=(row, nv)).tocsr()
    bounds = [(None, None)] * nv
    bounds[3 * ci[cams[1]]] = (sc.x2, sc.x2)             # scale gauge: x of camera 2's centre = +-1 (sign of COLMAP's)
    if rot is not None:
        for i, m in ci.items():
            lo, hi = rot[0][i]
            for d in range(3):
                bounds[nP + 3 * m + d] = (float(lo[d]), float(hi[d]))
    return A, np.array(b), bounds, L, ci, nC, np.array(row_pt)


# ---------------------------------------------------------------------------- solvers
class HiGHS:
    def __init__(self, A, b, bounds, method, time_limit=None):
        from scipy.optimize import linprog
        self.lp, self.A, self.b, self.bounds, self.method = linprog, A, b, bounds, method
        self.opt = {"time_limit": float(time_limit)} if time_limit else {}
        self.last = None                                                # scipy result of the last extreme() solve

    def feasibility(self):
        m, n = self.A.shape
        A2 = hstack([self.A, csr_matrix(-np.ones((m, 1)))]).tocsr()
        c = np.zeros(n + 1)
        c[-1] = 1
        r = self.lp(c, A_ub=A2, b_ub=self.b, bounds=self.bounds + [(0, None)], method=self.method)
        return (r.fun if r.status == 0 else None), r.message

    def extreme(self, j, sense):
        """min (sense=+1) or max (sense=-1) of x_j; None if not solved."""
        c = np.zeros(self.A.shape[1])
        c[j] = sense
        r = self.lp(c, A_ub=self.A, b_ub=self.b, bounds=self.bounds, method=self.method, options=self.opt)
        if r.status not in (0, 3) and self.method == "highs-ipm":
            # interior point stalls on unbounded / nearly unbounded directions; dual simplex detects them exactly
            r = self.lp(c, A_ub=self.A, b_ub=self.b, bounds=self.bounds, method="highs-ds", options=self.opt)
        self.last = r
        if r.status == 3:
            return -sense * np.inf                                      # unbounded: min = -inf / max = +inf
        return sense * r.fun if r.status == 0 else None                 # None: time limit


def duals(S, j, sense, row_pt, L, rel_tol=1e-9):
    """Dual weights of S's last solve (min of sense * x_j): y_r = -marginal_r >= 0 per row, with
    sense * e_j = -sum_r y_r a_r + (duals of the variable bounds), i.e. the bound is a y-weighted sum of rows.
    Returned summed per track (tracks below rel_tol * the largest weight dropped), with the duals of the
    variable bounds (the gauge coordinate) and the residual of that identity (checks the certificate)."""
    r = S.last
    if r is None or r.status != 0:
        return None
    m = np.asarray(r.ineqlin.marginals)
    mb = np.asarray(r.lower.marginals) + np.asarray(r.upper.marginals)
    c = np.zeros(S.A.shape[1])
    c[j] = sense
    resid = float(np.max(np.abs(c - S.A.T @ m - mb)))
    w = np.bincount(row_pt, weights=-m, minlength=len(L))
    keep = np.flatnonzero(w > rel_tol * max(w.max(), 1e-300))
    return {"tracks": {int(L[q]): float(w[q]) for q in keep}, "total": float(w.sum()),
            "bound_duals": {int(q): float(mb[q]) for q in np.flatnonzero(mb)}, "certificate_residual": resid}


_W = {}


def _init_worker(A, b, bounds, method, time_limit, row_pt=None, L=None):
    _W["S"] = HiGHS(A, b, bounds, method, time_limit)
    _W["row_pt"], _W["L"] = row_pt, L


def _extreme(task):
    j, sense = task
    t = time.time()
    v = _W["S"].extreme(j, sense)
    d = duals(_W["S"], j, sense, _W["row_pt"], _W["L"]) if _W["row_pt"] is not None else None
    return j, sense, v, time.time() - t, d


class CuOpt:
    """NVIDIA cuOpt (https://github.com/NVIDIA/cuopt), Python API cuopt.linear_programming (docs 26.10).
    The model is built once; only the objective changes between solves."""

    def __init__(self, A, b, bounds, method, tol):
        from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
        from cuopt.linear_programming.solver_settings import SolverSettings, SolverMethod
        self.MIN = MINIMIZE
        self.settings = SolverSettings()
        names = {"pdlp": ["PDLP"], "barrier": ["Barrier", "BARRIER"], "concurrent": ["Concurrent", "CONCURRENT"]}[method]
        member = next((getattr(SolverMethod, n) for n in names if hasattr(SolverMethod, n)), None)
        if member is None:
            raise SystemExit("cuOpt SolverMethod has no %s member (available: %s)" % (method, list(SolverMethod.__members__)))
        self.settings.set_parameter("method", member)
        for name in ("absolute_primal_tolerance", "relative_primal_tolerance", "absolute_dual_tolerance",
                     "relative_dual_tolerance", "absolute_gap_tolerance", "relative_gap_tolerance"):
            self.settings.set_parameter(name, tol)
        self.settings.set_parameter("infeasibility_detection", True)

        def var(p, lo, hi):
            return p.addVariable(lb=-float("inf") if lo is None else lo, ub=float("inf") if hi is None else hi,
                                 vtype=CONTINUOUS)

        def rows(p, x, extra=None):
            for r in range(A.shape[0]):
                s, e = A.indptr[r], A.indptr[r + 1]
                expr = x[A.indices[s]] * float(A.data[s])
                for q in range(s + 1, e):
                    expr = expr + x[A.indices[q]] * float(A.data[q])
                p.addConstraint((expr - extra if extra is not None else expr) <= float(b[r]))

        self.p = Problem("known_rotation")                      # ranges: A x <= b
        self.x = [var(self.p, lo, hi) for lo, hi in bounds]
        rows(self.p, self.x)
        self.pf = Problem("known_rotation_feasibility")         # feasibility: A x - s <= b, min s
        self.xf = [var(self.pf, lo, hi) for lo, hi in bounds]
        self.s = self.pf.addVariable(lb=0.0, vtype=CONTINUOUS)
        rows(self.pf, self.xf, self.s)
        self.pf.setObjective(self.s * 1.0, sense=MINIMIZE)

    def feasibility(self):
        self.pf.solve(self.settings)
        st = self.pf.Status.name
        return (self.pf.ObjValue if st == "Optimal" else None), "%s, cuOpt solve time %.1fs" % (st, self.pf.SolveTime)

    def extreme(self, j, sense):
        self.p.setObjective(self.x[j] * float(sense), sense=self.MIN)
        self.p.solve(self.settings)
        return sense * self.p.ObjValue if self.p.Status.name == "Optimal" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-cams", type=int, default=30, help="first n cameras of COLMAP's registration order")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--scheme", default="fixed", choices=["fixed", "mult", "add"])
    ap.add_argument("--solver", default="highs-ipm", choices=["highs-ipm", "highs-ds", "highs", "cuopt"])
    ap.add_argument("--cuopt-method", default="barrier", choices=["barrier", "pdlp", "concurrent"],
                    help="barrier: 1e-8-accurate; pdlp: fastest on big GPUs but ~1e-4 accurate")
    ap.add_argument("--tol", type=float, default=1e-8, help="cuOpt tolerances")
    ap.add_argument("--feas-tol", type=float, default=1e-7,
                    help="min violation counted as zero (solver round-off; rows are unit-normal, ~1e-7 << one pixel)")
    ap.add_argument("--camera-ranges", type=int, default=-1, help="cameras whose centre ranges are computed (-1: all)")
    ap.add_argument("--point-ranges", type=int, default=0, help="points whose box is computed (0: none)")
    ap.add_argument("--point-seed", type=int, default=0, help="random sample of --point-ranges points (-1: first by id)")
    ap.add_argument("--procs", type=int, default=1, help="parallel range LPs (HiGHS only)")
    ap.add_argument("--lp-time-limit", type=float, default=900, help="seconds per range LP (HiGHS); then reported unsolved")
    ap.add_argument("--save-duals", action="store_true",
                    help="HiGHS: save each range LP's dual weights per track (<checkpoint>.duals.jsonl)")
    ap.add_argument("--rot", default=None,
                    help="rotation bounds (json from rotation_bounds.py): rotations become unknowns "
                         "R_i = R_hat_i Exp(delta_i) with delta_i in the camera's bounds (linearised + slack)")
    ap.add_argument("--rot-D", type=float, default=0.5,
                    help="bound on |(X_k - C_i) - COLMAP's| where the --rot file has none (gauge units)")
    ap.add_argument("--rot-ranges", action="store_true", help="also compute min / max of every delta_i component")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.save_duals and a.solver == "cuopt":
        raise SystemExit("--save-duals is implemented for the HiGHS solvers only")
    T0 = time.time()
    model = a.model or os.path.join(a.scene, "COLMAP results")
    _, imgs_all, _ = read_model(model)
    cams = registration_order(a.scene, imgs_all, a.n_cams, model)
    sc = Scene(a.scene, model, cams)
    zmin = 1e-3
    rot = None
    if a.rot:
        J = json.load(open(a.rot))
        by_name = {sc.imgs[i].name: i for i in cams}
        box, theta = {}, {}
        for nm, r in J["cameras"].items():
            if nm in by_name:
                box[by_name[nm]], theta[by_name[nm]] = (r["lo"], r["hi"]), r["theta"]
        missing = [sc.imgs[i].name for i in cams[1:] if i not in box]
        if missing:
            raise SystemExit("--rot file has no bounds for %d cameras, e.g. %s" % (len(missing), missing[:3]))
        Dc = {by_name[nm]: v for nm, v in J.get("D_cam", {}).items() if nm in by_name}
        Dt = {int(k): v for k, v in J.get("D_track", {}).items()}
        fin = lambda v: v if v is not None and np.isfinite(v) else a.rot_D
        rot = (box, theta, lambda i, k: fin(Dc.get(i)) + fin(Dt.get(k)))
        print("rotation model: |delta| median %.3f deg, max %.3f deg (from %s)"
              % (np.degrees(np.median(list(theta.values()))), np.degrees(max(theta.values())), a.rot), flush=True)
    A, b, bounds, L, ci, nC, row_pt = build(sc, cams, a.eps, a.scheme, zmin, rot)
    nP = nC + 3 * len(L)
    print("%d cameras, %d points, %d rows, %d unknowns, %d nonzeros (built in %.0fs)"
          % (len(cams), len(L), A.shape[0], A.shape[1], A.nnz, time.time() - T0), flush=True)
    t = time.time()
    if a.solver == "cuopt":
        S = CuOpt(A, b, bounds, a.cuopt_method, a.tol)
        print("cuOpt model built (%.0fs)" % (time.time() - t), flush=True)
    else:
        S = HiGHS(A, b, bounds, a.solver)
    # COLMAP's own solution: largest row violation (it is a candidate point of F(eps | R))
    x_hat = np.concatenate([sc.pose(i)[1] for i in cams[1:]] + [sc.point(k) for k in L]
                           + ([np.zeros(3 * len(ci))] if rot else []))
    viol_hat = float(np.max(A @ x_hat - b))
    t = time.time()
    s, msg = S.feasibility()
    t_feas = time.time() - t
    print("feasibility: min violation %s (%s), %.1fs; COLMAP's solution violates by %.2e"
          % ("%.2e" % s if s is not None else "n/a", msg, t_feas, viol_hat), flush=True)
    out = {"scene": a.scene, "cameras": [sc.imgs[i].name for i in cams], "eps_px": a.eps, "scheme": a.scheme,
           "solver": a.solver + ("/" + a.cuopt_method if a.solver == "cuopt" else ""), "points": len(L),
           "rows": int(A.shape[0]), "unknowns": int(A.shape[1]), "min_violation": s, "colmap_violation": viol_hat,
           "feasibility_seconds": round(t_feas, 1), "centre_ranges": {}, "point_ranges": {},
           "rotation_model": a.rot, "rotation_ranges": {}}
    npts = min(max(a.point_ranges, 0), len(L))
    pts = (L[:npts] if a.point_seed < 0 else
           sorted(np.random.default_rng(a.point_seed).choice(L, npts, replace=False).tolist()))
    pos = {k: m for m, k in enumerate(L)}
    gt = sc.load_gt()
    gt_pose, Xgt = ({}, {})
    if gt is not None and all(sc.imgs[i].name in gt for i in cams):
        gt_pose, Xgt = sc.gt_in_gauge(gt, cams, pts)
    if s is not None and s <= a.feas_tol:
        todo = cams[1:] if a.camera_ranges < 0 else cams[1:1 + a.camera_ranges]
        fixed = lambda j: bounds[j][0] is not None and bounds[j][0] == bounds[j][1]
        cols = [3 * ci[i] + d for i in todo for d in range(3) if not fixed(3 * ci[i] + d)]
        cols += [nC + 3 * pos[k] + d for k in pts for d in range(3)]
        if rot and a.rot_ranges:
            cols += [nP + 3 * ci[i] + d for i in todo for d in range(3)]
        tasks = [(j, sg) for j in cols for sg in (+1, -1)]
        # checkpoint: every solved range LP is appended here; a rerun with the same arguments skips them
        os.makedirs(a.out, exist_ok=True)
        ckpt = os.path.join(a.out, "%s_known_rotation_%dcams_%s_eps%g%s.ranges.jsonl"
                            % (run_name(a.scene, a.model), len(cams), a.scheme, a.eps, "_rot" if rot else ""))
        ext, times = {}, []
        # duals checkpoint: with --save-duals an LP counts as done only once its duals are saved too
        dckpt = ckpt[:-len(".ranges.jsonl")] + ".duals.jsonl"
        have_d = set()
        if a.save_duals and os.path.exists(dckpt):
            for line in open(dckpt):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                have_d.add((e["j"], e["sense"]))
        if os.path.exists(ckpt):
            for line in open(ckpt):
                try:
                    j, sg, v, dt = json.loads(line)
                except ValueError:
                    continue
                if v is not None:                        # unsolved LPs are retried
                    ext[(j, sg)] = v
        todo_lps = [tk for tk in tasks if tk not in ext
                    or (a.save_duals and tk not in have_d and np.isfinite(ext[tk]))]   # unbounded: no duals
        t = time.time()
        print("%d range LPs (%d already in %s) on %d process(es)"
              % (len(tasks), len(tasks) - len(todo_lps), os.path.basename(ckpt), a.procs if a.solver != "cuopt" else 1), flush=True)
        f_ck = open(ckpt, "a")
        f_d = open(dckpt, "a") if a.save_duals else None
        # what each column is, so the duals file can be read on its own
        name = {3 * ci[i] + d: ["centre", sc.imgs[i].name, "xyz"[d]] for i in todo for d in range(3)}
        name.update({nC + 3 * pos[k] + d: ["point", int(k), "xyz"[d]] for k in pts for d in range(3)})
        if rot:
            name.update({nP + 3 * ci[i] + d: ["rotation", sc.imgs[i].name, "xyz"[d]] for i in todo for d in range(3)})

        def record(j, sg, v, dt, d=None):
            ext[(j, sg)] = v
            times.append(dt)
            f_ck.write(json.dumps([j, sg, v, dt]) + "\n")
            f_ck.flush()
            if f_d is not None and d is not None:
                f_d.write(json.dumps(dict(j=j, sense=sg, target=name[j], value=v, **d)) + "\n")
                f_d.flush()

        if a.solver != "cuopt" and a.procs > 1 and todo_lps:
            stall = 3 * a.lp_time_limit + 120                # no result for this long: a worker died
            pool = Pool(a.procs, initializer=_init_worker,
                        initargs=(A, b, bounds, a.solver, a.lp_time_limit) + ((row_pt, L) if a.save_duals else ()))
            it = pool.imap_unordered(_extreme, todo_lps)
            try:
                for n_done in range(1, len(todo_lps) + 1):
                    record(*it.next(timeout=stall))
                    if n_done % max(1, len(todo_lps) // 20) == 0:
                        print("  %d / %d range LPs (%.0fs)" % (n_done, len(todo_lps), time.time() - t), flush=True)
                pool.close()
            except PoolTimeout:
                print("  no LP finished for %.0fs (a worker probably died, e.g. out of memory): stopping; "
                      "%d LPs missing; rerun the same command to finish them" % (stall, len(tasks) - len(ext)), flush=True)
                pool.terminate()
            pool.join()
        else:
            if a.solver != "cuopt":
                S = HiGHS(A, b, bounds, a.solver, a.lp_time_limit)
            for j, sg in todo_lps:
                t1 = time.time()
                v = S.extreme(j, sg)
                record(j, sg, v, time.time() - t1, duals(S, j, sg, row_pt, L) if a.save_duals else None)
        f_ck.close()
        if f_d is not None:
            f_d.close()
            out["duals_file"] = os.path.basename(dckpt)

        def rng(j):
            if fixed(j):
                return bounds[j]
            lo_, hi_ = ext.get((j, +1)), ext.get((j, -1))
            return (np.nan if lo_ is None else lo_), (np.nan if hi_ is None else hi_)

        for i in todo:
            lo, hi = map(np.array, zip(*[rng(3 * ci[i] + d) for d in range(3)]))
            C_hat = sc.pose(i)[1]
            r = {"lo": lo.tolist(), "hi": hi.tolist(), "size": (hi - lo).tolist(), "colmap": C_hat.tolist()}
            if i in gt_pose:
                Cg = gt_pose[i][1]
                r["gt"] = Cg.tolist()
                r["gt_inside"] = bool(np.all((Cg >= lo - 1e-9) & (Cg <= hi + 1e-9)))
            out["centre_ranges"][sc.imgs[i].name] = r
            print("  %-14s centre size %s  (COLMAP inside: %s; GT inside: %s)"
                  % (sc.imgs[i].name, np.round(hi - lo, 4), bool(np.all((C_hat >= lo - 1e-9) & (C_hat <= hi + 1e-9))),
                     r.get("gt_inside", "n/a")), flush=True)
        if rot and a.rot_ranges:
            from scipy.spatial.transform import Rotation
            for i in todo:
                lo, hi = map(np.array, zip(*[rng(nP + 3 * ci[i] + d) for d in range(3)]))
                r = {"lo_deg": np.degrees(lo).tolist(), "hi_deg": np.degrees(hi).tolist(),
                     "bound_lo_deg": np.degrees(rot[0][i][0]).tolist(), "bound_hi_deg": np.degrees(rot[0][i][1]).tolist()}
                if i in gt_pose:              # true delta: R_true = R_hat Exp(delta)
                    dg = Rotation.from_matrix(sc.pose(i)[0].T @ gt_pose[i][0]).as_rotvec()
                    r["gt_deg"] = np.degrees(dg).tolist()
                    r["gt_inside"] = bool(np.all((dg >= lo - 1e-9) & (dg <= hi + 1e-9)))
                out["rotation_ranges"][sc.imgs[i].name] = r
            R_ = [r for r in out["rotation_ranges"].values() if "gt_inside" in r]
            print("rotations: true delta inside its range for %d / %d cameras" % (sum(r["gt_inside"] for r in R_), len(R_)))
        for k in pts:
            lo, hi = map(np.array, zip(*[rng(nC + 3 * pos[k] + d) for d in range(3)]))
            X_hat = sc.point(k)
            r = {"lo": lo.tolist(), "hi": hi.tolist(), "size": (hi - lo).tolist(), "colmap": X_hat.tolist(),
                 "views": int(sum(k in sc.det[i] for i in cams))}
            if k in Xgt:
                r["gt"] = Xgt[k].tolist()
                r["gt_inside"] = bool(np.all((Xgt[k] >= lo - 1e-9) & (Xgt[k] <= hi + 1e-9)))
            out["point_ranges"][int(k)] = r
        if pts:
            P = [r for r in out["point_ranges"].values() if np.all(np.isfinite(r["size"]))]
            diag = np.array([np.linalg.norm(r["size"]) for r in P])
            g = [r for r in P if "gt" in r]
            err = np.array([np.linalg.norm(np.subtract(r["colmap"], r["gt"])) for r in g])
            print("points: %d with ranges (%d unsolved); box diagonal median %.4f; COLMAP error median %s; GT inside %d / %d"
                  % (len(P), len(pts) - len(P), np.median(diag) if len(diag) else np.nan,
                     "%.4f" % np.median(err) if len(err) else "n/a", sum(r["gt_inside"] for r in g), len(g)), flush=True)
        out["seconds_per_range_lp"] = float(np.mean(times)) if times else None
        out["range_lps_seconds"] = round(time.time() - t)
        out["unsolved_range_lps"] = int(sum(ext.get(tk) is None for tk in tasks))
    else:
        print("F(eps | R) is empty at these rotations: no ranges (try --scheme mult or a larger --eps)")
    out["seconds"] = round(time.time() - T0)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "%s_known_rotation_%dcams%s.json" % (run_name(a.scene, a.model), len(cams), "_rot" if rot else ""))
    json.dump(out, open(path, "w"), indent=2)
    print("-> %s  (%ds)" % (path, out["seconds"]))


if __name__ == "__main__":
    main()
