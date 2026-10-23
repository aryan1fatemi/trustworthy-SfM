"""Known-rotation joint set: every rotation fixed at COLMAP's, all camera centres and all points solved together.

    python known_rotation.py --scene /path/to/airplane --n-cams 30 --solver highs-ipm
    python known_rotation.py --scene /path/to/airplane --n-cams 200 --solver cuopt      (GPU, Linux / WSL2)

With R_i fixed, every window condition pi((R_i, C_i), X_k) in W_ik (with depth > z_min) is linear in (C_i, X_k):
    n^T R_i (X_k - C_i) <= 0   for the 4 faces n of the window's pyramid,     r_i3^T (X_k - C_i) >= z_min,
so the joint set F(eps | R) = {(C_1..C_N, X_1..X_M) : all rows hold} is one polytope (one LP, no linearisation).
Gauge: camera 1 at the origin, x of camera 2's centre = 1 (rotations are fixed, so the rotation gauge is too).
Steps:
  1. feasibility: min s  s.t. every row holds up to s >= 0  (F(eps | R) non-empty iff s* = 0);
  2. ranges: min / max of each centre coordinate over F(eps | R) (the exact projection; other cameras and all
     points free), optionally the same for --point-ranges points.
The sets are conditional on the rotations being COLMAP's (the true rotations differ slightly).
Writes <out>/<dataset>_known_rotation_<n>cams.json.
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
def build(sc, cams, eps, scheme, zmin):
    """rows A x <= b over x = (C of cams[1:], X of every landmark seen >= 2 times)."""
    cnt = {}
    for i in cams:
        for k in sc.det[i]:
            cnt[k] = cnt.get(k, 0) + 1
    L = sorted(k for k, c in cnt.items() if c >= 2)
    pi = {k: m for m, k in enumerate(L)}
    ci = {i: m for m, i in enumerate(cams[1:])}
    nC = 3 * len(ci)
    ri, cj, val, b = [], [], [], []
    row = 0
    for i in cams:
        R = sc.pose(i)[0]
        for k, xs in sc.det[i].items():
            if k not in pi:
                continue
            for xn, e in zip(xs, sc.eps_list(i, k, eps, scheme)):
                G = [n @ R for n in box_normals(xn, sc.f[i], e)] + [-R[2]]
                for g, rhs in zip(G, (0, 0, 0, 0, -zmin)):
                    for d in range(3):
                        ri.append(row); cj.append(nC + 3 * pi[k] + d); val.append(g[d])
                        if i in ci:
                            ri.append(row); cj.append(3 * ci[i] + d); val.append(-g[d])
                    b.append(rhs)
                    row += 1
    nv = nC + 3 * len(L)
    A = coo_matrix((val, (ri, cj)), shape=(row, nv)).tocsr()
    bounds = [(None, None)] * nv
    bounds[3 * ci[cams[1]]] = (1.0, 1.0)                 # scale gauge
    return A, np.array(b), bounds, L, ci, nC


# ---------------------------------------------------------------------------- solvers
class HiGHS:
    def __init__(self, A, b, bounds, method, time_limit=None):
        from scipy.optimize import linprog
        self.lp, self.A, self.b, self.bounds, self.method = linprog, A, b, bounds, method
        self.opt = {"time_limit": float(time_limit)} if time_limit else {}

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
        if r.status == 3:
            return -sense * np.inf                                      # unbounded: min = -inf / max = +inf
        return sense * r.fun if r.status == 0 else None                 # None: time limit


_W = {}


def _init_worker(A, b, bounds, method, time_limit):
    _W["S"] = HiGHS(A, b, bounds, method, time_limit)


def _extreme(task):
    j, sense = task
    t = time.time()
    return j, sense, _W["S"].extreme(j, sense), time.time() - t


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
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    T0 = time.time()
    model = a.model or os.path.join(a.scene, "COLMAP results")
    _, imgs_all, _ = read_model(model)
    cams = registration_order(a.scene, imgs_all, a.n_cams, model)
    sc = Scene(a.scene, model, cams)
    zmin = 1e-3
    A, b, bounds, L, ci, nC = build(sc, cams, a.eps, a.scheme, zmin)
    print("%d cameras, %d points, %d rows, %d unknowns, %d nonzeros (built in %.0fs)"
          % (len(cams), len(L), A.shape[0], A.shape[1], A.nnz, time.time() - T0), flush=True)
    t = time.time()
    if a.solver == "cuopt":
        S = CuOpt(A, b, bounds, a.cuopt_method, a.tol)
        print("cuOpt model built (%.0fs)" % (time.time() - t), flush=True)
    else:
        S = HiGHS(A, b, bounds, a.solver)
    # COLMAP's own solution: largest row violation (it is a candidate point of F(eps | R))
    x_hat = np.concatenate([sc.pose(i)[1] for i in cams[1:]] + [sc.point(k) for k in L])
    viol_hat = float(np.max(A @ x_hat - b))
    t = time.time()
    s, msg = S.feasibility()
    t_feas = time.time() - t
    print("feasibility: min violation %s (%s), %.1fs; COLMAP's solution violates by %.2e"
          % ("%.2e" % s if s is not None else "n/a", msg, t_feas, viol_hat), flush=True)
    out = {"scene": a.scene, "cameras": [sc.imgs[i].name for i in cams], "eps_px": a.eps, "scheme": a.scheme,
           "solver": a.solver + ("/" + a.cuopt_method if a.solver == "cuopt" else ""), "points": len(L),
           "rows": int(A.shape[0]), "unknowns": int(A.shape[1]), "min_violation": s, "colmap_violation": viol_hat,
           "feasibility_seconds": round(t_feas, 1), "centre_ranges": {}, "point_ranges": {}}
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
        cols = [3 * ci[i] + d for i in todo for d in range(3) if bounds[3 * ci[i] + d] != (1.0, 1.0)]
        cols += [nC + 3 * pos[k] + d for k in pts for d in range(3)]
        tasks = [(j, sg) for j in cols for sg in (+1, -1)]
        # checkpoint: every solved range LP is appended here; a rerun with the same arguments skips them
        os.makedirs(a.out, exist_ok=True)
        ckpt = os.path.join(a.out, "%s_known_rotation_%dcams_%s_eps%g.ranges.jsonl"
                            % (run_name(a.scene, a.model), len(cams), a.scheme, a.eps))
        ext, times = {}, []
        if os.path.exists(ckpt):
            for line in open(ckpt):
                try:
                    j, sg, v, dt = json.loads(line)
                except ValueError:
                    continue
                if v is not None:                        # unsolved LPs are retried
                    ext[(j, sg)] = v
        todo_lps = [tk for tk in tasks if tk not in ext]
        t = time.time()
        print("%d range LPs (%d already in %s) on %d process(es)"
              % (len(tasks), len(tasks) - len(todo_lps), os.path.basename(ckpt), a.procs if a.solver != "cuopt" else 1), flush=True)
        f_ck = open(ckpt, "a")

        def record(j, sg, v, dt):
            ext[(j, sg)] = v
            times.append(dt)
            f_ck.write(json.dumps([j, sg, v, dt]) + "\n")
            f_ck.flush()

        if a.solver != "cuopt" and a.procs > 1 and todo_lps:
            stall = 3 * a.lp_time_limit + 120                # no result for this long: a worker died
            pool = Pool(a.procs, initializer=_init_worker, initargs=(A, b, bounds, a.solver, a.lp_time_limit))
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
                record(j, sg, S.extreme(j, sg), time.time() - t1)
        f_ck.close()

        def rng(j):
            if bounds[j] == (1.0, 1.0):
                return 1.0, 1.0
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
    path = os.path.join(a.out, "%s_known_rotation_%dcams.json" % (run_name(a.scene, a.model), len(cams)))
    json.dump(out, open(path, "w"), indent=2)
    print("-> %s  (%ds)" % (path, out["seconds"]))


if __name__ == "__main__":
    main()
