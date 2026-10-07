"""Figures for cp_normalized.py.   python cp_normalized_figs.py --scene airplane"""
import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model

COL = {"plain": "#718096", "scale": "#2b6cb0", "tensor": "#c05621", "scale+aniso": "#2f855a"}
BINS = ["2", "3-5", "6-10", "11-20", ">20"]


def rd(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    a = ap.parse_args()
    scene = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene))
    OUT = os.path.join(scene, "COLMAP results", "conformal", "normalized")
    cov = rd(os.path.join(OUT, "nm_coverage.csv"))
    eff = rd(os.path.join(OUT, "nm_efficiency.csv"))
    obs = rd(os.path.join(OUT, "nm_observations.csv"))
    V = [r["variant"] for r in cov if r["alpha"] == "0.05"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    # 1. coverage
    a_ = ax[0, 0]
    x = np.arange(len(V))
    for k, (al, mk) in enumerate(((0.05, "o"), (0.10, "s"), (0.20, "^"))):
        rr = [r for r in cov if float(r["alpha"]) == al]
        m = np.array([float(r["coverage_mean"]) for r in rr])
        lo = np.array([float(r["coverage_min"]) for r in rr])
        hi = np.array([float(r["coverage_max"]) for r in rr])
        a_.errorbar(x + 0.12 * (k - 1), m, yerr=[m - lo, hi - m], fmt=mk, color="#1a202c",
                    capsize=3, label="alpha = %.2f" % al)
        a_.hlines(1 - al, -0.5, len(V) - 0.5, color="#a0aec0", lw=0.8, ls="--")
    a_.set_xticks(x, V)
    a_.set(ylabel="landmark coverage  P($X^*_k\\in\\mathcal{F}_k$)",
           title="Coverage is nominal for every shape\n(mean, range over 20 seeds; dashed = 1-alpha)",
           ylim=(0.75, 1.0))
    a_.legend(frameon=False, fontsize=8, loc="lower right")

    # 2. set size
    a_ = ax[0, 1]
    w = 0.38
    for k, al in enumerate((0.05, 0.10)):
        rr = [r for r in eff if float(r["alpha"]) == al]
        vals = [float(r["median_diam_over_dk"]) for r in rr]
        bars = a_.bar(x + w * (k - 0.5), vals, w, color=[COL[r["variant"]] for r in rr],
                      alpha=1.0 if al == 0.10 else 0.55, label="alpha = %.2f" % al)
        for b, v in zip(bars, vals):
            a_.text(b.get_x() + b.get_width() / 2, v, "%.3f" % v, ha="center",
                    va="bottom", fontsize=7)
    a_.set_xticks(x, V)
    a_.set(ylabel="median diameter($\\mathcal{F}_k$) / $d_k$",
           title="Set size at the calibrated $\\lambda$ (seed 0, test landmarks)\n"
                 "same coverage -- smaller is better")
    a_.legend(frameon=False, fontsize=8)

    # 3. set size by track length
    a_ = ax[1, 0]
    rr = [r for r in eff if float(r["alpha"]) == 0.10]
    for r in rr:
        a_.plot(BINS, [float(r["median_diam_nk_" + b]) for b in BINS], "o-",
                color=COL[r["variant"]], label=r["variant"])
    a_.set(yscale="log", xlabel="$n_k$ (distinct images)", ylabel="median diameter / $d_k$",
           title="Set size by track length   (alpha = 0.10)")
    a_.legend(frameon=False, fontsize=8)

    # 4. noise model vs reality
    a_ = ax[1, 1]
    pred = np.array([float(r["noise_model_std_weak_axis"]) for r in obs])
    meas = np.array([float(r["abs_r_along_weak_axis"]) for r in obs])
    m = (pred > 0) & (meas > 0) & np.isfinite(pred)
    sub = np.random.default_rng(0).choice(np.flatnonzero(m), min(15000, m.sum()), replace=False)
    a_.scatter(pred[sub], meas[sub], s=2, lw=0, alpha=0.2, color="#2b6cb0")
    lim = [min(pred[m].min(), meas[m].min()), max(pred[m].max(), meas[m].max())]
    a_.plot(lim, lim, color="#1a202c", lw=1, label="measured = predicted")
    a_.plot(lim, [2.0 * v for v in lim], color="#a0aec0", lw=0.8, ls="--")
    rat = np.median(meas[m] / pred[m])
    a_.set(xscale="log", yscale="log",
           xlabel="noise-model std along weak axis  $\\sigma/\\sqrt{4\\pi\\sigma_I^2\\lambda_2}$  [px]",
           ylabel="measured |error| along weak axis  [px]",
           title="Forstner noise model vs. measured error\nmedian measured / predicted = %.0f x" % rat)
    a_.legend(frameon=False, fontsize=8, loc="upper left")
    for a_ in ax.ravel():
        a_.grid(alpha=0.25, which="both")
    fig.suptitle("Normalized landmark-level conformal calibration -- %s" % a.scene, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "nm_summary.png"), dpi=140)

    # ---- example regions on one image -----------------------------------
    cams, imgs, pts = read_model(os.path.join(scene, "COLMAP results"))
    by_img = {}
    for r in obs:
        by_img.setdefault(int(r["image_id"]), []).append(r)
    iid = max(by_img, key=lambda i: len(by_img[i]))
    im = imgs[iid]
    rows = by_img[iid]
    # pick keypoints spanning isotropic -> strongly anisotropic
    an = np.array([float(r["lambda1"]) / max(float(r["lambda2"]), 1e-12) for r in rows])
    pick = [rows[i] for i in np.argsort(an)[np.linspace(0, len(rows) - 1, 6).astype(int)]]
    lam = {r["variant"]: float(r["lambda_seed0"]) for r in eff if float(r["alpha"]) == 0.10}
    img = Image.open(os.path.join(scene, "images", im.name)).convert("RGB")
    fig, axes = plt.subplots(len(pick), 3, figsize=(8.4, 2.9 * len(pick)))
    import cp_normalized as NM
    for row_ax, r in zip(axes, pick):
        j = int(r["point2D_idx"])
        y = im.xys[j]
        s = float(r["scale_px"])
        l1, l2 = float(r["lambda1"]), float(r["lambda2"])
        for a_, v in zip(row_ax, ("plain", "scale", "scale+aniso")):
            # rebuild A from the stored eigenvalues is not possible (no eigenvectors), so
            # recompute M locally for this keypoint
            M, _, _ = NM.structure_tensors(os.path.join(scene, "images", im.name),
                                           y[None], np.array([s]))
            reg = 0.0 if v == "plain" else 1e-9
            A = NM.shape_matrix(v, s, M[0], max(reg, 1e-9 * np.trace(M[0])))
            corners = y + lam[v] * (A @ np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]).T).T
            h = max(np.ptp(corners, 0).max() * 0.7, 8)
            a_.imshow(np.asarray(img), extent=(0, img.size[0], img.size[1], 0),
                      interpolation="nearest")
            a_.add_patch(Polygon(corners, closed=True, fill=False, ec=COL[v], lw=1.8))
            a_.plot(*y, "o", ms=3, color="#c05621")
            a_.set(xlim=(y[0] - h, y[0] + h), ylim=(y[1] + h, y[1] - h))
            a_.set_xticks([]); a_.set_yticks([])
            a_.set_title("%s   (s=%.1f px, anis=%.0f)" % (v, s, l1 / max(l2, 1e-12)), fontsize=8)
    fig.suptitle("Tolerance regions $\\mathcal{X}_{ik}$ at the calibrated $\\lambda$ (alpha = 0.10)\n"
                 "%s, keypoints from isotropic (top) to anisotropic (bottom)" % im.name, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "nm_regions.png"), dpi=140)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
