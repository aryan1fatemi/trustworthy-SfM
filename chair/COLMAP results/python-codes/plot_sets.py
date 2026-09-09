"""Plots from sets_<tag>.csv produced by analyse_sets.py."""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feasible_set import OUTDIR

BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#c05621", "#2f855a", "#718096"


def load(tag):
    with open(os.path.join(OUTDIR, "sets_%s.csv" % tag)) as fh:
        rows = list(csv.DictReader(fh))
    def col(name, rs):
        return np.array([float(r[name]) for r in rs])
    ok = [r for r in rows if r["empty"] == "0" and r.get("diameter")]
    return rows, ok, col


def trend(x, y, bins):
    """Median of y within each x bin; returns (centres, medians, counts)."""
    cx, cy, cn = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 8:
            cx.append(np.sqrt(lo * hi))
            cy.append(np.median(y[m]))
            cn.append(int(m.sum()))
    return np.array(cx), np.array(cy), np.array(cn)


def main(tag="eps1_all"):
    rows, ok, col = load(tag)
    nv_all = np.array([float(r["n_views"]) for r in rows])
    empty = np.array([r["empty"] == "1" for r in rows])

    nv = col("n_views", ok)
    dia = col("diameter", ok)
    vol = col("volume", ok)
    ani = col("anisotropy", ok)
    sdia = col("s_diameter", ok)
    disp = col("max_displacement_from_X_k", ok)
    dk = col("d_k", ok)
    fill = col("fill_ratio", ok)

    vb = np.array([2, 3, 4, 6, 9, 14, 22, 35, 60, 120])
    jit = nv * np.exp(np.random.default_rng(0).normal(0, 0.045, nv.size))
    dot = dict(s=3, lw=0, alpha=0.13)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # 1. diameter vs views
    a = ax[0, 0]
    a.scatter(jit, dia, c=BLUE, **dot)
    cx, cy, _ = trend(nv, dia, vb)
    a.plot(cx, cy, "o-", color="#1a365d", ms=5, lw=2, label="median")
    a.set(xscale="log", yscale="log", xlabel="track length  |views(k)|",
          ylabel="diameter of $\\mathcal{F}_k$   [scene units]",
          title="Diameter vs. number of views")
    a.legend(frameon=False, fontsize=9)

    # 2. volume vs views
    a = ax[0, 1]
    a.scatter(jit, vol, c=ORANGE, **dot)
    cx, cy, _ = trend(nv, vol, vb)
    a.plot(cx, cy, "o-", color="#7b341e", ms=5, lw=2, label="median")
    a.set(xscale="log", yscale="log", xlabel="track length  |views(k)|",
          ylabel="volume of $\\mathcal{F}_k$   [units$^3$]",
          title="Volume vs. number of views")
    a.legend(frameon=False, fontsize=9)

    # 3. anisotropy vs views
    a = ax[0, 2]
    a.scatter(jit, ani, c=GREEN, **dot)
    cx, cy, _ = trend(nv, ani, vb)
    a.plot(cx, cy, "o-", color="#22543d", ms=5, lw=2, label="median")
    a.axhline(1, color=GREY, lw=0.8, ls=":")
    a.set(xscale="log", yscale="log", xlabel="track length  |views(k)|",
          ylabel="anisotropy   $L_1 / L_3$",
          title="Shape vs. number of views\n(1 = ball, large = needle)")
    a.legend(frameon=False, fontsize=9)

    # 4. distribution of normalised spread
    a = ax[1, 0]
    a.hist(np.log10(sdia), bins=70, color=BLUE, alpha=0.85)
    for q, c in ((50, "#1a365d"), (95, ORANGE)):
        x = np.log10(np.percentile(sdia, q))
        a.axvline(x, color=c, lw=1.6,
                  label="p%d = %.1e" % (q, np.percentile(sdia, q)))
    a.set(xlabel="$\\log_{10}$  ( diameter / $d_k$ )", ylabel="landmarks",
          title="Normalised spread  $\\tilde s_k$")
    a.legend(frameon=False, fontsize=9)

    # 5. does COLMAP's reported error predict the 3D extent?  (it does not)
    a = ax[1, 1]
    err = col("colmap_error_px", ok)
    m = err > 1e-3
    a.scatter(err[m], dia[m], c=GREY, s=3, lw=0, alpha=0.13)
    eb = np.logspace(np.log10(err[m].min()), np.log10(err[m].max()), 14)
    cx, cy, _ = trend(err[m], dia[m], eb)
    a.plot(cx, cy, "o-", color="#1a365d", ms=5, lw=2, label="median")
    w = m & (err > 0.95) & (err < 1.05)          # one narrow slice of error
    if w.sum() > 30:
        p5, p95 = np.percentile(dia[w], [5, 95])
        a.vlines(1.0, p5, p95, color=ORANGE, lw=3.5, alpha=0.9)
        a.text(1.15, p95, "%.0f x spread in 3D\nat one error value\n(n = %d)"
               % (p95 / p5, w.sum()), fontsize=8, color=ORANGE, va="top")
    a.set(xscale="log", yscale="log",
          xlabel="COLMAP reported error   [px]   (mean $L_2$)",
          ylabel="diameter   [scene units]",
          title="Reprojection error does not determine\n3D uncertainty")
    a.legend(frameon=False, fontsize=9, loc="lower right")

    # 6. empty fraction + how full the box is
    a = ax[1, 2]
    ex_x, ex_y = [], []
    for lo, hi in zip(vb[:-1], vb[1:]):
        mm = (nv_all >= lo) & (nv_all < hi)
        if mm.sum() >= 8:
            ex_x.append(np.sqrt(lo * hi))
            ex_y.append(100 * empty[mm].mean())   # a fraction, not a median
    a.plot(ex_x, ex_y, "o-", color=ORANGE, ms=5, lw=2,
           label="$\\mathcal{F}_k=\\emptyset$")
    cx2, cy2, _ = trend(nv, fill, vb)
    a.plot(cx2, 100 * cy2, "s-", color=BLUE, ms=5, lw=2,
           label="volume / OBB volume")
    a.set(xscale="log", xlabel="track length  |views(k)|", ylabel="percent",
          title="Empty sets, and how fully $\\mathcal{F}_k$\nfills its bounding box")
    a.legend(frameon=False, fontsize=9)

    for a in ax.ravel():
        a.grid(True, which="both", alpha=0.25)
    fig.suptitle("Feasible sets $\\mathcal{F}_k$ for all %d landmarks, "
                 "$\\epsilon$ = 1 px  (%d non-empty)" % (len(rows), len(ok)),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(OUTDIR, "sets_overview_%s.png" % tag)
    fig.savefig(out, dpi=140)
    print("wrote", out)

    # ---- focused figure: diameter vs views, with quartile band ----------
    fig2, a = plt.subplots(figsize=(7.2, 5.2))
    a.scatter(jit, dia, c=BLUE, s=4, lw=0, alpha=0.13)
    lo, med, hi, cx = [], [], [], []
    for l, h in zip(vb[:-1], vb[1:]):
        m = (nv >= l) & (nv < h)
        if m.sum() >= 8:
            cx.append(np.sqrt(l * h))
            lo.append(np.percentile(dia[m], 25))
            med.append(np.median(dia[m]))
            hi.append(np.percentile(dia[m], 75))
    a.fill_between(cx, lo, hi, color="#1a365d", alpha=0.18,
                   label="inter-quartile range")
    a.plot(cx, med, "o-", color="#1a365d", ms=6, lw=2.2, label="median")
    a.set(xscale="log", yscale="log",
          xlabel="track length   |views($k$)|",
          ylabel="diameter of $\\mathcal{F}_k$   [scene units]",
          title="How far a landmark can move vs. how many views see it\n"
                "($\\epsilon$ = 1 px, %d landmarks)" % len(ok))
    a.grid(True, which="both", alpha=0.25)
    a.legend(frameon=False)
    fig2.tight_layout()
    out2 = os.path.join(OUTDIR, "diameter_vs_views_%s.png" % tag)
    fig2.savefig(out2, dpi=150)
    print("wrote", out2)

    # ---- console summary by track length -------------------------------
    print("\n%8s %7s | %9s %9s %9s | %8s %8s" %
          ("views", "n", "diam p25", "diam med", "diam p75", "aniso", "empty%"))
    print("-" * 70)
    for l, h in zip(vb[:-1], vb[1:]):
        m = (nv >= l) & (nv < h)
        ma = (nv_all >= l) & (nv_all < h)
        if m.sum() < 8:
            continue
        lbl = "%d-%d" % (l, h - 1) if h - 1 > l else "%d" % l
        print("%8s %7d | %9.3e %9.3e %9.3e | %8.1f %7.1f%%" %
              (lbl, int(m.sum()), np.percentile(dia[m], 25), np.median(dia[m]),
               np.percentile(dia[m], 75), np.median(ani[m]),
               100 * empty[ma].mean()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eps1_all")
