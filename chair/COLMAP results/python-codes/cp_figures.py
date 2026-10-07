"""Step 9 strata table + the five figures, from cp_conformal.py outputs.

    python cp_figures.py --scene airplane
"""
import os
import sys
import csv
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cp_conformal import wilson, ALPHAS, SEEDS

C_TOT, C_DET, C_B, GREY = "#2b6cb0", "#c05621", "#2f855a", "#718096"
NK_BINS = [(2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10 ** 6, ">20")]


def rcsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="airplane")
    a = ap.parse_args()
    OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", a.scene, "COLMAP results",
                                       "conformal"))
    T = np.load(os.path.join(OUT, "thresholds.npz"))
    pids = T["pid"]
    idx = {int(p): i for i, p in enumerate(pids)}
    n_k, m_k, rmax = T["n_k"], T["m_k"], T["r_max"]
    es, cont, flag, has = T["eps_star"], T["cont"], T["flag"].astype(bool), T["has_ref"]

    cal = rcsv(os.path.join(OUT, "cp_calibration.csv"))
    cov = rcsv(os.path.join(OUT, "cp_coverage.csv"))
    scores = rcsv(os.path.join(OUT, "cp_scores.csv"))
    lmrows = rcsv(os.path.join(OUT, "cp_landmarks.csv"))

    def eps_of(sc, al, scheme, seed, subset="all"):
        if scheme == "naive":
            r = [x for x in cal if x["score"] == sc and float(x["alpha"]) == al
                 and x["scheme"] == "naive" and int(x["seed"]) == seed and x["subset"] == subset]
            return float(r[0]["eps"])
        return {int(x["m_k"]): float(x["eps"]) for x in cal
                if x["score"] == sc and float(x["alpha"]) == al and x["scheme"] == "bonferroni"
                and int(x["seed"]) == seed and x["subset"] == subset}

    # per seed test landmarks + per-landmark eps -> contains / empty
    def per_seed(sc, al, scheme, subset="all"):
        out = []
        for seed in SEEDS:
            perm = np.random.default_rng(seed).permutation(pids)
            calset = set(perm[: len(perm) // 2].tolist())
            test = np.array([(int(p) not in calset) and has[i] and (subset == "all" or not flag[i])
                             for i, p in enumerate(pids)])
            e = eps_of(sc, al, scheme, seed, subset)
            ek = np.full(len(pids), e) if scheme == "naive" else \
                np.array([e[int(m)] for m in m_k])
            out.append((test, ek >= cont, ek < es))
        return out

    # ---------------- Step 9 strata table ----------------
    rq = np.quantile(rmax[has.astype(bool)], [0.2, 0.4, 0.6, 0.8])
    qbin = np.searchsorted(rq, rmax)
    strata = [("n_k", lab, (n_k >= lo) & (n_k <= hi)) for lo, hi, lab in NK_BINS]
    strata += [("r_max_quintile", "Q%d [%.2f-%.2f px]" %
                (q + 1, ([0] + list(rq))[q], (list(rq) + [rmax.max()])[q]), qbin == q)
               for q in range(5)]
    strata += [("track", "unflagged", ~flag), ("track", "flagged", flag)]

    dia = {}
    for r in lmrows:
        if r["diam_over_dk"]:
            dia.setdefault((r["score"], float(r["alpha"]), r["scheme"]), {})[int(r["point3D_id"])] \
                = float(r["diam_over_dk"])

    srows = []
    for sc in ("tot", "det"):
        for al in (0.05, 0.10):
            for scheme in ("naive", "bonferroni"):
                runs = per_seed(sc, al, scheme)
                for kind, lab, mask in strata:
                    covs, emps, ns = [], [], []
                    for test, contains, empty in runs:
                        sel = test & mask
                        if sel.sum() == 0:
                            continue
                        covs.append(contains[sel].mean())
                        emps.append(empty[sel].mean())
                        ns.append(int(sel.sum()))
                    d0 = dia.get((sc, al, scheme), {})
                    dd = [d0[int(p)] for p in pids[mask] if int(p) in d0]
                    test0 = runs[0][0] & mask
                    k0 = int(runs[0][1][test0].sum())
                    _, lo, hi = wilson(k0, int(test0.sum())) if test0.sum() else (0, np.nan, np.nan)
                    srows.append([sc, al, scheme, kind, lab,
                                  np.mean(ns) if ns else 0,
                                  np.mean(covs) if covs else np.nan,
                                  np.min(covs) if covs else np.nan,
                                  np.max(covs) if covs else np.nan,
                                  lo, hi,
                                  np.mean(emps) if emps else np.nan,
                                  np.median(dd) if dd else np.nan,
                                  float(np.mean(m_k[mask])) if mask.any() else np.nan])
    with open(os.path.join(OUT, "cp_strata.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "alpha", "scheme", "stratum", "label", "mean_n_test",
                    "coverage_mean", "coverage_min", "coverage_max",
                    "wilson_lo_seed0", "wilson_hi_seed0", "empty_rate_mean",
                    "median_diam_over_dk_seed0", "mean_m_k"])
        w.writerows(srows)

    # ---------------- Figure 1: score histograms ----------------
    st = np.array([float(r["s_tot"]) for r in scores if r["s_tot"] != "nan"])
    sd = np.array([float(r["s_det"]) for r in scores if r["s_det"] != "nan"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    bins = np.logspace(np.log10(max(1e-3, min(st.min(), sd.min()))),
                       np.log10(max(st.max(), sd.max())), 70)
    for a_, s_, col, lab, sc in ((ax[0], st, C_TOT, "$s^{tot}$", "tot"),
                                (ax[1], sd, C_DET, "$s^{det}$", "det")):
        a_.hist(s_, bins=bins, color=col, alpha=0.8)
        for al, ls in ((0.10, "--"), (0.05, "-")):
            e = eps_of(sc, al, "naive", 0)
            a_.axvline(e, color="k", ls=ls, lw=1.4,
                       label="$\\epsilon(\\alpha=%.2f)$ = %.2f px" % (al, e))
        a_.set(xscale="log", xlabel=lab + "  [px, $\\|\\cdot\\|_\\infty$]",
               title="%s   (n = %d, median %.2f px)" % (lab, len(s_), np.median(s_)))
        a_.legend(frameon=False, fontsize=9)
        a_.grid(alpha=0.25)
    ax[0].set_ylabel("observations")
    fig.suptitle("Conformity scores with calibrated $\\epsilon$ (seed 0, naive)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_score_histograms.png"), dpi=150)

    # ---------------- Figure 2: calibration plot ----------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    nom = np.array([1 - al for al in ALPHAS])
    for a_, sc in ((ax[0], "tot"), (ax[1], "det")):
        a_.plot([0.75, 1], [0.75, 1], color=GREY, lw=1, ls=":")
        for level, scheme, col, mk, lab in (("observation", "naive", C_TOT if sc == "tot" else C_DET, "o", "per observation"),
                                            ("landmark", "naive", "#1a202c", "s", "per landmark, naive"),
                                            ("landmark", "bonferroni", C_B, "^", "per landmark, Bonferroni")):
            m, lo, hi = [], [], []
            for al in ALPHAS:
                v = [float(r["coverage"]) for r in cov if r["score"] == sc and r["level"] == level
                     and r["scheme"] == scheme and float(r["alpha"]) == al and r["subset"] == "all"]
                m.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
            m, lo, hi = map(np.array, (m, lo, hi))
            a_.errorbar(nom, m, yerr=[m - lo, hi - m], fmt=mk + "-", color=col, capsize=3,
                        label=lab)
        a_.set(xlabel="nominal  $1-\\alpha$", title="score $s^{%s}$" % sc,
               xlim=(0.77, 1.0))
        a_.grid(alpha=0.25)
        a_.legend(frameon=False, fontsize=9, loc="lower right")
    ax[0].set_ylabel("empirical coverage (mean, range over 20 seeds)")
    fig.suptitle("Calibration: empirical vs nominal coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_calibration.png"), dpi=150)

    # ---------------- Figure 3: landmark coverage vs n_k ----------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for a_, al in ((ax[0], 0.10), (ax[1], 0.05)):
        x = np.arange(len(NK_BINS))
        for scheme, col, mk in (("naive", "#1a202c", "s"), ("bonferroni", C_B, "^")):
            runs = per_seed("tot", al, scheme)
            m, lo, hi = [], [], []
            for lo_, hi_, _ in NK_BINS:
                mask = (n_k >= lo_) & (n_k <= hi_)
                c = [cn[t & mask].mean() for t, cn, _ in runs if (t & mask).any()]
                m.append(np.mean(c)); lo.append(np.min(c)); hi.append(np.max(c))
            m, lo, hi = map(np.array, (m, lo, hi))
            a_.errorbar(x, m, yerr=[m - lo, hi - m], fmt=mk + "-", color=col, capsize=3,
                        label="empirical, " + scheme)
        mm = np.array([m_k[(n_k >= lo_) & (n_k <= hi_)].mean() for lo_, hi_, _ in NK_BINS])
        a_.plot(x, np.maximum(0, 1 - mm * al), ":", color=GREY, label="$1-m_k\\alpha$ (union bound)")
        a_.plot(x, (1 - al) ** mm, "--", color=GREY, label="$(1-\\alpha)^{m_k}$ (independent)")
        a_.plot(x, np.full(len(x), 1 - al), "-", color=GREY, lw=0.8, label="$1-\\alpha$")
        a_.set_xticks(x, [b[2] for b in NK_BINS])
        a_.set(xlabel="$n_k$ (distinct images)", title="$s^{tot}$,  $\\alpha$ = %.2f" % al,
               ylim=(-0.02, 1.02))
        a_.grid(alpha=0.25)
        a_.legend(frameon=False, fontsize=8, loc="lower left")
    ax[0].set_ylabel("per-landmark coverage  $X^*_k\\in\\mathcal{F}_k$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_coverage_vs_nk.png"), dpi=150)

    # ---------------- Figure 4: coverage vs r_max quintile ----------------
    fig, a_ = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(5)
    for scheme, col, mk in (("naive", "#1a202c", "s"), ("bonferroni", C_B, "^")):
        runs = per_seed("tot", 0.10, scheme)
        m = [np.mean([cn[t & (qbin == q)].mean() for t, cn, _ in runs]) for q in range(5)]
        e_ = [np.mean([em[t & (qbin == q)].mean() for t, _, em in runs]) for q in range(5)]
        a_.plot(x, m, mk + "-", color=col, label="coverage, " + scheme)
        a_.plot(x, e_, mk + ":", color=col, alpha=0.6, label="empty rate, " + scheme)
    a_.set_xticks(x, ["Q%d\n%.2f-%.2f" % (q + 1, ([0] + list(rq))[q], (list(rq) + [rmax.max()])[q])
                      for q in range(5)])
    a_.axhline(0.9, color=GREY, lw=0.8)
    a_.set(xlabel="worst-view residual $r_k^{max}$ quintile [px]", ylabel="fraction of test landmarks",
           title="Per-landmark coverage vs. $r_k^{max}$   ($s^{tot}$, $\\alpha$ = 0.10)",
           ylim=(-0.02, 1.02))
    a_.grid(alpha=0.25)
    a_.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_coverage_vs_rmax.png"), dpi=150)

    # ---------------- Figure 5: emptiness vs eps ----------------
    fig, a_ = plt.subplots(figsize=(7.5, 4.8))
    grid = np.logspace(-1.5, 1.2, 300)
    a_.plot(grid, [(es > g).mean() for g in grid], color="#1a202c", lw=2,
            label="all landmarks")
    for lo_, hi_, lab in NK_BINS:
        mask = (n_k >= lo_) & (n_k <= hi_)
        a_.plot(grid, [(es[mask] > g).mean() for g in grid], lw=1, alpha=0.7,
                label="$n_k$ " + lab)
    for sc, col in (("tot", C_TOT), ("det", C_DET)):
        for al, ls in ((0.10, "--"), (0.05, "-")):
            a_.axvline(eps_of(sc, al, "naive", 0), color=col, ls=ls, lw=1.2,
                       label="$\\epsilon^{%s}(%.2f)$" % (sc, al))
    a_.set(xscale="log", xlabel="$\\epsilon$ [px]", ylabel="fraction with $\\mathcal{F}_k=\\emptyset$",
           title="Emptiness rate vs. $\\epsilon$")
    a_.grid(alpha=0.25, which="both")
    a_.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_emptiness_vs_eps.png"), dpi=150)
    print("figures + cp_strata.csv ->", OUT)


if __name__ == "__main__":
    main()
