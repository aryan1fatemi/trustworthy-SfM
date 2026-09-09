# SLUE feasible sets — chair scene

Per-landmark computation of

    F_k = { X in R^3 : ||pi(theta_i, X) - y_ik||_inf <= eps  for all i in views(k) }

with the cameras held fixed at their COLMAP estimates.

Code lives in `../python-codes/`.

## Files

| file | contents |
|---|---|
| `feasible_sets_eps<E>.csv` | one row per landmark: track length, `d_k`, interval-hull extent, `s_tilde`, volume, anisotropy |
| `feasible_sets_eps<E>.json` | same plus interval hull corners, PCA axes, weak direction `v3`, solver diagnostics |
| `clouds_eps<E>.npz` | the raw feasible grid samples, keyed by POINT3D_ID (N x 3, world coords) |
| `validation_eps1.json` | brute force vs. exact-polytope LP, per landmark |
| `feasible_sets_eps1.png` | each `F_k` in its own PCA frame, true aspect ratio |
| `spread_vs_views_eps1.png` | `s_tilde` vs. track length |
| `sets_eps1_all.csv` | **main table** — all 26 545 landmarks: volume, diameter, OBB sides, anisotropy, fill ratio, normalised spreads (`../python-codes/analyse_sets.py`) |
| `sets_overview_eps1_all.png` | six-panel overview: size/shape vs. track length, spread histogram, error-vs-extent, empty fraction |
| `diameter_vs_views_eps1_all.png` | diameter vs. track length with inter-quartile band |
| `slue_feasible_set_diagram.svg/.png` | labelled theory figure; panel A is generated 3D geometry (`../python-codes/gen_panelA.py`), spliced in by `../python-codes/assemble_diagram.py` |

## Reproducing

    cd "../python-codes"
    python residual_stats.py                       # epsilon selection evidence
    python feasible_set.py --eps 1.0 --n-per-bin 3 # 15 landmarks, ~4 s
    python validate_and_plot.py eps1               # validation + figures
    python assemble_diagram.py                     # rebuild the theory figure

    python analyse_sets.py --eps 1.0 --n 0         # exact LP, all 26545 pts, ~90 s
    python plot_sets.py eps1_all                   # overview + diameter plots

    python feasible_set.py --eps 1.0 --ids 8246 17861 --tag mypick

## Choice of epsilon

`residual_stats.py` reports, over all 26 545 landmarks, the largest per-view
L-inf residual at the COLMAP point (median 0.41 px, 75th pct 1.02 px). `F_k` is
guaranteed non-empty when that value is <= eps, which holds for

    eps = 0.5 -> 56 % of landmarks       eps = 2.0 -> 89 %
    eps = 1.0 -> 75 %                    eps = 3.0 -> 97 %

`eps = 1.0` px is used as the default: it matches the stated detector noise and
keeps three quarters of the reconstruction feasible. Landmarks whose max
residual already exceeds eps are reported as `empty` rather than silently
enlarged — an empty `F_k` means the track contains a mismatch or a pose error,
which is itself a useful signal.

Note that COLMAP's own `ERROR` column is the *mean L2* residual, which is the
wrong statistic here: feasibility is set by the *worst* view under the L-inf
norm.

## Validation (eps = 1 px, 15 landmarks)

- 0 of ~4 million reported-feasible samples violate the exact non-linear constraint.
- Grid interval-hull extents are 98.0–99.4 % of the exact LP values (a grid can
  only under-cover), and grid volumes match the exact polytope volumes to 0.3 %.
- Our reprojection reproduces COLMAP's `ERROR` column to 3.6e-13 px, confirming
  the SIMPLE_RADIAL intrinsics and the world->camera pose convention.

## Result at a glance

Scene units: cameras sit ~4.0–4.8 units from the object.

| track length | max interval extent | `s_tilde` | anisotropy |
|---|---|---|---|
| 2–3 views  | 0.04 – 0.15 | 1.0e-2 – 3.4e-2 | 8 – 20 |
| 6–9 views  | 0.007 – 0.012 | 1.9e-3 – 2.7e-3 | 1.7 – 3 |
| 14–40 views | 0.009 – 0.014 | 1.9e-3 – 2.9e-3 | 1.7 – 3 |

Two-view landmarks are needle-shaped and roughly 15x more uncertain along `v3`
than well-tracked ones; past ~6 views the set is nearly isotropic and stops
shrinking, because the residual budget rather than the geometry becomes binding.
Extents scale linearly in eps and volumes as eps^3 (verified at eps = 0.5, 1, 2).
