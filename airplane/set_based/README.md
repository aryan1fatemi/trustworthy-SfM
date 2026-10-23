# Set-based triangulation and back-projection (airplane)

Self-contained scripts for the two latest steps. They read the COLMAP model in
`../COLMAP results/` and write everything to `results/`.

| file | what it does |
|---|---|
| `colmap_io.py` | reader for COLMAP's text model (cameras, images, points3D) |
| `setlib.py` | shared geometry: undistortion, pixel windows as linear inequalities, gauge, LP helpers, ground-truth poses |
| `triangulate_pair.py` | **Step 1.** Set-based triangulation from the initial pair (#162 r_82, #15 r_15, poses held at COLMAP's). Each landmark seen in both images gets the polytope `F_k = {X : G_k X <= h_k}` of positions within eps pixels in both images. |
| `backproject_image3.py` | **Step 2.** Back-projects image 3's eps windows (#17 r_16_val) onto the sets `F_k`: samples exactly feasible camera-3 poses (fixed-rotation LP, radial search), and for each pose cuts every set with the back-projected window (the slice). |

## Run

```
python triangulate_pair.py --eps 3
python backproject_image3.py --eps 3
```

## Results (`results/`)

| file | content |
|---|---|
| `pair_sets.npz` | the polytopes: stacked rows `G`, `h`, `row_landmark`; boxes `lo`, `hi`; Chebyshev `centre`; `colmap_point` |
| `pair_sets.csv` | one row per landmark: box size, box diagonal, long-axis length, Chebyshev radius, COLMAP point inside |
| `pair_summary.json` | counts and medians for step 1 |
| `image3_poses.npz` | the exactly feasible camera-3 poses: `R`, `C`, rotation offset from COLMAP `w_rel_colmap`, `angle_deg` |
| `image3_slices.csv` | one row per landmark: set length, slice at COLMAP's pose, median slice length, union of slices, whether all slices share a point |
| `image3_summary.json` | pose-set extent (inner approximation), ground-truth comparison, slice statistics |
| `image3_slices.png` | one landmark: the slice for each feasible pose along the set's long axis |
| `image3_slice_sizes.png` | all landmarks: slice length / set length, union / set length |

## Conventions

* gauge: camera 1 at the origin with `R = I`, x-coordinate of camera 2's centre = 1 (lengths in baseline units)
* eps lives in the undistorted image; windows are written as four faces `n^T R (X - C) <= 0` plus depth
* the pose set from step 2 is an **inner** approximation: every listed pose satisfies all observations exactly
