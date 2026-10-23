# joint_sets: set-based camera-pose and landmark sets

Computes, for a window of cameras of a COLMAP reconstruction, a **set of camera poses** for each camera and a
**box of 3D positions** for each landmark, such that every detection reprojects within `eps` pixels. Each step
only removes what is impossible, so the true poses and points stay inside if every true detection is within
`eps` of its true projection (checked automatically when ground truth is available).

## Files

| file | what it is |
|---|---|
| `joint_sets.py` | the method (run this) |
| `geom.py` | data loading, undistortion, pixel windows, optional ground truth |
| `colmap_io.py` | reader for COLMAP's text model |
| `run_server.sh` | SLURM job: all windows of one dataset on one CPU node |
| `known_rotation.py` | all cameras + points in one LP with rotations fixed (CPU HiGHS or GPU cuOpt) |
| `run_known_rotation_gpu.sh` | SLURM job: `known_rotation.py` on one GPU with cuOpt |

Requirements: Python 3.9+, `numpy`, `scipy` (`pip install numpy scipy`). `joint_sets.py` is CPU-only; `known_rotation.py` can also use a GPU through cuOpt (see below).

## Dataset layout (airplane as the example)

```
airplane/
  COLMAP results/        cameras.txt, images.txt, points3D.txt   (COLMAP text model, required)
  colmap_log.txt         COLMAP mapper log (optional: gives the registration order)
  transforms_train.json  transforms_val.json + train/*_depth.npz, val/*_depth.npz
                         (optional: ground truth for the checks; NeRF-style poses + z-depth maps)
```

## Run

One window (6 cameras in COLMAP's registration order, 3 iterations):

```
python joint_sets.py --scene /path/to/airplane --window 0 --n-cams 6 --eps 3 --procs 32
```

All windows of a dataset on a server (edit the path and the SLURM account in the script first):

```
sbatch run_server.sh /path/to/airplane
```

Windows overlap by two cameras: window 0 = cameras 1-6 of the registration order, window 1 = cameras 5-10, ...
Each window uses its own first camera as the origin and its second camera's centre x = 1 as the scale.

Useful options: `--eps` (pixels), `--scheme mult` (wider windows where COLMAP's residual is large),
`--iters`, `--w0-deg` / `--c0` / `--x0` (initial rotation, centre and landmark half-widths around COLMAP's
estimate; results are conditional on the truth lying inside them).

## Output (`results/<dataset>_window<w>/`)

* `summary.json` per iteration: for each camera the number of rotation cells, the rotation range (degrees,
  relative to COLMAP's) and the centre range; for the landmarks the median box size; with ground truth,
  whether each true pose lies in its set and how many true points lie in their boxes.
* `pose_cells_<image>.npz`: the pose set of each camera (rotation cells `w_rad`, `h_rad`, centre boxes).
* `landmark_boxes.npz`: the landmark boxes (`lo`, `hi`).

## Known-rotation joint set (`known_rotation.py`): all cameras and points in one LP

Rotations fixed at COLMAP's; every camera centre and every point is an unknown, every detection's +-eps window is a
linear row. One LP gives feasibility; min / max of each centre coordinate gives its exact range (all other cameras
and points free). Results are conditional on the rotations.

```
python known_rotation.py --scene /path/to/airplane --n-cams 30 --scheme mult --solver highs-ipm      # CPU
sbatch run_known_rotation_gpu.sh /path/to/airplane 200 mult                                          # GPU (cuOpt)
```

* Another COLMAP model of the same dataset (e.g. the fixed-intrinsics one written by
  `nerf_synthetic/airplane/run_colmap.sh` to `COLMAP results fixedK/`): pass `--model "/path/to/airplane/COLMAP results fixedK"`.
  Its mapper log next to the model gives the registration order, and the output and checkpoint names get a
  `_COLMAP_results_fixedK` suffix, so earlier results in the same `--out` folder are not reused or overwritten.
* `--scheme fixed` (3 px everywhere) is usually infeasible (some detections are > 3 px off at COLMAP's rotations);
  use `--scheme mult`.
* Solvers: `highs-ipm` (CPU interior point; 30 airplane cameras: 2 s feasibility, ~4.5 s per range LP) or `cuopt`
  ([NVIDIA cuOpt](https://github.com/NVIDIA/cuopt), GPU, Linux / WSL2 only, Python >= 3.12). With cuOpt use
  `--cuopt-method barrier` (default, 1e-8 accurate). `pdlp` is faster on large problems but only ~1e-4 accurate,
  so its ranges are approximate rather than guaranteed.
* cuOpt install: the script runs `pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12`; check the
  package name against the install selector at https://docs.nvidia.com/cuopt/user-guide/latest/install.html
  (`cuopt-cu13` for CUDA 13). The cuOpt backend has not been run yet (no GPU on the development machine).
* Cost: 1 feasibility LP + 6 LPs per camera (+ 6 per point with `--point-ranges`). Output:
  `results/<dataset>_known_rotation_<n>cams.json` (centre ranges, COLMAP / GT inside, timings).

## Notes

* Speed: the work is many small linear programs solved by HiGHS on the CPU; `--procs` parallelises them.
  A GPU does not help without replacing the LP solver, so request CPU cores (32-64) rather than a GPU.
* Cost grows with landmarks x track length x rotation cells; keep windows at 6-10 cameras.
* Current status (first version): the sets are sound (on the airplane window 0 the true poses and 99% of the
  true points stay inside) but stay close to the initial boxes, because landmarks are kept as boxes and updated
  one at a time. Tighter results need the landmarks kept as polytopes and solved jointly with each camera
  centre; that is the next version.
