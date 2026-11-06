# LEGO dataset log

## Status (2026-11-06)
- Done: exact ground-truth render, conversion to the chair format (lego_canon/), COLMAP run (200/200 registered).
- Not done: pose accuracy vs ground truth (script ready, not yet run), choice of next experiment, where the dataset should live.

## Decisions from Ariyan
- New poses are needed (new images), so the poses in lego_gt/ are kept.
- No test split. All 200 views in one folder, named 1..200.
- The .obj is not needed in the repo (too large). Keep it outside git or in .gitignore.
- License: he thinks it is fine. The NeRF-synthetic README lists the LEGO model (BlendSwap 11490) as CC-BY-NC: attribution required, no commercial use.

## Setup
- Blender 5.2.2 LTS, headless from the terminal; Cycles on the Apple M4 Pro GPU (Metal), about 3 s per view after the first one.
- Workspace lego_work/ (the existing lego/ and chair/ are untouched).
- Model: src/Lego_856_Bulldozer.blend from BlendSwap, plus an .obj/.mtl export. Neither is tracked in git (.blend 117 MB, .obj 148 MB, above GitHub's 100 MB limit).
- COLMAP 3.11.1 (CPU only, macOS). Ariyan's script comment says 3.9.1.

## Why a new dataset
- Existing lego/ has RGB and poses. Depth exists only for the test split, as 8-bit RGBA PNGs (values 0-162), so it cannot be exact GT.
- No float depth, no mesh, no train/val depth. Its transforms_*.json keys: camera_angle_x, frames (file_path, rotation, transform_matrix).
- Its camera_angle_x (0.6911) matches render_airplane.py.

## Model findings
- The imported .obj has 441 objects. After deleting 3 non-model objects: 438 meshes, 1,089,885 vertices, about 32 x 57.6 x 27.6 units.
- Deleted by name: LegoPlane_huge (material Sand, floor plate), Plane (Fill_Light) and Plane.001 (Light), two light panels about 100 units away. Left in, they would shrink the normalized model.
- Orientation: import with forward=-Z, up=Y gives an upright model. The airplane script's axes (Y/Z) lay it on its side.
- The brick baseplate under the bulldozer is part of the model (Brick_flat_* parts) and stays.
- The .obj export lost the colors: all 11 materials in the .mtl have Kd 0.8 0.8 0.8. Colors are restored from the .blend by material name (9 materials).
- The .blend cannot be loaded directly: 497 meshes, 56 hidden template meshes, modifiers on all (SUBSURF 492, BEVEL 107, EDGE_SPLIT 107, MIRROR 5, ARMATURE 2, CURVE 2), one armature, animation range 1-250. The depth step ray-casts raw meshes and ignores modifiers.
- Evaluated .blend vertices: 1,090,053. The .obj has about 1,090,045 (difference of 8, not explained). So geometry comes from the .obj and colors from the .blend.
- Glass/transparent parts (Red_Glass, transparent) count as solid in the depth maps.

## Render (render_lego.py, from render_airplane.py)
- 800x800, 64 samples, FOV 0.6911 rad, camera radius 4.0311, model normalized to a sphere of radius 1.30 (scale factor in transforms.json), poses on the upper hemisphere.
- Views 1-100 from seed 0, 101-200 from seed 1000 (same generator as render_chair.py).
- Depth: float32 z-depth along the optical axis, 0 = background, exact ray cast through the pixel centre.

## Validation of the render (6 sampled views, not all 200)
- Alpha mask vs depth>0: IoU 0.9990-0.9999.
- Depth on object: 3.097-5.220 (expected about 2.73-5.33).
- Max distance of back-projected points from origin: 1.2870-1.2991 (<= 1.30).
- Colored vs gray render: identical poses and depth on 4 views (compare_gray.py).
- Visual check of lego_canon images 1 and 150: colored bulldozer, upright, white background.

## Conversion to chair format (convert_lego.py -> lego_canon/)
- images/1..200.png composited on white, depth/1..200.npz, transforms.json (per-frame keys same as chair: file_path, depth_path, same_view_as, transform_matrix).
- Checked: 200 images, 200 depth files, frame 101 = "val r_0".
- Top-level metadata differs from the chair: LEGO has source_obj, colours_from, import_axes; the chair has source_blend, objects. To ask Ariyan whether any code reads those.

## COLMAP (run_colmap_lego.sh = chair/run_colmap.sh with only ROOT changed)
- Settings: PINHOLE, fx=fy=1111.111, cx=cy=400, one shared camera, intrinsics not refined, exhaustive matching, CPU.
- Camera in the result stayed PINHOLE 800 800 1111.111 1111.111 400 400.

| | Chair (Ariyan, COLMAP 3.9.1) | LEGO (mine, COLMAP 3.11.1) |
|---|---|---|
| Registered images | 200 | 200 |
| Points | 30,361 | 31,772 |
| Observations | 224,371 | 183,604 |
| Mean track length | 7.39 | 5.78 |
| Observations per image | 1,122 | 918 |
| Mean reprojection error | 0.453 px | 0.549 px |

- LEGO has about 5% more points but about 18% fewer observations, shorter tracks and about 20% higher reprojection error. Cause not determined: scene content and COLMAP version are confounded, and each number is a single run.
- Reprojection error is COLMAP's agreement with itself. It is not pose accuracy.

## Pose accuracy vs ground truth (NOT RUN YET)
- chair/align_translations.py is stale: it reads transforms_train/val.json and images named r_N.png, but the chair now has transforms.json and images named 1..200. The existing translation_comparison.csv uses the old names, so it appears to come from an older run and is not a baseline for these results.
- pose_check.py (in lego_work/) imports its alignment functions (Umeyama Sim(3) fit on camera centres) and works on the new layout. Usage: python pose_check.py lego_canon, then python pose_check.py ../chair. It compares positions only, not rotations.
- Results (2026-11-06, positions only, Sim(3)-aligned, scene units): chair mean 0.0052 / median 0.0046 / RMSE 0.0064 / max 0.0430 (120.png), 0.044% of the camera extent 11.758; LEGO mean 0.0089 / median 0.0076 / RMSE 0.0110 / max 0.0698 (176.png), 0.076%. LEGO error is about 1.7x the chair on every statistic. Cause not determined (scene content vs COLMAP 3.9.1 vs 3.11.1, one run each). Both are very accurate; rotations not compared.

## Open
1. Run pose_check.py on LEGO and chair; compare mean, median, max.
2. Which experiment next (joint_sets/ ?), and whether Bugatti is also mine.
3. Where lego_canon/ and the COLMAP text model should live (not committed; check du -sh first).
4. Whether transparent parts should give depth at the glass or through it.
5. Rotation error (needs the OpenGL to OpenCV flip), as a follow-up.

## Files
- Tracked: scripts (render_lego.py, convert_lego.py, run_colmap_lego.sh, check_output.py, compare_gray.py, pose_check.py, diag_*.py, check_obj.py), logs, NOTES.md.
- Not tracked: lego_gt/, lego_canon/, src/.