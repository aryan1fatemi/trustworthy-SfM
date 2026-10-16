# LEGO dataset log

## Setup
- Blender 5.2.2 LTS installed; runs headless from the terminal
- Separate workspace lego_work/ (existing lego/ and chair/ untouched)
- Model: src/Lego_856_Bulldozer.blend, exported to .obj (+ .mtl)

## Findings
- Existing lego/ has depth only for the test split, as 8-bit RGBA PNGs (values 0-162 across all channels), so it can't be exact ground truth.
- Its camera_angle_x (0.6911) matches render_airplane.py.
- Imported .obj contains ~450 mesh objects (bricks, gears, axles, treads).
- LegoPlane_huge (32 x 0.22 x 57.6) is a floor baseplate and must be removed.
- Plane and Plane.001 are unidentified (large); need to check.
- The model appears to be lying on its side with the axes used for the import (hypothesis, to confirm with diag_obj.py).
- Glass/transparent parts exist; ray-cast depth will treat them as solid.

## Open questions for senior
1. Reuse existing transforms_*.json poses, or new poses like the airplane script?
2. Splits: train/val only, or train/val/test (100/100/200)?
3. OK to lose colors/materials with the .obj route?
4. Transparent parts: depth at the glass surface, or through it?

## Next
- Run diag_obj.py; delete floor/helper planes; fix up-axis
- 2-view test render; validate depth
- diag_obj.py: with import forward=-Z, up=Y the model is upright (floor thin along z, model starts at the floor's top surface). The script's original import axes (Y/Z) lay it on its side.
- LegoPlane_huge (material Sand) = floor; Plane (Fill_Light) and Plane.001 (Light) = light panels ~100 units away. All three to be deleted; otherwise they would shrink the normalized model.
- Bulldozer = 438 meshes, 1,089,885 vertices, ~32 x 57.6 x 27.6 units.

## Changes from prev
- Existing lego/: poses + RGB only. Depth only in test/ as 8-bit RGBA PNG (no train/val depth). transforms_*.json have keys camera_angle_x, frames (file_path, rotation, transform_matrix). No mesh, no float depth.
- lego_gt/ (mine): float32 z-depth (.npz) for train+val, mesh_normalized.obj, documented conventions; poses newly sampled, not the existing ones.
- Awaiting answer: keep new poses, or render exact depth at the existing lego/ poses?