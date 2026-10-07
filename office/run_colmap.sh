#!/usr/bin/env bash
# COLMAP reconstruction of the rendered office scene. Same settings as chair/airplane:
# single_camera=1 (all 200 renders share one camera). Images are already plain
# RGB with the studio backdrop as background, so no compositing step. Run in WSL:
#     wsl -e bash /mnt/c/Users/fatem/COLMAP/nerf_synthetic/office/run_colmap.sh
# CPU-only COLMAP 3.9.1 build, so SIFT runs on the CPU.
set -euo pipefail
ROOT=/mnt/c/Users/fatem/COLMAP/nerf_synthetic/office
DB=$ROOT/database.db
OUT=$ROOT/sparse
TXT="$ROOT/COLMAP results"

rm -f "$DB"; rm -rf "$OUT"; mkdir -p "$OUT" "$TXT"

colmap feature_extractor \
  --database_path "$DB" --image_path "$ROOT/images" \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.max_image_size 3200 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.estimate_affine_shape 0 \
  --SiftExtraction.domain_size_pooling 0

colmap exhaustive_matcher \
  --database_path "$DB" \
  --SiftMatching.use_gpu 0 \
  --SiftMatching.cross_check 1 \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.max_distance 0.7 \
  --SiftMatching.guided_matching 0

colmap mapper \
  --database_path "$DB" --image_path "$ROOT/images" --output_path "$OUT" \
  --Mapper.multiple_models 1 \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 \
  --Mapper.min_num_matches 15 \
  --Mapper.init_min_num_inliers 100 \
  --Mapper.init_min_tri_angle 16 \
  --Mapper.abs_pose_min_num_inliers 30 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.tri_ignore_two_view_tracks 1

# report every model, export the largest to text
best=""; bestn=0
for m in "$OUT"/*/; do
  n=$(colmap model_analyzer --path "$m" 2>&1 | grep -i "Registered images" | grep -o '[0-9]*' | tail -1)
  echo "model $m : ${n:-0} registered images"
  if [ "${n:-0}" -gt "$bestn" ]; then bestn=$n; best=$m; fi
done
echo "exporting $best ($bestn images) -> $TXT"
colmap model_converter --input_path "$best" --output_path "$TXT" --output_type TXT
colmap model_analyzer --path "$best"
