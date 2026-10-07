#!/usr/bin/env bash
# SIFT extraction + exhaustive matching (with two-view geometric verification) only; no mapper.
# Same settings as airplane/run_colmap.sh. Writes <dataset>/database_features.db.
#     wsl -e bash /mnt/c/Users/fatem/COLMAP/nerf_synthetic/run_features_only.sh airplane_noise_low
set -euo pipefail
ROOT=/mnt/c/Users/fatem/COLMAP/nerf_synthetic/$1
DB=$ROOT/database_features.db
rm -f "$DB"
start=$(date +%s)
colmap feature_extractor --database_path "$DB" --image_path "$ROOT/images" \
  --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0 --SiftExtraction.max_image_size 3200 \
  --SiftExtraction.max_num_features 8192 --SiftExtraction.estimate_affine_shape 0 \
  --SiftExtraction.domain_size_pooling 0 > "$ROOT/features_log.txt" 2>&1
mid=$(date +%s)
colmap exhaustive_matcher --database_path "$DB" \
  --SiftMatching.use_gpu 0 --SiftMatching.cross_check 1 --SiftMatching.max_ratio 0.8 \
  --SiftMatching.max_distance 0.7 --SiftMatching.guided_matching 0 >> "$ROOT/features_log.txt" 2>&1
end=$(date +%s)
echo "$1: extraction $((mid-start)) s, matching $((end-mid)) s"
