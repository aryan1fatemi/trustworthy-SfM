#!/usr/bin/env bash
# COLMAP on an ordered spiral render (render_chair.py --path spiral), intrinsics FIXED as in run_colmap.sh.
#     wsl -e bash /mnt/c/Users/fatem/COLMAP/nerf_synthetic/chair/run_colmap_spiral.sh chair_spiral30 sequential
# matcher:
#   exhaustive  every image pair
#   sequential  each image with its OVERLAP neighbours in capture order, cyclic (the orbit closes, so the last
#               images are also matched with the first); pairs are imported as a list, no vocabulary tree needed
# Outputs, per matcher: database_<m>.db, sparse_<m>/, "COLMAP results <m>/" (text model + mapper log).
set -euo pipefail
SCENE=${1:?scene folder under nerf_synthetic, e.g. chair_spiral30}
M=${2:?exhaustive or sequential}
OVERLAP=${OVERLAP:-5}
ROOT=/mnt/c/Users/fatem/COLMAP/nerf_synthetic/$SCENE
DB=$ROOT/database_$M.db
OUT=$ROOT/sparse_$M
TXT="$ROOT/COLMAP results $M"
FX=1111.111031193768                      # Blender fl_x (transforms.json)
CX=400
CY=400

rm -f "$DB"; rm -rf "$OUT"; mkdir -p "$OUT" "$TXT"

colmap feature_extractor \
  --database_path "$DB" --image_path "$ROOT/images" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "$FX,$FX,$CX,$CY" \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.max_image_size 3200 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.estimate_affine_shape 0 \
  --SiftExtraction.domain_size_pooling 0

MATCH_OPTS="--SiftMatching.use_gpu 0 --SiftMatching.cross_check 1 --SiftMatching.max_ratio 0.8
            --SiftMatching.max_distance 0.7 --SiftMatching.guided_matching 0"
if [ "$M" = exhaustive ]; then
  colmap exhaustive_matcher --database_path "$DB" $MATCH_OPTS
else
  N=$(ls "$ROOT/images" | grep -c '\.png$')
  PAIRS=$TXT/pairs.txt
  : > "$PAIRS"
  for ((i = 1; i <= N; i++)); do
    for ((d = 1; d <= OVERLAP; d++)); do
      j=$(( (i - 1 + d) % N + 1 ))
      echo "$i.png $j.png" >> "$PAIRS"
    done
  done
  colmap matches_importer --database_path "$DB" --match_list_path "$PAIRS" --match_type pairs $MATCH_OPTS
fi

colmap mapper \
  --database_path "$DB" --image_path "$ROOT/images" --output_path "$OUT" \
  --Mapper.multiple_models 1 \
  --Mapper.min_model_size 3 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 \
  --Mapper.min_num_matches 15 \
  --Mapper.init_min_num_inliers 100 \
  --Mapper.init_min_tri_angle 16 \
  --Mapper.abs_pose_min_num_inliers 30 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.tri_ignore_two_view_tracks 1 2>&1 | tee "$TXT/colmap_log.txt"

best=""; bestn=0
for m in "$OUT"/*/; do
  n=$(colmap model_analyzer --path "$m" 2>&1 | grep -i "Registered images" | grep -o '[0-9]*' | tail -1)
  echo "model $m : ${n:-0} registered images"
  if [ "${n:-0}" -gt "$bestn" ]; then bestn=$n; best=$m; fi
done
echo "exporting $best ($bestn images) -> $TXT"
colmap model_converter --input_path "$best" --output_path "$TXT" --output_type TXT
