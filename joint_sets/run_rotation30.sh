#!/bin/bash
#SBATCH --job-name=rotation30
#SBATCH --output=rotation30_%j.out
#SBATCH --error=rotation30_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sfatemi@ufl.edu
#SBATCH --account=eel5840
#SBATCH --qos=eel5840-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
# Rotation bounds and the joint set with rotations as unknowns, 30-view chair spiral (exhaustive matching).
#     sbatch run_rotation30.sh
#   1. known_rotation.py      fixed COLMAP rotations      -> camera-centre boxes (input of step 2)
#   2. rotation_bounds.py     per-camera rotation boxes, starting from |delta| <= RHO0 (an assumption on the
#                             largest rotation error, not tuned), shrunk over ROUNDS decoupled LP rounds
#   3. known_rotation.py --rot  centre ranges (and rotation ranges) with R_i = R_hat_i Exp(delta_i), delta_i in its box
# Needs in $DATA: "COLMAP results exhaustive/" (with colmap_log.txt) and transforms.json (GT, evaluation only).

set -euo pipefail

SRC=/home/sfatemi/Desktop/colmap/joint_sets     # known_rotation.py, rotation_bounds.py, geom.py, colmap_io.py
DATA=/blue/eel5840/sfatemi/chair_spiral30
MODEL="$DATA/COLMAP results exhaustive"
OUT=/blue/eel5840/sfatemi/rotation30_results
PY=/blue/eel5840/sfatemi/envs/cuopt/bin/python
NCAMS=30
SCHEME=mult
EPS=3
RHO0=2.0                # deg; every true rotation error must be below this (spiral30: max 0.71 deg)
ROUNDS=3
NPOINTS=0               # >0: also point boxes for a random sample of landmarks (seed 0)
P=${SLURM_CPUS_PER_TASK:-8}

echo "host: $(hostname)   started: $(date)   cpus: $P"
mkdir -p "$OUT/fixedR" "$OUT/rot"
cd "$SRC"

echo "=== 1. fixed rotations: centre boxes ($(date)) ==="
"$PY" -u known_rotation.py --scene "$DATA" --model "$MODEL" --n-cams $NCAMS --scheme $SCHEME --eps $EPS \
    --solver highs-ipm --procs $P --lp-time-limit 1800 --out "$OUT/fixedR"
CENTRES=$(ls -t "$OUT"/fixedR/*_known_rotation_${NCAMS}cams.json | head -1)     # the name comes from geom.run_name
echo "centre boxes: $CENTRES"

echo "=== 2. rotation bounds ($(date)) ==="
"$PY" -u rotation_bounds.py --scene "$DATA" --model "$MODEL" --centres "$CENTRES" --n-cams $NCAMS \
    --scheme $SCHEME --eps $EPS --rho0 $RHO0 --rounds $ROUNDS --procs $P --out "$OUT/rot"
ROT=$(ls -t "$OUT"/rot/*_rotation_bounds.json | head -1)
echo "rotation bounds: $ROT"

echo "=== 3. joint set with rotation unknowns ($(date)) ==="
"$PY" -u known_rotation.py --scene "$DATA" --model "$MODEL" --n-cams $NCAMS --scheme $SCHEME --eps $EPS \
    --solver highs-ipm --procs $P --lp-time-limit 1800 --rot "$ROT" --rot-ranges \
    --point-ranges $NPOINTS --point-seed 0 --out "$OUT/rot"

echo "finished: $(date)"
