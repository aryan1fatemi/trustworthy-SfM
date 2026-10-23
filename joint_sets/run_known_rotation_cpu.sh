#!/bin/bash
#SBATCH --job-name=known_rotation
#SBATCH --output=known_rot_%j.out
#SBATCH --error=known_rot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sfatemi@ufl.edu
#SBATCH --account=eel5840
#SBATCH --qos=eel5840-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=48:00:00

set -euo pipefail

SRC=/home/sfatemi/Desktop/colmap/joint_sets     # known_rotation.py, geom.py, colmap_io.py
DATA=/blue/eel5840/sfatemi/airplane
OUT=/blue/eel5840/sfatemi/known_rotation_results
PY=/blue/eel5840/sfatemi/envs/cuopt/bin/python
NCAMS=200
SCHEME=mult             # fixed 3 px is infeasible at COLMAP's rotations

echo "host: $(hostname)   started: $(date)   cpus: ${SLURM_CPUS_PER_TASK}"
mkdir -p "$OUT"
cd "$SRC"
"$PY" -u known_rotation.py --scene "$DATA" --n-cams "$NCAMS" --scheme "$SCHEME" \
    --solver highs-ipm --procs "${SLURM_CPUS_PER_TASK}" --out "$OUT"

echo "finished: $(date)"
