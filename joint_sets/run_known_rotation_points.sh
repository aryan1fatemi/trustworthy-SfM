#!/bin/bash
#SBATCH --job-name=known_rot_points
#SBATCH --output=known_rot_points_%j.out
#SBATCH --error=known_rot_points_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sfatemi@ufl.edu
#SBATCH --account=eel5840
#SBATCH --qos=eel5840-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=06:00:00

set -euo pipefail

SRC=/home/sfatemi/Desktop/colmap/joint_sets     # known_rotation.py, geom.py, colmap_io.py
DATA=/blue/eel5840/sfatemi/airplane
OUT=/blue/eel5840/sfatemi/known_rotation_results  # same folder as the camera run: its LPs are reused
PY=/blue/eel5840/sfatemi/envs/cuopt/bin/python
NCAMS=200
SCHEME=mult
NPOINTS=500             # random sample of landmarks (seed 0); 6 LPs each

echo "host: $(hostname)   started: $(date)   cpus: ${SLURM_CPUS_PER_TASK}"
mkdir -p "$OUT"
cd "$SRC"
"$PY" -u known_rotation.py --scene "$DATA" --n-cams "$NCAMS" --scheme "$SCHEME" \
    --solver highs-ipm --procs "${SLURM_CPUS_PER_TASK}" --lp-time-limit 600 \
    --point-ranges "$NPOINTS" --point-seed 0 --out "$OUT"

echo "finished: $(date)"
