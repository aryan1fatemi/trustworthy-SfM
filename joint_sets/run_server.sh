#!/bin/bash
#SBATCH --job-name=joint_sets
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=joint_sets_%j.log
# Runs every window of one dataset on one CPU node (no GPU needed).
#     sbatch run_server.sh /path/to/airplane [n_cams] [eps]
set -u
SCENE=${1:?usage: sbatch run_server.sh /path/to/dataset [n_cams] [eps]}
NCAMS=${2:-6}
EPS=${3:-3}
cd "$(dirname "$0")"
module load python 2>/dev/null || true
python -c "import numpy, scipy" || pip install --user numpy scipy
NIMG=$(grep -c "\.png\|\.jpg\|\.JPG" "$SCENE/COLMAP results/images.txt")
NIMG=$((NIMG / 1))
W=0
while true; do
    START=$((W * (NCAMS - 2)))
    if [ $((START + 3)) -gt $NIMG ]; then break; fi
    echo "=== window $W (cameras $START..$((START + NCAMS - 1))) ==="
    python -u joint_sets.py --scene "$SCENE" --window $W --n-cams $NCAMS --eps $EPS \
        --procs ${SLURM_CPUS_PER_TASK:-8} --out results || echo "window $W failed"
    W=$((W + 1))
done
echo "done"
