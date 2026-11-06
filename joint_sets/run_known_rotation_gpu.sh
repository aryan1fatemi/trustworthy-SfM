#!/bin/bash
#SBATCH --job-name=known_rotation
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=known_rotation_%j.log
# Known-rotation joint set on one GPU with NVIDIA cuOpt (Linux only; Python >= 3.12, CUDA 12 or 13, Volta or newer).
#     sbatch run_known_rotation_gpu.sh /path/to/airplane [n_cams] [scheme]
set -u
SCENE=${1:?usage: sbatch run_known_rotation_gpu.sh /path/to/dataset [n_cams] [scheme]}
NCAMS=${2:-200}
SCHEME=${3:-mult}
cd "$(dirname "$0")"
module load cuda python 2>/dev/null || true
python -c "import cuopt" 2>/dev/null || pip install --user --extra-index-url=https://pypi.nvidia.com cuopt-cu12 numpy scipy
nvidia-smi
python -u known_rotation.py --scene "$SCENE" --n-cams $NCAMS --scheme $SCHEME --solver cuopt --cuopt-method barrier --out results
