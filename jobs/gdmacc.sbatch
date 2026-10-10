#!/bin/bash
#SBATCH --job-name=gdmacc
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=gdmacc_%j.out
#SBATCH --error=gdmacc_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python gia_dose_matched.py --units acc_model,acc_agdep --all_bg --suffix _modelacc
echo "[done] $(date -Is)"
