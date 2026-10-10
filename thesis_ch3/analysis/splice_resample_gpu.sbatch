#!/bin/bash
#SBATCH --job-name=ss_resample
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/ss_resample_%j.out
#SBATCH --error=logs/ss_resample_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3
source ~/splice_env2/bin/activate
python py/splice_scramble.py scan 4 resample
python py/splice_scramble.py score resample
echo "[done] $(date -Is)"
