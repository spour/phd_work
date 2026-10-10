#!/bin/bash
#SBATCH --job-name=acc_ctl
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=acc_ctl_%j.out
#SBATCH --error=acc_ctl_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python acceptor_vs_intron_positional.py
echo "[done] $(date -Is)"
