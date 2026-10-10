#!/bin/bash
#SBATCH --job-name=acc_fu
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=acc_fu_%j.out
#SBATCH --error=acc_fu_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate

python acceptor_arch_train.py --variants base,aux --seeds 42,43,44,45,46,47,48,49
echo "[done] $(date -Is)"
