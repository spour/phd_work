#!/bin/bash
#SBATCH --job-name=design_relay
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --output=logs/design_relay_%j.out
#SBATCH --error=logs/design_relay_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3
source ~/splice_env2/bin/activate
python py/design_relay.py
echo "[done] $(date -Is)"
