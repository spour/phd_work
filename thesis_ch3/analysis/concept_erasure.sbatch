#!/bin/bash
#SBATCH --job-name=concept_er
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/concept_er_%j.out
#SBATCH --error=logs/concept_er_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3
source ~/splice_env2/bin/activate
python py/concept_erasure.py within
echo "[done] $(date -Is)"
