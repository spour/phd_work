#!/bin/bash
#SBATCH --job-name=spldiag
#SBATCH --account=def-thughes_gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=spldiag_%j.out
#SBATCH --error=spldiag_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python splice_diagnose.py
echo "[done] $(date -Is)"
