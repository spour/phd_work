#!/bin/bash
#SBATCH --job-name=motifnet4
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=motifnet4_%j.out
#SBATCH --error=motifnet4_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate

python motifnet4.py --lam 1.0
mv motifnet4_results.tsv motifnet4_results_lam1.tsv
python motifnet4.py --lam 0.0
mv motifnet4_results.tsv motifnet4_results_lam0.tsv
echo "[done] $(date -Is)"
