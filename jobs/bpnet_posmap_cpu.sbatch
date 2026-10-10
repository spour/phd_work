#!/bin/bash
# the interpretability suite on the bpnet variant checkpoint
#SBATCH --job-name=bp_pm
#SBATCH --account=def-thughes_cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --output=bp_pm_%j.out
#SBATCH --error=bp_pm_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python bpnet_interp.py --stages 3 --posmap_genes 200
echo "[done] $(date -Is)"
