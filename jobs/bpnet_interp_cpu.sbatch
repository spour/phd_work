#!/bin/bash
# the interpretability suite on the bpnet variant checkpoint
#SBATCH --job-name=bp_int12
#SBATCH --account=def-thughes_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=bp_int12_%j.out
#SBATCH --error=bp_int12_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python bpnet_interp.py --stages 1,2
echo "[done] $(date -Is)"
