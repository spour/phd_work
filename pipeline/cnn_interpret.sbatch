#!/bin/bash
#SBATCH --job-name=cnn_interpret
#SBATCH --account=def-thughes           # <-- set to your allocation
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cnn_interpret_%j.out
#SBATCH --error=logs/cnn_interpret_%j.err

set -euo pipefail
mkdir -p logs

module purge
module load StdEnv/2023 gcc python/3.11 arrow/17.0.0   # arrow OK here: torch reads no parquet via arrow at score time
source ~/splice_env2/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub

python3 scripts/cnn_interpret.py \
  --shortlist shortlist_crossfit_windows.parquet \
  --reference_fasta ../train/data/hg38.fa \
  --upstream_nt 2000 --downstream_nt 500 \
  --n_folds 4 --activation exp \
  --modisco \
  --out cnn_interp

echo "done: $(date)"
