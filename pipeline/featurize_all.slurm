#!/bin/bash
#SBATCH --job-name=feat_array
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/feat_%A_%a.out
#SBATCH --error=logs/feat_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ~/venv_2025/bin/activate

PWM_LIST="${PWM_LIST:-/scratch/spour98/identify_model/pwm_paths}"
NAMES=(fold0 fold1 fold2 fold3 fold4 prod)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
DIR=splits/$NAME
mkdir -p "$DIR/features"

echo "[$(date)] featurize $NAME  (pwms=$PWM_LIST)"
for split in train val test; do
  for cls in pos neg; do
    fa="$DIR/${cls}_${split}.fa"; out="$DIR/features/${cls}_${split}.tsv"
    if [ -s "$out" ]; then echo "  skip existing $out"; continue; fi
    python3 scripts/featurization.py -f "$fa" -pwms "$PWM_LIST" \
        -out "$out" --bg 0.3 0.2 0.2 0.3
  done
done
echo "[$(date)] featurize $NAME done -> $DIR/features/"
