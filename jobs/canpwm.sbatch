#!/bin/bash
#SBATCH --job-name=canpwm
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=canpwm_%j.out
#SBATCH --error=canpwm_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python model_acc_pwm.py --pyr 0.80 --no_g --seed_seq TTTAACTTTCTTTTTTACAGACG --auto_temp --out canon_acc_pwm.tsv
echo "[done] $(date -Is)"
