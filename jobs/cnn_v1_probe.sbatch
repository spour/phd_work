#!/bin/bash
#SBATCH --job-name=cnn3p_v1p
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=cnn_v1_probe_%j.out
#SBATCH --error=cnn_v1_probe_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
echo "=== retrain best exponential variant ==="
python cnn_train2.py --only V1_exp_wd3e-4 --out cnn_model_v1
echo "=== metrics ==="
python cnn_eval2.py --scores cnn_model_v1_scores.npz --out cnn_metrics_retrained_v1.tsv
echo "=== occlusion ==="
python cnn_occlusion.py --model cnn_model_v1.pt --out occlusion_profile_v1.tsv
echo "=== filters + attribution + ISM + GIA ==="
python cnn_interpret2.py --model cnn_model_v1.pt --suffix _v1
