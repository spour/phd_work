#!/bin/bash
#SBATCH --job-name=cnn3p_v2p
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=cnn_v2_probe_%j.out
#SBATCH --error=cnn_v2_probe_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
echo "=== within-gene metrics + GC removal, retrained ==="
python cnn_eval2.py --scores cnn_model_v2_scores.npz --out cnn_metrics_retrained.tsv
echo "=== occlusion, retrained ==="
python cnn_occlusion.py --model cnn_model_v2.pt --out occlusion_profile.tsv
echo "=== filters + attribution + ISM + GIA, retrained ==="
python cnn_interpret2.py --model cnn_model_v2.pt --suffix _retrained
