#!/bin/bash
#SBATCH --job-name=cnn3p_leg
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=cnn_legacy_probe_%j.out
#SBATCH --error=cnn_legacy_probe_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
echo "=== occlusion, reference model ==="
python cnn_occlusion.py --legacy --model cnn_model.pt --out occlusion_profile_reference.tsv
echo "=== ISM + DeepLift vs dinuc null, reference model ==="
python cnn_interpret2.py --legacy --model cnn_model.pt --suffix _reference
