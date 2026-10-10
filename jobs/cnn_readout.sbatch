#!/bin/bash
#SBATCH --job-name=cnn_readout
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=cnn_readout_%j.out
#SBATCH --error=cnn_readout_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
set -e
echo "=== 1. grafted readout on the frozen model ==="
python cnn_readout_train.py
echo "=== 2. signed filter x bin occlusion map (unchanged model) ==="
python cnn_posmap_occlusion.py
echo "=== 3. pre-registered donor/U1 test ==="
python cnn_donor_test.py
