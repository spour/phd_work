#!/bin/bash
#SBATCH --job-name=cnnshuf
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=cnn_shuf_%j.out
#SBATCH --error=cnn_shuf_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python cnn_shuffle_control.py
