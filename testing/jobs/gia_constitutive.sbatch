#!/bin/bash
#SBATCH --job-name=gia_const
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=gia_const_%j.out
#SBATCH --error=gia_const_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python gia_constitutive.py
