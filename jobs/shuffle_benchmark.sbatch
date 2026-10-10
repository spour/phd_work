#!/bin/bash
#SBATCH --job-name=shufbench
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=shuffle_benchmark_%j.out
#SBATCH --error=shuffle_benchmark_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python shuffle_benchmark.py --k 3 --jobs 32
