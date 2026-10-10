#!/bin/bash
# multi-scale occlusion + rank-stratified profile + ig cross-check on the reported cnn
#SBATCH --job-name=cnn_interp
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=cnn_interp_%j.out
#SBATCH --error=cnn_interp_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python cnn_occlusion_multiscale.py --widths 25,50,100,200,400 --nrep 10 --batch 1024

python ig_positional.py --steps 128 --batch 256

echo "[done] $(date -Is)"
