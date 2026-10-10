#!/bin/bash
# one-shot installer for the aparent2 (tf 1.13.1 / keras 2.2.4 / py 3.6) stack
set -euo pipefail

ENV_NAME=aparent2_env
APARENT_DIR="${HOME}/tools/aparent-resnet"   # where to clone the repo
ISOLEARN_DIR="${HOME}/tools/isolearn"

mkdir -p "${HOME}/tools"

conda create -y -n "${ENV_NAME}" python=3.6
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install --no-cache-dir \
    "tensorflow==1.13.1" \
    "keras==2.2.4" \
    "h5py==2.10.0" \
    "numpy==1.16.6" \
    "scipy==1.5.4" \
    "pandas==1.1.5" \
    "matplotlib==3.3.4" \
    "protobuf==3.19.6" \
    "pysam>=0.15"

if [ ! -d "${ISOLEARN_DIR}" ]; then
    git clone https://github.com/johli/isolearn.git "${ISOLEARN_DIR}"
fi
cd "${ISOLEARN_DIR}"
python setup.py install

if [ ! -d "${APARENT_DIR}" ]; then
    git clone https://github.com/johli/aparent-resnet.git "${APARENT_DIR}"
fi
cd "${APARENT_DIR}"
python setup.py install

python - <<'PYTEST'
import sys
print(f"Python {sys.version.split()[0]}")
import tensorflow as tf
print(f"TF      {tf.__version__}")
import keras
print(f"Keras   {keras.__version__}")
import numpy as np
print(f"numpy   {np.__version__}")
from aparent.predictor import find_polya_peaks_simple
print("aparent.predictor import OK")
PYTEST

echo ""
echo "Done. APARENT2 env is '${ENV_NAME}'."
echo "  repo:    ${APARENT_DIR}"
echo "  weights: ${APARENT_DIR}/saved_models/"
echo ""
echo "Activate with:  conda activate ${ENV_NAME}"