#!/usr/bin/env bash
# install maxentpy without the failing c extension

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: no virtualenv active. Activate first:"
    echo "  source ~/bpnet_env/bin/activate"
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap "rm -rf $WORKDIR" EXIT
cd "$WORKDIR"

echo "[noc] cloning maxentpy"
git clone --depth 1 https://github.com/kepbod/maxentpy.git
cd maxentpy

echo "[noc] overwriting setup.py to drop the C extension"
cat > setup.py << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Patched by install_maxentpy_noc.sh: ext_modules removed so install
# succeeds without the _hashseq C extension. The pure-Python maxent.py
# module works fine without it; only maxent_fast.py is affected.

from setuptools import setup

setup(name='maxentpy',
      version='0.0.3',
      description='Calculate splice site strength',
      url='https://github.com/kepbod/maxentpy',
      packages=['maxentpy'],
      package_data={'maxentpy': ['data/*']},
      install_requires=['msgpack-python']
      )
EOF
echo "[noc] wrote new setup.py"

echo "[noc] installing (pure-Python path)"
pip install . --no-deps

echo "[noc] verifying"
python3 -c "
from maxentpy import maxent
from maxentpy.maxent import load_matrix5, load_matrix3
m5 = load_matrix5(); m3 = load_matrix3()
s5 = maxent.score5('GAGGTGCTG', matrix=m5)
s3 = maxent.score3('TTCCAAACGAACTTTTGTAGGGA', matrix=m3)
print(f'  score5(GAGGTGCTG)            = {s5:.4f}  (expected ~9.79)')
print(f'  score3(TTCCAAACGAACTTTTGTAGGGA) = {s3:.4f}  (expected ~9.4)')
print('[noc] maxentpy works in pure-Python mode')
"