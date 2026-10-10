#!/bin/bash
# setup script for maxentscan and rnahybrid

set -euo pipefail

TOOLS_DIR="${HOME}/tools"
mkdir -p "$TOOLS_DIR"
cd "$TOOLS_DIR"

if [[ ! -d "$TOOLS_DIR/maxentscan" ]]; then
    echo "[setup] Installing MaxEntScan..."
    git clone https://github.com/matthdsm/MaxEntScan.git "$TOOLS_DIR/maxentscan_repo"
    mkdir -p "$TOOLS_DIR/maxentscan"
    cp -r "$TOOLS_DIR/maxentscan_repo/"* "$TOOLS_DIR/maxentscan/"
    chmod +x "$TOOLS_DIR/maxentscan/score5.pl" "$TOOLS_DIR/maxentscan/score3.pl"
fi

echo "[setup] Testing MaxEntScan score5..."
cd "$TOOLS_DIR/maxentscan"
echo "cagGTAAGT" > /tmp/_test5.txt
RESULT_5=$(perl score5.pl /tmp/_test5.txt | awk '{print $NF}')
echo "    score5 result for 'cagGTAAGT': $RESULT_5 (expect ~10.86)"

echo "[setup] Testing MaxEntScan score3..."
echo "ttccaaacgaacttttgtagGGA" > /tmp/_test3.txt
RESULT_3=$(perl score3.pl /tmp/_test3.txt | awk '{print $NF}')
echo "    score3 result for 'ttccaaacgaacttttgtagGGA': $RESULT_3 (expect ~2.89)"

RNAH_REPO="$TOOLS_DIR/rnahybrid"
RNAH_PREFIX="$TOOLS_DIR/rnahybrid_prefix"
 
if [[ ! -d "$RNAH_REPO" ]]; then
    echo "[setup] Cloning RNAhybrid from Bielefeld GitLab..."
    git clone https://gitlab.ub.uni-bielefeld.de/bibiserv/rnahybrid.git \
              "$RNAH_REPO"
fi
 
if [[ ! -x "$RNAH_PREFIX/bin/RNAhybrid" ]]; then
    echo "[setup] Building RNAhybrid..."
    cd "$RNAH_REPO"
    if [[ ! -x configure ]]; then
        if command -v autoreconf >/dev/null 2>&1; then
            echo "[setup]   running autoreconf -fi (configure not found)"
            autoreconf -fi
        else
            echo "[setup]   ERROR: configure missing AND autoreconf unavailable."
            echo "[setup]   Load an autoconf module or install one and re-run."
            exit 1
        fi
    fi
    ./configure --prefix="$RNAH_PREFIX"
    make
    make install
    cd "$TOOLS_DIR"
fi
 
echo "[setup] Testing RNAhybrid..."
RNAH_BIN="$RNAH_PREFIX/bin/RNAhybrid"
"$RNAH_BIN" \
    -s 3utr_human -c \
    -t CAGGTAAGTATCAAGGTTACAAGAC \
    -q ACTTACCTG \
    | head -1 || echo "    (no output, see error above)"
 
cat <<EOF
 
[setup] Done. Tool paths:
  MaxEntScan score5: $TOOLS_DIR/maxentscan/score5.pl
  MaxEntScan score3: $TOOLS_DIR/maxentscan/score3.pl
  RNAhybrid:         $RNAH_PREFIX/bin/RNAhybrid
 
Add to your shell rc if you want them on PATH:
  export PATH="\$HOME/tools/maxentscan:\$HOME/tools/rnahybrid_prefix/bin:\$PATH"
 
For extract_splice_features.py, pass:
  --maxentscan  $TOOLS_DIR/maxentscan
  --rnahybrid   $RNAH_PREFIX/bin/RNAhybrid
 
EOF
 