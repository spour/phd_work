#!/usr/bin/env bash
# set up the toolchain for the splicing model

set -euo pipefail

INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/tools/rnahybrid}"
VENV_PATH="${VENV_PATH:-$HOME/venv_2025}"
SKIP_RNAHYBRID="${SKIP_RNAHYBRID:-0}"
WORKDIR="$(mktemp -d)"
trap "rm -rf $WORKDIR" EXIT

echo "=========================================="
echo "Splicing toolchain install"
echo "=========================================="
echo "  target prefix:   $INSTALL_PREFIX"
echo "  venv:            $VENV_PATH"
echo "  skip RNAhybrid:  $SKIP_RNAHYBRID"
echo "  build dir:       $WORKDIR"
echo ""

echo "=========================================="
echo "Step 1: installing maxentpy (REQUIRED)"
echo "=========================================="

if [[ ! -d "$VENV_PATH" ]]; then
    echo "[install] ERROR: $VENV_PATH not found"
    echo "[install] create it first: python3 -m venv $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"
echo "[install] activated $VENV_PATH"

if pip install maxentpy 2>&1 | tail -1 | grep -qE "Successfully installed|already satisfied"; then
    echo "[install] maxentpy installed via PyPI"
else
    echo "[install] PyPI install failed; trying GitHub source"
    pip install "git+https://github.com/kepbod/maxentpy.git"
fi

python3 -c "from maxentpy import maxent; from maxentpy.maxent import load_matrix5, load_matrix3; \
m5 = load_matrix5(); m3 = load_matrix3(); \
s = maxent.score5('GAGGTGCTG', matrix=m5); \
print(f'[install] maxentpy smoke test: score5(GAGGTGCTG) = {s:.4f}  (expected approx 9.79)')"

echo ""

echo "=========================================="
echo "Step 2: verifying ViennaRNA Python bindings (REQUIRED for build_u1_lookup.py)"
echo "=========================================="

if python3 -c "import RNA" 2>/dev/null; then
    python3 -c "import RNA; print(f'[install] ViennaRNA RNA module OK (version: {RNA.__version__ if hasattr(RNA,\"__version__\") else \"unknown\"})')"
    python3 -c "import RNA; e = RNA.duplexfold('ACUUACCUG', 'AGGUAAGUC').energy; print(f'[install] duplexfold smoke test: U1 vs canonical 5SS = {e:.4f} kcal/mol  (expected: strongly negative, around -8 to -10)')"
else
    echo "[install] ViennaRNA Python bindings NOT found in venv."
    echo "[install] On Compute Canada (Alliance Canada) clusters:"
    echo "[install]     module load StdEnv/2023 viennarna/2.6.4"
    echo "[install]     (then re-run this script with the module loaded)"
    echo "[install] Alternatively try: pip install ViennaRNA"
    echo "[install] You only need this for build_u1_lookup.py; once you have"
    echo "[install] u1_lookup.npy on disk, no further ViennaRNA calls happen."
fi

echo ""

if [[ "$SKIP_RNAHYBRID" == "1" ]]; then
    echo "=========================================="
    echo "Step 3: SKIPPED (SKIP_RNAHYBRID=1)"
    echo "=========================================="
    echo "[install] RNAhybrid is not required for the splicing model."
    echo "[install] If you want it later for orthogonal U1 verification,"
    echo "[install] re-run with SKIP_RNAHYBRID=0."
    echo ""
else
    echo "=========================================="
    echo "Step 3: building RNAhybrid from source (OPTIONAL)"
    echo "=========================================="
    echo "[install] NOTE: this is optional. RNAhybrid is only needed as an"
    echo "[install] orthogonal U1-affinity check, not for the main benchmark."
    echo "[install] Skip with: SKIP_RNAHYBRID=1 bash install_splicing_tools.sh"
    echo ""

    cd "$WORKDIR"

    PRIMARY="http://ftp.debian.org/debian/pool/main/r/rnahybrid/rnahybrid_2.1.2.orig.tar.gz"
    FALLBACK="http://deb.debian.org/debian/pool/main/r/rnahybrid/rnahybrid_2.1.2.orig.tar.gz"

    GOT_SOURCE=0
    for URL in "$PRIMARY" "$FALLBACK"; do
        echo "[install] trying: $URL"
        if curl -sSLf -o rnahybrid.tar.gz "$URL"; then
            if file rnahybrid.tar.gz | grep -qi "gzip compressed"; then
                echo "[install]   downloaded, file type OK"
                GOT_SOURCE=1
                break
            else
                echo "[install]   downloaded but not a gzip (likely HTML); trying next URL"
                rm -f rnahybrid.tar.gz
            fi
        else
            echo "[install]   curl failed"
        fi
    done

    if [[ "$GOT_SOURCE" == "0" ]]; then
        echo ""
        echo "[install] ERROR: could not download RNAhybrid from any mirror."
        echo "[install] Manual download workaround:"
        echo "[install]   1. From a machine with browser access, fetch:"
        echo "[install]      $PRIMARY"
        echo "[install]   2. Copy the .orig.tar.gz to the cluster, then:"
        echo "[install]      mkdir -p $INSTALL_PREFIX && cd <dir with tarball>"
        echo "[install]      tar xzf rnahybrid_2.1.2.orig.tar.gz"
        echo "[install]      cd RNAhybrid-2.1.2"
        echo "[install]      ./configure --prefix=$INSTALL_PREFIX"
        echo "[install]      make -j4 && make install"
        echo ""
        echo "[install] The splicing model and benchmark will still work without RNAhybrid."
        echo "[install] Re-run with SKIP_RNAHYBRID=1 to dismiss this step."
        exit 1
    fi

    tar xzf rnahybrid.tar.gz
    cd RNAhybrid-2.1.2
    ./configure --prefix="$INSTALL_PREFIX"
    make -j 4
    make install
    echo ""
    echo "[install] RNAhybrid built; binary at $INSTALL_PREFIX/bin/RNAhybrid"

    "$INSTALL_PREFIX/bin/RNAhybrid" -s 3utr_human -t ACGUACGUACGU -q ACGUACGU 2>&1 | head -5 || \
        echo "[install] (smoke test failed but binary is present; verify manually)"
fi

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
echo ""
echo "Add to your job scripts:"
echo "  source $VENV_PATH/bin/activate"
if [[ "$SKIP_RNAHYBRID" == "0" ]]; then
    echo "  export PATH=$INSTALL_PREFIX/bin:\$PATH"
fi
echo ""
echo "Next steps:"
echo "  1. Generate the U1 lookup (once):"
echo "       python3 build_u1_lookup.py --out data/u1_lookup.npy"
echo "     (Requires ViennaRNA Python bindings; module load if needed.)"
echo ""
echo "  2. Extract splicing features:"
echo "       python3 extract_splicing_features.py \\"
echo "           --pos_bed canonical.bed --neg_bed 3way_intragenic.bed \\"
echo "           --reference_fasta hg38.fa --u1_lookup data/u1_lookup.npy \\"
echo "           --upstream_nt 3000 --downstream_nt 1000 \\"
echo "           --out_tsv splicing_features.tsv"
echo ""
echo "  3. Or sweep all window configs:"
echo "       bash benchmark_splicing_windows.sh"