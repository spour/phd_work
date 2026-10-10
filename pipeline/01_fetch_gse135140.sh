#!/usr/bin/env bash
# pull individual sample bigwigs from gse135140

set -uo pipefail
ROOT="${EXTERNAL_ROOT:-$PWD/external}"
DEST="$ROOT/geo/GSE135140/bw"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36'
mkdir -p "$DEST"

SAMPLES="
GSM3989596 EU-labeled_polyA_cAMO         1
GSM3989597 EU-labeled_polyA_U1AMO        1
GSM3989602 U1A_XLIP_cAMO                 2
GSM3989603 U1C_XLIP_cAMO                 2
GSM3989616 Bio-cAMO_XLPD                 2
GSM3989617 Bio-U1AMO_XLPD                2
GSM3989606 CFIm25_XLIP_cAMO              2
GSM3989607 CFIm25_XLIP_U1AMO             2
GSM3989610 CstF64_XLIP_cAMO              2
GSM3989611 CstF64_XLIP_U1AMO             2
GSM3989598 ctrl_siRNA_cAMO_labeled       3
GSM3989599 ctrl_siRNA_U1AMO_labeled      3
GSM3989600 siCPSF5_cAMO_labeled          3
GSM3989601 siCPSF6_cAMO_labeled          3
"

TIERS="${*:-1 2}"
note() { printf '\033[1m==> %s\033[0m\n' "$*"; }

get() {
  local gsm="$1" stem="$2" strand="$3"
  local fn="${gsm}_${stem}.${strand}.bw"
  local dst="$DEST/$fn"
  if [ -s "$dst" ]; then echo "  cached  $fn"; return 0; fi
  local url="https://ftp.ncbi.nlm.nih.gov/geo/samples/${gsm:0:7}nnn/${gsm}/suppl/${fn}"
  if curl -fsSL --retry 4 --retry-delay 3 --connect-timeout 30 --max-time 7200 \
          -A "$UA" -o "$dst.part" "$url" 2>/dev/null && [ -s "$dst.part" ]; then
    local m; m=$(od -An -tx1 -N4 "$dst.part" | tr -d ' \n')
    if [ "$m" = "26fc8f88" ] || [ "$m" = "888ffc26" ]; then
      mv "$dst.part" "$dst"
      echo "  got     $fn  $(du -h "$dst" | cut -f1)"; return 0
    fi
    echo "  BAD     $fn  bad magic '$m'" >&2
  else
    echo "  FAIL    $fn" >&2
  fi
  rm -f "$dst.part"; return 1
}

note "GSE135140 bigWigs -> $DEST  (tiers: $TIERS)"
while read -r gsm stem tier; do
  [ -z "${gsm:-}" ] && continue
  case " $TIERS " in *" $tier "*) ;; *) continue ;; esac
  get "$gsm" "$stem" fwd
  get "$gsm" "$stem" rev
done <<< "$SAMPLES"

echo
echo "total: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo
echo "Next, confirm these really are hg38 before trusting any coordinate:"
echo "  python 02_u1_usage_at_peaks.py --check-only \\"
echo "    --bw-dir $DEST --peaks <your peak table>"