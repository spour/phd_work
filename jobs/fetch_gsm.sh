#!/usr/bin/env bash
# download named geo samples, never the raw tar

set -uo pipefail
ROOT="${EXTERNAL_ROOT:-$PWD/external}"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'

note() { printf '\033[1m==> %s\033[0m\n' "$*"; }

PCF11_3P="GSM3536477 GSM3536478 GSM3536479 GSM3536480 \
GSM3536481 GSM3536482 GSM3536483 GSM3536484"
PCF11_3P_LABEL="siLUC rep1-4, siPCF11 rep1-4"

PCF11_CHR="GSM3483754 GSM3483755 GSM3483756 GSM3483757"
PCF11_CHR_LABEL="chrRNA-seq siLUC rep1-2, siPCF11 rep1-2"

SO_SI="GSM3989598 GSM3989600 GSM3989601"
SO_SI_LABEL="ctrl siRNA (background), siCPSF5, siCPSF6 -- all in cAMO"

SO_XLIP="GSM3989602 GSM3989603 GSM3989604 GSM3989606 GSM3989608 GSM3989610 \
GSM3989612 GSM3989614"
SO_XLIP_LABEL="U1A, U1C, SF3B1, CFIm25, Fip1, CstF64, hnRNPC, Input (all cAMO)"

gsm_dir() { echo "https://ftp.ncbi.nlm.nih.gov/geo/samples/${1:0:7}nnn/${1}/suppl"; }

list_gsm() {
  local g="$1"
  local base; base=$(gsm_dir "$g")
  local title
  title=$(curl -fsSL -A "$UA" \
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${g}&targ=self&form=text&view=brief" \
    2>/dev/null | awk -F' = ' '/^!Sample_title/{print $2; exit}')
  echo "  $g  ${title:-<no title>}"
  curl -fsSL -A "$UA" "$base/" 2>/dev/null \
    | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//' \
    | grep -vE '^\?|^/|^$|hhs.gov' | sed 's/^/      /'
}

get_gsm() {
  local g="$1" dest="$2"
  local base; base=$(gsm_dir "$g")
  mkdir -p "$dest"
  local files
  files=$(curl -fsSL -A "$UA" "$base/" 2>/dev/null \
    | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//' \
    | grep -vE '^\?|^/|^$|hhs.gov')
  if [ -z "$files" ]; then echo "  no files listed for $g" >&2; return 1; fi
  for f in $files; do
    case "$f" in
      *.bw|*.bigWig|*.bedGraph*|*.bed.gz|*.bed|*.broadPeak*|*.txt.gz|*.tsv.gz) ;;
      *) continue ;;
    esac
    local out="$dest/$f"
    if [ -s "$out" ]; then echo "  cached  $f"; continue; fi
    if curl -fsSL --retry 3 --connect-timeout 30 -A "$UA" -o "$out.part" \
         "$base/$f" 2>/dev/null && [ -s "$out.part" ]; then
      mv "$out.part" "$out"
      echo "  got     $f  $(du -h "$out" | cut -f1)"
    else
      rm -f "$out.part"; echo "  FAIL    $f" >&2
    fi
  done
}

case "${1:-help}" in
  list)
    acc="${2:-}"
    [ -z "$acc" ] && { echo "usage: $0 list GSE123105"; exit 2; }
    pre="${acc:0:${#acc}-3}nnn"
    note "$acc sample titles"
    curl -fsSL -A "$UA" \
      "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${acc}&targ=gsm&form=text&view=brief" \
      2>/dev/null | awk -F' = ' '/^\^SAMPLE/{g=$2} /^!Sample_title/{printf "  %-12s %s\n", g, $2}'
    ;;
  probe)
    shift
    note "listing files for the named samples"
    for g in "$@"; do list_gsm "$g"; done
    ;;
  pcf11)
    note "PCF11 3' mRNA-seq: $PCF11_3P_LABEL"
    echo "  human HeLa only; zebrafish samples deliberately excluded"
    for g in $PCF11_3P; do
      echo "-- $g"; get_gsm "$g" "$ROOT/geo/GSE123105/3pmrnaseq"
    done
    ;;
  pcf11_chr)
    note "PCF11 chrRNA-seq: $PCF11_CHR_LABEL"
    for g in $PCF11_CHR; do
      echo "-- $g"; get_gsm "$g" "$ROOT/geo/GSE123105/chrrnaseq"
    done
    ;;
  so2019_xlip)
    note "So 2019 XLIP: $SO_XLIP_LABEL"
    echo "  ~1 Gb total; hg38, no liftOver needed"
    for g in $SO_XLIP; do
      echo "-- $g"; get_gsm "$g" "$ROOT/geo/GSE135140/xlip"
    done
    ;;
  so2019_si)
    note "So 2019 CFIm knockdowns: $SO_SI_LABEL"
    for g in $SO_SI; do
      echo "-- $g"; get_gsm "$g" "$ROOT/geo/GSE135140/sirna"
    done
    ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac