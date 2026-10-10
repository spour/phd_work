#!/usr/bin/env bash
# validate everything 00_download_and_liftover.sh fetched, re-download whatever is corrupt

set -uo pipefail
ROOT="${EXTERNAL_ROOT:-$PWD/external}"
GEO="$ROOT/geo"; CHAIN="$ROOT/chain"; BIN="$ROOT/bin"
REPORT="$ROOT/verify_report.tsv"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36'
MODE="${1:-repair}"

note() { printf '\033[1m==> %s\033[0m\n' "$*"; }
bad()  { printf '\033[31m  BAD  %s  (%s)\033[0m\n' "$1" "$2"; }
good() { printf '  ok   %s  %s bytes\n' "$1" "$2"; }

validate() {
  local f="$1"
  [ -s "$f" ] || { echo "empty or missing"; return; }
  local sz; sz=$(stat -c%s "$f")
  if head -c 512 "$f" | grep -qiE '<html|<!doctype html|<title>'; then
    echo "HTML error page (${sz} B)"; return
  fi
  case "$f" in
    *.gz)  gzip -t "$f" 2>/dev/null || { echo "not valid gzip (${sz} B)"; return; } ;;
    *.tar) tar -tf "$f" >/dev/null 2>&1 || { echo "not valid tar (${sz} B)"; return; } ;;
    *.bw|*.bigWig)
      local m; m=$(od -An -tx1 -N4 "$f" | tr -d ' \n')
      [ "$m" = "26fc8f88" ] || [ "$m" = "888ffc26" ] \
        || { echo "bad bigWig magic '$m' (${sz} B)"; return; } ;;
    */liftOver)
      head -c 4 "$f" | grep -q $'\x7fELF' || { echo "not an ELF binary (${sz} B)"; return; } ;;
  esac
  echo ""
}

refetch() {
  local url="$1" dest="$2"
  curl -fsSL --retry 4 --retry-delay 3 --connect-timeout 30 --max-time 3600 \
       -A "$UA" -o "$dest.part" "$url" 2>/dev/null
  if [ -s "$dest.part" ]; then
    mv "$dest.part" "$dest"
    local why; why=$(validate "$dest")
    if [ -z "$why" ]; then good "$(basename "$dest")" "$(stat -c%s "$dest")"; return 0; fi
    bad "$(basename "$dest")" "still bad after refetch: $why"
  else
    rm -f "$dest.part"; bad "$(basename "$dest")" "refetch returned nothing"
  fi
  return 1
}

geo_url_for() {
  local f="$1" acc base
  acc=$(basename "$(dirname "$f")")
  base="https://ftp.ncbi.nlm.nih.gov/geo/series/${acc:0:6}nnn/${acc}/suppl"
  echo "$base/$(basename "$f")"
}

note "validating $ROOT"
printf 'status\tbytes\tpath\treason\n' > "$REPORT"
declare -a BROKEN=()
while IFS= read -r -d '' f; do
  case "$f" in *_files.txt|*_listing.html|*manifest.tsv|*verify_report.tsv) continue ;; esac
  why=$(validate "$f"); sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ -z "$why" ]; then
    good "${f#$ROOT/}" "$sz"; printf 'ok\t%s\t%s\t\n' "$sz" "$f" >> "$REPORT"
  else
    bad "${f#$ROOT/}" "$why"; printf 'BAD\t%s\t%s\t%s\n' "$sz" "$f" "$why" >> "$REPORT"
    BROKEN+=("$f")
  fi
done < <(find "$ROOT" -type f \( -name '*.gz' -o -name '*.tar' -o -name '*.bw' \
         -o -name '*.bigWig' -o -name 'liftOver' \) -print0 2>/dev/null)

echo
echo "  $(grep -c '^ok' "$REPORT") valid, ${#BROKEN[@]} broken. Detail: $REPORT"
[ "$MODE" = "verify" ] && exit 0

if [ "${#BROKEN[@]}" -gt 0 ]; then
  note "re-downloading ${#BROKEN[@]} file(s) with a browser user-agent"
  for f in "${BROKEN[@]}"; do
    case "$f" in
      "$GEO"/*)   refetch "$(geo_url_for "$f")" "$f" ;;
      "$CHAIN"/hg19ToHg38.over.chain.gz)
        refetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz" "$f" ;;
      "$BIN"/liftOver)
        refetch "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver" "$f" && chmod +x "$f" ;;
      *) bad "$(basename "$f")" "no known source URL, refetch by hand" ;;
    esac
  done
fi

note "GEO filelist.txt (contents of each RAW tar, no tar download needed)"
for acc in GSE135140 GSE103252 GSE111310 GSE111793; do
  url="https://ftp.ncbi.nlm.nih.gov/geo/series/${acc:0:6}nnn/${acc}/suppl/filelist.txt"
  dest="$GEO/$acc/filelist.txt"
  mkdir -p "$GEO/$acc"
  if [ ! -s "$dest" ]; then
    curl -fsSL --retry 3 --connect-timeout 30 -A "$UA" -o "$dest" "$url" 2>/dev/null || true
  fi
  if [ -s "$dest" ]; then
    echo "  $acc: $(( $(wc -l < "$dest") - 1 )) entries"
    awk -F'\t' 'NR>1{print "    " $1 "  " $2}' "$dest" | head -25
    [ "$(wc -l < "$dest")" -gt 26 ] && echo "    ... see $dest"
  else
    echo "  $acc: no filelist.txt"
  fi
done

note "hg38ToHg19 chain (to lift our 70,723 centers onto the Oh 2017 bigWigs)"
d="$CHAIN/hg38ToHg19.over.chain.gz"
if [ -s "$d" ] && [ -z "$(validate "$d")" ]; then
  good "hg38ToHg19.over.chain.gz" "$(stat -c%s "$d")"
else
  refetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz" "$d"
fi

note "apa_2018 clone: candidate atlas tables"
if [ -d "$ROOT/apa_2018" ]; then
  find "$ROOT/apa_2018" -type f \( -name '*.tsv' -o -name '*.txt' -o -name '*.csv' \
       -o -name '*.bed*' -o -name '*.rds' -o -name '*.RData' \) \
    -size +10k -printf '  %s\t%p\n' 2>/dev/null | sort -rn | head -25
else
  echo "  not cloned"
fi

echo
note "next"
echo "  1. bash $0 verify        # confirm everything is valid now"
echo "  2. cat $GEO/GSE135140/filelist.txt"
echo "     If it lists per-sample bedGraph or bigWig, So 2019 is usable and stays"
echo "     primary. If it is FASTQ only, Oh 2017 is the U1 arm."
echo "  3. zcat $GEO/GSE111310/GSE111310_ipa_tpm.txt.gz | head -3"
echo "     This is the quantified IpA matrix. If it carries coordinates it"
echo "     replaces the Singh 2018 supplement entirely."