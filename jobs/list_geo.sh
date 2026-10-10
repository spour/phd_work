#!/usr/bin/env bash
# gse103252 [gse...] -- what is actually deposited, per sample
set -uo pipefail
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'
for acc in "$@"; do
  pre="${acc:0:${#acc}-3}nnn"
  echo "== $acc"
  echo "-- series-level supplementary files"
  curl -fsSL -A "$UA" "https://ftp.ncbi.nlm.nih.gov/geo/series/${pre}/${acc}/suppl/" 2>/dev/null \
    | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//' \
    | grep -vE '^\?|^/|^$|hhs.gov' | sed 's/^/   /'
  echo "-- per-sample files (from filelist.txt of the RAW tar)"
  curl -fsSL -A "$UA" "https://ftp.ncbi.nlm.nih.gov/geo/series/${pre}/${acc}/suppl/filelist.txt" 2>/dev/null \
    | awk -F'\t' 'NR>1{printf "   %-58s %s\n", $2, $4}'
  echo "-- sample titles (dose/time/condition live here)"
  curl -fsSL -A "$UA" \
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${acc}&targ=gsm&form=text&view=brief" 2>/dev/null \
    | awk -F' = ' '/^\^SAMPLE/{g=$2} /^!Sample_title/{printf "   %-12s %s\n", g, $2}'
  echo
done