#!/usr/bin/env bash
# fetch external icpa datasets and lift hg19 tables to hg38

set -uo pipefail

ROOT="${EXTERNAL_ROOT:-$PWD/external}"
GEO="$ROOT/geo"; SUPP="$ROOT/supp"; CHAIN="$ROOT/chain"
BIN="$ROOT/bin"; LIFT="$ROOT/lifted"; LOG="$ROOT/manifest.tsv"
mkdir -p "$GEO" "$SUPP" "$CHAIN" "$BIN" "$LIFT"
[ -s "$LOG" ] || printf 'status\tbytes\tpath\tsource\n' > "$LOG"

note() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*" >&2; }

fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    printf 'cached\t%s\t%s\t%s\n' "$(stat -c%s "$dest")" "$dest" "$url" >> "$LOG"
    echo "  cached  $(basename "$dest")"; return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 \
          -o "$dest.part" "$url" 2>/dev/null && [ -s "$dest.part" ]; then
    mv "$dest.part" "$dest"
    printf 'ok\t%s\t%s\t%s\n' "$(stat -c%s "$dest")" "$dest" "$url" >> "$LOG"
    echo "  got     $(basename "$dest")  $(du -h "$dest" | cut -f1)"; return 0
  fi
  rm -f "$dest.part"
  printf 'FAIL\t0\t%s\t%s\n' "$dest" "$url" >> "$LOG"
  warn "failed: $url"; return 1
}

geo_series() {
  local acc="$1" pref="${1:0:6}nnn"
  local base="https://ftp.ncbi.nlm.nih.gov/geo/series/${pref}/${acc}/suppl"
  note "GEO $acc"
  mkdir -p "$GEO/$acc"
  if curl -fsSL --retry 2 --connect-timeout 20 "$base/" \
       -o "$GEO/$acc/_listing.html" 2>/dev/null; then
    grep -oE 'href="[^"]+"' "$GEO/$acc/_listing.html" \
      | sed 's/href="//;s/"//' | grep -vE '^\?|^/|^$' | sort -u \
      > "$GEO/$acc/_files.txt"
    echo "  $(wc -l < "$GEO/$acc/_files.txt") supplementary files listed:"
    sed 's/^/    /' "$GEO/$acc/_files.txt"
    while read -r f; do
      case "$f" in
        *.bed*|*.bedGraph*|*.bw|*.bigWig|*.txt.gz|*.tsv.gz|*.csv.gz|*.xlsx)
          fetch "$base/$f" "$GEO/$acc/$f" ;;
      esac
    done < "$GEO/$acc/_files.txt"
  else
    warn "no supplementary directory for $acc; trying series tarball"
    fetch "https://www.ncbi.nlm.nih.gov/geo/download/?acc=${acc}&format=file" \
          "$GEO/$acc/${acc}_RAW.tar"
  fi
}

springer_probe() {
  local doi_enc="$1" stem="$2" label="$3" nmax="${4:-15}"
  note "supplementary: $label"
  local got=0
  for n in $(seq 1 "$nmax"); do
    for ext in xlsx xls pdf txt zip; do
      local url="https://static-content.springer.com/esm/${doi_enc}/MediaObjects/${stem}_MOESM${n}_ESM.${ext}"
      local dest="$SUPP/${label}_MOESM${n}.${ext}"
      [ -s "$dest" ] && { got=$((got+1)); echo "  cached  $(basename "$dest")"; break; }
      if curl -fsSL --retry 2 --connect-timeout 20 -o "$dest.part" "$url" 2>/dev/null \
         && [ -s "$dest.part" ] \
         && ! head -c 512 "$dest.part" | grep -qiE '<html|<!doctype html'; then
        mv "$dest.part" "$dest"
        printf 'ok\t%s\t%s\t%s\n' "$(stat -c%s "$dest")" "$dest" "$url" >> "$LOG"
        echo "  got     $(basename "$dest")  $(du -h "$dest" | cut -f1)"
        got=$((got+1)); break
      fi
      rm -f "$dest.part"
    done
  done
  if [ "$got" -eq 0 ]; then
    warn "no MOESM resolved for $label; download by hand from the article page"
  else
    echo "  $got supplementary file(s) for $label"
  fi
  return 0
}

if [ "${1:-run}" = "manifest" ]; then
  command -v column >/dev/null && column -t -s$'\t' "$LOG" || cat "$LOG"
  exit 0
fi

note "target: $ROOT"

geo_series GSE135140        # So 2019, hg38, primary
geo_series GSE103252        # Oh 2017, hg19, secondary
geo_series GSE111310        # Singh 2018 + Lee 2018 3'-seq
geo_series GSE111793        # Lee 2018 RNA-seq

springer_probe '10.1038%2Fs41586-018-0465-8'  '41586_2018_465'         lee2018_nature
springer_probe '10.1038%2Fs41467-018-04112-z' '41467_2018_4112'        singh2018_natcommun
springer_probe '10.1038%2Fnsmb.3473'          '41594_2017_BFnsmb3473'  oh2017_nsmb

note "Leslie lab code and processed data"
if command -v git >/dev/null && [ ! -d "$ROOT/apa_2018" ]; then
  git clone --depth 1 https://bitbucket.org/leslielab/apa_2018.git "$ROOT/apa_2018" \
    || warn "bitbucket clone failed; browse https://bitbucket.org/leslielab/apa_2018/"
fi

note "liftOver toolchain"
fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz" \
      "$CHAIN/hg19ToHg38.over.chain.gz"
if [ ! -x "$BIN/liftOver" ]; then
  fetch "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver" "$BIN/liftOver" \
    && chmod +x "$BIN/liftOver"
fi

cat > "$ROOT/liftover_table.sh" <<'INNER'
#!/usr/bin/env bash
# liftover_table.sh IN.tsv CHROM_COL POS_COL STRAND_COL OUT.tsv
#
# Lifts a headed TSV from hg19 to hg38 by column NAME. Adds hg38_chrom and
# hg38_pos, drops rows that fail or map more than once, and writes a report
# next to the output. Positions are treated as 1-based in the input.
set -euo pipefail
IN="$1"; CC="$2"; PC="$3"; SC="$4"; OUT="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LO="$ROOT/bin/liftOver"; CH="$ROOT/chain/hg19ToHg38.over.chain.gz"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

awk -v FS='\t' -v OFS='\t' -v cc="$CC" -v pc="$PC" -v sc="$SC" '
NR==1 { for (i=1;i<=NF;i++) h[$i]=i
        if (!(cc in h) || !(pc in h) || !(sc in h)) {
          print "column not found; header is:" > "/dev/stderr"
          for (i=1;i<=NF;i++) print "  " $i > "/dev/stderr"; exit 1 }
        next }
{ c=$h[cc]; p=$h[pc]+0; s=$h[sc]
  if (c !~ /^chr/) c="chr" c
  print c, p-1, p, NR, 0, s }' "$IN" > "$TMP/in.bed"

"$LO" -minMatch=0.95 "$TMP/in.bed" "$CH" "$TMP/out.bed" "$TMP/unmapped.bed" 2>/dev/null

cut -f4 "$TMP/out.bed" | sort | uniq -d > "$TMP/dupes.txt"
awk -v FS='\t' -v OFS='\t' 'NR==FNR{d[$1];next} !($4 in d)' \
    "$TMP/dupes.txt" "$TMP/out.bed" > "$TMP/uniq.bed"

awk -v FS='\t' -v OFS='\t' '
NR==FNR { hc[$4]=$1; hp[$4]=$3; next }
FNR==1  { print $0, "hg38_chrom", "hg38_pos"; next }
(FNR in hc) { print $0, hc[FNR], hp[FNR] }' "$TMP/uniq.bed" "$IN" > "$OUT"

n_in=$(($(wc -l < "$IN") - 1)); n_out=$(($(wc -l < "$OUT") - 1))
n_dup=$(wc -l < "$TMP/dupes.txt"); n_un=$(( $(wc -l < "$TMP/unmapped.bed") / 2 ))
{ echo "input        $n_in"
  echo "lifted       $n_out"
  echo "unmapped     $n_un"
  echo "multi-mapped $n_dup  (dropped)"
  echo "failure rate $(awk -v a="$n_in" -v b="$n_out" 'BEGIN{printf "%.2f%%",100*(a-b)/a}')"
  echo
  echo "Check whether failures are class-biased before using this table."
} | tee "${OUT%.tsv}.liftover_report.txt"
INNER
chmod +x "$ROOT/liftover_table.sh"

note "done"
echo
echo "manifest:      bash $0 manifest"
echo "failures:      grep -c FAIL $LOG"
echo
echo "First thing to read, it decides Step 1 of the handoff:"
echo "  cat $GEO/GSE135140/_files.txt"
echo
echo "LiftOver usage:"
echo "  $ROOT/liftover_table.sh in.tsv chrom position strand out_hg38.tsv"