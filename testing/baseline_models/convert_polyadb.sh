#!/usr/bin/env bash
# convert polya_db v4 tsv -> bed, auto-detecting max vs main layout
set -euo pipefail
IN="$1"; OUT="$2"

HEADER=$(head -1 "$IN")
RPM_COL=$(echo "$HEADER" | tr '\t' '\n' | grep -n -x "AvgRPM.3READS" | cut -d: -f1 || true)
[ -z "${RPM_COL:-}" ] && RPM_COL=$(echo "$HEADER" | tr '\t' '\n' | grep -n -x "AvgRPM" | cut -d: -f1 || true)
PSE_COL=$(echo "$HEADER" | tr '\t' '\n' | grep -n -x "PSE.3READS" | cut -d: -f1 || true)
[ -z "${PSE_COL:-}" ] && PSE_COL=$(echo "$HEADER" | tr '\t' '\n' | grep -n -x "PSE" | cut -d: -f1 || true)

if [ -z "${RPM_COL:-}" ]; then echo "ERROR: no AvgRPM column found"; exit 1; fi
echo "RPM -> column $RPM_COL   PSE -> column ${PSE_COL:-none}"

tail -n +2 "$IN" | awk -F'\t' -v rc="$RPM_COL" -v pc="${PSE_COL:-0}" 'BEGIN{OFS="\t"}
{
  split($1, a, ":");
  chrom=a[1]; strand=a[2]; pos=a[3]+0;
  rpm=$rc; if (rpm=="NA" || rpm=="") rpm=0;
  pse=(pc>0 ? $pc : "NA"); if (pse=="NA" || pse=="") pse=0;
  print chrom, pos-1, pos, $1, rpm, strand, pse
}' | sort -k1,1 -k2,2n > "$OUT"

echo "wrote $OUT  ($(wc -l < "$OUT") sites)"
echo "sample:"; head -3 "$OUT"
echo "RPM>0 count:"; awk '$5>0{n++} END{print n}' "$OUT"