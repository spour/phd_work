#!/usr/bin/env bash
# build a size-matched, randomized, intragenic, non-3'utr random control bed

PC_GENES_BED="${1:-${PC_GENES_BED:-data/pc_genes.bed}}"
THREE_PRIME_BED="${2:-${THREE_PRIME_BED:-data/annotation/three_prime_zone.bed}}"
GENOME_FILE="${3:-${GENOME_FILE:-}}"
TARGET_N="${4:-${TARGET_N:-0}}"
MATCH_BED="${5:-${MATCH_BED:-}}"
OUT_BED="${6:-${OUT_BED:-data/random_intragenic.bed}}"
WIN_SIZE="${WIN_SIZE:-500}"
OVERSAMPLE="${OVERSAMPLE:-10}"
SEED="${SEED:-42}"

if [[ -z "$GENOME_FILE" || ! -f "$GENOME_FILE" ]]; then
    echo "ERROR: GENOME_FILE missing or unset. Must be a bedtools chrom.sizes file."
    echo "  Produce with: samtools faidx hg38.fa && cut -f1,2 hg38.fa.fai > hg38.chrom.sizes"
    exit 1
fi

if [[ "$TARGET_N" == "0" ]]; then
    if [[ -z "$MATCH_BED" || ! -f "$MATCH_BED" ]]; then
        echo "ERROR: TARGET_N=0 requires MATCH_BED to match by line count"
        exit 1
    fi
    TARGET_N=$(grep -v '^#' "$MATCH_BED" | grep -vc '^$' || true)
    echo "  matched line count from $MATCH_BED: $TARGET_N"
fi

DRAW_N=$((TARGET_N * OVERSAMPLE))
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
echo "  draw $DRAW_N candidate windows (target $TARGET_N, oversample ${OVERSAMPLE}x)"
echo "  window size: $WIN_SIZE nt"
echo "  seed:        $SEED"
echo "  intermediate dir: $TMPDIR"

mkdir -p "$(dirname "$OUT_BED")"

sort -k1,1 -k2,2n "$PC_GENES_BED" > "$TMPDIR/pc.sorted.bed"

bedtools random -l "$WIN_SIZE" -n "$DRAW_N" -seed "$SEED" \
    -g "$GENOME_FILE" \
    > "$TMPDIR/random.raw.bed"
echo "  drew $(wc -l < $TMPDIR/random.raw.bed) raw windows"

sort -k1,1 -k2,2n "$TMPDIR/random.raw.bed" > "$TMPDIR/random.sorted.bed"
bedtools intersect -u -s \
    -a "$TMPDIR/random.sorted.bed" \
    -b "$TMPDIR/pc.sorted.bed" \
    > "$TMPDIR/random.intragenic.bed"
echo "  after pc_genes filter (strand-aware): $(wc -l < $TMPDIR/random.intragenic.bed)"

sort -k1,1 -k2,2n "$THREE_PRIME_BED" > "$TMPDIR/three_prime.sorted.bed"
bedtools intersect -v -s \
    -a "$TMPDIR/random.intragenic.bed" \
    -b "$TMPDIR/three_prime.sorted.bed" \
    > "$TMPDIR/random.intragenic.no3utr.bed"
N_AFTER_3UTR=$(wc -l < "$TMPDIR/random.intragenic.no3utr.bed")
echo "  after 3'UTR-zone subtraction: $N_AFTER_3UTR"

if (( N_AFTER_3UTR < TARGET_N )); then
    echo "  WARNING: only $N_AFTER_3UTR candidates after filtering, "
    echo "           but TARGET_N=$TARGET_N. Increase OVERSAMPLE and re-run."
    echo "           Using all $N_AFTER_3UTR available."
    TARGET_N=$N_AFTER_3UTR
fi

yes "$SEED" | head -c 2M > "$TMPDIR/seed_file" 2>/dev/null || true
shuf --random-source="$TMPDIR/seed_file" \
    "$TMPDIR/random.intragenic.no3utr.bed" \
    | head -n "$TARGET_N" \
    > "$TMPDIR/random.subset.bed"
echo "  shuffled and subset to $(wc -l < $TMPDIR/random.subset.bed) sites"

sort -k1,1 -k2,2n "$TMPDIR/random.subset.bed" \
    | awk -v OFS='\t' '{print $1, $2, $3, "random_"NR, "0", $6}' \
    > "$OUT_BED"

echo ""
echo "  wrote $OUT_BED ($(wc -l < $OUT_BED) lines)"
echo ""
echo "  sanity:"
echo "    window widths (should all be $WIN_SIZE):"
awk '{print $3-$2}' "$OUT_BED" | sort -u | head -5
echo "    strand split:"
awk -F'\t' '{print $6}' "$OUT_BED" | sort | uniq -c
echo "    chrom distribution (top 10):"
cut -f1 "$OUT_BED" | sort | uniq -c | sort -rn | head -10