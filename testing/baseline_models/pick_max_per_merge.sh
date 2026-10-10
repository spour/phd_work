#!/bin/bash
# pick the maximum-scoring window per merged region, for each model

set -euo pipefail

DIR=./train/cryptic_sites_paper
MERGE_DIST=30   # nt; matches the per-paper definition (group adjacent above-threshold bases)

declare -A INPUT=(
    [rf]="cryptic_rf.gt0.86.bed"
    [lr]="cryptic_lr.gt0.9.bed"
    [cnn]="cryptic_cnn.gt0.87.bed"
)

for MODEL in rf lr cnn; do
    IN="$DIR/${INPUT[$MODEL]}"
    SORTED="$DIR/_tmp_${MODEL}_sorted.bed"
    ENV_BED="$DIR/_tmp_${MODEL}_env.bed"
    OUT="$DIR/cryptic_${MODEL}.rep.bed"

    echo "=== $MODEL ==="
    [[ ! -f "$IN" ]] && { echo "  WARN: $IN not found, skipping"; continue; }
    N_IN=$(wc -l < "$IN")
    echo "  input peaks: $N_IN"

    sort -k1,1 -k2,2n "$IN" > "$SORTED"

    bedtools merge -i "$SORTED" -s -d "$MERGE_DIST" -c 6 -o distinct \
        | awk 'BEGIN{OFS="\t"} {print $1, $2, $3, NR, 0, $4}' \
        > "$ENV_BED"
    N_ENV=$(wc -l < "$ENV_BED")
    echo "  merged envelopes: $N_ENV"

    bedtools intersect -a "$ENV_BED" -b "$SORTED" -wa -wb -s \
        | awk 'BEGIN{OFS="\t"}
               {
                   env_id = $4
                   score  = $11
                   if (!(env_id in best_score) || score > best_score[env_id]) {
                       best_score[env_id] = score
                       best_row[env_id]   = $7"\t"$8"\t"$9"\t"$10"\t"$11"\t"$12
                   }
               }
               END {
                   for (id in best_row) print best_row[id]
               }' \
        | sort -k1,1 -k2,2n \
        > "$OUT"

    N_OUT=$(wc -l < "$OUT")
    echo "  representatives: $N_OUT"
    printf "  reduction: %d -> %d (%.1f%%)\n" \
        "$N_IN" "$N_OUT" "$(awk "BEGIN{print $N_OUT/$N_IN*100}")"

    rm -f "$SORTED" "$ENV_BED"
done

echo
echo "Done. Outputs:"
ls -la "$DIR"/cryptic_*.rep.bed