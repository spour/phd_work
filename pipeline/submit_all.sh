#!/bin/bash
# (run from pub/) queues the whole overnight training: featurize -> rf/lr (after featurize)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

[ -d splits/prod ] || { echo "[err] run scripts/make_prod_split.sh first"; exit 1; }
[ -s splits/prod/pos_train.fa ] || { echo "[err] run scripts/extract_fastas.sh first"; exit 1; }

FEAT=$(sbatch --parsable scripts/featurize_all.slurm)
echo "featurize array : $FEAT"

RF=$(sbatch --parsable --dependency=aftercorr:$FEAT scripts/train_rf_array.slurm)
echo "RF array        : $RF   (after matching featurize task)"

LR=$(sbatch --parsable --dependency=aftercorr:$FEAT scripts/train_lr_array.slurm)
echo "LR array        : $LR   (after matching featurize task)"

CNN=$(sbatch --parsable scripts/train_cnn_array.slurm)
echo "CNN array       : $CNN  (independent, reads FASTA)"

echo
echo "Submitted. Overnight plan:"
echo "  featurize 6 splits -> RF (6) + LR (6) once features ready; CNN (6) in parallel."
echo "  Models land in models/{fold0..fold4,prod}/{rf,lr,cnn}/"
echo "Check in the morning: sacct -j $FEAT,$RF,$LR,$CNN --format=JobID,JobName,State,Elapsed"