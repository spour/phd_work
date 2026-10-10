#!/usr/bin/env python3
# cache the analysis-set columns the cnn evaluation needs
import sys
import numpy as np
from fastparquet import ParquetFile

def log(*a): print(*a, file=sys.stderr, flush=True)
COLS = ["gene", "chrom", "center", "strand", "is_canonical", "in_analysis_set",
        "gc_contrast", "spl_upstream_max_me3"]
pf = ParquetFile("cpa_peaks_final.parquet")
T = pf.to_pandas(columns=COLS)
A = T[T.in_analysis_set].reset_index(drop=True)
log(f"analysis set {len(A)} peaks: {int(A.is_canonical.sum())} real + "
    f"{int((~A.is_canonical).sum())} competitor in {A.gene.nunique()} genes")

D = np.load("cnn_data.npz", allow_pickle=True)
ok = D["ok"]
assert len(A) == len(ok), f"row count {len(A)} != cnn_data.npz {len(ok)}"
assert (A.is_canonical.values[ok] == (D["y"][ok] == 1)).all(), "label alignment"
assert (A.gene.values[ok].astype(str) == D["gene"][ok]).all(), "gene alignment"
assert np.nanmax(np.abs(A.gc_contrast.values[ok] - D["gc_contrast"][ok])) == 0, "gc_contrast"
log("[check] index-aligned with cnn_data.npz on labels, genes and gc_contrast: OK")

np.savez_compressed("analysis_columns.npz",
                    gene=A.gene.values.astype(str), chrom=A.chrom.values.astype(str),
                    center=A.center.values, strand=A.strand.values.astype(str),
                    is_canonical=A.is_canonical.values,
                    gc_contrast=A.gc_contrast.values,
                    spl_upstream_max_me3=A.spl_upstream_max_me3.values)
log(f"[out] analysis_columns.npz  ({len(A)} rows, {len(COLS)} columns)")
