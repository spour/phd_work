#!/usr/bin/env python3
# dataset for the interpretable cnn
import sys, os, time
import numpy as np
from multiprocessing import Pool
import u1_5ss_landscape_data as L

log = L.log
UP, DN = 2000, 500
SEQLEN = UP + DN
GC_HALF = 250
PARQ = "cpa_peaks_final.parquet"
OUT = "cnn_data.npz"
TEST_CHROMS = ["chr1", "chr8", "chr15", "chr21"]
VAL_CHROMS = ["chr3", "chr18"]
CODE = np.full(256, 255, np.uint8)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): CODE[ch] = v

SITES = {}
def _init(): L.G_["G"].fh = open(L.FASTA, "rb")

def fetch_site(i):
    ch = SITES["chrom"][i]; c = int(SITES["center"][i]); st = SITES["strand"][i]
    a, b = (c - UP, c + DN) if st == "+" else (c - DN + 1, c + UP + 1)
    s = L.G_["G"].fetch(ch, a, b, st)
    if not s or len(s) != SEQLEN:
        return np.full(SEQLEN, 255, np.uint8), np.nan
    x = CODE[np.frombuffer(s.encode("ascii"), np.uint8)]
    g = L.G_["G"].fetch(ch, c - GC_HALF, c + GC_HALF, "+")
    gc = np.nan
    if g:
        t = CODE[np.frombuffer(g.encode("ascii"), np.uint8)]
        n = int((t < 4).sum())
        gc = float(((t == 1) | (t == 2)).sum()) / n if n else np.nan
    return x, gc

if __name__ == "__main__":
    import pyarrow.parquet as pq
    T = pq.read_table(PARQ).to_pandas()
    A = T[T.in_analysis_set].reset_index(drop=True)
    log(f"analysis set {len(A)} peaks: {int(A.is_canonical.sum())} real + "
        f"{int((~A.is_canonical).sum())} competitor in {A.gene.nunique()} genes")

    SITES.update(chrom=A.chrom.values, center=A.center.values, strand=A.strand.values)
    t = time.time()
    X = np.empty((len(A), SEQLEN), np.uint8); GC = np.empty(len(A))
    with Pool(min(32, os.cpu_count() or 8), initializer=_init) as p:
        for i, (x, gc) in enumerate(p.imap(fetch_site, range(len(A)), chunksize=64)):
            X[i] = x; GC[i] = gc
            if (i + 1) % 20000 == 0: log(f"  fetch {i+1}/{len(A)}  {time.time()-t:.0f}s")
    log(f"fetch done {time.time()-t:.0f}s;  N bases {float((X==255).mean()):.5f} of all positions;"
        f"  sequences all-N: {int((X==255).all(1).sum())}")

    G = L.G_["G"]
    bad = 0
    rc = str.maketrans("ACGT", "TGCA")
    for i in np.random.default_rng(0).choice(len(A), 200, replace=False):
        b = G.fetch(A.chrom[i], int(A.center[i]), int(A.center[i]) + 1, "+")
        if not b: continue
        want = b if A.strand[i] == "+" else b.translate(rc)
        if CODE[ord(want)] != X[i, UP]: bad += 1
    assert bad == 0, f"orientation check failed on {bad}/200 sites"
    log("[check] transcript-orientation offset 0 == base at center on 200 random sites: OK")

    y = A.is_canonical.values.astype(np.int8)
    gene = A.gene.values.astype(str); chrom = A.chrom.values.astype(str)
    split = np.where(np.isin(chrom, TEST_CHROMS), "test",
             np.where(np.isin(chrom, VAL_CHROMS), "val", "train"))
    hold_genes = set(A.gene[A.in_gene_holdout])
    split = np.where(np.array([g in hold_genes for g in gene]) & (split == "train"), "test", split)
    ok = np.isfinite(GC) & np.isfinite(A.gc_contrast.values) & ~(X == 255).all(1)
    log(f"usable {int(ok.sum())}/{len(A)}")
    for s in ("train", "val", "test"):
        m = (split == s) & ok
        log(f"  {s:5s} {int(m.sum()):6d} peaks  {int(y[m].sum()):5d} real  "
            f"{len(set(gene[m])):5d} genes  {len(set(chrom[m]))} chroms")
    assert not (set(gene[(split=='train')&ok]) & set(gene[(split=='test')&ok])), "gene spans train/test"
    assert not (set(gene[(split=='val')&ok]) & set(gene[(split=='test')&ok])), "gene spans val/test"
    log("[check] no gene spans splits: OK")

    np.savez_compressed(OUT, X=X, y=y, gene=gene, chrom=chrom, strand=A.strand.values.astype(str),
                        center=A.center.values, window_gc=GC, gc_contrast=A.gc_contrast.values,
                        split=split, ok=ok, loco=A.loco_chromosome.values.astype(str),
                        gene_holdout=A.in_gene_holdout.values, up=UP, dn=DN)
    log(f"[out] {OUT}  X {X.shape} uint8")
