#!/usr/bin/env python3
# complete random-forest coverage for the shared-gene scope
import sys, csv, gzip, os, time
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
DENSE = ("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome/"
         "out_scores/per_gene_dir")
WIN = 500

SH = {r["gene"] for r in csv.DictReader(open("overlap_shared_genes.tsv"), delimiter="\t")}
log(f"[shared] {len(SH):,} genes")
SITES = [r for r in csv.DictReader(open("overlap_scores_sites.tsv"), delimiter="\t")
         if r["in_shared"] == "1"]
log(f"[sites] {len(SITES):,} sites in those genes")
BY = {}
for r in SITES: BY.setdefault(r["gene"], []).append(r)

FILE = {}
for ch in sorted(os.listdir(DENSE)):
    d = os.path.join(DENSE, ch)
    if not os.path.isdir(d): continue
    for fn in os.listdir(d):
        if fn.endswith(".bed6.gz"): FILE[fn[:-8]] = os.path.join(d, fn)
log(f"[scan] {len(FILE):,} per-gene scans on disk; {len(SH & set(FILE)):,} of the "
    f"{len(SH):,} shared genes are present")

res = {}; t0 = time.time(); miss = 0
for i, (g, rows) in enumerate(sorted(BY.items())):
    p = FILE.get(g)
    if p is None:
        miss += len(rows); continue
    cen, cal = [], []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) < 7: continue
            cen.append((int(f[1]) + int(f[2])) // 2); cal.append(float(f[4]))
    if not cen: continue
    cen = np.asarray(cen); cal = np.asarray(cal)
    o = np.argsort(cen); cen = cen[o]; cal = cal[o]
    for r in rows:
        pos = int(r["pos"])
        lo = np.searchsorted(cen, pos - WIN, "left"); hi = np.searchsorted(cen, pos + WIN, "right")
        res[(r["set"], r["chrom"], r["pos"])] = (
            float(cal[lo:hi].max()) if hi > lo else np.nan, int(hi - lo))
    if (i + 1) % 60 == 0:
        log(f"[dense] {i+1}/{len(BY)} genes, {time.time()-t0:.0f}s")
log(f"[dense] done, {time.time()-t0:.0f}s; {miss:,} sites in genes with no scan on disk")

with open("overlap_rf_dense.tsv", "w") as fh:
    fh.write("set\tchrom\tpos\trf_dense_max\tn_dense_windows\n")
    for (s, c, p), (v, n) in sorted(res.items()):
        fh.write(f"{s}\t{c}\t{p}\t{'' if not np.isfinite(v) else f'{v:.5f}'}\t{n}\n")
log(f"[out] overlap_rf_dense.tsv ({len(res):,} sites)")
SETS = sorted({s for s, _, _ in res})
log(f"\n{'set':<12} {'n':>6} {'scored':>7} {'median':>8} {'mean':>7} {'q25':>7} {'q75':>7}")
for s in SETS:
    v = np.array([res[k][0] for k in res if k[0] == s], float)
    ok = v[np.isfinite(v)]
    log(f"{s:<12} {len(v):>6,} {len(ok):>7,} {np.median(ok):>8.4f} {ok.mean():>7.4f} "
        f"{np.percentile(ok,25):>7.4f} {np.percentile(ok,75):>7.4f}")
