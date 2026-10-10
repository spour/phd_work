#!/usr/bin/env python3
# plain rf-argmax p@1 for the cpa model
import argparse, os, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

TEST_CHROMS = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS  = {"chr3","chr13"}
LEN_BINS = [0, 5000, 20000, 50000, 100000, 500000, 5e8]
LEN_LABELS = ["<5kb", "5-20kb", "20-50kb", "50-100kb", "100-500kb", ">500kb"]

def split_of(chrom):
    c = chrom if str(chrom).startswith("chr") else "chr"+str(chrom)
    if c in TEST_CHROMS: return "test"
    if c in VAL_CHROMS:  return "val"
    return "train"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gene_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--limit_genes", type=int, default=0)
    ap.add_argument("--out", default="rf_baseline_pergene.tsv")
    a = ap.parse_args()

    pc = pd.read_csv(a.pc_genes_bed, sep="\t", header=None,
                     names=["chrom","start","end","name","score","strand"])
    pc["gene"] = pc["name"].astype(str).str.split("_").str[-1]
    pc = pc.drop_duplicates("gene"); pc["gene_length"] = pc["end"] - pc["start"]
    meta = pc.set_index("gene")

    genes = sorted(f[:-8] for f in os.listdir(a.per_gene_dir) if f.endswith(".parquet"))
    if a.limit_genes > 0:
        genes = genes[:a.limit_genes]

    rows = []
    drop = dict(no_meta=0, read_fail=0, empty=0, no_canon=0)
    for gene in tqdm(genes, desc="genes", unit="gene"):
        if gene not in meta.index:
            drop["no_meta"] += 1; continue
        fp = os.path.join(a.per_gene_dir, f"{gene}.parquet")
        try:
            g = pd.read_parquet(fp, columns=["window_start_genomic","rf_prob","is_constitutive_cpa"])
        except Exception:
            drop["read_fail"] += 1; continue
        g = g.dropna(subset=["rf_prob"])
        if len(g) == 0:
            drop["empty"] += 1; continue
        canon = g.loc[g["is_constitutive_cpa"] == True, "window_start_genomic"].to_numpy()
        if len(canon) == 0:
            drop["no_canon"] += 1; continue
        pos = g["window_start_genomic"].to_numpy(); rf = g["rf_prob"].to_numpy()
        amax = int(pos[int(np.argmax(rf))])
        hit = int(np.min(np.abs(canon - amax)) <= a.tolerance_nt)
        gm = meta.loc[gene]
        rows.append((gene, str(gm["chrom"]), split_of(gm["chrom"]),
                     int(gm["gene_length"]), len(g), hit))

    R = pd.DataFrame(rows, columns=["gene","chrom","split","gene_length","n_windows","rf_hit"])
    R.to_csv(a.out, sep="\t", index=False)

    print(f"\n[drops] {drop}   scorable genes = {len(R):,}", file=sys.stderr)
    print(f"\n[RF-argmax P@1 from raw scans, by split]")
    print(f"  {'split':<8}{'genes':>8}{'P@1':>8}")
    for s in ["train", "val", "test"]:
        sub = R[R.split == s]
        if len(sub) == 0:
            continue
        print(f"  {s:<8}{len(sub):>8}{sub.rf_hit.mean():>8.3f}")
    print(f"  {'ALL':<8}{len(R):>8}{R.rf_hit.mean():>8.3f}")

    tr = R[R.split == "train"]; te = R[R.split == "test"]
    if len(tr) and len(te):
        gap = tr.rf_hit.mean() - te.rf_hit.mean()
        verdict = ("large gap: RF saw the train genes (or test chroms are harder); "
                   "use TEST as the honest baseline, and the TEST contested fraction for biology"
                   if gap > 0.05 else
                   "small gap: RF generalizes; the genome-wide baseline is usable")
        print(f"\n  train - test gap = {gap:+.3f}   ({verdict})")

    if len(te):
        te = te.copy(); te["lb"] = pd.cut(te.gene_length, LEN_BINS, labels=LEN_LABELS)
        print(f"\n[TEST RF P@1 by gene length, the honest baseline by length]")
        print(f"  {'length':<10}{'genes':>7}{'P@1':>8}")
        for b in LEN_LABELS:
            gb = te[te.lb == b]
            if len(gb) == 0:
                continue
            print(f"  {b:<10}{len(gb):>7}{gb.rf_hit.mean():>8.3f}")
    print(f"\n[out] {a.out}", file=sys.stderr)

if __name__ == "__main__":
    main()