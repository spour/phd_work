#!/usr/bin/env python3
# for each test-chrom gene, take rf's per-gene argmax
import argparse, os, sys, math
from collections import Counter, defaultdict
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}

def parse_gencode_per_gene(gtf):
    tx_exons=defaultdict(list); tx_meta={}
    for line in tqdm(open(gtf),desc="gtf"):
        if line.startswith("#"): continue
        f=line.rstrip("\n").split("\t")
        if len(f)<9 or f[2]!="exon": continue
        gn=tx=None
        for kv in f[8].split(";"):
            kv=kv.strip()
            if kv.startswith("gene_name "): gn=kv.split('"')[1]
            elif kv.startswith("transcript_id "): tx=kv.split('"')[1]
        if not gn or not tx: continue
        tx_exons[tx].append((int(f[3]),int(f[4]))); tx_meta[tx]=(f[0],f[6],gn)
    genes=defaultdict(lambda: dict(chrom=None,strand=None,exons=set(),term=[]))
    for tx,exons in tx_exons.items():
        chrom,strand,gn=tx_meta[tx]; g=genes[gn]; g["chrom"]=chrom; g["strand"]=strand
        for e in exons: g["exons"].add(e)
        te=max(exons,key=lambda e:e[1]) if strand=="+" else min(exons,key=lambda e:e[0])
        g["term"].append(te)
    def union(ivs):
        ivs=sorted(ivs); out=[list(ivs[0])]
        for s,e in ivs[1:]:
            if s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
            else: out.append([s,e])
        return [(s,e) for s,e in out]
    for gn,g in genes.items():
        g["term"]=union(g["term"]); g["exons"]=union(list(g["exons"]))
        g["span"]=(min(s for s,_ in g["exons"]), max(e for _,e in g["exons"]))
    print(f"[gtf] {len(genes):,} genes",file=sys.stderr)
    return genes

def classify_region(pos, g):
    for s,e in g["term"]:
        if s<=pos<=e: return "terminal_exon"
    for s,e in g["exons"]:
        if s<=pos<=e: return "internal_exon"
    s,e=g["span"]
    if s<=pos<=e: return "intron"
    return "outside_gene"

def load_atlas_sites(atlas):
    raw=pd.read_csv(atlas,sep="\t",header=0,dtype=str,low_memory=False)
    rc={c.lower().lstrip("#"):c for c in raw.columns}
    def col(*n):
        for x in n:
            if x in rc: return rc[x]
        return None
    rep=col("rep","repsite"); ann=col("annotation"); chrom=col("chrom")
    sym=col("gene_name"); strand=col("strand")
    d=pd.DataFrame({"chrom":raw[chrom].astype(str),"ann":raw[ann],
                    "pos":pd.to_numeric(raw[rep],errors="coerce")-1,
                    "gene":raw[sym] if sym else np.nan})
    d["chrom"]=d["chrom"].where(d["chrom"].str.startswith("chr"),"chr"+d["chrom"])
    d=d.dropna(subset=["pos"])
    by_chrom={c:(gg["pos"].to_numpy(),gg["ann"].to_numpy()) for c,gg in d.groupby("chrom")}
    print(f"[atlas] {len(d):,} sites across {len(by_chrom)} chroms",file=sys.stderr)
    return by_chrom

def nearest_atlas(chrom, pos, by_chrom, tol):
    if chrom not in by_chrom: return "none"
    P,A=by_chrom[chrom]; j=int(np.argmin(np.abs(P-pos)))
    return A[j] if abs(P[j]-pos)<=tol else "none"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--per_gene_dir",required=True)
    ap.add_argument("--gtf",required=True); ap.add_argument("--atlas",required=True)
    ap.add_argument("--pc_genes_bed",required=True)
    ap.add_argument("--win",type=int,default=500)
    ap.add_argument("--atlas_tol",type=int,default=250)
    ap.add_argument("--out",default="argmax_classify")
    a=ap.parse_args()
    off=a.win//2

    genes_gtf=parse_gencode_per_gene(a.gtf)
    atlas=load_atlas_sites(a.atlas)
    pc=pd.read_csv(a.pc_genes_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","score","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]; pc=pc.drop_duplicates("gene").set_index("gene")
    genes=[g for g in genes_gtf if g in pc.index and genes_gtf[g]["chrom"] in TEST_CHROMS]
    print(f"[genes] {len(genes):,} test-chrom genes",file=sys.stderr)

    rows=[]
    for g in tqdm(genes,desc="classify"):
        p=os.path.join(a.per_gene_dir,f"{g}.parquet")
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=["window_start_genomic","rf_prob"]).dropna(subset=["rf_prob"])
        if len(d)==0: continue
        ws=int(d["window_start_genomic"].to_numpy()[int(np.argmax(d["rf_prob"].to_numpy()))])
        gg=genes_gtf[g]; pos=ws+off
        region=classify_region(pos,gg)
        ann=nearest_atlas(gg["chrom"],pos,atlas,a.atlas_tol)
        rows.append(dict(gene=g,chrom=gg["chrom"],pos=pos,region=region,atlas_ann=ann))
    R=pd.DataFrame(rows); R.to_csv(f"{a.out}.tsv",sep="\t",index=False)
    n=len(R)
    print(f"\n=== where does RF's argmax land?  (n={n} test genes) ===",file=sys.stderr)
    print("[GENCODE region]",file=sys.stderr)
    for k,v in R["region"].value_counts().items():
        print(f"    {k:<16} {v:>5}  {v/n:.3f}",file=sys.stderr)
    print("[atlas annotation at argmax (+/- %dnt)]"%a.atlas_tol,file=sys.stderr)
    for k,v in R["atlas_ann"].value_counts().items():
        print(f"    {k:<8} {v:>5}  {v/n:.3f}",file=sys.stderr)
    miss=R[R.region!="terminal_exon"]; m=len(miss)
    print(f"\n=== of the {m} terminal-exon MISSES ({m/n:.1%} of genes) ===",file=sys.stderr)
    print("[GENCODE region of misses]",file=sys.stderr)
    for k,v in miss["region"].value_counts().items():
        print(f"    {k:<16} {v:>5}  {v/m:.3f}",file=sys.stderr)
    print("[atlas annotation of misses]",file=sys.stderr)
    for k,v in miss["atlas_ann"].value_counts().items():
        print(f"    {k:<8} {v:>5}  {v/m:.3f}",file=sys.stderr)
    intronic_real=((miss.region=="intron") & (miss.atlas_ann=="IN")).sum()
    on_any_atlas=(miss.atlas_ann!="none").sum()
    print(f"\n[headline] of misses: {on_any_atlas}/{m} ({on_any_atlas/m:.1%}) sit on a REAL atlas "
          f"polyA site; {intronic_real}/{m} ({intronic_real/m:.1%}) are intronic AND on an "
          f"atlas IN site (suppressed cryptic CPA = your phenomenon)",file=sys.stderr)
    print(f"[out] {a.out}.tsv",file=sys.stderr)

if __name__=="__main__": main()