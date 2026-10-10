#!/usr/bin/env python3
# metric: how often does the per-gene argmax
import argparse, os, sys, math, gzip
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def _open(p): return gzip.open(p,"rt") if p.endswith(".gz") else open(p)

def parse_terminal_exons(gtf):
    tx_exons={}
    tx_meta={}
    n=0
    for line in tqdm(_open(gtf),desc="gtf"):
        if line.startswith("#"): continue
        f=line.rstrip("\n").split("\t")
        if len(f)<9 or f[2]!="exon": continue
        chrom=f[0]; start=int(f[3]); end=int(f[4]); strand=f[6]; attr=f[8]
        gn=None; tx=None
        for kv in attr.split(";"):
            kv=kv.strip()
            if kv.startswith("gene_name "): gn=kv.split('"')[1]
            elif kv.startswith("transcript_id "): tx=kv.split('"')[1]
        if gn is None or tx is None: continue
        tx_exons.setdefault(tx,[]).append((start,end))
        tx_meta[tx]=(chrom,strand,gn)
        n+=1
    print(f"[gtf] {n:,} exon lines, {len(tx_exons):,} transcripts",file=sys.stderr)
    gene_te={}
    for tx,exons in tx_exons.items():
        chrom,strand,gn=tx_meta[tx]
        if strand=="+": te=max(exons,key=lambda e:e[1])
        else:           te=min(exons,key=lambda e:e[0])
        g=gene_te.setdefault(gn,{"chrom":chrom,"strand":strand,"iv":[]})
        g["iv"].append(te)
    def union(ivs):
        ivs=sorted(ivs); out=[list(ivs[0])]
        for s,e in ivs[1:]:
            if s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
            else: out.append([s,e])
        return [(s,e) for s,e in out]
    for g in gene_te: gene_te[g]["iv"]=union(gene_te[g]["iv"])
    print(f"[gtf] {len(gene_te):,} genes with terminal exons",file=sys.stderr)
    return gene_te

def gene_meta(pc_bed):
    pc=pd.read_csv(pc_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","score","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]
    return pc.drop_duplicates("gene").set_index("gene")

def in_te(pos, ivs, pad):
    for s,e in ivs:
        if s-pad<=pos<=e+pad: return True
    return False

def selfcheck(per_gene_dir, genes, tol):
    hit=0; n=0; dists=[]
    for g in tqdm(genes,desc="selfcheck"):
        p=os.path.join(per_gene_dir,f"{g}.parquet")
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=["window_start_genomic","rf_prob","is_constitutive_cpa"]).dropna(subset=["rf_prob"])
        if len(d)==0 or d["is_constitutive_cpa"].sum()==0: continue
        ws=d["window_start_genomic"].to_numpy(); rf=d["rf_prob"].to_numpy()
        amax=ws[int(np.argmax(rf))]; canon=ws[d["is_constitutive_cpa"].to_numpy().astype(bool)]
        hit+=int(np.min(np.abs(canon-amax))<=tol); n+=1
    p,lo,hi=wilson(hit,n)
    print(f"[selfcheck] constitutive P@1 = {p:.4f} [{lo:.4f},{hi:.4f}] (n={n}); "
          f"the TE-region rate MUST be >= this",file=sys.stderr)

def argmax_scan(per_gene_dir, genes):
    out={}
    for g in tqdm(genes,desc="scan argmax"):
        p=os.path.join(per_gene_dir,f"{g}.parquet")
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=["window_start_genomic","rf_prob"]).dropna(subset=["rf_prob"])
        if len(d)==0: continue
        out[g]=int(d["window_start_genomic"].to_numpy()[int(np.argmax(d["rf_prob"].to_numpy()))])
    return out

def argmax_table(eval_table, score_col):
    d=pd.read_parquet(eval_table)
    if score_col not in d.columns: sys.exit(f"[err] {score_col} not in {list(d.columns)[:8]}")
    return {g:int(gg["position"].to_numpy()[int(np.argmax(gg[score_col].to_numpy()))])
            for g,gg in d.groupby("gene",sort=False)}


def calibrate_te(per_gene_dir, gene_te, genes, center_offset):
    inside=0; n=0; sdist=[]
    for g in genes:
        p=os.path.join(per_gene_dir,f"{g}.parquet")
        if not os.path.exists(p) or g not in gene_te: continue
        d=pd.read_parquet(p,columns=["window_start_genomic","is_constitutive_cpa"])
        cc=d.loc[d["is_constitutive_cpa"]==True,"window_start_genomic"]
        if len(cc)==0: continue
        ws=int(cc.iloc[0]); pos = ws + center_offset
        ivs=gene_te[g]["iv"]; n+=1
        if in_te(pos,ivs,0): inside+=1; sdist.append(0); continue
        best=min(((pos-e) if pos>e else (pos-s) for s,e in ivs), key=abs)
        sdist.append(int(best))
    sdist=np.array(sdist)
    print(f"[calib] constitutive site inside terminal exon: {inside}/{n} = {inside/max(n,1):.3f} "
          f"(MUST be ~1.0)",file=sys.stderr)
    if len(sdist):
        miss=sdist[sdist!=0]
        print(f"[calib] signed dist of MISSES to nearest TE edge: median={np.median(miss) if len(miss) else 0:.0f} "
              f"mean={miss.mean() if len(miss) else 0:.0f} "
              f"(consistent ~+250 => use --center_offset 249; ~ -1 => off-by-one base)",file=sys.stderr)
    return inside/max(n,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["scan","table"],required=True)
    ap.add_argument("--per_gene_dir"); ap.add_argument("--eval_table"); ap.add_argument("--score_col",default="rf_prob")
    ap.add_argument("--gtf",required=True)
    ap.add_argument("--pc_genes_bed",required=True)
    ap.add_argument("--pad",type=int,default=0,help="pad terminal-exon interval by this many nt")
    ap.add_argument("--win",type=int,default=500,help="scan window length; cleavage = window_start + win/2")
    ap.add_argument("--center_offset",type=int,default=None,
                    help="scan: window_start -> cleavage shift. Default = win//2 (window center).")
    ap.add_argument("--selfcheck",action="store_true")
    ap.add_argument("--calibrate_te",action="store_true",
                    help="check that constitutive sites fall in the GTF terminal exon")
    ap.add_argument("--out",default="argmax_te_region")
    a=ap.parse_args()

    if a.center_offset is None: a.center_offset=a.win//2
    print(f"[coord] cleavage pos = window_start +/- {a.center_offset} (strand-aware)",file=sys.stderr)
    gene_te=parse_terminal_exons(a.gtf)
    pc=gene_meta(a.pc_genes_bed)
    ov=len(set(gene_te)&set(pc.index))
    print(f"[match] gtf<->pc_genes gene_name overlap: {ov:,}",file=sys.stderr)
    genes=[g for g in gene_te if g in pc.index and gene_te[g]["chrom"] in TEST_CHROMS]
    print(f"[genes] {len(genes):,} test-chrom genes with a terminal exon",file=sys.stderr)
    if len(genes)<50: print("[match] WARNING low overlap; check pc_genes name field vs gtf gene_name",file=sys.stderr)

    if a.mode=="scan":
        if not a.per_gene_dir: sys.exit("[err] --per_gene_dir required")
        if a.selfcheck: selfcheck(a.per_gene_dir,genes,250)
        if a.calibrate_te: calibrate_te(a.per_gene_dir,gene_te,genes,a.center_offset)
        amax=argmax_scan(a.per_gene_dir,genes)
    else:
        if not a.eval_table: sys.exit("[err] --eval_table required")
        amax=argmax_table(a.eval_table,a.score_col)

    rows=[]; hit=0; n=0
    for g in genes:
        if g not in amax: continue
        strand=gene_te[g]["strand"]; ws=amax[g]
        pos = ws + a.center_offset if a.mode=="scan" else ws
        h=int(in_te(pos,gene_te[g]["iv"],a.pad)); hit+=h; n+=1
        rows.append(dict(gene=g,chrom=gene_te[g]["chrom"],argmax_pos=int(pos),hit=h))
    p,lo,hi=wilson(hit,n)
    R=pd.DataFrame(rows); R.to_csv(f"{a.out}.tsv",sep="\t",index=False)
    print(f"\n[result] argmax-in-terminal-exon rate = {p:.4f} [{lo:.4f},{hi:.4f}] "
          f"(n={n} test genes; pad={a.pad})",file=sys.stderr)
    for c,gg in R.groupby("chrom"):
        pc_,lo_,hi_=wilson(int(gg.hit.sum()),len(gg))
        print(f"    {c:<7} n={len(gg):<5} rate={pc_:.3f} [{lo_:.3f},{hi_:.3f}]",file=sys.stderr)
    print(f"[out] {a.out}.tsv",file=sys.stderr)

if __name__=="__main__": main()