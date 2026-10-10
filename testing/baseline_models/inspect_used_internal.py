#!/usr/bin/env python3
# show which dominant/used sites are not in a terminal exon
import argparse, sys, time
from collections import defaultdict
import numpy as np, pandas as pd

def load_bed(path):
    rows=[]
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip(): continue
            f=ln.rstrip("\n").split("\t")
            rows.append(dict(chrom=f[0],start=int(f[1]),end=int(f[2]),
                             name=f[3] if len(f)>3 else ".",
                             strand=f[5] if len(f)>5 else "+"))
    return pd.DataFrame(rows)

def parse_gencode(gtf):
    gene_span={}; tx_exons=defaultdict(list)
    nl=0; t0=time.time()
    with open(gtf) as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            f=ln.split("\t")
            if len(f)<9 or f[2] not in ("gene","exon"): continue
            name=None; tid=None
            for kv in f[8].split(";"):
                kv=kv.strip()
                if kv.startswith("gene_name"): name=kv.split('"')[1]
                elif kv.startswith("transcript_id"): tid=kv.split('"')[1]
            if name is None: continue
            if f[2]=="gene": gene_span[name]=[f[0],f[6],int(f[3]),int(f[4])]
            else: tx_exons[(name,tid)].append((int(f[3]),int(f[4])))
            nl+=1
            if nl%1000000==0: print(f"[gtf] {nl:,} lines {(time.time()-t0)/60:.1f}m",file=sys.stderr)
    term_exons=defaultdict(list); internal_exons=defaultdict(list); all_exons=defaultdict(list)
    tx_ends=defaultdict(list)
    by_gene=defaultdict(list)
    for (name,tid),ex in tx_exons.items(): by_gene[name].append(ex)
    for name,txs in by_gene.items():
        if name not in gene_span: continue
        strand=gene_span[name][1]
        for ex in txs:
            ex=sorted(ex)
            all_exons[name].extend(ex)
            last = ex[-1] if strand=="+" else ex[0]
            term_exons[name].append(last)
            tx_ends[name].append(last[1] if strand=="+" else last[0])
            for k,e in enumerate(ex):
                is_term = (e==ex[-1]) if strand=="+" else (e==ex[0])
                if not is_term: internal_exons[name].append(e)
    print(f"[gtf] {len(gene_span):,} genes  {(time.time()-t0)/60:.1f}m",file=sys.stderr)
    return gene_span,term_exons,internal_exons,all_exons,tx_ends

def in_any(spans, c, pad=0):
    for (a,b) in spans:
        if a-pad<=c<=b+pad: return True
    return False

def dist_to_nearest_end(ends, c):
    if not ends: return np.nan
    return float(min(abs(c-e) for e in ends))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--used",required=True); ap.add_argument("--gtf",required=True)
    ap.add_argument("--te_pad",type=int,default=200,help="how close to a terminal exon counts as TE")
    ap.add_argument("--out",default="used_internal")
    a=ap.parse_args()
    df=load_bed(a.used); df["center"]=((df.start+df.end)//2).astype(int)
    df["gene"]=df["name"].str.split("_").str[-1]
    print(f"[data] {len(df):,} used sites",file=sys.stderr)
    gene_span,term_exons,internal_exons,all_exons,tx_ends=parse_gencode(a.gtf)

    out=[]
    for r in df.itertuples(index=False):
        g=r.gene; c=int(r.center)
        d=dict(in_terminal_exon=False,in_internal_exon=False,in_exon_any=False,
               dist_to_tx_end=np.nan,compartment="UNKNOWN",gene_matched=(g in gene_span))
        if g in gene_span:
            d["in_terminal_exon"]=in_any(term_exons.get(g,[]),c,a.te_pad)
            d["in_internal_exon"]=in_any(internal_exons.get(g,[]),c)
            d["in_exon_any"]=in_any(all_exons.get(g,[]),c)
            d["dist_to_tx_end"]=dist_to_nearest_end(tx_ends.get(g,[]),c)
            if d["in_terminal_exon"]: d["compartment"]="TE"
            elif d["in_internal_exon"]: d["compartment"]="INTERNAL_EXON"
            elif d["in_exon_any"]: d["compartment"]="EXON_OTHER"
            else: d["compartment"]="INTRON"
        out.append(d)
    O=pd.DataFrame(out); res=pd.concat([df.reset_index(drop=True),O],axis=1)
    res.to_parquet(f"{a.out}.parquet",index=False)
    res.to_csv(f"{a.out}.tsv",sep="\t",index=False)

    print(f"\n=== compartment breakdown of USED sites (recomputed from GENCODE) ===",file=sys.stderr)
    print(res["compartment"].value_counts().to_string(),file=sys.stderr)
    nonterm=res[~res["in_terminal_exon"] & res["gene_matched"]]
    print(f"\n=== USED sites NOT in a terminal exon: {len(nonterm):,} ===",file=sys.stderr)
    print(nonterm["compartment"].value_counts().to_string(),file=sys.stderr)
    print(f"\n  of these, how many are in an INTERNAL EXON (likely real internal APA) "
          f"vs INTRON (intronic PA) vs EXON_OTHER:",file=sys.stderr)
    nt=nonterm.copy()
    q=nt["dist_to_tx_end"].quantile([0,.25,.5,.75,1.0])
    print(f"\n  dist_to_nearest_transcript_end (small = near SOME isoform's end -> maybe a",file=sys.stderr)
    print(f"  shorter-isoform terminal exon, not truly internal):",file=sys.stderr)
    print(f"    min {q.iloc[0]:.0f}  25% {q.iloc[1]:.0f}  median {q.iloc[2]:.0f}  "
          f"75% {q.iloc[3]:.0f}  max {q.iloc[4]:.0f}",file=sys.stderr)
    near=(nt["dist_to_tx_end"]<200).sum()
    print(f"    {near:,}/{len(nt):,} are within 200nt of SOME transcript's annotated end "
          f"(could be terminal exons of shorter isoforms, not real internal sites)",file=sys.stderr)
    print(f"\n[out] {a.out}.tsv (all used sites + compartment + diagnostics)",file=sys.stderr)
    print(f"      filter to compartment != 'TE' to see the non-terminal ones.",file=sys.stderr)

if __name__=="__main__": main()