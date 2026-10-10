#!/usr/bin/env python3
# 95th percentile (and full distribution) of human terminal-exon length
import argparse, sys, re
import numpy as np, pandas as pd

def attr(s, key):
    m=re.search(key+r' "([^"]+)"', s)
    return m.group(1) if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gtf",required=True)
    ap.add_argument("--out",default="terminal_exon_length")
    a=ap.parse_args()

    tx_exons={}
    tx_meta={}
    cds_len={}
    n=0
    with open(a.gtf) as fh:
        for line in fh:
            if line.startswith("#"): continue
            f=line.rstrip("\n").split("\t")
            if len(f)<9: continue
            feat=f[2]
            if feat not in ("exon","CDS"): continue
            attrs=f[8]
            ttype=attr(attrs,"transcript_type") or attr(attrs,"transcript_biotype")
            if ttype!="protein_coding": continue
            tx=attr(attrs,"transcript_id")
            if tx is None: continue
            start=int(f[3]); end=int(f[4]); strand=f[6]
            if feat=="CDS":
                cds_len[tx]=cds_len.get(tx,0)+(end-start+1)
                continue
            tx_exons.setdefault(tx,[]).append((start,end))
            if tx not in tx_meta:
                is_mane = ("MANE_Select" in attrs) or ("MANE Select" in attrs) or (attr(attrs,"tag")=="MANE_Select") or ("tag \"MANE_Select\"" in attrs)
                tx_meta[tx]=dict(gene=attr(attrs,"gene_id"), gname=attr(attrs,"gene_name"),
                                 strand=strand, is_mane=("MANE_Select" in attrs))
            n+=1
            if n%2_000_000==0: print(f"[gtf] {n:,} exon lines", file=sys.stderr)

    rows=[]
    for tx,exs in tx_exons.items():
        m=tx_meta[tx]; strand=m["strand"]
        exs=sorted(exs)
        term = exs[-1] if strand=="+" else exs[0]
        tlen = term[1]-term[0]+1
        txlen = sum(e-s+1 for s,e in exs)
        rows.append(dict(tx=tx, gene=m["gene"], gname=m["gname"], strand=strand,
                         is_mane=m["is_mane"], term_exon_len=tlen,
                         cds_len=cds_len.get(tx,0), tx_len=txlen))
    T=pd.DataFrame(rows)
    print(f"[tx] {len(T):,} protein-coding transcripts over {T.gene.nunique():,} genes",file=sys.stderr)

    T["rank_key"]=list(zip(T["is_mane"].astype(int), T["cds_len"], T["tx_len"]))
    pick=T.sort_values("rank_key").groupby("gene").tail(1).copy()
    mane_frac=pick["is_mane"].mean()
    print(f"[pick] one tx/gene: {len(pick):,} genes ({mane_frac:.0%} via MANE Select)",file=sys.stderr)

    for label, D in [("ALL protein-coding transcripts", T),
                     ("one transcript per gene (MANE/longest)", pick)]:
        L=D["term_exon_len"].to_numpy()
        qs=np.percentile(L,[50,75,90,95,99])
        print(f"\n=== terminal-exon length: {label} (n={len(L):,}) ===",file=sys.stderr)
        print(f"  mean={L.mean():.0f}  median={qs[0]:.0f}  q75={qs[1]:.0f}  "
              f"q90={qs[2]:.0f}  q95={qs[3]:.0f}  q99={qs[4]:.0f}  max={L.max():,.0f}",file=sys.stderr)

    L95=float(np.percentile(pick["term_exon_len"],95))
    print(f"\n>>> L95 (one-tx-per-gene 95th pct terminal-exon length) = {L95:,.0f} nt <<<",file=sys.stderr)

    pick[["gene","gname","strand","term_exon_len","cds_len","tx_len","is_mane"]].to_csv(
        f"{a.out}_pergene.tsv",sep="\t",index=False)
    pd.DataFrame([dict(scope="one_tx_per_gene", n=len(pick),
        q50=float(np.percentile(pick.term_exon_len,50)),
        q90=float(np.percentile(pick.term_exon_len,90)),
        q95=L95, q99=float(np.percentile(pick.term_exon_len,99)),
        mane_frac=mane_frac)]).to_csv(f"{a.out}_summary.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_summary.tsv, {a.out}_pergene.tsv",file=sys.stderr)

if __name__=="__main__": main()