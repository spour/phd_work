#!/usr/bin/env python3
# build the used-vs-suppressed contrast for motif discovery, the right scientific question
import argparse, sys
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

def load_gene_spans(gtf):
    genes={}; te={}
    exons={}
    with open(gtf) as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            f=ln.split("\t")
            if len(f)<9: continue
            if f[2] not in ("gene","exon"): continue
            attrs=f[8]
            gname=None
            for kv in attrs.split(";"):
                kv=kv.strip()
                if kv.startswith("gene_name"):
                    gname=kv.split('"')[1]; break
            if gname is None: continue
            if f[2]=="gene":
                genes[gname]=(f[0],f[6],int(f[3]),int(f[4]))
            else:
                exons.setdefault(gname,[]).append((int(f[3]),int(f[4])))
    for g,(chrom,strand,gs,ge) in genes.items():
        ex=sorted(exons.get(g,[]))
        if not ex: continue
        last = ex[-1] if strand=="+" else ex[0]
        te[g]=(chrom,strand,last[0],last[1])
    return genes, te

def compartment(row, te_by_gene, gene_of):
    g=gene_of(row["name"])
    if g and g in te_by_gene:
        chrom,strand,ts,tend=te_by_gene[g]
        c=row["center"]
        if ts-200<=c<=tend+200: return "TE"
        return "INTERNAL"
    return "UNKNOWN"

def gene_from_name(name):
    parts=name.split("_")
    return parts[-1] if parts else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--used",required=True); ap.add_argument("--cryptic",required=True)
    ap.add_argument("--gtf",required=True)
    ap.add_argument("--mode",choices=["unmatched","matched"],default="unmatched")
    ap.add_argument("--win",type=int,default=500,help="bed window width (center = mid)")
    ap.add_argument("--out",default="supp_labels")
    a=ap.parse_args()

    used=load_bed(a.used); used["label"]=1
    cryp=load_bed(a.cryptic); cryp["label"]=0
    for df in (used,cryp): df["center"]=((df.start+df.end)//2).astype(int)
    print(f"[load] used {len(used):,}, cryptic {len(cryp):,}",file=sys.stderr)

    print(f"[gtf] parsing gene/terminal-exon spans...",file=sys.stderr)
    genes,te=load_gene_spans(a.gtf)
    print(f"[gtf] {len(genes):,} genes, {len(te):,} terminal exons",file=sys.stderr)

    both=pd.concat([used,cryp],ignore_index=True)
    both["gene"]=both["name"].map(gene_from_name)
    both["compartment"]=both.apply(lambda r: compartment(r,te,gene_from_name),axis=1)
    print(f"[compartment] used:",file=sys.stderr)
    print(both[both.label==1].compartment.value_counts().to_string(),file=sys.stderr)
    print(f"[compartment] cryptic:",file=sys.stderr)
    print(both[both.label==0].compartment.value_counts().to_string(),file=sys.stderr)

    if a.mode=="matched":
        sub=both[both.compartment=="INTERNAL"].copy()
        nu=(sub.label==1).sum(); nc=(sub.label==0).sum()
        print(f"[matched] INTERNAL-only: used {nu:,} vs cryptic {nc:,}",file=sys.stderr)
        if nu<50:
            print(f"[matched][WARN] very few INTERNAL used sites ({nu}); matched contrast is "
                  f"underpowered. Consider gene-matched negatives instead.",file=sys.stderr)
        out=sub
    else:
        out=both

    out=out[["chrom","start","end","name","strand","center","label","gene","compartment"]]
    out.to_parquet(f"{a.out}.parquet",index=False)
    print(f"[out] {a.out}.parquet  (n={len(out):,}, pos={int((out.label==1).sum()):,}, "
          f"neg={int((out.label==0).sum()):,})",file=sys.stderr)

if __name__=="__main__": main()