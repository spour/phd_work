#!/usr/bin/env python3
# figure out why threess_distance_profile produced all-nan
import argparse, sys
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--shortlist",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--maxw",type=int,default=6200)
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import GenomeFasta, init_maxentscan
    ret=init_maxentscan()
    print(f"[init] init_maxentscan returned {len(ret)} objects: "
          f"{[type(x).__name__ for x in ret]}",file=sys.stderr)
    _,score3,_,m3=ret
    fa=GenomeFasta(a.reference_fasta)

    df=pd.read_parquet(a.shortlist)
    if "n_internal" in df.columns: df=df[df.n_internal>0]
    r=df.iloc[0]
    c=int(r["center"]); strand=r["strand"]; chrom=r["chrom"]; pad=10
    if strand=="+": w0=max(0,c-a.maxw); w1=c+pad
    else:           w0=max(0,c-pad);   w1=c+a.maxw
    seq=fa.fetch(chrom,w0,w1,strand=strand)
    print(f"[fetch] {chrom}:{w0}-{w1} strand={strand}  requested_len={w1-w0}  "
          f"got_len={len(seq) if seq else None}",file=sys.stderr)
    print(f"[fetch] first 60: {seq[:60] if seq else None}",file=sys.stderr)
    print(f"[fetch] last 60:  {seq[-60:] if seq else None}",file=sys.stderr)

    if not seq: sys.exit("[err] fetch returned None")
    su=seq.upper()
    ag=[i for i in range(len(su)-1) if su[i]=="A" and su[i+1]=="G"]
    print(f"[AG] {len(ag):,} AG dinucleotides in the window (expect thousands)",file=sys.stderr)
    if not ag: sys.exit("[err] no AGs -> sequence is masked/N? print above shows it")

    p=ag[len(ag)//2]+2
    s0=p-20
    win=su[s0:p+3]
    print(f"\n[score3] testing on a 23-mer: '{win}' (len {len(win)})",file=sys.stderr)
    attempts=[
        ("score3(win, m3)",      lambda: score3(win, m3)),
        ("score3(win)",          lambda: score3(win)),
        ("score3([win], m3)",    lambda: score3([win], m3)),
        ("score3(win, m3)[0]",   lambda: score3(win, m3)[0]),
        ("m3 is None?",          lambda: ("m3="+str(m3))),
    ]
    for name,fn in attempts:
        try:
            v=fn()
        except Exception as e:
            print(f"  ERR {name:<22} -> {type(e).__name__}: {e}",file=sys.stderr)

    print(f"\n[note] MaxEntScan score3 expects a 23nt acceptor window. If the working "
          f"call above differs from 'score3(win, m3)', that mismatch is the all-NaN cause.",
          file=sys.stderr)

if __name__=="__main__": main()