#!/usr/bin/env python3
# positional density of the four core cpa sequence elements around canonical vs competitor
import argparse, os, sys
import numpy as np, pandas as pd

PAS_HEXAMERS = ["AATAAA","ATTAAA","AGTAAA","TATAAA","CATAAA","GATAAA",
                "AATATA","AATACA","AATAGA","AAAAAG","ACTAAA","AATAAT"]

def revcomp(s):
    t=str.maketrans("ACGTacgtNn","TGCAtgcaNn"); return s.translate(t)[::-1]

def get_seq(fa, chrom, center, flank, strand):
    s = center - flank; e = center + flank
    try:
        seq = fa.fetch(chrom, max(0,s), e).upper()
    except Exception:
        return None
    if len(seq) < 2*flank: seq = seq + "N"*(2*flank-len(seq))
    if strand == "-": seq = revcomp(seq)
    return seq

def motif_hits(seq, motif):
    L=len(seq); m=len(motif); out=np.zeros(L)
    idx=seq.find(motif)
    while idx!=-1:
        out[idx]=1; idx=seq.find(motif, idx+1)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--flank", type=int, default=250)
    ap.add_argument("--smooth", type=int, default=15, help="rolling window for density smoothing")
    ap.add_argument("--max_per_class", type=int, default=40000, help="subsample for speed")
    ap.add_argument("--contested_only", action="store_true", default=True)
    ap.add_argument("--out", default="motif_density")
    a=ap.parse_args()
    import pysam
    fa=pysam.FastaFile(a.reference_fasta)

    W=pd.read_parquet(a.shortlist)
    if a.contested_only: W=W[W.n_internal>0].copy()
    if "center" not in W.columns: W["center"]=W["window_start_genomic"]+250
    W["cls"]=np.where(W.is_canonical,"canonical","competitor")
    rng=np.random.default_rng(0)
    parts=[]
    for cls,gg in W.groupby("cls"):
        if len(gg)>a.max_per_class: gg=gg.sample(a.max_per_class, random_state=0)
        parts.append(gg)
    W=pd.concat(parts, ignore_index=True)
    print(f"[data] {len(W):,} peaks ({(W.cls=='competitor').sum():,} competitor / "
          f"{(W.cls=='canonical').sum():,} canonical); flank +/-{a.flank}", file=sys.stderr)

    L=2*a.flank
    elems=["PAS","CFIm(UGUA)","U-rich","GU-rich"]
    acc={c:{e:np.zeros(L) for e in elems} for c in ["canonical","competitor"]}
    cnt={c:0 for c in ["canonical","competitor"]}
    present={c:{e:0 for e in elems} for c in ["canonical","competitor"]}
    core=slice(a.flank-50, a.flank+50)

    for _, r in W.iterrows():
        seq=get_seq(fa, r["chrom"], int(r["center"]), a.flank, r["strand"])
        if seq is None or "N"*10 in seq: continue
        c=r["cls"]; cnt[c]+=1
        pas=np.zeros(L)
        for h in PAS_HEXAMERS: pas=np.maximum(pas, motif_hits(seq,h))
        acc[c]["PAS"]+=pas
        cf=motif_hits(seq,"TGTA"); acc[c]["CFIm(UGUA)"]+=cf
        tarr=np.frombuffer(seq.encode(),dtype=np.uint8); tind=(tarr==ord("T")).astype(float)
        acc[c]["U-rich"]+=tind
        gu=motif_hits(seq,"GT"); acc[c]["GU-rich"]+=gu
        for e,arr in [("PAS",pas),("CFIm(UGUA)",cf),("U-rich",tind),("GU-rich",gu)]:
            if arr[core].sum()>0: present[c][e]+=1

    def smooth(x,w):
        if w<=1: return x
        k=np.ones(w)/w; return np.convolve(x,k,mode="same")
    rows=[]
    pos=np.arange(-a.flank, a.flank)
    for c in ["canonical","competitor"]:
        if cnt[c]==0: continue
        for e in elems:
            dens=smooth(acc[c][e]/cnt[c], a.smooth)
            for p,v in zip(pos,dens):
                rows.append(dict(cls=c, element=e, pos=int(p), density=float(v)))
    D=pd.DataFrame(rows)
    D.to_csv(f"{a.out}_density.tsv", sep="\t", index=False)

    srows=[]
    for e in elems:
        pc=present["canonical"][e]/max(1,cnt["canonical"])
        px=present["competitor"][e]/max(1,cnt["competitor"])
        srows.append(dict(element=e, canonical_present=round(pc,4),
                          competitor_present=round(px,4), diff=round(pc-px,4),
                          n_canonical=cnt["canonical"], n_competitor=cnt["competitor"]))
    S=pd.DataFrame(srows)
    S.to_csv(f"{a.out}_composition.tsv", sep="\t", index=False)
    print("\n=== core-window (+/-50nt) motif presence: canonical vs competitor ===", file=sys.stderr)
    print(S[["element","canonical_present","competitor_present","diff"]].to_string(index=False), file=sys.stderr)
    print(f"\n[interpret] near-zero diffs => canonical and competitor peaks have the same "
          f"CPA motif composition; sequence grammar cannot separate them.", file=sys.stderr)
    print(f"[out] {a.out}_density.tsv, {a.out}_composition.tsv", file=sys.stderr)

if __name__=="__main__": main()