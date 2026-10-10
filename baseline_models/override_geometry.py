#!/usr/bin/env python3
# for genes where the reranker overrode the rf argmax
import argparse, sys
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scored",required=True)
    ap.add_argument("--max_kb",type=float,default=200.0,help="histogram x-range, kb")
    ap.add_argument("--out",default="override_geometry")
    a=ap.parse_args()

    df=pd.read_parquet(a.scored)
    need={"gene","center","is_canonical","rf_prob","splice_score"}
    if not need.issubset(df.columns): sys.exit(f"[err] scored file missing {need-set(df.columns)}")
    has_strand="strand" in df.columns
    sz=df.groupby("gene").size(); df=df[df.gene.isin(set(sz[sz>1].index))]

    recs=[]
    for g,gg in df.groupby("gene",sort=False):
        if not gg["is_canonical"].any(): continue
        canon_c=int(gg.loc[gg.is_canonical,"center"].iloc[0])
        strand=gg["strand"].iloc[0] if has_strand else "+"
        rf_i =int(np.argmax(gg["rf_prob"].to_numpy()))
        spl_i=int(np.argmax(gg["splice_score"].to_numpy()))
        rf_c =int(gg["center"].to_numpy()[rf_i]); spl_c=int(gg["center"].to_numpy()[spl_i])
        overrode = (rf_c!=spl_c)
        def signed(c):
            d=c-canon_c
            return d if strand=="+" else -d
        recs.append(dict(gene=g, overrode=overrode,
            rf_signed=signed(rf_c), spl_signed=signed(spl_c),
            rf_hit=bool(gg["is_canonical"].to_numpy()[rf_i]),
            spl_hit=bool(gg["is_canonical"].to_numpy()[spl_i])))
    R=pd.DataFrame(recs)
    ov=R[R.overrode].copy()
    print(f"[data] {len(R):,} contested genes; {len(ov):,} where reranker overrode RF "
          f"({len(ov)/max(1,len(R)):.1%})",file=sys.stderr)

    def summarize(name, signed, hit):
        s=signed.to_numpy().astype(float); s=s[np.isfinite(s)]
        up=float(np.mean(s<0)); dn=float(np.mean(s>0))
        print(f"  {name:<22} median |dist|={np.median(np.abs(s)):,.0f} nt  "
              f"upstream={up:.0%}  downstream={dn:.0%}  on-canonical={np.mean(hit):.0%}",file=sys.stderr)
        return s

    print(f"\n=== where picks land relative to the canonical (strand-aware, +=downstream) ===",file=sys.stderr)
    s_rf  = summarize("RF argmax pick",      ov.rf_signed,  ov.rf_hit)
    s_spl = summarize("reranker pick",       ov.spl_signed, ov.spl_hit)

    lim=a.max_kb*1000
    edges=np.linspace(-lim,lim,41)
    h_spl,_=np.histogram(np.clip(ov.spl_signed,-lim,lim),bins=edges)
    h_rf,_ =np.histogram(np.clip(ov.rf_signed, -lim,lim),bins=edges)
    centers=((edges[:-1]+edges[1:])/2/1000)
    pd.DataFrame(dict(bin_center_kb=centers, reranker=h_spl, rf=h_rf)).to_csv(
        f"{a.out}_hist.tsv",sep="\t",index=False)
    R.to_csv(f"{a.out}_pergene.tsv",sep="\t",index=False)
    pd.DataFrame([
        dict(metric="n_contested",value=len(R)),
        dict(metric="n_overrode",value=len(ov)),
        dict(metric="reranker_median_abs_nt",value=float(np.median(np.abs(s_spl)))),
        dict(metric="reranker_frac_upstream",value=float(np.mean(s_spl<0))),
        dict(metric="reranker_frac_downstream",value=float(np.mean(s_spl>0))),
    ]).to_csv(f"{a.out}_metrics.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_metrics.tsv, {a.out}_hist.tsv, {a.out}_pergene.tsv",file=sys.stderr)

if __name__=="__main__": main()