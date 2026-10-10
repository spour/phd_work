#!/usr/bin/env python3
# cheap within-gene-vs-random screen for candidate features
import argparse, math
import numpy as np, pandas as pd
TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--peak_table",required=True)
    ap.add_argument("--features",nargs="+",required=True)
    ap.add_argument("--n_boot",type=int,default=1000)
    a=ap.parse_args()
    df=pd.read_parquet(a.peak_table)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"].astype(str))
    te=df[df.chrom.isin(TEST)].copy()
    def cmarg(g):
        c=g.loc[g.in_zone==1,"rf_prob"]; i=g.loc[g.in_zone==0,"rf_prob"]
        return float(c.max()-i.max()) if len(c) and len(i) else np.nan
    cm=te.groupby("gene").apply(cmarg,include_groups=False)
    contested=set(cm[cm<=0].index)
    tec=te[te.gene.isin(contested)]
    feats=[f for f in a.features if f in tec.columns]
    missing=[f for f in a.features if f not in tec.columns]
    if missing: print(f"[warn] not in table, skipped: {missing}")

    genes=[]; cand_idx=[]; canon_pos=[]; m_list=[]
    for gene,g in tec.groupby("gene",sort=False):
        iz=g.in_zone.to_numpy()
        if iz.sum()!=1: continue
        rf=g.rf_prob.to_numpy(); crf=rf[iz==1][0]
        cand=np.where((iz==1)|(rf>=crf))[0]
        genes.append(gene); cand_idx.append(cand)
        canon_pos.append(int(np.where(iz[cand]==1)[0][0])); m_list.append(len(cand))
    Gv={gene:g for gene,g in tec.groupby("gene",sort=False)}
    m_arr=np.array(m_list); null=float(np.mean(1.0/m_arr))

    def winvec(feat,sign):
        w=[]
        for gene,cand,cpos in zip(genes,cand_idx,canon_pos):
            v=Gv[gene][feat].to_numpy()[cand]*sign
            if not np.isfinite(v).any(): w.append(0); continue
            w.append(int(np.nanargmax(v)==cpos))
        return np.array(w)

    print(f"\ncontested test genes: {len(genes)}   random null (mean 1/m): {null:.3f}   "
          f"mean contest size: {m_arr.mean():.1f}")
    print(f"{'feature':<24}{'pol':>4}{'win':>8}{'win-null':>10}{'CI(win-null)':>20}{'verdict':>10}")
    rng=np.random.default_rng(0)
    rows=[]
    for f in feats:
        best=None
        for sign in (+1,-1):
            w=winvec(f,sign); wr=w.mean(); diff=wr-null
            d=[]
            nullvec=1.0/m_arr
            for _ in range(a.n_boot):
                idx=rng.integers(0,len(w),len(w))
                d.append(w[idx].mean()-nullvec[idx].mean())
            lo,hi=np.percentile(d,[2.5,97.5])
            if best is None or diff>best[1]: best=(sign,diff,wr,lo,hi)
        sign,diff,wr,lo,hi=best
        verdict="PROMISING" if lo>0 else ("dead" if hi<0.02 else "weak")
        print(f"{f:<24}{('+' if sign>0 else '-'):>4}{wr:>8.3f}{diff:>+10.3f}"
              f"   [{lo:+.3f},{hi:+.3f}]{verdict:>10}")
        rows.append({"feature":f,"polarity":int(sign),"win_rate":wr,"null":null,
                     "win_minus_null":diff,"ci_lo":lo,"ci_hi":hi,"verdict":verdict})
    pd.DataFrame(rows).to_csv("feature_screen.tsv",sep="\t",index=False)
    print("\n[out] feature_screen.tsv   (PROMISING = CI above 0; build only these into the model)")

if __name__=="__main__": main()