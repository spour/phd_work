#!/usr/bin/env python3
# is the forced-canonical window needlessly low-rf?
import argparse
import numpy as np, pandas as pd
TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--peak_table",required=True)
    ap.add_argument("--tol",type=int,default=250)
    a=ap.parse_args()
    df=pd.read_parquet(a.peak_table)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"].astype(str))
    te=df[df.chrom.isin(TEST)]
    n=0; q1=0; q2=0; rf_gap=[]; near=[]
    for gene,g in te.groupby("gene",sort=False):
        iz=g.in_zone.to_numpy()
        if iz.sum()==0: continue
        n+=1
        rf=g.rf_prob.to_numpy(); pos=g.position.to_numpy()
        canon_rf=rf[iz==1].max()
        int_mask=iz==0
        if int_mask.sum()==0: continue
        top_int_rf=rf[int_mask].max()
        if top_int_rf>=canon_rf:
            q1+=1; rf_gap.append(top_int_rf-canon_rf)
            canon_pos=pos[iz==1]
            ti=np.where(int_mask)[0][np.argmax(rf[int_mask])]
            d=np.min(np.abs(pos[ti]-canon_pos))
            near.append(d)
            if d<=2*a.tol: q2+=1
    print(f"scorable test genes: {n}")
    print(f"Q1  top internal RF >= top in_zone RF (genuinely contested): {q1} ({q1/n:.3f})")
    print(f"Q2  of those, winning internal peak is within {2*a.tol}nt of the true end")
    print(f"    (likely the true end's OWN peak, mislabelled internal): {q2} ({q2/max(q1,1):.3f} of contested)")
    if rf_gap:
        rf_gap=np.array(rf_gap); near=np.array(near)
        print(f"\nRF gap (top_internal - canonical) on contested genes:")
        print(f"   median={np.median(rf_gap):.3f}  mean={rf_gap.mean():.3f}  "
              f"frac<0.02 (near-tie)={np.mean(rf_gap<0.02):.3f}")
        print(f"distance from winning internal peak to nearest true-end window:")
        print(f"   median={np.median(near):.0f}nt  frac<=500nt={np.mean(near<=500):.3f}  "
              f"frac<=1000nt={np.mean(near<=1000):.3f}")
        print(f"\nREAD: if a large fraction of 'contested' genes have the winning internal")
        print(f"peak within ~500nt of the true end, those are the true end's own RF peak")
        print(f"labelled as a competitor (tolerance too tight) -> inflates contested count")
        print(f"and makes 'reranking' a contest against itself. Fix: widen in_zone tolerance")
        print(f"or merge near-duplicate peaks before labelling.")

if __name__=="__main__": main()