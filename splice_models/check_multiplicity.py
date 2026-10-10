#!/usr/bin/env python3
# is the true end occupying multiple in_zone windows in the candidate set
import argparse, numpy as np, pandas as pd
TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}
ap=argparse.ArgumentParser(); ap.add_argument("--peak_table",required=True); a=ap.parse_args()
df=pd.read_parquet(a.peak_table)
df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"].astype(str))
te=df[df.chrom.isin(TEST)]
n_iz=[]; set_sz=[]; canon_in_set=[]; randp=[]
for gene,g in te.groupby("gene",sort=False):
    iz=g.in_zone.to_numpy()
    if iz.sum()==0: continue
    rf=g.rf_prob.to_numpy()
    n_iz.append(int(iz.sum()))
    crf=rf[iz==1].max(); cand=np.where((iz==1)|(rf>=crf))[0]
    set_sz.append(len(cand))
    cic=int((iz[cand]==1).sum())
    canon_in_set.append(cic)
    randp.append(cic/len(cand))
n_iz=np.array(n_iz); set_sz=np.array(set_sz); cic=np.array(canon_in_set); randp=np.array(randp)
print(f"genes: {len(n_iz)}")
print(f"in_zone windows per gene:        mean={n_iz.mean():.2f} median={np.median(n_iz):.0f} max={n_iz.max()} frac>1={np.mean(n_iz>1):.3f}")
print(f"in_zone members inside contest:  mean={cic.mean():.2f} median={np.median(cic):.0f} frac>1={np.mean(cic>1):.3f}")
print(f"contest set size m:              mean={set_sz.mean():.2f} median={np.median(set_sz):.0f}")
print(f"random hit prob (cic/m):         mean={randp.mean():.3f}  <- this is the 'random tiebreak' number")
print(f"if true end were ONE ticket (1/m): mean={np.mean(1.0/set_sz):.3f}  <- what random SHOULD be")
print(f"\nfrac of genes with >1 in_zone window in the set: {np.mean(cic>1):.3f}")
print("if that frac is high, the true end is multiple lottery tickets -> random inflated above 1/m")