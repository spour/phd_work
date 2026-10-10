#!/usr/bin/env python3
# compute the positioned upstream g-tract / g4 / at features
import argparse, sys, math
import numpy as np, pandas as pd
BASES=np.array(list("ACGT"))

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def onehot_to_seq(oh):
    # oh: (l,4) -> string
    idx=oh.argmax(1); has=oh.sum(1)>0
    s=np.where(has, BASES[idx], "N")
    return "".join(s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",required=True);  ap.add_argument("--test_meta",required=True)
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--out",default="gtract")
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    import splice_extra_features as XF

    rows=[]; metas=[]
    for npz,meta,split in [(a.train_npz,a.train_meta,"train"),(a.test_npz,a.test_meta,"test")]:
        X=np.load(npz)["X"]; m=pd.read_parquet(meta).reset_index(drop=True)
        for i in range(len(X)):
            seq=onehot_to_seq(X[i])
            f=XF.positioned_gtract(seq,a.upstream_nt,a.downstream_nt)
            f["split"]=split; f["is_canonical"]=bool(m["is_canonical"].iloc[i])
            f["gene"]=m["gene"].iloc[i]; f["center"]=int(m["center"].iloc[i])
            rows.append(f)
    F=pd.DataFrame(rows)
    F.to_parquet(f"{a.out}_features.parquet",index=False)

    feat_cols=["up_g_content","up_gc_content","up_max_g_run","up_g4_score",
               "comp_at_content","comp_u_richness","up_minus_comp_g"]
    print(f"\n=== positioned G-tract features: canonical vs competitor (test split) ===",file=sys.stderr)
    te=F[F.split=="test"]
    print(f"  {'feature':<18}{'canon':>8}{'comp':>8}{'cohens_d':>10}",file=sys.stderr)
    for c in feat_cols:
        ac=te[te.is_canonical][c].dropna(); co=te[~te.is_canonical][c].dropna()
        if len(ac)<10 or len(co)<10: continue
        d=(ac.mean()-co.mean())/(np.sqrt((ac.var()+co.var())/2)+1e-9)
        print(f"  {c:<18}{ac.mean():>8.3f}{co.mean():>8.3f}{d:>+10.3f}",file=sys.stderr)
    print(f"\n  positive cohens_d on up_g_content/up_max_g_run/up_g4_score => the upstream",file=sys.stderr)
    print(f"  G-tract marks true ends (CNN hypothesis confirmed); these are addable features.",file=sys.stderr)
    print(f"[out] {a.out}_features.parquet (join into the reranker on gene+center)",file=sys.stderr)

if __name__=="__main__": main()