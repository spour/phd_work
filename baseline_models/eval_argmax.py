#!/usr/bin/env python3
# the ablation headline
import argparse, json, os, math
import numpy as np, pandas as pd, joblib

def load_model(d):
    clf=joblib.load(os.path.join(d,"rf_uncalibrated.joblib"))
    cols=json.load(open(os.path.join(d,"metadata.json")))["winner_meta"]["columns"]
    return clf,cols

def score(df,clf,cols):
    X=df.reindex(columns=cols).fillna(0.0).to_numpy()
    return clf.predict_proba(X)[:,1]

def p1_mrr(df,sccol):
    p1=0; rr=0.0; n=0
    for gene,g in df.groupby("gene",sort=False):
        if g["is_constitutive"].sum()==0:
            n+=1; continue
        order=np.argsort(g[sccol].to_numpy())[::-1]
        ranks=g["is_constitutive"].to_numpy()[order]
        r=int(np.argmax(ranks==1))+1
        p1+=int(r==1); rr+=1.0/r; n+=1
    return p1/n, rr/n, n

def wilson(k,n,z=1.96):
    if n==0: return (0,0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval_table",required=True)
    ap.add_argument("--baseline_dir",required=True); ap.add_argument("--augmented_dir",required=True)
    a=ap.parse_args()
    df=pd.read_parquet(a.eval_table)
    bclf,bcols=load_model(a.baseline_dir); aclf,acols=load_model(a.augmented_dir)
    df["s_base"]=score(df,bclf,bcols); df["s_aug"]=score(df,aclf,acols)

    rows=[]
    for chrom,sub in list(df.groupby("chrom"))+[("ALL",df)]:
        b_p1,b_mrr,n=p1_mrr(sub,"s_base"); a_p1,a_mrr,_=p1_mrr(sub,"s_aug")
        _,blo,bhi=wilson(round(b_p1*n),n); _,alo,ahi=wilson(round(a_p1*n),n)
        rows.append((chrom,n,b_p1,blo,bhi,a_p1,alo,ahi,a_p1-b_p1,b_mrr,a_mrr))
    print(f"{'chrom':>7}{'genes':>7}{'base_P@1':>10}{'aug_P@1':>10}{'dP@1':>8}{'base_MRR':>10}{'aug_MRR':>9}")
    for c,n,bp,blo,bhi,apv,alo,ahi,d,bm,am in rows:
        star=" *" if (alo>bhi or ahi<blo) else ""
        print(f"{c:>7}{n:>7}{bp:>10.3f}{apv:>10.3f}{d:>+8.3f}{bm:>10.3f}{am:>9.3f}{star}")
    print("\n base_P@1 should reproduce your ~0.41 on test chroms (faithful harness).")
    print(" dP@1 > 0 with CI separation (*) on ALL = splicing improves argmax. MRR is the")
    print(" softer signal (rank of the constitutive site), useful if P@1 moves little.")
    pd.DataFrame(rows,columns=["chrom","genes","base_p1","base_lo","base_hi","aug_p1",
        "aug_lo","aug_hi","delta_p1","base_mrr","aug_mrr"]).to_csv("eval_argmax.tsv",sep="\t",index=False)
if __name__=="__main__": main()