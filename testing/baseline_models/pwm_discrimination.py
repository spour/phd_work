#!/usr/bin/env python3
# for each pwm the cnn learned (from a .meme file)
import argparse, sys, re
import numpy as np, pandas as pd
B={"A":0,"C":1,"G":2,"T":3}

def auc_mw(s,y):
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank(method="average").to_numpy()
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def parse_meme(path):
    pwms={}; txt=open(path).read()
    for b in re.split(r"\nMOTIF ",txt)[1:]:
        name=b.split()[0]; rows=[]
        for ln in b.splitlines():
            nums=re.findall(r"[0-9]*\.?[0-9]+(?:e-?\d+)?",ln)
            v=[float(x) for x in nums]
            if len(v)==4 and 0.9<sum(v)<1.1: rows.append(v)
        if len(rows)>=4: pwms[name]=np.array(rows)
    return pwms

def pwm_logodds(P):
    return np.log2(np.clip(P,1e-4,1)/0.25)

def scan_max(X, lom):
    N,L,_=X.shape; w=lom.shape[0]
    scores=np.full(N,-1e9,np.float32); pos=np.zeros(N,np.int32)
    for start in range(0,L-w+1):
        seg=X[:,start:start+w,:]
        sc=np.einsum("nwb,wb->n",seg,lom)
        upd=sc>scores
        scores[upd]=sc[upd]; pos[upd]=start
    return scores, pos

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--meme",required=True); ap.add_argument("--npz",required=True); ap.add_argument("--meta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000)
    ap.add_argument("--out",default="pwm_disc")
    a=ap.parse_args()
    X=np.load(a.npz)["X"].astype(np.float32); m=pd.read_parquet(a.meta).reset_index(drop=True)
    y=m["label"].to_numpy().astype(int); cl=a.upstream_nt
    has_comp = "compartment" in m.columns
    internal = (m["compartment"]=="INTERNAL").to_numpy() if has_comp else np.ones(len(m),bool)
    pwms=parse_meme(a.meme)
    print(f"[data] {len(X):,} sites ({int((y==1).sum())} used, {int((y==0).sum())} supp), "
          f"{len(pwms)} PWMs",file=sys.stderr)
    rows=[]
    for name,P in pwms.items():
        lom=pwm_logodds(P)
        sc,ps=scan_max(X,lom)
        auc=auc_mw(sc,y)
        mk=internal & (~np.isnan(sc))
        auc_int=auc_mw(sc[mk],y[mk]) if mk.sum()>50 and len(np.unique(y[mk]))==2 else np.nan
        medpos_used=int(np.median((ps[y==1]-cl))) if (y==1).any() else 0
        medpos_supp=int(np.median((ps[y==0]-cl))) if (y==0).any() else 0
        cons="".join("ACGT"[i] for i in P.argmax(1))
        rows.append(dict(motif=name,consensus=cons,auc=auc,auc_internal=auc_int,
                         abs_auc=abs(auc-0.5),abs_auc_int=abs(auc_int-0.5) if not np.isnan(auc_int) else np.nan,
                         medpos_used=medpos_used,medpos_supp=medpos_supp))
    R=pd.DataFrame(rows).sort_values("abs_auc",ascending=False)
    R.to_csv(f"{a.out}.tsv",sep="\t",index=False)
    print(f"\n=== PWM discrimination (used vs suppressed), sorted by |AUC-0.5| ===",file=sys.stderr)
    print(f"  {'consensus':>16} {'AUC':>6} {'AUC_int':>8} {'pos_used':>9} {'pos_supp':>9}",file=sys.stderr)
    for _,r in R.iterrows():
        ai=f"{r.auc_internal:.3f}" if not np.isnan(r.auc_internal) else "  n/a"
        print(f"  {r.consensus:>16} {r.auc:>6.3f} {ai:>8} {int(r.medpos_used):>9} {int(r.medpos_supp):>9}",file=sys.stderr)
    print(f"\n  AUC>0.5 => motif stronger on USED; <0.5 => stronger on SUPPRESSED.",file=sys.stderr)
    print(f"  AUC_int near 0.5 => that motif's discrimination was COMPARTMENT, not mechanism.",file=sys.stderr)
    print(f"  AUC_int still away from 0.5 => differentiates beyond compartment (real).",file=sys.stderr)
    print(f"[out] {a.out}.tsv",file=sys.stderr)

if __name__=="__main__": main()
