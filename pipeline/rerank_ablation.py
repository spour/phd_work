#!/usr/bin/env python3
# feature-group ablation of the reranker on contested test genes
import argparse, sys, math
import numpy as np, pandas as pd

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}
DIRP={"spl_u1_suppression_up":-1,"spl_u1_suppression_down":-1,"spl_bp_strength":+1,
      "spl_u2af65_ppt":+1,"spl_u2af35_ag":+1,"spl_term_3ss":+1,"spl_upstream_max_me3":+1,
      "spl_local_me5":-1,"spl_interv_me5_max":-1,"spl_dn5ss_has_strong":-1,
      "spl_pas_prox_u1":-1,"spl_upstream_mean_u1":-1,"spl_term_ppt":+1,"rf_prob":+1}

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def per_gene_hits(df, score, seed=0):
    rng=np.random.default_rng(seed); hits=[]
    for g,gg in df.groupby("gene",sort=False):
        s=score[gg.index.to_numpy()]
        mx=s.max(); cand=np.flatnonzero(s>=mx-1e-12)
        top=int(rng.choice(cand))
        hits.append(int(bool(gg["is_canonical"].to_numpy()[top])))
    return np.array(hits)

def boot_diff(ha,hb,n=5000,seed=0):
    rng=np.random.default_rng(seed); ha=np.asarray(ha,float); hb=np.asarray(hb,float)
    d=[hb[ix].mean()-ha[ix].mean() for ix in (rng.integers(0,len(ha),len(ha)) for _ in range(n))]
    return float(hb.mean()-ha.mean()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

def resolve(cols, spec):
    hit=[c for c in cols if c==spec or c.startswith(spec)]
    return hit

def train_score(tr,te,feats,ytr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mc=[int(DIRP.get(f,0)) for f in feats]
    clf=HistGradientBoostingClassifier(max_depth=3,max_iter=400,learning_rate=0.05,
        min_samples_leaf=50,class_weight="balanced",monotonic_cst=mc,random_state=0)
    clf.fit(tr[feats].to_numpy(),ytr)
    return clf.predict_proba(te[feats].to_numpy())[:,1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_feat",required=True); ap.add_argument("--test_feat",required=True)
    ap.add_argument("--summary",required=True)
    ap.add_argument("--groups",default="splice:spl_,cpa:rf_prob")
    ap.add_argument("--out",default="rerank_ablation")
    a=ap.parse_args()

    tr=pd.read_parquet(a.train_feat).reset_index(drop=True)
    te=pd.read_parquet(a.test_feat).reset_index(drop=True)
    cols=list(tr.columns)
    groups={}
    for tok in a.groups.split(","):
        name,spec=tok.split(":"); g=resolve(cols,spec)
        if g: groups[name.strip()]=g
        else: print(f"[warn] group '{name}' (spec '{spec}') matched no columns",file=sys.stderr)
    allfeat=sorted(set(sum(groups.values(),[])))
    for d in (tr,te): d[allfeat]=d[allfeat].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    ytr=tr["is_canonical"].astype(int).to_numpy()
    print(f"[data] {te.gene.nunique():,} contested test genes; groups: "
          + ", ".join(f"{k}={len(v)}" for k,v in groups.items()),file=sys.stderr)

    S=pd.read_csv(a.summary,sep="\t"); S=S[S.chrom.isin(TEST_CHROMS)]
    n_unc=int((S.n_internal==0).sum()); n_tot=len(S)

    runs={}
    order=list(groups)+["combined"]
    for name in order:
        feats = allfeat if name=="combined" else groups[name]
        sc=train_score(tr,te,feats,ytr)
        h=per_gene_hits(te,sc); n=len(h); p,lo,hi=wilson(int(h.sum()),n)
        e2e=(p*n+1.0*n_unc)/n_tot
        runs[name]=dict(hits=h,p1=p,lo=lo,hi=hi,e2e=e2e,nfeat=len(feats))

    print(f"\n========== reranker P@1 by feature group (contested n={len(te.gene.unique())}) ==========",file=sys.stderr)
    base="splice" if "splice" in runs else order[0]
    rows=[]
    for name in order:
        r=runs[name]
        d=lo=hi=np.nan; star=""
        if name!=base:
            d,lo,hi=boot_diff(runs[base]["hits"],r["hits"]); star="**" if (lo>0 or hi<0) else ""
        diff=f"  vs {base}: {d:+.3f} [{lo:+.3f},{hi:+.3f}] {star}" if name!=base else "  (reference)"
        print(f"  {name:<10} P@1={r['p1']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]  "
              f"end-to-end={r['e2e']:.3f}  ({r['nfeat']} feats){diff}",file=sys.stderr)
        rows.append(dict(group=name,p1=r['p1'],ci_lo=r['lo'],ci_hi=r['hi'],end_to_end=r['e2e'],
                         n_feat=r['nfeat'],diff_vs_base=d,diff_lo=lo,diff_hi=hi))
    pd.DataFrame(rows).to_csv(f"{a.out}_metrics.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_metrics.tsv",file=sys.stderr)

if __name__=="__main__": main()