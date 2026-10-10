#!/usr/bin/env python3
# two permutation nulls for the splice reranker, on contested test genes
import argparse, sys, math
import numpy as np, pandas as pd

DIRP={"spl_u1_suppression_up":-1,"spl_u1_suppression_down":-1,"spl_bp_strength":+1,
      "spl_u2af65_ppt":+1,"spl_u2af35_ag":+1,"spl_term_3ss":+1,"spl_upstream_max_me3":+1,
      "spl_local_me5":-1,"spl_interv_me5_max":-1,"spl_dn5ss_has_strong":-1,
      "spl_pas_prox_u1":-1,"spl_upstream_mean_u1":-1,"spl_term_ppt":+1}

def gene_groups(df):
    idx=[]; iscanon=[]
    for g,gg in df.groupby("gene",sort=False):
        idx.append(gg.index.to_numpy())
        iscanon.append(gg["is_canonical"].to_numpy().astype(bool))
    return idx, iscanon

def p1_from_scores(score, idx, iscanon, perm_rng=None):
    hits=0; n=len(idx)
    for ii,ic in zip(idx, iscanon):
        s=score[ii]
        lab=ic
        if perm_rng is not None:
            lab=np.zeros(len(ic),bool); lab[perm_rng.integers(len(ic))]=True
        hits+=int(bool(lab[int(np.argmax(s))]))
    return hits/n

def train_score(tr, te, spl, ytr, monotone=True):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mc=[int(DIRP.get(f,0)) for f in spl] if monotone else None
    clf=HistGradientBoostingClassifier(max_depth=3,max_iter=400,learning_rate=0.05,
        min_samples_leaf=50,class_weight="balanced",monotonic_cst=mc,random_state=0)
    clf.fit(tr[spl].to_numpy(), ytr)
    return clf.predict_proba(te[spl].to_numpy())[:,1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_feat",required=True); ap.add_argument("--test_feat",required=True)
    ap.add_argument("--n_perm",type=int,default=1000)
    ap.add_argument("--n_retrain",type=int,default=25)
    ap.add_argument("--out",default="rerank_permnull")
    a=ap.parse_args()

    tr=pd.read_parquet(a.train_feat).reset_index(drop=True)
    te=pd.read_parquet(a.test_feat).reset_index(drop=True)
    spl=[c for c in tr.columns if c.startswith("spl_")]
    for d in (tr,te): d[spl]=d[spl].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    print(f"[data] train {tr.gene.nunique():,} genes / {len(tr):,} peaks; "
          f"test {te.gene.nunique():,} genes / {len(te):,} peaks; {len(spl)} feats",file=sys.stderr)

    te_idx, te_iscanon = gene_groups(te)
    ytr=tr["is_canonical"].astype(int).to_numpy()

    real_score=train_score(tr,te,spl,ytr)
    real_p1=p1_from_scores(real_score, te_idx, te_iscanon)
    print(f"\n[real] reranker contested P@1 = {real_p1:.3f} (n={len(te_idx)} genes)",file=sys.stderr)

    # (a) score-vs-shuffled-label null
    rng=np.random.default_rng(0)
    nullA=np.array([p1_from_scores(real_score, te_idx, te_iscanon, perm_rng=rng)
                    for _ in range(a.n_perm)])
    pA=(np.sum(nullA>=real_p1)+1)/(a.n_perm+1)
    print(f"[null A: shuffled labels] mean={nullA.mean():.3f} "
          f"[{np.percentile(nullA,2.5):.3f},{np.percentile(nullA,97.5):.3f}]  "
          f"p(real<=null)={pA:.4f}",file=sys.stderr)

    # (b) retrain-on-shuffled-train null
    nullB=[]
    for b in range(a.n_retrain):
        rb=np.random.default_rng(1000+b)
        y=np.zeros(len(tr),int)
        for ii in [g.index.to_numpy() for _,g in tr.groupby("gene",sort=False)]:
            y[ii[rb.integers(len(ii))]]=1
        sc=train_score(tr,te,spl,y,monotone=False)
        nullB.append(p1_from_scores(sc, te_idx, te_iscanon))
    nullB=np.array(nullB)
    pB=(np.sum(nullB>=real_p1)+1)/(len(nullB)+1)
    print(f"[null B: retrain on shuffled train, NO monotone prior] mean={nullB.mean():.3f} "
          f"[{nullB.min():.3f},{nullB.max():.3f}]  p(real<=null)={pB:.4f}",file=sys.stderr)

    pd.DataFrame([
        dict(metric="real_p1",value=real_p1),
        dict(metric="nullA_mean",value=nullA.mean(),lo=np.percentile(nullA,2.5),hi=np.percentile(nullA,97.5),p=pA),
        dict(metric="nullB_mean",value=nullB.mean(),lo=nullB.min(),hi=nullB.max(),p=pB),
    ]).to_csv(f"{a.out}_metrics.tsv",sep="\t",index=False)
    pd.DataFrame(dict(nullA=nullA)).to_csv(f"{a.out}_nullA.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_metrics.tsv, {a.out}_nullA.tsv",file=sys.stderr)

if __name__=="__main__": main()