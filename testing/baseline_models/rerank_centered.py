#!/usr/bin/env python3
# quick side-by-side: the same splice reranker (gbt, depth-3, monotone
import argparse, sys, math
import numpy as np, pandas as pd

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}

DIRP={"spl_u1_suppression_up":-1,"spl_u1_suppression_down":-1,"spl_bp_strength":+1,
      "spl_u2af65_ppt":+1,"spl_u2af35_ag":+1,"spl_term_3ss":+1,"spl_upstream_max_me3":+1,
      "spl_local_me5":-1,"spl_interv_me5_max":-1,"spl_dn5ss_has_strong":-1,
      "spl_pas_prox_u1":-1,"spl_upstream_mean_u1":-1,"spl_term_ppt":+1}

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def per_gene_p1(df, score_col):
    hits=[]
    for g,gg in df.groupby("gene",sort=False):
        s=gg[score_col].to_numpy().astype(float)
        top=int(np.argmax(s))
        hits.append(int(bool(gg["is_canonical"].to_numpy()[top])))
    hits=np.array(hits)
    return hits

def gene_center(df, spl):
    out=df.copy()
    gm=out.groupby("gene")[spl].transform("mean")
    out[spl]=out[spl].to_numpy()-gm.to_numpy()
    return out

def train_apply(tr, te, spl, monotone=True):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ytr=tr["is_canonical"].astype(int).to_numpy()
    mc=[int(DIRP.get(f,0)) for f in spl] if monotone else None
    clf=HistGradientBoostingClassifier(max_depth=3,max_iter=400,learning_rate=0.05,
        min_samples_leaf=50,class_weight="balanced",monotonic_cst=mc,random_state=0)
    clf.fit(tr[spl].to_numpy(), ytr)
    return clf.predict_proba(te[spl].to_numpy())[:,1]

def report(name, te, hits, summary, n_boot=0):
    n=len(hits); p,lo,hi=wilson(int(hits.sum()),n)
    S=pd.read_csv(summary,sep="\t"); S=S[S.chrom.isin(TEST_CHROMS)]
    n_unc=int((S.n_internal==0).sum()); n_tot=len(S)
    o=(p*n + 1.0*n_unc)/n_tot
    print(f"  {name:<10} contested P@1={p:.3f} [{lo:.3f},{hi:.3f}]  (n={n})   "
          f"end-to-end={o:.3f}  (n={n_tot}, {n_unc} free wins)", file=sys.stderr)
    return p,lo,hi,o

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_feat",required=True); ap.add_argument("--test_feat",required=True)
    ap.add_argument("--summary",required=True)
    ap.add_argument("--out",default="rerank_centered")
    a=ap.parse_args()

    tr=pd.read_parquet(a.train_feat); te=pd.read_parquet(a.test_feat)
    spl=[c for c in tr.columns if c.startswith("spl_")]
    for d in (tr,te): d[spl]=d[spl].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    print(f"[data] train {tr.gene.nunique():,} genes / {len(tr):,} peaks; "
          f"test {te.gene.nunique():,} genes / {len(te):,} peaks; {len(spl)} splice features",
          file=sys.stderr)

    const_within=[c for c in spl
                  if (tr.groupby("gene")[c].nunique()<=1).mean()>0.99]
    print(f"[centering] {len(const_within)} of {len(spl)} features are constant within gene "
          f"(zeroed by centering, effectively dropped): {const_within}", file=sys.stderr)

    # raw reranker
    te_raw=te.copy()
    te_raw["score"]=train_apply(tr, te, spl)
    h_raw=per_gene_p1(te_raw, "score")

    # gene-centered reranker
    tr_c=gene_center(tr, spl); te_c=gene_center(te, spl)
    te_c["score"]=train_apply(tr_c, te_c, spl)
    h_cen=per_gene_p1(te_c, "score")

    print(f"\n========== contested TEST reranker P@1: raw vs gene-centered ==========",
          file=sys.stderr)
    pr=report("raw",      te_raw, h_raw, a.summary)
    pc=report("centered", te_c,   h_cen, a.summary)

    d=h_cen.mean()-h_raw.mean()
    rng=np.random.default_rng(0); nb=[]
    for _ in range(5000):
        ix=rng.integers(0,len(h_raw),len(h_raw)); nb.append(h_cen[ix].mean()-h_raw[ix].mean())
    lo,hi=np.percentile(nb,[2.5,97.5])
    star="**" if (lo>0 or hi<0) else ""
    print(f"\n  centered - raw = {d:+.3f} [{lo:+.3f},{hi:+.3f}] {star}", file=sys.stderr)
    print(f"  (raw reproduces your standing reranker number; centered = within-gene "
          f"contrasts only, gene axes dropped)", file=sys.stderr)

    pd.DataFrame([
        dict(model="raw",      contested_p1=pr[0], ci_lo=pr[1], ci_hi=pr[2], end_to_end=pr[3]),
        dict(model="centered", contested_p1=pc[0], ci_lo=pc[1], ci_hi=pc[2], end_to_end=pc[3]),
        dict(model="centered_minus_raw", contested_p1=d, ci_lo=lo, ci_hi=hi, end_to_end=np.nan),
    ]).to_csv(f"{a.out}_metrics.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_metrics.tsv", file=sys.stderr)

if __name__=="__main__": main()