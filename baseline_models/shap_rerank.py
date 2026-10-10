#!/usr/bin/env python3
# shap analysis of the splice reranker (monotone gbt)
import argparse, sys
import numpy as np, pandas as pd

DIRP={"spl_u1_suppression_up":-1,"spl_u1_suppression_down":-1,"spl_bp_strength":+1,
      "spl_u2af65_ppt":+1,"spl_u2af35_ag":+1,"spl_term_3ss":+1,"spl_upstream_max_me3":+1,
      "spl_local_me5":-1,"spl_interv_me5_max":-1,"spl_dn5ss_has_strong":-1,
      "spl_pas_prox_u1":-1,"spl_upstream_mean_u1":-1,"spl_term_ppt":+1,
      "spl_rel_3ss_up":+1,"spl_rel_3ss_dn":+1,"spl_term_3ss_abs":+1,
      "spl_comp_up_3ss":-1,"spl_comp_dn_3ss":-1,
      "spl_dist_nearest_5ss":+1,"spl_dist_up_5ss":+1,"spl_dist_dn_5ss":+1,
      "spl_n_5ss_500":-1,"spl_n_5ss_1k":-1,"spl_sum_5ss":-1,"spl_mean_top3_5ss":-1,
      "spl_dist_to_max_3ss":0}

PRETTY={
  "spl_upstream_max_me3":"strongest upstream 3' splice site",
  "spl_u1_suppression_up":"U1 / donor density (upstream)",
  "spl_pas_prox_u1":"U1 near the polyA signal",
  "spl_comp_dn_3ss":"competing 3' splice site (downstream)",
  "spl_comp_up_3ss":"competing 3' splice site (upstream)",
  "spl_interv_me5_max":"strongest intervening 5' splice site",
  "spl_term_3ss":"terminal 3' splice site strength",
  "spl_term_ppt":"terminal polypyrimidine tract",
  "spl_u2af35_ag":"U2AF35 (3' AG) strength",
  "spl_dist_to_max_3ss":"distance to strongest 3' splice site",
  "spl_sum_5ss":"total donor strength nearby",
  "spl_mean_top3_5ss":"mean of top-3 donors nearby",
  "spl_upstream_mean_u1":"mean upstream U1 density",
}
def pretty(f): return PRETTY.get(f, f.replace("spl_",""))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_feat",required=True)
    ap.add_argument("--test_feat",required=True)
    ap.add_argument("--no_monotone",action="store_true",
                    help="train WITHOUT monotone (exploration only; not the deployed model)")
    ap.add_argument("--max_explain",type=int,default=8000,
                    help="subsample test peaks for the SHAP computation (speed)")
    ap.add_argument("--top_k",type=int,default=14,help="features to show in the plots")
    ap.add_argument("--out",default="shap")
    a=ap.parse_args()

    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    tr=pd.read_parquet(a.train_feat); te=pd.read_parquet(a.test_feat)
    SPL=[c for c in tr.columns if c.startswith("spl_")]
    for d in (tr,te): d[SPL]=d[SPL].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    ytr=tr["is_canonical"].astype(int).to_numpy()
    monotone = not a.no_monotone
    mc=[int(DIRP.get(f,0)) for f in SPL] if monotone else None
    clf=HistGradientBoostingClassifier(max_depth=3,max_iter=400,learning_rate=0.05,
        min_samples_leaf=50,class_weight="balanced",monotonic_cst=mc,random_state=0)
    clf.fit(tr[SPL].to_numpy(),ytr)
    print(f"[model] {'monotone' if monotone else 'UNCONSTRAINED'} GBT, {len(SPL)} features, "
          f"trained on {len(tr):,} peaks",file=sys.stderr)

    Xte=te[SPL].copy()
    if a.max_explain>0 and len(Xte)>a.max_explain:
        Xte=Xte.sample(a.max_explain,random_state=0)
    print(f"[shap] explaining {len(Xte):,} test peaks ...",file=sys.stderr)
    expl=shap.TreeExplainer(clf)
    sv=expl.shap_values(Xte.to_numpy())
    if isinstance(sv,list): sv=sv[1] if len(sv)>1 else sv[0]
    sv=np.asarray(sv)
    print(f"[shap] shap matrix {sv.shape}",file=sys.stderr)

    # global importance: mean |shap|
    mean_abs=np.abs(sv).mean(0)
    order=np.argsort(mean_abs)[::-1]
    feats_sorted=[SPL[i] for i in order]
    imp=pd.DataFrame(dict(feature=feats_sorted,
                          mean_abs_shap=mean_abs[order],
                          monotone_dir=[DIRP.get(f,0) for f in feats_sorted]))
    imp.to_csv(f"{a.out}_importance.tsv",sep="\t",index=False)
    topf=[SPL[i] for i in order[:a.top_k]]
    longrows=[]
    Xv=Xte.to_numpy()
    for j,f in enumerate(SPL):
        if f not in topf: continue
        col=SPL.index(f)
        for r in range(len(Xte)):
            longrows.append(dict(feature=f, shap=float(sv[r,col]), value=float(Xv[r,col])))
    pd.DataFrame(longrows).to_csv(f"{a.out}_beeswarm_long.tsv",sep="\t",index=False)

    print(f"\n=== SHAP global importance (mean |SHAP|) ===",file=sys.stderr)
    for f,m in zip(feats_sorted[:a.top_k],mean_abs[order][:a.top_k]):
        bar="#"*int(round(m/mean_abs.max()*40))
        print(f"  {pretty(f):<38}{m:.4f}  {bar}",file=sys.stderr)

    # plots: beeswarm + bar (top_k)
    idx=order[:a.top_k]
    sv_top=sv[:,idx]; X_top=Xte.to_numpy()[:,idx]; names=[pretty(SPL[i]) for i in idx]
    try:
        plt.figure(figsize=(8,6))
        shap.summary_plot(sv_top, X_top, feature_names=names, show=False, max_display=a.top_k)
        plt.tight_layout(); plt.savefig(f"{a.out}_beeswarm.png",dpi=200,bbox_inches="tight")
        plt.savefig(f"{a.out}_beeswarm.pdf",bbox_inches="tight"); plt.close()
        plt.figure(figsize=(7,5))
        shap.summary_plot(sv_top, X_top, feature_names=names, plot_type="bar",
                          show=False, max_display=a.top_k)
        plt.tight_layout(); plt.savefig(f"{a.out}_bar.png",dpi=200,bbox_inches="tight"); plt.close()
        print(f"\n[out] {a.out}_beeswarm.png/.pdf, {a.out}_bar.png, "
              f"{a.out}_importance.tsv, {a.out}_beeswarm_long.tsv",file=sys.stderr)
    except Exception as e:
        print(f"[warn] plotting failed ({e}); TSVs written, plot from {a.out}_beeswarm_long.tsv",
              file=sys.stderr)

if __name__=="__main__": main()