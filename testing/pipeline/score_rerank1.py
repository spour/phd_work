#!/usr/bin/env python3
# (step 3, corrected for the consensus shortlist) within-shortlist reranking on contested
import argparse, sys, math
import numpy as np, pandas as pd
TEST_CHROMS = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS  = {"chr3","chr13"}
CPA_SCORES  = ["rf_prob","lr_prob","cnn_prob"]
DIRP = {"spl_u1_suppression_up":-1,"spl_u1_suppression_down":-1,"spl_bp_strength":+1,
        "spl_u2af65_ppt":+1,"spl_u2af35_ag":+1,"spl_term_3ss":+1,"spl_upstream_max_me3":+1,
        "spl_local_me5":-1,"spl_interv_me5_max":-1,"spl_dn5ss_has_strong":-1,
        "spl_pas_prox_u1":-1,"spl_upstream_mean_u1":-1,"spl_term_ppt":+1,
        "spl_rel_3ss_up":+1,"spl_rel_3ss_dn":+1,"spl_term_3ss_abs":+1,
        "spl_comp_up_3ss":-1,"spl_comp_dn_3ss":-1,
        "spl_dist_nearest_5ss":+1,"spl_dist_up_5ss":+1,"spl_dist_dn_5ss":+1,
        "spl_dist_interv_me5":+1,
        "spl_n_5ss_500":-1,"spl_n_5ss_1k":-1,"spl_sum_5ss":-1,"spl_mean_top3_5ss":-1,
        "spl_site_has_donor":-1,
        "spl_acceptor_minus_dndonor":+1,
        "spl_bp_to_3ss_dist":0,
        "spl_ppt_length":+1,
        "spl_dn_u1_decay_far":-1}
def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)
def fit_model(model, Xtr, ytr, feats, monotone):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if model=="gbt":
        mc=[int(DIRP.get(f,0)) for f in feats] if monotone else None
        clf=HistGradientBoostingClassifier(max_depth=3,max_iter=400,learning_rate=0.05,
            min_samples_leaf=50,class_weight="balanced",monotonic_cst=mc,random_state=0)
        clf.fit(Xtr,ytr); return lambda X: clf.predict_proba(X)[:,1]
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(Xtr),ytr)
    if monotone:
        co=clf.coef_[0].copy(); mcv=np.array([DIRP.get(f,0) for f in feats],float)
        co[(mcv!=0)&(np.sign(co)!=np.sign(mcv))]=0.0; clf.coef_[0]=co
    return lambda X: clf.predict_proba(sc.transform(X))[:,1]
def p1_from_scores(df, score_col, rng=None):
    if rng is None: rng=np.random.default_rng(0)
    g=df["gene"].to_numpy()
    s=df[score_col].to_numpy().astype(float)
    yc=df["is_canonical"].to_numpy()
    order=np.argsort(g, kind="stable")
    g=g[order]; s=s[order]; yc=yc[order]
    rank=s + rng.random(len(s))*1e-9
    bnd=np.flatnonzero(g[1:]!=g[:-1])+1
    starts=np.concatenate([[0], bnd]); ends=np.concatenate([bnd, [len(g)]])
    hits=np.empty(len(starts), dtype=int)
    for i,(a_,b_) in enumerate(zip(starts,ends)):
        hits[i]=int(bool(yc[a_+int(np.argmax(rank[a_:b_]))]))
    return hits
def random_expected(df):
    return np.array([1.0/len(gg) for _,gg in df.groupby("gene",sort=False)])
def boot_diff(ha,hb,n_boot=5000,seed=0):
    rng=np.random.default_rng(seed); ha=np.asarray(ha,float); hb=np.asarray(hb,float)
    n=len(ha); d=[hb[ix].mean()-ha[ix].mean() for ix in (rng.integers(0,n,n) for _ in range(n_boot))]
    return float(hb.mean()-ha.mean()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))
def loco_p1(pool, feats, model, monotone, rng, return_scored=False):
    hh=[]; ff=[]; scored=[]
    for c in sorted(pool.chrom.unique()):
        trc=pool[pool.chrom!=c]; tec=pool[pool.chrom==c]
        if tec.gene.nunique()==0 or trc["is_canonical"].sum()==0: continue
        pred=fit_model(model,trc[feats].to_numpy(),trc["is_canonical"].astype(int).to_numpy(),feats,monotone)
        d2=tec.copy(); d2["_s"]=pred(tec[feats].to_numpy())
        hh.append(p1_from_scores(d2,"_s",rng)); ff.append(random_expected(tec))
        if return_scored:
            keep=[k for k in ("gene","chrom","center","strand","is_canonical","n_internal") if k in d2.columns]
            scored.append(d2[keep+["_s"]].rename(columns={"_s":"splice_score"}))
    H=np.concatenate(hh); F=np.concatenate(ff)
    out=(H,F)
    if return_scored: out=(H,F,pd.concat(scored,ignore_index=True) if scored else None)
    return out
# fast permutation-null workers (module scope so processpool can pickle them)
_PN = {}
def _pn_init(Xtr, Xte, tr, te, starts, gsz, gs, ge, floor, mc, model, iters=150):
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    _PN.update(Xtr=Xtr, Xte=Xte, tr=tr, te=te, starts=starts, gsz=gsz,
               gs=gs, ge=ge, floor=floor, mc=mc, model=model, N=len(tr), iter=iters)
def _pn_boost(labels_full):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ytr = labels_full[_PN["tr"]].astype(int)
    if _PN["model"] == "gbt":
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=_PN.get("iter", 150),
              learning_rate=0.08, min_samples_leaf=50, class_weight="balanced",
              monotonic_cst=_PN["mc"], random_state=0)
        clf.fit(_PN["Xtr"], ytr); s = clf.predict_proba(_PN["Xte"])[:, 1]
    else:
        sc = StandardScaler().fit(_PN["Xtr"])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(_PN["Xtr"]), ytr)
        s = clf.predict_proba(sc.transform(_PN["Xte"]))[:, 1]
    isc = labels_full[_PN["te"]]
    gs, ge = _PN["gs"], _PN["ge"]
    hits = 0.0
    for a_, b_ in zip(gs, ge):
        hits += isc[a_ + int(np.argmax(s[a_:b_]))]
    return hits / len(gs) - _PN["floor"]
def _pn_null(seed):
    rr = np.random.default_rng(seed)
    lab = np.zeros(_PN["N"], bool)
    lab[_PN["starts"] + rr.integers(0, _PN["gsz"])] = True
    return _pn_boost(lab)
FEATURE_GROUPS={
    "acceptor":["spl_upstream_max_me3","spl_term_3ss","spl_term_3ss_abs","spl_term_ppt",
                "spl_dist_to_max_3ss","spl_comp_up_3ss","spl_comp_dn_3ss","spl_rel_3ss_up",
                "spl_rel_3ss_dn","spl_u2af65_ppt","spl_u2af35_ag","spl_bp_strength"],
    "donor":["spl_interv_me5_max","spl_local_me5","spl_dn5ss_has_strong","spl_sum_5ss",
             "spl_mean_top3_5ss","spl_n_5ss_500","spl_n_5ss_1k","spl_dist_nearest_5ss",
             "spl_dist_up_5ss","spl_dist_dn_5ss","spl_dist_interv_me5"],
    "u1":["spl_upstream_mean_u1","spl_u1_suppression_up","spl_u1_suppression_down","spl_pas_prox_u1"],
}
def permutation_importance(df, feats, predict, base_p1, n_rep=3, seed=0):
    rng=np.random.default_rng(seed); out=[]
    X=df[feats].to_numpy()
    base=df[["gene","is_canonical"]].copy()
    for j,f in enumerate(feats):
        drops=[]
        col=X[:,j].copy()
        for _ in range(n_rep):
            Xp=X.copy(); Xp[:,j]=rng.permutation(col)
            base["_s"]=predict(Xp)
            drops.append(base_p1 - p1_from_scores(base,"_s").mean())
        out.append((f,float(np.mean(drops)),float(np.std(drops))))
    return sorted(out,key=lambda x:-x[1])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat", default=None, help="single featurized parquet (all chroms)")
    ap.add_argument("--train_feat", default=None); ap.add_argument("--test_feat", default=None)
    ap.add_argument("--summary", required=True, help="shortlist_crossfit_summary.tsv")
    ap.add_argument("--model", default="gbt", choices=["gbt","logistic"])
    ap.add_argument("--monotone", action="store_true")
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--select", action="store_true",
                    help="build a parsimonious final model: rank features by permutation "
                         "importance, keep the smallest set whose LOCO boost is within "
                         "--select_tol of the full-model boost, refit, and report it "
                         "-> rerank_selected_features.tsv, rerank_selection_path.tsv")
    ap.add_argument("--select_tol", type=float, default=0.005,
                    help="how much LOCO boost you'll give up for parsimony (default 0.005 P@1)")
    ap.add_argument("--select_full_path", action="store_true",
                    help="evaluate the whole 1..K feature path (for the parsimony curve) "
                         "instead of stopping once within --select_tol (slower)")
    ap.add_argument("--pdp", action="store_true",
                    help="partial dependence of P(canonical) on the top features -> rerank_pdp.tsv")
    ap.add_argument("--pdp_k", type=int, default=4, help="how many top features to profile for --pdp")
    ap.add_argument("--examples", type=int, default=0,
                    help="emit peaks for N illustrative genes where the reranker corrected a "
                         "cryptic RF pick (for a worked-example track) -> rerank_examples.tsv")
    ap.add_argument("--gene_split_check", action="store_true")
    ap.add_argument("--capacity_sweep", action="store_true",
                    help="train vs test P@1 across model sizes (over/underfit diagnostic)")
    ap.add_argument("--fixed_split_check", action="store_true",
                    help="also report the legacy chr18-22,X split as a consistency check")
    ap.add_argument("--ablation", action="store_true",
                    help="LOCO P@1 for feature groups (acceptor / donor / U1 / all)")
    ap.add_argument("--permnull", type=int, default=0,
                    help="N label-permutation nulls (fixed-split, 1 fit each). 200 is plenty "
                         "for p<0.005; use 1000 only if you want a very smooth tail.")
    ap.add_argument("--permnull_iters", type=int, default=150,
                    help="boosting iters per null fit (default 150; the null needs less "
                         "convergence than the headline model, so this is the main speed knob)")
    ap.add_argument("--permnull_backend", default="thread", choices=["thread","process","serial"],
                    help="parallelism for the null: 'thread' (default; no pickling, robust), "
                         "'process' (fastest but can hang on some clusters), 'serial' (safe fallback)")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="parallel worker processes (permutation null + heavy fits)")
    ap.add_argument("--by_contest", action="store_true",
                    help="LOCO P@1 stratified by number of competitors in the gene")
    ap.add_argument("--out", default="rerank")
    ap.add_argument("--shap", action="store_true",
                    help="per-peak SHAP on the pooled reranker -> rerank_shap_long.tsv")
    ap.add_argument("--violins", action="store_true",
                    help="per-feature value distributions by class -> rerank_violins_long/_stats.tsv")
    ap.add_argument("--override_geom", action="store_true",
                    help="RF-pick vs splice-pick distance + outcome -> rerank_override_hist/_pergene.tsv")
    ap.add_argument("--atlas", default=None,
                    help="polyA atlas BED; overrides landing on real APA sites -> rerank_atlas.tsv")
    ap.add_argument("--atlas_usage_col", type=int, default=4,
                    help="0-indexed BED column holding site usage/expression for the usage-weighted "
                         "panel (PolyASite: 4=avg expression, 7=n supporting samples). Default 4.")
    ap.add_argument("--shap_max", type=int, default=8000,
                    help="cap peaks per class in the SHAP dump (speed)")
    ap.add_argument("--winrate", action="store_true",
                    help="per-feature: fraction of genes where the canonical peak has the "
                         "within-gene extreme value (max, or min for distances) -> rerank_winrate.tsv")
    a=ap.parse_args()
    parts=[]
    for f in (a.feat, a.train_feat, a.test_feat):
        if f: parts.append(pd.read_parquet(f))
    if not parts: sys.exit("[err] pass --feat or --train_feat/--test_feat")
    pool=pd.concat(parts, ignore_index=True).drop_duplicates(subset=["gene","center"])
    SPLICE_WHITELIST = [
        "spl_upstream_mean_u1","spl_upstream_max_me3","spl_term_3ss","spl_term_ppt",
        "spl_local_me5","spl_pas_prox_u1","spl_dn5ss_has_strong","spl_interv_me5_max",
        "spl_dist_interv_me5",
        "spl_bp_strength","spl_u2af65_ppt","spl_u2af35_ag","spl_u1_suppression_up",
        "spl_u1_suppression_down","spl_dist_to_max_3ss",
        "spl_term_3ss_abs","spl_comp_up_3ss","spl_comp_dn_3ss","spl_rel_3ss_up","spl_rel_3ss_dn",
        "spl_dist_nearest_5ss","spl_dist_up_5ss","spl_dist_dn_5ss",
        "spl_n_5ss_500","spl_n_5ss_1k","spl_sum_5ss","spl_mean_top3_5ss",
    ]
    SPL=[c for c in SPLICE_WHITELIST if c in pool.columns]
    missing=[c for c in SPLICE_WHITELIST if c not in pool.columns]
    if missing:
        print(f"[warn] whitelist features absent from parquet: {missing}", file=sys.stderr)
    pool[SPL]=pool[SPL].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    pool["is_canonical"]=pool["is_canonical"].astype(bool)
    print(f"[data] {pool.gene.nunique():,} contested genes / {len(pool):,} peaks / "
          f"{len(SPL)} splice features", file=sys.stderr)
    unconstrained=[f for f in SPL if f not in DIRP]
    if unconstrained:
        print(f"[monotone] unconstrained by design: {unconstrained}", file=sys.stderr)
    rng=np.random.default_rng(0)
    e_rnd_all=random_expected(pool); floor_all=float(e_rnd_all.mean())
    VAL=[]
    deg_rows=[]
    print(f"\n===== within-shortlist ranking by each CPA score (contested, n={pool.gene.nunique()}) =====",
          file=sys.stderr)
    for c in CPA_SCORES:
        if c not in pool.columns: continue
        h=p1_from_scores(pool,c,rng); p,lo,hi=wilson(int(h.sum()),len(h))
        deg_rows.append(dict(score=c,p1=round(p,4),ci_lo=round(lo,4),ci_hi=round(hi,4)))
        print(f"  {c:<10} P@1={p:.4f} [{lo:.4f},{hi:.4f}]", file=sys.stderr)
    deg_rows.append(dict(score="random_floor",p1=round(floor_all,4),ci_lo=np.nan,ci_hi=np.nan))
    print(f"  {'random':<10} P@1={floor_all:.4f}  (floor = mean 1/n)", file=sys.stderr)
    print("  -> the canonical sits at the bottom of its own gene under every CPA score,"
          "\n     by the consensus definition. Random is the only honest baseline.", file=sys.stderr)
    pd.DataFrame(deg_rows).to_csv(f"{a.out}_degeneracy.tsv",sep="\t",index=False)
    chroms=sorted(pool.chrom.unique())
    all_h=[]; all_f=[]; per_chrom=[]; gw=[]
    for c in chroms:
        trc=pool[pool.chrom!=c]; tec=pool[pool.chrom==c]
        if tec.gene.nunique()==0 or trc["is_canonical"].sum()==0: continue
        pred=fit_model(a.model,trc[SPL].to_numpy(),trc["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
        d2=tec.copy(); d2["splice_score"]=pred(tec[SPL].to_numpy())
        h=p1_from_scores(d2,"splice_score",rng); f=random_expected(tec)
        all_h.append(h); all_f.append(f)
        pr,plo,phi=wilson(int(h.sum()),len(h))
        per_chrom.append(dict(chrom=c,n=len(h),p1=pr,ci_lo=plo,ci_hi=phi,floor=float(f.mean()),
                              boost=pr-float(f.mean())))
        keep=[k for k in ("gene","chrom","center","strand","rf_prob","lr_prob","cnn_prob",
                          "is_canonical","n_internal") if k in d2.columns]
        gw.append(d2[keep+["splice_score"]])
    H=np.concatenate(all_h); F=np.concatenate(all_f)
    P,LO,HI=wilson(int(H.sum()),len(H)); floor=float(F.mean())
    d_sr,lo_sr,hi_sr=boot_diff(F,H)
    print(f"\n===== HEADLINE: genome-wide LOCO (contested genes, n={len(H):,}) =====", file=sys.stderr)
    print(f"  splice reranker P@1 = {P:.3f} [{LO:.3f},{HI:.3f}]", file=sys.stderr)
    print(f"  random floor        = {floor:.3f}", file=sys.stderr)
    print(f"  splice - random     = {d_sr:+.3f} [{lo_sr:+.3f},{hi_sr:+.3f}]"
          f"{' **' if lo_sr>0 else ''}   <- the splicing contribution", file=sys.stderr)
    VAL.append(dict(scheme="LOCO (chromosome)",n=len(H),p1=P,ci_lo=LO,ci_hi=HI,
                    floor=floor,boost=P-floor,boost_lo=lo_sr,boost_hi=hi_sr))
    pd.concat(gw,ignore_index=True).to_parquet(f"{a.out}_genomewide_scored.parquet",index=False)
    pc=pd.DataFrame(per_chrom); pc.to_csv(f"{a.out}_genomewide_bychrom.tsv",sep="\t",index=False)
    # recall@k: is the canonical in the top-k under the reranker?
    GWr = pd.concat(gw, ignore_index=True)
    has_rf = "rf_prob" in GWr.columns
    rk_spl=[]; rk_rf=[]; ns_rk=[]
    for _g, d in GWr.groupby("gene", sort=False):
        isc = d["is_canonical"].to_numpy(); ci = np.flatnonzero(isc)
        if len(ci)==0: continue
        ci=ci[0]; ns_rk.append(len(d))
        s=d["splice_score"].to_numpy()
        rk_spl.append((int((s>s[ci]).sum()), int((s==s[ci]).sum())))
        if has_rf:
            r=d["rf_prob"].to_numpy(); rk_rf.append((int((r>r[ci]).sum()), int((r==r[ci]).sum())))
    ns_rk=np.array(ns_rk)
    def _recall_at(pairs, k):
        v=[0.0 if k<=gt else 1.0 if k>=gt+eq else (k-gt)/eq for gt,eq in pairs]
        return float(np.mean(v)) if v else np.nan
    rk_rows=[]
    for k in range(1,6):
        row=dict(k=k, recall_splice=_recall_at(rk_spl,k),
                 recall_random=float(np.mean(np.minimum(k,ns_rk)/ns_rk)))
        if has_rf: row["recall_rf"]=_recall_at(rk_rf,k)
        rk_rows.append(row)
    pd.DataFrame(rk_rows).to_csv(f"{a.out}_recall_at_k.tsv", sep="\t", index=False)
    # calibration: are the reranker's p(canonical) honest?
    sc = GWr["splice_score"].to_numpy().astype(float)
    yc = GWr["is_canonical"].to_numpy().astype(int)
    edges = np.linspace(0.0, 1.0, 11)
    bidx = np.clip(np.digitize(sc, edges) - 1, 0, len(edges) - 2)
    cal_rows = []; ece = 0.0; Ntot = len(sc)
    for b in range(len(edges) - 1):
        m = bidx == b
        if m.sum() == 0: continue
        pmean = float(sc[m].mean()); ofrac = float(yc[m].mean()); nb = int(m.sum())
        _, lo, hi = wilson(int(yc[m].sum()), nb)
        cal_rows.append(dict(bin_mid=float((edges[b] + edges[b+1]) / 2),
                             pred_mean=pmean, obs_frac=ofrac, n=nb, ci_lo=lo, ci_hi=hi))
        ece += (nb / Ntot) * abs(pmean - ofrac)
    pd.DataFrame(cal_rows).to_csv(f"{a.out}_calibration.tsv", sep="\t", index=False)
    pd.DataFrame([dict(metric="ece", value=float(ece)),
                  dict(metric="n_peaks", value=int(Ntot))]).to_csv(f"{a.out}_calibration_metrics.tsv", sep="\t", index=False)
    # score margin: how decisively is the canonical ranked?
    mar_rows = []
    for _g, d in GWr.groupby("gene", sort=False):
        isc = d["is_canonical"].to_numpy(); ci = np.flatnonzero(isc)
        if len(ci) == 0 or len(d) < 2: continue
        s = d["splice_score"].to_numpy(); cval = float(s[ci[0]])
        comp = np.delete(s, ci[0])
        mar_rows.append(dict(gene=_g, margin=cval - float(np.max(comp)),
                             correct=bool(cval > float(np.max(comp))), n=int(len(d))))
    pd.DataFrame(mar_rows).to_csv(f"{a.out}_margin.tsv", sep="\t", index=False)
    def grp(c): return "CPA-heldout" if c in TEST_CHROMS else ("CPA-val" if c in VAL_CHROMS else "CPA-trained")
    pc["cpa_group"]=pc.chrom.map(grp)
    print(f"\n  [boost by whether the CPA models saw the chromosome]", file=sys.stderr)
    for g,gg in pc.groupby("cpa_group"):
        w=gg.n/gg.n.sum()
        print(f"    {g:<14} n={int(gg.n.sum()):<6} P@1={float((gg.p1*w).sum()):.3f} "
              f"floor={float((gg.floor*w).sum()):.3f}  boost={float((gg.boost*w).sum()):+.3f}",
              file=sys.stderr)
    S=pd.read_csv(a.summary,sep="\t")
    n_tot=len(S); n_unc=int((S.n_internal==0).sum())
    unc_frac=n_unc/n_tot
    e2e_splice=(P*len(H)+1.0*n_unc)/n_tot
    e2e_random=(floor*len(H)+1.0*n_unc)/n_tot
    print(f"\n===== end-to-end over all {n_tot:,} genes ({n_unc:,} uncontested) =====", file=sys.stderr)
    print(f"  uncontested fraction (free wins, no method earns these) = {unc_frac:.3f}", file=sys.stderr)
    print(f"  shortlist + random tie-break = {e2e_random:.3f}", file=sys.stderr)
    print(f"  shortlist + splice reranker  = {e2e_splice:.3f}", file=sys.stderr)
    print(f"  NOTE: every CPA score scores ~0 on contested genes BY CONSTRUCTION, so an\n"
          f"        'RF argmax' end-to-end number is just the uncontested fraction. The\n"
          f"        splicing result is the contested boost above: {d_sr:+.3f}.", file=sys.stderr)
    if a.gene_split_check:
        genes=np.array(sorted(pool.gene.unique())); rng2=np.random.default_rng(0); rng2.shuffle(genes)
        folds=np.array_split(genes,5); hh=[]; ff=[]
        for i in range(5):
            teg=set(folds[i]); trd=pool[~pool.gene.isin(teg)]; ted=pool[pool.gene.isin(teg)]
            if trd["is_canonical"].sum()==0 or ted.gene.nunique()==0: continue
            pred=fit_model(a.model,trd[SPL].to_numpy(),trd["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
            d2=ted.copy(); d2["_s"]=pred(ted[SPL].to_numpy())
            hh.append(p1_from_scores(d2,"_s",rng)); ff.append(random_expected(ted))
        Hg=np.concatenate(hh); Fg=np.concatenate(ff)
        Pg,LOg,HIg=wilson(int(Hg.sum()),len(Hg))
        dg,dglo,dghi=boot_diff(Fg,Hg)
        VAL.append(dict(scheme="5-fold by gene",n=len(Hg),p1=Pg,ci_lo=LOg,ci_hi=HIg,
                        floor=float(Fg.mean()),boost=Pg-float(Fg.mean()),
                        boost_lo=dglo,boost_hi=dghi))
        print(f"\n[gene-split leak check, 5-fold by gene, n={len(Hg):,}]", file=sys.stderr)
        print(f"  P@1={Pg:.3f} [{LOg:.3f},{HIg:.3f}]  floor={Fg.mean():.3f}  boost={Pg-Fg.mean():+.3f}",
              file=sys.stderr)
        print(f"  (chromosome-split boost was {d_sr:+.3f}; if similar, the features are not "
              f"encoding gene identity)", file=sys.stderr)
    if a.importance:
        pred=fit_model(a.model,pool[SPL].to_numpy(),pool["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
        base=p1_from_scores(pool.assign(_s=pred(pool[SPL].to_numpy())),"_s",rng).mean()
        imp=permutation_importance(pool,SPL,pred,base)
        print(f"\n[feature importance] drop in P@1 when shuffled (base {base:.3f}):", file=sys.stderr)
        for f,dv,s in imp[:12]:
            print(f"    {f:<24}{dv:>+7.4f} +/-{s:.4f}  {'#'*max(0,int(round(dv*200)))}", file=sys.stderr)
        idf=pd.DataFrame(imp,columns=["feature","p1_drop","p1_drop_sd"])
        idf["direction"]=idf.feature.map(lambda f:int(DIRP.get(f,0)))
        idf.to_csv(f"{a.out}_importance.tsv",sep="\t",index=False)
        nneg=int((idf.p1_drop<=0).sum())
        print(f"  ({nneg}/{len(idf)} features have <=0 importance: shuffling them does not hurt)",
              file=sys.stderr)
    if a.select:
        if not a.importance:
            pred0=fit_model(a.model,pool[SPL].to_numpy(),pool["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
            base0=p1_from_scores(pool.assign(_s=pred0(pool[SPL].to_numpy())),"_s",rng).mean()
            imp=permutation_importance(pool,SPL,pred0,base0)
        ranked=[f for f,_,_ in imp]
        Hf,Ff=loco_p1(pool,SPL,a.model,a.monotone,rng); full_boost=float(Hf.mean()-Ff.mean())
        try:
            from tqdm import tqdm as _tqdm
        except Exception:
            def _tqdm(x, **k): return x
        path=[]; chosen=[]; best_k=None; patience=2; hit=0
        print(f"[select] forward selection over {len(ranked)} features (target boost {full_boost:+.3f}, "
              f"tol {a.select_tol:+.3f})", file=sys.stderr)
        for k in _tqdm(range(1,len(ranked)+1), desc="select", file=sys.stderr):
            feats_k=ranked[:k]
            Hk,Fk=loco_p1(pool,feats_k,a.model,a.monotone,rng)
            bk=float(Hk.mean()-Fk.mean())
            path.append(dict(k=k,added=ranked[k-1],boost=bk,gap_to_full=full_boost-bk))
            print(f"    [select] k={k:<2} +{ranked[k-1]:<24} boost={bk:+.3f} (gap {full_boost-bk:+.3f})", file=sys.stderr)
            if (full_boost-bk) <= a.select_tol:
                if best_k is None: best_k=k; chosen=feats_k
                hit+=1
                if hit>=patience and not a.select_full_path:
                    break
            else:
                hit=0; best_k=None; chosen=[]
        if best_k is None:
            best_k=len(ranked); chosen=ranked
        pd.DataFrame(path).to_csv(f"{a.out}_selection_path.tsv",sep="\t",index=False)
        Hs,Fs=loco_p1(pool,chosen,a.model,a.monotone,rng)
        Ps,LOs,HIs=wilson(int(Hs.sum()),len(Hs)); ds,dslo,dshi=boot_diff(Fs,Hs)
        sel=pd.DataFrame(dict(feature=chosen,
                              importance=[dict((f,d) for f,d,_ in imp)[f] for f in chosen],
                              direction=[int(DIRP.get(f,0)) for f in chosen]))
        sel.to_csv(f"{a.out}_selected_features.tsv",sep="\t",index=False)
        VAL.append(dict(scheme=f"parsimony ({best_k} feat)",n=len(Hs),p1=Ps,ci_lo=LOs,ci_hi=HIs,
                        floor=float(Fs.mean()),boost=ds,boost_lo=dslo,boost_hi=dshi))
        print(f"\n[select] kept {best_k}/{len(ranked)} features "
              f"(tol {a.select_tol:+.3f} P@1): {chosen}", file=sys.stderr)
        print(f"[select] parsimonious LOCO P@1={Ps:.3f} boost={ds:+.3f} "
              f"vs full boost {full_boost:+.3f}", file=sys.stderr)
    if a.pdp:
        pred_full=fit_model(a.model,pool[SPL].to_numpy(),pool["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
        if a.importance:
            top_feats=[f for f,_,_ in imp][:a.pdp_k]
        else:
            central=["spl_upstream_max_me3","spl_interv_me5_max","spl_dist_to_max_3ss","spl_comp_up_3ss",
                     "spl_pas_prox_u1","spl_term_ppt"]
            top_feats=[f for f in central if f in SPL][:a.pdp_k]
        Xful=pool[SPL].to_numpy(); pdp_rows=[]
        for f in top_feats:
            j=SPL.index(f); col=Xful[:,j]
            lo,hi=np.percentile(col,[2,98])
            if hi<=lo: hi=lo+1.0
            grid=np.linspace(lo,hi,25); Xt=Xful.copy()
            for v in grid:
                Xt[:,j]=v
                pdp_rows.append(dict(feature=f,direction=int(DIRP.get(f,0)),x=float(v),
                                     pd=float(pred_full(Xt).mean())))
        pd.DataFrame(pdp_rows).to_csv(f"{a.out}_pdp.tsv",sep="\t",index=False)
        print(f"[pdp] partial dependence for {top_feats}", file=sys.stderr)
    if a.examples > 0:
        G=pd.concat(gw, ignore_index=True)
        key_feats=[f for f in ("spl_upstream_max_me3","spl_interv_me5_max","spl_dn5ss_has_strong") if f in SPL]
        if key_feats:
            G=G.merge(pool[["gene","center"]+key_feats], on=["gene","center"], how="left")
        if all(c in G.columns for c in ("rf_prob","strand","center")):
            cand=[]
            for g,d in G.groupby("gene", sort=False):
                if len(d)<3 or len(d)>7: continue
                ci=np.flatnonzero(d["is_canonical"].to_numpy())
                if len(ci)==0: continue
                ci=ci[0]; s=d["splice_score"].to_numpy(); r=d["rf_prob"].to_numpy()
                if int(np.argmax(s))==ci and int(np.argmax(r))!=ci:
                    cand.append((g, float(s[ci]-np.delete(s,ci).max())))
            cand.sort(key=lambda x:-x[1])
            chosen=[g for g,_ in cand[:a.examples]]
            rows=[]
            for g in chosen:
                d=G[G.gene==g]
                true_c=float(d.loc[d.is_canonical,"center"].iloc[0]); strand=d["strand"].iloc[0]
                cen=d["center"].to_numpy()
                rf_c=cen[int(np.argmax(d["rf_prob"].to_numpy()))]
                re_c=cen[int(np.argmax(d["splice_score"].to_numpy()))]
                for _,row in d.iterrows():
                    rel=float(row["center"])-true_c
                    if strand=="-": rel=-rel
                    rd_={ "gene":g, "rel_kb":rel/1000.0, "splice_score":float(row["splice_score"]),
                          "rf_prob":float(row["rf_prob"]), "is_canonical":bool(row["is_canonical"]),
                          "is_rf_pick":bool(row["center"]==rf_c),
                          "is_rerank_pick":bool(row["center"]==re_c) }
                    for kf in key_feats:
                        rd_[kf]=float(row[kf]) if pd.notna(row[kf]) else np.nan
                    rows.append(rd_)
            pd.DataFrame(rows).to_csv(f"{a.out}_examples.tsv",sep="\t",index=False)
            print(f"[examples] {len(chosen)} illustrative genes: {chosen}", file=sys.stderr)
        else:
            print(f"[examples] need rf_prob/strand/center columns; skipped", file=sys.stderr)
    if a.ablation:
        GROUPS = {
            "acceptor (3'ss)": [f for f in SPL if "3ss" in f or "ppt" in f or "u2af" in f
                                or "bp_strength" in f or "bp_to_3ss" in f
                                or f == "spl_acceptor_minus_dndonor"],
            "donor (5'ss)":    [f for f in SPL if "5ss" in f
                                or f in ("spl_local_me5","spl_interv_me5_max","spl_dn5ss_has_strong",
                                         "spl_dist_interv_me5","spl_site_has_donor")],
            "U1":              [f for f in SPL if "u1" in f],
            "all splice":      SPL,
        }
        abl = []
        for name, cols in GROUPS.items():
            cols = [c for c in cols if c in SPL]
            if len(cols) == 0: continue
            Hb, Fb = loco_p1(pool, cols, a.model, a.monotone, rng)
            p, lo, hi = wilson(int(Hb.sum()), len(Hb))
            d, dlo, dhi = boot_diff(Fb, Hb)
            abl.append(dict(group=name, n_feat=len(cols), p1=p, ci_lo=lo, ci_hi=hi,
                            floor=float(Fb.mean()), boost=d, boost_lo=dlo, boost_hi=dhi))
        pd.DataFrame(abl).to_csv(f"{a.out}_ablation.tsv", sep="\t", index=False)
        print(f"\n[ablation] LOCO P@1 by feature group:", file=sys.stderr)
        for r in abl:
            print(f"    {r['group']:<16} ({r['n_feat']:>2}f) P@1={r['p1']:.3f} boost={r['boost']:+.3f}",
                  file=sys.stderr)
    if a.permnull > 0:
        ps = pool.sort_values("gene", kind="stable").reset_index(drop=True)
        gsz = ps.groupby("gene", sort=False).size().to_numpy()
        starts = np.concatenate([[0], np.cumsum(gsz)[:-1]])
        Xall = ps[SPL].to_numpy(); chrom_all = ps["chrom"].to_numpy()
        real_lab = ps["is_canonical"].to_numpy().astype(bool)
        sizes = ps.groupby("chrom").size().sort_values(ascending=False)
        binA, sA, sB = set(), 0, 0
        for c, sz in sizes.items():
            if sA <= sB: binA.add(c); sA += int(sz)
            else: sB += int(sz)
        tr = np.array([c in binA for c in chrom_all]); te = ~tr
        Xtr, Xte = Xall[tr], Xall[te]
        te_gene = ps["gene"].to_numpy()[te]
        chg = np.flatnonzero(te_gene[1:] != te_gene[:-1]) + 1
        gs = np.concatenate([[0], chg]); ge = np.concatenate([chg, [len(te_gene)]])
        floor_pn = float(np.mean([1.0/(e-s) for s, e in zip(gs, ge)]))
        mc = [int(DIRP.get(f, 0)) for f in SPL] if a.monotone else None
        payload = (Xtr, Xte, tr, te, starts, gsz, gs, ge, floor_pn, mc, a.model, a.permnull_iters)
        _pn_init(*payload)
        real_split = _pn_boost(real_lab)
        seeds = list(range(1000, 1000 + a.permnull))
        try:
            from tqdm import tqdm as _tqdm
        except Exception:
            def _tqdm(x, **k): return x
        backend = a.permnull_backend
        if backend == "process" and a.n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            print(f"[perm null] {a.permnull} nulls, process backend x{a.n_jobs} "
                  f"({a.permnull_iters}-iter fits; train {int(tr.sum()):,} / test {int(te.sum()):,})", file=sys.stderr)
            with ProcessPoolExecutor(max_workers=a.n_jobs, initializer=_pn_init, initargs=payload) as ex:
                nb = np.array(list(_tqdm(ex.map(_pn_null, seeds, chunksize=8),
                                         total=len(seeds), desc="perm-null", file=sys.stderr)))
        elif backend == "thread" and a.n_jobs > 1:
            import os
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(v, "1")
            from concurrent.futures import ThreadPoolExecutor
            print(f"[perm null] {a.permnull} nulls, thread backend x{a.n_jobs} "
                  f"({a.permnull_iters}-iter fits; train {int(tr.sum()):,} / test {int(te.sum()):,})", file=sys.stderr)
            with ThreadPoolExecutor(max_workers=a.n_jobs) as ex:
                nb = np.array(list(_tqdm(ex.map(_pn_null, seeds),
                                         total=len(seeds), desc="perm-null", file=sys.stderr)))
        else:
            print(f"[perm null] {a.permnull} nulls, serial ({a.permnull_iters}-iter fits; "
                  f"train {int(tr.sum()):,} / test {int(te.sum()):,})", file=sys.stderr)
            nb = np.array([_pn_null(sd) for sd in _tqdm(seeds, total=len(seeds), desc="perm-null", file=sys.stderr)])
        z = (real_split - nb.mean()) / (nb.std(ddof=1) + 1e-9)
        pval = (1 + int((nb >= real_split).sum())) / (1 + len(nb))
        pd.DataFrame(dict(kind=["real"] + ["null"] * len(nb),
                          boost=[real_split] + list(nb),
                          p1=[real_split + floor_pn] + list(nb + floor_pn))).to_csv(
                          f"{a.out}_permnull.tsv", sep="\t", index=False)
        pd.DataFrame([dict(metric="real_boost", value=real_split),
                      dict(metric="real_p1", value=float(real_split + floor_pn)),
                      dict(metric="floor", value=float(floor_pn)),
                      dict(metric="loco_boost", value=d_sr),
                      dict(metric="null_mean", value=float(nb.mean())),
                      dict(metric="null_sd", value=float(nb.std(ddof=1))),
                      dict(metric="z", value=float(z)),
                      dict(metric="n_nulls", value=int(len(nb))),
                      dict(metric="p_value", value=float(pval))]).to_csv(f"{a.out}_permnull_metrics.tsv", sep="\t", index=False)
        print(f"\n[perm null] fixed-split real boost {real_split:+.3f} (LOCO headline {d_sr:+.3f}); "
              f"null {nb.mean():+.3f} \u00b1 {nb.std(ddof=1):.3f} (n={len(nb)}); z={z:.1f}; p={pval:.4g}",
              file=sys.stderr)
    if a.capacity_sweep:
        from sklearn.ensemble import HistGradientBoostingClassifier
        cap = []
        hold = "chr1" if "chr1" in chroms else chroms[0]
        trc = pool[pool.chrom != hold]; tec = pool[pool.chrom == hold]
        yt = trc["is_canonical"].astype(int).to_numpy()
        for depth, iters, leaf in [(2,300,80),(3,400,50),(5,600,30),(7,800,20),(None,1200,10)]:
            mc = [int(DIRP.get(f,0)) for f in SPL] if a.monotone else None
            clf = HistGradientBoostingClassifier(max_depth=depth,max_iter=iters,learning_rate=0.05,
                min_samples_leaf=leaf,class_weight="balanced",monotonic_cst=mc,random_state=0)
            clf.fit(trc[SPL].to_numpy(), yt)
            f = lambda X: clf.predict_proba(X)[:,1]
            ptr = p1_from_scores(trc.assign(_s=f(trc[SPL].to_numpy())), "_s", rng).mean()
            pte = p1_from_scores(tec.assign(_s=f(tec[SPL].to_numpy())), "_s", rng).mean()
            cap.append(dict(depth=("Inf" if depth is None else depth), iters=iters, leaf=leaf,
                            train_p1=ptr, test_p1=pte, gap=ptr-pte))
        pd.DataFrame(cap).to_csv(f"{a.out}_capacity.tsv", sep="\t", index=False)
        print(f"\n[capacity] train vs test P@1 (held-out {hold}):", file=sys.stderr)
        for r in cap: print(f"    depth={str(r['depth']):>3} train={r['train_p1']:.3f} test={r['test_p1']:.3f} gap={r['gap']:+.3f}", file=sys.stderr)
    GW = pd.concat(gw, ignore_index=True)
    cs_rows = []
    GW["bucket"] = pd.cut(GW.n_internal, [0,1,2,4,8,10**9], labels=["1","2","3-4","5-8","9+"])
    for b, sub in GW.groupby("bucket", observed=True):
        hh = p1_from_scores(sub, "splice_score", rng)
        ff = random_expected(sub)
        p, lo, hi = wilson(int(hh.sum()), len(hh))
        cs_rows.append(dict(bucket=str(b), n_genes=sub.gene.nunique(), p1=p, ci_lo=lo, ci_hi=hi,
                            floor=float(ff.mean()), boost=p-float(ff.mean())))
    pd.DataFrame(cs_rows).to_csv(f"{a.out}_by_contest.tsv", sep="\t", index=False)
    print(f"\n[by contest size] splice P@1 vs floor:", file=sys.stderr)
    for r in cs_rows:
        print(f"    {r['bucket']:<4} n={r['n_genes']:<5} P@1={r['p1']:.3f} floor={r['floor']:.3f} boost={r['boost']:+.3f}", file=sys.stderr)
    if a.fixed_split_check:
        tr=pool[~pool.chrom.isin(TEST_CHROMS)]; te=pool[pool.chrom.isin(TEST_CHROMS)]
        if len(te)==0 or len(tr)==0 or tr["is_canonical"].sum()==0:
            print(f"\n[consistency] legacy fixed split unavailable "
                  f"(test peaks={len(te)}, train peaks={len(tr)}); skipped", file=sys.stderr)
        else:
            pred=fit_model(a.model,tr[SPL].to_numpy(),tr["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
            d2=te.copy(); d2["_s"]=pred(te[SPL].to_numpy())
            h=p1_from_scores(d2,"_s",rng); f=random_expected(te)
            p,lo,hi=wilson(int(h.sum()),len(h))
            dfx,dfxlo,dfxhi=boot_diff(f,h)
            VAL.append(dict(scheme="fixed chr18-22,X",n=len(h),p1=p,ci_lo=lo,ci_hi=hi,
                            floor=float(f.mean()),boost=p-float(f.mean()),
                            boost_lo=dfxlo,boost_hi=dfxhi))
            print(f"\n[consistency: legacy chr18-22,X fixed split, n={len(h)} genes]", file=sys.stderr)
            print(f"  P@1={p:.3f} [{lo:.3f},{hi:.3f}]  floor={f.mean():.3f}  boost={p-f.mean():+.3f}",
                  file=sys.stderr)
    if a.violins:
        from scipy.stats import mannwhitneyu
        long_rows = []; stat_rows = []
        yv = pool["is_canonical"].to_numpy()
        for f in SPL:
            v = pool[f].to_numpy().astype(float)
            a_ = v[yv]; b_ = v[~yv]
            a_ = a_[np.isfinite(a_)]; b_ = b_[np.isfinite(b_)]
            if len(a_) < 10 or len(b_) < 10:
                continue
            try:
                U, _ = mannwhitneyu(a_, b_, alternative="two-sided")
                rb = 2.0 * U / (len(a_) * len(b_)) - 1.0
            except Exception:
                rb = np.nan
            stat_rows.append(dict(feature=f, label=f.replace("spl_", ""),
                                  rank_biserial=float(rb),
                                  median_canon=float(np.median(a_)),
                                  median_comp=float(np.median(b_))))
        st = pd.DataFrame(stat_rows).reindex(
            pd.DataFrame(stat_rows).rank_biserial.abs().sort_values(ascending=False).index)
        keep = st.head(9)["feature"].tolist()
        for f in keep:
            v = pool[f].to_numpy().astype(float)
            for grp_, mask in (("canonical", yv), ("competitor", ~yv)):
                vv = v[mask]; vv = vv[np.isfinite(vv)]
                if len(vv) > 6000:
                    vv = np.random.default_rng(0).choice(vv, 6000, replace=False)
                for x in vv:
                    long_rows.append(dict(feature=f, group=grp_, value=float(x)))
        pd.DataFrame(long_rows).to_csv(f"{a.out}_violins_long.tsv", sep="\t", index=False)
        st[st.feature.isin(keep)].to_csv(f"{a.out}_violins_stats.tsv", sep="\t", index=False)
        print(f"\n[violins] wrote {len(keep)} feature distributions (top |rank-biserial|)", file=sys.stderr)
    if a.shap:
        try:
            import shap
            from sklearn.ensemble import HistGradientBoostingClassifier
            mc = [int(DIRP.get(f, 0)) for f in SPL] if a.monotone else None
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.05,
                    min_samples_leaf=50, class_weight="balanced", monotonic_cst=mc, random_state=0)
            clf.fit(pool[SPL].to_numpy(), pool["is_canonical"].astype(int).to_numpy())
            idx = []
            for mask in (pool["is_canonical"].to_numpy(), ~pool["is_canonical"].to_numpy()):
                ii = np.flatnonzero(mask)
                if len(ii) > a.shap_max:
                    ii = np.random.default_rng(0).choice(ii, a.shap_max, replace=False)
                idx.append(ii)
            idx = np.concatenate(idx)
            Xs = pool[SPL].to_numpy()[idx]
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(Xs)
            sv = sv[1] if isinstance(sv, list) else sv
            rows = []
            for j, f in enumerate(SPL):
                for i in range(len(idx)):
                    rows.append(dict(feature=f, shap=float(sv[i, j]), value=float(Xs[i, j])))
            pd.DataFrame(rows).to_csv(f"{a.out}_shap_long.tsv", sep="\t", index=False)
            print(f"\n[shap] wrote per-peak SHAP for {len(SPL)} features on {len(idx):,} peaks", file=sys.stderr)
        except ImportError:
            print("\n[shap] `shap` package not installed; skipping. "
                  "pip install shap to enable this panel.", file=sys.stderr)
    if a.override_geom:
        GWo = pd.concat(gw, ignore_index=True)
        has_rf = "rf_prob" in GWo.columns
        strand_col = "strand" if "strand" in GWo.columns else None
        pergene = []
        for g, sub in GWo.groupby("gene", sort=False):
            if len(sub) < 2:
                continue
            true_row = sub[sub.is_canonical]
            if len(true_row) == 0:
                continue
            true_c = float(true_row.center.iloc[0])
            strand = true_row[strand_col].iloc[0] if strand_col else "+"
            spl_pick = sub.loc[sub.splice_score.idxmax()]
            rf_pick = sub.loc[sub.rf_prob.idxmax()] if has_rf else spl_pick
            def signed(pick_c):
                d = float(pick_c) - true_c
                return d if strand == "+" else -d
            overrode = bool(rf_pick.center != spl_pick.center)
            pergene.append(dict(gene=g, overrode=overrode,
                                spl_signed=signed(spl_pick.center),
                                rf_signed=signed(rf_pick.center),
                                spl_correct=bool(spl_pick.is_canonical),
                                rf_correct=bool(rf_pick.is_canonical)))
        PG = pd.DataFrame(pergene)
        PG.to_csv(f"{a.out}_override_pergene.tsv", sep="\t", index=False)
        ov = PG[PG.overrode]
        if len(ov):
            edges = np.arange(-5, 5.01, 0.25)
            ctr = (edges[:-1] + edges[1:]) / 2
            hs = np.histogram(ov.spl_signed / 1000.0, bins=edges)[0]
            hr = np.histogram(ov.rf_signed / 1000.0, bins=edges)[0]
            pd.DataFrame(dict(bin_center_kb=ctr, reranker=hs, rf=hr)).to_csv(
                f"{a.out}_override_hist.tsv", sep="\t", index=False)
        print(f"\n[override geom] {len(PG):,} multi-peak genes, "
              f"{int(PG.overrode.sum()):,} overridden by splice", file=sys.stderr)
    if a.atlas:
        try:
            atl = pd.read_csv(a.atlas, sep="\t", header=None, comment="#")
            atl = atl.rename(columns={0: "chrom", 1: "start", 2: "end"})
            atl["mid"] = ((atl["start"] + atl["end"]) // 2).astype(int)
            usage_ok = False
            ucol = a.atlas_usage_col
            if atl.shape[1] > ucol:
                u = pd.to_numeric(atl[ucol], errors="coerce")
                if u.notna().mean() > 0.5 and u.nunique() > 5:
                    atl["usage"] = u; usage_ok = True
            if not usage_ok:
                atl["usage"] = np.nan
            by_chrom = {}
            for c, d in atl.groupby("chrom"):
                o = np.argsort(d["mid"].to_numpy())
                by_chrom[c] = (d["mid"].to_numpy()[o], d["usage"].to_numpy()[o])
            GWa = pd.concat(gw, ignore_index=True)
            def nearest(chrom, center, tol=50):
                pack = by_chrom.get(chrom)
                if pack is None: return (False, np.nan)
                mids, us = pack
                i = int(np.searchsorted(mids, center)); best = None; bd = tol + 1
                for j in (i - 1, i):
                    if 0 <= j < len(mids):
                        dd = abs(int(center) - int(mids[j]))
                        if dd < bd: bd = dd; best = j
                if best is None or bd > tol: return (False, np.nan)
                return (True, float(us[best]))
            picks_re = GWa.loc[GWa.groupby("gene")["splice_score"].idxmax()]
            picks_rf = GWa.loc[GWa.groupby("gene")["rf_prob"].idxmax()] if "rf_prob" in GWa.columns else None
            rows = []
            for who, picks in (("RF would pick", picks_rf), ("reranker moves to", picks_re)):
                if picks is None: continue
                hits = [nearest(r["chrom"], r["center"]) for _, r in picks.iterrows()]
                rows.append(dict(which=who, on_atlas_rate=float(np.mean([h[0] for h in hits])),
                                 n_changed=len(picks)))
            pd.DataFrame(rows).to_csv(f"{a.out}_atlas.tsv", sep="\t", index=False)
            print(f"\n[atlas] on-atlas rates written (usage column {'found' if usage_ok else 'absent'})", file=sys.stderr)
            if usage_ok and picks_rf is not None:
                rf_c = {g: (r["chrom"], r["center"]) for g, r in picks_rf.set_index("gene").iterrows()}
                re_c = {g: (r["chrom"], r["center"]) for g, r in picks_re.set_index("gene").iterrows()}
                urows = []
                for g, rec in re_c.items():
                    if g not in rf_c: continue
                    rfc = rf_c[g]
                    rf_on, rf_u = nearest(*rfc); re_on, re_u = nearest(*rec)
                    urows.append(dict(gene=g, overrode=bool(rfc[1] != rec[1]),
                                      rf_on=rf_on, rf_usage=rf_u, rerank_on=re_on, rerank_usage=re_u))
                ud = pd.DataFrame(urows)
                ud.to_csv(f"{a.out}_atlas_usage.tsv", sep="\t", index=False)
                sub = ud[(ud.overrode) & ud.rf_usage.notna() & ud.rerank_usage.notna()
                         & (ud.rf_usage > 0) & (ud.rerank_usage > 0)]
                if len(sub) >= 10:
                    up = int((sub.rerank_usage > sub.rf_usage).sum())
                    dn = int((sub.rerank_usage < sub.rf_usage).sum())
                    try:
                        from scipy.stats import binomtest
                        psign = binomtest(up, up + dn, 0.5, alternative="greater").pvalue if (up + dn) else np.nan
                    except Exception:
                        psign = np.nan
                    pd.DataFrame([dict(metric="n_overridden_both_atlas", value=int(len(sub))),
                                  dict(metric="frac_rerank_higher_usage", value=float(up / max(up + dn, 1))),
                                  dict(metric="median_log2_ratio", value=float(np.median(np.log2(sub.rerank_usage / sub.rf_usage)))),
                                  dict(metric="p_sign", value=float(psign))]).to_csv(
                        f"{a.out}_atlas_usage_metrics.tsv", sep="\t", index=False)
                    print(f"[atlas] usage-weighting: {up}/{up+dn} overrides move to a higher-usage site "
                          f"(p={psign:.3g})", file=sys.stderr)
        except Exception as e:
            print(f"\n[atlas] failed ({e}); skipping", file=sys.stderr)
    if a.winrate:
        try:
            from scipy.stats import binomtest
        except Exception:
            binomtest = None
        genes_grp = list(pool.groupby("gene", sort=False))
        ns = np.array([len(gg) for _, gg in genes_grp], float)
        floor_wr = float(np.mean(1.0 / ns))
        B = 2000
        rng_wr = np.random.default_rng(0)
        null_rates = np.array([float(np.mean(rng_wr.random(len(ns)) < 1.0/ns)) for _ in range(B)])
        def _credit(feature, direction):
            cr = []
            for _, gg in genes_grp:
                v = gg[feature].to_numpy().astype(float)
                fin = np.isfinite(v)
                if not fin.any():
                    continue
                isc = gg["is_canonical"].to_numpy()
                ext = np.nanmax(v) if direction >= 0 else np.nanmin(v)
                at = fin & (np.abs(v - ext) <= 1e-9)
                k = int(at.sum())
                if k == 0:
                    continue
                cr.append(int((at & isc).sum()) / k)
            return np.array(cr, float)
        wr_rows = []
        for f in SPL:
            s = int(DIRP.get(f, 0))
            if s == 0:
                c_hi = _credit(f, +1); c_lo = _credit(f, -1)
                if (c_hi.mean() if len(c_hi) else -1) >= (c_lo.mean() if len(c_lo) else -1):
                    cr, s_used = c_hi, +1
                else:
                    cr, s_used = c_lo, -1
            else:
                cr = _credit(f, s); s_used = s
            if len(cr) == 0:
                continue
            usable = len(cr)
            rate = float(cr.mean())
            bi = rng_wr.integers(0, usable, (2000, usable))
            bs = cr[bi].mean(axis=1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            p_mc = (1 + int((null_rates >= rate).sum())) / (1 + B)
            if binomtest is not None:
                try:
                    p_binom = binomtest(int(round(rate*usable)), usable, floor_wr,
                                        alternative="greater").pvalue
                except Exception:
                    p_binom = np.nan
            else:
                p_binom = np.nan
            wr_rows.append(dict(feature=f, direction=s_used, n_genes=usable,
                                canonical_win_rate=rate, ci_lo=lo, ci_hi=hi,
                                random_rate=floor_wr, enrichment=rate - floor_wr,
                                p_mc=float(p_mc), p_binom=p_binom,
                                min_better=(s_used < 0)))
        wr = pd.DataFrame(wr_rows).sort_values("enrichment", ascending=False)
        wr.to_csv(f"{a.out}_winrate.tsv", sep="\t", index=False)
        print(f"\n[winrate] canonical within-gene extreme rate "
              f"(floor={floor_wr:.3f}, N={len(genes_grp)} genes):", file=sys.stderr)
        for r in wr.head(12).itertuples():
            arrow = "min" if r.min_better else "max"
            print(f"    {r.feature:<26} rate={r.canonical_win_rate:.3f} "
                  f"[{r.ci_lo:.3f},{r.ci_hi:.3f}] enrich={r.enrichment:+.3f} "
                  f"({arrow}, n={r.n_genes}, p_mc={r.p_mc:.2g})", file=sys.stderr)

    pd.DataFrame(VAL).to_csv(f"{a.out}_validation.tsv",sep="\t",index=False)
    pd.DataFrame([
        dict(metric="n_contested_genes",value=len(H)),
        dict(metric="n_all_genes",value=n_tot),
        dict(metric="loco_p1_splice",value=P,ci_lo=LO,ci_hi=HI),
        dict(metric="loco_p1_random_floor",value=floor),
        dict(metric="loco_splice_minus_random",value=d_sr,ci_lo=lo_sr,ci_hi=hi_sr),
        dict(metric="uncontested_fraction",value=unc_frac),
        dict(metric="e2e_shortlist_random",value=e2e_random),
        dict(metric="e2e_shortlist_splice",value=e2e_splice),
    ]).to_csv(f"{a.out}_metrics.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_metrics.tsv, {a.out}_degeneracy.tsv, {a.out}_validation.tsv, "
          f"{a.out}_genomewide_bychrom.tsv, {a.out}_genomewide_scored.parquet", file=sys.stderr)
if __name__=="__main__": main()


