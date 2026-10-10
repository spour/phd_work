#!/usr/bin/env python3
# (step 3 of the clean rebuild) within-shortlist reranking on contested genes
import argparse, sys, math
import numpy as np, pandas as pd

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}

DIRP={"spl_u1_suppression_up":-1,"spl_u1_suppression_down":-1,"spl_bp_strength":+1,
      "spl_u2af65_ppt":+1,"spl_u2af35_ag":+1,"spl_term_3ss":+1,"spl_upstream_max_me3":+1,
      "spl_local_me5":-1,"spl_interv_me5_max":-1,"spl_dn5ss_has_strong":-1,
      "spl_pas_prox_u1":-1,"spl_upstream_mean_u1":-1,"spl_term_ppt":+1,
      "spl_rel_3ss_up":+1,"spl_rel_3ss_dn":+1,"spl_term_3ss_abs":+1,
      "spl_comp_up_3ss":-1,"spl_comp_dn_3ss":-1,
      "spl_dist_nearest_5ss":+1,"spl_dist_up_5ss":+1,"spl_dist_dn_5ss":+1,
      "spl_n_5ss_500":-1,"spl_n_5ss_1k":-1,"spl_sum_5ss":-1,"spl_mean_top3_5ss":-1}


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

def p1_from_scores(df, score_col):
    h=[]
    for g,gg in df.groupby("gene",sort=False):
        top=int(np.argmax(gg[score_col].to_numpy()))
        h.append(int(bool(gg["is_canonical"].to_numpy()[top])))
    return np.array(h)

def permutation_importance(df, feats, predict, base_p1, n_rep=5, seed=0):
    rng=np.random.default_rng(seed); out=[]
    X=df[feats].to_numpy()
    for j,f in enumerate(feats):
        drops=[]
        for _ in range(n_rep):
            Xp=X.copy(); Xp[:,j]=rng.permutation(Xp[:,j])
            d2=df.copy(); d2["_s"]=predict(Xp)
            drops.append(base_p1 - p1_from_scores(d2,"_s").mean())
        out.append((f,float(np.mean(drops)),float(np.std(drops))))
    return sorted(out,key=lambda x:-x[1])

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def per_gene_hits(df, score_col, rng=None, jitter=False):
    genes=[]; hits=[]
    for g,gg in df.groupby("gene",sort=False):
        s=gg[score_col].to_numpy().astype(float)
        if jitter: s=rng.random(len(s))
        top=int(np.argmax(s))
        hits.append(int(bool(gg["is_canonical"].to_numpy()[top]))); genes.append(g)
    return np.array(genes), np.array(hits)

def random_expected(df):
    e=[]
    for g,gg in df.groupby("gene",sort=False):
        e.append(1.0/len(gg))
    return np.array(e)

def boot_diff(ha,hb,n_boot=5000,seed=0):
    rng=np.random.default_rng(seed); ha=np.asarray(ha,float); hb=np.asarray(hb,float)
    n=len(ha); d=[]
    for _ in range(n_boot):
        ix=rng.integers(0,n,n); d.append(hb[ix].mean()-ha[ix].mean())
    return float(hb.mean()-ha.mean()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_feat",required=True); ap.add_argument("--test_feat",required=True)
    ap.add_argument("--summary",required=True,help="shortlist_summary.tsv (for uncontested free wins)")
    ap.add_argument("--model",default="gbt",choices=["gbt","logistic"])
    ap.add_argument("--monotone",action="store_true")
    ap.add_argument("--genomewide",action="store_true",
                    help="leave-one-chromosome-out P@1 over ALL chromosomes (pools train+test feat)")
    ap.add_argument("--importance",action="store_true",help="permutation feature importance")
    ap.add_argument("--gene_split_check",action="store_true",
                    help="leak test: split train/test by GENE within the pooled peaks; "
                         "if the boost survives, features are not encoding gene identity")
    ap.add_argument("--capacity_sweep",action="store_true",
                    help="report train vs test P@1 and sweep model capacity to test under/overfit")
    ap.add_argument("--lean",action="store_true",
                    help="refit using only features with positive permutation importance")
    ap.add_argument("--out",default="rerank")
    a=ap.parse_args()

    tr=pd.read_parquet(a.train_feat); te=pd.read_parquet(a.test_feat)
    SPL=[c for c in tr.columns if c.startswith("spl_")]
    print(f"[data] train {tr.gene.nunique():,} genes / {len(tr):,} peaks; "
          f"test {te.gene.nunique():,} genes / {len(te):,} peaks; {len(SPL)} splice features",file=sys.stderr)
    for d in (tr,te):
        d[SPL]=d[SPL].apply(pd.to_numeric,errors="coerce").fillna(0.0)

    # train the splice reranker on contested train peaks (label = is_canonical)
    ytr=tr["is_canonical"].astype(int).to_numpy()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if a.model=="gbt":
        mc=[int(DIRP.get(f,0)) for f in SPL] if a.monotone else None
        clf=HistGradientBoostingClassifier(max_depth=3,max_iter=400,learning_rate=0.05,
            min_samples_leaf=50,class_weight="balanced",monotonic_cst=mc,random_state=0)
        clf.fit(tr[SPL].to_numpy(),ytr)
        te=te.copy(); te["splice_score"]=clf.predict_proba(te[SPL].to_numpy())[:,1]
    else:
        sc=StandardScaler().fit(tr[SPL]); 
        clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(tr[SPL]),ytr)
        if a.monotone:
            co=clf.coef_[0].copy(); mcv=np.array([DIRP.get(f,0) for f in SPL],float)
            co[(mcv!=0)&(np.sign(co)!=np.sign(mcv))]=0.0; clf.coef_[0]=co
        te=te.copy(); te["splice_score"]=clf.predict_proba(sc.transform(te[SPL]))[:,1]

    # score on contested test genes
    rng=np.random.default_rng(0)
    g_spl,h_spl=per_gene_hits(te,"splice_score")
    e_rnd=random_expected(te)
    n=len(g_spl)
    def rep(h): p,lo,hi=wilson(int(h.sum()),len(h)); return p,lo,hi
    p_spl=rep(h_spl); p_rnd=(e_rnd.mean(),)+wilson(int(round(e_rnd.mean()*n)),n)[1:]

    _,h_cpa=per_gene_hits(te,"rf_prob"); p_cpa_deg=float(h_cpa.mean())

    print(f"\n========== CONTESTED test genes (n={n}) ==========",file=sys.stderr)
    print(f"{'ranker':<12}{'P@1':>8}{'95% CI':>20}",file=sys.stderr)
    print(f"{'random':<12}{p_rnd[0]:>8.3f}   [{p_rnd[1]:.3f},{p_rnd[2]:.3f}]  (floor = mean 1/n; the baseline)",file=sys.stderr)
    print(f"{'splice':<12}{p_spl[0]:>8.3f}   [{p_spl[1]:.3f},{p_spl[2]:.3f}]",file=sys.stderr)
    print(f"  (within-shortlist rf-argmax = {p_cpa_deg:.3f} is DEGENERATE by the >= "
          f"definition and is not a valid baseline; use random)",file=sys.stderr)

    # train-set p@1 (to see the train/test gap
    pred_tr=fit_model(a.model,tr[SPL].to_numpy(),tr["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
    tr2=tr.copy(); tr2["_s"]=pred_tr(tr[SPL].to_numpy())
    h_tr=p1_from_scores(tr2,"_s"); p_tr=h_tr.mean()
    print(f"  [train P@1 = {p_tr:.3f} in-sample | test {p_spl[0]:.3f}]  "
          f"gap={p_tr-p_spl[0]:+.3f}  (small gap + low test => ceiling, not underfit; "
          f"large gap => overfit)",file=sys.stderr)

    # headline: splice - random (the isolated splicing contribution)
    d_sr,lo_sr,hi_sr=boot_diff(e_rnd,h_spl)
    sig=lambda lo,hi:"**" if (lo>0 or hi<0) else ""
    print(f"\n[isolated splicing contribution] splice - random = {d_sr:+.3f} "
          f"[{lo_sr:+.3f},{hi_sr:+.3f}] {sig(lo_sr,hi_sr)}",file=sys.stderr)

    S=pd.read_csv(a.summary,sep="\t")
    S=S[S.chrom.isin(TEST_CHROMS)]
    n_uncontested=int((S.n_internal==0).sum()); n_total=len(S)
    def overall(p_contested): return (p_contested*n + 1.0*n_uncontested)/n_total
    o_cpa = n_uncontested/n_total
    o_rnd = overall(p_rnd[0]); o_spl = overall(p_spl[0])
    print(f"\n========== END-TO-END overall test P@1 (n={n_total}, "
          f"{n_uncontested} uncontested free wins) ==========",file=sys.stderr)
    print(f"  RF argmax (deployed baseline) {o_cpa:.3f}   [sanity: should ~match your 0.41]",file=sys.stderr)
    print(f"  shortlist + random tie-break  {o_rnd:.3f}   (narrowing alone, no splice)",file=sys.stderr)
    print(f"  shortlist + splice reranker   {o_spl:.3f}",file=sys.stderr)
    print(f"\n  end-to-end gain over RF argmax: splice {o_spl-o_cpa:+.3f}  (random {o_rnd-o_cpa:+.3f})",file=sys.stderr)
    print(f"  -> of the {o_spl-o_cpa:+.3f} two-stage gain, {o_rnd-o_cpa:+.3f} is candidate "
          f"narrowing (a coin flip on the shortlist) and {o_spl-o_rnd:+.3f} is splice "
          f"reranking over the floor. The {o_spl-o_rnd:+.3f} is the part that is actually splicing.",file=sys.stderr)

    if a.importance:
        base=h_spl.mean()
        pred=fit_model(a.model,tr[SPL].to_numpy(),tr["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
        imp=permutation_importance(te.assign(**{c:te[c] for c in SPL}),SPL,pred,base)
        print(f"\n[feature importance] drop in contested test P@1 when feature is shuffled:",file=sys.stderr)
        for f,d,s in imp:
            bar="#"*max(0,int(round(d*200)))
            print(f"    {f:<24}{d:>+7.4f} +/-{s:.4f}  {bar}",file=sys.stderr)
        impdf=pd.DataFrame(imp,columns=["feature","p1_drop","p1_drop_sd"])
        impdf["direction"]=impdf["feature"].map(lambda f:int(DIRP.get(f,0)))
        impdf.to_csv(f"{a.out}_importance.tsv",sep="\t",index=False)

    if a.genomewide:
        pool=pd.concat([tr,te],ignore_index=True)
        for c in SPL: pool[c]=pd.to_numeric(pool[c],errors="coerce").fillna(0.0)
        pool["is_canonical"]=pool["is_canonical"].astype(bool)
        chroms=sorted(pool.chrom.unique())
        all_h=[]; per_chrom=[]
        for c in chroms:
            tr_c=pool[pool.chrom!=c]; te_c=pool[pool.chrom==c]
            if te_c.gene.nunique()==0 or tr_c["is_canonical"].sum()==0: continue
            pred=fit_model(a.model,tr_c[SPL].to_numpy(),tr_c["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
            d2=te_c.copy(); d2["_s"]=pred(te_c[SPL].to_numpy())
            h=p1_from_scores(d2,"_s"); all_h.append(h)
            pr,plo,phi=wilson(int(h.sum()),len(h))
            rnd=float((1.0/te_c.groupby("gene").size()).mean())
            per_chrom.append((c,len(h),pr,plo,phi,rnd))
        gw_scored=[]
        for c in chroms:
            tr_c=pool[pool.chrom!=c]; te_c=pool[pool.chrom==c]
            if te_c.gene.nunique()==0 or tr_c["is_canonical"].sum()==0: continue
            predc=fit_model(a.model,tr_c[SPL].to_numpy(),tr_c["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
            tmp=te_c[["gene","chrom","center","rf_prob","is_canonical","n_internal"]].copy()
            tmp["splice_score"]=predc(te_c[SPL].to_numpy())
            gw_scored.append(tmp)
        if gw_scored:
            GW=pd.concat(gw_scored,ignore_index=True)
            GW.to_parquet(f"{a.out}_genomewide_scored.parquet",index=False)
            print(f"  [saved genome-wide LOCO picks -> {a.out}_genomewide_scored.parquet "
                  f"({len(GW):,} peaks, {GW.gene.nunique():,} genes)]",file=sys.stderr)
        H=np.concatenate(all_h); P,LO,HI=wilson(int(H.sum()),len(H))
        rnd_all=float((1.0/pool.groupby("gene").size()).mean())
        print(f"\n========== GENOME-WIDE leave-one-chromosome-out (contested genes, n={len(H)}) ==========",file=sys.stderr)
        print(f"  splice reranker P@1 = {P:.3f}  [{LO:.3f},{HI:.3f}]   (random floor {rnd_all:.3f}, supplementary)",file=sys.stderr)
        VAL_CHROMS={"chr3","chr13"}
        def group_of(c): return "RF-heldout(test)" if c in TEST_CHROMS else ("RF-val" if c in VAL_CHROMS else "RF-trained")
        pc=pd.DataFrame(per_chrom,columns=["chrom","n","p1","ci_lo","ci_hi","random"])
        pc["rf_group"]=pc["chrom"].map(group_of)
        print(f"\n  [splice boost where RF did vs did NOT see the chromosome]",file=sys.stderr)
        print(f"  (if the splice contribution P@1-floor is similar in both, the reranking",file=sys.stderr)
        print(f"   effect is real and not riding on RF overfitting its training chroms)",file=sys.stderr)
        for grp,gg in pc.groupby("rf_group"):
            sub=pool[pool.chrom.isin(set(gg.chrom))]
            hh=[]; 
            for c in gg.chrom:
                trc=pool[pool.chrom!=c]; tec=pool[pool.chrom==c]
                if tec.gene.nunique()==0 or trc["is_canonical"].sum()==0: continue
                pr=fit_model(a.model,trc[SPL].to_numpy(),trc["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
                d2=tec.copy(); d2["_s"]=pr(tec[SPL].to_numpy()); hh.append(p1_from_scores(d2,"_s"))
            if not hh: continue
            hh=np.concatenate(hh); pP,pLO,pHI=wilson(int(hh.sum()),len(hh))
            fl=float((1.0/sub.groupby("gene").size()).mean())
            print(f"    {grp:<18} n={len(hh):<5} splice P@1={pP:.3f} [{pLO:.3f},{pHI:.3f}] "
                  f"floor={fl:.3f}  boost(P@1-floor)={pP-fl:+.3f}",file=sys.stderr)
        pc.to_csv(f"{a.out}_genomewide_bychrom.tsv",sep="\t",index=False)
        Sall=pd.read_csv(a.summary,sep="\t"); n_unc=int((Sall.n_internal==0).sum()); n_tot=len(Sall)
        gw_o_rf=n_unc/n_tot; gw_o_spl=(P*len(H)+1.0*n_unc)/n_tot
        print(f"  end-to-end overall: RF argmax {gw_o_rf:.3f} -> RF+splice reranking {gw_o_spl:.3f}  ({gw_o_spl-gw_o_rf:+.3f})",file=sys.stderr)
        pd.DataFrame([
            dict(metric="rf_argmax",value=gw_o_rf),
            dict(metric="rf_plus_splice",value=gw_o_spl),
            dict(metric="n_genes",value=n_tot),
        ]).to_csv(f"{a.out}_genomewide_endtoend.tsv",sep="\t",index=False)
        print(f"  per chromosome:",file=sys.stderr)
        for c,nn,pr,plo,phi,rnd in per_chrom:
            print(f"    {c:<7} n={nn:<5} P@1={pr:.3f} [{plo:.3f},{phi:.3f}]",file=sys.stderr)
        pd.DataFrame(per_chrom,columns=["chrom","n","p1","ci_lo","ci_hi","random"]).to_csv(f"{a.out}_genomewide.tsv",sep="\t",index=False)

    if a.gene_split_check:
        pool=pd.concat([tr,te],ignore_index=True)
        for c in SPL: pool[c]=pd.to_numeric(pool[c],errors="coerce").fillna(0.0)
        pool["is_canonical"]=pool["is_canonical"].astype(bool)
        genes=np.array(sorted(pool.gene.unique()))
        rng=np.random.default_rng(0); rng.shuffle(genes)
        k=5; folds=np.array_split(genes,k)
        hh=[]; fl=[]
        for i in range(k):
            te_g=set(folds[i]); tr_g=set(np.concatenate([folds[j] for j in range(k) if j!=i]))
            trd=pool[pool.gene.isin(tr_g)]; ted=pool[pool.gene.isin(te_g)]
            if trd["is_canonical"].sum()==0 or ted.gene.nunique()==0: continue
            pred=fit_model(a.model,trd[SPL].to_numpy(),trd["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
            d2=ted.copy(); d2["_s"]=pred(ted[SPL].to_numpy())
            hh.append(p1_from_scores(d2,"_s"))
            fl.append(1.0/ted.groupby("gene").size().to_numpy())
        H=np.concatenate(hh); F=np.concatenate(fl)
        P,LO,HI=wilson(int(H.sum()),len(H)); floor=float(F.mean())
        print(f"\n[GENE-SPLIT leak check, 5-fold by gene, n={len(H)}]",file=sys.stderr)
        print(f"  splice P@1={P:.3f} [{LO:.3f},{HI:.3f}]  floor={floor:.3f}  boost={P-floor:+.3f}",file=sys.stderr)
        print(f"  compare to chromosome-split contested boost (+0.10ish). If similar, the "
              f"features are NOT encoding gene identity; if the boost collapses, they leak.",file=sys.stderr)

    if a.lean:
        predf=fit_model(a.model,tr[SPL].to_numpy(),tr["is_canonical"].astype(int).to_numpy(),SPL,a.monotone)
        base=p1_from_scores(te.assign(_s=predf(te[SPL].to_numpy())),"_s").mean()
        imp=permutation_importance(te,SPL,predf,base,n_rep=5)
        keep=[f for f,d,s in imp if d>0.001]
        if len(keep)<2: keep=[f for f,_,_ in imp[:6]]
        predk=fit_model(a.model,tr[keep].to_numpy(),tr["is_canonical"].astype(int).to_numpy(),keep,a.monotone)
        d2=te.copy(); d2["_s"]=predk(te[keep].to_numpy()); hk=p1_from_scores(d2,"_s")
        P,LO,HI=wilson(int(hk.sum()),len(hk)); rnd=float((1.0/te.groupby("gene").size()).mean())
        print(f"\n[LEAN model: {len(keep)} features with importance>0.001]",file=sys.stderr)
        print(f"  {keep}",file=sys.stderr)
        print(f"  contested test P@1={P:.3f} [{LO:.3f},{HI:.3f}]  (full-25 was {h_spl.mean():.3f}; "
              f"floor {rnd:.3f})",file=sys.stderr)

    if a.capacity_sweep:
        from sklearn.ensemble import HistGradientBoostingClassifier
        _sweep_rows=[]
        print(f"\n[capacity sweep] if test is flat while train rises => at ceiling (not underfit)",file=sys.stderr)
        print(f"  {'depth':>5}{'iters':>7}{'leaf':>6}{'train':>8}{'test':>8}{'gap':>8}",file=sys.stderr)
        ytr=tr['is_canonical'].astype(int).to_numpy()
        for depth,iters,leaf in [(3,400,50),(5,600,30),(7,800,20),(9,1200,10),(None,2000,5)]:
            mc=[int(DIRP.get(f,0)) for f in SPL] if a.monotone else None
            clf=HistGradientBoostingClassifier(max_depth=depth,max_iter=iters,learning_rate=0.05,
                min_samples_leaf=leaf,class_weight='balanced',monotonic_cst=mc,random_state=0)
            clf.fit(tr[SPL].to_numpy(),ytr)
            f=lambda X: clf.predict_proba(X)[:,1]
            ptr=p1_from_scores(tr.assign(_s=f(tr[SPL].to_numpy())),'_s').mean()
            pte=p1_from_scores(te.assign(_s=f(te[SPL].to_numpy())),'_s').mean()
            ds="None" if depth is None else str(depth)
            print(f"  {ds:>5}{iters:>7}{leaf:>6}{ptr:>8.3f}{pte:>8.3f}{ptr-pte:>+8.3f}",file=sys.stderr)
            _sweep_rows.append(dict(depth=ds,iters=iters,leaf=leaf,train=ptr,test=pte))
        print(f"  (read: TEST column. If it does not rise with capacity, the {p_spl[0]:.3f} is "
              f"a real ceiling, consistent with the tissue-specificity result.)",file=sys.stderr)
        pd.DataFrame(_sweep_rows).assign(capacity=range(1,len(_sweep_rows)+1),
            label=[r["depth"] if r["depth"]!="None" else "dInf" for r in _sweep_rows]
            ).to_csv(f"{a.out}_capacity_sweep.tsv",sep="\t",index=False)

    out=te[["gene","chrom","center","rf_prob","is_canonical","n_internal","splice_score"]].copy()
    out.to_parquet(f"{a.out}_test_scored.parquet",index=False)
    pd.DataFrame([
        dict(metric="n_contested_test",value=n),
        dict(metric="n_endtoend_test",value=n_total),
        dict(metric="contested_p1_random_floor",value=p_rnd[0],ci_lo=p_rnd[1],ci_hi=p_rnd[2]),
        dict(metric="contested_p1_splice",value=p_spl[0],ci_lo=p_spl[1],ci_hi=p_spl[2]),
        dict(metric="splice_minus_random",value=d_sr,ci_lo=lo_sr,ci_hi=hi_sr),
        dict(metric="contested_rf_argmax_DEGENERATE",value=p_cpa_deg),
        dict(metric="overall_rf_argmax",value=o_cpa),
        dict(metric="overall_shortlist_random",value=o_rnd),
        dict(metric="overall_shortlist_splice",value=o_spl),
        dict(metric="gain_narrowing_part",value=o_rnd-o_cpa),
        dict(metric="gain_splice_part",value=o_spl-o_rnd),
    ]).to_csv(f"{a.out}_metrics.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_test_scored.parquet, {a.out}_metrics.tsv",file=sys.stderr)

if __name__=="__main__": main()


