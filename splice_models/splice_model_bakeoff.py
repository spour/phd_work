#!/usr/bin/env python3
import math
"""
splice_model_bakeoff.py

Find a splicing scorer that improves CPA gene-end identification on the TEST
chromosomes by a non-trivial, honest amount. Trains on train chroms, SELECTS the
winner on val chroms, reports test once. Test-set selection is forbidden here on
purpose: a number you tuned on test is not a result.

Searches a grid of (architecture x feature subset). For each candidate:
  splice_prob = model(features)              fit on train (canonical vs ALL internal RF peaks)
  combiner    = logit(rf), logit(splice)     fit on val
  gated       = rf if rf-confident-and-winning else combiner   (deployable; gate tuned on val)
Reports on test with bootstrap CIs:
  cpa_only P@1 (X), splice_only, combiner, gated (Y), paired lift Y-X,
  union-oracle ceiling, canonical-margin recovery, length-stratified lift,
  and a feature-direction audit vs mechanistic priors.

Winner = highest VAL gated P@1, tie-broken by val lift then direction sanity.
Its TEST row is the reportable result.

INPUT peak_table (parquet/tsv): one row per RF peak, columns
  gene, chrom, position, rf_prob, in_zone(1=canonical/0=internal), gene_length
  + feature columns (default: every col ending __wg; must be label-free).
If no __wg cols exist, pass --feature_cols with the model-output columns you do
have (e.g. splicing_prob lr_prob cnn_prob aparent2_prob) and it searches
combiners over those instead of training new splicing models.
"""
import argparse, os, sys, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
TEST_CHROMS = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS  = {"chr3","chr13"}
LEN_BINS = [0,5000,20000,50000,100000,500000,5e8]
LEN_LABELS = ["<5kb","5-20kb","20-50kb","50-100kb","100-500kb",">500kb"]
MBINS = [-1.01,-0.2,-0.05,0.05,0.2,1.01]
MLABELS = ["internal_wins_clear","internal_wins_narrow","tied",
           "canonical_wins_narrow","canonical_wins_clear"]
CONTESTED_MARGIN = 0.0
TOPK_VALUES = [2,3,5]

DIRECTION_PRIOR = {
    "prox_u1_sum_t200__wg": -1, "prox_u1_sum_t400__wg": -1, "prox_u1_max__wg": -1,
    "upstream_max_me3__wg": +1, "term_3ss__wg": +1, "term_me3__wg": +1,
    "dn5ss_has_strong__wg": -1, "dn5ss_max_me5__wg": -1,
    "bp_best_score__wgp": +1, "py_tract_u2af__wgp": +1, "bp_py_colocal__wgp": +1,
    "term_ppt__wg": +1,
}

def safe_logit(p):
    p = np.clip(np.asarray(p,float), EPS, 1-EPS); return np.log(p/(1-p))

def assign_split(df):
    if "split" in df.columns: return df
    c = df["chrom"].astype(str)
    c = c.where(c.str.startswith("chr"), "chr"+c).to_numpy()
    s = np.full(len(df),"train",object)
    s[np.isin(c,list(VAL_CHROMS))]="val"; s[np.isin(c,list(TEST_CHROMS))]="test"
    df=df.copy(); df["split"]=s; return df

def feature_subsets(all_feats):
    subs = {"all": all_feats}
    core = [c for c in all_feats if c in DIRECTION_PRIOR]
    if core and set(core)!=set(all_feats): subs["mechanistic_core"]=core
    u1 = [c for c in all_feats if "u1" in c.lower()]
    me = [c for c in all_feats if "me3" in c.lower() or "3ss" in c.lower()]
    if u1 and me: subs["u1_plus_terminal"]=sorted(set(u1+me))
    compact=[c for c in ("upstream_max_me3__wg","prox_u1_sum_t200__wg") if c in all_feats]
    if len(compact)==2: subs["compact2"]=compact
    bp=[c for c in all_feats if c.startswith(("bp_","py_"))]
    if bp: subs["bp_py_only"]=sorted(bp)
    if me and bp: subs["terminal_full"]=sorted(set(me+bp))
    if u1 and bp and me: subs["u1_terminal_full"]=sorted(set(u1+me+bp))
    term=[c for c in all_feats if c in ("term_3ss__wg","term_ppt__wg","upstream_max_me3__wg")]
    if len(term)>=2: subs["terminal_exon"]=sorted(term)
    return subs

def make_arch(kind, monotonic_cst=None):
    if kind=="logistic":
        return LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000)
    if kind=="gbt":
        return HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05,
            min_samples_leaf=200,l2_regularization=1.0,class_weight="balanced",random_state=0,
            monotonic_cst=monotonic_cst)
    if kind=="mlp":
        return MLPClassifier(hidden_layer_sizes=(16,),alpha=1e-2,max_iter=500,
            early_stopping=True,random_state=0)
    raise ValueError(kind)

def fit_predict(kind, Xtr, ytr, monotonic_cst=None):
    est = make_arch(kind, monotonic_cst=monotonic_cst if kind=="gbt" else None)
    if kind in ("logistic","mlp"):
        sc = StandardScaler().fit(Xtr); est.fit(sc.transform(Xtr), ytr)
        if kind=="logistic" and monotonic_cst is not None:
            co=est.coef_[0].copy(); mc=np.asarray(monotonic_cst,float)
            bad=(mc!=0)&(np.sign(co)!=np.sign(mc))
            if bad.any():
                co[bad]=0.0; est.coef_[0]=co
        return (lambda X: est.predict_proba(sc.transform(X))[:,1]), est
    est.fit(Xtr,ytr)
    return (lambda X: est.predict_proba(X)[:,1]), est

# per-gene scoring
def top1(df, col):
    rec=[]
    for gene,g in df.groupby("gene",sort=False):
        gl=float(g["gene_length"].iloc[0])
        if g["in_zone"].sum()==0: rec.append((gene,0,gl)); continue
        rec.append((gene,int(g.iloc[int(np.argmax(g[col].to_numpy()))]["in_zone"]==1),gl))
    return pd.DataFrame(rec,columns=["gene","hit","gene_length"])

def gated_hit(df, w, hi, mm):
    rec=[]
    for gene,g in df.groupby("gene",sort=False):
        gl=float(g["gene_length"].iloc[0])
        if g["in_zone"].sum()==0: rec.append((gene,0,gl)); continue
        rf=g["rf_prob"].to_numpy(); sp=g["splice_prob"].to_numpy()
        pr=np.sort(rf)[::-1]; top=float(pr[0]); marg=top-(float(pr[1]) if len(pr)>1 else 0.0)
        s = rf if (top>=hi and marg>=mm) else w[0]*safe_logit(rf)+w[1]*safe_logit(sp)
        rec.append((gene,int(g.iloc[int(np.argmax(s))]["in_zone"]==1),gl))
    return pd.DataFrame(rec,columns=["gene","hit","gene_length"])

def boot(v,B=4000,seed=0):
    v=np.asarray(v,float)
    if len(v)==0: return (np.nan,np.nan,np.nan)
    r=np.random.default_rng(seed); m=v[r.integers(0,len(v),(B,len(v)))].mean(1)
    return float(v.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))

def paired(a,b,B=4000,seed=0):
    d=np.asarray(a,float)-np.asarray(b,float)
    if len(d)==0: return (np.nan,np.nan,np.nan)
    r=np.random.default_rng(seed); m=d[r.integers(0,len(d),(B,len(d)))].mean(1)
    return float(d.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))

def cmargin(df):
    out={}
    for gene,g in df.groupby("gene",sort=False):
        if g["in_zone"].sum()==0: out[gene]=np.nan; continue
        intr=g[g["in_zone"]==0]
        out[gene]=np.nan if len(intr)==0 else float(g[g["in_zone"]==1]["rf_prob"].max())-float(intr["rf_prob"].max())
    return out

def pick_gate(va, w):
    best,bp1=(0.95,0.1),-1
    for hi in [0.7,0.8,0.85,0.9,0.95,0.99]:
        for mm in [0.0,0.05,0.1,0.15,0.2,0.3]:
            p1=float(gated_hit(va,w,hi,mm)["hit"].mean())
            if p1>bp1: bp1,best=p1,(hi,mm)
    return best[0],best[1],bp1

def audit(kind, est, feats, tr):
    rows=[]
    if kind=="logistic":
        for c,wt in zip(feats, np.asarray(est.coef_).reshape(-1)):
            pr=DIRECTION_PRIOR.get(c); obs=int(np.sign(wt))
            rows.append((c,obs,pr,pr is None or obs==pr))
    else:
        can=tr[tr["in_zone"]==1]; ii=tr[tr["in_zone"]==0]
        for c in feats:
            obs=int(np.sign(float(can[c].mean())-float(ii[c].mean()))); pr=DIRECTION_PRIOR.get(c)
            rows.append((c,obs,pr,pr is None or obs==pr))
    return rows

def atorabove_rerank_hit(df, margin=0.0, cap=None, by="splice", w=None):
    rec=[]
    for gene,g in df.groupby("gene",sort=False):
        gl=float(g["gene_length"].iloc[0]); iz=g["in_zone"].to_numpy()
        if iz.sum()==0: rec.append((gene,0,gl,0)); continue
        rf=g["rf_prob"].to_numpy(); sp=g["splice_prob"].to_numpy()
        crf=float(rf[iz==1].max())
        cand=np.where((iz==1)|(rf>=crf-margin))[0]
        if cap is not None and len(cand)>cap:
            cand=cand[np.argsort(rf[cand])[::-1][:cap]]
        if by=="splice":  s=sp[cand]
        elif by=="sum":   s=safe_logit(rf[cand])+safe_logit(sp[cand])
        else:             s=w[0]*safe_logit(rf[cand])+w[1]*safe_logit(sp[cand])
        win=cand[int(np.argmax(s))]
        rec.append((gene,int(iz[win]==1),gl,len(cand)))
    return pd.DataFrame(rec,columns=["gene","hit","gene_length","contest_size"])

def topk_rerank_hit(df, w, k, by="comb"):
    rec=[]
    for gene,g in df.groupby("gene",sort=False):
        gl=float(g["gene_length"].iloc[0])
        if g["in_zone"].sum()==0: rec.append((gene,0,gl)); continue
        rf=g["rf_prob"].to_numpy(); sp=g["splice_prob"].to_numpy(); iz=g["in_zone"].to_numpy()
        kk=min(k,len(rf))
        top=np.argpartition(rf,-kk)[-kk:]
        if by=="comb":   s=w[0]*safe_logit(rf[top])+w[1]*safe_logit(sp[top])
        elif by=="splice": s=sp[top]
        elif by=="sum":  s=safe_logit(rf[top])+safe_logit(sp[top])
        else:            s=rf[top]
        win=top[int(np.argmax(s))]
        rec.append((gene,int(iz[win]==1),gl))
    return pd.DataFrame(rec,columns=["gene","hit","gene_length"])

def hard_negatives(tr, margin=0.0):
    keep=[]
    for gene,g in tr.groupby("gene",sort=False):
        canon=g[g.in_zone==1]
        if len(canon)==0: continue
        crf=float(canon.rf_prob.max())
        keep.append(g[(g.in_zone==1)|(g.rf_prob>=crf-margin)])
    return pd.concat(keep,ignore_index=True) if keep else tr.iloc[0:0]

def evaluate(name, kind, feats, df, seed=0, hard_neg=False, hard_neg_margin=0.0, monotone=False):
    tr=df[df.split=="train"]; va=df[df.split=="val"].copy(); te=df[df.split=="test"].copy()
    tr_fit = hard_negatives(tr, hard_neg_margin) if hard_neg else tr
    mcst = [int(DIRECTION_PRIOR.get(f,0)) for f in feats] if monotone else None
    predict,est = fit_predict(kind, tr_fit[feats].to_numpy(), tr_fit["in_zone"].astype(int).to_numpy(), monotonic_cst=mcst)
    for s in (va,te): s["splice_prob"]=predict(s[feats].to_numpy())
    va_fit = hard_negatives(va, hard_neg_margin) if hard_neg else va
    Xv=np.column_stack([safe_logit(va_fit.rf_prob),safe_logit(va_fit.splice_prob)])
    comb=LogisticRegression(max_iter=1000).fit(Xv, va_fit["in_zone"].astype(int)); w=comb.coef_[0]
    hi,mm,val_g=pick_gate(va,w)
    best_k,val_topk=1,-1.0
    for k in TOPK_VALUES:
        p1=float(topk_rerank_hit(va,w,k,'comb')['hit'].mean())
        if p1>val_topk: val_topk,best_k=p1,k
    va_g=val_g
    te["comb"]=w[0]*safe_logit(te.rf_prob)+w[1]*safe_logit(te.splice_prob)
    hx=top1(te,"rf_prob"); hs=top1(te,"splice_prob"); hc=top1(te,"comb"); hg=gated_hit(te,w,hi,mm)
    hk=topk_rerank_hit(te,w,best_k,"comb"); hks=topk_rerank_hit(te,w,best_k,"splice")
    hksum=topk_rerank_hit(te,w,best_k,"sum")
    row={"candidate":name,"arch":kind,"n_feats":len(feats),
         "w_rf":float(w[0]),"w_splice":float(w[1]),"gate_hi":hi,"gate_mm":mm,
         "val_gated_p1":va_g,"val_topk_p1":val_topk,"best_k":best_k}
    for lab,h in [("cpa",hx),("splice",hs),("comb",hc),("gated",hg),("topk",hk),("topk_splice",hks),("topk_sum",hksum)]:
        m,lo,h2=boot(h["hit"],seed=seed); row[f"{lab}_p1"]=m; row[f"{lab}_lo"]=lo; row[f"{lab}_hi"]=h2
    dl,dlo,dhi=paired(hg["hit"],hx["hit"],seed=seed); row["lift"]=dl; row["lift_lo"]=dlo; row["lift_hi"]=dhi
    row["union_oracle"]=float(((hx.hit.to_numpy()==1)|(hs.hit.to_numpy()==1)).mean())
    a=audit(kind,est,feats,tr_fit); nb=sum(1 for _,_,p,_ in a if p is not None)
    row["dir_ok"]=f"{sum(1 for _,_,p,ok in a if p is not None and ok)}/{nb}"
    mp=cmargin(te)
    hgl=hg.copy(); hgl["lb"]=pd.cut(hgl.gene_length,LEN_BINS,labels=LEN_LABELS)
    hxl=hx.copy(); hxl["lb"]=pd.cut(hxl.gene_length,LEN_BINS,labels=LEN_LABELS)
    lrows=[]; worst=np.inf
    for b in LEN_LABELS:
        gy=hgl[hgl.lb==b]; gx=hxl[hxl.lb==b]
        if len(gx)==0: continue
        X=float(gx.hit.mean()); Y=float(gy.hit.mean()); lrows.append({"candidate":name,"length_bin":b,"n":len(gx),"cpa":X,"gated":Y,"lift":Y-X}); worst=min(worst,Y-X)
    row["worst_len_lift"]=float(worst)
    hsm=hs.copy(); hxm=hx.copy(); hsm["cm"]=hsm.gene.map(mp); hxm["cm"]=hxm.gene.map(mp)
    hsm["mb"]=pd.cut(hsm.cm,MBINS,labels=MLABELS); hxm["mb"]=pd.cut(hxm.cm,MBINS,labels=MLABELS)
    mrows=[]
    for b in MLABELS:
        gs=hsm[hsm.mb==b]; gx=hxm[hxm.mb==b]
        if len(gs)==0: continue
        mrows.append({"candidate":name,"margin_bin":b,"n":len(gs),"cpa":float(gx.hit.mean()),"splice":float(gs.hit.mean())})
    # the headline: contested genes (internal ties/beats canonical on rf)
    cm_series = hx.gene.map(mp)
    contested = (cm_series <= CONTESTED_MARGIN).to_numpy()
    hxc=hx[contested]; hgc=hk[contested]; hsc=hks[contested]; hcc=hc[contested]
    n_cont=int(contested.sum())
    if n_cont>0:
        cX=float(hxc.hit.mean()); cG=float(hgc.hit.mean()); cS=float(hsc.hit.mean()); cC=float(hcc.hit.mean())
        cdl,cdlo,cdhi=paired(hgc.hit,hxc.hit,seed=seed)
        hsumc=hksum[contested]
        sdl,sdlo,sdhi=paired(hsc.hit,hxc.hit,seed=seed)
        qdl,qdlo,qdhi=paired(hsumc.hit,hxc.hit,seed=seed)
        row["contested_splice_lift"]=sdl; row["contested_splice_lift_lo"]=sdlo; row["contested_splice_lift_hi"]=sdhi
        row["contested_sum"]=float(hsumc.hit.mean()); row["contested_sum_lift"]=qdl
        row["contested_sum_lift_lo"]=qdlo; row["contested_sum_lift_hi"]=qdhi
        row["contested_n"]=n_cont
        row["contested_cpa"]=cX; row["contested_splice"]=cS
        row["contested_comb"]=cC; row["contested_gated"]=cG
        row["contested_lift"]=cdl; row["contested_lift_lo"]=cdlo; row["contested_lift_hi"]=cdhi
    else:
        for k in ("contested_n","contested_cpa","contested_splice","contested_comb",
                  "contested_gated","contested_lift","contested_lift_lo","contested_lift_hi"):
            row[k]=0.0
    crows=[]
    hb=hk.copy(); hb["lb"]=pd.cut(hb.gene_length,LEN_BINS,labels=LEN_LABELS); hb["cont"]=contested
    xb=hx.copy(); xb["lb"]=pd.cut(xb.gene_length,LEN_BINS,labels=LEN_LABELS); xb["cont"]=contested
    sb=hks.copy(); sb["lb"]=pd.cut(sb.gene_length,LEN_BINS,labels=LEN_LABELS); sb["cont"]=contested
    cb=hc.copy(); cb["lb"]=pd.cut(cb.gene_length,LEN_BINS,labels=LEN_LABELS); cb["cont"]=contested
    for b in LEN_LABELS:
        gx=xb[(xb.lb==b)&(xb.cont)]; gg=hb[(hb.lb==b)&(hb.cont)]
        gs=sb[(sb.lb==b)&(sb.cont)]; gc=cb[(cb.lb==b)&(cb.cont)]
        if len(gx)==0: continue
        crows.append({"candidate":name,"length_bin":b,"n_contested":len(gx),
                      "cpa":float(gx.hit.mean()),"splice":float(gs.hit.mean()),
                      "comb":float(gc.hit.mean()),"gated":float(gg.hit.mean()),
                      "lift":float(gg.hit.mean()-gx.hit.mean())})
    return row, pd.DataFrame(lrows), pd.DataFrame(mrows), a, pd.DataFrame(crows)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peak_table",required=True)
    ap.add_argument("--feature_cols",nargs="+",default=None)
    ap.add_argument("--archs",nargs="+",default=["logistic","gbt","mlp"])
    ap.add_argument("--out_dir",default=".")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--hard_neg",action="store_true",
                    help="train splice+combiner only on canonical + internal peaks "
                         "scoring >= canonical_rf - margin (the actual competitors)")
    ap.add_argument("--hard_neg_margin",type=float,default=0.0)
    ap.add_argument("--monotone",action="store_true",
                    help="constrain each feature to its DIRECTION_PRIOR sign "
                         "(GBT monotonic_cst; logistic sign-projection). Prevents the model "
                         "from using a feature against its mechanism.")
    a=ap.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    df=pd.read_parquet(a.peak_table) if a.peak_table.endswith(".parquet") else pd.read_csv(a.peak_table,sep="\t")
    df=assign_split(df)
    feats=a.feature_cols or [c for c in df.columns if c.endswith("__wg") or c.endswith("__wgp")]
    feats=[c for c in feats if c in df.columns]
    if not feats: sys.exit("[err] no feature columns; pass --feature_cols")
    nan_cells=int(df[feats].isna().sum().sum())
    if nan_cells:
        df[feats]=df[feats].fillna(0.0)
        print(f"[data] filled {nan_cells:,} NaN feature cells with 0 "
              f"(centered-feature neutral; mostly short genes lacking a baseline)",file=sys.stderr)
    for s in ["train","val","test"]:
        sub=df[df.split==s]; print(f"[data] {s}: {len(sub):,} peaks, {sub.gene.nunique():,} genes",file=sys.stderr)
    print(f"[data] {len(feats)} features; train canonical rate {df[df.split=='train'].in_zone.mean():.4f}",file=sys.stderr)
    if a.hard_neg:
        _h=hard_negatives(df[df.split=="train"], a.hard_neg_margin)
        print(f"[hard_neg] margin={a.hard_neg_margin}: train fit set {len(_h):,} rows, "
              f"{_h.gene.nunique():,} genes, positive rate {_h.in_zone.mean():.3f} "
              f"(canonical + internal peaks scoring >= canonical_rf - margin)",file=sys.stderr)

    subs=feature_subsets(feats)
    grid=list(itertools.product(a.archs, subs.items()))
    print(f"[grid] {len(grid)} candidates = {a.archs} x {list(subs)}",file=sys.stderr)

    summ,lens,margins,conts=[],[],[],[]
    for kind,(sname,scols) in grid:
        name=f"{kind}:{sname}"
        try:
            row,ldf,mdf,aud,cdf=evaluate(name,kind,scols,df,seed=a.seed,hard_neg=a.hard_neg,hard_neg_margin=a.hard_neg_margin,monotone=a.monotone)
        except Exception as e:
            print(f"[skip] {name}: {e}",file=sys.stderr); continue
        summ.append(row); lens.append(ldf); margins.append(mdf); conts.append(cdf)
        print(f"  {name:28s} k={int(row['best_k'])} val_topk={row['val_topk_p1']:.3f} | "
              f"TEST cpa={row['cpa_p1']:.3f} topk={row['topk_p1']:.3f} "
              f"topk_splice={row['topk_splice_p1']:.3f} | "
              f"CONTESTED(n={int(row['contested_n'])}) cpa={row['contested_cpa']:.3f} "
              f"topk={row['contested_gated']:.3f} "
              f"lift={row['contested_lift']:+.3f}[{row['contested_lift_lo']:+.3f},{row['contested_lift_hi']:+.3f}] "
              f"dir={row['dir_ok']}",file=sys.stderr)

    S=pd.DataFrame(summ)
    S.to_csv(f"{a.out_dir}/bakeoff_summary.tsv",sep="\t",index=False)
    if lens: pd.concat(lens,ignore_index=True).to_csv(f"{a.out_dir}/bakeoff_length.tsv",sep="\t",index=False)
    if margins: pd.concat(margins,ignore_index=True).to_csv(f"{a.out_dir}/bakeoff_canonical_margin.tsv",sep="\t",index=False)
    if conts: pd.concat(conts,ignore_index=True).to_csv(f"{a.out_dir}/bakeoff_contested_by_length.tsv",sep="\t",index=False)

    win=S.sort_values(["val_topk_p1","contested_lift_lo"],ascending=False).iloc[0]
    cpa=S["cpa_p1"].iloc[0]
    print(f"\n[winner|selected on val top-k] {win['candidate']}  (k={int(win['best_k'])})",file=sys.stderr)
    print(f"  TEST  cpa_only={cpa:.4f}  ->  topk_rerank={win['topk_p1']:.4f}  "
          f"(splice-among-topk={win['topk_splice_p1']:.4f})",file=sys.stderr)
    print(f"  splice_only(all-peaks)={win['splice_p1']:.4f}  union_oracle={win['union_oracle']:.4f}  "
          f"directions={win['dir_ok']}",file=sys.stderr)
    sig = win["contested_lift_lo"]>0
    print(f"  contested lift CI excludes 0: {sig}  ->  "
          f"{'splice reranks non-trivially on contested genes' if sig else 'no reranking signal'}",file=sys.stderr)
    # adaptive at-or-above contest-size sweep (winner candidate, splice-among)
    wname=win["candidate"]; wkind=wname.split(":")[0]; wsub=wname.split(":")[1]
    _tr=df[df.split=="train"]; _te=df[df.split=="test"].copy()
    _trf = hard_negatives(_tr, a.hard_neg_margin) if a.hard_neg else _tr
    _subs=feature_subsets(feats)
    _sc=_subs.get(wsub,feats)
    _pred,_=fit_predict(wkind,_trf[_sc].to_numpy(),_trf["in_zone"].astype(int).to_numpy())
    _te["splice_prob"]=_pred(_te[_sc].to_numpy())
    mp_te=cmargin(_te)
    print(f"  [adaptive contest sweep: canonical + internal sites >= canonical_rf - {a.hard_neg_margin}, splice picks]",file=sys.stderr)
    print(f"    {'cap':>5s}{'cont_n':>8s}{'recovery':>10s}{'ci':>16s}{'random':>9s}{'med_m':>9s}",file=sys.stderr)
    base_cm=_te.assign(cm=_te.gene.map(mp_te))
    contested_genes=set(base_cm[base_cm.cm<=CONTESTED_MARGIN].gene.unique())
    def _wilson(k,n,z=1.96):
        if n==0: return (0.0,0.0,0.0)
        p=k/n; d=1+z*z/n
        c=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        return p, max(0.0,c-half), min(1.0,c+half)
    sweep_rows=[]
    for cap in [2,3,5,10,None]:
        h=atorabove_rerank_hit(_te, margin=a.hard_neg_margin, cap=cap, by="splice")
        hc=h[h.gene.isin(contested_genes)]
        k=int(hc.hit.sum()); n=len(hc); p,lo,hi=_wilson(k,n)
        rnull=float((1.0/hc.contest_size).mean())
        lbl="all" if cap is None else str(cap)
        print(f"    {lbl:>5s}{n:>8d}{p:>10.3f}   [{lo:.3f},{hi:.3f}]   rnd={rnull:.3f}{hc.contest_size.median():>9.1f}",file=sys.stderr)
        sweep_rows.append({"cap":lbl,"contested_n":n,"recovered":k,"recovery":p,
                           "ci_lo":lo,"ci_hi":hi,"random_null":rnull,"median_contest_size":float(hc.contest_size.median())})
    pd.DataFrame(sweep_rows).to_csv(f"{a.out_dir}/sweep_recovery.tsv",sep="\t",index=False)
    hall=atorabove_rerank_hit(_te, margin=a.hard_neg_margin, cap=None, by="splice")
    hall=hall[hall.gene.isin(contested_genes)].copy()
    hall["lb"]=pd.cut(hall.gene_length,LEN_BINS,labels=LEN_LABELS)
    len_rows=[]
    for b in LEN_LABELS:
        gb=hall[hall.lb==b]
        if len(gb)==0: continue
        k=int(gb.hit.sum()); n=len(gb); p,lo,hi=_wilson(k,n)
        len_rows.append({"length_bin":b,"contested_n":n,"recovered":k,"recovery":p,"ci_lo":lo,"ci_hi":hi})
    pd.DataFrame(len_rows).to_csv(f"{a.out_dir}/sweep_recovery_by_length.tsv",sep="\t",index=False)
    print(f"    by length (cap=all): " + "  ".join(
        f"{r['length_bin']}:{r['recovery']:.3f}(n{r['contested_n']})" for r in len_rows), file=sys.stderr)

    # overall reranked p@1 on all test genes (same denominator as cpa_only)
    def overall_rerank(_df, margin, cap, seed=0):
        cpa_hit={}; rr_hit={}; rand_p={}
        rng=np.random.default_rng(seed)
        for gene,g in _df.groupby("gene",sort=False):
            iz=g["in_zone"].to_numpy()
            if iz.sum()==0: continue
            rf=g["rf_prob"].to_numpy(); sp=g["splice_prob"].to_numpy()
            cpa_hit[gene]=int(iz[int(np.argmax(rf))]==1)
            crf=float(rf[iz==1].max())
            cand=np.where((iz==1)|(rf>=crf-margin))[0]
            if cap is not None and len(cand)>cap:
                cand=cand[np.argsort(rf[cand])[::-1][:cap]]
            win=cand[int(np.argmax(sp[cand]))]
            rr_hit[gene]=int(iz[win]==1)
            rand_p[gene]=float(iz[cand].sum())/len(cand)
        genes=list(cpa_hit); cp=np.array([cpa_hit[x] for x in genes]); rr=np.array([rr_hit[x] for x in genes])
        rnd=float(np.mean([rand_p[x] for x in genes]))
        fixes=int(((cp==0)&(rr==1)).sum()); breaks=int(((cp==1)&(rr==0)).sum())
        return len(genes),float(cp.mean()),float(rr.mean()),rnd,fixes,breaks
    print(f"  [OVERALL P@1 on all scorable test genes (same denominator as cpa_only)]",file=sys.stderr)
    print(f"    {'cap':>5s}{'genes':>7s}{'cpa':>7s}{'random':>8s}{'splice':>8s}{'spl-rnd':>9s}{'fixes':>7s}{'breaks':>8s}",file=sys.stderr)
    ov_rows=[]
    for cap in [2,3,5,10,None]:
        N,cp,rr,rnd,fx,bk=overall_rerank(_te, a.hard_neg_margin, cap)
        lbl="all" if cap is None else str(cap)
        print(f"    {lbl:>5s}{N:>7d}{cp:>7.3f}{rnd:>8.3f}{rr:>8.3f}{rr-rnd:>+9.3f}{fx:>7d}{bk:>8d}",file=sys.stderr)
        ov_rows.append({"cap":lbl,"scorable_genes":N,"cpa_p1":cp,"random_p1":rnd,"splice_p1":rr,
                        "splice_minus_random":rr-rnd,"net_vs_cpa":rr-cp,"fixes":fx,"breaks":bk})
    pd.DataFrame(ov_rows).to_csv(f"{a.out_dir}/overall_rerank_p1.tsv",sep="\t",index=False)
    print(f"    NOTE: 'random' = pick uniformly among the candidate set (the honest control).",file=sys.stderr)
    print(f"          splice's real claim is the spl-rnd column, not cpa->splice.",file=sys.stderr)

    print(f"  [contested reranker variants, vs cpa=0.000]",file=sys.stderr)
    print(f"    fitted-combiner (w_rf={win['w_rf']:+.2f} w_splice={win['w_splice']:+.2f}): "
          f"{win['contested_gated']:.3f}  lift {win['contested_lift']:+.3f}  <- audit: RF weight may be inverted",file=sys.stderr)
    print(f"    splice-among-topk (RF only picks top-k): {win['contested_splice']:.3f}  "
          f"lift {win['contested_splice_lift']:+.3f} "
          f"[{win['contested_splice_lift_lo']:+.3f},{win['contested_splice_lift_hi']:+.3f}]  <- clean, no RF weight",file=sys.stderr)
    print(f"    fixed-sum logit(rf)+logit(splice): {win['contested_sum']:.3f}  "
          f"lift {win['contested_sum_lift']:+.3f} "
          f"[{win['contested_sum_lift_lo']:+.3f},{win['contested_sum_lift_hi']:+.3f}]  <- clean, RF positive",file=sys.stderr)
    print(f"\n[CONTESTED genes: internal site ties/beats canonical on RF]",file=sys.stderr)
    print(f"  overall (n={int(win['contested_n'])}): cpa={win['contested_cpa']:.3f} -> "
          f"gated={win['contested_gated']:.3f}  lift={win['contested_lift']:+.3f} "
          f"[{win['contested_lift_lo']:+.3f},{win['contested_lift_hi']:+.3f}]  "
          f"(splice_only={win['contested_splice']:.3f})",file=sys.stderr)
    cw=pd.concat(conts,ignore_index=True)
    cw=cw[cw.candidate==win['candidate']]
    if len(cw):
        print(f"  by gene length (contested genes only):",file=sys.stderr)
        print(f"    {'length':<10s}{'n':>7s}{'cpa':>8s}{'topk':>8s}{'lift':>9s}",file=sys.stderr)
        for _,r in cw.iterrows():
            print(f"    {r['length_bin']:<10s}{int(r['n_contested']):>7d}{r['cpa']:>8.3f}"
                  f"{r['gated']:>8.3f}{r['lift']:>+9.3f}",file=sys.stderr)
    print(f"\n[out] {a.out_dir}/bakeoff_summary.tsv (+ length, canonical_margin, contested_by_length)",file=sys.stderr)

if __name__=="__main__":
    main()