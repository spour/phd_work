#!/usr/bin/env python3
# find the used 3' end among cpa-competent candidates, labeled by measured usage
import argparse, os, sys, re, time, tempfile
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

t0=time.time(); LOG=lambda m: print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

def read_chroms(p): 
    return set(x.strip() for x in open(p) if x.strip())

# conditional logit (gene-stratified choice)
def seg_softmax_ll(eta, starts, counts, true_pos):
    gmax=np.maximum.reduceat(eta,starts); gmax_row=np.repeat(gmax,counts)
    ex=np.exp(eta-gmax_row); gsum=np.add.reduceat(ex,starts)
    p=ex/np.repeat(gsum,counts); ll=eta[true_pos]-(np.log(gsum)+gmax)
    return p,ll
def fit_clogit(X,starts,counts,true_pos,y):
    def ng(b):
        eta=X@b; p,ll=seg_softmax_ll(eta,starts,counts,true_pos)
        return -ll.sum(), X.T@(p-y)
    return minimize(ng,np.zeros(X.shape[1]),jac=True,method="L-BFGS-B",
                    options=dict(maxiter=500)).x
def logit(p,eps=1e-6):
    p=np.clip(np.asarray(p,float),eps,1-eps); return np.log(p/(1-p))

def build_groups(df):
    df=df.sort_values("gene",kind="stable").reset_index(drop=True)
    codes=pd.factorize(df["gene"].to_numpy())[0]
    starts=np.r_[0,np.flatnonzero(np.diff(codes))+1]; counts=np.diff(np.r_[starts,len(df)])
    usage=df["usage"].to_numpy()
    keep_rows=[]; k_starts=[]; k_counts=[]; k_true=[]; cur=0
    for s,c in zip(starts,counts):
        u=usage[s:s+c]
        if c>=2 and np.nanmax(u)>0:
            keep_rows.extend(range(s,s+c)); k_starts.append(cur); k_counts.append(c)
            k_true.append(cur+int(np.nanargmax(u))); cur+=c
    dfk=df.iloc[keep_rows].reset_index(drop=True)
    return (dfk, np.array(k_starts), np.array(k_counts), np.array(k_true),
            list(zip(k_starts,k_counts)))

def contested_mask(dfk, pig):
    out=[]
    rf=dfk["rf_prob"].to_numpy(); us=dfk["usage"].to_numpy()
    for (s,c) in pig:
        u=us[s:s+c]; r=rf[s:s+c]; ti=int(np.argmax(u))
        out.append(bool((np.delete(r,ti)>=r[ti]).any()))
    return np.array(out)

def p1_by_score(dfk, pig, score):
    hit=[]; rnd=[]
    us=dfk["usage"].to_numpy(); s=dfk[score].to_numpy()
    for (a,c) in pig:
        u=us[a:a+c]; sc=s[a:a+c]; ti=int(np.argmax(u))
        hit.append(int(np.argmax(sc)==ti)); rnd.append(1.0/c)
    return np.array(hit), np.array(rnd)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--scripts_dir", required=True)
    ap.add_argument("--pwms", required=True)
    ap.add_argument("--cache", default="train/data/choice_features_cache.parquet")
    ap.add_argument("--cpa_window", type=int, default=500)
    ap.add_argument("--up", type=int, default=3000)
    ap.add_argument("--dn", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--center_offset", type=int, default=0,
                    help="PAS bed is the cleavage site; window centered on it directly")
    ap.add_argument("--max_genes", type=int, default=0, help="0 = all genes")
    ap.add_argument("--bg", nargs=4, type=float, default=[0.3,0.2,0.2,0.3])
    ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    D=a.data_dir

    if os.path.exists(a.cache):
        LOG(f"loading cached features {a.cache}")
        feat=pd.read_parquet(a.cache)
    else:
        LOG("building choice dataset from polyADB PSE + CPCCPA")
        # polyadb pas with measured usage
        pa=pd.read_csv(f"{D}/polyadb_v4/HumanPas/hg38.PAS.main.tsv",sep="\t")
        s=pa["PAS_ID"].str.split(":",expand=True)
        pa["chrom"]=s[0]; pa["strand"]=s[1]; pa["pos"]=s[2].astype(int)
        pa["gene"]=pa["GeneSymbol"].astype(str)
        for c in ["PSE.3READS","PSE.ENCODE.PB","PSE.GTEx.ONT"]:
            pa[c]=pd.to_numeric(pa[c],errors="coerce")
        pa["usage"]=pa[["PSE.3READS","PSE.ENCODE.PB","PSE.GTEx.ONT"]].mean(axis=1,skipna=True)
        pa=pa.dropna(subset=["usage"])
        pa=pa[["gene","chrom","strand","pos","usage"]]; pa["src"]="pas"
        # cpccpa competitors (usage 0)
        cp=pd.read_csv(f"{D}/cpccpa_3model_intragenic.bed",sep="\t",header=None,
                       names=["chrom","start","end","name","score","strand"])
        cp["gene"]=cp["name"].astype(str).str.split("_").str[-1]
        cp["pos"]=((cp["start"]+cp["end"])//2).astype(int)
        cp["usage"]=0.0; cp["src"]="cpccpa"
        cp=cp[["gene","chrom","strand","pos","usage","src"]]
        cand=pd.concat([pa,cp],ignore_index=True)
        cand["chrom"]=cand["chrom"].astype(str)
        cand["chrom"]=cand["chrom"].where(cand["chrom"].str.startswith("chr"),"chr"+cand["chrom"])
        cand=cand.sort_values("usage",ascending=False).drop_duplicates(["gene","chrom","pos"])
        vc=cand.gene.value_counts(); keep=set(vc[vc>=2].index)
        cand=cand[cand.gene.isin(keep)].reset_index(drop=True)
        if a.max_genes>0:
            g=pd.Series(sorted(cand.gene.unique()))
            cand=cand[cand.gene.isin(set(g.sample(min(a.max_genes,len(g)),random_state=a.seed)))]
        LOG(f"candidates {len(cand):,} across {cand.gene.nunique():,} genes "
            f"(pas {(cand.src=='pas').sum():,}, cpccpa {(cand.src=='cpccpa').sum():,})")

        # featurize each candidate once: cpa (pwm) + splice axes
        from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup,
                                               init_maxentscan, extract_one_site)
        from compute_gene_baselines import derive_axes
        from featurization import read_pwms, featurize_fasta_to_df
        fa=GenomeFasta(f"{D}/hg38.fa")
        u1=load_u1_lookup(f"{D}/u1_lookup.npy") if os.path.exists(f"{D}/u1_lookup.npy") else None
        me5=load_me5_lookup(f"{D}/me5_lookup.npy") if os.path.exists(f"{D}/me5_lookup.npy") else None
        _,score3,_,m3=init_maxentscan()
        pwms=read_pwms(a.pwms,max_len=20,bg=np.array(a.bg))
        AXES=["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt",
              "local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max"]

        half=a.cpa_window//2; ids=[]; seqs=[]; ax_rows=[]
        chroms=cand["chrom"].to_numpy(); poss=cand["pos"].to_numpy(); sts=cand["strand"].to_numpy()
        LOG("extracting + splice-featurizing candidates...")
        for i in range(len(cand)):
            st=sts[i]; pos=int(poss[i]); ch=chroms[i]
            c=pos+a.center_offset if st=="+" else pos-a.center_offset
            cseq=fa.fetch(ch,max(0,c-half),c+half,strand=st)
            ids.append(f"row{i}"); seqs.append(cseq if cseq else "N"*a.cpa_window)
            if st=="+": w0,w1=max(0,c-a.up),c+a.dn
            else:       w0,w1=max(0,c-a.dn),c+a.up
            sseq=fa.fetch(ch,w0,w1,strand=st)
            try:
                feat=extract_one_site(sseq,a.up,a.dn,a.n_bins,me5,score3,m3,u1)
                ax=derive_axes(feat,sseq,a.up)
                ax_rows.append([float(ax.get(k,np.nan)) if ax.get(k,None) is not None else np.nan for k in AXES])
            except Exception:
                ax_rows.append([np.nan]*len(AXES))
            if (i+1)%20000==0: LOG(f"  {i+1:,}/{len(cand):,} featurized")
        tmp=tempfile.NamedTemporaryFile("w",suffix=".fa",delete=False)
        for cid,seq in zip(ids,seqs): tmp.write(f">{cid}\n{seq}\n")
        tmp.close()
        cpa_df=featurize_fasta_to_df(tmp.name,pwms,bg=np.array(a.bg)); os.unlink(tmp.name)
        cpa_df=cpa_df.reindex([f"row{i}" for i in range(len(cand))]).reset_index(drop=True)
        ax_df=pd.DataFrame(ax_rows,columns=AXES)
        feat=pd.concat([cand.reset_index(drop=True),
                        ax_df,
                        cpa_df.add_prefix("CPA__")],axis=1)
        feat["rf_prob"]=np.nan
        feat.to_parquet(a.cache,index=False)
        LOG(f"cached features -> {a.cache}  shape {feat.shape}")

    if feat["rf_prob"].isna().all():
        pt=pd.read_parquet(f"{D}/peak_table_8axes_rank.parquet")
        pt["chrom"]=pt["chrom"].astype(str)
        pt["chrom"]=pt["chrom"].where(pt["chrom"].str.startswith("chr"),"chr"+pt["chrom"])
        by={g:(d["position"].to_numpy(),d["rf_prob"].to_numpy()) for g,d in pt.groupby("gene")}
        rf=np.full(len(feat),np.nan)
        gv=feat["gene"].to_numpy(); pv=feat["pos"].to_numpy()
        for i in range(len(feat)):
            g=gv[i]
            if g in by:
                pos,pr=by[g]; j=int(np.argmin(np.abs(pos-pv[i])))
                if abs(int(pos[j])-int(pv[i]))<=500: rf[i]=pr[j]
        feat["rf_prob"]=rf
    feat["rf_prob"]=feat["rf_prob"].fillna(0.0)
    feat["rf_logit"]=logit(feat["rf_prob"])

    TRAIN=read_chroms(f"{D}/../train_chroms.txt") if os.path.exists(f"{D}/../train_chroms.txt") else \
          {"chr1","chr2","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr14","chr15","chr16","chr17"}
    TEST =read_chroms(f"{D}/../test_chroms.txt") if os.path.exists(f"{D}/../test_chroms.txt") else \
          {"chr18","chr19","chr20","chr21","chr22","chrX"}
    feat["chrom"]=feat["chrom"].astype(str)
    AXES=["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt","local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max"]
    CPA_COLS=[c for c in feat.columns if c.startswith("CPA__")]
    for c in AXES+CPA_COLS+["rf_logit"]: feat[c]=feat[c].fillna(0.0)

    tr=feat[feat.chrom.isin(TRAIN)].copy(); te=feat[feat.chrom.isin(TEST)].copy()
    trk,s_tr,c_tr,tp_tr,pig_tr=build_groups(tr)
    tek,s_te,c_te,tp_te,pig_te=build_groups(te)
    LOG(f"train choice-genes {len(pig_tr):,}  test choice-genes {len(pig_te):,}")
    con=contested_mask(tek,pig_te)
    LOG(f"contested test genes: {con.sum():,}/{len(con):,}")

    rf_hit,rnd=p1_by_score(tek,pig_te,"rf_prob")
    def show(tag,hit):
        print(f"  {tag:<18} all_P@1={hit.mean():.3f}  contested_P@1={hit[con].mean():.3f}  "
              f"(random {rnd.mean():.3f}/{rnd[con].mean():.3f})")
    print("\n=== USED-END RECOVERY (held-out test) ===")
    show("rf_baseline",rf_hit)

    def clogit_eval(feature_list):
        mean=trk[feature_list].to_numpy().mean(0); sd=trk[feature_list].to_numpy().std(0); sd[sd==0]=1
        Xtr=(trk[feature_list].to_numpy()-mean)/sd; ytr=np.zeros(len(trk)); ytr[tp_tr]=1.0
        b=fit_clogit(Xtr,s_tr,c_tr,tp_tr,ytr)
        Xte=(tek[feature_list].to_numpy()-mean)/sd
        eta=Xte@b; 
        hit=[]; lls=[]
        for (s,c) in pig_te:
            e=eta[s:s+c]; ti=int(np.argmax(tek["usage"].to_numpy()[s:s+c]))
            m=e.max(); lse=np.log(np.exp(e-m).sum())+m
            lls.append(float(e[ti]-lse)); hit.append(int(np.argmax(e)==ti))
        return np.array(hit),np.array(lls),b
    h0,ll0,_=clogit_eval(["rf_logit"])
    h1,ll1,b1=clogit_eval(["rf_logit","upstream_max_me3"])
    show("clogit_cpa",h0); show("clogit_cpa+me3",h1)
    bits_all=(ll1-ll0).mean()/np.log(2); bits_con=((ll1-ll0)[con]).mean()/np.log(2)
    print(f"  >>> held-out info gain (CPA+me3 vs CPA): all {bits_all:+.4f} bits, "
          f"contested {bits_con:+.4f} bits")

    def gbt_eval(cols,labels_tr=None):
        Xtr=trk[cols].fillna(0).to_numpy()
        ytr=np.zeros(len(trk),int); 
        if labels_tr is None: ytr[tp_tr]=1
        else: ytr=labels_tr
        clf=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,
             max_leaf_nodes=63,random_state=a.seed)
        clf.fit(Xtr,ytr)
        sc=clf.predict_proba(tek[cols].fillna(0).to_numpy())[:,1]
        tek2=tek.assign(_s=sc); hit=[]
        for (s,c) in pig_te:
            sub=tek2["_s"].to_numpy()[s:s+c]; ti=int(np.argmax(tek["usage"].to_numpy()[s:s+c]))
            hit.append(int(np.argmax(sub)==ti))
        return np.array(hit),clf
    hg,clf=gbt_eval(["rf_prob"]+AXES+CPA_COLS)
    show("gbt_cpa+splice",hg)

    rng=np.random.default_rng(a.seed)
    ysh=np.zeros(len(trk),int)
    for (s,c) in pig_tr: ysh[s+int(rng.integers(c))]=1
    hsh,_=gbt_eval(["rf_prob"]+AXES+CPA_COLS,labels_tr=ysh)
    print(f"\n[shuffle null] GBT contested P@1 on shuffled labels = {hsh[con].mean():.3f} "
          f"(should ~ random {rnd[con].mean():.3f})")

    print("\n=== VERDICT ===")
    print(f"  rf_baseline contested P@1 = {rf_hit[con].mean():.3f}")
    print(f"  best model  contested P@1 = {max(h1[con].mean(),hg[con].mean()):.3f}")
    print(f"  held-out info gain        = {bits_con:+.4f} bits (contested)")
    print(f"  shuffle null clean?       = {'YES' if abs(hsh[con].mean()-rnd[con].mean())<0.02 else 'NO'}")
    print("  Beat the RF iff best contested P@1 > rf_baseline AND null is clean.")

if __name__=="__main__":
    main()