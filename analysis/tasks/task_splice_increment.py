#!/usr/bin/env python3
# correction to task_splice_rebuild.py
import csv, numpy as np
from collections import defaultdict
NFOLD=5; RIDGE=1.0
rng=np.random.default_rng(42)

R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t")); NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
comp=np.load("peak_compartment.npy"); F1=np.load("hypothesis_test_features.npy")
S=np.load("splice_feats_cache.npz")

CURRENT=F1[:,66:70]; COMP=F1[:,0:60]
spl_acc=F1[:,66]; spl_max5=F1[:,69]
Dmax_up=S["Dmax_up"]; Dmax_dn=S["Dmax_dn"]
strongdon=np.maximum(np.where(np.isfinite(Dmax_up),Dmax_up,-99.0),np.where(np.isfinite(Dmax_dn),Dmax_dn,-99.0))
den=np.where(np.abs(strongdon)<1.0,np.sign(strongdon)+(strongdon==0),strongdon)
FEAT={n:S[n] for n in ["A_ndon_up","A_ndon_dn","A_sumdon_up","A_dstrong_up_pA","A_dstrong_up_anchor",
                       "B_acc_rank2k","B_acc_margin2k","C_dist_3ss_up","C_dist_3ss_dn","C_dist_5ss_up",
                       "C_dist_5ss_dn","C_n3ss_up5k","C_n3ss_dn5k","C_n5ss_up5k","C_n5ss_dn5k","C_in_intron"]}
FEAT["B_acc_minus_max5"]=spl_acc-spl_max5; FEAT["B_acc_ratio_strongdon"]=spl_acc/den
A_names=["A_ndon_up","A_ndon_dn","A_sumdon_up","A_dstrong_up_pA","A_dstrong_up_anchor"]
B_names=["B_acc_minus_max5","B_acc_ratio_strongdon","B_acc_rank2k","B_acc_margin2k"]
C_names=["C_dist_3ss_up","C_dist_3ss_dn","C_dist_5ss_up","C_dist_5ss_dn","C_n3ss_up5k","C_n3ss_dn5k","C_n5ss_up5k","C_n5ss_dn5k","C_in_intron"]
def M(names): return np.column_stack([FEAT[n] for n in names]) if names else np.empty((NP,0))
A=M(A_names); Bm=M(B_names); Cm=M(C_names)
SPLSETS={"current":CURRENT,"current+A":np.column_stack([CURRENT,A]),"current+B":np.column_stack([CURRENT,Bm]),
         "current+C":np.column_stack([CURRENT,Cm]),"current+A+B+C":np.column_stack([CURRENT,A,Bm,Cm])}

have=np.isfinite(F1[:,0:60]).all(1)
allg=sorted(set(gene)); permf=np.random.default_rng(7).permutation(len(allg))
fold_of={g:permf[i]%NFOLD for i,g in enumerate(allg)}
def fit_ridge(X,yv,lam):
    P=np.eye(X.shape[1])*lam; P[0,0]=0.0; beta=np.zeros(X.shape[1])
    for _ in range(100):
        eta=np.clip(X@beta,-30,30); mu=1/(1+np.exp(-eta)); w=np.maximum(mu*(1-mu),1e-9)
        z=eta+(yv-mu)/w; XtW=X.T*w
        try: new=np.linalg.solve(XtW@X+P,XtW@z+P@beta)
        except np.linalg.LinAlgError: new=np.linalg.lstsq(XtW@X+P,XtW@z+P@beta,rcond=None)[0]
        if np.max(np.abs(new-beta))<1e-9: beta=new; break
        beta=new
    return beta
def auroc(yv,s):
    order=np.argsort(s,kind="mergesort"); sv=s[order]; n=len(s); rk=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sv[j+1]==sv[i]: j+=1
        rk[i:j+1]=(i+j)/2.0+1.0; i=j+1
    rr=np.empty(n); rr[order]=rk; npos=yv.sum(); nneg=n-npos
    return np.nan if npos==0 or nneg==0 else (rr[yv==1].sum()-npos*(npos+1)/2.0)/(npos*nneg)
def cv_pred(Xfull,rows):
    yv=isc[rows].astype(float); fold=np.array([fold_of[gene[i]] for i in rows]); pred=np.full(len(rows),np.nan); Xr=Xfull[rows]
    for f in range(NFOLD):
        tr=fold!=f; te=fold==f
        if tr.sum()==0 or te.sum()==0: continue
        Xtr=Xr[tr].astype(float).copy(); Xte=Xr[te].astype(float).copy()
        mu=np.nanmean(Xtr,0); sd=np.nanstd(Xtr,0); sd[sd==0]=1.0
        Xtr=np.where(np.isfinite(Xtr),Xtr,mu); Xte=np.where(np.isfinite(Xte),Xte,mu)
        Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
        Xtr=np.column_stack([np.ones(len(Xtr)),Xtr]); Xte=np.column_stack([np.ones(len(Xte)),Xte])
        beta=fit_ridge(Xtr,yv[tr],RIDGE); pred[te]=Xte@beta
    return yv,pred,np.array([gene[i] for i in rows])
def ggroups(genes):
    by=defaultdict(list)
    for k in range(len(genes)): by[genes[k]].append(k)
    return {g:np.array(v) for g,v in by.items()}
def p1_cred(yv,pred,gg):
    cred={}; chance={}
    for g,ix in gg.items():
        if (yv[ix]==1).sum()!=1 or len(ix)<2: continue
        sc=pred[ix]; mx=np.nanmax(sc)
        if not np.isfinite(mx): continue
        am=sc==mx; cred[g]=float((yv[ix][am]==1).sum())/am.sum(); chance[g]=1.0/len(ix)
    return cred,chance
def mrr_gene(yv,pred,gg):
    rr={}
    for g,ix in gg.items():
        if (yv[ix]==1).sum()!=1 or len(ix)<2: continue
        sc=pred[ix]; sca=sc[yv[ix]==1][0]
        if not np.isfinite(sca): continue
        rank=int((sc>sca).sum())+(int((sc==sca).sum())+1)/2.0; rr[g]=1.0/rank
    return rr

def block_alone(X,rows,gg,yv):
    yv2,pred,_=cv_pred(X,rows); ar=auroc(yv2,pred)
    cred,chance=p1_cred(yv2,pred,gg); gl=list(cred); credv=np.array([cred[g] for g in gl]); chc=float(np.mean([chance[g] for g in gl]))
    mrrv=np.array(list(mrr_gene(yv2,pred,gg).values())); gk=list(gg)
    arb=[]
    for _ in range(400):
        samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gg[g] for g in samp]); arb.append(auroc(yv2[idx],pred[idx]))
    p1b=[credv[rng.integers(0,len(credv),len(credv))].mean() for _ in range(400)]
    mrb=[mrrv[rng.integers(0,len(mrrv),len(mrrv))].mean() for _ in range(400)]
    return dict(ar=ar,alo=np.percentile(arb,2.5),ahi=np.percentile(arb,97.5),p1=credv.mean(),
                plo=np.percentile(p1b,2.5),phi=np.percentile(p1b,97.5),ch=chc,
                mrr=mrrv.mean(),mlo=np.percentile(mrb,2.5),mhi=np.percentile(mrb,97.5),cred=cred)

def increment(splX,rows,gg,yv,predC,crC):
    Xcs=np.column_stack([COMP,splX]); yv2,predB,_=cv_pred(Xcs,rows)
    d_au=auroc(yv2,predB)-auroc(yv2,predC); gk=list(gg); daub=[]
    for _ in range(400):
        samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gg[g] for g in samp])
        daub.append(auroc(yv2[idx],predB[idx])-auroc(yv2[idx],predC[idx]))
    crB,_=p1_cred(yv2,predB,gg); gpe=[g for g in crC if g in crB]; dcred=np.array([crB[g]-crC[g] for g in gpe])
    d_p1=dcred.mean(); dp1b=[dcred[rng.integers(0,len(dcred),len(dcred))].mean() for _ in range(400)]
    cred,chance=p1_cred(yv2,predB,gg); gl=list(cred); credv=np.array([cred[g] for g in gl])
    return dict(cs_au=auroc(yv2,predB),cs_p1=credv.mean(),d_au=d_au,daulo=np.percentile(daub,2.5),dauhi=np.percentile(daub,97.5),
                fau=float(np.mean(np.array(daub)>0)),d_p1=d_p1,dp1lo=np.percentile(dp1b,2.5),dp1hi=np.percentile(dp1b,97.5),
                fp1=float(np.mean(np.array(dp1b)>0)))

def run(mask,tag):
    rows=np.where((isc==1)|mask)[0]; rows=rows[have[rows]]
    yv=isc[rows].astype(float); gg=ggroups(np.array([gene[i] for i in rows]))
    _,predC,_=cv_pred(COMP,rows); crC,_=p1_cred(yv,predC,gg); auC=auroc(yv,predC)
    BA={}; INC={}
    for snm,splX in SPLSETS.items():
        BA[snm]=block_alone(splX,rows,gg,yv); INC[snm]=increment(splX,rows,gg,yv,predC,crC)
    return rows,gg,yv,predC,crC,auC,BA,INC

C=isc==0; NONTERM=C&np.isin(comp,[1,2])
rowsA,ggA,yvA,predCA,crCA,auCA,BAa,INCa=run(C,"ALL_comp")
rowsN,ggN,yvN,predCN,crCN,auCN,BAn,INCn=run(NONTERM,"NONTERM")

best=max(SPLSETS,key=lambda s:INCa[s]["d_au"])
def splice_pruned(setname):
    parts=[F1[:,[66,68]]]
    if "A" in setname: parts.append(A)
    if "B" in setname: parts.append(Bm)
    if "C" in setname: parts.append(Cm)
    return np.column_stack(parts)
DPR_ba=block_alone(splice_pruned(best),rowsA,ggA,yvA)
DPR_inc=increment(splice_pruned(best),rowsA,ggA,yvA,predCA,crCA)

def ci(v,lo,hi): return f"{v:.4f}[{lo:.4f},{hi:.4f}]"
def show(tag,auC,BA,INC):
    print(f"\n== {tag}  (COMP-only AUROC {auC:.4f}) ==")
    print(f"  {'set':14s} {'block-alone AUROC':22s} {'block P@1(ch)':22s} {'MRR':16s} | {'COMP+set AUROC':16s} {'incr AUROC vs COMP':24s} f>0  {'incr P@1':18s}")
    for s in ["current","current+A","current+B","current+C","current+A+B+C"]:
        b=BA[s]; c=INC[s]
        print(f"  {s:14s} {ci(b['ar'],b['alo'],b['ahi']):22s} {ci(b['p1'],b['plo'],b['phi']):22s} {b['mrr']:.4f}          | "
              f"{c['cs_au']:.4f}           {ci(c['d_au'],c['daulo'],c['dauhi']):24s} {c['fau']:.2f} {ci(c['d_p1'],c['dp1lo'],c['dp1hi'])}")
show("PRIMARY ALL_comp",auCA,BAa,INCa)
show("ROBUSTNESS NONTERM",auCN,BAn,INCn)
print(f"\n  best-by-increment (ALL_comp) = {best}  incr AUROC {ci(INCa[best]['d_au'],INCa[best]['daulo'],INCa[best]['dauhi'])}")
print(f"  D-pruned {best} [drop spl_dpa,spl_max5]: block AUROC {ci(DPR_ba['ar'],DPR_ba['alo'],DPR_ba['ahi'])} "
      f"P@1 {ci(DPR_ba['p1'],DPR_ba['plo'],DPR_ba['phi'])} MRR {DPR_ba['mrr']:.4f} | COMP+set AUROC {DPR_inc['cs_au']:.4f} "
      f"incr AUROC {ci(DPR_inc['d_au'],DPR_inc['daulo'],DPR_inc['dauhi'])} f>0 {DPR_inc['fau']:.2f} incr P@1 {ci(DPR_inc['d_p1'],DPR_inc['dp1lo'],DPR_inc['dp1hi'])}")

with open("competitor_subclass_analysis.tsv","a") as fh:
    W=lambda *a: fh.write("\t".join(str(x) for x in a)+"\n")
    W("#","CORRECTION (supersedes incrAUROCvsCOMP above): increment = AUROC(COMP+set) - AUROC(COMP). baseline current=+0.026.")
    for tag,auC,BA,INC in [("ALL_comp",auCA,BAa,INCa),("NONTERM",auCN,BAn,INCn)]:
        W("scope",tag,f"COMP_only_AUROC={auC:.4f}")
        W("set","blk_AUROC","blk_AUROC_lo","blk_AUROC_hi","blk_P@1","blk_MRR","COMP+set_AUROC","incrAUROC","incr_lo","incr_hi","frac>0","incrP@1","incrP@1_lo","incrP@1_hi")
        for s in ["current","current+A","current+B","current+C","current+A+B+C"]:
            b=BA[s]; c=INC[s]
            W(s,f"{b['ar']:.4f}",f"{b['alo']:.4f}",f"{b['ahi']:.4f}",f"{b['p1']:.4f}",f"{b['mrr']:.4f}",f"{c['cs_au']:.4f}",
              f"{c['d_au']:.4f}",f"{c['daulo']:.4f}",f"{c['dauhi']:.4f}",f"{c['fau']:.2f}",f"{c['d_p1']:.4f}",f"{c['dp1lo']:.4f}",f"{c['dp1hi']:.4f}")
    W("best_by_increment",best,f"{INCa[best]['d_au']:.4f}")
    W("Dpruned",best,"drop=spl_dpa,spl_max5",f"blk_AUROC={DPR_ba['ar']:.4f}",f"blk_P@1={DPR_ba['p1']:.4f}",f"MRR={DPR_ba['mrr']:.4f}",
      f"COMP+set_AUROC={DPR_inc['cs_au']:.4f}",f"incrAUROC={DPR_inc['d_au']:.4f}",f"[{DPR_inc['daulo']:.4f},{DPR_inc['dauhi']:.4f}]",f"incrP@1={DPR_inc['d_p1']:.4f}")
print("\nappended corrected increment to competitor_subclass_analysis.tsv")
