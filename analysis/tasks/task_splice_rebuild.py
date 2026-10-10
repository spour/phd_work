#!/usr/bin/env python3
# rebuild splice_arch and test whether better maxent-scan features raise the increment
import csv, numpy as np
from collections import defaultdict
NFOLD=5; RIDGE=1.0
rng=np.random.default_rng(42)

R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t")); NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
comp=np.load("peak_compartment.npy"); F1=np.load("hypothesis_test_features.npy")
At=np.load("atlas_peak_cache.npz"); matched=At["dist"]<=25
S=np.load("splice_feats_cache.npz")

CURRENT=F1[:,66:70]
COMP=F1[:,0:60]
spl_acc=F1[:,66]; spl_max5=F1[:,69]
Dmax_up=S["Dmax_up"]; Dmax_dn=S["Dmax_dn"]
strongdon=np.maximum(np.where(np.isfinite(Dmax_up),Dmax_up,-99.0),np.where(np.isfinite(Dmax_dn),Dmax_dn,-99.0))
den=np.where(np.abs(strongdon)<1.0,np.sign(strongdon)+(strongdon==0),strongdon)
B_acc_minus_max5=spl_acc-spl_max5
B_acc_ratio=spl_acc/den

# feature groups (new features only)
A_names=["A_ndon_up","A_ndon_dn","A_sumdon_up","A_dstrong_up_pA","A_dstrong_up_anchor"]
B_names=["B_acc_minus_max5","B_acc_ratio_strongdon","B_acc_rank2k","B_acc_margin2k"]
C_names=["C_dist_3ss_up","C_dist_3ss_dn","C_dist_5ss_up","C_dist_5ss_dn",
         "C_n3ss_up5k","C_n3ss_dn5k","C_n5ss_up5k","C_n5ss_dn5k","C_in_intron"]
FEAT={n:S[n] for n in A_names}
FEAT["B_acc_minus_max5"]=B_acc_minus_max5; FEAT["B_acc_ratio_strongdon"]=B_acc_ratio
for n in ["B_acc_rank2k","B_acc_margin2k"]+C_names: FEAT[n]=S[n]
def M(names): return np.column_stack([FEAT[n] for n in names])
A=M(A_names); Bm=M(B_names); Cm=M(C_names)

SETS={
 "current":CURRENT,
 "current+A":np.column_stack([CURRENT,A]),
 "current+B":np.column_stack([CURRENT,Bm]),
 "current+C":np.column_stack([CURRENT,Cm]),
 "current+A+B+C":np.column_stack([CURRENT,A,Bm,Cm]),
}
NEWFEATS={"current":[], "current+A":A_names, "current+B":B_names, "current+C":C_names,
          "current+A+B+C":A_names+B_names+C_names}

# cv machinery
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
    yv=isc[rows].astype(float); fold=np.array([fold_of[gene[i]] for i in rows]); pred=np.full(len(rows),np.nan)
    Xr=Xfull[rows]
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

def evaluate(mask,label,do_feat_wr):
    rows=np.where((isc==1)|mask)[0]; rows=rows[have[rows]]
    yv0=isc[rows].astype(float); genes0=np.array([gene[i] for i in rows]); gg=ggroups(genes0)
    _,predC,_=cv_pred(COMP,rows)
    res={}
    for snm,X in SETS.items():
        yv,pred,genes=cv_pred(X,rows)
        ar=auroc(yv,pred)
        cred,chance=p1_cred(yv,pred,gg); gl=list(cred)
        credv=np.array([cred[g] for g in gl]); chc=float(np.mean([chance[g] for g in gl]))
        mrrd=mrr_gene(yv,pred,gg); mrrv=np.array(list(mrrd.values()))
        gk=list(gg)
        arb=[];
        for _ in range(400):
            samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gg[g] for g in samp])
            arb.append(auroc(yv[idx],pred[idx]))
        p1b=[credv[rng.integers(0,len(credv),len(credv))].mean() for _ in range(400)]
        mrb=[mrrv[rng.integers(0,len(mrrv),len(mrrv))].mean() for _ in range(400)]
        d_au=auroc(yv,pred)-auroc(yv,predC)
        daub=[]
        for _ in range(400):
            samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gg[g] for g in samp])
            daub.append(auroc(yv[idx],pred[idx])-auroc(yv[idx],predC[idx]))
        crC,_=p1_cred(yv0,predC,gg); crX=cred
        gpe=[g for g in crC if g in crX]; dcred=np.array([crX[g]-crC[g] for g in gpe]); d_p1=dcred.mean() if len(dcred) else np.nan
        dp1b=[dcred[rng.integers(0,len(dcred),len(dcred))].mean() for _ in range(400)] if len(dcred) else [np.nan]
        res[snm]=dict(ar=ar,alo=np.percentile(arb,2.5),ahi=np.percentile(arb,97.5),
                      p1=credv.mean(),plo=np.percentile(p1b,2.5),phi=np.percentile(p1b,97.5),ch=chc,
                      mrr=mrrv.mean(),mlo=np.percentile(mrb,2.5),mhi=np.percentile(mrb,97.5),
                      d_au=d_au,daulo=np.percentile(daub,2.5),dauhi=np.percentile(daub,97.5),fau=float(np.mean(np.array(daub)>0)),
                      d_p1=d_p1,dp1lo=np.nanpercentile(dp1b,2.5),dp1hi=np.nanpercentile(dp1b,97.5))
    return rows,gg,res

# per-feature within-gene win-rate (canonical value > competitor value)
gmap=defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
def build_pairs(cm):
    P={}
    for g,idxs in gmap.items():
        idx=np.array(idxs); c=idx[isc[idx]==1]; k=idx[(isc[idx]==0)&cm[idx]]
        if len(c)>=1 and len(k)>=1: P[g]=(c,k)
    return P
def wg(val,P):
    GK=list(P)
    def rate(perm=False,rs=None):
        rr=[]
        for g in GK:
            ci,ki=P[g]; idx=np.concatenate([ci,ki]); v=val[idx]; mm=np.isfinite(v)
            if mm.sum()<2: continue
            v=v[mm]; nc=int(mm[:len(ci)].sum())
            lab=rs.permutation(np.r_[np.ones(nc),np.zeros(len(v)-nc)]) if perm else np.r_[np.ones(len(ci)),np.zeros(len(ki))][mm]
            cv=v[lab==1]; kv=v[lab==0]
            if not len(cv) or not len(kv): continue
            cv=cv[:,None]; kv=kv[None,:]
            w=((cv>kv).sum()+0.5*(cv==kv).sum())
            rr.append(w/(cv.shape[0]*kv.shape[1]))
        return rr
    obs=rate(); wr=float(np.mean(obs)); ng=len(obs)
    pg=np.array(obs); bs=pg[rng.integers(0,len(pg),size=(1000,len(pg)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
    rs=np.random.default_rng(7); nul=np.array([np.mean(rate(True,rs)) for _ in range(500)])
    p=(np.sum(np.abs(nul-0.5)>=abs(wr-0.5))+1)/501
    return dict(wr=wr,exc=wr-0.5,lo=lo,hi=hi,p=p,ng=ng)

C=isc==0
NONTERM=C&np.isin(comp,[1,2])
rowsA,ggA,RESA=evaluate(C,"ALL_comp",True)
rowsN,ggN,RESN=evaluate(NONTERM,"NONTERM",False)

best=max(SETS,key=lambda s:RESA[s]["d_au"])
CUR_keep=F1[:,[66,68]]
def dprune_matrix(setname):
    parts=[CUR_keep]
    if "A" in setname: parts.append(A)
    if "B" in setname: parts.append(Bm)
    if "C" in setname: parts.append(Cm)
    return np.column_stack(parts)
Xbest_d=dprune_matrix(best)
def eval_one(X,mask):
    rows=np.where((isc==1)|mask)[0]; rows=rows[have[rows]]; gg=ggroups(np.array([gene[i] for i in rows]))
    yv,pred,genes=cv_pred(X,rows); _,predC,_=cv_pred(COMP,rows)
    ar=auroc(yv,pred); cred,chance=p1_cred(yv,pred,gg); gl=list(cred)
    credv=np.array([cred[g] for g in gl]); chc=float(np.mean([chance[g] for g in gl]))
    mrrv=np.array(list(mrr_gene(yv,pred,gg).values()))
    gk=list(gg); arb=[]; daub=[]
    for _ in range(400):
        samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gg[g] for g in samp])
        arb.append(auroc(yv[idx],pred[idx])); daub.append(auroc(yv[idx],pred[idx])-auroc(yv[idx],predC[idx]))
    p1b=[credv[rng.integers(0,len(credv),len(credv))].mean() for _ in range(400)]
    crC,_=p1_cred(yv,predC,gg); gpe=[g for g in crC if g in cred]; dcred=np.array([cred[g]-crC[g] for g in gpe])
    return dict(ar=ar,alo=np.percentile(arb,2.5),ahi=np.percentile(arb,97.5),p1=credv.mean(),
                plo=np.percentile(p1b,2.5),phi=np.percentile(p1b,97.5),ch=chc,mrr=mrrv.mean(),
                d_au=auroc(yv,pred)-auroc(yv,predC),daulo=np.percentile(daub,2.5),dauhi=np.percentile(daub,97.5),
                fau=float(np.mean(np.array(daub)>0)),d_p1=dcred.mean(),n=X.shape[1])
DPR=eval_one(Xbest_d,C)

WR={}
for n in A_names+B_names+C_names:
    WR[("ALL_comp",n)]=wg(FEAT[n],build_pairs(C))
    WR[("NONTERM",n)]=wg(FEAT[n],build_pairs(NONTERM))

# report
def ci(v,lo,hi): return f"{v:.4f}[{lo:.4f},{hi:.4f}]"
def block_table(RES,title):
    print("\n== "+title+" ==")
    order=["current","current+A","current+B","current+C","current+A+B+C"]
    print(f"  {'set':16s} {'AUROC':22s} {'P@1(chance)':24s} {'MRR':22s} {'incr_AUROC_vsCOMP':24s} {'incr_P@1':10s}")
    for s in order:
        r=RES[s]
        p1s=ci(r['p1'],r['plo'],r['phi'])+"(ch%.3f)"%r['ch']
        print(f"  {s:16s} {ci(r['ar'],r['alo'],r['ahi']):22s} {p1s:26s} "
              f"{ci(r['mrr'],r['mlo'],r['mhi']):22s} {ci(r['d_au'],r['daulo'],r['dauhi']):24s} {r['d_p1']:+.4f}")
block_table(RESA,"PRIMARY: ALL_comp  (canonical vs all competitors)")
block_table(RESN,"ROBUSTNESS: NONTERM  (canonical vs non-terminal competitors)")
print(f"\n  best-by-increment = {best}  (incr AUROC {RESA[best]['d_au']:+.4f})")
print(f"  D-pruned {best} [drop spl_dpa,spl_max5]  ncols {DPR['n']}: AUROC {ci(DPR['ar'],DPR['alo'],DPR['ahi'])} "
      f"P@1 {ci(DPR['p1'],DPR['plo'],DPR['phi'])}({DPR['ch']:.3f}) MRR {DPR['mrr']:.4f} "
      f"incr_AUROC {ci(DPR['d_au'],DPR['daulo'],DPR['dauhi'])} frac>0 {DPR['fau']:.3f} incr_P@1 {DPR['d_p1']:+.4f}")

print("\n== PER-FEATURE within-gene win-rate (canonical>competitor; excess, perm p, CI) ==")
print(f"  {'feature':24s} | {'ALL_comp exc(p)':28s} | {'NONTERM exc(p)':28s}")
for n in A_names+B_names+C_names:
    a=WR[("ALL_comp",n)]; b=WR[("NONTERM",n)]
    print(f"  {n:24s} | {a['wr']:.4f} exc{a['exc']:+.4f} p{a['p']:.4f} | {b['wr']:.4f} exc{b['exc']:+.4f} p{b['p']:.4f}")

with open("competitor_subclass_analysis.tsv","a") as fh:
    W=lambda *a: fh.write("\t".join(str(x) for x in a)+"\n")
    W("#","SPLICE_arch REBUILD: do better MaxEnt-scan features raise the increment over COMP? (5-fold gene CV, ridge1.0, EXPLORATORY)")
    W("#","current=spl_acc,spl_dpa,spl_strong5,spl_max5. A=donor dosage; B=acc/donor balance; C=scan-derived local context.")
    for tag,RES in [("ALL_comp",RESA),("NONTERM",RESN)]:
        W("scope",tag)
        W("set","AUROC","AUROC_lo","AUROC_hi","P@1","P@1_lo","P@1_hi","chance","MRR","incrAUROCvsCOMP","incr_lo","incr_hi","frac_boot>0","incrP@1")
        for s in ["current","current+A","current+B","current+C","current+A+B+C"]:
            r=RES[s]
            W(s,f"{r['ar']:.4f}",f"{r['alo']:.4f}",f"{r['ahi']:.4f}",f"{r['p1']:.4f}",f"{r['plo']:.4f}",f"{r['phi']:.4f}",
              f"{r['ch']:.4f}",f"{r['mrr']:.4f}",f"{r['d_au']:.4f}",f"{r['daulo']:.4f}",f"{r['dauhi']:.4f}",f"{r['fau']:.3f}",f"{r['d_p1']:+.4f}")
    W("best_by_increment",best,f"{RESA[best]['d_au']:.4f}")
    W("Dpruned",best,"drop=spl_dpa,spl_max5",f"ncols={DPR['n']}",f"AUROC={DPR['ar']:.4f}",f"P@1={DPR['p1']:.4f}",
      f"chance={DPR['ch']:.4f}",f"MRR={DPR['mrr']:.4f}",f"incrAUROC={DPR['d_au']:.4f}",f"[{DPR['daulo']:.4f},{DPR['dauhi']:.4f}]",f"incrP@1={DPR['d_p1']:+.4f}")
    W("#","per-feature within-gene win-rate (canonical>competitor)")
    W("feature","scope","winrate","excess","ci_lo","ci_hi","perm_p","n_genes")
    for n in A_names+B_names+C_names:
        for tag in ["ALL_comp","NONTERM"]:
            r=WR[(tag,n)]; W(n,tag,f"{r['wr']:.4f}",f"{r['exc']:+.4f}",f"{r['lo']:.4f}",f"{r['hi']:.4f}",f"{r['p']:.4f}",r['ng'])
print("\nappended to competitor_subclass_analysis.tsv")
