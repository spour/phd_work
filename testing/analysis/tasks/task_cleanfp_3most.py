#!/usr/bin/env python3
# one extra row: clean_fp restricted to the genes where the canonical is the 3'-most peak
import csv, numpy as np
from collections import defaultdict
NFOLD=5; RIDGE=1.0; THR=8.46
rng=np.random.default_rng(42)

R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t")); NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
cen=np.array([int(r["center"]) for r in R]); strd=np.array([r["strand"] for r in R])
acc=np.array([float(r["acc"]) if r["acc"] not in("","nan") else np.nan for r in R])
comp=np.load("peak_compartment.npy"); F1=np.load("hypothesis_test_features.npy")
Zg=np.load("geom_cache.npz"); max_up=Zg["max_up"]; max_dn=Zg["max_dn"]
At=np.load("atlas_peak_cache.npz"); a_dist=At["dist"]
matched=a_dist<=25

tc=np.where(strd=="+",cen,-cen).astype(float)
gmap=defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
inG3=np.zeros(NP,bool)
for g,idxs in gmap.items():
    idx=np.array(idxs); c=idx[isc[idx]==1]
    if len(c)!=1 or len(idx)<2: continue
    if tc[int(c[0])]==tc[idx].max():
        for i in idx: inG3[i]=True
presUP=np.where(np.isfinite(max_up),(max_up>=THR).astype(float),np.nan)
presDN=np.where(np.isfinite(max_dn),(max_dn>=THR).astype(float),np.nan)

CLEAN_FP=(isc==0)&(~matched)&(comp==2)
MASK=CLEAN_FP&inG3
SUB="CLEAN_FP_canon3most"
ng=len({gene[i] for i in np.where(MASK)[0]})
print(f"{SUB}: {int(MASK.sum())} peaks  {ng} genes")

def build_pairs(cm):
    P={}
    for g,idxs in gmap.items():
        idx=np.array(idxs); c=idx[isc[idx]==1]; k=idx[(isc[idx]==0)&cm[idx]]
        if len(c)>=1 and len(k)>=1: P[g]=(c,k)
    return P
def wg(val,P,higher):
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
            w=((kv>cv).sum()+0.5*(kv==cv).sum()) if higher=="comp" else ((cv>kv).sum()+0.5*(cv==kv).sum())
            rr.append(w/(cv.shape[0]*kv.shape[1]))
        return rr
    obs=rate(); wr=float(np.mean(obs)); n=len(obs)
    pg=np.array(obs); bs=pg[rng.integers(0,len(pg),size=(1000,len(pg)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
    rs=np.random.default_rng(7); nul=np.array([np.mean(rate(True,rs)) for _ in range(500)])
    p=(np.sum(np.abs(nul-0.5)>=abs(wr-0.5))+1)/501
    return dict(wr=wr,exc=wr-0.5,lo=lo,hi=hi,p=p,ng=n)

WM={}
for mnm,val,hi in [("acc_acceptor",acc,"canon"),("geom_presUP(anchor,pA)",presUP,"comp"),("geom_presDN(pA,pA+d)_matched",presDN,"comp")]:
    r=wg(val,build_pairs(MASK),hi); WM[mnm]=r
    print(f"  {mnm:28s} wr {r['wr']:.4f} exc {r['exc']:+.4f} CI[{r['lo']:.4f},{r['hi']:.4f}] p {r['p']:.4f} ng {r['ng']}")

have=np.isfinite(F1[:,0:60]).all(1)
allg=sorted(set(gene)); permf=np.random.default_rng(7).permutation(len(allg))
fold_of={g:permf[i]%NFOLD for i,g in enumerate(allg)}
BLOCKS={"COMP":F1[:,0:60],"SPLICE_arch":F1[:,66:70],"COMP+SPLICE_arch":np.column_stack([F1[:,0:60],F1[:,66:70]])}
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
def p1_pooled(yv,pred,genes):
    by=defaultdict(list)
    for k in range(len(yv)): by[genes[k]].append(k)
    cred={}; chance={}
    for g,ix in by.items():
        ix=np.array(ix)
        if (yv[ix]==1).sum()!=1 or len(ix)<2: continue
        sc=pred[ix]; mx=np.nanmax(sc)
        if not np.isfinite(mx): continue
        am=sc==mx; cred[g]=float((yv[ix][am]==1).sum())/am.sum(); chance[g]=1.0/len(ix)
    return cred,chance

rows_all=np.where(((isc==1)&inG3)|MASK)[0]; rows_all=rows_all[have[rows_all]]
ncan=int((isc[rows_all]==1).sum()); ncmp=int((isc[rows_all]==0).sum())
FB={}
for bnm,Xf in BLOCKS.items():
    yv,pred,genes=cv_pred(Xf,rows_all); ar=auroc(yv,pred)
    cred,chance=p1_pooled(yv,pred,genes); gl=list(cred); credv=np.array([cred[g] for g in gl])
    chc=float(np.mean([chance[g] for g in gl])); p1=credv.mean()
    gpool=defaultdict(list)
    for k in range(len(yv)): gpool[genes[k]].append(k)
    gk=list(gpool); arb=[]; p1b=[]
    for _ in range(400):
        samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gpool[g] for g in samp]); arb.append(auroc(yv[idx],pred[idx]))
        p1b.append(credv[rng.integers(0,len(credv),len(credv))].mean())
    alo,ahi=np.percentile(arb,[2.5,97.5]); plo,phi=np.percentile(p1b,[2.5,97.5])
    FB[bnm]=dict(ar=ar,alo=alo,ahi=ahi,p1=p1,plo=plo,phi=phi,ch=chc)
    print(f"  {bnm:18s} nC{ncan} nK{ncmp} AUROC {ar:.4f}[{alo:.4f},{ahi:.4f}] P@1 {p1:.4f}[{plo:.4f},{phi:.4f}] chance {chc:.4f}")

# append to tsv
with open("competitor_subclass_analysis.tsv","a") as fh:
    W=lambda *a: fh.write("\t".join(str(x) for x in a)+"\n")
    W("#","ADDED ROW: CLEAN_FP restricted to genes where canonical IS 3'-most (not-3'-most genes fully dropped)")
    W("subclass_def",SUB,int(MASK.sum()),ng)
    for mnm,hi in [("acc_acceptor","canon_higher"),("geom_presUP(anchor,pA)","comp_higher"),("geom_presDN(pA,pA+d)_matched","comp_higher")]:
        r=WM[mnm]; W("winrate",mnm,SUB,f"{r['wr']:.4f}",f"{r['exc']:+.4f}",f"{r['lo']:.4f}",f"{r['hi']:.4f}",f"{r['p']:.4f}",r['ng'],hi)
    for bnm in BLOCKS:
        r=FB[bnm]; W("fitblock",bnm,SUB,ncan,ncmp,f"{r['ar']:.4f}",f"{r['alo']:.4f}",f"{r['ahi']:.4f}",f"{r['p1']:.4f}",f"{r['plo']:.4f}",f"{r['phi']:.4f}",f"{r['ch']:.4f}")
print("appended to competitor_subclass_analysis.tsv")
