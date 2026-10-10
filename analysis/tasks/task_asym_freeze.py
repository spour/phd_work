#!/usr/bin/env python3
# characterise the acceptor-count asymmetry (n3ss_up - n3ss_dn)
import csv, numpy as np
from collections import defaultdict
NFOLD=5; RIDGE=1.0
rng=np.random.default_rng(42)

R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t")); NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
comp=np.load("peak_compartment.npy").astype(int); F1=np.load("hypothesis_test_features.npy")
AR=np.load("asym_real_cache.npz"); AD=np.load("asym_decoy_cache.npz")
SF=np.load("splice_feats_cache.npz")
gmap=defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)

# within-gene win-rate on an arbitrary pair dict
def wg(val,P,higher="canon"):
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
    obs=rate(); wr=float(np.mean(obs)); ng=len(obs)
    pg=np.array(obs); bs=pg[rng.integers(0,len(pg),size=(1000,len(pg)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
    rs=np.random.default_rng(7); nul=np.array([np.mean(rate(True,rs)) for _ in range(500)])
    p=(np.sum(np.abs(nul-0.5)>=abs(wr-0.5))+1)/501
    return dict(wr=wr,exc=wr-0.5,lo=lo,hi=hi,p=p,ng=ng)

def pairs(cm_comp,cm_canon=None):
    P={}
    for g,idxs in gmap.items():
        idx=np.array(idxs)
        c=idx[(isc[idx]==1)&(cm_canon[idx] if cm_canon is not None else True)]
        k=idx[(isc[idx]==0)&cm_comp[idx]]
        if len(c)>=1 and len(k)>=1: P[g]=(c,k)
    return P

def pairs_samecomp():
    P={}
    for g,idxs in gmap.items():
        idx=np.array(idxs); c=idx[isc[idx]==1]
        if len(c)!=1: continue
        cc=comp[c[0]]; k=idx[(isc[idx]==0)&(comp[idx]==cc)]
        if len(k)>=1: P[g]=(c,k)
    return P

def auroc(yv,s):
    order=np.argsort(s,kind="mergesort"); sv=s[order]; n=len(s); rk=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sv[j+1]==sv[i]: j+=1
        rk[i:j+1]=(i+j)/2.0+1.0; i=j+1
    rr=np.empty(n); rr[order]=rk; npos=yv.sum(); nneg=n-npos
    return np.nan if npos==0 or nneg==0 else (rr[yv==1].sum()-npos*(npos+1)/2.0)/(npos*nneg)

# part 1: asymmetry characterisation
WINS=[1,2,5,10,20]
Pall=pairs(isc==0)
CANrows=np.where((isc==1)|(isc==0))[0]
ycan=isc[CANrows].astype(float)
def pooled_auroc(v):
    m=np.isfinite(v[CANrows]); return auroc(ycan[m],v[CANrows][m])

print("\n=== PART 1a: acceptor-count asymmetry n3ss_up - n3ss_dn (canonical>competitor) ===")
ASY3={}; ASY5={}
for w in WINS:
    a3=AR[f"n3ss_up_{w}"]-AR[f"n3ss_dn_{w}"]; ASY3[w]=a3
    a5=AR[f"n5ss_up_{w}"]-AR[f"n5ss_dn_{w}"]; ASY5[w]=a5
rows_asy=[]
for lab,STORE in [("3'ss",ASY3),("5'ss",ASY5)]:
    for w in WINS:
        r=wg(STORE[w],Pall,"canon"); au=pooled_auroc(STORE[w])
        rows_asy.append((lab,w,r["wr"],r["exc"],r["lo"],r["hi"],r["p"],au,r["ng"]))
        print(f"  {lab} {w:>2}kb  winrate {r['wr']:.4f}[{r['lo']:.4f},{r['hi']:.4f}] exc {r['exc']:+.4f} p{r['p']:.4f}  AUROC {au:.4f}  ({r['ng']}g)")

# part 1b: compartment-matching
print("\n=== PART 1b: does the 5kb 3'ss asymmetry survive compartment-matching? ===")
prim=ASY3[5]
Pterm=pairs(comp==0,comp==0)
Psame=pairs_samecomp()
match_rows=[]
for nm,P in [("ALL_comp",Pall),("TERM-TERM",Pterm),("same-compartment(pooled)",Psame)]:
    r=wg(prim,P,"canon")
    match_rows.append((nm,r["wr"],r["exc"],r["lo"],r["hi"],r["p"],r["ng"]))
    print(f"  {nm:<26} winrate {r['wr']:.4f}[{r['lo']:.4f},{r['hi']:.4f}] exc {r['exc']:+.4f} p{r['p']:.4f} ({r['ng']}g)")

# part 1c: matched-intronic-decoy control (b13.2)
print("\n=== PART 1c: matched intronic decoy control on the 5kb 3'ss asymmetry ===")
D=np.load("decoy_set.npz"); dcomp=D["comp"].astype(int)
dec_asy=AD["d_n3ss_up5k"]-AD["d_n3ss_dn5k"]
dec_by_gene=defaultdict(list)
for j in range(len(dcomp)): dec_by_gene[gene[dcomp[j]]].append(j)
def canon_of(g):
    idx=np.array(gmap[g]); c=idx[isc[idx]==1]; return int(c[0]) if len(c)==1 else None
def decoy_arm(mode):
    rates=[]
    for g,js in dec_by_gene.items():
        js=np.array(js); dv=dec_asy[js]; ok=np.isfinite(dv)
        if ok.sum()==0: continue
        dv=dv[ok]; jj=js[ok]
        if mode=="comp":
            rv=prim[dcomp[jj]]
        else:
            c=canon_of(g)
            if c is None: continue
            rv=np.full(len(dv),prim[c])
        m2=np.isfinite(rv)
        if m2.sum()==0: continue
        d=rv[m2]-dv[m2]
        rates.append(((d>0).sum()+0.5*(d==0).sum())/len(d))
    rates=np.array(rates); bs=rates[rng.integers(0,len(rates),size=(2000,len(rates)))].mean(1)
    return rates.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),len(rates)
dec_rows=[]
for nm,mode in [("competitor>decoy (decisive)","comp"),("canonical>decoy (placebo)","canon")]:
    m,lo,hi,ng=decoy_arm(mode); dec_rows.append((nm,m,m-0.5,lo,hi,ng))
    print(f"  {nm:<28} winrate {m:.4f}[{lo:.4f},{hi:.4f}] exc {m-0.5:+.4f} ({ng}g)")

print("\n=== PART 3: C_in_intron vs peak_compartment.npy (INTRON=2) ===")
cii=SF["C_in_intron"]; okc=np.isfinite(cii)
is_intr=(comp==2)
frac1=float((cii[okc]==1).mean())
agree=float((cii[okc]==is_intr[okc]).mean())
p_intr_given1=float(is_intr[okc&(cii==1)].mean())
p_1_given_intr=float((cii[okc&is_intr]==1).mean())
p_1_given_notintr=float((cii[okc&(~is_intr)]==1).mean())
print(f"  C_in_intron mean (frac==1)           {frac1:.4f}")
print(f"  agreement (C_in_intron == comp==INTRON) {agree:.4f}")
print(f"  P(comp==INTRON | C_in_intron==1)     {p_intr_given1:.4f}")
print(f"  P(C_in_intron==1 | comp==INTRON)     {p_1_given_intr:.4f}")
print(f"  P(C_in_intron==1 | comp!=INTRON)     {p_1_given_notintr:.4f}")

print("\n=== PART 2: FROZEN SPLICE block = spl_acc + spl_strong5 + group C (11 feats) ===")
Ccols=["C_dist_3ss_up","C_dist_3ss_dn","C_dist_5ss_up","C_dist_5ss_dn",
       "C_n3ss_up5k","C_n3ss_dn5k","C_n5ss_up5k","C_n5ss_dn5k","C_in_intron"]
Cmat=np.column_stack([SF[c] for c in Ccols])
FROZEN=np.column_stack([F1[:,66],F1[:,68],Cmat])
COMP=F1[:,0:60]
BLOCKS={"FROZEN":FROZEN,"COMP":COMP,"COMP+FROZEN":np.column_stack([COMP,FROZEN])}
have=np.isfinite(COMP).all(1)
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
    return yv,pred
def gene_groups(genes):
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
def mrr_by_gene(yv,pred,gg):
    rr={}
    for g,ix in gg.items():
        if (yv[ix]==1).sum()!=1 or len(ix)<2: continue
        sc=pred[ix]; sc_can=sc[yv[ix]==1][0]
        if not np.isfinite(sc_can): continue
        greater=int((sc>sc_can).sum()); tied=int((sc==sc_can).sum())
        rank=greater+(tied+1)/2.0; rr[g]=1.0/rank
    return rr
def fmtci(v,lo,hi): return f"{v:.4f}[{lo:.4f},{hi:.4f}]"

FREEZE_DEF={}
for snm,m in [("ALL_comp",isc==0),("NONTERM",(isc==0)&np.isin(comp,[1,2]))]:
    rows=np.where((isc==1)|m)[0]; rows=rows[have[rows]]
    genes0=np.array([gene[i] for i in rows]); gg=gene_groups(genes0)
    preds={}; blk={}
    for bnm,Xf in BLOCKS.items():
        yv,pred=cv_pred(Xf,rows); preds[bnm]=pred; ar=auroc(yv,pred)
        cred,chance=p1_cred(yv,pred,gg); gl=list(cred)
        credv=np.array([cred[g] for g in gl]); chc=float(np.mean([chance[g] for g in gl]))
        mrrd=mrr_by_gene(yv,pred,gg); mrrv=np.array([mrrd[g] for g in mrrd])
        gk=list(gg); arb=[]
        for _ in range(400):
            samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]
            idx=np.concatenate([gg[g] for g in samp]); arb.append(auroc(yv[idx],pred[idx]))
        p1b=[credv[rng.integers(0,len(credv),len(credv))].mean() for _ in range(400)]
        mrb=[mrrv[rng.integers(0,len(mrrv),len(mrrv))].mean() for _ in range(400)]
        blk[bnm]=dict(ar=ar,alo=np.percentile(arb,2.5),ahi=np.percentile(arb,97.5),
                      p1=credv.mean(),plo=np.percentile(p1b,2.5),phi=np.percentile(p1b,97.5),ch=chc,
                      mrr=mrrv.mean(),mlo=np.percentile(mrb,2.5),mhi=np.percentile(mrb,97.5),cred=cred)
    yv=isc[rows].astype(float); pc=preds["COMP"]; pb=preds["COMP+FROZEN"]
    d_au=auroc(yv,pb)-auroc(yv,pc); gk=list(gg); dau_b=[]
    for _ in range(400):
        samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gg[g] for g in samp])
        dau_b.append(auroc(yv[idx],pb[idx])-auroc(yv[idx],pc[idx]))
    crC=blk["COMP"]["cred"]; crB=blk["COMP+FROZEN"]["cred"]
    gpe=[g for g in crC if g in crB]; dcred=np.array([crB[g]-crC[g] for g in gpe]); d_p1=dcred.mean()
    dp1_b=[dcred[rng.integers(0,len(dcred),len(dcred))].mean() for _ in range(400)]
    FREEZE_DEF[snm]=dict(blk=blk,d_au=d_au,daulo=np.percentile(dau_b,2.5),dauhi=np.percentile(dau_b,97.5),
                         d_p1=d_p1,dp1lo=np.percentile(dp1_b,2.5),dp1hi=np.percentile(dp1_b,97.5),
                         comp_ar=auroc(yv,pc),frac_au=float(np.mean(np.array(dau_b)>0)))
    b=blk
    print(f"\n  [{snm}] COMP-only AUROC {auroc(yv,pc):.4f}")
    print(f"    FROZEN     AUROC {fmtci(b['FROZEN']['ar'],b['FROZEN']['alo'],b['FROZEN']['ahi'])} "
          f"P@1 {fmtci(b['FROZEN']['p1'],b['FROZEN']['plo'],b['FROZEN']['phi'])}(ch{b['FROZEN']['ch']:.3f}) "
          f"MRR {b['FROZEN']['mrr']:.4f}")
    print(f"    COMP+FROZEN AUROC {fmtci(b['COMP+FROZEN']['ar'],b['COMP+FROZEN']['alo'],b['COMP+FROZEN']['ahi'])}")
    print(f"    incr AUROC vs COMP {fmtci(FREEZE_DEF[snm]['d_au'],FREEZE_DEF[snm]['daulo'],FREEZE_DEF[snm]['dauhi'])} "
          f"(frac>0 {FREEZE_DEF[snm]['frac_au']:.2f})  incr P@1 {fmtci(d_p1,FREEZE_DEF[snm]['dp1lo'],FREEZE_DEF[snm]['dp1hi'])}")

with open("competitor_subclass_analysis.tsv","a") as fh:
    W=lambda *a: fh.write("\t".join(str(x) for x in a)+"\n")
    W("#","ASYMMETRY CHARACTERISATION + FROZEN SPLICE BLOCK + C_in_intron audit.")
    W("#","asymmetry = strong-3'ss count upstream minus downstream (MaxEnt scan, >=6.5), canonical>competitor.")
    W("## Part1a  scale sweep: 3'ss and 5'ss asymmetry, within-gene win-rate (ALL_comp) + pooled AUROC")
    W("site","window_kb","winrate","excess","ci_lo","ci_hi","perm_p","pooled_AUROC","n_genes")
    for lab,w,wr,exc,lo,hi,p,au,ng in rows_asy: W(lab,w,f"{wr:.4f}",f"{exc:+.4f}",f"{lo:.4f}",f"{hi:.4f}",f"{p:.4f}",f"{au:.4f}",ng)
    W("## Part1b  compartment-matching on the 5kb 3'ss asymmetry")
    W("pairset","winrate","excess","ci_lo","ci_hi","perm_p","n_genes")
    for nm,wr,exc,lo,hi,p,ng in match_rows: W(nm,f"{wr:.4f}",f"{exc:+.4f}",f"{lo:.4f}",f"{hi:.4f}",f"{p:.4f}",ng)
    W("## Part1c  matched intronic decoy control (5kb 3'ss asymmetry)")
    W("arm","winrate","excess","ci_lo","ci_hi","n_genes")
    for nm,m,exc,lo,hi,ng in dec_rows: W(nm,f"{m:.4f}",f"{exc:+.4f}",f"{lo:.4f}",f"{hi:.4f}",ng)
    W("## Part3  C_in_intron audit vs peak_compartment.npy")
    W("C_in_intron_mean",f"{frac1:.4f}"); W("agreement_with_INTRON",f"{agree:.4f}")
    W("P(INTRON|inintron=1)",f"{p_intr_given1:.4f}"); W("P(inintron=1|INTRON)",f"{p_1_given_intr:.4f}")
    W("P(inintron=1|notINTRON)",f"{p_1_given_notintr:.4f}")
    W("## Part2  FROZEN SPLICE block = spl_acc + spl_strong5 + group C (11 features)")
    W("subclass","block","AUROC","P@1(chance)","MRR","COMP+FROZEN_AUROC","incr_AUROC_vs_COMP","incr_P@1")
    for snm in ["ALL_comp","NONTERM"]:
        d=FREEZE_DEF[snm]; b=d["blk"]
        W(snm,"FROZEN",fmtci(b["FROZEN"]["ar"],b["FROZEN"]["alo"],b["FROZEN"]["ahi"]),
          fmtci(b["FROZEN"]["p1"],b["FROZEN"]["plo"],b["FROZEN"]["phi"])+f"(ch{b['FROZEN']['ch']:.3f})",
          f"{b['FROZEN']['mrr']:.4f}",fmtci(b["COMP+FROZEN"]["ar"],b["COMP+FROZEN"]["alo"],b["COMP+FROZEN"]["ahi"]),
          fmtci(d["d_au"],d["daulo"],d["dauhi"]),fmtci(d["d_p1"],d["dp1lo"],d["dp1hi"]))
        W(snm,"COMP",fmtci(b["COMP"]["ar"],b["COMP"]["alo"],b["COMP"]["ahi"]),
          fmtci(b["COMP"]["p1"],b["COMP"]["plo"],b["COMP"]["phi"])+f"(ch{b['COMP']['ch']:.3f})",
          f"{b['COMP']['mrr']:.4f}","-","-","-")
print("\nappended to competitor_subclass_analysis.tsv  DONE_ASYM_FREEZE")
