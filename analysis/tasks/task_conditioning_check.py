import csv, numpy as np
from collections import defaultdict
POS="fig3_new_positional_features.tsv"; NFOLD=5
R=list(csv.DictReader(open(POS),delimiter="\t")); NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
F1=np.load("hypothesis_test_features.npy")
have=np.isfinite(F1[:,0:60]).all(1); keep=np.where(have)[0]
gene_k=gene[keep]; y=isc[keep].astype(float); ISC=isc[keep]
X=np.column_stack([F1[keep,0:60], F1[keep,66:71]])
NK=len(keep); NCOMP=60
genes=sorted(set(gene_k)); rs=np.random.default_rng(7); perm=rs.permutation(len(genes))
fold_of={g:perm[i]%NFOLD for i,g in enumerate(genes)}; fold=np.array([fold_of[g] for g in gene_k])
gp=defaultdict(list)
for i in range(NK): gp[gene_k[i]].append(i)
GEN=[np.array(v) for v in gp.values()]; NG=len(GEN)
GENP=[idx for idx in GEN if len(idx)>=2 and (ISC[idx]==1).any()]; NGP=len(GENP)

FALLBACK=[0]
def fit_ridge(Xd,yv,lam):
    P=np.eye(Xd.shape[1])*lam; P[0,0]=0.0; beta=np.zeros(Xd.shape[1])
    for _ in range(100):
        eta=np.clip(Xd@beta,-30,30); mu=1/(1+np.exp(-eta)); w=np.maximum(mu*(1-mu),1e-9); z=eta+(yv-mu)/w; XtW=Xd.T*w
        A=XtW@Xd+P; b=XtW@z+P@beta
        try: new=np.linalg.solve(A,b)
        except np.linalg.LinAlgError:
            new=np.linalg.lstsq(A,b,rcond=None)[0]; FALLBACK[0]+=1
        if np.max(np.abs(new-beta))<1e-9: beta=new; break
        beta=new
    return beta
def prep(Xtr,Xte,standardize):
    mu=np.nanmean(Xtr,0); sd=np.nanstd(Xtr,0); sd[sd==0]=1.0
    Xtr=np.where(np.isfinite(Xtr),Xtr,mu); Xte=np.where(np.isfinite(Xte),Xte,mu)
    if standardize: Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    return np.column_stack([np.ones(len(Xtr)),Xtr]), np.column_stack([np.ones(len(Xte)),Xte])
def cv(standardize,lam):
    FALLBACK[0]=0; pred=np.full(NK,np.nan)
    for f in range(NFOLD):
        tr=fold!=f; te=fold==f
        Xtr,Xte=prep(X[tr].astype(float).copy(),X[te].astype(float).copy(),standardize)
        beta=fit_ridge(Xtr,y[tr],lam); pred[te]=Xte@beta
    return pred,FALLBACK[0]
def auroc(yv,s):
    order=np.argsort(s,kind="mergesort"); sv=s[order]; n=len(s); rk=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sv[j+1]==sv[i]: j+=1
        rk[i:j+1]=(i+j)/2.0+1.0; i=j+1
    rr=np.empty(n); rr[order]=rk; npos=yv.sum(); nneg=n-npos
    return (rr[yv==1].sum()-npos*(npos+1)/2.0)/(npos*nneg)
def auprc(yv,s):
    order=np.argsort(-s,kind="mergesort"); ys=yv[order]; tp=np.cumsum(ys); fp=np.cumsum(1-ys); P=ys.sum()
    prec=tp/(tp+fp); rec=tp/P; ap=0.0; prev=0.0
    for k in range(len(ys)):
        if ys[k]==1: ap+=prec[k]*(rec[k]-prev); prev=rec[k]
    return ap
def mrr(s):
    tot=0.0
    for idx in GEN:
        sc=s[idx]; yv=y[idx]; order=np.argsort(-sc,kind="mergesort"); ranks=np.empty(len(sc)); ranks[order]=np.arange(1,len(sc)+1); pos=ranks[yv==1]
        if len(pos): tot+=1.0/pos.min()
    return tot/NG
def p1(s):
    tot=0.0
    for idx in GENP:
        sc=s[idx]; mx=np.nanmax(sc); am=sc==mx; tot+=float((ISC[idx][am]==1).sum())/am.sum()
    return tot/NGP
def M(s): return (auroc(y,s),auprc(y,s),mrr(s),p1(s))

variants=[("a. std + lambda=1.0",True,1.0),("b. unstd + lambda=1.0",False,1.0),
          ("c. std + lambda=0",True,0.0),("d. unstd + lambda=0",False,0.0)]
res={}
print(f"{'variant':22s} {'AUROC':>8s} {'AUPRC':>8s} {'MRR':>8s} {'P@1':>8s}  lstsq_fallbacks")
for nm,stz,lam in variants:
    s,fb=cv(stz,lam); ar,ap,mr,pp=M(s); res[nm]=(ar,ap,mr,pp,s)
    print(f"{nm:22s} {ar:8.4f} {ap:8.4f} {mr:8.4f} {pp:8.4f}  {fb}")

cd_max=max(abs(res['c. std + lambda=0'][i]-res['d. unstd + lambda=0'][i]) for i in range(4))
pred_maxdiff=np.nanmax(np.abs(res['c. std + lambda=0'][4]-res['d. unstd + lambda=0'][4]))
print(f"\n(c) vs (d) max metric diff = {cd_max:.2e} ; max per-peak prediction diff = {pred_maxdiff:.2e}")
print("  identical" if cd_max<1e-3 else "  DIFFER >1e-3 -> conditioning problem in IRLS solve (see lstsq fallback counts above)")

ac_max=max(abs(res['a. std + lambda=1.0'][i]-res['c. std + lambda=0'][i]) for i in range(4))
print(f"(a) vs (c) max metric diff = {ac_max:.4f}")

def full_coef(standardize,lam):
    Xtr,_=prep(X.astype(float).copy(),X[:1].astype(float).copy(),standardize)
    beta=fit_ridge(Xtr,y,lam)[1:]
    comp=np.mean(np.abs(beta[:NCOMP])); spl=np.mean(np.abs(beta[NCOMP:])); return comp,spl
print("\ncoefficient redistribution (full-data fit, lambda=1.0), mean|coef|:")
for nm,stz in [("a. standardized",True),("b. unstandardized",False)]:
    c,s=full_coef(stz,1.0); print(f"  {nm:18s} COMP {c:.5g}  SPLICE {s:.5g}  ratio COMP/SPLICE {c/s:.4g}")
