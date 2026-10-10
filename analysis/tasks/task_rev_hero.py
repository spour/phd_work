#!/usr/bin/env python3
# reversal: regenerate panels g / i / j at u1 cutoff 5.25 (was 4.0)
import sys, csv, math
import numpy as np
from collections import defaultdict
def log(*a): print(*a,file=sys.stderr,flush=True)
Z95=1.959963984540054
ACC_EDGES=np.arange(4.5,15.0+1e-9,1.5)
U1_EDGES=np.linspace(0.0,90.0,10)
MIN_CELL=200

def fnum(s):
    try:
        v=float(s); return v if np.isfinite(v) else np.nan
    except (TypeError,ValueError): return np.nan
def two_sided_p(z): return math.erfc(abs(z)/math.sqrt(2.0))
def p_label(p):
    if not np.isfinite(p): return "n/a"
    if p<=0: return "p < 1e-300"
    if p<1e-3: return "p = %.0e"%p
    return "p = %.3f"%p
def logistic_irls(X,y,max_iter=60,tol=1e-10):
    beta=np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta=np.clip(X@beta,-30,30); mu=1/(1+np.exp(-eta)); w=np.maximum(mu*(1-mu),1e-10)
        z=eta+(y-mu)/w; XtW=X.T*w
        try: new=np.linalg.solve(XtW@X,XtW@z)
        except np.linalg.LinAlgError: new=np.linalg.lstsq(XtW@X,XtW@z,rcond=None)[0]
        if np.max(np.abs(new-beta))<tol: beta=new; break
        beta=new
    eta=np.clip(X@beta,-30,30); mu=1/(1+np.exp(-eta)); w=np.maximum(mu*(1-mu),1e-10)
    return beta,np.sqrt(np.diag(np.linalg.inv((X.T*w)@X)))
def zscore(v):
    ok=np.isfinite(v); mu,sd=v[ok].mean(),v[ok].std(); return (v-mu)/(sd if sd>0 else 1.0)
def wilcoxon_signed_rank_p(dd):
    dd=dd[dd!=0]; n=dd.size
    if n<10: return float("nan")
    a=np.abs(dd); o=np.argsort(a,kind="mergesort"); sa=a[o]
    ng=np.r_[True,sa[1:]!=sa[:-1]]; g=np.cumsum(ng); st=np.r_[np.nonzero(ng)[0],n]
    av=(st[:-1]+st[1:]+1)/2.0; r=np.empty(n); r[o]=av[g-1]
    W=r[dd>0].sum(); mu=n*(n+1)/4.0; tie=np.bincount(g)[1:].astype(float)
    var=(n*(n+1)*(2*n+1)-(tie**3-tie).sum()/2.0)/24.0
    return two_sided_p((W-mu)/math.sqrt(var)) if var>0 else float("nan")
def within_gene(score,y,genes,comp_hi):
    by=defaultdict(lambda:([],[]))
    for i in range(score.size):
        if np.isfinite(score[i]): by[genes[i]][0 if y[i]==1 else 1].append(score[i])
    rates,tw,tp=[],0.0,0
    for _g,(can,comp) in by.items():
        if not can or not comp: continue
        c=np.asarray(can)[:,None]; k=np.asarray(comp)[None,:]
        w=float((k>c).sum() if comp_hi else (c>k).sum()); w+=0.5*float((c==k).sum())
        n=c.size*k.size; rates.append(w/n); tw+=w; tp+=n
    r=np.asarray(rates); return r,(tw/tp if tp else float("nan")),tp

# load aligned peaks
R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t"))
genes=np.array([r["gene"] for r in R]); y=np.array([int(float(r["is_canonical"])) for r in R],np.int8)
acc_me=np.array([fnum(r["acc"]) for r in R])
S=list(csv.DictReader(open("sa_peak_scalars_final.tsv"),delimiter="\t"))
assert [r["gene"] for r in S]==list(genes),"SA row order mismatch"
acc_sa=np.array([fnum(r["sa_acc_max"]) for r in S])
u1=np.load("u1_window_cache.npz")["dens_5.25"].astype(float)
base=float((y==1).mean())
log(f"peaks {len(R)} genes {len(set(genes))} base {base:.4f}  u1@5.25 median {np.nanmedian(u1):.3g}")
FEATS=[("acc_MaxEnt",acc_me,False),("acc_SpliceAI",acc_sa,False),("u1_pasprox",u1,True)]

rows=[]; summ={}
for name,v,comp_hi in FEATS:
    vz=zscore(v); by=defaultdict(lambda:([],[],[],[]))
    for i in range(v.size):
        if not np.isfinite(v[i]): continue
        g=by[genes[i]]
        if y[i]==1: g[0].append(v[i]); g[2].append(vz[i])
        else: g[1].append(v[i]); g[3].append(vz[i])
    for gname,(cv,kv,cz,kz) in by.items():
        if not cv or not kv: continue
        rows.append(dict(gene=gname,feature=name,canonical_val=float(np.mean(cv)),
            competitor_val=float(np.mean(kv)),canonical_z=float(np.mean(cz)),
            competitor_z=float(np.mean(kz)),n_competitors=len(kv),
            expected_direction=("competitor" if comp_hi else "canonical"),
            winrate_expected="",pair_wtd="",wilcoxon_p="",p_label="",n_genes="",n_pairs=""))
    r,pw,npair=within_gene(v,y,genes,comp_hi); p=wilcoxon_signed_rank_p(r-0.5)
    summ[name]=(float(r.mean()),pw,p,int(r.size),npair)
    rows.append(dict(gene="__SUMMARY__",feature=name,canonical_val="",competitor_val="",
        canonical_z="",competitor_z="",n_competitors="",
        expected_direction=("competitor" if comp_hi else "canonical"),
        winrate_expected=float(r.mean()),pair_wtd=float(pw),wilcoxon_p=float(p),
        p_label=p_label(p),n_genes=int(r.size),n_pairs=int(npair)))
K1=["gene","feature","canonical_val","competitor_val","canonical_z","competitor_z",
    "n_competitors","expected_direction","winrate_expected","pair_wtd","wilcoxon_p",
    "p_label","n_genes","n_pairs"]
with open("hero_within_gene_525.tsv","w") as fh:
    fh.write("\t".join(K1)+"\n")
    for r in rows:
        fh.write("\t".join(("%.6g"%r[c]) if isinstance(r[c],float) else str(r[c]) for c in K1)+"\n")
log(f"hero win-rates: acc_MaxEnt {summ['acc_MaxEnt'][0]:.4f} acc_SpliceAI {summ['acc_SpliceAI'][0]:.4f} u1 {summ['u1_pasprox'][0]:.4f}")

ok=np.isfinite(acc_me)&np.isfinite(u1)
ai=np.digitize(acc_me,ACC_EDGES)-1; ui=np.digitize(u1,U1_EDGES)-1
inb=ok&(ai>=0)&(ai<len(ACC_EDGES)-1)&(ui>=0)&(ui<len(U1_EDGES)-1)
cells=[]; kept=0
for bi in range(len(ACC_EDGES)-1):
    for bj in range(len(U1_EDGES)-1):
        m=inb&(ai==bi)&(ui==bj); n=int(m.sum())
        if n<MIN_CELL: continue
        pc=float(y[m].mean()); odds=(pc/(1-pc)) if 0<pc<1 else np.nan; ob=base/(1-base)
        l2=math.log2(odds/ob) if np.isfinite(odds) and odds>0 else np.nan
        xc=float((U1_EDGES[bj]+U1_EDGES[bj+1])/2)
        cells.append(dict(acceptor=float((ACC_EDGES[bi]+ACC_EDGES[bi+1])/2),interv=xc,u1=xc,
                          n=n,p_canonical=pc,log2_enrich=l2,base_rate=base)); kept+=n
K2=["acceptor","interv","u1","n","p_canonical","log2_enrich","base_rate"]
with open("acc_x_u1_525.tsv","w") as fh:
    fh.write("\t".join(K2)+"\n")
    for c in cells:
        fh.write("\t".join(("%.6g"%c[k]) if isinstance(c[k],float) else str(c[k]) for k in K2)+"\n")
lo=min((c["log2_enrich"] for c in cells if np.isfinite(c["log2_enrich"])),default=np.nan)
hi=max((c["log2_enrich"] for c in cells if np.isfinite(c["log2_enrich"])),default=np.nan)
log(f"acc_x_u1: {len(cells)} cells>= {MIN_CELL}, {kept} peaks, log2 OR {lo:.3f}..{hi:.3f}")

frows=[]
m=np.isfinite(u1); X=np.column_stack([np.ones(int(m.sum())),zscore(u1)[m]])
beta,se=logistic_irls(X,y[m].astype(float)); b,s=beta[1],se[1]; p=two_sided_p(b/s)
frows.append(dict(block="effect",effect_label="PAS-proximal U1",scorer="U1",
    estimate=math.exp(b),ci_lo=math.exp(b-Z95*s),ci_hi=math.exp(b+Z95*s),p=p,
    p_label=p_label(p),null_value=1.0,n=int(m.sum()),source="computed"))
ga,pw,pp,ng,_np=summ["u1_pasprox"]
r,_pw,_n=within_gene(u1,y,genes,True); hse=float(r.std()/math.sqrt(r.size)) if r.size>1 else float("nan")
frows.append(dict(block="ranking",effect_label="within-gene win-rate, U1 PAS-proximal",scorer="U1",
    estimate=ga,ci_lo=ga-Z95*hse,ci_hi=ga+Z95*hse,p=pp,p_label=p_label(pp),null_value=0.5,n=ng,source="computed"))
mi=np.isfinite(acc_me)&np.isfinite(u1); za,zu=zscore(acc_me),zscore(u1)
Xi=np.column_stack([np.ones(int(mi.sum())),za[mi],zu[mi],za[mi]*zu[mi]])
bi_,si_=logistic_irls(Xi,y[mi].astype(float)); b,s=bi_[3],si_[3]; p=two_sided_p(b/s)
frows.append(dict(block="interaction",effect_label="acceptor x U1",scorer="U1",
    estimate=math.exp(b),ci_lo=math.exp(b-Z95*s),ci_hi=math.exp(b+Z95*s),p=p,
    p_label=p_label(p),null_value=1.0,n=int(mi.sum()),source="computed"))
qt=np.nanquantile(acc_me,[1/3.0,2/3.0]); tert=np.digitize(acc_me,qt); tert_out=[]
for k,nm in enumerate(("weak","medium","strong")):
    v=np.where(tert==k,u1,np.nan); rr,ppw,_n=within_gene(v,y,genes,True); pt=wilcoxon_signed_rank_p(rr-0.5)
    se_=float(rr.std()/math.sqrt(rr.size)) if rr.size>1 else float("nan")
    tert_out.append((nm,float(rr.mean()),ppw,int(rr.size)))
    frows.append(dict(block="tertile",effect_label="U1 win-rate, %s acceptor"%nm,scorer="U1",
        estimate=float(rr.mean()),ci_lo=float(rr.mean())-Z95*se_,ci_hi=float(rr.mean())+Z95*se_,
        p=pt,p_label=p_label(pt),null_value=0.5,n=int(rr.size),source="computed"))
K3=["block","effect_label","scorer","estimate","ci_lo","ci_hi","p","p_label","null_value","n","source"]
with open("forest_u1_525.tsv","w") as fh:
    fh.write("\t".join(K3)+"\n")
    for r_ in frows:
        fh.write("\t".join(("%.6g"%r_[c]) if isinstance(r_[c],float) else str(r_[c]) for c in K3)+"\n")
f=frows[0]
print(f"[panelJ] U1 OR/SD {f['estimate']:.4f} [{f['ci_lo']:.4f},{f['ci_hi']:.4f}] {f['p_label']}")
print(f"[panelJ] within-gene win-rate {ga:.4f}  interaction OR {frows[2]['estimate']:.4f} {frows[2]['p_label']}")
print(f"[panelJ] U1 win-rate by acceptor tertile: "+" ".join(f"{nm} {v:.3f}" for nm,v,_,_ in tert_out))
print("[out] hero_within_gene_525.tsv acc_x_u1_525.tsv forest_u1_525.tsv")
