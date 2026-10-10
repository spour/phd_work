#!/usr/bin/env python3
# does the null survive the choice of variability threshold?
import sys, math, subprocess
import numpy as np, pandas as pd
from collections import defaultdict
def two_sided_p(z): return math.erfc(abs(z)/math.sqrt(2.0))
def plab(p): return "p<1e-300" if not (p > 1e-300) else ("p=%.1e" % p if p < 1e-3 else "p=%.3f" % p)
def woolf(a,b,c,d):
    if min(a,b,c,d)==0: a,b,c,d=a+.5,b+.5,c+.5,d+.5
    lor=math.log(a*d/(b*c)); se=math.sqrt(1/a+1/b+1/c+1/d)
    return math.exp(lor),math.exp(lor-1.96*se),math.exp(lor+1.96*se),two_sided_p(lor/se)
def mh_or(a,b,c,d):
    a,b,c,d=(np.asarray(x,float) for x in (a,b,c,d)); n=a+b+c+d
    ok=(n>0)&((a+b)>0)&((c+d)>0)&((a+c)>0)&((b+d)>0); a,b,c,d,n=a[ok],b[ok],c[ok],d[ok],n[ok]
    R,S=a*d/n,b*c/n
    if a.size==0 or R.sum()==0 or S.sum()==0: return (float("nan"),)*4
    orr=R.sum()/S.sum(); P,Q=(a+d)/n,(b+c)/n
    se=math.sqrt((P*R).sum()/(2*R.sum()**2)+(P*S+Q*R).sum()/(2*R.sum()*S.sum())+(Q*S).sum()/(2*S.sum()**2))
    E=(a+b)*(a+c)/n; V=np.where(n>1,(a+b)*(c+d)*(a+c)*(b+d)/(n*n*np.maximum(n-1,1)),0.0)
    chi=(abs(a.sum()-E.sum())-0.5)**2/V.sum() if V.sum()>0 else float("nan")
    return orr,math.exp(math.log(orr)-1.96*se),math.exp(math.log(orr)+1.96*se),(two_sided_p(math.sqrt(chi)) if chi==chi else float("nan"))
def merge(iv):
    iv=sorted(iv); out=[]
    for a,b in iv:
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return np.array([o[0] for o in out]),np.array([o[1] for o in out])
def inside(M,chrom,strand,pos):
    out=np.zeros(pos.size,bool)
    for k in np.unique(np.char.add(np.char.add(chrom,"\t"),strand)):
        c,s=k.split("\t"); m=M.get((c,s))
        if m is None: continue
        sel=(chrom==c)&(strand==s); st,en=m; p=pos[sel]
        j=np.searchsorted(st,p,"right")-1; ok=(j>=0)&(en[np.clip(j,0,len(en)-1)]>p)
        o=out[sel]; o[ok]=True; out[sel]=o
    return out

Z=np.load("as_overlap_flags.npz",allow_pickle=True)
S={nm:{k.split("__")[1]:Z[k] for k in Z.files if k.startswith(nm+"__")} for nm in ("COMP","DEC")}
rows=[]
print(f"{'R>=':>4} {'type':10s} {'nEv':>7} {'comp%':>7} {'dec%':>7} {'OR':>6} {'95% CI':>16} {'p':>10} "
      f"{'MH OR':>6} {'MH 95% CI':>16} {'MH p':>10}")
for RMIN in (10,20,30,50):
    f=f"as_events_R{RMIN}.tsv"
    subprocess.run([sys.executable,"as_events_build.py",str(RMIN),f],check=True,stderr=subprocess.DEVNULL)
    E=pd.read_csv(f,sep="\t"); E=E[E.variable==1]
    flags={}
    for t in ("INT","EXFLANK","ALTA_W","ALTD_W"):
        sub=E[E.etype==t]; d=defaultdict(list)
        for c,a,b,st in zip(sub.chrom,sub.start0,sub.end0,sub.strand): d[(c,st)].append((a,b))
        M={k:merge(v) for k,v in d.items()}
        for nm in ("COMP","DEC"):
            flags[(nm,t)]=inside(M,S[nm]["chrom"],S[nm]["strand"],S[nm]["pos"])
    for nm in ("COMP","DEC"):
        flags[(nm,"ANYAS")]=np.zeros(S[nm]["pos"].size,bool)
        for t in ("INT","EXFLANK","ALTA_W","ALTD_W"): flags[(nm,"ANYAS")]|=flags[(nm,t)]
    for t in ("INT","EXFLANK","ANYAS"):
        nev=int((E.etype==t).sum()) if t!="ANYAS" else int(E.etype.isin(["INT","EXFLANK","ALTA_W","ALTD_W"]).sum())
        yc=flags[("COMP",t)][S["COMP"]["pure_intron"]]; yd=flags[("DEC",t)][S["DEC"]["pure_intron"]]
        gc=S["COMP"]["gene"][S["COMP"]["pure_intron"]]; gd=S["DEC"]["gene"][S["DEC"]["pure_intron"]]
        a,b=int(yc.sum()),int((~yc).sum()); c,d=int(yd.sum()),int((~yd).sum())
        o,l,h,p=woolf(a,b,c,d)
        g=np.unique(np.concatenate([gc,gd])); gi={x:i for i,x in enumerate(g)}
        ic=np.array([gi[x] for x in gc]); idd=np.array([gi[x] for x in gd])
        ac=np.bincount(ic,yc.astype(float),len(g)); nc=np.bincount(ic,None,len(g))
        ad=np.bincount(idd,yd.astype(float),len(g)); nd=np.bincount(idd,None,len(g))
        mo,ml,mh,mp=mh_or(ac,nc-ac,ad,nd-ad)
        print(f"{RMIN:4d} {t:10s} {nev:7,} {a/(a+b):7.3%} {c/(c+d):7.3%} {o:6.3f} "
              f"[{l:5.3f},{h:5.3f}] {plab(p):>10} {mo:6.3f} [{ml:5.3f},{mh:5.3f}] {plab(mp):>10}")
        rows.append(dict(range_min=RMIN,etype=t,n_intervals=nev,comp_rate=a/(a+b),dec_rate=c/(c+d),
                         or_pooled=o,ci_lo=l,ci_hi=h,p_pooled=p,or_mh=mo,mh_lo=ml,mh_hi=mh,p_mh=mp))
pd.DataFrame(rows).to_csv("as_threshold_sweep.tsv",sep="\t",index=False,float_format="%.6g")
print("\n[out] as_threshold_sweep.tsv")
