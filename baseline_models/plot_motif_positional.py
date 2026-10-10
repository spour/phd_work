#!/usr/bin/env python3
# scan chosen motif pwms (from the cnn's extracted .meme) across the full window of every
import argparse, sys, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASES=["A","C","G","T"]; BIDX={b:i for i,b in enumerate(BASES)}

def parse_meme(path):
    pwms={}; txt=open(path).read()
    for b in re.split(r"\nMOTIF ",txt)[1:]:
        name=b.split()[0]; rows=[]
        for ln in b.splitlines():
            nums=re.findall(r"[0-9]*\.?[0-9]+(?:e-?\d+)?",ln)
            v=[float(x) for x in nums]
            if len(v)==4 and 0.9<sum(v)<1.1: rows.append(v)
        if rows: pwms[name]=np.array(rows)
    return pwms

def consensus(P): return "".join(BASES[i] for i in P.argmax(1))

def is_ga(P):
    c=consensus(P); return (c.count("G")+c.count("A"))/len(c)>=0.6
def is_t(P):
    c=consensus(P); return (c.count("T")+c.count("C"))/len(c)>=0.6 and c.count("T")>=len(c)*0.4

def log_odds(P, bg=0.25, pcount=0.01):
    Pn=(P+pcount); Pn=Pn/Pn.sum(1,keepdims=True)
    return np.log2(Pn/bg)

def scan_scores(seqidx, has, lom):
    N,L=seqidx.shape; w=lom.shape[0]
    sc=np.full((N,L),-np.inf,dtype=np.float32)
    for st in range(L-w+1):
        window=seqidx[:,st:st+w]
        valid=has[:,st:st+w].all(1)
        s=lom[np.arange(w),window].sum(1)
        s=np.where(valid,s,-np.inf)
        sc[:,st]=s
    return sc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",required=True); ap.add_argument("--meta",required=True)
    ap.add_argument("--meme",required=True)
    ap.add_argument("--motifs",default=None,help="comma list of MOTIF names from the meme")
    ap.add_argument("--names",default=None,help="comma display names matching --motifs")
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--score_pct",type=float,default=90,help="hit threshold = this percentile of the motif's own scores")
    ap.add_argument("--bin",type=int,default=25,help="histogram bin width (nt)")
    ap.add_argument("--out",default="motif_positional")
    a=ap.parse_args()

    X=np.load(a.npz)["X"].astype(np.int8); m=pd.read_parquet(a.meta).reset_index(drop=True)
    y=m["label"].to_numpy().astype(int); L=X.shape[1]; cl=a.upstream_nt
    seqidx=X.argmax(2); has=X.sum(2)>0
    print(f"[data] {len(X):,} sites ({int((y==1).sum())} dominant, {int((y==0).sum())} cryptic)",file=sys.stderr)

    pwms=parse_meme(a.meme)
    if a.motifs:
        chosen=a.motifs.split(","); names=(a.names.split(",") if a.names else chosen)
    else:
        ga=next((n for n in pwms if is_ga(pwms[n])),None)
        tt=next((n for n in pwms if is_t(pwms[n])),None)
        chosen=[x for x in [ga,tt] if x]; names=["ESE(GA-rich)","PPT(T-rich)"][:len(chosen)]
        print(f"[auto] picked {chosen}",file=sys.stderr)

    rel=np.arange(L)-cl
    fig,axes=plt.subplots(len(chosen),1,figsize=(11,3.2*len(chosen)),sharex=True)
    if len(chosen)==1: axes=[axes]
    col_dom="#5B7C6F"; col_cry="#C65B3C"
    summary=[]
    for ax,mt,nm in zip(axes,chosen,names):
        if mt not in pwms:
            print(f"[warn] {mt} not in meme; available e.g. {list(pwms)[:3]}",file=sys.stderr); continue
        P=pwms[mt]; lom=log_odds(P); cons=consensus(P); w=P.shape[0]
        sc=scan_scores(seqidx,has,lom)
        thr=np.percentile(sc[np.isfinite(sc)],a.score_pct)
        hits=sc>=thr
        hitpos_center=[]
        center_rel=rel + w//2
        for cls,yy,colr,lab in [(1,col_dom,col_dom,"dominant"),(0,col_cry,col_cry,"cryptic")]:
            mask=(y==cls)
            freq=hits[mask].mean(0)
            nb=L//a.bin
            bz=(center_rel[:nb*a.bin].reshape(nb,a.bin)).mean(1)
            fz=(freq[:nb*a.bin].reshape(nb,a.bin)).mean(1)
            ax.plot(bz,fz,color=colr,lw=1.8,label=f"{lab} (n={int(mask.sum())})")
            ax.fill_between(bz,fz,alpha=0.12,color=colr)
        ax.axvline(0,color="k",lw=1); ax.axvline(-21,ls=":",color="grey")
        ax.axvspan(13,30,color="grey",alpha=0.08)
        ax.text(21,ax.get_ylim()[1]*0.9,"DSE\n+13–30",fontsize=6,ha="center",color="grey")
        ax.set_title(f"{nm}: {cons}  (PWM hit frequency by position, >{a.score_pct:.0f}th pct)",fontsize=10,loc="left")
        ax.set_ylabel("hit frequency"); ax.legend(fontsize=8,frameon=False)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        up=(center_rel<-50); dn=(center_rel>50)
        for cls,lab in [(1,"dominant"),(0,"cryptic")]:
            f=hits[y==cls].mean(0)
            summary.append(dict(motif=nm,consensus=cons,clas=lab,
                up_freq=float(f[up].mean()),dn_freq=float(f[dn].mean()),
                dse_freq=float(f[(center_rel>=13)&(center_rel<=30)].mean())))
    axes[-1].set_xlabel("position of motif center relative to cleavage site (nt)")
    fig.tight_layout()
    fig.savefig(f"{a.out}.png",dpi=200,bbox_inches="tight"); fig.savefig(f"{a.out}.pdf",bbox_inches="tight")
    S=pd.DataFrame(summary); S.to_csv(f"{a.out}_summary.tsv",sep="\t",index=False)
    print(f"\n=== upstream vs downstream hit frequency ===",file=sys.stderr)
    print(S.to_string(index=False),file=sys.stderr)
    print(f"\n[out] {a.out}.png, {a.out}.pdf, {a.out}_summary.tsv",file=sys.stderr)

if __name__=="__main__": main()