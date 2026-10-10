#!/usr/bin/env python3
# plot at% and gc% in 10-nt bins across the window, dominant vs cryptic
import argparse, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A,C,G,T=0,1,2,3

def binned_frac(X, channels, bin_):
    N,L,_=X.shape
    nb=L//bin_
    sel=X[:,:nb*bin_,channels].sum(2)
    has=X[:,:nb*bin_,:].sum(2)>0
    sel=sel.reshape(N,nb,bin_); has=has.reshape(N,nb,bin_)
    num=(sel*has).sum((0,2)); den=has.sum((0,2))
    return np.where(den>0, num/np.maximum(den,1), np.nan)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",required=True); ap.add_argument("--meta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--bin",type=int,default=10)
    ap.add_argument("--out",default="at_gc_profile")
    a=ap.parse_args()
    X=np.load(a.npz)["X"].astype(np.float32); m=pd.read_parquet(a.meta).reset_index(drop=True)
    y=m["label"].to_numpy().astype(int); L=X.shape[1]; cl=a.upstream_nt
    nb=L//a.bin
    rel=(np.arange(nb)*a.bin + a.bin/2.0) - cl
    print(f"[data] {len(X):,} sites ({int((y==1).sum())} dominant, {int((y==0).sum())} cryptic), "
          f"L={L}, {nb} bins of {a.bin}nt",file=sys.stderr)

    Xd=X[y==1]; Xc=X[y==0]
    at_d=binned_frac(Xd,[A,T],a.bin); at_c=binned_frac(Xc,[A,T],a.bin)
    gc_d=binned_frac(Xd,[G,C],a.bin); gc_c=binned_frac(Xc,[G,C],a.bin)

    fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True,
                        gridspec_kw={"height_ratios":[1,1,0.8],"hspace":0.18})
    cd="#5B7C6F"; cc="#C65B3C"
    ax[0].plot(rel,at_d,color=cd,lw=1.4,label="dominant/used")
    ax[0].plot(rel,at_c,color=cc,lw=1.4,label="cryptic")
    ax[0].set_ylabel("AT fraction"); ax[0].legend(fontsize=9,frameon=False)
    ax[0].set_title(f"AT and GC richness per {a.bin} nt: dominant vs cryptic",fontsize=12,weight="bold",loc="left")
    ax[1].plot(rel,gc_d,color=cd,lw=1.4); ax[1].plot(rel,gc_c,color=cc,lw=1.4)
    ax[1].set_ylabel("GC fraction")
    ax[2].axhline(0,color="grey",lw=0.8)
    ax[2].plot(rel,at_d-at_c,color="#9467bd",lw=1.2,label="AT diff (dom-cry)")
    ax[2].plot(rel,gc_d-gc_c,color="#1f77b4",lw=1.2,label="GC diff (dom-cry)")
    ax[2].set_ylabel("dominant - cryptic"); ax[2].legend(fontsize=8,frameon=False,ncol=2)
    ax[2].set_xlabel("position relative to cleavage site (nt)")
    for x in ax:
        x.axvline(0,color="k",lw=1); x.axvline(-21,ls=":",color="grey",lw=1)
        x.axvspan(13,30,color="grey",alpha=0.07)
        for s in ["top","right"]: x.spines[s].set_visible(False)
    ax[0].text(0,ax[0].get_ylim()[1],"cleavage",fontsize=7,ha="center",va="bottom",color="k")
    ax[0].text(-21,ax[0].get_ylim()[0],"PAS",fontsize=7,ha="center",va="top",color="grey")
    fig.savefig(f"{a.out}.png",dpi=200,bbox_inches="tight"); fig.savefig(f"{a.out}.pdf",bbox_inches="tight")
    print(f"[out] {a.out}.png, {a.out}.pdf",file=sys.stderr)

    prof=pd.DataFrame(dict(rel=rel,at_dominant=at_d,at_cryptic=at_c,
                           gc_dominant=gc_d,gc_cryptic=gc_c,
                           at_diff=at_d-at_c,gc_diff=gc_d-gc_c))
    prof.to_csv(f"{a.out}.tsv",sep="\t",index=False)
    k=int(np.nanargmax(np.abs(prof["gc_diff"].to_numpy())))
    print(f"[note] largest GC diff at {rel[k]:+.0f} nt: dom-cry = {prof['gc_diff'][k]:+.3f}",file=sys.stderr)
    k=int(np.nanargmax(np.abs(prof["at_diff"].to_numpy())))
    print(f"[note] largest AT diff at {rel[k]:+.0f} nt: dom-cry = {prof['at_diff'][k]:+.3f}",file=sys.stderr)

if __name__=="__main__": main()