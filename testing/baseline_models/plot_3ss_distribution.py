#!/usr/bin/env python3
# distribution of 3' splice-site (acceptor) strength for the splicing reranker
import argparse, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAND_LABEL=["is_canonical","is_true_end","label","y","canonical"]
CAND_3SS=["upstream_max_me3","term_3ss","spl_upstream_max_me3","spl_term_3ss",
          "upstream_max_me3__wg","term_3ss__wg"]

def auc_mw(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int); ok=np.isfinite(s); s=s[ok]; y=y[ok]
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank(method="average").to_numpy()
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--peak_table",required=True)
    ap.add_argument("--label_col",default=None)
    ap.add_argument("--cols",default=None,help="comma list of 3'ss columns to plot; default=auto")
    ap.add_argument("--split",default=None,help="optional: only this split (train/test/val) if a 'split' col exists")
    ap.add_argument("--out",default="fig_3ss_dist")
    a=ap.parse_args()
    df=pd.read_parquet(a.peak_table)
    print(f"[data] {len(df):,} peaks, {df.shape[1]} cols",file=sys.stderr)

    lab=a.label_col
    if lab is None:
        for c in CAND_LABEL:
            if c in df.columns: lab=c; break
    if lab is None or lab not in df.columns:
        sys.exit(f"[fatal] no label column found; pass --label_col (looked for {CAND_LABEL}, "
                 f"available: {[c for c in df.columns if df[c].dropna().isin([0,1,True,False]).all()][:10]})")
    y=df[lab].astype(int).to_numpy()
    if a.split and "split" in df.columns:
        m=df["split"]==a.split; df=df[m].reset_index(drop=True); y=y[m.to_numpy()]
        print(f"[split] {a.split}: {len(df):,} peaks",file=sys.stderr)

    cols=(a.cols.split(",") if a.cols else [c for c in CAND_3SS if c in df.columns])
    cols=[c for c in cols if c in df.columns]
    if not cols:
        sys.exit(f"[fatal] no 3'ss columns found; available numeric cols include: "
                 f"{[c for c in df.columns if '3ss' in c or 'me3' in c][:15]}")
    print(f"[plot] 3'ss columns: {cols}",file=sys.stderr)
    print(f"[label] {lab}: {int((y==1).sum())} true-end / {int((y==0).sum())} competitor",file=sys.stderr)

    n=len(cols)
    fig,axes=plt.subplots(1,n,figsize=(4.6*n,4.2),squeeze=False)
    axes=axes[0]
    ct="#5B7C6F"; cc="#C65B3C"
    for ax,c in zip(axes,cols):
        x=df[c].to_numpy(float)
        xt=x[(y==1)&np.isfinite(x)]; xc=x[(y==0)&np.isfinite(x)]
        lo,hi=np.nanpercentile(x[np.isfinite(x)],[1,99])
        bins=np.linspace(lo,hi,50)
        ax.hist(xc,bins=bins,density=True,color=cc,alpha=0.55,label=f"competitor (n={len(xc):,})")
        ax.hist(xt,bins=bins,density=True,color=ct,alpha=0.55,label=f"true end (n={len(xt):,})")
        ax.axvline(np.median(xc),color=cc,ls="--",lw=1.2); ax.axvline(np.median(xt),color=ct,ls="--",lw=1.2)
        au=auc_mw(x,y)
        ax.set_title(f"{c}\nAUC {au:.3f}  (med: end {np.median(xt):.2f} vs comp {np.median(xc):.2f})",fontsize=9)
        ax.set_xlabel("3'ss strength" + (" (within-gene centered)" if c.endswith("__wg") else " (MaxEntScan score3)"))
        ax.set_ylabel("density")
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        ax.legend(fontsize=7,frameon=False)
    fig.suptitle("3' splice-site strength: gene's true end vs internal competitors",fontsize=12,weight="bold",y=1.02)
    fig.tight_layout()
    fig.savefig(f"{a.out}.png",dpi=200,bbox_inches="tight"); fig.savefig(f"{a.out}.pdf",bbox_inches="tight")
    print(f"[out] {a.out}.png, {a.out}.pdf",file=sys.stderr)
    for c in cols:
        x=df[c].to_numpy(float)
        print(f"  {c:<24} AUC {auc_mw(x,y):.3f}  med_end {np.nanmedian(x[y==1]):+.3f}  "
              f"med_comp {np.nanmedian(x[y==0]):+.3f}",file=sys.stderr)

if __name__=="__main__": main()