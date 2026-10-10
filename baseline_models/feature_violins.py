#!/usr/bin/env python3
# distributions of the two top reranker features, canonical
import argparse, sys
import numpy as np, pandas as pd

FEATS = [
    ("spl_upstream_max_me3",  "strongest upstream\n3' splice site",       +1),
    ("spl_u1_suppression_up", "U1 / donor density\n(upstream)",           -1),
]
COL_CANON = "#C65B3C"
COL_COMP  = "#5B7C6F"

def load(train_feat, test_feat):
    parts=[]
    for f in (train_feat, test_feat):
        d=pd.read_parquet(f)
        parts.append(d)
    df=pd.concat(parts, ignore_index=True)
    sz=df.groupby("gene").size()
    df=df[df.gene.isin(set(sz[sz>1].index))].copy()
    for col,_,_ in FEATS:
        df[col]=pd.to_numeric(df[col], errors="coerce")
    return df

def mannwhitney_rbc(a, b):
    from scipy.stats import mannwhitneyu
    a=a[~np.isnan(a)]; b=b[~np.isnan(b)]
    if len(a)<2 or len(b)<2: return np.nan, np.nan, len(a), len(b)
    U,p=mannwhitneyu(a, b, alternative="two-sided")
    rbc=2.0*U/(len(a)*len(b)) - 1.0
    return p, rbc, len(a), len(b)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_feat", required=True)
    ap.add_argument("--test_feat",  required=True)
    ap.add_argument("--out", default="feature_violins")
    a=ap.parse_args()

    df=load(a.train_feat, a.test_feat)
    print(f"[data] {len(df):,} contested peaks over {df.gene.nunique():,} genes; "
          f"{int(df.is_canonical.sum()):,} canonical, {int((~df.is_canonical).sum()):,} competitor",
          file=sys.stderr)

    for col,_,_ in FEATS:
        df[col+"_ctr"]=df[col]-df.groupby("gene")[col].transform("mean")

    emit(df, FEATS, a.out, False,
         "Top reranker features: true ends vs internal competitors")
    emit(df, FEATS, a.out+"_centered", True,
         "Gene-baseline-centered: within-gene contrast")

def emit(df, feats, out, centered, title):
    canon=df[df.is_canonical]; comp=df[~df.is_canonical]
    rows=[]; long=[]
    for col,label,direction in feats:
        vcol=col+"_ctr" if centered else col
        ca=canon[vcol].to_numpy(); co=comp[vcol].to_numpy()
        p,rbc,na,nb=mannwhitney_rbc(ca, co)
        pair=[]
        for g,gg in df.groupby("gene", sort=False):
            cc=gg.loc[gg.is_canonical, vcol].to_numpy(); kk=gg.loc[~gg.is_canonical, vcol].to_numpy()
            if len(cc) and len(kk): pair.append(np.nanmean(cc)-np.nanmean(kk))
        pair=np.array([x for x in pair if np.isfinite(x)])
        rows.append(dict(feature=col, label=label.replace("\n"," "), direction=direction,
            canon_median=np.nanmedian(ca), comp_median=np.nanmedian(co),
            mannwhitney_p=p, rank_biserial=rbc,
            paired_diff_median=float(np.nanmedian(pair)) if len(pair) else np.nan,
            paired_frac_positive=float(np.nanmean(pair>0)) if len(pair) else np.nan,
            n_canon=na, n_comp=nb, n_genes_paired=len(pair)))
        for grp,sub in [("canonical",canon),("competitor",comp)]:
            v=sub[vcol].to_numpy(); v=v[~np.isnan(v)]
            long.append(pd.DataFrame(dict(feature=col, label=label.replace("\n"," "),
                                          group=grp, value=v)))
    st=pd.DataFrame(rows)
    st.to_csv(f"{out}_stats.tsv", sep="\t", index=False)
    pd.concat(long, ignore_index=True).to_csv(f"{out}_long.tsv", sep="\t", index=False)

    print(f"\n=== {'centered' if centered else 'raw'}: canonical vs competitor ===", file=sys.stderr)
    for _,r in st.iterrows():
        extra=(f"  within-gene diff med={r.paired_diff_median:+.3f} "
               f"({r.paired_frac_positive:.0%} genes canon>comp)") if np.isfinite(r.paired_diff_median) else ""
        print(f"  {r.feature:<24} canon med={r.canon_median:+.3f}  comp med={r.comp_median:+.3f}  "
              f"rbc={r.rank_biserial:+.3f}  p={r.mannwhitney_p:.1e}{extra}", file=sys.stderr)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11, "svg.fonttype":"none",
                         "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, len(feats), figsize=(4.0*len(feats), 4.6))
    if len(feats)==1: axes=[axes]
    for ax,(col,label,direction) in zip(axes, feats):
        vcol=col+"_ctr" if centered else col
        ca=canon[vcol].to_numpy(); ca=ca[~np.isnan(ca)]
        co=comp[vcol].to_numpy(); co=co[~np.isnan(co)]
        lo=np.nanpercentile(np.concatenate([ca,co]), 0.5)
        hi=np.nanpercentile(np.concatenate([ca,co]), 99.5)
        if hi<=lo: hi=lo+1e-6
        cac=np.clip(ca,lo,hi); coc=np.clip(co,lo,hi)
        parts=ax.violinplot([cac, coc], positions=[1,2], widths=0.8,
                            showmeans=False, showextrema=False, showmedians=False)
        for pc,c in zip(parts["bodies"], [COL_CANON, COL_COMP]):
            pc.set_facecolor(c); pc.set_edgecolor("none"); pc.set_alpha(0.85)
        for pos,arr in [(1,ca),(2,co)]:
            q1,med,q3=np.nanpercentile(arr,[25,50,75])
            ax.plot([pos,pos],[q1,q3], color="black", lw=4, solid_capstyle="round", zorder=3)
            ax.plot(pos, med, "o", color="white", mec="black", ms=6, zorder=4)
        if centered: ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
        rbc=float(st.loc[st.feature==col,"rank_biserial"].iloc[0])
        ax.set_title(label.replace("\n"," "), fontsize=11)
        ax.set_xticks([1,2]); ax.set_xticklabels(["true end\n(canonical)","competitor\n(internal)"])
        ax.set_ylim(lo-(hi-lo)*0.05, hi+(hi-lo)*0.12)
        ax.set_ylabel("peak \u2212 gene mean" if centered else "feature value")
        mk="true end" if direction>0 else "suppressed"; sign="+" if direction>0 else "\u2212"
        ax.text(0.5, 0.97, f"rank-biserial = {rbc:+.2f}  ({sign} marks {mk})",
                transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555555")
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    print(f"[out] {out}.pdf, {out}.png, {out}_stats.tsv, {out}_long.tsv", file=sys.stderr)

if __name__=="__main__": main()