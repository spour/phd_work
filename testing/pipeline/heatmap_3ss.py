#!/usr/bin/env python3
# 2d joint distribution of (distance to the strongest upstream 3'ss) x
import argparse, sys
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="featurized shortlist parquet")
    ap.add_argument("--x", default="spl_dist_to_max_3ss")
    ap.add_argument("--y", default="spl_upstream_max_me3")
    ap.add_argument("--xbins", type=int, default=40)
    ap.add_argument("--ybins", type=int, default=40)
    ap.add_argument("--xclip", type=float, nargs=2, default=None,
                    help="clip x to [lo,hi]; default = 1st/99th pct")
    ap.add_argument("--yclip", type=float, nargs=2, default=None)
    ap.add_argument("--contested_only", action="store_true", default=True)
    ap.add_argument("--center_by_gene", action="store_true",
                    help="subtract each gene's mean over its shortlist peaks from both axes. "
                         "Removes the between-gene offset (gene size / intron architecture) and "
                         "isolates the WITHIN-gene contrast the reranker actually uses, since "
                         "argmax is per gene. Strongly recommended: pooled absolute distance is "
                         "dominated by which gene a peak sits in, not by canonical vs competitor.")
    ap.add_argument("--mode", default="enrichment", choices=["enrichment","density"],
                    help="enrichment (default): per bin, log2 odds that a peak is canonical "
                         "vs the global base rate. Uses ALL peaks per bin, so it is well "
                         "powered. density: per-class joint density (underpowered: canonical "
                         "n is ~10x smaller, so its map is Poisson noise).")
    ap.add_argument("--min_n", type=int, default=40,
                    help="bins with fewer than this many total peaks are masked (NA)")
    ap.add_argument("--quantile_bins", action="store_true", default=False,
                    help="quantile edges per axis. NOTE: independent per-axis quantiles do "
                         "NOT make the JOINT grid evenly populated; linear bins + --min_n "
                         "masking is usually better.")
    ap.add_argument("--out", default="fig3_3ss")
    a = ap.parse_args()

    d = pd.read_parquet(a.feat)
    if a.contested_only and "n_internal" in d.columns:
        d = d[d.n_internal > 0]
    for c in (a.x, a.y):
        if c not in d.columns: sys.exit(f"[err] missing column {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[a.x, a.y]).copy()
    d["cls"] = np.where(d.is_canonical, "Canonical", "Competitor")
    print(f"[data] {len(d):,} peaks "
          f"({int((d.cls=='Canonical').sum()):,} canonical, "
          f"{int((d.cls=='Competitor').sum()):,} competitor)", file=sys.stderr)

    xname, yname = a.x, a.y
    if a.center_by_gene:
        if "gene" not in d.columns: sys.exit("[err] --center_by_gene needs a `gene` column")
        ok = d.groupby("gene")["is_canonical"].agg(["any","all"])
        good = ok[(ok["any"]) & (~ok["all"])].index
        n_before = d.gene.nunique()
        d = d[d.gene.isin(good)].copy()
        print(f"[center] {d.gene.nunique():,}/{n_before:,} genes retain both classes", file=sys.stderr)
        for c in (a.x, a.y):
            d[c + "_ctr"] = d[c] - d.groupby("gene")[c].transform("mean")
        xname, yname = a.x + "_ctr", a.y + "_ctr"
        for c, lab in ((xname,"x"), (yname,"y")):
            mu = d.groupby("cls")[c].mean()
            print(f"[center] {lab} mean by class: "
                  + "  ".join(f"{k}={v:+.3f}" for k, v in mu.items()), file=sys.stderr)
    a.x, a.y = xname, yname

    xlo, xhi = a.xclip if a.xclip else (np.percentile(d[a.x], 1), np.percentile(d[a.x], 99))
    ylo, yhi = a.yclip if a.yclip else (np.percentile(d[a.y], 1), np.percentile(d[a.y], 99))
    d[a.x] = d[a.x].clip(xlo, xhi); d[a.y] = d[a.y].clip(ylo, yhi)
    if a.quantile_bins:
        xe = np.unique(np.quantile(d[a.x], np.linspace(0, 1, a.xbins + 1)))
        ye = np.unique(np.quantile(d[a.y], np.linspace(0, 1, a.ybins + 1)))
    else:
        xe = np.linspace(xlo, xhi, a.xbins + 1)
        ye = np.linspace(ylo, yhi, a.ybins + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2

    rows = []; grids = {}
    for cls, sub in d.groupby("cls"):
        H, _, _ = np.histogram2d(sub[a.x].values, sub[a.y].values, bins=[xe, ye])
        P = H / max(H.sum(), 1)
        grids[cls] = P
        for i in range(a.xbins):
            for j in range(a.ybins):
                if H[i, j] == 0: continue
                rows.append(dict(cls=cls, x=float(xc[i]), y=float(yc[j]),
                                 n=int(H[i, j]), prop=float(P[i, j])))
    long = pd.DataFrame(rows)
    long.to_csv(f"{a.out}_joint.tsv", sep="\t", index=False)

    if {"Canonical","Competitor"} <= set(grids):
        D = grids["Competitor"] - grids["Canonical"]
        drows = [dict(x=float(xc[i]), y=float(yc[j]), diff=float(D[i, j]))
                 for i in range(a.xbins) for j in range(a.ybins)]
        pd.DataFrame(drows).to_csv(f"{a.out}_diff.tsv", sep="\t", index=False)
        m = np.abs(D).max()
        print(f"[diff] max |competitor - canonical| proportion = {m:.5f}", file=sys.stderr)

    # enrichment: per bin, log2( p(canonical|bin) / p(canonical) )
    Hc, _, _ = np.histogram2d(d.loc[d.is_canonical, a.x], d.loc[d.is_canonical, a.y], bins=[xe, ye])
    Hk, _, _ = np.histogram2d(d.loc[~d.is_canonical, a.x], d.loc[~d.is_canonical, a.y], bins=[xe, ye])
    Ht = Hc + Hk
    base = float(d.is_canonical.mean())
    frac = (Hc + 0.5) / (Ht + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo2 = np.log2(frac / base)
    lo2 = np.where(Ht >= a.min_n, lo2, np.nan)
    erows = []
    nxb, nyb = len(xe) - 1, len(ye) - 1
    for i in range(nxb):
        for j in range(nyb):
            if not np.isfinite(lo2[i, j]): continue
            erows.append(dict(x=float((xe[i]+xe[i+1])/2), y=float((ye[j]+ye[j+1])/2),
                              xmin=float(xe[i]), xmax=float(xe[i+1]),
                              ymin=float(ye[j]), ymax=float(ye[j+1]),
                              n=int(Ht[i, j]), n_canon=int(Hc[i, j]),
                              frac_canon=float(frac[i, j]), log2_enrich=float(lo2[i, j])))
    E = pd.DataFrame(erows)
    E.to_csv(f"{a.out}_enrich.tsv", sep="\t", index=False)
    kept = int(np.isfinite(lo2).sum()); tot = nxb * nyb
    print(f"[enrich] base rate P(canonical)={base:.4f}; {kept}/{tot} bins pass min_n={a.min_n}; "
          f"log2 range [{np.nanmin(lo2):+.2f}, {np.nanmax(lo2):+.2f}]", file=sys.stderr)

    mrows = []
    for cls, sub in d.groupby("cls"):
        for ax, name in ((a.x, "x"), (a.y, "y")):
            hh, ee = np.histogram(sub[ax].values, bins=(xe if name == "x" else ye))
            cc = (ee[:-1] + ee[1:]) / 2
            for v, c_ in zip(hh / max(hh.sum(), 1), cc):
                mrows.append(dict(cls=cls, axis=name, center=float(c_), prop=float(v)))
    pd.DataFrame(mrows).to_csv(f"{a.out}_marginals.tsv", sep="\t", index=False)

    with open(f"{a.out}_axes.tsv", "w") as fh:
        fh.write("axis\tcolumn\tlo\thi\tcentered\n")
        cen = int(bool(a.center_by_gene))
        fh.write(f"x\t{a.x}\t{xlo:.4f}\t{xhi:.4f}\t{cen}\n")
        fh.write(f"y\t{a.y}\t{ylo:.4f}\t{yhi:.4f}\t{cen}\n")
    pr = []
    for g, gg in d.groupby("gene"):
        cc = gg.loc[gg.is_canonical, a.y]; kk = gg.loc[~gg.is_canonical, a.y]
        if len(cc) and len(kk): pr.append(float(cc.mean() - kk.mean()))
    if pr:
        pr = np.array(pr)
        frac = float((pr > 0).mean())
        print(f"[paired] within-gene canonical - competitor on {a.y}: "
              f"median={np.median(pr):+.3f}, {frac:.1%} of genes positive (n={len(pr):,})",
              file=sys.stderr)
        pd.DataFrame({"gene_diff": pr}).to_csv(f"{a.out}_paired.tsv", sep="\t", index=False)
    print(f"[out] {a.out}_enrich.tsv, {a.out}_joint.tsv, {a.out}_diff.tsv, "
          f"{a.out}_marginals.tsv, {a.out}_axes.tsv", file=sys.stderr)

if __name__ == "__main__":
    main()