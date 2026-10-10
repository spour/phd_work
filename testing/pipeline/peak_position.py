#!/usr/bin/env python3
# where do the internal (competitor) peaks sit?
import argparse, os, sys, re
import numpy as np, pandas as pd
from collections import defaultdict

def parse_gtf(gtf):
    genes = {}
    exons = defaultdict(list)
    pat = re.compile(r'gene_name "([^"]+)"')
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"): continue
            f = line.split("\t")
            if len(f) < 9: continue
            typ = f[2]
            if typ not in ("gene", "exon"): continue
            m = pat.search(f[8])
            if not m: continue
            sym = m.group(1); chrom=f[0]; s=int(f[3]); e=int(f[4]); strand=f[6]
            if typ == "gene":
                if sym not in genes or (e-s) > (genes[sym]["end"]-genes[sym]["start"]):
                    genes[sym] = dict(chrom=chrom, strand=strand, start=s, end=e)
            else:
                exons[sym].append((s, e))
    exon_iv = {}; intron_iv = {}
    for sym, ev in exons.items():
        ev = sorted(set(ev))
        merged = []
        for s,e in ev:
            if merged and s <= merged[-1][1]+1: merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else: merged.append((s,e))
        exon_iv[sym] = merged
        intr = [(merged[i][1]+1, merged[i+1][0]-1) for i in range(len(merged)-1)
                if merged[i+1][0]-1 >= merged[i][1]+1]
        intron_iv[sym] = intr
    print(f"[gtf] {len(genes):,} genes, exon models for {len(exon_iv):,}", file=sys.stderr)
    return genes, exon_iv, intron_iv

def classify_and_position(center, gene, gmeta, exon_iv, intron_iv):
    g = gmeta.get(gene)
    if g is None: return ("unknown", np.nan, np.nan)
    strand = g["strand"]; gs, ge = g["start"], g["end"]
    if ge > gs:
        gf = (center - gs) / (ge - gs)
        if strand == "-": gf = 1.0 - gf
    else: gf = np.nan
    gf = float(min(max(gf, 0.0), 1.0)) if np.isfinite(gf) else np.nan
    for (s,e) in exon_iv.get(gene, []):
        if s <= center <= e: return ("exon", np.nan, gf)
    for (s,e) in intron_iv.get(gene, []):
        if s <= center <= e:
            frac = (center - s) / (e - s) if e > s else np.nan
            if strand == "-": frac = 1.0 - frac
            return ("intron", float(frac), gf)
    return ("outside", np.nan, gf)

def load_atlas_types(atlas_bed):
    rows = []
    with open(atlas_bed) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 10: continue
            try: rep = int(p[3].split(":")[1]) - 1
            except Exception: continue
            chrom = p[0] if p[0].startswith("chr") else "chr"+p[0]
            rows.append((chrom, rep, p[9]))
    a = pd.DataFrame(rows, columns=["chrom","mid","type"])
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--contested_only", action="store_true", default=True)
    ap.add_argument("--out", default="peak_position")
    a = ap.parse_args()

    W = pd.read_parquet(a.shortlist)
    if a.contested_only: W = W[W.n_internal > 0].copy()
    if "center" not in W.columns: W["center"] = W["window_start_genomic"] + 250
    W["cls"] = np.where(W.is_canonical, "canonical", "competitor")
    print(f"[data] {len(W):,} peaks ({(W.cls=='competitor').sum():,} competitor / "
          f"{(W.cls=='canonical').sum():,} canonical)", file=sys.stderr)

    gmeta, exon_iv, intron_iv = parse_gtf(a.gtf)
    ann=[]; ifrac=[]; gfrac=[]
    for _, r in W.iterrows():
        an, inf, gf = classify_and_position(int(r["center"]), r["gene"], gmeta, exon_iv, intron_iv)
        ann.append(an); ifrac.append(inf); gfrac.append(gf)
    W["annotation"] = ann; W["intron_frac"] = ifrac; W["gene_frac"] = gfrac

    brk = (W.groupby(["cls","annotation"]).size().reset_index(name="n"))
    brk["frac"] = brk.groupby("cls")["n"].transform(lambda x: x/x.sum())
    brk.to_csv(f"{a.out}_annotation.tsv", sep="\t", index=False)

    if a.atlas and os.path.exists(a.atlas):
        at = load_atlas_types(a.atlas)
        at_by = {c: g.sort_values("mid") for c,g in at.groupby("chrom")}
        types=[]
        for _, r in W.iterrows():
            sub = at_by.get(r["chrom"])
            if sub is None: types.append("none"); continue
            i = sub["mid"].searchsorted(int(r["center"]))
            best="none"; bd=251
            for j in (i-1,i):
                if 0<=j<len(sub):
                    d=abs(int(sub["mid"].iloc[j])-int(r["center"]))
                    if d<bd: bd=d; best=sub["type"].iloc[j]
            types.append(best)
        W["atlas_type"]=types
        tb=(W.groupby(["cls","atlas_type"]).size().reset_index(name="n"))
        tb["frac"]=tb.groupby("cls")["n"].transform(lambda x:x/x.sum())
        tb.to_csv(f"{a.out}_atlastype.tsv", sep="\t", index=False)

    intr = W[W.annotation=="intron"].dropna(subset=["intron_frac"])
    ih=[]
    for cls,gg in intr.groupby("cls"):
        h,edges=np.histogram(gg.intron_frac, bins=20, range=(0,1))
        for k in range(20):
            ih.append(dict(cls=cls, bin_mid=(edges[k]+edges[k+1])/2, n=int(h[k]),
                           frac=float(h[k]/max(1,h.sum()))))
    pd.DataFrame(ih).to_csv(f"{a.out}_intronpos.tsv", sep="\t", index=False)

    gh=[]
    for cls,gg in W.dropna(subset=["gene_frac"]).groupby("cls"):
        h,edges=np.histogram(gg.gene_frac, bins=20, range=(0,1))
        for k in range(20):
            gh.append(dict(cls=cls, bin_mid=(edges[k]+edges[k+1])/2, n=int(h[k]),
                           frac=float(h[k]/max(1,h.sum()))))
    pd.DataFrame(gh).to_csv(f"{a.out}_genepos.tsv", sep="\t", index=False)

    comp = W[W.cls=="competitor"]
    print(f"\n=== competitor peak annotation ===", file=sys.stderr)
    vc = comp.annotation.value_counts(normalize=True)
    for k,v in vc.items(): print(f"  {k:<9} {v:.1%}", file=sys.stderr)
    med_if = intr[intr.cls=='competitor'].intron_frac.median()
    print(f"\n[intron pos] competitor median intron_frac = {med_if:.2f} "
          f"(0=5' end, 1=3' end of intron)", file=sys.stderr)
    print(f"[gene pos] competitor median gene_frac = {comp.gene_frac.median():.2f} "
          f"(0=TSS, 1=polyA)", file=sys.stderr)
    print(f"\n[out] {a.out}_annotation.tsv, _atlastype.tsv, _intronpos.tsv, _genepos.tsv", file=sys.stderr)

if __name__ == "__main__":
    main()
