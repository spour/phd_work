#!/usr/bin/env python3
# how should duplex stability be measured?
import argparse, os, sys, json, math
from collections import defaultdict

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")
try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# tables

DNA_SL98 = {"AA": -1.00, "TT": -1.00, "AT": -0.88, "TA": -0.58,
            "CA": -1.45, "TG": -1.45, "GT": -1.44, "AC": -1.44,
            "CT": -1.28, "AG": -1.28, "GA": -1.30, "TC": -1.30,
            "CG": -2.17, "GC": -2.24, "GG": -1.84, "CC": -1.84}

DNA_BRES = {"AA": -1.9, "TT": -1.9, "AT": -1.5, "TA": -0.9,
            "CA": -1.9, "TG": -1.9, "GT": -1.3, "AC": -1.3,
            "CT": -1.6, "AG": -1.6, "GA": -1.6, "TC": -1.6,
            "CG": -3.6, "GC": -3.1, "GG": -3.1, "CC": -3.1}

DNA_DH = {"AA": -7.9, "TT": -7.9, "AT": -7.2, "TA": -7.2,
          "CA": -8.5, "TG": -8.5, "GT": -8.4, "AC": -8.4,
          "CT": -7.8, "AG": -7.8, "GA": -8.2, "TC": -8.2,
          "CG": -10.6, "GC": -9.8, "GG": -8.0, "CC": -8.0}

RNA_TURNER = {"AA": -0.93, "AT": -1.10, "AC": -2.24, "AG": -2.08,
              "TA": -1.33, "TT": -0.93, "TC": -2.35, "TG": -2.11,
              "CA": -2.11, "CT": -2.08, "CC": -3.26, "CG": -2.36,
              "GA": -2.35, "GT": -2.24, "GC": -3.42, "GG": -3.26}

HYBRID_SUG = {"AA": -1.0, "AC": -2.1, "AG": -1.8, "AT": -0.9,
              "CA": -0.9, "CC": -2.1, "CG": -1.7, "CT": -0.9,
              "GA": -1.3, "GC": -2.7, "GG": -2.9, "GT": -1.1,
              "TA": -0.6, "TC": -1.5, "TG": -1.6, "TT": -0.2}

GC_NULL = {k: (-1.0 if set(k) <= set("GC") else
               (-0.5 if len(set(k) & set("GC")) == 1 else 0.0))
           for k in DNA_SL98}

TABLES = {"SantaLucia98_dG": DNA_SL98,
          "Breslauer86_dG": DNA_BRES,
          "SantaLucia98_dH": DNA_DH,
          "TurnerRNA_dG": RNA_TURNER,
          "SugimotoHybrid_dG": HYBRID_SUG,
          "GCidentity_null": GC_NULL}

WINDOWS = [(-25, 25), (-50, 50), (-100, 100), (-150, 150), (-250, 250),
           (-400, 400), (-600, 600), (-1000, 1000), (-1500, 1500),
           (-2500, 2500),
           (-100, 0), (0, 100), (-250, 0), (0, 250), (-500, 0), (0, 500),
           (-60, -10), (10, 60), (-40, 40), (-1000, 0), (0, 1000)]


class Genome:
    def __init__(self, path):
        fai = path + ".fai"
        if not os.path.exists(fai):
            sys.exit("need " + fai + "; run `samtools faidx " + path + "`")
        self.idx = {}
        for line in open(fai):
            f = line.split("\t")
            self.idx[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self.fh = open(path, "rb")

    def fetch(self, chrom, start, end):
        if chrom not in self.idx or end <= start:
            return ""
        ln, off, lb, lw = self.idx[chrom]
        start, end = max(0, start), min(end, ln)
        if end <= start:
            return ""
        def seek(p):
            return off + p // lb * lw + p % lb
        self.fh.seek(seek(start))
        return self.fh.read(seek(end) - seek(start)).decode().replace("\n", "").upper()


def oriented(gen, chrom, pos, strand, up, dn):
    if strand == "-":
        s = gen.fetch(chrom, int(pos) - dn, int(pos) + up)
        return s.translate(RC)[::-1] if len(s) >= up + dn else ""
    s = gen.fetch(chrom, int(pos) - up, int(pos) + dn)
    return s if len(s) >= up + dn else ""


def di_track(seq, table):
    v = np.full(len(seq) - 1, np.nan)
    for i in range(len(seq) - 1):
        v[i] = table.get(seq[i:i + 2], np.nan)
    return v


def sliding(v, w):
    ok = np.isfinite(v)
    x = np.where(ok, v, 0.0)
    k = np.ones(w)
    num = np.convolve(x, k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def summaries(v, melt_w):
    out = {}
    x = v[np.isfinite(v)]
    if x.size < 5:
        return {k: np.nan for k in
                ("mean", "median", "sd", "range", "slidmin", "slidmax",
                 "frac_weak", "sum")}
    out["mean"] = float(x.mean())
    out["median"] = float(np.median(x))
    out["sd"] = float(x.std(ddof=1))
    out["range"] = float(x.max() - x.min())
    out["sum"] = float(x.sum())
    if len(v) >= melt_w:
        s = sliding(v, melt_w)
        s = s[np.isfinite(s)]
        if s.size:
            out["slidmin"] = float(s.max())
            out["slidmax"] = float(s.min())
        else:
            out["slidmin"] = out["slidmax"] = np.nan
    else:
        out["slidmin"] = out["slidmax"] = np.nan
    out["frac_weak"] = float((x > -1.2).mean())
    return out


def gc_of(seq):
    n = sum(seq.count(b) for b in "ACGT")
    return (seq.count("G") + seq.count("C")) / n if n else np.nan


def paired(dfr, dff, col):
    a = dfr[col].astype(float)
    b = dff[col].astype(float)
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).dropna()


def dz_of(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 20:
        return np.nan, len(v)
    sd = v.std(ddof=1)
    return (float(v.mean() / sd) if sd > 0 else np.nan), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--pad", type=int, default=3000,
                    help="sequence fetched each side; must exceed the widest "
                         "window")
    ap.add_argument("--melt_w", type=int, default=13,
                    help="sliding width for the melting-bubble summaries, in "
                         "dinucleotides; Pol II opens roughly 12-14 bp")
    ap.add_argument("--max_ccpa", type=int, default=8)
    ap.add_argument("--nperm", type=int, default=500)
    ap.add_argument("--out", default="stab")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)
    PAD = a.pad
    if PAD < max(abs(w[0]) for w in WINDOWS) + 50:
        sys.exit("--pad must exceed the widest window")

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    pos = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[pos].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    if a.max_ccpa > 0:
        keep = []
        for g, sub in d.groupby("gene", sort=False):
            r = sub[sub.is_real]; c = sub[~sub.is_real]
            if len(c) > a.max_ccpa:
                c = c.sample(a.max_ccpa, random_state=0)
            keep.append(pd.concat([r, c]))
        d = pd.concat(keep, ignore_index=True)
    print(f"[set]  {int(d.is_real.sum()):,} real, "
          f"{int((~d.is_real).sum()):,} false positives, "
          f"{d.gene.nunique():,} genes", file=sys.stderr)
    print(f"[cfg]  {len(TABLES)} parameter sets x {len(WINDOWS)} windows x 8 "
          f"summaries", file=sys.stderr)

    gen = Genome(a.genome)
    rows = []
    nbad = 0
    for i, r in enumerate(d.itertuples()):
        if i and i % 5000 == 0:
            print(f"[run]  {i:,}/{len(d):,} sites...", file=sys.stderr)
        seq = oriented(gen, r.chrom, r.pos, r.strand, PAD, PAD)
        if len(seq) != 2 * PAD:
            nbad += 1
            continue
        rec = dict(gene=r.gene, is_real=bool(r.is_real))
        tracks = {name: di_track(seq, tab) for name, tab in TABLES.items()}
        for w0, w1 in WINDOWS:
            tag = f"{w0}:{w1}"
            i0, i1 = PAD + w0, PAD + w1
            rec["GC." + tag] = gc_of(seq[i0:i1])
            for name in TABLES:
                seg = tracks[name][max(0, i0):max(0, i1 - 1)]
                for k, v in summaries(seg, a.melt_w).items():
                    rec[f"{name}.{tag}.{k}"] = v
        rows.append(rec)
    if nbad:
        print(f"[seq]  {nbad:,} sites dropped at contig edges", file=sys.stderr)
    P = pd.DataFrame(rows)
    print(f"[seq]  {len(P):,} sites, {P.shape[1]:,} columns", file=sys.stderr)

    R = P[P.is_real].drop_duplicates("gene").set_index("gene")
    F = P[~P.is_real].groupby("gene").mean(numeric_only=True)
    feats = [c for c in P.columns if c not in ("gene", "is_real")]

    res = []
    for c in feats:
        v = paired(R, F, c)
        dz, n = dz_of(v.to_numpy())
        if not np.isfinite(dz):
            continue
        parts = c.split(".")
        res.append(dict(feature=c,
                        table=parts[0] if len(parts) == 3 else "GC",
                        window=parts[1] if len(parts) == 3 else parts[1],
                        summary=parts[2] if len(parts) == 3 else "gc",
                        dz=round(dz, 4), n_genes=n,
                        mean_diff=round(float(v.mean()), 6),
                        frac_pos=round(float((v > 0).mean()), 4)))
    S = pd.DataFrame(res)
    S["abs_dz"] = S.dz.abs()
    S = S.sort_values("abs_dz", ascending=False)
    W("sweep_all.tsv", S.drop(columns="abs_dz"))

    print("\n[top]  best 20 of "
          f"{len(S):,} combinations, by |dz|", file=sys.stderr)
    print(f"       {'table':<20}{'window':>12}{'summary':>10}{'dz':>9}"
          f"{'%genes':>8}", file=sys.stderr)
    for r_ in S.head(20).itertuples():
        print(f"       {r_.table:<20}{r_.window:>12}{r_.summary:>10}"
              f"{r_.dz:>+9.3f}{100*r_.frac_pos:>7.1f}%", file=sys.stderr)

    base = S[(S.table == "SantaLucia98_dG") & (S.window == "-250:250") &
             (S.summary == "mean")]
    if len(base):
        b = float(base.dz.iloc[0])
        best = float(S.dz.iloc[0])
        print(f"\n[cmp]  the original core_dG (SantaLucia, +/-250, mean) is "
              f"dz {b:+.3f}\n       the best combination is {best:+.3f}, an "
              f"improvement of {abs(best)-abs(b):+.3f}", file=sys.stderr)

    # window profile at fixed table and summary
    prof = S[(S.table == "SantaLucia98_dG") & (S.summary == "mean")]
    prof = prof.assign(w0=[int(w.split(":")[0]) for w in prof.window],
                       w1=[int(w.split(":")[1]) for w in prof.window])
    sym = prof[prof.w0 == -prof.w1].sort_values("w1")
    print("\n[win]  symmetric windows, SantaLucia dG, mean:", file=sys.stderr)
    for r_ in sym.itertuples():
        print(f"       +/-{r_.w1:>5} nt   dz {r_.dz:+.3f}", file=sys.stderr)
    asym = prof[prof.w0 != -prof.w1].sort_values("dz", key=abs, ascending=False)
    print("\n[win]  asymmetric windows:", file=sys.stderr)
    for r_ in asym.head(8).itertuples():
        print(f"       {r_.window:>12}   dz {r_.dz:+.3f}", file=sys.stderr)
    W("window_profile.tsv", prof.drop(columns="abs_dz"))

    # table comparison at the best window
    bw = S.iloc[0].window
    tc = S[(S.window == bw) & (S.summary == "mean")]
    print(f"\n[tab]  parameter sets at window {bw}, mean summary:",
          file=sys.stderr)
    for r_ in tc.sort_values("dz", key=abs, ascending=False).itertuples():
        note = ""
        if r_.table == "GCidentity_null":
            note = "   <- this one IS GC content"
        print(f"       {r_.table:<22} dz {r_.dz:+.3f}{note}", file=sys.stderr)
    W("table_comparison.tsv", tc.drop(columns="abs_dz"))

    # is it just gc?
    best_feat = S.iloc[0].feature
    bwin = S.iloc[0].window
    gcf = "GC." + bwin
    print(f"\n[gc]   decomposing {best_feat} against {gcf}", file=sys.stderr)
    A = paired(R, F, best_feat)
    B = paired(R, F, gcf)
    common = A.index.intersection(B.index)
    A, B = A.loc[common].to_numpy(), B.loc[common].to_numpy()
    m = np.isfinite(A) & np.isfinite(B)
    A, B = A[m], B[m]
    r = float(np.corrcoef(A, B)[0, 1])
    bt = np.polyfit(B, A, 1)
    resA = A - np.polyval(bt, B)
    bt2 = np.polyfit(A, B, 1)
    resB = B - np.polyval(bt2, A)
    dzA, _ = dz_of(A); dzB, _ = dz_of(B)
    dzRA, _ = dz_of(resA); dzRB, _ = dz_of(resB)
    print(f"       r(dG diff, GC diff) = {r:+.3f}", file=sys.stderr)
    print(f"       dG alone                dz {dzA:+.3f}", file=sys.stderr)
    print(f"       GC alone                dz {dzB:+.3f}", file=sys.stderr)
    print(f"       dG given GC             dz {dzRA:+.3f}", file=sys.stderr)
    print(f"       GC given dG             dz {dzRB:+.3f}", file=sys.stderr)
    gcnull = S[(S.table == "GCidentity_null") & (S.window == bwin) &
               (S.summary == "mean")]
    if len(gcnull):
        print(f"       GC-identity table       dz {float(gcnull.dz.iloc[0]):+.3f}",
              file=sys.stderr)
    print("       if dG given GC keeps most of its effect, stacking preference "
          "is\n       carrying signal that base composition does not",
          file=sys.stderr)
    W("gc_decomposition.tsv", pd.DataFrame([dict(
        feature=best_feat, gc_feature=gcf, r=round(r, 4),
        dz_dG=round(dzA, 4), dz_GC=round(dzB, 4),
        dz_dG_given_GC=round(dzRA, 4), dz_GC_given_dG=round(dzRB, 4),
        n_genes=len(A))]))

    print(f"\n[perm] {a.nperm} within-gene label permutations, "
          f"tracking the maximum |dz| over the top 50 combinations",
          file=sys.stderr)
    top50 = S.head(50).feature.tolist()
    gl = {}
    for g, sub in P.groupby("gene", sort=False):
        if len(sub) >= 2:
            gl[g] = sub[top50].to_numpy(float)
    obs_max = float(S.abs_dz.iloc[0])
    nulls = []
    for it in range(a.nperm):
        if it and it % 100 == 0:
            print(f"       {it}/{a.nperm}", file=sys.stderr)
        acc = []
        for g, M in gl.items():
            k = rng.integers(0, len(M))
            acc.append(M[k] - np.nanmean(np.delete(M, k, axis=0), axis=0))
        Aa = np.array(acc, float)
        sd = np.nanstd(Aa, axis=0, ddof=1)
        dz = np.nanmean(Aa, axis=0) / np.where(sd > 0, sd, np.nan)
        nulls.append(float(np.nanmax(np.abs(dz))))
    nulls = np.array(nulls)
    p = (1 + int((nulls >= obs_max).sum())) / (len(nulls) + 1)
    print(f"\n[perm] observed max |dz| {obs_max:.3f}; null median "
          f"{np.median(nulls):.3f}, 95th {np.percentile(nulls, 95):.3f}, "
          f"p {p:.3g}", file=sys.stderr)
    W("permutation.tsv", pd.DataFrame(dict(max_abs_dz=nulls)))

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_sites=len(P), n_genes=int(P.gene.nunique()),
                       n_combinations=len(S), best=S.iloc[0].feature,
                       best_dz=float(S.dz.iloc[0]), perm_p=p,
                       melt_w=a.melt_w), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead [gc] first. If dG given GC collapses, the stability finding "
          "is GC\ncontent and should be called that. Then [win]: a narrow "
          "optimum means a local\nproperty of the site, a flat one means a "
          "regional property and the 'core'\nlabel is wrong.", file=sys.stderr)


if __name__ == "__main__":
    main()