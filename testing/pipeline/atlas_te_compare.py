#!/usr/bin/env python3
# are poly(a) sites enriched for transposable elements?
import argparse, os, sys, json, gzip, math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, mannwhitneyu, spearmanr, wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RC = str.maketrans("ACGTN", "TGCAN")
CANON = ("AATAAA", "ATTAAA")
DSE = ("TTTT", "TGTG")

FAMS = ["L1", "Alu", "L2", "MIR", "hAT-Charlie", "TcMar-Tigger", "ERVL-MaLR",
        "ERV1", "ERVL", "CR1", "ERVK", "SVA", "hAT-Tip100", "RTE-X"]

NOT_TE = {"Simple_repeat", "Low_complexity", "Satellite", "rRNA", "scRNA",
          "snRNA", "srpRNA", "tRNA", "RNA", "Unknown", "Other"}

GROUPS = ["architecture pool", "used intronic", "unused, model-flagged",
          "used terminal exon"]


# sequence
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


def gc_of(s):
    n = sum(s.count(b) for b in "ACGT")
    return (s.count("G") + s.count("C")) / n if n else np.nan


# rmsk
def load_rmsk(path, chroms):
    op = gzip.open if path.endswith(".gz") else open
    by = defaultdict(list)
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in chroms:
                continue
            by[f[5]].append((int(f[6]), int(f[7]), f[9], f[11], f[12],
                             float(f[2]) / 10.0))
            n += 1
    out = {}
    for c, v in by.items():
        v.sort(key=lambda x: x[0])
        out[c] = dict(rows=v, start=np.array([x[0] for x in v]))
    print(f"[rmsk] {n:,} intervals", file=sys.stderr)
    return out


def repeat_at(rm, chrom, pos, strand):
    blank = dict(rep=0, fam="none", cls="none", sense=np.nan, is_te=0,
                 el_div=np.nan, el_frac=np.nan)
    e = rm.get(chrom)
    if e is None or not len(e["start"]):
        return blank
    j = np.searchsorted(e["start"], pos, side="right") - 1
    if j < 0 or pos > e["rows"][j][1]:
        return blank
    s0, s1, estr, ecls, efam, ediv = e["rows"][j]
    ln = max(s1 - s0, 1)
    frac = (pos - s0) / ln
    if estr == "-":
        frac = 1.0 - frac
    return dict(rep=1, fam=efam, cls=ecls, sense=float(estr == strand),
                is_te=0 if ecls in NOT_TE else 1,
                el_div=ediv, el_frac=float(frac))


# architectures
def scan_architectures(gen, chrom, lo, hi, strand, dse_lo, dse_hi,
                       site_lo=10, site_hi=30):
    seq = gen.fetch(chrom, lo, hi)
    if len(seq) != hi - lo:
        return []
    if strand == "-":
        seq = seq.translate(RC)[::-1]
    n = len(seq)
    out = []
    mid = (site_lo + site_hi) // 2
    for h in CANON:
        i = seq.find(h)
        while i != -1:
            w0, w1 = i + 6 + dse_lo, min(n, i + 6 + dse_hi)
            if w1 > w0 and any(p in seq[w0:w1] for p in DSE):
                off = i + 6 + mid
                if 0 <= off < n:
                    gpos = (lo + off) if strand != "-" else (hi - off - 1)
                    out.append(gpos)
            i = seq.find(h, i + 1)
    return out


# atlas
def load_atlas(path, codes):
    rows = []
    n = 0
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 11 or f[9] not in codes:
                continue
            n += 1
            hexs = f[10]
            motif = hexs.split("@")[0] if hexs and hexs != "NA" else None
            st, en, strand = int(f[1]), int(f[2]), f[5]
            pos = en - 1 if strand == "+" else st
            try:
                nsamp = int(f[7])
            except ValueError:
                nsamp = 0
            try:
                tpm = float(f[8])
            except ValueError:
                tpm = np.nan
            rows.append(dict(chrom=f[0], pos=pos, strand=strand, code=f[9],
                             n_samples=nsamp, tpm=tpm, motif=motif,
                             has_canon=bool(motif in CANON),
                             has_any_hex=bool(motif is not None)))
    A = pd.DataFrame(rows)
    print(f"[atlas] {n:,} entries in {sorted(codes)}", file=sys.stderr)
    return A


def assign_to_genes(A, span):
    byc = defaultdict(list)
    for r in span.itertuples():
        byc[r.chrom].append((r.lo, r.hi, r.strand, r.gene))
    for c in byc:
        byc[c].sort()
    starts = {c: np.array([x[0] for x in v]) for c, v in byc.items()}
    genes = []
    for r in A.itertuples():
        v = byc.get(r.chrom)
        g = ""
        if v:
            j = np.searchsorted(starts[r.chrom], r.pos, side="right") - 1
            for k in range(max(0, j - 4), min(len(v), j + 1)):
                loo, hii, gstr, gene = v[k]
                if loo <= r.pos <= hii and gstr == r.strand:
                    g = gene
                    break
        genes.append(g)
    A = A.assign(gene=genes)
    return A[A.gene != ""].reset_index(drop=True)


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return q
    pv = p[ok]
    o = np.argsort(pv)
    n = len(pv)
    adj = pv[o] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty(n)
    tmp[o] = np.clip(adj, 0, 1)
    q[ok] = tmp
    return q


def rate_test(a_mask, b_mask):
    ka, na = int(a_mask.sum()), len(a_mask)
    kb, nb = int(b_mask.sum()), len(b_mask)
    ra = ka / na if na else np.nan
    rb = kb / nb if nb else np.nan
    p = np.nan
    if HAVE_SCIPY and na and nb:
        try:
            p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        except Exception:
            pass
    return dict(k_a=ka, n_a=na, rate_a=ra, k_b=kb, n_b=nb, rate_b=rb,
                ratio=(ra / rb) if (rb and rb > 0) else np.nan, p=p)


def boot_ratio(a, b, rng, n=2000):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not len(a) or not len(b):
        return (np.nan, np.nan)
    ra = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    rb = b[rng.integers(0, len(b), size=(n, len(b)))].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(rb > 0, ra / rb, np.nan)
    r = r[np.isfinite(r)]
    if not len(r):
        return (np.nan, np.nan)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def gc_match(target, pool, col="gc_500", nbin=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    both = pd.concat([target[col], pool[col]]).dropna()
    if len(both) < nbin * 5:
        return pool.iloc[[]], 0
    edges = np.unique(np.quantile(both, np.linspace(0, 1, nbin + 1)))
    tb = np.digitize(target[col].to_numpy(), edges[1:-1])
    pb = np.digitize(pool[col].to_numpy(), edges[1:-1])
    keep, short = [], 0
    for b, k in Counter(tb).items():
        idx = np.where(pb == b)[0]
        if len(idx) == 0:
            short += k
            continue
        if len(idx) < k:
            short += k - len(idx)
            keep.append(idx)
        else:
            keep.append(rng.choice(idx, k, replace=False))
    if not keep:
        return pool.iloc[[]], short
    return pool.iloc[np.concatenate(keep)], short


def paired_within_gene(M, ga, gb, fn, min_genes=30):
    rows = []
    for g, sub in M.groupby("gene", sort=False):
        a = sub[sub.group == ga]
        b = sub[sub.group == gb]
        if not len(a) or not len(b):
            continue
        rows.append(dict(gene=g, diff=float(fn(a).mean()) - float(fn(b).mean()),
                         n_a=len(a), n_b=len(b)))
    D = pd.DataFrame(rows)
    if len(D) < min_genes:
        return D, dict(n_genes=len(D))
    v = D["diff"].to_numpy(float)
    v = v[np.isfinite(v)]
    sd = v.std(ddof=1)
    se = sd / math.sqrt(len(v))
    p = np.nan
    if HAVE_SCIPY and sd > 0:
        try:
            p = float(wilcoxon(v, zero_method="zsplit").pvalue)
        except Exception:
            pass
    return D, dict(n_genes=len(v), mean_diff=float(v.mean()),
                   lo=float(v.mean() - 1.96 * se),
                   hi=float(v.mean() + 1.96 * se),
                   dz=float(v.mean() / sd) if sd > 0 else np.nan, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--pad", type=int, default=5000)
    ap.add_argument("--dse_lo", type=int, default=15)
    ap.add_argument("--dse_hi", type=int, default=70)
    ap.add_argument("--gc_win", type=int, default=500)
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--max_pool", type=int, default=120000,
                    help="cap on the architecture pool; sampled if exceeded, "
                         "since measuring every one is the slow step")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out", default="atlaste")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)
    half = a.gc_win // 2

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    POS = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[POS].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    span = (d.groupby(["gene", "chrom", "strand"], as_index=False)
              .agg(lo=("pos", "min"), hi=("pos", "max")))
    span["lo"] = (span.lo - a.pad).clip(lower=0)
    span["hi"] = span.hi + a.pad
    print(f"[set]  {len(span):,} genes, {int(d.is_real.sum()):,} canonical, "
          f"{int((~d.is_real).sum()):,} false positives, "
          f"{(span.hi-span.lo).sum()/1e6:,.0f} Mb", file=sys.stderr)

    gen = Genome(a.genome)
    rm = load_rmsk(a.rmsk, set(d.chrom.unique()))

    print("\n[pool] scanning gene spans for complete architectures...",
          file=sys.stderr)
    pool = []
    for i, r in enumerate(span.itertuples()):
        if i and i % 1000 == 0:
            print(f"       {i:,}/{len(span):,} genes, {len(pool):,} found",
                  file=sys.stderr)
        for p in scan_architectures(gen, r.chrom, r.lo, r.hi, r.strand,
                                    a.dse_lo, a.dse_hi):
            pool.append((r.gene, r.chrom, p, r.strand))
    P = pd.DataFrame(pool, columns=["gene", "chrom", "pos", "strand"])
    n_pool_all = len(P)
    print(f"[pool] {n_pool_all:,} complete architectures across "
          f"{P.gene.nunique():,} genes", file=sys.stderr)
    if n_pool_all > a.max_pool:
        P = P.sample(a.max_pool, random_state=0).reset_index(drop=True)
        print(f"       sampled to {len(P):,} for measurement", file=sys.stderr)

    A_all = assign_to_genes(load_atlas(a.atlas, {"IN", "TE"}), span)
    print(f"[atlas] {len(A_all):,} entries in an analysis gene on its own "
          f"strand: " + ", ".join(f"{k} {v:,}"
                                  for k, v in A_all.code.value_counts().items()),
          file=sys.stderr)

    fl = []
    for code in ("IN", "TE"):
        sub = A_all[A_all.code == code]
        if not len(sub):
            continue
        te = [repeat_at(rm, r.chrom, r.pos, r.strand)["is_te"]
              for r in sub.itertuples()]
        sub = sub.assign(is_te=te)
        for lab, m in (("all", np.ones(len(sub), bool)),
                       ("canonical hexamer", sub.has_canon.to_numpy()),
                       ("no canonical hexamer", ~sub.has_canon.to_numpy())):
            if m.sum() < 20:
                continue
            fl.append(dict(code=code, subset=lab, n=int(m.sum()),
                           frac_TE=round(float(sub.is_te[m].mean()), 4)))
    FL = pd.DataFrame(fl)
    W("hexamer_filter_effect.tsv", FL)
    print("\n[flt]  effect of requiring a canonical hexamer on TE content",
          file=sys.stderr)
    print(FL.to_string(index=False), file=sys.stderr)

    A = A_all[A_all.has_canon]
    if a.min_samples > 1:
        A = A[A.n_samples >= a.min_samples]
    print(f"[atlas] {len(A):,} kept with a canonical hexamer and >= "
          f"{a.min_samples} samples", file=sys.stderr)

    fpset = defaultdict(set)
    for r in d[~d.is_real].itertuples():
        for off in range(-25, 26):
            fpset[r.chrom].add(r.pos + off)
    A = A.assign(also_fp=[bool(r.pos in fpset[r.chrom]) for r in A.itertuples()])

    rows = []
    for r in P.itertuples():
        rows.append(dict(group="architecture pool", gene=r.gene,
                         chrom=r.chrom, pos=r.pos, strand=r.strand,
                         n_samples=0, tpm=np.nan))
    for r in A[(A.code == "IN") & (~A.also_fp)].itertuples():
        rows.append(dict(group="used intronic", gene=r.gene, chrom=r.chrom,
                         pos=r.pos, strand=r.strand,
                         n_samples=r.n_samples, tpm=r.tpm))
    for r in d[~d.is_real].itertuples():
        rows.append(dict(group="unused, model-flagged", gene=r.gene,
                         chrom=r.chrom, pos=r.pos, strand=r.strand,
                         n_samples=0, tpm=np.nan))
    for r in A[A.code == "TE"].itertuples():
        rows.append(dict(group="used terminal exon", gene=r.gene,
                         chrom=r.chrom, pos=r.pos, strand=r.strand,
                         n_samples=r.n_samples, tpm=r.tpm))
    G = pd.DataFrame(rows)
    print("\n[grp]  " + ", ".join(f"{k}: {v:,}"
                                  for k, v in G.group.value_counts().items()),
          file=sys.stderr)

    print(f"[msr]  repeat context and local GC for {len(G):,} positions...",
          file=sys.stderr)
    recs = []
    for i, r in enumerate(G.itertuples()):
        if i and i % 50000 == 0:
            print(f"       {i:,}/{len(G):,}", file=sys.stderr)
        rec = repeat_at(rm, r.chrom, r.pos, r.strand)
        s = gen.fetch(r.chrom, r.pos - half, r.pos + half)
        rec["gc_500"] = gc_of(s) if len(s) >= half else np.nan
        recs.append(rec)
    M = pd.concat([G.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    M["te_sense"] = ((M.is_te == 1) & (M.sense == 1)).astype(int)
    M["te_anti"] = ((M.is_te == 1) & (M.sense == 0)).astype(int)
    W("per_site.tsv", M)

    pool_m = M[M.group == "architecture pool"]
    used = M[M.group == "used intronic"]
    unused = M[M.group == "unused, model-flagged"]
    term = M[M.group == "used terminal exon"]

    print("\n" + "=" * 74, file=sys.stderr)
    print("Q1  How much of the cryptic poly(A) landscape is TE-borne?",
          file=sys.stderr)
    print("=" * 74, file=sys.stderr)
    print(f"       {n_pool_all:,} complete intronic architectures in these "
          f"genes", file=sys.stderr)
    print(f"       {100*pool_m.is_te.mean():.1f}% lie in transposable element "
          f"sequence", file=sys.stderr)
    print(f"         sense-oriented     {100*pool_m.te_sense.mean():5.1f}%",
          file=sys.stderr)
    print(f"         antisense          {100*pool_m.te_anti.mean():5.1f}%",
          file=sys.stderr)
    fam_pool = []
    for f in FAMS:
        k = int((pool_m.fam == f).sum())
        if k < 20:
            continue
        fam_pool.append(dict(family=f, n=k,
                             pct_of_pool=round(100 * k / len(pool_m), 3),
                             pct_of_TE_pool=round(100 * k /
                                                  max(int(pool_m.is_te.sum()), 1), 2),
                             est_total=int(round(n_pool_all * k / len(pool_m)))))
    FP_ = pd.DataFrame(fam_pool).sort_values("n", ascending=False)
    W("architecture_pool_families.tsv", FP_)
    print("\n       families carrying the TE-borne architectures:",
          file=sys.stderr)
    print(f"       {'family':<16}{'% of pool':>11}{'% of TE pool':>14}"
          f"{'est. total':>12}", file=sys.stderr)
    for r in FP_.head(10).itertuples():
        print(f"       {r.family:<16}{r.pct_of_pool:>10.2f}%"
              f"{r.pct_of_TE_pool:>13.1f}%{r.est_total:>12,}", file=sys.stderr)

    print("\n" + "=" * 74, file=sys.stderr)
    print("Q2  Do the models sample that pool evenly?   "
          "Q3  Does the cell?", file=sys.stderr)
    print("=" * 74, file=sys.stderr)
    feats = [("any TE", lambda s: s.is_te == 1),
             ("TE, sense", lambda s: s.te_sense == 1),
             ("TE, antisense", lambda s: s.te_anti == 1)]
    feats += [(f, (lambda f_: lambda s: s.fam == f_)(f)) for f in FAMS]

    rows = []
    for lab, fn in feats:
        mp, mu, mv = fn(pool_m), fn(used), fn(unused)
        if int(mp.sum()) < 20:
            continue
        su = rate_test(mu, mp)
        sv = rate_test(mv, mp)
        lo_u, hi_u = boot_ratio(mu.to_numpy(), mp.to_numpy(), rng, a.nboot)
        lo_v, hi_v = boot_ratio(mv.to_numpy(), mp.to_numpy(), rng, a.nboot)
        rows.append(dict(feature=lab,
                         pool_rate=round(su["rate_b"], 5),
                         used_rate=round(su["rate_a"], 5),
                         unused_rate=round(sv["rate_a"], 5),
                         used_over_pool=round(su["ratio"], 3),
                         used_ci_lo=round(lo_u, 3), used_ci_hi=round(hi_u, 3),
                         used_p=su["p"],
                         unused_over_pool=round(sv["ratio"], 3),
                         unused_ci_lo=round(lo_v, 3), unused_ci_hi=round(hi_v, 3),
                         unused_p=sv["p"],
                         n_pool=su["k_b"], n_used=su["k_a"], n_unused=sv["k_a"]))
    R = pd.DataFrame(rows)
    R["used_q"] = bh(R.used_p)
    R["unused_q"] = bh(R.unused_p)
    W("vs_architecture_pool.tsv", R)
    print(f"       {'feature':<16}{'pool':>8}{'used':>8}{'unused':>9}"
          f"{'used/pool':>11}{'unused/pool':>13}", file=sys.stderr)
    for r in R.itertuples():
        su = "*" if r.used_q < 0.05 else " "
        sv = "*" if r.unused_q < 0.05 else " "
        print(f"       {r.feature:<16}{100*r.pool_rate:>7.2f}%"
              f"{100*r.used_rate:>7.2f}%{100*r.unused_rate:>8.2f}%"
              f"{r.used_over_pool:>10.2f}{su}{r.unused_over_pool:>12.2f}{sv}",
              file=sys.stderr)
    print("       * marks q < 0.05. a ratio near 1.00 means that group samples "
          "the pool\n       in proportion to what the pool contains",
          file=sys.stderr)

    print("\n[gc]   the same comparisons at matched local GC", file=sys.stderr)
    print(f"       median GC: pool {pool_m.gc_500.median():.3f}, "
          f"used {used.gc_500.median():.3f}, "
          f"unused {unused.gc_500.median():.3f}", file=sys.stderr)
    mu_pool, short_u = gc_match(used, pool_m, "gc_500", 20, rng)
    mv_pool, short_v = gc_match(unused, pool_m, "gc_500", 20, rng)
    grows = []
    if len(mu_pool) > 200 and len(mv_pool) > 200:
        print(f"       matched pools: {len(mu_pool):,} for used "
              f"({short_u:,} short), {len(mv_pool):,} for unused "
              f"({short_v:,} short)", file=sys.stderr)
        for lab, fn in feats:
            if int(fn(pool_m).sum()) < 20:
                continue
            su = rate_test(fn(used), fn(mu_pool))
            sv = rate_test(fn(unused), fn(mv_pool))
            grows.append(dict(feature=lab,
                              used_over_matched_pool=round(su["ratio"], 3),
                              used_p=su["p"],
                              unused_over_matched_pool=round(sv["ratio"], 3),
                              unused_p=sv["p"]))
        GM = pd.DataFrame(grows)
        GM["used_q"] = bh(GM.used_p)
        GM["unused_q"] = bh(GM.unused_p)
        W("gc_matched.tsv", GM)
        print(f"       {'feature':<16}{'used/pool':>12}{'q':>10}"
              f"{'unused/pool':>14}{'q':>10}", file=sys.stderr)
        for r in GM.itertuples():
            print(f"       {r.feature:<16}{r.used_over_matched_pool:>12.2f}"
                  f"{r.used_q:>10.2e}{r.unused_over_matched_pool:>14.2f}"
                  f"{r.unused_q:>10.2e}", file=sys.stderr)
        print("       compare against the uncorrected ratios above. anything "
              "that moves to\n       1.00 here was base composition",
              file=sys.stderr)
    else:
        print("       too few matched positions", file=sys.stderr)

    print("\n[pair] paired within gene, against the architecture pool",
          file=sys.stderr)
    prows = []
    for lab, fn in feats[:6]:
        for grp, tag in (("used intronic", "used"),
                         ("unused, model-flagged", "unused")):
            D, st = paired_within_gene(M, grp, "architecture pool", fn)
            if "mean_diff" not in st:
                continue
            prows.append(dict(feature=lab, group=tag, n_genes=st["n_genes"],
                              mean_diff=round(st["mean_diff"], 5),
                              lo=round(st["lo"], 5), hi=round(st["hi"], 5),
                              dz=round(st["dz"], 4), p=st["p"]))
            print(f"       {lab:<16} {tag:<8} diff {st['mean_diff']:+.4f} "
                  f"[{st['lo']:+.4f}, {st['hi']:+.4f}]  dz {st['dz']:+.3f}  "
                  f"n {st['n_genes']:,}  p {st['p']:.2e}", file=sys.stderr)
    W("paired_within_gene.tsv", pd.DataFrame(prows))

    u = used[used.n_samples > 0].copy()
    if len(u) > 400:
        print("\n[dose] TE overlap against how broadly a used site is detected",
              file=sys.stderr)
        u["q"] = pd.qcut(u.n_samples.rank(method="first"), 4, labels=False)
        drows = []
        for q, t in u.groupby("q"):
            drows.append(dict(quartile=int(q) + 1, n=len(t),
                              median_samples=float(t.n_samples.median()),
                              frac_TE=round(float(t.is_te.mean()), 4),
                              frac_TE_sense=round(float(t.te_sense.mean()), 4)))
        DS = pd.DataFrame(drows)
        W("usage_dose.tsv", DS)
        print(DS.to_string(index=False), file=sys.stderr)
        if HAVE_SCIPY:
            rho, prho = spearmanr(u.n_samples, u.is_te)
            print(f"       Spearman {rho:+.3f} (p {prho:.2e})", file=sys.stderr)

    print(f"\n[ref]  used terminal-exon sites: {100*term.is_te.mean():.1f}% in "
          f"a TE (n {len(term):,})", file=sys.stderr)
    print("       exonic, and exons are repeat-poor because they are under "
          "selection, so\n       this is location rather than usage and is not "
          "compared with the rest", file=sys.stderr)

    print("\n" + "=" * 74, file=sys.stderr)
    te_pool = float(pool_m.is_te.mean())
    te_u = float(used.is_te.mean())
    te_v = float(unused.is_te.mean())
    print(f"       {100*te_pool:.1f}% of complete intronic architectures are "
          f"TE-borne.", file=sys.stderr)
    print(f"       the models flag them at {te_v/te_pool:.2f}x that rate; "
          f"the cell uses them at\n       {te_u/te_pool:.2f}x.",
          file=sys.stderr)
    if abs(te_v / te_pool - 1) < 0.12:
        print("       the models sample the pool close to evenly: TE origin "
              "neither attracts\n       nor repels them.", file=sys.stderr)
    if len(grows):
        strong = GM[(GM.used_q < 0.05) &
                    ((GM.used_over_matched_pool < 0.7) |
                     (GM.used_over_matched_pool > 1.4))]
        if len(strong):
            print("       families whose architectures are used at a rate "
                  "distinct from the pool,\n       after GC matching:",
                  file=sys.stderr)
            for r in strong.itertuples():
                print(f"         {r.feature:<16} "
                      f"{r.used_over_matched_pool:.2f}x", file=sys.stderr)
        else:
            print("       no family's architectures are used at a rate "
                  "distinguishable from the\n       pool once GC is matched.",
                  file=sys.stderr)
    print("=" * 74, file=sys.stderr)

    W("plot_pool_families.tsv", FP_)
    W("plot_ratios.tsv", R)
    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_pool_all=n_pool_all, n_pool_measured=len(pool_m),
                       n_used_intronic=len(used), n_unused=len(unused),
                       n_terminal=len(term),
                       te_frac_pool=te_pool, te_frac_used=te_u,
                       te_frac_unused=te_v), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()