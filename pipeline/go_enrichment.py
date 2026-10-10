#!/usr/bin/env python3
# do the genes that accumulate model false positives share any functional character?
import argparse, os, sys, json, gzip, math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom, mannwhitneyu, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

EXCLUDE_EVIDENCE = {"IEA", "ND"}
NS = {"P": "biological process", "F": "molecular function",
      "C": "cellular component"}


def load_obo(path):
    name, ns, parents, obsolete = {}, {}, defaultdict(set), set()
    cur = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {}
                continue
            if line.startswith("[") and line != "[Term]":
                cur = None
                continue
            if cur is None or not line:
                continue
            if line.startswith("id: GO:"):
                cur["id"] = line[4:].strip()
            elif line.startswith("name:"):
                if "id" in cur:
                    name[cur["id"]] = line[5:].strip()
            elif line.startswith("namespace:"):
                if "id" in cur:
                    v = line[10:].strip()
                    ns[cur["id"]] = {"biological_process": "P",
                                     "molecular_function": "F",
                                     "cellular_component": "C"}.get(v, "?")
            elif line.startswith("is_a: GO:"):
                if "id" in cur:
                    parents[cur["id"]].add(line[6:].split("!")[0].strip())
            elif line.startswith("is_obsolete: true"):
                if "id" in cur:
                    obsolete.add(cur["id"])
    for t in obsolete:
        name.pop(t, None)
        ns.pop(t, None)
        parents.pop(t, None)
    print(f"[obo]  {len(name):,} terms, {len(obsolete):,} obsolete dropped",
          file=sys.stderr)
    return name, ns, parents


def ancestors(term, parents, cache):
    if term in cache:
        return cache[term]
    out = set()
    stack = list(parents.get(term, ()))
    while stack:
        t = stack.pop()
        if t in out:
            continue
        out.add(t)
        stack.extend(parents.get(t, ()))
    cache[term] = out
    return out


def load_gaf(path, parents, keep_iea=False):
    op = gzip.open if path.endswith(".gz") else open
    g2t = defaultdict(set)
    cache = {}
    n = skipped = 0
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 15:
                continue
            sym, term, ev, qual = f[2], f[4], f[6], f[3]
            if "NOT" in qual:
                continue
            if not keep_iea and ev in EXCLUDE_EVIDENCE:
                skipped += 1
                continue
            g2t[sym].add(term)
            g2t[sym] |= ancestors(term, parents, cache)
            n += 1
    print(f"[gaf]  {n:,} annotations over {len(g2t):,} symbols "
          f"({skipped:,} skipped by evidence code)", file=sys.stderr)
    return g2t


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


def enrich(fg, bg, g2t, name, ns, min_term, max_term, label):
    fg = set(fg) & set(g2t)
    bg = set(bg) & set(g2t)
    if not fg or not bg:
        return pd.DataFrame()
    cnt_fg, cnt_bg = Counter(), Counter()
    for g in fg:
        cnt_fg.update(g2t[g])
    for g in bg:
        cnt_bg.update(g2t[g])
    N, K = len(bg), len(fg)
    rows = []
    for term, k in cnt_fg.items():
        n_bg = cnt_bg.get(term, 0)
        if n_bg < min_term or n_bg > max_term or k < 3:
            continue
        p = hypergeom.sf(k - 1, N, n_bg, K) if HAVE_SCIPY else np.nan
        exp = K * n_bg / N
        rows.append(dict(comparison=label, term=term,
                         name=name.get(term, term),
                         namespace=NS.get(ns.get(term, "?"), "?"),
                         n_fg=k, n_bg=n_bg, K=K, N=N,
                         expected=round(exp, 2),
                         fold=round(k / exp, 3) if exp > 0 else np.nan,
                         p=p))
    R = pd.DataFrame(rows)
    if not len(R):
        return R
    R["q"] = np.nan
    for nsp, idx in R.groupby("namespace").groups.items():
        R.loc[idx, "q"] = bh(R.loc[idx, "p"].to_numpy())
    return R.sort_values("q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--gaf", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--pad", type=int, default=5000)
    ap.add_argument("--top_frac", type=float, default=0.10,
                    help="foreground is the top fraction by FP rate")
    ap.add_argument("--n_length_bins", type=int, default=10)
    ap.add_argument("--min_term", type=int, default=10)
    ap.add_argument("--max_term", type=int, default=1000)
    ap.add_argument("--keep_iea", action="store_true")
    ap.add_argument("--out", default="go")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    if not HAVE_SCIPY:
        sys.exit("scipy required for the hypergeometric test")

    # per-gene burden
    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    POS = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[POS].astype(int),
                      "is_real": pk.is_canonical.astype(bool)})

    G = (d.groupby("gene", as_index=False)
           .agg(chrom=("chrom", "first"), lo=("pos", "min"), hi=("pos", "max"),
                n_fp=("is_real", lambda s: int((~s).sum())),
                n_real=("is_real", "sum")))
    G["span_kb"] = (G.hi - G.lo + 2 * a.pad) / 1000.0
    G["fp_per_kb"] = G.n_fp / G.span_kb.clip(lower=1)
    W("gene_burden.tsv", G)

    rho, prho = spearmanr(G.span_kb, G.n_fp)
    rho2, prho2 = spearmanr(G.span_kb, G.fp_per_kb)
    print(f"[burden] {len(G):,} genes", file=sys.stderr)
    print(f"       span vs FP COUNT: Spearman {rho:+.3f} (p {prho:.2e})",
          file=sys.stderr)
    print(f"       span vs FP RATE:  Spearman {rho2:+.3f} (p {prho2:.2e})",
          file=sys.stderr)
    print("       the first is the confound; the second is what the "
          "foreground is\n       defined on, and should be much weaker",
          file=sys.stderr)

    # foreground and backgrounds
    G = G.sort_values("fp_per_kb", ascending=False).reset_index(drop=True)
    k = max(50, int(a.top_frac * len(G)))
    fg = G.head(k).gene.tolist()
    bg_all = G.gene.tolist()

    G["len_bin"] = pd.qcut(G.span_kb.rank(method="first"), a.n_length_bins,
                           labels=False)
    fg_set = set(fg)
    rng = np.random.default_rng(0)
    matched = []
    need = Counter(G[G.gene.isin(fg_set)].len_bin)
    for b, n_need in need.items():
        pool = G[(G.len_bin == b) & (~G.gene.isin(fg_set))].gene.tolist()
        if not pool:
            continue
        take = min(len(pool), n_need * 5)
        matched.extend(rng.choice(pool, take, replace=False).tolist())
    print(f"[fg]   {len(fg):,} foreground genes (top {100*a.top_frac:.0f}% by "
          f"FP/kb), {len(matched):,} length-matched background genes",
          file=sys.stderr)
    fgd = G[G.gene.isin(fg_set)]
    mbd = G[G.gene.isin(set(matched))]
    print(f"       median span: foreground {fgd.span_kb.median():,.0f} kb, "
          f"matched background {mbd.span_kb.median():,.0f} kb, "
          f"all genes {G.span_kb.median():,.0f} kb", file=sys.stderr)
    u, pu = mannwhitneyu(fgd.span_kb, mbd.span_kb)
    print(f"       span difference after matching: p {pu:.3f} "
          f"(a large p means the matching worked)", file=sys.stderr)
    W("gene_sets.tsv", pd.DataFrame(dict(
        gene=fg + matched,
        set=["foreground"] * len(fg) + ["matched_background"] * len(matched))))

    # annotations
    name, ns, parents = load_obo(a.obo)
    g2t = load_gaf(a.gaf, parents, a.keep_iea)
    cov = len(set(G.gene) & set(g2t)) / max(len(G), 1)
    print(f"[map]  {100*cov:.1f}% of genes have an annotation; the rest are "
          f"dropped from\n       both foreground and background",
          file=sys.stderr)

    # the two tests
    R1 = enrich(fg, bg_all, g2t, name, ns, a.min_term, a.max_term,
                "unmatched")
    R2 = enrich(fg, matched, g2t, name, ns, a.min_term, a.max_term,
                "length-matched")
    R = pd.concat([R1, R2], ignore_index=True)
    W("enrichment.tsv", R)

    def show(tab, lab):
        s = tab[tab.q < 0.05].sort_values("q").head(15)
        print(f"\n[{lab}]", file=sys.stderr)
        if not len(s):
            print("       nothing at q < 0.05", file=sys.stderr)
            return
        print(f"       {'term':<44}{'ns':>4}{'fg':>6}{'exp':>8}{'fold':>7}"
              f"{'q':>11}", file=sys.stderr)
        for r in s.itertuples():
            nm = r.name if len(r.name) <= 42 else r.name[:39] + "..."
            print(f"       {nm:<44}{r.namespace[:1].upper():>4}{r.n_fg:>6}"
                  f"{r.expected:>8.1f}{r.fold:>7.2f}{r.q:>11.2e}",
                  file=sys.stderr)

    show(R1, "unmatched  -- includes anything driven by gene length")
    show(R2, "length-matched  -- what survives holding length fixed")

    s1 = set(R1[R1.q < 0.05].term)
    s2 = set(R2[R2.q < 0.05].term)
    only1 = s1 - s2
    print(f"\n[cmp]  {len(s1)} terms significant unmatched, {len(s2)} "
          f"length-matched, {len(s1 & s2)} shared", file=sys.stderr)
    if only1:
        nm = [name.get(t, t) for t in list(only1)[:8]]
        print(f"       lost to length matching: {'; '.join(nm)}",
              file=sys.stderr)
    W("length_artefacts.tsv", pd.DataFrame(dict(
        term=sorted(only1), name=[name.get(t, t) for t in sorted(only1)])))

    W("plot_burden.tsv", G[["gene", "span_kb", "n_fp", "n_real", "fp_per_kb",
                            "len_bin"]].assign(
        is_foreground=G.gene.isin(fg_set)))
    top = (R2[R2.q < 0.05].sort_values("q").groupby("namespace")
             .head(8).copy())
    W("plot_terms.tsv", top)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_genes=len(G), n_foreground=len(fg),
                       n_matched=len(matched),
                       rho_span_count=float(rho), rho_span_rate=float(rho2),
                       n_sig_unmatched=len(s1), n_sig_matched=len(s2),
                       top_frac=a.top_frac), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead the length-matched table. Terms present only in the "
          "unmatched one are\nexplained by gene length and should be reported "
          "as such rather than dropped.", file=sys.stderr)


if __name__ == "__main__":
    main()