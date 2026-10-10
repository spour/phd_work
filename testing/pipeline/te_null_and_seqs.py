#!/usr/bin/env python3
# are competing sites in transposable element sequence more often than random positions
import argparse, os, sys, gzip, json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

NOT_TE = {"Simple_repeat", "Low_complexity", "Satellite", "rRNA", "scRNA",
          "snRNA", "srpRNA", "tRNA", "RNA", "Unknown", "Other"}


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
    blank = dict(rep=0, is_te=0, fam="none", sense=np.nan, frac=np.nan,
                 div=np.nan)
    e = rm.get(chrom)
    if e is None or not len(e["start"]):
        return blank
    j = int(np.searchsorted(e["start"], pos, side="right") - 1)
    if j < 0 or pos > e["rows"][j][1]:
        return blank
    s0, s1, estr, ecls, efam, ediv = e["rows"][j]
    ln = max(s1 - s0, 1)
    fr = (pos - s0) / ln
    if estr == "-":
        fr = 1.0 - fr
    return dict(rep=1, is_te=0 if ecls in NOT_TE else 1, fam=efam,
                sense=float(estr == strand), frac=float(fr), div=ediv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--pad", type=int, default=5000)
    ap.add_argument("--n_null", type=int, default=5,
                    help="random positions drawn per competing site")
    ap.add_argument("--exclude", type=int, default=100,
                    help="a random position is rejected if it lies within this "
                         "many nt of any known site")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tenull")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    POS = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[POS].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    canon = d[d.is_real].groupby("gene").pos.first().to_dict()
    span = (d.groupby(["gene", "chrom", "strand"], as_index=False)
              .agg(lo=("pos", "min"), hi=("pos", "max")))
    span["lo"] = (span.lo - a.pad).clip(lower=0)
    span["hi"] = span.hi + a.pad
    sp = {r.gene: (r.chrom, r.lo, r.hi, r.strand) for r in span.itertuples()}

    taken = defaultdict(set)
    for r in d.itertuples():
        for o in range(-a.exclude, a.exclude + 1):
            taken[r.chrom].add(r.pos + o)

    rm = load_rmsk(a.rmsk, set(d.chrom.unique()))

    rows = []
    fp = d[~d.is_real]
    print(f"[site] {len(fp):,} competing sites", file=sys.stderr)
    for r in fp.itertuples():
        rec = repeat_at(rm, r.chrom, r.pos, r.strand)
        rec.update(group="competing site", gene=r.gene, chrom=r.chrom,
                   pos=r.pos, dist_canon=abs(r.pos - canon.get(r.gene, r.pos)))
        rows.append(rec)

    print(f"[null] drawing {a.n_null} matched positions per site",
          file=sys.stderr)
    n_ok = n_fail = 0
    for r in fp.itertuples():
        g = sp.get(r.gene)
        cp = canon.get(r.gene)
        if g is None or cp is None:
            continue
        chrom, lo, hi, strand = g
        off = abs(r.pos - cp)
        if off < 200:
            continue
        for _ in range(a.n_null):
            placed = False
            for _try in range(12):
                f = float(rng.uniform(0.5, 2.0))
                p = cp - int(off * f) if r.pos < cp else cp + int(off * f)
                if p < lo or p > hi or p in taken[chrom]:
                    continue
                rec = repeat_at(rm, chrom, p, strand)
                rec.update(group="matched random", gene=r.gene, chrom=chrom,
                           pos=p, dist_canon=abs(p - cp))
                rows.append(rec)
                placed = True
                n_ok += 1
                break
            if not placed:
                n_fail += 1
        if n_ok and n_ok % 50000 == 0:
            print(f"       {n_ok:,}", file=sys.stderr)

    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(a.out, "sites.tsv"), sep="\t", index=False)
    print(f"[null] {n_ok:,} placed, {n_fail:,} attempts abandoned",
          file=sys.stderr)

    C = M[M.group == "competing site"]
    B = M[M.group == "matched random"]
    print("\n" + "=" * 66, file=sys.stderr)
    print(f"       {'':<22}{'competing':>12}{'matched random':>17}",
          file=sys.stderr)
    print(f"       {'n':<22}{len(C):>12,}{len(B):>17,}", file=sys.stderr)
    for lab, col in (("in any repeat", "rep"), ("in a TE", "is_te")):
        print(f"       {lab:<22}{100*C[col].mean():>11.1f}%"
              f"{100*B[col].mean():>16.1f}%", file=sys.stderr)
    ts = ((C.is_te == 1) & (C.sense == 1)).mean()
    bs = ((B.is_te == 1) & (B.sense == 1)).mean()
    print(f"       {'TE, sense':<22}{100*ts:>11.1f}%{100*bs:>16.1f}%",
          file=sys.stderr)
    ta = ((C.is_te == 1) & (C.sense == 0)).mean()
    ba = ((B.is_te == 1) & (B.sense == 0)).mean()
    print(f"       {'TE, antisense':<22}{100*ta:>11.1f}%{100*ba:>16.1f}%",
          file=sys.stderr)
    odds, p = fisher_exact([[int((C.is_te == 1).sum()),
                             int((C.is_te == 0).sum())],
                            [int((B.is_te == 1).sum()),
                             int((B.is_te == 0).sum())]])
    print(f"\n       any TE: odds ratio {odds:.3f}, Fisher p = {p:.3g}",
          file=sys.stderr)
    print("=" * 66, file=sys.stderr)

    fams = sorted(set(C[C.is_te == 1].fam) | set(B[B.is_te == 1].fam))
    rows2 = []
    for f in fams:
        kc = int((C.fam == f).sum())
        kb = int((B.fam == f).sum())
        if kc + kb < 60:
            continue
        pc, pb = kc / len(C), kb / len(B)
        _, pv = fisher_exact([[kc, len(C) - kc], [kb, len(B) - kb]])
        ksc = int(((C.fam == f) & (C.sense == 1)).sum())
        ksb = int(((B.fam == f) & (B.sense == 1)).sum())
        rows2.append(dict(family=f, n_competing=kc, n_null=kb,
                          pct_competing=round(100 * pc, 3),
                          pct_null=round(100 * pb, 3),
                          ratio=round(pc / pb, 3) if pb > 0 else np.nan,
                          pct_competing_sense=round(100 * ksc / len(C), 3),
                          pct_null_sense=round(100 * ksb / len(B), 3),
                          sense_ratio=round((ksc / len(C)) / (ksb / len(B)), 3)
                          if ksb > 0 else np.nan, p=pv))
    F = pd.DataFrame(rows2)
    pv = F.p.to_numpy(float)
    o = np.argsort(pv); n = len(pv)
    adj = pv[o] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty(n); q[o] = np.clip(adj, 0, 1)
    F["q"] = q
    F = F.sort_values("n_competing", ascending=False)
    F.to_csv(os.path.join(a.out, "families.tsv"), sep="\t", index=False)

    print("\n[fam]  by family", file=sys.stderr)
    print(f"       {'family':<15}{'competing':>11}{'null':>9}{'ratio':>8}"
          f"{'sense ratio':>13}{'q':>10}", file=sys.stderr)
    for r in F.itertuples():
        star = " *" if r.q < 0.05 else "  "
        print(f"       {r.family:<15}{r.pct_competing:>10.2f}%"
              f"{r.pct_null:>8.2f}%{r.ratio:>8.2f}{r.sense_ratio:>13.2f}"
              f"{r.q:>10.1e}{star}", file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_competing=len(C), n_null=len(B),
                       te_competing=float(C.is_te.mean()),
                       te_null=float(B.is_te.mean()),
                       odds_ratio=float(odds), p=float(p)), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()