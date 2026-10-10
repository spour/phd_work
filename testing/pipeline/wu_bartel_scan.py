#!/usr/bin/env python3
# the wu & bartel (cell 2017) analysis
import argparse, os, sys, json, math, random
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import RNA
except ImportError:
    sys.exit("ViennaRNA not importable; pip install ViennaRNA")
try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RC = str.maketrans("ACGTN", "TGCAN")
PAS_CANON = ["AATAAA", "ATTAAA"]
PAS_VARIANT = ["AGTAAA", "TATAAA", "CATAAA", "AATACA", "GATAAA", "AATATA",
               "AATAGA", "ACTAAA", "AATGAA", "ATTATA", "ATTACA", "AATAAT",
               "TTTAAA", "AATAAG", "CTTAAA", "AATAAC", "GTTAAA", "ATTAGA",
               "AACAAA", "ATTGAA", "AAGAAA", "ATTAAC"]


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


def dinuc_shuffle(s, rng):
    n = len(s)
    if n < 4:
        return s
    edges = defaultdict(list)
    for i in range(n - 1):
        edges[s[i]].append(s[i + 1])
    last = s[-1]
    for _ in range(25):
        lastedge, ok = {}, True
        for v in edges:
            if v == last:
                continue
            if not edges[v]:
                ok = False
                break
            lastedge[v] = rng.choice(edges[v])
        if not ok:
            return s
        good = True
        for v in lastedge:
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in lastedge:
                    good = False
                    break
                seen.add(cur)
                cur = lastedge[cur]
            if not good:
                break
        if good:
            break
    else:
        return s
    rem = {v: list(e) for v, e in edges.items()}
    for v, e in lastedge.items():
        rem[v].remove(e)
    for v in rem:
        rng.shuffle(rem[v])
    for v, e in lastedge.items():
        rem[v].append(e)
    out, cur = [s[0]], s[0]
    for _ in range(n - 1):
        if not rem.get(cur):
            break
        nxt = rem[cur].pop(0)
        out.append(nxt)
        cur = nxt
    return "".join(out) if len(out) == n else s


def efe(rna, md):
    fc = RNA.fold_compound(rna, md)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    _, g = fc.pf()
    return float(g), float(mfe)


def pair_probs(rna, md):
    fc = RNA.fold_compound(rna, md)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = fc.bpp()
    n = len(rna)
    p = np.zeros(n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            v = bpp[i][j]
            if v:
                p[i] += v
                p[j] += v
    return np.clip(p[1:], 0, 1)


def find_pas(seq, centre, maxdist, exclude_ambiguous=True):
    best = None
    lo = max(0, centre - maxdist)
    for h in PAS_CANON:
        i = seq.find(h, lo, centre)
        while i != -1:
            d = centre - (i + 6)
            if best is None or d < best[1]:
                best = (i, d)
            i = seq.find(h, i + 1, centre)
    if best is None:
        return None
    if exclude_ambiguous:
        for h in PAS_VARIANT:
            j = seq.find(h, max(0, centre - 30), centre - 10)
            if j != -1:
                return None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--pad", type=int, default=200)
    ap.add_argument("--max_pas_dist", type=int, default=100,
                    help="how far upstream to look for the PAS; Wu and Bartel "
                         "used the last 100 nt")
    ap.add_argument("--min_interval", type=int, default=8,
                    help="intervals shorter than this cannot fold")
    ap.add_argument("--n_shuffle", type=int, default=100,
                    help="dinucleotide shuffles per site for the empirical "
                         "p-value. Wu and Bartel used 1,000-10,000; 100 gives "
                         "a resolution of 0.01, adequate for a group mean")
    ap.add_argument("--max_ccpa", type=int, default=6)
    ap.add_argument("--heatmap_span", type=int, default=100,
                    help="positions upstream of the site for the pairing "
                         "heatmap, their Figure 2A")
    ap.add_argument("--heatmap_n", type=int, default=1500,
                    help="sites per class sampled for the heatmap")
    ap.add_argument("--temperature", type=float, default=37.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="wb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(a.seed)
    pyrng = random.Random(a.seed)
    md = RNA.md()
    md.temperature = a.temperature

    print(f"[cfg]  {a.temperature} C, {a.n_shuffle} dinucleotide shuffles per "
          f"site, PAS searched to -{a.max_pas_dist}", file=sys.stderr)

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

    gen = Genome(a.genome)
    PAD = a.pad
    rows = []
    n_nopas = n_ambig = n_short = n_bad = 0

    for i, r in enumerate(d.itertuples()):
        if i and i % 2000 == 0:
            print(f"[run]  {i:,}/{len(d):,}...", file=sys.stderr)
        dna = oriented(gen, r.chrom, r.pos, r.strand, PAD, PAD)
        if len(dna) != 2 * PAD or "N" in dna[PAD - a.max_pas_dist:PAD + 50]:
            n_bad += 1
            continue
        c = PAD
        hit = find_pas(dna, c, a.max_pas_dist, exclude_ambiguous=True)
        if hit is None:
            if find_pas(dna, c, a.max_pas_dist, exclude_ambiguous=False) is None:
                n_nopas += 1
            else:
                n_ambig += 1
            continue
        idx, dist = hit
        if dist < a.min_interval:
            n_short += 1
            continue

        interval = dna[idx + 6:c].replace("T", "U")
        g_obs, mfe_obs = efe(interval, md)

        u0 = max(0, idx - len(interval))
        upstream = dna[u0:idx].replace("T", "U")
        g_up = efe(upstream, md)[0] if len(upstream) == len(interval) else np.nan

        gs, gs_up = [], []
        for _ in range(a.n_shuffle):
            gs.append(efe(dinuc_shuffle(interval, pyrng), md)[0])
            if len(upstream) == len(interval):
                gs_up.append(efe(dinuc_shuffle(upstream, pyrng), md)[0])
        gs = np.array(gs, float)
        p_emp = float((1 + (gs <= g_obs).sum()) / (len(gs) + 1))
        z = float((g_obs - gs.mean()) / gs.std(ddof=1)) if gs.std(ddof=1) > 0 else np.nan

        rec = dict(gene=r.gene, chrom=r.chrom, pos=r.pos,
                   is_real=bool(r.is_real), pas_dist=int(dist),
                   interval_len=len(interval),
                   dG=g_obs, mfe=mfe_obs,
                   dG_shuf_mean=float(gs.mean()),
                   dG_minus_shuf=float(g_obs - gs.mean()),
                   z_vs_shuf=z, p_folded=p_emp,
                   folded_sig=float(p_emp < 0.05),
                   gc=float((interval.count("G") + interval.count("C")) /
                            max(len(interval), 1)))
        if len(gs_up):
            gsu = np.array(gs_up, float)
            rec["dG_up"] = g_up
            rec["dG_up_minus_shuf"] = float(g_up - gsu.mean())
            rec["p_folded_up"] = float((1 + (gsu <= g_up).sum()) / (len(gsu) + 1))
        rows.append(rec)

    print(f"[flt]  dropped: {n_nopas:,} no canonical PAS, {n_ambig:,} "
          f"ambiguous PAS, {n_short:,} interval < {a.min_interval} nt, "
          f"{n_bad:,} bad sequence", file=sys.stderr)
    P = pd.DataFrame(rows)
    W("per_site.tsv", P)
    print(f"[seq]  {len(P):,} sites analysed: "
          f"{int(P.is_real.sum()):,} real, {int((~P.is_real).sum()):,} fp",
          file=sys.stderr)

    # distance distribution
    print("\n[dist] PAS-to-site distance", file=sys.stderr)
    for lab, m in (("real 3' end", P.is_real), ("false positive", ~P.is_real)):
        s = P[m]
        print(f"       {lab:<16} median {s.pas_dist.median():.0f} nt, "
              f"{100*(s.pas_dist > 27).mean():5.1f}% beyond 27 nt "
              f"(Wu and Bartel: 13% of RefSeq mRNAs), n {len(s):,}",
              file=sys.stderr)
    dd = []
    for lab, m in (("real", P.is_real), ("fp", ~P.is_real)):
        h, e = np.histogram(P.loc[m, "pas_dist"], bins=np.arange(0, 102, 1))
        for x, y in zip(e[:-1], h):
            dd.append(dict(cls=lab, dist=int(x), n=int(y)))
    W("distance_distribution.tsv", pd.DataFrame(dd))

    # their figure 2b and 2c, by distance group
    print("\n[fold] folding excess over dinucleotide shuffles, by "
          "PAS-to-site distance", file=sys.stderr)
    print("       negative dG minus shuffle = more stably folded than "
          "composition alone predicts", file=sys.stderr)
    grp = []
    for lo in range(a.min_interval, 80):
        hi = lo + 10
        for lab, m in (("real", P.is_real), ("fp", ~P.is_real)):
            s = P[m & (P.pas_dist >= lo) & (P.pas_dist < hi)]
            if len(s) < 25:
                continue
            row = dict(cls=lab, dist_lo=lo, dist_hi=hi, n=len(s),
                       mean_dG=round(float(s.dG.mean()), 4),
                       excess=round(float(s.dG_minus_shuf.mean()), 4),
                       excess_se=round(float(s.dG_minus_shuf.std(ddof=1) /
                                             math.sqrt(len(s))), 4),
                       frac_sig=round(float(s.folded_sig.mean()), 4),
                       mean_z=round(float(s.z_vs_shuf.mean()), 4))
            if "dG_up_minus_shuf" in s.columns:
                u = s.dG_up_minus_shuf.dropna()
                row["excess_upstream"] = (round(float(u.mean()), 4)
                                          if len(u) else np.nan)
                row["frac_sig_upstream"] = (round(float((s.p_folded_up < 0.05).mean()), 4)
                                            if "p_folded_up" in s else np.nan)
            grp.append(row)
    G = pd.DataFrame(grp)
    W("folding_by_distance.tsv", G)

    for lab in ("real", "fp"):
        s = G[G.cls == lab]
        if not len(s):
            continue
        print(f"\n       {lab}:", file=sys.stderr)
        print(f"       {'distance':>10}{'n':>7}{'excess dG':>12}"
              f"{'up ctrl':>10}{'% p<0.05':>10}{'% up sig':>10}",
              file=sys.stderr)
        for r_ in s[s.dist_lo % 8 == 0].itertuples():
            up = getattr(r_, "excess_upstream", np.nan)
            us = getattr(r_, "frac_sig_upstream", np.nan)
            print(f"       {str(r_.dist_lo)+'-'+str(r_.dist_hi):>10}"
                  f"{r_.n:>7,}{r_.excess:>+12.3f}{up:>+10.3f}"
                  f"{100*r_.frac_sig:>9.1f}%{100*us:>9.1f}%", file=sys.stderr)

    # the conditional test
    print("\n[cond] real versus false positive, inside distance strata",
          file=sys.stderr)
    STRATA = [(a.min_interval, 17, "short, below the optimum"),
              (17, 27, "optimal, structure disfavoured"),
              (27, 45, "distal, structure predicted to rescue"),
              (45, 100, "very distal")]
    crow = []
    for lo, hi, note in STRATA:
        rs = P[P.is_real & (P.pas_dist >= lo) & (P.pas_dist < hi)]
        fs = P[(~P.is_real) & (P.pas_dist >= lo) & (P.pas_dist < hi)]
        if len(rs) < 20 or len(fs) < 20:
            continue
        x = rs.dG_minus_shuf.dropna(); y = fs.dG_minus_shuf.dropna()
        sd = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        dd_ = (x.mean() - y.mean()) / sd if sd > 0 else np.nan
        p = np.nan
        if HAVE_SCIPY:
            p = float(sps.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        crow.append(dict(stratum=f"{lo}-{hi}", note=note,
                         n_real=len(rs), n_fp=len(fs),
                         real_excess=round(float(x.mean()), 4),
                         fp_excess=round(float(y.mean()), 4),
                         d=round(float(dd_), 4), p=p,
                         real_frac_sig=round(float(rs.folded_sig.mean()), 4),
                         fp_frac_sig=round(float(fs.folded_sig.mean()), 4)))
        print(f"       {lo:>3}-{hi:<3} {note:<36} real {x.mean():+.3f}  "
              f"fp {y.mean():+.3f}  d {dd_:+.3f}  p {p:.2e}", file=sys.stderr)
    W("conditional.tsv", pd.DataFrame(crow))

    print("\n[pair] paired within gene, distal sites only (> 27 nt)",
          file=sys.stderr)
    dp = P[P.pas_dist > 27]
    rb = dp[dp.is_real].drop_duplicates("gene").set_index("gene")
    fb = dp[~dp.is_real].groupby("gene")[["dG_minus_shuf", "dG", "z_vs_shuf",
                                          "pas_dist"]].mean()
    com = rb.index.intersection(fb.index)
    if len(com) >= 20:
        v = (rb.loc[com, "dG_minus_shuf"].astype(float) -
             fb.loc[com, "dG_minus_shuf"].astype(float)).dropna()
        sd = v.std(ddof=1)
        p = np.nan
        if HAVE_SCIPY and sd > 0:
            try:
                p = float(sps.wilcoxon(v, zero_method="zsplit").pvalue)
            except Exception:
                pass
        print(f"       n {len(v):,} genes with a distal PAS at both a real end "
              f"and a false positive", file=sys.stderr)
        print(f"       real minus fp folding excess: {v.mean():+.4f} "
              f"(dz {v.mean()/sd:+.3f}, p {p:.2e})", file=sys.stderr)
        W("paired_distal.tsv",
          pd.DataFrame(dict(gene=v.index, diff=v.to_numpy())))
    else:
        print(f"       only {len(com)} genes qualify; too few to pair",
              file=sys.stderr)

    # their figure 2a: pairing heatmap
    print("\n[heat] pairing-probability heatmap, their Figure 2A",
          file=sys.stderr)
    HS = a.heatmap_span
    hrows = []
    for lab, m in (("real", P.is_real), ("fp", ~P.is_real)):
        s = P[m]
        if len(s) > a.heatmap_n:
            s = s.sample(a.heatmap_n, random_state=0)
        acc = defaultdict(list)
        for k, r_ in enumerate(s.itertuples()):
            if k and k % 500 == 0:
                print(f"       {lab} {k:,}/{len(s):,}", file=sys.stderr)
            row = d[(d.gene == r_.gene) & (d.pos == r_.pos)]
            if not len(row):
                continue
            st = row.strand.iloc[0]
            dna = oriented(gen, r_.chrom, r_.pos, st, HS, 0)
            if len(dna) != HS or "N" in dna:
                continue
            pp = pair_probs(dna.replace("T", "U"), md)
            acc[int(r_.pas_dist)].append(pp)
        for dist, mats in acc.items():
            if len(mats) < 10:
                continue
            mean = np.mean(np.vstack(mats), axis=0)
            for j, val in enumerate(mean):
                hrows.append(dict(cls=lab, pas_dist=dist,
                                  position=j - HS, paired=round(float(val), 5),
                                  n=len(mats)))
    W("pairing_heatmap.tsv", pd.DataFrame(hrows))
    print(f"       wrote {len(hrows):,} cells", file=sys.stderr)

    print("\n[sum]  overall, ignoring distance (the comparison that fails)",
          file=sys.stderr)
    for lab, m in (("real 3' end", P.is_real), ("false positive", ~P.is_real)):
        s = P[m]
        print(f"       {lab:<16} excess dG {s.dG_minus_shuf.mean():+.3f}, "
              f"{100*s.folded_sig.mean():.1f}% with p < 0.05",
              file=sys.stderr)
    print("       Wu and Bartel found ~13-20% of endogenous PAS-downstream "
          "regions reach\n       p < 0.05, against ~5% expected, and their "
          "Figure 7 makes the point that\n       structures no more stable "
          "than random can still be functional -- so a\n       low fraction "
          "here is not evidence against the mechanism.", file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_sites=len(P), n_real=int(P.is_real.sum()),
                       n_fp=int((~P.is_real).sum()),
                       n_genes=int(P.gene.nunique()),
                       dropped_no_pas=n_nopas, dropped_ambiguous=n_ambig,
                       dropped_short=n_short,
                       n_shuffle=a.n_shuffle), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead [cond]. The mechanism predicts a difference only in the "
          "distal stratum,\nand possibly the opposite sign at 17-27 nt. A flat "
          "result across all strata\nmeans the mechanism does not distinguish "
          "these classes.", file=sys.stderr)


if __name__ == "__main__":
    main()