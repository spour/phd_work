#!/usr/bin/env python3
# what distinguishes real 3' ends from model false positives
import argparse, os, sys, json, math
from collections import defaultdict

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")

# maxent
try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5, load_matrix3
    _M5, _M3 = load_matrix5(), load_matrix3()
    HAVE_MAXENT = True
except Exception:
    HAVE_MAXENT = False

PWM5 = [
    {"A": .33, "C": .36, "G": .18, "T": .13},
    {"A": .61, "C": .13, "G": .13, "T": .13},
    {"A": .10, "C": .03, "G": .80, "T": .07},
    {"A": .00, "C": .00, "G": 1.0, "T": .00},
    {"A": .00, "C": .00, "G": .00, "T": 1.0},
    {"A": .53, "C": .03, "G": .42, "T": .02},
    {"A": .71, "C": .08, "G": .12, "T": .09},
    {"A": .07, "C": .06, "G": .81, "T": .06},
    {"A": .16, "C": .17, "G": .19, "T": .48},
]
PWM3 = ([{"A": .10, "C": .32, "G": .06, "T": .52}] * 18 +
        [{"A": .24, "C": .25, "G": .13, "T": .38},
         {"A": .06, "C": .17, "G": .04, "T": .73},
         {"A": 1.0, "C": .00, "G": .00, "T": .00},
         {"A": .00, "C": .00, "G": 1.0, "T": .00},
         {"A": .25, "C": .14, "G": .48, "T": .13}])
BG = {"A": .27, "C": .23, "G": .23, "T": .27}


def pwm_score(seq, pwm):
    if len(seq) != len(pwm):
        return np.nan
    s = 0.0
    for ch, col in zip(seq, pwm):
        p = col.get(ch)
        if p is None:
            return np.nan
        s += math.log2(max(p, 1e-3) / BG[ch])
    return s


def score5(seq):
    if HAVE_MAXENT:
        try:
            return float(maxent.score5(seq, matrix=_M5))
        except Exception:
            return np.nan
    return pwm_score(seq, PWM5)


def score3(seq):
    if HAVE_MAXENT:
        try:
            return float(maxent.score3(seq, matrix=_M3))
        except Exception:
            return np.nan
    return pwm_score(seq, PWM3)


U1_TEMPLATE = "ACUUACCUG"
NN = {
    "AA": -0.93, "AU": -1.10, "AC": -2.24, "AG": -2.08,
    "UA": -1.33, "UU": -0.93, "UC": -2.35, "UG": -2.11,
    "CA": -2.11, "CU": -2.08, "CC": -3.26, "CG": -2.36,
    "GA": -2.35, "GU": -2.24, "GC": -3.42, "GG": -3.26,
}
PAIR = {"A": "U", "U": "A", "G": "C", "C": "G"}


def u1_energy(seq9):
    s = seq9.replace("T", "U")
    if len(s) != 9:
        return np.nan
    paired = [PAIR.get(a) == b for a, b in zip(s, U1_TEMPLATE)]
    e = 0.0
    for i in range(8):
        if paired[i] and paired[i + 1]:
            e += NN.get(s[i:i + 2], 0.0)
    return e


# sequence
def gc(s):
    n = sum(s.count(b) for b in "ACGT")
    return (s.count("G") + s.count("C")) / n if n else np.nan


def gc_skew(s):
    g, c = s.count("G"), s.count("C")
    return (g - c) / (g + c) if (g + c) else np.nan


def at_skew(s):
    a, t = s.count("A"), s.count("T")
    return (a - t) / (a + t) if (a + t) else np.nan


def cpg_oe(s):
    n = len(s)
    if n < 2:
        return np.nan
    c, g = s.count("C"), s.count("G")
    obs = s.count("CG")
    exp = c * g / n
    return obs / exp if exp > 0 else np.nan


def shannon_k(s, k=3):
    if len(s) < k:
        return np.nan
    cnt = defaultdict(int)
    tot = 0
    for i in range(len(s) - k + 1):
        w = s[i:i + k]
        if "N" in w:
            continue
        cnt[w] += 1
        tot += 1
    if not tot:
        return np.nan
    return -sum((v / tot) * math.log2(v / tot) for v in cnt.values())


def longest_run(s, chars):
    best = run = 0
    for ch in s:
        run = run + 1 if ch in chars else 0
        best = max(best, run)
    return best


def ww_ss_periodicity(s, period=10):
    n = len(s)
    if n < 3 * period:
        return np.nan
    v = np.zeros(n - 1)
    for i in range(n - 1):
        d = s[i:i + 2]
        if d in ("AA", "TT", "AT", "TA"):
            v[i] = 1.0
        elif d in ("GG", "CC", "GC", "CG"):
            v[i] = -1.0
    v = v - v.mean()
    if not np.any(v):
        return np.nan
    f = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(len(v))
    j = int(np.argmin(np.abs(freqs - 1.0 / period)))
    return float(np.abs(f[j]) / len(v))


G4_RE = None


def g4_density(s):
    global G4_RE
    if G4_RE is None:
        import re
        G4_RE = re.compile(r"G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}")
    return len(G4_RE.findall(s)) / max(len(s), 1) * 1000


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


# splice search
def splice_landscape(seq, centre, up, dn, search_up, search_dn, topn=3):
    out = {}
    i0 = max(0, centre - search_up)
    i1 = min(len(seq), centre + search_dn)

    d5, s5, e5 = [], [], []
    for i in range(i0, i1 - 9):
        if seq[i + 3:i + 5] != "GT":
            continue
        w = seq[i:i + 9]
        if "N" in w:
            continue
        sc = score5(w)
        if not np.isfinite(sc):
            continue
        d5.append(i + 3 - centre)
        s5.append(sc)
        e5.append(u1_energy(w))

    d3, s3 = [], []
    for i in range(i0, i1 - 23):
        if seq[i + 18:i + 20] != "AG":
            continue
        w = seq[i:i + 23]
        if "N" in w:
            continue
        sc = score3(w)
        if not np.isfinite(sc):
            continue
        d3.append(i + 20 - centre)
        s3.append(sc)

    def summarize(dists, scores, tag, energies=None):
        r = {}
        if not dists:
            r[tag + " n"] = 0
            for k in ("best score", "best dist", "nearest score",
                      "nearest dist", "mean top3"):
                r[tag + " " + k] = np.nan
            if energies is not None:
                r[tag + " best dG"] = np.nan
            return r
        d = np.asarray(dists, float)
        s = np.asarray(scores, float)
        r[tag + " n"] = len(d)
        j = int(np.argmax(s))
        r[tag + " best score"] = float(s[j])
        r[tag + " best dist"] = float(d[j])
        k = int(np.argmin(np.abs(d)))
        r[tag + " nearest score"] = float(s[k])
        r[tag + " nearest dist"] = float(d[k])
        r[tag + " mean top3"] = float(np.mean(np.sort(s)[::-1][:topn]))
        if energies is not None:
            e = np.asarray(energies, float)
            r[tag + " best dG"] = float(np.nanmin(e))
            r[tag + " dG at best score"] = float(e[j])
        far = np.abs(d) > 500
        r[tag + " n beyond500"] = int(far.sum())
        r[tag + " best score beyond500"] = float(s[far].max()) if far.any() else np.nan
        near = ~far
        r[tag + " best score within500"] = float(s[near].max()) if near.any() else np.nan
        return r

    out.update(summarize(d5, s5, "5'ss", e5))
    out.update(summarize(d3, s3, "3'ss"))

    for tag, dd, ss in (("5'ss", d5, s5), ("3'ss", d3, s3)):
        d = np.asarray(dd, float); s = np.asarray(ss, float)
        for side, m in (("up", d < 0), ("dn", d > 0)):
            if len(d) and m.any():
                out[tag + " " + side + " best"] = float(s[m].max())
                out[tag + " " + side + " nearest dist"] = float(
                    np.abs(d[m]).min())
            else:
                out[tag + " " + side + " best"] = np.nan
                out[tag + " " + side + " nearest dist"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--up", type=int, default=5000,
                    help="sequence fetched upstream; ten times the CNN window")
    ap.add_argument("--dn", type=int, default=3000)
    ap.add_argument("--search_up", type=int, default=3000,
                    help="splice-site search range upstream of the site")
    ap.add_argument("--search_dn", type=int, default=2000)
    ap.add_argument("--bin", type=int, default=100,
                    help="bin width for the composition profiles")
    ap.add_argument("--max_ccpa", type=int, default=8,
                    help="cap false positives per gene. Genes range from 1 to "
                         "~900; without a cap a handful dominate every pooled "
                         "statistic and the paired means are noisy")
    ap.add_argument("--out", default="mech")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    UP, DN, L = a.up, a.dn, a.up + a.dn

    print("[cfg]  5'/3' splice scoring: " +
          ("MaxEnt (maxentpy)" if HAVE_MAXENT else
           "PWM log-odds (maxentpy not installed; weaker but monotone)"),
          file=sys.stderr)

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    pos = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[pos].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})

    if a.max_ccpa > 0:
        keep = []
        rng = np.random.default_rng(0)
        for g, sub in d.groupby("gene", sort=False):
            r = sub[sub.is_real]
            c = sub[~sub.is_real]
            if len(c) > a.max_ccpa:
                c = c.sample(a.max_ccpa, random_state=0)
            keep.append(pd.concat([r, c]))
        n0 = len(d)
        d = pd.concat(keep, ignore_index=True)
        print(f"[cap]  {n0:,} -> {len(d):,} sites "
              f"({a.max_ccpa} false positives per gene)", file=sys.stderr)

    print(f"[set]  {int(d.is_real.sum()):,} real, "
          f"{int((~d.is_real).sum()):,} false positives, "
          f"{d.gene.nunique():,} genes", file=sys.stderr)
    print(f"[win]  sequence -{UP} to +{DN}; splice search -{a.search_up} to "
          f"+{a.search_dn}; the CNN saw 500 nt", file=sys.stderr)

    gen = Genome(a.genome)

    edges = list(range(-UP, DN + 1, a.bin))
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    rows, prof_rows = [], []
    n_bad = 0
    for gi, (g, sub) in enumerate(d.groupby("gene", sort=False)):
        if gi and gi % 250 == 0:
            print(f"[run]  {gi:,} genes...", file=sys.stderr)
        for r in sub.itertuples():
            seq = oriented(gen, r.chrom, r.pos, r.strand, UP, DN)
            if len(seq) != L:
                n_bad += 1
                continue
            rec = dict(gene=g, chrom=r.chrom, pos=r.pos, strand=r.strand,
                       is_real=bool(r.is_real))

            # composition, per bin, for the profile
            for b0, b1 in bins:
                s = seq[b0 + UP:b1 + UP]
                if len(s) < a.bin // 2:
                    continue
                prof_rows.append(dict(gene=g, is_real=bool(r.is_real),
                                      pos=(b0 + b1) // 2,
                                      gc=gc(s), gc_skew=gc_skew(s),
                                      at_skew=at_skew(s), cpg=cpg_oe(s),
                                      ent3=shannon_k(s, 3)))

            for wn, w0, w1 in (("core", -250, 250),
                               ("prox up", -1000, -250),
                               ("prox dn", 250, 1000),
                               ("dist up", -3000, -1000),
                               ("dist dn", 1000, 3000),
                               ("far up", -5000, -3000)):
                s = seq[w0 + UP:w1 + UP]
                if len(s) < 100:
                    continue
                rec[wn + " GC"] = gc(s)
                rec[wn + " GC skew"] = gc_skew(s)
                rec[wn + " AT skew"] = at_skew(s)
                rec[wn + " CpG o/e"] = cpg_oe(s)
                rec[wn + " entropy3"] = shannon_k(s, 3)
                rec[wn + " nucleosome WW/SS"] = ww_ss_periodicity(s)
                rec[wn + " G4 per kb"] = g4_density(s)
                rec[wn + " maxrun A"] = longest_run(s, "A")
                rec[wn + " maxrun T"] = longest_run(s, "T")
                rec[wn + " maxrun AT"] = longest_run(s, "AT")

            rec.update(splice_landscape(seq, UP, UP, DN,
                                        a.search_up, a.search_dn))
            rows.append(rec)

    if n_bad:
        print(f"[seq]  {n_bad:,} sites dropped: window off the contig",
              file=sys.stderr)
    PER = pd.DataFrame(rows)
    W("per_site.tsv", PER)
    print(f"[seq]  {len(PER):,} sites measured", file=sys.stderr)

    PR = pd.DataFrame(prof_rows)
    prof = []
    for metric in ("gc", "gc_skew", "at_skew", "cpg", "ent3"):
        for p_, sp in PR.groupby("pos"):
            diffs = []
            for g, gg in sp.groupby("gene", sort=False):
                rv = gg.loc[gg.is_real, metric]
                cv = gg.loc[~gg.is_real, metric]
                if len(rv) != 1 or not len(cv):
                    continue
                v = float(rv.iloc[0]) - float(cv.mean())
                if np.isfinite(v):
                    diffs.append(v)
            if len(diffs) < 50:
                continue
            v = np.asarray(diffs)
            se = v.std(ddof=1) / np.sqrt(len(v))
            prof.append(dict(metric=metric, pos=int(p_),
                             delta=round(float(v.mean()), 6),
                             lo=round(float(v.mean() - 1.96 * se), 6),
                             hi=round(float(v.mean() + 1.96 * se), 6),
                             n_genes=len(v)))
    PF = pd.DataFrame(prof)
    W("profile_paired.tsv", PF)

    pooled = (PR.groupby(["pos", "is_real"])
                [["gc", "gc_skew", "at_skew", "cpg", "ent3"]]
                .mean().reset_index())
    pooled["cls"] = np.where(pooled.is_real, "real 3' end", "false positive")
    W("profile_pooled.tsv", pooled.drop(columns="is_real"))

    if len(PF):
        g_ = PF[PF.metric == "gc"]
        print(f"\n[gc]   paired GC: minimum {g_.delta.min():+.4f} at "
              f"{int(g_.loc[g_.delta.idxmin(),'pos'])} nt; "
              f"beyond -1 kb mean {g_[g_.pos < -1000].delta.mean():+.4f}",
              file=sys.stderr)

    cols = [c for c in PER.columns
            if c not in ("gene", "chrom", "pos", "strand", "is_real")]
    pr = []
    for c in cols:
        vals = []
        for g, sub in PER.groupby("gene", sort=False):
            rv = sub.loc[sub.is_real, c]
            cv = sub.loc[~sub.is_real, c]
            if len(rv) != 1 or not len(cv):
                continue
            rr = pd.to_numeric(rv.iloc[0], errors="coerce")
            cc = pd.to_numeric(cv, errors="coerce").mean()
            if np.isfinite(rr) and np.isfinite(cc):
                vals.append(float(rr - cc))
        if len(vals) < 100:
            continue
        v = np.asarray(vals)
        sd = v.std(ddof=1)
        se = sd / np.sqrt(len(v))
        pr.append(dict(feature=c, n_genes=len(v),
                       paired_diff=round(float(v.mean()), 5),
                       lo=round(float(v.mean() - 1.96 * se), 5),
                       hi=round(float(v.mean() + 1.96 * se), 5),
                       frac_positive=round(float((v > 0).mean()), 4),
                       cohens_dz=round(float(v.mean() / sd), 4)
                       if sd > 0 else np.nan,
                       outside_cnn=not c.startswith("core")))
    PRD = pd.DataFrame(pr).sort_values("cohens_dz", key=abs, ascending=False)
    W("paired_effects.tsv", PRD)

    print("\n[out]  strongest paired differences OUTSIDE the 500 nt the CNN "
          "saw:", file=sys.stderr)
    for r_ in PRD[PRD.outside_cnn].head(20).itertuples():
        sig = "" if (r_.lo <= 0 <= r_.hi) else "  *"
        print(f"       {r_.feature:<34} {r_.paired_diff:+9.4f}  "
              f"dz {r_.cohens_dz:+.3f}  {100*r_.frac_positive:5.1f}% of "
              f"genes{sig}", file=sys.stderr)
    print("\n[out]  inside the CNN window, for contrast (confounded by "
          "construction):", file=sys.stderr)
    for r_ in PRD[~PRD.outside_cnn].head(8).itertuples():
        print(f"       {r_.feature:<34} {r_.paired_diff:+9.4f}  "
              f"dz {r_.cohens_dz:+.3f}", file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_real=int(PER.is_real.sum()),
                       n_fp=int((~PER.is_real).sum()),
                       n_genes=int(PER.gene.nunique()),
                       up=UP, dn=DN, bin=a.bin,
                       search_up=a.search_up, search_dn=a.search_dn,
                       maxent=HAVE_MAXENT, max_ccpa=a.max_ccpa), fh, indent=2)

    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead paired_effects.tsv filtered to outside_cnn=True. Those are "
          "the features the models could not have selected on, so a difference "
          "there is a candidate mechanism rather than a restatement of how the "
          "false positives were chosen.", file=sys.stderr)


if __name__ == "__main__":
    main()