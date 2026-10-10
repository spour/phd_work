#!/usr/bin/env python3
# what do real 3' ends and their competitors look like?
import argparse, os, sys, json, itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")

PAS_STRONG = ["AATAAA", "ATTAAA"]
PAS_WEAK = ["TATAAA", "AGTAAA", "AAGAAA", "AATATA", "AATACA", "CATAAA",
            "GATAAA", "AATGAA", "TTTAAA", "ACTAAA", "AATAGA"]

U1_CORE = ["GTAAGT", "GTGAGT", "GTAAGC", "GTGAGC", "GTAAGA", "GTAAGG"]

MOTIFS = {
    "PAS canonical":      PAS_STRONG,
    "PAS variant":        PAS_WEAK,
    "U1 5'ss core":       U1_CORE,
    "CFIm UGUA":          ["TGTA"],
    "CstF GU-rich":       ["TGTGT", "GTGTG"],
    "U-rich (U5)":        ["TTTTT"],
    "CA dinucleotide":    ["CA"],
    "polypyrimidine":     ["TTTCT", "TCTTT", "CTTTC"],
}

SUBWIN = [
    ("far upstream",     -2000, -500),
    ("upstream",          -500, -100),
    ("PAS zone",          -100,  -10),
    ("cleavage",           -10,   10),
    ("downstream DSE",      10,  100),
    ("far downstream",     100, 1000),
]


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


def oriented_window(gen, chrom, pos, strand, up, dn):
    if strand == "-":
        s = gen.fetch(chrom, int(pos) - dn, int(pos) + up)
        if len(s) < up + dn:
            return ""
        return s.translate(RC)[::-1]
    s = gen.fetch(chrom, int(pos) - up, int(pos) + dn)
    return s if len(s) >= up + dn else ""


def motif_hits(seq, pats):
    out = []
    for p in pats:
        i = seq.find(p)
        while i != -1:
            out.append(i)
            i = seq.find(p, i + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--up", type=int, default=2000)
    ap.add_argument("--dn", type=int, default=1000)
    ap.add_argument("--gc_win", type=int, default=101,
                    help="sliding window for the GC profile, odd")
    ap.add_argument("--bin", type=int, default=10,
                    help="bin width for the profiles, in nt")
    ap.add_argument("--max_ccpa", type=int, default=0,
                    help="cap competitors per gene (0 = all). Genes range from "
                         "1 to ~900 competitors, so a few peak-dense genes "
                         "otherwise dominate the pooled competitor profile")
    ap.add_argument("--kmer_k", type=int, default=6)
    ap.add_argument("--kmer_top", type=int, default=40)
    ap.add_argument("--out", default="context")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)
    UP, DN, L = a.up, a.dn, a.up + a.dn

    # sites
    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    pos = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str),
                      "chrom": pk.chrom.astype(str),
                      "pos": pk[pos].astype(int),
                      "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)}).reset_index(drop=True)

    if a.max_ccpa > 0:
        keep = []
        for g, sub in d.groupby("gene", sort=False):
            r = sub[sub.is_real]
            c = sub[~sub.is_real]
            if len(c) > a.max_ccpa:
                c = c.sample(a.max_ccpa, random_state=0)
            keep.append(pd.concat([r, c]))
        n0 = len(d)
        d = pd.concat(keep, ignore_index=True)
        print(f"[cap]  competitors capped at {a.max_ccpa}/gene: "
              f"{n0:,} -> {len(d):,} sites", file=sys.stderr)

    print(f"[set]  {int(d.is_real.sum()):,} real 3' ends, "
          f"{int((~d.is_real).sum()):,} competitors, "
          f"{d.gene.nunique():,} genes", file=sys.stderr)
    print(f"[win]  -{UP} to +{DN} nt, transcript-oriented", file=sys.stderr)

    gen = Genome(a.genome)

    # one pass: fetch, accumulate
    xs = np.arange(-UP, DN)
    base_acc = {c: {b: np.zeros(L) for b in "ACGT"} for c in (True, False)}
    motif_acc = {c: {m: np.zeros(L) for m in MOTIFS} for c in (True, False)}
    n_seq = {True: 0, False: 0}

    rows = []
    pair_gc = []
    pair_motif = defaultdict(list)

    gc_half = a.gc_win // 2
    kmer_counts = {c: {w[0]: Counter() for w in SUBWIN} for c in (True, False)}
    dinuc_counts = {c: {w[0]: Counter() for w in SUBWIN} for c in (True, False)}

    n_bad = 0
    gene_iter = d.groupby("gene", sort=False)
    for gi, (g, sub) in enumerate(gene_iter):
        if gi and gi % 500 == 0:
            print(f"[run]  {gi:,} genes...", file=sys.stderr)

        gc_profiles = {True: [], False: []}
        mo_profiles = {True: defaultdict(list), False: defaultdict(list)}

        for r in sub.itertuples():
            seq = oriented_window(gen, r.chrom, r.pos, r.strand, UP, DN)
            if len(seq) != L:
                n_bad += 1
                continue
            cls = bool(r.is_real)
            n_seq[cls] += 1
            arr = np.frombuffer(seq.encode(), dtype=np.uint8)

            for b in "ACGT":
                hit = (arr == ord(b))
                base_acc[cls][b] += hit
            gcv = ((arr == ord("G")) | (arr == ord("C"))).astype(float)
            k = np.ones(a.gc_win)
            num = np.convolve(gcv, k, mode="same")
            den = np.convolve(np.ones(L), k, mode="same")
            gcprof = num / den
            gc_profiles[cls].append(gcprof)

            for mname, pats in MOTIFS.items():
                v = np.zeros(L)
                for i in motif_hits(seq, pats):
                    v[i] = 1.0
                motif_acc[cls][mname] += v
                mo_profiles[cls][mname].append(v)

            rec = dict(gene=g, chrom=r.chrom, pos=r.pos, strand=r.strand,
                       is_real=cls)
            for wname, w0, w1 in SUBWIN:
                i0, i1 = w0 + UP, w1 + UP
                s = seq[i0:i1]
                n = len(s)
                if n == 0:
                    continue
                gcf = (s.count("G") + s.count("C")) / n
                rec[wname + " GC"] = gcf
                rec[wname + " A"] = s.count("A") / n
                rec[wname + " T"] = s.count("T") / n
                for mname, pats in MOTIFS.items():
                    rec[wname + " " + mname] = len(motif_hits(s, pats))
                for i in range(0, n - a.kmer_k + 1):
                    kmer_counts[cls][wname][s[i:i + a.kmer_k]] += 1
                for i in range(0, n - 1):
                    dinuc_counts[cls][wname][s[i:i + 2]] += 1
            rows.append(rec)

        if len(gc_profiles[True]) == 1 and len(gc_profiles[False]) >= 1:
            pair_gc.append(gc_profiles[True][0] -
                           np.mean(gc_profiles[False], axis=0))
            for mname in MOTIFS:
                if mo_profiles[True][mname] and mo_profiles[False][mname]:
                    pair_motif[mname].append(
                        mo_profiles[True][mname][0] -
                        np.mean(mo_profiles[False][mname], axis=0))

    if n_bad:
        print(f"[seq]  {n_bad:,} sites dropped: window ran off the contig",
              file=sys.stderr)
    print(f"[seq]  {n_seq[True]:,} real and {n_seq[False]:,} competitor "
          f"windows read; {len(pair_gc):,} genes usable for pairing",
          file=sys.stderr)

    PER = pd.DataFrame(rows)
    W("per_site.tsv", PER)

    # binned profiles
    def binned(v, w=a.bin):
        n = (len(v) // w) * w
        return v[:n].reshape(-1, w).mean(axis=1)
    xb = binned(xs.astype(float))

    br = []
    for cls in (True, False):
        lab = "real 3' end" if cls else "competitor"
        for b in "ACGT":
            v = base_acc[cls][b] / max(n_seq[cls], 1)
            for x_, y_ in zip(xb, binned(v)):
                br.append(dict(cls=lab, base=b, pos=int(x_),
                               freq=round(float(y_), 5)))
    W("profile_base.tsv", pd.DataFrame(br))

    gr = []
    for cls in (True, False):
        lab = "real 3' end" if cls else "competitor"
        prof = ((base_acc[cls]["G"] + base_acc[cls]["C"]) /
                max(n_seq[cls], 1))
        sm = np.convolve(prof, np.ones(a.gc_win) / a.gc_win, mode="same")
        for x_, y_ in zip(xb, binned(sm)):
            gr.append(dict(cls=lab, pos=int(x_), gc=round(float(y_), 5)))
    W("profile_gc.tsv", pd.DataFrame(gr))

    if pair_gc:
        P = np.vstack(pair_gc)
        mu = P.mean(axis=0)
        se = P.std(axis=0, ddof=1) / np.sqrt(P.shape[0])
        W("profile_gc_paired.tsv", pd.DataFrame(dict(
            pos=xb.astype(int), delta_gc=np.round(binned(mu), 5),
            lo=np.round(binned(mu - 1.96 * se), 5),
            hi=np.round(binned(mu + 1.96 * se), 5),
            n_genes=P.shape[0])))
        i0, i1 = UP - 500, UP + 500
        print(f"[gc]   paired GC difference, -500..+500: "
              f"mean {mu[i0:i1].mean():+.4f}", file=sys.stderr)

    mr = []
    for cls in (True, False):
        lab = "real 3' end" if cls else "competitor"
        for mname in MOTIFS:
            v = motif_acc[cls][mname] / max(n_seq[cls], 1)
            for x_, y_ in zip(xb, binned(v)):
                mr.append(dict(cls=lab, motif=mname, pos=int(x_),
                               density=round(float(y_), 6)))
    W("profile_motif.tsv", pd.DataFrame(mr))

    pm = []
    for mname, lst in pair_motif.items():
        if not lst:
            continue
        P = np.vstack(lst)
        mu = P.mean(axis=0)
        se = P.std(axis=0, ddof=1) / np.sqrt(P.shape[0])
        for x_, m_, l_, h_ in zip(xb, binned(mu),
                                  binned(mu - 1.96 * se),
                                  binned(mu + 1.96 * se)):
            pm.append(dict(motif=mname, pos=int(x_),
                           delta=round(float(m_), 6),
                           lo=round(float(l_), 6), hi=round(float(h_), 6),
                           n_genes=P.shape[0]))
    if pm:
        W("profile_motif_paired.tsv", pd.DataFrame(pm))

    # position-free summaries
    cols = [c for c in PER.columns
            if c not in ("gene", "chrom", "pos", "strand", "is_real")]
    ws = []
    for c in cols:
        r_ = PER.loc[PER.is_real, c].to_numpy(float)
        k_ = PER.loc[~PER.is_real, c].to_numpy(float)
        r_ = r_[np.isfinite(r_)]; k_ = k_[np.isfinite(k_)]
        if not len(r_) or not len(k_):
            continue
        pooled = r_.mean() - k_.mean()
        sd = np.sqrt((r_.var(ddof=1) + k_.var(ddof=1)) / 2)
        ws.append(dict(feature=c, n_real=len(r_), n_comp=len(k_),
                       mean_real=round(float(r_.mean()), 5),
                       mean_comp=round(float(k_.mean()), 5),
                       pooled_diff=round(float(pooled), 5),
                       cohens_d=round(float(pooled / sd), 4) if sd > 0 else np.nan))
    W("window_summary.tsv", pd.DataFrame(ws).sort_values(
        "cohens_d", key=abs, ascending=False))

    pr = []
    for c in cols:
        vals = []
        for g, sub in PER.groupby("gene", sort=False):
            r_ = sub.loc[sub.is_real, c]
            k_ = sub.loc[~sub.is_real, c]
            if len(r_) != 1 or not len(k_):
                continue
            v = float(r_.iloc[0]) - float(k_.mean())
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            continue
        v = np.asarray(vals)
        se = v.std(ddof=1) / np.sqrt(len(v))
        pr.append(dict(feature=c, n_genes=len(v),
                       paired_diff=round(float(v.mean()), 5),
                       lo=round(float(v.mean() - 1.96 * se), 5),
                       hi=round(float(v.mean() + 1.96 * se), 5),
                       frac_positive=round(float((v > 0).mean()), 4),
                       cohens_dz=round(float(v.mean() / v.std(ddof=1)), 4)
                       if v.std(ddof=1) > 0 else np.nan))
    PRD = pd.DataFrame(pr).sort_values("cohens_dz", key=abs, ascending=False)
    W("window_paired.tsv", PRD)

    print("\n[paired] largest within-gene differences (real minus its own "
          "competitors):", file=sys.stderr)
    for r_ in PRD.head(15).itertuples():
        star = "" if (r_.lo <= 0 <= r_.hi) else "  *"
        print(f"         {r_.feature:<34} {r_.paired_diff:+8.4f}  "
              f"[{r_.lo:+.4f}, {r_.hi:+.4f}]  {100*r_.frac_positive:5.1f}% "
              f"of genes{star}", file=sys.stderr)

    dr = []
    for cls in (True, False):
        lab = "real 3' end" if cls else "competitor"
        for wname, _, _ in SUBWIN:
            cnt = dinuc_counts[cls][wname]
            tot = sum(cnt.values())
            if not tot:
                continue
            for dn_ in ["".join(p) for p in itertools.product("ACGT", repeat=2)]:
                dr.append(dict(cls=lab, window=wname, dinuc=dn_,
                               freq=round(cnt[dn_] / tot, 6)))
    W("dinuc.tsv", pd.DataFrame(dr))

    kr = []
    for wname, _, _ in SUBWIN:
        cr = kmer_counts[True][wname]; cc = kmer_counts[False][wname]
        tr, tc = sum(cr.values()), sum(cc.values())
        if not tr or not tc:
            continue
        keys = set(cr) | set(cc)
        for kk in keys:
            fr_ = cr[kk] / tr; fc_ = cc[kk] / tc
            if cr[kk] + cc[kk] < 50:
                continue
            kr.append(dict(window=wname, kmer=kk,
                           freq_real=round(fr_, 8), freq_comp=round(fc_, 8),
                           log2_ratio=round(float(np.log2((fr_ + 1e-9) /
                                                          (fc_ + 1e-9))), 4),
                           n_real=cr[kk], n_comp=cc[kk]))
    K = pd.DataFrame(kr)
    W("kmer_enrichment.tsv", K)
    if len(K):
        print(f"\n[kmer] most differential {a.kmer_k}-mers per window "
              f"(real vs competitor):", file=sys.stderr)
        for wname, _, _ in SUBWIN:
            s = K[K.window == wname]
            if not len(s):
                continue
            up_ = s.nlargest(3, "log2_ratio")
            dn_ = s.nsmallest(3, "log2_ratio")
            print(f"         {wname}:", file=sys.stderr)
            for r_ in up_.itertuples():
                print(f"           {r_.kmer}  {r_.log2_ratio:+.2f}  "
                      f"(enriched at real)", file=sys.stderr)
            for r_ in dn_.itertuples():
                print(f"           {r_.kmer}  {r_.log2_ratio:+.2f}  "
                      f"(enriched at competitors)", file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_real=n_seq[True], n_comp=n_seq[False],
                       n_genes=int(d.gene.nunique()),
                       n_genes_paired=len(pair_gc),
                       up=UP, dn=DN, gc_win=a.gc_win, bin=a.bin,
                       subwindows={w[0]: [w[1], w[2]] for w in SUBWIN}), fh,
                  indent=2)

    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead window_paired.tsv first: it is the within-gene contrast, "
          "which is where the signal is. Then compare it against "
          "profile_gc_paired.tsv and profile_motif_paired.tsv -- a feature "
          "strong in the window summary but flat in the profile is present "
          "without fixed geometry, and no position-anchored analysis will "
          "find it.", file=sys.stderr)


if __name__ == "__main__":
    main()