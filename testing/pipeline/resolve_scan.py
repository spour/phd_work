#!/usr/bin/env python3
# four questions left open by context_scan.py
import argparse, os, sys, json, gzip
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")

try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5
    _M5 = load_matrix5()
    HAVE_MAXENT = True
except Exception:
    HAVE_MAXENT = False


class Genome:
    def __init__(self, path):
        fai = path + ".fai"
        if not os.path.exists(fai):
            sys.exit("need " + fai)
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


def cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / sd) if sd > 0 else np.nan


def donor_profile(seq, centre, span):
    d, s = [], []
    i0, i1 = max(0, centre - span), min(len(seq), centre + span)
    for i in range(i0, max(i0, i1 - 9)):
        if seq[i + 3:i + 5] != "GT":
            continue
        w = seq[i:i + 9]
        if "N" in w:
            continue
        try:
            sc = float(maxent.score5(w, matrix=_M5))
        except Exception:
            continue
        if np.isfinite(sc):
            d.append(i + 3 - centre)
            s.append(sc)
    return np.array(d, float), np.array(s, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_site", required=True,
                    help="ctx/per_site.tsv from context_scan.py")
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--rmsk", default=None)
    ap.add_argument("--up", type=int, default=3000)
    ap.add_argument("--dn", type=int, default=3000)
    ap.add_argument("--gc_win", type=int, default=201)
    ap.add_argument("--span", type=int, default=2500)
    ap.add_argument("--thresholds", default="-99,0,3,6,8,10",
                    help="MaxEnt 5'ss cutoffs for the proximity test")
    ap.add_argument("--out", default="resolve")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    UP, DN = a.up, a.dn
    if not HAVE_MAXENT:
        sys.exit("maxentpy required")

    P = pd.read_csv(a.per_site, sep="\t")
    print(f"[load] {len(P):,} positions: "
          f"{int((P.cls=='real').sum()):,} real, "
          f"{int((P.cls=='fp').sum()):,} false positives, "
          f"{int((P.cls=='control').sum()):,} controls", file=sys.stderr)
    gen = Genome(a.genome)

    # q1: is the near donor a splice site?
    THR = [float(x) for x in a.thresholds.split(",")]
    print("\n[Q1]   nearest-donor distance at increasing MaxEnt cutoffs",
          file=sys.stderr)
    print("       a GT dinucleotide scores around -10; a genuine 5' splice "
          "site scores\n       above ~6. If the fp-vs-control gap closes as "
          "the cutoff rises, the\n       proximity was dinucleotide density, "
          "not U1.", file=sys.stderr)

    rows = []
    per_thr = {t: {"real": [], "fp": [], "control": []} for t in THR}
    ndone = 0
    for r in P.itertuples():
        seq = oriented(gen, r.chrom, r.pos, r.strand, UP, DN)
        if len(seq) != UP + DN:
            continue
        d, s = donor_profile(seq, UP, a.span)
        ndone += 1
        if ndone % 5000 == 0:
            print(f"       {ndone:,} positions scanned...", file=sys.stderr)
        dn_ = d > 0
        for t in THR:
            m = dn_ & (s >= t)
            per_thr[t][r.cls].append(float(d[m].min()) if m.any() else np.nan)

    for t in THR:
        fp = np.array(per_thr[t]["fp"], float)
        ct = np.array(per_thr[t]["control"], float)
        rl = np.array(per_thr[t]["real"], float)
        fpf = fp[np.isfinite(fp)]; ctf = ct[np.isfinite(ct)]
        rlf = rl[np.isfinite(rl)]
        rows.append(dict(
            maxent_cutoff=t,
            frac_fp_with_donor=round(float(np.isfinite(fp).mean()), 4),
            frac_control_with_donor=round(float(np.isfinite(ct).mean()), 4),
            median_fp=round(float(np.median(fpf)), 1) if len(fpf) else np.nan,
            median_control=round(float(np.median(ctf)), 1) if len(ctf) else np.nan,
            median_real=round(float(np.median(rlf)), 1) if len(rlf) else np.nan,
            d_fp_vs_control=round(cohens_d(fpf, ctf), 4),
            d_fp_vs_real=round(cohens_d(fpf, rlf), 4)))
        print(f"       cutoff {t:>5.0f}: fp median {rows[-1]['median_fp']:>8.1f} nt, "
              f"control {rows[-1]['median_control']:>8.1f} nt, "
              f"d {rows[-1]['d_fp_vs_control']:+.3f}   "
              f"({100*rows[-1]['frac_fp_with_donor']:.0f}% of fp have one)",
              file=sys.stderr)
    W("q1_donor_thresholds.tsv", pd.DataFrame(rows))
    strong = [r for r in rows if r["maxent_cutoff"] >= 6]
    if strong:
        d6 = strong[0]["d_fp_vs_control"]
        print(f"\n       at MaxEnt >= 6 the effect is d {d6:+.3f}. "
              + ("U1 occlusion remains plausible."
                 if abs(d6) > 0.15 else
                 "The proximity was GT density, not splice sites."),
              file=sys.stderr)

    # q2: depth vs width vs area
    print("\n[Q2]   which summary of the GC notch separates best?",
          file=sys.stderr)
    shapes = []
    for r in P.itertuples():
        seq = oriented(gen, r.chrom, r.pos, r.strand, UP, DN)
        if len(seq) != UP + DN:
            continue
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        v = ((arr == ord("G")) | (arr == ord("C"))).astype(float)
        k = np.ones(a.gc_win)
        gcv = np.convolve(v, k, "same") / np.convolve(np.ones(len(v)), k, "same")
        c = UP
        flank = float(np.nanmean(np.r_[gcv[c-3000:c-2000], gcv[c+2000:c+3000]]))
        inner = gcv[c-1000:c+1000]
        below = flank - inner
        rec = dict(gene=r.gene, cls=r.cls,
                   depth=float(np.nanmax(below)),
                   width=float(np.sum(below > 0.02)),
                   area=float(np.sum(np.clip(below, 0, None))),
                   frac_below=float(np.mean(below > 0)),
                   mean_below=float(np.nanmean(np.clip(below, 0, None))))
        j = int(np.nanargmin(inner))
        left = below[:j]; right = below[j:]
        rec["asymmetry"] = (float(np.nansum(left) - np.nansum(right)) /
                            max(float(np.nansum(below)), 1e-9))
        rec["min_offset"] = float(j - 1000)
        shapes.append(rec)
    SH = pd.DataFrame(shapes)
    W("q2_notch_shapes.tsv", SH)

    q2 = []
    for c in ("depth", "width", "area", "frac_below", "mean_below",
              "asymmetry", "min_offset"):
        rl = SH.loc[SH.cls == "real", c]
        fp = SH.loc[SH.cls == "fp", c]
        vals = []
        for g, sub in SH.groupby("gene", sort=False):
            a_ = sub.loc[sub.cls == "real", c]
            b_ = sub.loc[sub.cls == "fp", c]
            if len(a_) == 1 and len(b_):
                vals.append(float(a_.iloc[0]) - float(b_.mean()))
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        q2.append(dict(summary=c, pooled_d=round(cohens_d(rl, fp), 4),
                       paired_dz=round(float(v.mean() / v.std(ddof=1)), 4)
                       if len(v) > 1 and v.std(ddof=1) > 0 else np.nan,
                       n_genes=len(v),
                       real_mean=round(float(rl.mean()), 4),
                       fp_mean=round(float(fp.mean()), 4)))
    Q2 = pd.DataFrame(q2).sort_values("paired_dz", key=abs, ascending=False)
    W("q2_summary.tsv", Q2)
    print(Q2.to_string(index=False), file=sys.stderr)

    # q3: sense-oriented tes
    if a.rmsk and "rep_family" in P.columns:
        print("\n[Q3]   transposable elements split by orientation",
              file=sys.stderr)
        print("       a retrotransposon's poly(A) signal only reads as one when "
              "the element\n       lies in the transcript's orientation, so "
              "pooling both halves any effect",
              file=sys.stderr)
        rows3 = []
        fams = [f for f in P.rep_family.dropna().unique() if f != "none"]
        fp = P[P.cls == "fp"]; ct = P[P.cls == "control"]
        for f in fams:
            for sense, lab in ((1.0, "sense"), (0.0, "antisense")):
                x = ((fp.rep_family == f) &
                     (fp.rep_same_strand == sense)).astype(float)
                y = ((ct.rep_family == f) &
                     (ct.rep_same_strand == sense)).astype(float)
                if x.sum() < 20 and y.sum() < 20:
                    continue
                rows3.append(dict(family=f, orientation=lab,
                                  fp_rate=round(float(x.mean()), 5),
                                  control_rate=round(float(y.mean()), 5),
                                  ratio=round(float((x.mean() + 1e-6) /
                                                    (y.mean() + 1e-6)), 3),
                                  d=round(cohens_d(x, y), 4),
                                  n_fp=int(x.sum()), n_control=int(y.sum())))
        Q3 = pd.DataFrame(rows3).sort_values("ratio", ascending=False)
        W("q3_te_orientation.tsv", Q3)
        show = Q3[(Q3.n_fp + Q3.n_control) >= 100]
        print(show.head(16).to_string(index=False), file=sys.stderr)
        s_ = Q3[Q3.orientation == "sense"]
        if len(s_):
            print(f"\n       sense-oriented, ratio > 1.3: "
                  f"{', '.join(s_[s_.ratio > 1.3].family.tolist()) or 'none'}",
                  file=sys.stderr)

    # q4: one failure mode or several?
    print("\n[Q4]   do the features flag the same false positives?",
          file=sys.stderr)
    FEATS = ["nearest_donor_dist", "n5ss", "best3ss_up", "exon_acceptor_w",
             "gc_trough_depth", "gc_trough_width", "gc_core", "rep_cov"]
    FEATS = [f for f in FEATS if f in P.columns]
    RF = P[P.cls.isin(["real", "fp"])].copy()
    devs = []
    for g, sub in RF.groupby("gene", sort=False):
        r_ = sub[sub.cls == "real"]
        f_ = sub[sub.cls == "fp"]
        if len(r_) != 1 or not len(f_):
            continue
        for t in f_.itertuples():
            rec = dict(gene=g, pos=t.pos)
            for c in FEATS:
                rv = getattr(r_.iloc[0], c, np.nan)
                fv = getattr(t, c, np.nan)
                rec[c] = float(fv) - float(rv) if (np.isfinite(fv) and
                                                   np.isfinite(rv)) else np.nan
            devs.append(rec)
    D = pd.DataFrame(devs)
    W("q4_deviations.tsv", D)

    SIGN = {"nearest_donor_dist": -1, "n5ss": +1, "best3ss_up": -1,
            "exon_acceptor_w": -1, "gc_trough_depth": -1,
            "gc_trough_width": -1, "gc_core": +1, "rep_cov": +1}
    Z = pd.DataFrame(index=D.index)
    for c in FEATS:
        v = D[c].to_numpy(float)
        sd = np.nanstd(v)
        Z[c] = SIGN.get(c, 1) * (v - np.nanmean(v)) / (sd if sd > 0 else 1)

    print("\n       correlation between features, across false positives:",
          file=sys.stderr)
    C = Z.corr()
    print(C.round(2).to_string(), file=sys.stderr)
    W("q4_feature_corr.tsv", C.reset_index().rename(columns={"index": "feature"}))

    q = 0.90
    ext = {c: (Z[c] >= Z[c].quantile(q)).to_numpy() for c in FEATS}
    ov = []
    for x, y in combinations(FEATS, 2):
        a_, b_ = ext[x], ext[y]
        both = float((a_ & b_).mean())
        exp = float(a_.mean() * b_.mean())
        ov.append(dict(feature_a=x, feature_b=y,
                       observed=round(both, 5), expected=round(exp, 5),
                       enrichment=round(both / exp, 3) if exp > 0 else np.nan))
    OV = pd.DataFrame(ov).sort_values("enrichment", ascending=False)
    W("q4_extreme_overlap.tsv", OV)
    print("\n       overlap of the top-decile sites, observed vs independence:",
          file=sys.stderr)
    print(OV.head(12).to_string(index=False), file=sys.stderr)
    print(f"\n       enrichment near 1.00 means the features flag DIFFERENT "
          f"sites and the\n       false-positive set is a mixture of failure "
          f"modes", file=sys.stderr)

    n_ext = sum(ext[c] for c in FEATS)
    print(f"\n       false positives extreme on k features (independence "
          f"would give ~{len(FEATS)*(1-q):.1f} on average):", file=sys.stderr)
    for k in range(0, min(len(FEATS), 6) + 1):
        print(f"         k={k}: {int((n_ext == k).sum()):>6,} "
              f"({100*(n_ext == k).mean():5.1f}%)", file=sys.stderr)
    print(f"         none: {int((n_ext == 0).sum()):,} false positives are "
          f"extreme on nothing", file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_positions=len(P), n_fp_deviations=len(D),
                       features=FEATS, thresholds=THR), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()