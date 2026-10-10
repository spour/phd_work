#!/usr/bin/env python3
# what separates real 3' ends from model false positives
import argparse, os, sys, json, gzip
from collections import defaultdict

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")
PAS = ["AATAAA", "ATTAAA"]

try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5, load_matrix3
    _M5, _M3 = load_matrix5(), load_matrix3()
    HAVE_MAXENT = True
except Exception:
    HAVE_MAXENT = False


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


# repeats
def load_rmsk(path, want_chroms):
    cols = defaultdict(list)
    op = gzip.open if path.endswith(".gz") else open
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13:
                continue
            chrom = f[5]
            if chrom not in want_chroms:
                continue
            cols[chrom].append((int(f[6]), int(f[7]), f[9], f[10], f[11], f[12]))
            n += 1
    out = {}
    for c, v in cols.items():
        v.sort(key=lambda x: x[0])
        out[c] = dict(start=np.array([x[0] for x in v]),
                      end=np.array([x[1] for x in v]),
                      strand=np.array([x[2] for x in v]),
                      name=np.array([x[3] for x in v]),
                      cls=np.array([x[4] for x in v]),
                      fam=np.array([x[5] for x in v]))
    print(f"[rmsk] {n:,} repeat intervals over {len(out)} chromosomes",
          file=sys.stderr)
    return out


FAMS = ("Alu", "L1", "L2", "MIR", "ERV1", "ERVL", "hAT-Charlie")


def repeat_at(rm, chrom, pos, strand):
    e = rm.get(chrom)
    blank = dict(rep_class="none", rep_family="none", rep_name="none",
                 rep_same_strand=np.nan, rep_frac=np.nan, rep_len=np.nan)
    if e is None or not len(e["start"]):
        return blank
    j = np.searchsorted(e["start"], pos, side="right") - 1
    if j < 0 or pos > e["end"][j]:
        return blank
    ln = int(e["end"][j] - e["start"][j])
    return dict(rep_class=str(e["cls"][j]), rep_family=str(e["fam"][j]),
                rep_name=str(e["name"][j]),
                rep_same_strand=float(e["strand"][j] == strand),
                rep_frac=float((pos - e["start"][j]) / max(ln, 1)),
                rep_len=float(ln))


def repeat_coverage(rm, chrom, lo, hi):
    e = rm.get(chrom)
    out = dict(rep_cov=0.0, **{"cov_" + f: 0.0 for f in FAMS})
    if e is None or not len(e["start"]):
        return out
    i0 = np.searchsorted(e["end"], lo, side="left")
    i1 = np.searchsorted(e["start"], hi, side="right")
    span = max(hi - lo, 1)
    tot = 0
    byfam = defaultdict(int)
    for k in range(i0, i1):
        a, b = max(lo, e["start"][k]), min(hi, e["end"][k])
        if b <= a:
            continue
        tot += b - a
        if e["fam"][k] in FAMS:
            byfam[e["fam"][k]] += b - a
    out["rep_cov"] = tot / span
    for f in FAMS:
        out["cov_" + f] = byfam[f] / span
    return out


# splice sites, weighted
def scan_splice(seq, centre, lo, hi):
    i0, i1 = max(0, centre + lo), min(len(seq), centre + hi)
    d5, s5, d3, s3 = [], [], [], []
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
            d5.append(i + 3 - centre); s5.append(sc)
    for i in range(i0, max(i0, i1 - 23)):
        if seq[i + 18:i + 20] != "AG":
            continue
        w = seq[i:i + 23]
        if "N" in w:
            continue
        try:
            sc = float(maxent.score3(w, matrix=_M3))
        except Exception:
            continue
        if np.isfinite(sc):
            d3.append(i + 20 - centre); s3.append(sc)
    return (np.array(d5, float), np.array(s5, float),
            np.array(d3, float), np.array(s3, float))


def weighted_best(d, s, side, lam=500.0):
    m = (d < 0) if side == "up" else (d > 0)
    if not m.any():
        return np.nan, np.nan
    w = s[m] - np.log1p(np.abs(d[m]) / lam)
    j = int(np.argmax(w))
    return float(w[j]), float(d[m][j])


def measure(gen, rm, chrom, pos, strand, UP, DN, gc_win, S, lam):
    seq = oriented(gen, chrom, pos, strand, UP, DN)
    L = UP + DN
    if len(seq) != L:
        return None
    c = UP
    rec = {}

    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    v = ((arr == ord("G")) | (arr == ord("C"))).astype(float)
    k = np.ones(gc_win)
    gcv = np.convolve(v, k, "same") / np.convolve(np.ones(L), k, "same")
    flank = float(np.nanmean(np.r_[gcv[c-3000:c-2000], gcv[c+2000:c+3000]]))
    inner = gcv[c-1000:c+1000]
    rec["gc_flank"] = flank
    rec["gc_min"] = float(np.nanmin(inner))
    rec["gc_trough_depth"] = flank - rec["gc_min"]
    rec["gc_trough_width"] = float(np.sum(inner < flank - 0.02))
    rec["gc_trough_area"] = float(np.sum(np.clip(flank - inner, 0, None)))
    rec["gc_min_pos"] = float(int(np.nanargmin(inner)) - 1000)
    rec["gc_core"] = float(np.nanmean(gcv[c-250:c+250]))

    d5, s5, d3, s3 = scan_splice(seq, c, -S, S)
    rec["n5ss"] = int(len(d5)); rec["n3ss"] = int(len(d3))
    rec["n5ss_up"] = int((d5 < 0).sum()); rec["n5ss_dn"] = int((d5 > 0).sum())
    rec["n5ss_beyond500"] = int((np.abs(d5) > 500).sum())
    rec["best5ss"] = float(s5.max()) if len(s5) else np.nan
    rec["best3ss_up"] = float(s3[d3 < 0].max()) if (d3 < 0).any() else np.nan
    rec["best3ss_dn"] = float(s3[d3 > 0].max()) if (d3 > 0).any() else np.nan

    wu, du = weighted_best(d3, s3, "up", lam)
    wd, dd_ = weighted_best(d5, s5, "dn", lam)
    rec["exon_acceptor_w"] = wu
    rec["exon_donor_w"] = wd
    rec["exon_acceptor_dist"] = du
    rec["exon_donor_dist"] = dd_
    rec["exon_def_min"] = (float(np.nanmin([wu, wd]))
                           if np.isfinite([wu, wd]).any() else np.nan)
    rec["exon_def_balance"] = (wu - wd if np.isfinite(wu) and np.isfinite(wd)
                               else np.nan)
    m = d5 > 0
    rec["nearest_donor_dist"] = float(d5[m].min()) if m.any() else np.nan
    rec["nearest_donor_score"] = (float(s5[m][np.argmin(d5[m])])
                                  if m.any() else np.nan)
    mu_ = d3 < 0
    rec["local_ag_gt_span"] = (float(d5[m].min() - d3[mu_].max())
                               if m.any() and mu_.any() else np.nan)

    if rm is not None:
        rec.update(repeat_at(rm, chrom, pos, strand))
        rec.update(repeat_coverage(rm, chrom, pos - 1000, pos + 1000))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--rmsk", default=None)
    ap.add_argument("--up", type=int, default=3000)
    ap.add_argument("--dn", type=int, default=3000)
    ap.add_argument("--gc_win", type=int, default=201)
    ap.add_argument("--splice_span", type=int, default=2500)
    ap.add_argument("--lam", type=float, default=500.0)
    ap.add_argument("--max_ccpa", type=int, default=8)
    ap.add_argument("--n_control", type=int, default=2,
                    help="matched intronic controls drawn per false positive")
    ap.add_argument("--out", default="ctx")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    UP, DN, S = a.up, a.dn, a.splice_span
    rng = np.random.default_rng(0)

    if not HAVE_MAXENT:
        sys.exit("maxentpy required; the splice half of this analysis is the point")
    print("[cfg]  MaxEnt splice scoring", file=sys.stderr)

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    pos = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[pos].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    canon = d[d.is_real].groupby("gene").pos.first().to_dict()

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
    rm = load_rmsk(a.rmsk, set(d.chrom.unique())) if a.rmsk else None

    # measure everything
    rows = []
    n_bad = 0
    for gi, (g, sub) in enumerate(d.groupby("gene", sort=False)):
        if gi and gi % 250 == 0:
            print(f"[run]  {gi:,} genes...", file=sys.stderr)
        cpos = canon.get(g)
        for r in sub.itertuples():
            rec = measure(gen, rm, r.chrom, r.pos, r.strand, UP, DN,
                          a.gc_win, S, a.lam)
            if rec is None:
                n_bad += 1
                continue
            rec.update(gene=g, chrom=r.chrom, pos=r.pos, strand=r.strand,
                       cls="real" if r.is_real else "fp")
            if cpos is not None and not r.is_real:
                dist = (cpos - r.pos) if r.strand != "-" else (r.pos - cpos)
                rec["dist_to_canon"] = float(dist)
            rows.append(rec)

            if r.is_real or cpos is None:
                continue
            off = abs(r.pos - cpos)
            if off < 200:
                continue
            for _ in range(a.n_control):
                f = float(rng.uniform(0.5, 2.0))
                delta = int(off * f)
                cp = cpos - delta if r.pos < cpos else cpos + delta
                crec = measure(gen, rm, r.chrom, cp, r.strand, UP, DN,
                               a.gc_win, S, a.lam)
                if crec is None:
                    continue
                crec.update(gene=g, chrom=r.chrom, pos=cp, strand=r.strand,
                            cls="control", dist_to_canon=float(
                                (cpos - cp) if r.strand != "-" else (cp - cpos)))
                rows.append(crec)

    if n_bad:
        print(f"[seq]  {n_bad:,} sites dropped at contig edges", file=sys.stderr)
    PER = pd.DataFrame(rows)
    PER["is_real"] = PER.cls == "real"
    W("per_site.tsv", PER)
    print(f"[seq]  {int((PER.cls=='real').sum()):,} real, "
          f"{int((PER.cls=='fp').sum()):,} false positives, "
          f"{int((PER.cls=='control').sum()):,} matched controls",
          file=sys.stderr)

    # repeats, three ways
    if rm is not None:
        rr = []
        for cl, lab in (("real", "real 3' end"), ("fp", "false positive"),
                        ("control", "matched intronic control")):
            s = PER[PER.cls == cl]
            if not len(s):
                continue
            row = dict(cls=lab, n=len(s),
                       in_any_repeat=round(float((s.rep_class != "none").mean()), 4),
                       mean_rep_cov_2kb=round(float(s.rep_cov.mean()), 4))
            for f in FAMS:
                row["in_" + f] = round(float((s.rep_family == f).mean()), 4)
            rr.append(row)
        RR = pd.DataFrame(rr)
        W("repeats.tsv", RR)
        print("\n[rep]  repeat context, three ways", file=sys.stderr)
        print(RR.to_string(index=False), file=sys.stderr)

        fp = PER[PER.cls == "fp"]; ct = PER[PER.cls == "control"]
        if len(ct):
            print("\n[rep]  FALSE POSITIVES vs MATCHED INTRONIC CONTROLS",
                  file=sys.stderr)
            print("       this is the comparison that means something: both "
                  "are intronic, both\n       at a matched distance from the "
                  "canonical site in the same gene",
                  file=sys.stderr)
            for f in ("any",) + FAMS:
                if f == "any":
                    x = (fp.rep_class != "none").astype(float).to_numpy()
                    y = (ct.rep_class != "none").astype(float).to_numpy()
                else:
                    x = (fp.rep_family == f).astype(float).to_numpy()
                    y = (ct.rep_family == f).astype(float).to_numpy()
                if y.mean() <= 0 and x.mean() <= 0:
                    continue
                sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
                dd = (x.mean() - y.mean()) / sd if sd > 0 else np.nan
                ratio = (x.mean() + 1e-6) / (y.mean() + 1e-6)
                flag = "" if abs(dd) < 0.1 else ("   <- enriched" if dd > 0
                                                 else "   <- depleted")
                print(f"       {f:<14} fp {100*x.mean():6.2f}%   control "
                      f"{100*y.mean():6.2f}%   ratio {ratio:5.2f}x   "
                      f"d {dd:+.3f}{flag}", file=sys.stderr)

            crows = []
            for f in ("any",) + FAMS:
                if f == "any":
                    x = (fp.rep_class != "none").astype(float)
                    y = (ct.rep_class != "none").astype(float)
                else:
                    x = (fp.rep_family == f).astype(float)
                    y = (ct.rep_family == f).astype(float)
                crows.append(dict(family=f, fp_rate=round(float(x.mean()), 5),
                                  control_rate=round(float(y.mean()), 5),
                                  ratio=round(float((x.mean()+1e-6) /
                                                    (y.mean()+1e-6)), 3),
                                  n_fp=len(x), n_control=len(y)))
            W("repeats_vs_control.tsv", pd.DataFrame(crows))

        fam = (PER.groupby(["cls", "rep_family"]).size().rename("n")
                  .reset_index())
        fam["frac"] = fam.n / fam.groupby("cls").n.transform("sum")
        W("repeat_families.tsv", fam.sort_values("n", ascending=False))

    if PER.local_ag_gt_span.notna().any():
        R = PER[PER.cls.isin(["real", "fp"])].copy()
        R["_q"] = pd.qcut(R.local_ag_gt_span.rank(method="first"), 5,
                          labels=False)
        ctl = []
        for b, s in R.groupby("_q"):
            vals = []
            for g, gg in s.groupby("gene", sort=False):
                r_ = gg.loc[gg.cls == "real", "n5ss"]
                c_ = gg.loc[gg.cls == "fp", "n5ss"]
                if len(r_) == 1 and len(c_):
                    vals.append(float(r_.iloc[0]) - float(c_.mean()))
            if len(vals) < 30:
                continue
            v = np.asarray(vals)
            ctl.append(dict(quintile=int(b) + 1, n_genes=len(v),
                            median_span=round(float(s.local_ag_gt_span.median()), 0),
                            paired_diff=round(float(v.mean()), 3),
                            dz=round(float(v.mean() / v.std(ddof=1)), 3)))
        CT = pd.DataFrame(ctl)
        W("donor_deficit_by_span.tsv", CT)
        print("\n[ctl]  donor deficit within strata of local AG-to-GT spacing",
              file=sys.stderr)
        print(CT.to_string(index=False), file=sys.stderr)

    skip = {"gene", "chrom", "pos", "strand", "is_real", "cls",
            "rep_class", "rep_family", "rep_name", "dist_to_canon", "_q"}
    RF = PER[PER.cls.isin(["real", "fp"])]
    feats = [c for c in RF.columns if c not in skip]
    out = []
    for c in feats:
        r = pd.to_numeric(RF.loc[RF.cls == "real", c], errors="coerce").dropna()
        k = pd.to_numeric(RF.loc[RF.cls == "fp", c], errors="coerce").dropna()
        if len(r) < 50 or len(k) < 50:
            continue
        sd = np.sqrt((r.var(ddof=1) + k.var(ddof=1)) / 2)
        vals = []
        for g, sub in RF.groupby("gene", sort=False):
            rv = pd.to_numeric(sub.loc[sub.cls == "real", c], errors="coerce")
            cv = pd.to_numeric(sub.loc[sub.cls == "fp", c], errors="coerce")
            if len(rv) != 1 or not cv.notna().any() or not np.isfinite(rv.iloc[0]):
                continue
            vals.append(float(rv.iloc[0] - cv.mean()))
        if len(vals) < 100:
            continue
        v = np.asarray(vals)
        vsd = v.std(ddof=1)
        se = vsd / np.sqrt(len(v))
        out.append(dict(
            feature=c, n_genes=len(v),
            pooled_real=round(float(r.mean()), 4),
            pooled_fp=round(float(k.mean()), 4),
            pooled_d=round(float((r.mean() - k.mean()) / sd), 4) if sd > 0 else np.nan,
            paired_diff=round(float(v.mean()), 4),
            paired_lo=round(float(v.mean() - 1.96 * se), 4),
            paired_hi=round(float(v.mean() + 1.96 * se), 4),
            paired_dz=round(float(v.mean() / vsd), 4) if vsd > 0 else np.nan,
            frac_positive=round(float((v > 0).mean()), 4),
            var_ratio=round(float(k.var(ddof=1) / max(r.var(ddof=1), 1e-12)), 3),
            q10_diff=round(float(np.percentile(r, 10) - np.percentile(k, 10)), 4),
            q90_diff=round(float(np.percentile(r, 90) - np.percentile(k, 90)), 4),
            sign_agrees=bool(np.sign(r.mean() - k.mean()) == np.sign(v.mean()))))
    CMP = pd.DataFrame(out).sort_values("paired_dz", key=abs, ascending=False)
    W("within_vs_between.tsv", CMP)

    print("\n[cmp]  strongest paired differences, real vs false positive",
          file=sys.stderr)
    print(f"       {'feature':<24} {'paired dz':>10} {'pooled d':>9} "
          f"{'var ratio':>10} {'agree':>6}", file=sys.stderr)
    for r_ in CMP.head(22).itertuples():
        print(f"       {r_.feature:<24} {r_.paired_dz:>+10.3f} "
              f"{r_.pooled_d:>+9.3f} {r_.var_ratio:>10.2f} "
              f"{'yes' if r_.sign_agrees else 'NO':>6}", file=sys.stderr)
    dis = CMP[~CMP.sign_agrees]
    if len(dis):
        print(f"\n       {len(dis)} feature(s) reverse between the two "
              f"comparisons:", file=sys.stderr)
        for r_ in dis.head(8).itertuples():
            print(f"         {r_.feature:<24} pooled {r_.pooled_real:.4f} vs "
                  f"{r_.pooled_fp:.4f}, paired {r_.paired_diff:+.4f}",
                  file=sys.stderr)

    ct = PER[PER.cls == "control"]
    if len(ct):
        out2 = []
        for c in feats:
            x = pd.to_numeric(PER.loc[PER.cls == "fp", c], errors="coerce").dropna()
            y = pd.to_numeric(ct[c], errors="coerce").dropna()
            if len(x) < 50 or len(y) < 50:
                continue
            sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            out2.append(dict(feature=c, fp_mean=round(float(x.mean()), 4),
                             control_mean=round(float(y.mean()), 4),
                             diff=round(float(x.mean() - y.mean()), 4),
                             d=round(float((x.mean() - y.mean()) / sd), 4)
                             if sd > 0 else np.nan,
                             n_fp=len(x), n_control=len(y)))
        C2 = pd.DataFrame(out2).sort_values("d", key=abs, ascending=False)
        W("fp_vs_control.tsv", C2)
        print("\n[ctr]  false positives vs matched intronic controls",
              file=sys.stderr)
        print("       what makes a position a MODEL FAILURE rather than just "
              "intronic", file=sys.stderr)
        for r_ in C2.head(16).itertuples():
            print(f"       {r_.feature:<24} fp {r_.fp_mean:>9.4f}   control "
                  f"{r_.control_mean:>9.4f}   d {r_.d:>+7.3f}", file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_real=int((PER.cls == "real").sum()),
                       n_fp=int((PER.cls == "fp").sum()),
                       n_control=int((PER.cls == "control").sum()),
                       n_genes=int(PER.gene.nunique()),
                       up=UP, dn=DN, splice_span=S, lam=a.lam,
                       n_control_per_fp=a.n_control, rmsk=bool(rm)), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead [rep] FALSE POSITIVES vs MATCHED INTRONIC CONTROLS. The "
          "three-way table above it is descriptive; only that comparison "
          "controls for the fact that real 3' ends are exonic and everything "
          "else is intronic.", file=sys.stderr)


if __name__ == "__main__":
    main()