#!/usr/bin/env python3
# four candidate mechanisms, tested properly
import argparse, os, sys, json, gzip, math
from collections import defaultdict

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5, load_matrix3
    _M5, _M3 = load_matrix5(), load_matrix3()
    HAVE_MAXENT = True
except Exception:
    HAVE_MAXENT = False

DG_NN = {"AA": -1.00, "TT": -1.00, "AT": -0.88, "TA": -0.58,
         "CA": -1.45, "TG": -1.45, "GT": -1.44, "AC": -1.44,
         "CT": -1.28, "AG": -1.28, "GA": -1.30, "TC": -1.30,
         "CG": -2.17, "GC": -2.24, "GG": -1.84, "CC": -1.84}

FEATURES = ["core_dG", "notch_fracbelow", "exon_acc_w", "n5ss_any"]
NICE = {"core_dG": "duplex free energy, core +/-250 nt",
        "notch_fracbelow": "GC notch extent, fraction of +/-1 kb below flank",
        "exon_acc_w": "upstream 3'ss, distance-weighted MaxEnt",
        "n5ss_any": "GT dinucleotide density, +/-2.5 kb"}
EXPECT = {"core_dG": +1, "notch_fracbelow": +1, "exon_acc_w": +1,
          "n5ss_any": -1}


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


def load_rmsk(path, want):
    cols = defaultdict(list)
    op = gzip.open if path.endswith(".gz") else open
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in want:
                continue
            cols[f[5]].append((int(f[6]), int(f[7]), f[9], f[10], f[11],
                               f[12], int(f[2])))
            n += 1
    out = {}
    for c, v in cols.items():
        v.sort(key=lambda x: x[0])
        out[c] = dict(start=np.array([x[0] for x in v]),
                      end=np.array([x[1] for x in v]),
                      strand=np.array([x[2] for x in v]),
                      name=np.array([x[3] for x in v]),
                      cls=np.array([x[4] for x in v]),
                      fam=np.array([x[5] for x in v]),
                      div=np.array([x[6] for x in v]))
    print(f"[rmsk] {n:,} intervals, {len(out)} chromosomes", file=sys.stderr)
    return out


def repeat_at(rm, chrom, pos, strand):
    out = dict(te_hit=0.0, te_family="none", te_class="none", te_sense=np.nan,
               te_frac=np.nan, te_len=np.nan, te_div=np.nan, te_name="none")
    e = rm.get(chrom)
    if e is None or not len(e["start"]):
        return out
    j = np.searchsorted(e["start"], pos, side="right") - 1
    if j < 0 or pos > e["end"][j]:
        return out
    ln = int(e["end"][j] - e["start"][j])
    return dict(te_hit=1.0, te_family=str(e["fam"][j]),
                te_class=str(e["cls"][j]), te_name=str(e["name"][j]),
                te_sense=float(e["strand"][j] == strand),
                te_frac=float((pos - e["start"][j]) / max(ln, 1)),
                te_len=float(ln), te_div=float(e["div"][j]) / 10.0)


def measure(gen, rm, chrom, pos, strand, up, dn, gc_win, span, lam):
    seq = oriented(gen, chrom, pos, strand, up, dn)
    if len(seq) != up + dn:
        return None
    c = up
    out = {}

    core = seq[c - 250:c + 250]
    di = [core[i:i + 2] for i in range(len(core) - 1)]
    di = [x for x in di if x in DG_NN]
    out["core_dG"] = float(np.mean([DG_NN[x] for x in di])) if di else np.nan

    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    v = ((arr == ord("G")) | (arr == ord("C"))).astype(float)
    k = np.ones(gc_win)
    gcv = np.convolve(v, k, "same") / np.convolve(np.ones(len(v)), k, "same")
    flank = float(np.nanmean(np.r_[gcv[c-3000:c-2000], gcv[c+2000:c+3000]]))
    inner = gcv[c-1000:c+1000]
    below = flank - inner
    out["notch_fracbelow"] = float(np.nanmean(below > 0))
    out["notch_depth"] = float(np.nanmax(below))
    out["gc_core"] = float(np.nanmean(gcv[c-250:c+250]))
    out["gc_flank"] = flank

    if HAVE_MAXENT:
        d3, s3, n5 = [], [], 0
        i0, i1 = max(0, c - span), min(len(seq), c + span)
        for i in range(i0, max(i0, i1 - 9)):
            if seq[i + 3:i + 5] == "GT" and "N" not in seq[i:i + 9]:
                n5 += 1
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
                d3.append(i + 20 - c); s3.append(sc)
        out["n5ss_any"] = float(n5)
        d3 = np.array(d3, float); s3 = np.array(s3, float)
        m = d3 < 0
        if m.any():
            w = s3[m] - np.log1p(np.abs(d3[m]) / lam)
            j = int(np.argmax(w))
            out["exon_acc_w"] = float(w[j])
            out["exon_acc_dist"] = float(d3[m][j])
            out["exon_acc_raw"] = float(s3[m][j])
        else:
            out["exon_acc_w"] = np.nan
            out["exon_acc_dist"] = np.nan
            out["exon_acc_raw"] = np.nan

    if rm is not None:
        out.update(repeat_at(rm, chrom, pos, strand))
    return out


# statistics
def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return q
    pv = p[ok]; o = np.argsort(pv); n = len(pv)
    adj = pv[o] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty(n); tmp[o] = np.clip(adj, 0, 1)
    q[ok] = tmp
    return q


def boot_ci(v, rng, n=2000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 10:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_stats(v, rng, nboot=2000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    out = dict(n=len(v))
    if len(v) < 20:
        return out
    sd = v.std(ddof=1)
    out["mean"] = float(v.mean())
    out["dz"] = float(v.mean() / sd) if sd > 0 else np.nan
    out["frac_pos"] = float((v > 0).mean())
    lo, hi = boot_ci(v, rng, nboot)
    out["boot_lo"], out["boot_hi"] = lo, hi
    if HAVE_SCIPY:
        t, p = sps.ttest_1samp(v, 0.0)
        out["t"] = float(t); out["p_t"] = float(p)
        try:
            _, pw = sps.wilcoxon(v, zero_method="zsplit")
            out["p_wilcoxon"] = float(pw)
        except Exception:
            out["p_wilcoxon"] = np.nan
        k = int((v > 0).sum())
        out["p_sign"] = float(sps.binomtest(k, len(v), 0.5).pvalue)
    return out


def two_sample(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    out = dict(n_a=len(x), n_b=len(y))
    if len(x) < 20 or len(y) < 20:
        return out
    sd = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    out["mean_a"] = float(x.mean()); out["mean_b"] = float(y.mean())
    out["d"] = float((x.mean() - y.mean()) / sd) if sd > 0 else np.nan
    if HAVE_SCIPY:
        t, p = sps.ttest_ind(x, y, equal_var=False)
        out["p"] = float(p)
        try:
            u, pu = sps.mannwhitneyu(x, y, alternative="two-sided")
            out["auc"] = float(u / (len(x) * len(y)))
            out["p_mwu"] = float(pu)
        except Exception:
            out["auc"] = np.nan; out["p_mwu"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--rmsk", default=None)
    ap.add_argument("--up", type=int, default=3500)
    ap.add_argument("--dn", type=int, default=3500)
    ap.add_argument("--gc_win", type=int, default=201)
    ap.add_argument("--span", type=int, default=2500)
    ap.add_argument("--lam", type=float, default=500.0)
    ap.add_argument("--max_ccpa", type=int, default=8)
    ap.add_argument("--n_control", type=int, default=2)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--holdout_frac", type=float, default=0.3)
    ap.add_argument("--out", default="final")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)

    if not HAVE_MAXENT:
        print("[warn] maxentpy absent; the two splice features will be NaN",
              file=sys.stderr)
    print(f"[cfg]  scipy {HAVE_SCIPY}, maxent {HAVE_MAXENT}", file=sys.stderr)

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

    rows = []
    nbad = 0
    for gi, (g, sub) in enumerate(d.groupby("gene", sort=False)):
        if gi and gi % 400 == 0:
            print(f"[run]  {gi:,} genes...", file=sys.stderr)
        cpos = canon.get(g)
        todo = [(r.chrom, r.pos, r.strand, "real" if r.is_real else "fp")
                for r in sub.itertuples()]
        if cpos is not None:
            for r in sub.itertuples():
                if r.is_real:
                    continue
                off = abs(r.pos - cpos)
                if off < 200:
                    continue
                for _ in range(a.n_control):
                    f = float(rng.uniform(0.5, 2.0))
                    cp = cpos - int(off * f) if r.pos < cpos else cpos + int(off * f)
                    todo.append((r.chrom, cp, r.strand, "control"))
        for chrom, p_, st, cls in todo:
            rec = measure(gen, rm, chrom, p_, st, a.up, a.dn, a.gc_win,
                          a.span, a.lam)
            if rec is None:
                nbad += 1
                continue
            rec.update(gene=g, chrom=chrom, pos=p_, strand=st, cls=cls)
            rows.append(rec)
    if nbad:
        print(f"[seq]  {nbad:,} positions dropped at contig edges",
              file=sys.stderr)
    P = pd.DataFrame(rows)
    W("per_site.tsv", P)
    print(f"[seq]  {len(P):,} positions measured", file=sys.stderr)

    RF = P[P.cls.isin(["real", "fp"])]
    FP = P[P.cls == "fp"]
    CT = P[P.cls == "control"]

    rb = RF[RF.cls == "real"].drop_duplicates("gene").set_index("gene")
    fm = RF[RF.cls == "fp"].groupby("gene")[FEATURES].mean()
    common = rb.index.intersection(fm.index)
    DIFF = (rb.loc[common, FEATURES].astype(float) -
            fm.loc[common, FEATURES].astype(float))
    DIFF["chrom"] = rb.loc[common, "chrom"].to_numpy()
    W("paired_differences.tsv", DIFF.reset_index())

    # headline table
    res = []
    for f in FEATURES:
        pr = paired_stats(DIFF[f].to_numpy(), rng, a.nboot)
        po = two_sample(RF.loc[RF.cls == "real", f], RF.loc[RF.cls == "fp", f])
        cc = two_sample(FP[f], CT[f]) if len(CT) else {}
        res.append(dict(feature=f, description=NICE[f],
                        expected_sign=EXPECT[f],
                        n_genes=pr.get("n"), paired_diff=pr.get("mean"),
                        paired_dz=pr.get("dz"),
                        boot_lo=pr.get("boot_lo"), boot_hi=pr.get("boot_hi"),
                        frac_pos=pr.get("frac_pos"),
                        p_t=pr.get("p_t"), p_wilcoxon=pr.get("p_wilcoxon"),
                        p_sign=pr.get("p_sign"),
                        pooled_real=po.get("mean_a"), pooled_fp=po.get("mean_b"),
                        pooled_d=po.get("d"), pooled_auc=po.get("auc"),
                        pooled_p=po.get("p"),
                        fp_vs_control_d=cc.get("d"),
                        fp_vs_control_p=cc.get("p"),
                        control_mean=cc.get("mean_b")))
    R = pd.DataFrame(res)
    for c in ("p_t", "p_wilcoxon", "p_sign", "pooled_p", "fp_vs_control_p"):
        R["q_" + c.replace("p_", "")] = bh(R[c])
    W("headline.tsv", R)

    print("\n[main] paired, within gene", file=sys.stderr)
    print(f"       {'feature':<20}{'diff':>10}{'dz':>8}{'95% CI':>22}"
          f"{'%genes':>8}{'q (Wilcoxon)':>14}", file=sys.stderr)
    for r_ in R.itertuples():
        ci = f"[{r_.boot_lo:+.4f}, {r_.boot_hi:+.4f}]"
        print(f"       {r_.feature:<20}{r_.paired_diff:>+10.4f}"
              f"{r_.paired_dz:>+8.3f}{ci:>22}{100*r_.frac_pos:>7.1f}%"
              f"{r_.q_wilcoxon:>14.2e}", file=sys.stderr)

    print("\n[main] the same features, pooled across genes and against "
          "matched intronic controls", file=sys.stderr)
    print(f"       {'feature':<20}{'pooled d':>10}{'AUC':>8}"
          f"{'ctrl d':>9}   sign agrees", file=sys.stderr)
    for r_ in R.itertuples():
        ag = "yes" if np.sign(r_.pooled_d) == np.sign(r_.paired_dz) else "NO"
        print(f"       {r_.feature:<20}{r_.pooled_d:>+10.3f}"
              f"{r_.pooled_auc:>8.3f}{r_.fp_vs_control_d:>+9.3f}   {ag}",
              file=sys.stderr)
    print("       the control column answers 'model failure or just introns' "
          "and is\n       diagnostic only where the exon/intron contrast is a "
          "live alternative", file=sys.stderr)

    # within-gene permutation null
    print(f"\n[perm] within-gene label permutation, {a.nperm} rounds",
          file=sys.stderr)
    gl = {g: sub[FEATURES].to_numpy(float)
          for g, sub in RF.groupby("gene", sort=False) if len(sub) >= 2}
    obs = {f: R.loc[R.feature == f, "paired_dz"].iloc[0] for f in FEATURES}
    nulls = {f: [] for f in FEATURES}
    for it in range(a.nperm):
        if it and it % 250 == 0:
            print(f"       {it}/{a.nperm}", file=sys.stderr)
        acc = []
        for g, M in gl.items():
            k = rng.integers(0, len(M))
            others = np.delete(M, k, axis=0)
            acc.append(M[k] - np.nanmean(others, axis=0))
        A = np.array(acc, float)
        sd = np.nanstd(A, axis=0, ddof=1)
        dz = np.nanmean(A, axis=0) / np.where(sd > 0, sd, np.nan)
        for j, f in enumerate(FEATURES):
            nulls[f].append(dz[j])
    prow = []
    for f in FEATURES:
        nl = np.array(nulls[f], float)
        nl = nl[np.isfinite(nl)]
        p = (1 + int((np.abs(nl) >= abs(obs[f])).sum())) / (len(nl) + 1)
        prow.append(dict(feature=f, observed_dz=round(obs[f], 4),
                         null_mean=round(float(nl.mean()), 4),
                         null_sd=round(float(nl.std(ddof=1)), 4),
                         null_p95=round(float(np.percentile(np.abs(nl), 95)), 4),
                         p_perm=p, n_perm=len(nl)))
        print(f"       {f:<20} observed {obs[f]:+.3f}, null "
              f"{nl.mean():+.3f} +/- {nl.std(ddof=1):.3f}, p {p:.3g}",
              file=sys.stderr)
    W("permutation.tsv", pd.DataFrame(prow))

    # leave one chromosome out
    print("\n[loco] per chromosome, to check no single one carries the effect",
          file=sys.stderr)
    lrow = []
    for c in sorted(DIFF.chrom.dropna().unique()):
        s = DIFF[DIFF.chrom == c]
        if len(s) < 50:
            continue
        for f in FEATURES:
            v = s[f].to_numpy(float); v = v[np.isfinite(v)]
            if len(v) < 30:
                continue
            sd = v.std(ddof=1)
            lrow.append(dict(chrom=c, feature=f, n_genes=len(v),
                             dz=round(float(v.mean() / sd), 4) if sd > 0 else np.nan))
    L = pd.DataFrame(lrow)
    W("per_chromosome.tsv", L)
    for f in FEATURES:
        s = L[L.feature == f]
        if not len(s):
            continue
        agree = float((np.sign(s.dz) == np.sign(obs[f])).mean())
        print(f"       {f:<20} median dz {s.dz.median():+.3f}, range "
              f"{s.dz.min():+.3f} to {s.dz.max():+.3f}, "
              f"{100*agree:.0f}% of chromosomes agree in sign", file=sys.stderr)

    genes = DIFF.index.to_numpy()
    ho = set(rng.choice(genes, int(a.holdout_frac * len(genes)), replace=False))
    hrow = []
    for f in FEATURES:
        for lab, m in (("discovery", ~np.isin(genes, list(ho))),
                       ("holdout", np.isin(genes, list(ho)))):
            v = DIFF[f].to_numpy(float)[m]
            st = paired_stats(v, rng, 500)
            hrow.append(dict(feature=f, split=lab, n_genes=st.get("n"),
                             dz=round(st.get("dz", np.nan), 4),
                             p=st.get("p_wilcoxon", np.nan)))
    H = pd.DataFrame(hrow)
    W("holdout.tsv", H)
    print("\n[hold] discovery vs held-out genes", file=sys.stderr)
    for f in FEATURES:
        s = H[H.feature == f].set_index("split")
        print(f"       {f:<20} discovery dz {s.loc['discovery','dz']:+.3f}, "
              f"holdout dz {s.loc['holdout','dz']:+.3f}", file=sys.stderr)

    C = DIFF[FEATURES].corr()
    W("feature_correlation.tsv", C.reset_index().rename(columns={"index": "f"}))
    print("\n[vif]  are the four independent?", file=sys.stderr)
    print(C.round(3).to_string(), file=sys.stderr)
    Z = DIFF[FEATURES].apply(lambda s: (s - s.mean()) / (s.std(ddof=1) or 1))
    Z = Z.fillna(0.0).to_numpy()
    for j, f in enumerate(FEATURES):
        others = [k for k in range(len(FEATURES)) if k != j]
        A = np.column_stack([np.ones(len(Z)), Z[:, others]])
        beta = np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
        r2 = 1 - np.sum((Z[:, j] - A @ beta) ** 2) / np.sum((Z[:, j] - Z[:, j].mean()) ** 2)
        print(f"       {f:<20} R2 on the other three {r2:.3f}, "
              f"VIF {1/max(1-r2,1e-9):.2f}", file=sys.stderr)

    if rm is not None:
        print("\n[te]   transposable elements. Pol II reads sequence and has no "
              "category for\n       'retrotransposon', so TE overlap cannot be "
              "a proximate mechanism. The\n       question is whether TE-borne "
              "sites differ from the rest ON THE FOUR\n       FEATURES -- if "
              "they do, TE origin acts through them; if not, whatever\n       "
              "represses TE-borne signals is not in the local sequence.",
              file=sys.stderr)
        trow = []
        fams = [f for f in FP.te_family.dropna().unique() if f != "none"]
        for fam in fams:
            for sense, lab in ((1.0, "sense"), (0.0, "antisense")):
                x = ((FP.te_family == fam) & (FP.te_sense == sense)).astype(float)
                y = ((CT.te_family == fam) & (CT.te_sense == sense)).astype(float)
                if x.sum() + y.sum() < 40:
                    continue
                trow.append(dict(family=fam, orientation=lab,
                                 fp_rate=round(float(x.mean()), 5),
                                 control_rate=round(float(y.mean()), 5),
                                 ratio=round(float((x.mean()+1e-6)/(y.mean()+1e-6)), 3),
                                 n_fp=int(x.sum()), n_control=int(y.sum())))
        T = pd.DataFrame(trow).sort_values("ratio", ascending=False)
        W("te_enrichment.tsv", T)
        print("\n       enrichment over matched intronic controls:",
              file=sys.stderr)
        print(T.head(10).to_string(index=False), file=sys.stderr)

        senseTE = FP[(FP.te_hit == 1) & (FP.te_sense == 1)]
        noTE = FP[FP.te_hit == 0]
        srow = []
        for f in FEATURES:
            st = two_sample(senseTE[f], noTE[f])
            srow.append(dict(feature=f, sense_te_mean=st.get("mean_a"),
                             no_te_mean=st.get("mean_b"), d=st.get("d"),
                             p=st.get("p"), n_te=st.get("n_a"),
                             n_no_te=st.get("n_b")))
        S = pd.DataFrame(srow)
        S["q"] = bh(S.p)
        W("te_vs_features.tsv", S)
        print("\n       sense-oriented TE false positives vs non-TE false "
              "positives:", file=sys.stderr)
        for r_ in S.itertuples():
            print(f"         {r_.feature:<20} d {r_.d:+.3f}  q {r_.q:.2e}",
                  file=sys.stderr)
        big = S[S.d.abs() > 0.2]
        print(f"       {len(big)} of {len(S)} features differ by |d| > 0.2. "
              + ("TE origin acts partly through these features."
                 if len(big) else
                 "TE-borne sites look like any other false positive on all "
                 "four, so\n       their repression is not explained by local "
                 "sequence."), file=sys.stderr)

        st_ = senseTE[senseTE.te_family.isin(["ERV1", "ERVL", "ERVL-MaLR",
                                              "ERVK", "L1", "Alu"])]
        if len(st_):
            gr = (st_.groupby("te_family")
                  .agg(n=("te_frac", "size"),
                       median_frac_into_element=("te_frac", "median"),
                       median_element_len=("te_len", "median"),
                       median_divergence_pct=("te_div", "median")).round(3))
            W("te_position.tsv", gr.reset_index())
            print("\n       where in the element the site sits, and element "
                  "age:", file=sys.stderr)
            print(gr.to_string(), file=sys.stderr)

    plot = []
    for f in FEATURES:
        for cl in ("real", "fp", "control"):
            s = P[P.cls == cl]
            if not len(s):
                continue
            v = pd.to_numeric(s[f], errors="coerce").dropna()
            plot.append(pd.DataFrame(dict(feature=f, cls=cl, value=v.to_numpy())))
    W("plot_values.tsv", pd.concat(plot, ignore_index=True))
    pd_ = DIFF[FEATURES].reset_index().melt(id_vars="gene", var_name="feature",
                                            value_name="paired_diff")
    W("plot_paired.tsv", pd_)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_real=int((P.cls == "real").sum()),
                       n_fp=int((P.cls == "fp").sum()),
                       n_control=int((P.cls == "control").sum()),
                       n_genes=int(P.gene.nunique()),
                       features=FEATURES, nboot=a.nboot, nperm=a.nperm), fh,
                  indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nheadline.tsv carries all three comparisons with FDR-corrected q "
          "for each.\npermutation.tsv is the null that matters. "
          "per_chromosome.tsv and holdout.tsv\nsay whether the effects are "
          "stable. feature_correlation.tsv and the VIF block\nsay how many "
          "independent things you actually have.", file=sys.stderr)


if __name__ == "__main__":
    main()