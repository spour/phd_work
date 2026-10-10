#!/usr/bin/env python3
# quantify nascent polya signal from gse135140 at our own peak centers and turn it

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

try:
    import pyBigWig
except ImportError:
    sys.exit("pyBigWig missing:  pip install pyBigWig --no-binary pyBigWig")

BUILD_BY_CHR1 = {248956422: "hg38", 249250621: "hg19"}

COND = {
    "cAMO":  "GSM3989596_EU-labeled_polyA_cAMO",
    "U1AMO": "GSM3989597_EU-labeled_polyA_U1AMO",
}


def open_pair(bw_dir, stem):
    fwd = os.path.join(bw_dir, f"{stem}.fwd.bw")
    rev = os.path.join(bw_dir, f"{stem}.rev.bw")
    for p in (fwd, rev):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    return pyBigWig.open(fwd), pyBigWig.open(rev)


def check_build(bw, label):
    chroms = bw.chroms()
    key = "chr1" if "chr1" in chroms else ("1" if "1" in chroms else None)
    if key is None:
        print(f"  {label}: no chr1 or 1 in header; chroms look like "
              f"{list(chroms)[:5]}")
        return None, None
    n = chroms[key]
    build = BUILD_BY_CHR1.get(n, f"unknown (chr1 = {n})")
    print(f"  {label}: naming '{key}', chr1 = {n:,} -> {build}")
    return build, key.startswith("chr")


def signal(bw, chrom, start, end, prefix_chr):
    c = chrom if prefix_chr else chrom.replace("chr", "")
    if c not in bw.chroms():
        return 0.0
    L = bw.chroms()[c]
    s, e = max(0, int(start)), min(int(end), L)
    if e <= s:
        return 0.0
    try:
        vals = bw.values(c, s, e, numpy=True)
    except (RuntimeError, OverflowError):
        return 0.0
    vals = np.nan_to_num(vals, nan=0.0)
    return float(np.abs(vals).sum())


def extract(peaks, bw_fwd, bw_rev, half, prefix_chr):
    out = np.zeros(len(peaks))
    for i, (chrom, ctr, strand) in enumerate(
            zip(peaks["chrom"], peaks["center"], peaks["strand"])):
        bw = bw_fwd if strand == "+" else bw_rev
        out[i] = signal(bw, chrom, ctr - half, ctr + half, prefix_chr)
    return out


def strand_convention_check(peaks, bw_fwd, bw_rev, half, prefix_chr, n=400):
    can = peaks[peaks["is_canonical"]]
    plus = can[can["strand"] == "+"].head(n)
    minus = can[can["strand"] == "-"].head(n)
    if len(plus) < 20 or len(minus) < 20:
        print("  too few canonicals to test the strand convention")
        return True
    pf = extract(plus, bw_fwd, bw_fwd, half, prefix_chr).sum()
    pr = extract(plus, bw_rev, bw_rev, half, prefix_chr).sum()
    mf = extract(minus, bw_fwd, bw_fwd, half, prefix_chr).sum()
    mr = extract(minus, bw_rev, bw_rev, half, prefix_chr).sum()
    print(f"  plus-strand canonicals : fwd {pf:,.0f}  rev {pr:,.0f}")
    print(f"  minus-strand canonicals: fwd {mf:,.0f}  rev {mr:,.0f}")
    ok = pf > pr and mr > mf
    print("  convention: " + ("fwd = plus strand, as assumed" if ok else
          "MISMATCHED. Swap fwd and rev, or these are read-strand files."))
    return ok


def usage_fraction(peaks, sig, min_gene_signal):
    df = peaks[["gene"]].copy()
    df["sig"] = sig
    tot = df.groupby("gene")["sig"].transform("sum")
    frac = np.where(tot >= min_gene_signal, df["sig"] / tot.replace(0, np.nan),
                    np.nan)
    return np.asarray(frac, float), np.asarray(tot, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw-dir", required=True)
    ap.add_argument("--peaks", required=True,
                    help="TSV or parquet with gene, chrom, center, strand, is_canonical")
    ap.add_argument("--features", default=None,
                    help="table carrying the U1 density column, joined on gene+center")
    ap.add_argument("--u1-col", default="u1_dens_pa500")
    ap.add_argument("--half-window", type=int, default=50)
    ap.add_argument("--min-gene-signal", type=float, default=20.0)
    ap.add_argument("--min-competitors", type=int, default=3,
                    help="genes with fewer competitors are skipped in per-gene tests")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="report detection rate across window sizes, then exit")
    ap.add_argument("--sweep-windows", default="25,50,100,200,500,1000")
    ap.add_argument("--exclude-crowded", action="store_true",
                    help="drop peaks whose nearest same-gene neighbour is closer "
                         "than 2x half-window, so no read is counted twice")
    ap.add_argument("--pseudocount", default="auto",
                    help="number, or 'auto' = half the median nonzero signal")
    ap.add_argument("--out", default="u1_usage")
    args = ap.parse_args()

    peaks = (pd.read_parquet(args.peaks) if args.peaks.endswith(".parquet")
             else pd.read_csv(args.peaks, sep="\t", low_memory=False))
    need = {"gene", "chrom", "center", "strand", "is_canonical"}
    missing = need - set(peaks.columns)
    if missing:
        sys.exit(f"peaks table missing columns: {sorted(missing)}\n"
                 f"has: {list(peaks.columns)}")
    peaks = peaks.reset_index(drop=True)
    peaks["is_canonical"] = peaks["is_canonical"].astype(bool)
    print(f"peaks: {len(peaks):,} in {peaks['gene'].nunique():,} genes "
          f"({int(peaks['is_canonical'].sum()):,} canonical)")

    print("\n[build check] our peaks are hg38; these files must be too")
    handles, prefix = {}, None
    for cond, stem in COND.items():
        f, r = open_pair(args.bw_dir, stem)
        b, p = check_build(f, f"{cond}.fwd")
        check_build(r, f"{cond}.rev")
        if b not in (None, "hg38"):
            print(f"  WARNING: {cond} looks like {b}, not hg38. "
                  f"Stop and reconcile before going further.")
        try:
            hdrs = [h.header() for h in (f, r)]
            tot = sum(x.get("sumData", 0.0) for x in hdrs)
            cov = sum(x.get("nBasesCovered", 0) for x in hdrs)
            print(f"  {cond}: total signal {tot:,.0f}  bases covered {cov:,}  "
                  f"mean per covered base {tot / max(cov, 1):.2f}")
        except (RuntimeError, TypeError, AttributeError):
            pass
        handles[cond] = (f, r)
        prefix = p if prefix is None else prefix

    print("  Equal totals with unequal bases covered means the signal is")
    print("  redistributed, not shallower. Concentration is the U1 AMO phenotype.")

    print("\n[strand convention]")
    ok = strand_convention_check(peaks, *handles["cAMO"], args.half_window, prefix)
    if not ok:
        print("  Continuing, but delta_usage will be meaningless if this is wrong.")
    if args.sweep:
        gaps = []
        for _, d in peaks.groupby("gene"):
            c = np.sort(d["center"].to_numpy())
            if c.size > 1:
                gaps.append(np.diff(c))
        if gaps:
            gaps = np.concatenate(gaps)
            q = np.percentile(gaps, [5, 25, 50])
            print("\n[peak spacing] distance to the next peak in the same gene")
            print(f"  5th pct {q[0]:,.0f} nt   25th pct {q[1]:,.0f} nt   "
                  f"median {q[2]:,.0f} nt")
            print("  A half-window above half the 5th percentile starts double")
            print("  counting reads into two peaks of the same gene.")
            print(f"  Safe half-window ceiling: about {q[0]/2:,.0f} nt")

        print("\n[window sweep] fraction of peaks with any signal")
        print(f"  {'half-win':>9}  {'cAMO can':>9}  {'cAMO comp':>10}  "
              f"{'U1AMO can':>10}  {'U1AMO comp':>11}  {'genes both':>11}")
        can_m = peaks["is_canonical"].to_numpy()
        for w in [int(x) for x in args.sweep_windows.split(",")]:
            cols = {}
            for cond, (f, r) in handles.items():
                cols[cond] = extract(peaks, f, r, w, prefix) > 0
            both = peaks.assign(
                c=(cols["cAMO"] | cols["U1AMO"]) & can_m,
                k=(cols["cAMO"] | cols["U1AMO"]) & ~can_m,
            ).groupby("gene")[["c", "k"]].any()
            n_both = int((both["c"] & both["k"]).sum())
            print(f"  {w:>9}  {cols['cAMO'][can_m].mean():>9.3f}  "
                  f"{cols['cAMO'][~can_m].mean():>10.3f}  "
                  f"{cols['U1AMO'][can_m].mean():>10.3f}  "
                  f"{cols['U1AMO'][~can_m].mean():>11.3f}  {n_both:>11,}")
        print("\n  Pick the smallest window where 'genes both' plateaus.")
        print("  If every row is near zero the bigWigs are truncated; check")
        print("  stat -c%s against filelist.txt before anything else.")
        return

    if args.check_only:
        return

    if args.exclude_crowded:
        keep = np.ones(len(peaks), bool)
        for g, d in peaks.groupby("gene"):
            idx = d.index.to_numpy()
            c = d["center"].to_numpy()
            o = np.argsort(c)
            idx, c = idx[o], c[o]
            if c.size < 2:
                continue
            nn = np.full(c.size, np.inf)
            nn[:-1] = np.minimum(nn[:-1], np.diff(c))
            nn[1:] = np.minimum(nn[1:], np.diff(c))
            keep[idx[nn < 2 * args.half_window]] = False
        print(f"\n[crowding] dropped {int((~keep).sum()):,} peaks with a neighbour "
              f"closer than {2 * args.half_window} nt "
              f"({100 * (~keep).mean():.1f}%)")
        peaks = peaks[keep].reset_index(drop=True)
        print(f"  remaining: {len(peaks):,} peaks in {peaks['gene'].nunique():,} genes")

    print(f"\n[quantify] center +/- {args.half_window} nt")
    res = {}
    for cond, (f, r) in handles.items():
        s = extract(peaks, f, r, args.half_window, prefix)
        frac, tot = usage_fraction(peaks, s, args.min_gene_signal)
        res[cond] = dict(sig=s, usage=frac, gene_total=tot)
        n_ok = int(np.isfinite(frac).sum())
        print(f"  {cond}: nonzero peaks {int((s > 0).sum()):,}  "
              f"usable after gene floor {n_ok:,}")

    peaks["sig_cAMO"] = res["cAMO"]["sig"]
    peaks["sig_U1AMO"] = res["U1AMO"]["sig"]
    peaks["usage_cAMO"] = res["cAMO"]["usage"]
    peaks["usage_U1AMO"] = res["U1AMO"]["usage"]
    peaks["delta_usage"] = peaks["usage_U1AMO"] - peaks["usage_cAMO"]

    nz = np.concatenate([peaks["sig_cAMO"].to_numpy(), peaks["sig_U1AMO"].to_numpy()])
    nz = nz[nz > 0]
    pc = (float(np.percentile(nz, 10)) if args.pseudocount == "auto" and nz.size
          else float(args.pseudocount) if args.pseudocount != "auto" else 1.0)
    shrunk = float((nz < pc).mean()) if nz.size else 0.0
    print(f"\n[log2OR] pseudocount {pc:.4g}  "
          f"(median nonzero signal {np.median(nz):.4g}; "
          f"{100 * shrunk:.1f}% of nonzero peaks fall below it)")
    if pc > 0.25 * np.median(nz):
        print("  WARNING: pseudocount is large relative to typical signal; "
              "log2OR will be shrunk toward 0.")
    can = peaks[peaks["is_canonical"]].drop_duplicates("gene").set_index("gene")
    for cond in ("cAMO", "U1AMO"):
        peaks[f"can_{cond}"] = peaks["gene"].map(can[f"sig_{cond}"])
    peaks["log2OR"] = (
        np.log2((peaks["sig_U1AMO"] + pc) / (peaks["can_U1AMO"] + pc))
        - np.log2((peaks["sig_cAMO"] + pc) / (peaks["can_cAMO"] + pc)))

    if args.features:
        feat = (pd.read_parquet(args.features) if args.features.endswith(".parquet")
                else pd.read_csv(args.features, sep="\t", low_memory=False))
        if args.u1_col not in feat.columns:
            print(f"\n  '{args.u1_col}' not in features; available u1 columns: "
                  f"{[c for c in feat.columns if 'u1' in c.lower()]}")
        else:
            keys = [k for k in ("gene", "center") if k in feat.columns]
            peaks = peaks.merge(feat[keys + [args.u1_col]], on=keys, how="left")

    # detection diagnostics
    det_c = peaks["sig_cAMO"] > 0
    det_u = peaks["sig_U1AMO"] > 0
    peaks["detected"] = det_c | det_u
    can_m = peaks["is_canonical"]
    print("\n[detection] peaks with any signal")
    for lab, m in (("canonical", can_m), ("competitor", ~can_m)):
        print(f"  {lab:<11} cAMO {det_c[m].mean():.3f}  U1AMO {det_u[m].mean():.3f}  "
              f"either {peaks.loc[m, 'detected'].mean():.3f}  (n {int(m.sum()):,})")

    g = peaks.groupby("gene")
    gene_ok = (g.apply(lambda d: bool((d["detected"] & d["is_canonical"]).any()
                                      and (d["detected"] & ~d["is_canonical"]).any()),
                       include_groups=False))
    analyzable_genes = set(gene_ok[gene_ok].index)
    print(f"  genes with canonical AND >=1 competitor detected: "
          f"{len(analyzable_genes):,} / {peaks['gene'].nunique():,}")

    can_rows = peaks[peaks["is_canonical"]].drop_duplicates("gene").set_index("gene")
    can_both = set(can_rows.index[(can_rows["sig_cAMO"] > 0)
                                  & (can_rows["sig_U1AMO"] > 0)])
    eff = analyzable_genes & can_both
    print(f"  canonical detected in BOTH conditions: {len(can_both):,}")
    print(f"  effective n for log2OR: {len(eff):,} genes")
    peaks["log2OR_informative"] = peaks["gene"].isin(eff)
    if len(analyzable_genes) < 200:
        print("  WARNING: too few analyzable genes for a stable within-gene test.")
        print("  Widen the window (--sweep) or confirm the bigWigs are complete.")

    an = peaks[peaks["gene"].isin(analyzable_genes) & peaks["detected"]].copy()
    print(f"  analyzable peaks: {len(an):,}")

    # positive control
    print("\n[positive control] canonicals must lose usage share under U1 AMO")
    can = an[an["is_canonical"] & an["delta_usage"].notna()]
    com = an[~an["is_canonical"] & an["delta_usage"].notna()]
    if len(can) > 10 and len(com) > 10:
        mc, mm = can["delta_usage"].median(), com["delta_usage"].median()
        fc = float((can["delta_usage"] < 0).mean())
        fm = float((com["delta_usage"] > 0).mean())
        U, p = mannwhitneyu(can["delta_usage"], com["delta_usage"],
                            alternative="less")
        print(f"  canonical  median {mc:+.4f}  frac negative {fc:.3f}  (n {len(can):,})")
        print(f"  competitor median {mm:+.4f}  frac positive {fm:.3f}  (n {len(com):,})")
        print(f"  MWU one-sided p {p:.3g}")
        ok = fc > 0.5 and fm > 0.5 and p < 0.05
        print("  " + ("PASS" if ok else
                      "FAIL. Check strand convention, build, gene mapping, and "
                      "whether the bigWigs are complete."))
    else:
        print("  too few analyzable peaks to run the control")

    # control on the depth-invariant metric
    print("\n[positive control, log2OR] competitors must gain against their own canonical")
    lo = an.loc[~an["is_canonical"], "log2OR"].dropna()
    if len(lo) > 20:
        med = float(lo.median()); frac = float((lo > 0).mean())
        st, p = wilcoxon(lo) if (lo != 0).sum() >= 6 else (np.nan, np.nan)
        print(f"  n {len(lo):,}  median log2OR {med:+.4f}  frac positive {frac:.3f}  "
              f"Wilcoxon p {p:.3g}")
        print("  " + ("PASS" if med > 0 and frac > 0.5 else
                      "FAIL. Competitors are not gaining share under U1 AMO."))
    else:
        print("  too few analyzable competitors")

    # primary test
    rows = []
    METRICS = [("log2OR", "PRIMARY"), ("delta_usage", "secondary")]
    if args.u1_col in peaks.columns:
      for METRIC, TAG in METRICS:
        print(f"\n[{TAG}] {METRIC} vs baseline U1 PAS-proximal density, within gene")
        com = (an[~an["is_canonical"] & an["log2OR_informative"]]
               if METRIC == "log2OR" else an[~an["is_canonical"]])
        sub = com if args.u1_col in com.columns else com.merge(
            peaks[["gene", "center", args.u1_col]], on=["gene", "center"], how="left")
        sub = sub.dropna(subset=[args.u1_col, METRIC])

        rhos = []
        for g, s in sub.groupby("gene"):
            if len(s) < args.min_competitors:
                continue
            r, _ = spearmanr(s[args.u1_col], s[METRIC])
            if np.isfinite(r):
                rhos.append(r)
        rhos = np.array(rhos)
        if rhos.size >= 10:
            nz = rhos != 0
            stat, p = wilcoxon(rhos[nz]) if nz.sum() >= 6 else (np.nan, np.nan)
            print(f"  per-gene Spearman: n_genes {rhos.size:,}  "
                  f"median rho {np.median(rhos):+.3f}  "
                  f"frac positive {np.mean(rhos > 0):.3f}  Wilcoxon p {p:.3g}")
            rows.append(dict(metric=METRIC, test="per_gene_spearman", n=rhos.size,
                             estimate=float(np.median(rhos)), p=p))

        d = sub.copy()
        for c in (args.u1_col, METRIC):
            d[c + "_c"] = d[c] - d.groupby("gene")[c].transform("mean")
        r, p = spearmanr(d[args.u1_col + "_c"], d[METRIC + "_c"])
        print(f"  gene-demeaned pooled: n {len(d):,}  rho {r:+.3f}  p {p:.3g}")
        rows.append(dict(metric=METRIC, test="demeaned_pooled_spearman", n=len(d),
                         estimate=float(r), p=float(p)))

        hi, lo, genes = [], [], []
        for g, s in sub.groupby("gene"):
            if len(s) < 2:
                continue
            med = s[args.u1_col].median()
            a = s.loc[s[args.u1_col] > med, METRIC]
            b = s.loc[s[args.u1_col] <= med, METRIC]
            if len(a) and len(b):
                hi.append(a.mean()); lo.append(b.mean()); genes.append(g)
        if len(hi) >= 10:
            hi, lo = np.array(hi), np.array(lo)
            diff = hi - lo
            nz = diff != 0
            stat, p = wilcoxon(hi[nz], lo[nz]) if nz.sum() >= 6 else (np.nan, np.nan)
            print(f"  U1-high vs U1-low, paired: n_genes {len(hi):,}  "
                  f"median diff {np.median(diff):+.5f}  win-rate "
                  f"{np.mean(diff > 0):.3f}  Wilcoxon p {p:.3g}")
            rows.append(dict(metric=METRIC, test="u1_high_vs_low_paired", n=len(hi),
                             estimate=float(np.median(diff)), p=p))
    else:
        print("  skipped: no U1 density column. Pass --features and --u1-col.")

    print("\n[confound] metric vs gene total signal")
    for METRIC, _ in METRICS:
        d = an[~an["is_canonical"]].dropna(subset=[METRIC])
        tot = d["gene"].map(an.groupby("gene")["sig_cAMO"].sum())
        m = np.isfinite(tot) & (tot > 0)
        if m.sum() > 50:
            r, p = spearmanr(np.log10(tot[m]), d.loc[m, METRIC])
            print(f"  {METRIC:<12} rho vs log10(gene signal) {r:+.3f}  p {p:.3g}")
    print("  A large negative rho means shallow genes drive the result.")

    stats = pd.DataFrame(rows)
    if len(stats):
        m = stats["p"].notna()
        order = stats.loc[m, "p"].rank(method="first")
        stats.loc[m, "p_BH"] = (stats.loc[m, "p"] * m.sum() / order).clip(upper=1)
    peaks.to_csv(f"{args.out}_per_peak.tsv", sep="\t", index=False)
    stats.to_csv(f"{args.out}_stats.tsv", sep="\t", index=False)
    print(f"\nwrote {args.out}_per_peak.tsv and {args.out}_stats.tsv")


if __name__ == "__main__":
    main()