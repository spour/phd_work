#!/usr/bin/env python3
# do our peaks agree with published ones

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

CHROM_NAMES = ["chrom", "chr", "seqnames", "seqname", "chromosome"]
POS_NAMES = ["center", "pos", "position", "start", "site"]
STRAND_NAMES = ["strand", "Strand"]


def _self_id():
    import hashlib
    try:
        h = hashlib.md5(open(__file__, "rb").read()).hexdigest()[:8]
        print(f"[{os.path.basename(__file__)}  build {h}]", file=sys.stderr)
    except OSError:
        pass


def pick(cols, options, what):
    for o in options:
        if o in cols:
            return o
    found = list(cols)[:15]
    sys.exit(f"no {what} column. Looked for {options}. Found: {found}")


def load_table(path):
    if not os.path.exists(path):
        sys.exit(f"{path} does not exist")
    if os.path.getsize(path) == 0:
        sys.exit(f"{path} is empty. If liftover_table.sh made it, that step "
                 f"failed. Rerun it and read the error.")
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False,
                         compression="gzip" if path.endswith(".gz") else None)
    except pd.errors.EmptyDataError:
        sys.exit(f"{path} has no parseable columns. Check the separator:\n"
                 f"  head -1 {path} | cat -A | head -3")
    if "hg38_chrom" in df.columns and "hg38_pos" in df.columns:
        c, p = "hg38_chrom", "hg38_pos"
    else:
        c = pick(df.columns, CHROM_NAMES, "chromosome")
        p = pick(df.columns, POS_NAMES, "position")
    s = pick(df.columns, STRAND_NAMES, "strand")
    posv = pd.to_numeric(df[p], errors="coerce")
    if p in ("start", "hg38_pos") and "end" in df.columns:
        endv = pd.to_numeric(df["end"], errors="coerce")
        w = (endv - posv)
        if w.median() > 1:
            print(f"    peak width: median {w.median():.0f} nt "
                  f"(IQR {w.quantile(.25):.0f}-{w.quantile(.75):.0f}); "
                  f"using midpoint")
            posv = (posv + endv) // 2
    out = pd.DataFrame({"chrom": df[c].astype(str),
                        "pos": posv,
                        "strand": df[s].astype(str)}).dropna()
    out["pos"] = out["pos"].astype(int)
    out["chrom"] = out["chrom"].where(out["chrom"].str.startswith("chr"),
                                      "chr" + out["chrom"])
    out = out[out["strand"].isin(["+", "-"])].reset_index(drop=True)
    out["value"] = np.nan
    print(f"    columns used: {c}, {p}, {s}")
    if "hg38" in os.path.basename(path).lower() and c != "hg38_chrom":
        print("    *** WARNING: filename says hg38 but no hg38_chrom/hg38_pos")
        print("    *** column was found, so these are the ORIGINAL coordinates.")
        print("    *** If the source was hg19 this comparison is meaningless.")
        print(f"    *** check:  head -1 {path} | tr '\\t' '\\n' | tail -4")
    return out


def call_peaks_from_bw(fwd_path, min_value=None, top_n=None, merge_within=25):
    try:
        import pyBigWig
    except ImportError:
        sys.exit("pyBigWig needed to read bigWigs")
    rev_path = fwd_path.replace(".fwd.bw", ".rev.bw")
    if not os.path.exists(rev_path):
        sys.exit(f"expected a matching {rev_path}")
    floor = 0.0 if min_value is None else float(min_value)

    rows = []
    for path, strand in ((fwd_path, "+"), (rev_path, "-")):
        bw = pyBigWig.open(path)
        for chrom in bw.chroms():
            try:
                ivs = bw.intervals(chrom)
            except RuntimeError:
                continue
            if not ivs:
                continue
            keep = [(s, e, abs(v)) for s, e, v in ivs if abs(v) > floor]
            if not keep:
                continue
            b_end, best_v, best_p = keep[0][1], keep[0][2], keep[0][0]
            for s, e, v in keep[1:]:
                if s - b_end <= merge_within:
                    b_end = max(b_end, e)
                    if v > best_v:
                        best_v, best_p = v, s
                else:
                    rows.append((chrom, best_p, strand, best_v))
                    b_end, best_v, best_p = e, v, s
            rows.append((chrom, best_p, strand, best_v))
        bw.close()
    df = pd.DataFrame(rows, columns=["chrom", "pos", "strand", "value"])
    if top_n and len(df) > top_n:
        df = df.nlargest(top_n, "value").reset_index(drop=True)
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    s_ = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (c - s_) / d, (c + s_) / d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def find_first(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def liftover_df(df, chain, exe, label, min_match=0.95):
    if exe is None or chain is None:
        sys.exit(f"{label}: asked to lift but liftOver binary or chain not found.\n"
                 f"  pass --liftover-bin and --chain")
    tmp = tempfile.mkdtemp(prefix="lo_")
    src, dst, unm = (os.path.join(tmp, x) for x in ("in.bed", "out.bed", "un.bed"))
    df = df.reset_index(drop=True)
    with open(src, "w") as fh:
        for i, (c, pos, st) in enumerate(zip(df["chrom"], df["pos"], df["strand"])):
            fh.write(f"{c}\t{int(pos)-1}\t{int(pos)}\t{i}\t0\t{st}\n")
    r = subprocess.run([exe, f"-minMatch={min_match}", src, chain, dst, unm],
                       capture_output=True, text=True)
    if not os.path.exists(dst):
        shutil.rmtree(tmp, ignore_errors=True)
        err = r.stderr[-500:]
        sys.exit(f"{label}: liftOver produced no output\n{err}")
    out = pd.read_csv(dst, sep="\t", header=None,
                      names=["chrom", "s", "pos", "idx", "sc", "strand"])
    shutil.rmtree(tmp, ignore_errors=True)
    dup = out["idx"].duplicated(keep=False)
    n_multi = int(out.loc[dup, "idx"].nunique())
    out = out.loc[~dup]
    kept = df.loc[out["idx"].to_numpy()].copy()
    kept["chrom"] = out["chrom"].to_numpy()
    kept["pos"] = out["pos"].to_numpy()
    lost = len(df) - len(kept)
    print(f"    lifted hg19 -> hg38: {len(kept):,} of {len(df):,} "
          f"({100*lost/max(len(df),1):.2f}% lost, {n_multi} multi-mapped)")
    return kept.reset_index(drop=True)


def nearest(query, target):
    out = np.full(len(query), np.inf)
    idx = defaultdict(list)
    for c, s, p in zip(target["chrom"], target["strand"], target["pos"]):
        idx[(c, s)].append(p)
    idx = {k: np.sort(np.asarray(v)) for k, v in idx.items()}
    for (c, s), sub in query.groupby(["chrom", "strand"], sort=False):
        arr = idx.get((c, s))
        if arr is None or arr.size == 0:
            continue
        p = sub["pos"].to_numpy()
        i = np.searchsorted(arr, p)
        left = np.where(i > 0, np.abs(p - arr[np.clip(i - 1, 0, arr.size - 1)]), np.inf)
        right = np.where(i < arr.size, np.abs(arr[np.clip(i, 0, arr.size - 1)] - p), np.inf)
        out[sub.index.to_numpy()] = np.minimum(left, right)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--external", nargs="+", required=True,
                    help="name=path pairs; path is a table or a .fwd.bw")
    ap.add_argument("--tolerances", default="25,50,100")
    ap.add_argument("--bw-min-value", type=float, default=None)
    ap.add_argument("--bw-top-n", type=int, default=None,
                    help="keep the N strongest peaks per bigWig, so two "
                         "bigWigs are compared at matched set size")
    ap.add_argument("--hg19", nargs="*", default=[],
                    help="names of external sets that are hg19 and need lifting")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--liftover-bin", default=None)
    ap.add_argument("--shuffle", type=int, default=50,
                    help="displaced-control rounds for the empirical p value")
    ap.add_argument("--out", default="overlap")
    args = ap.parse_args()

    chain = find_first(args.chain, "external/chain/hg19ToHg38.over.chain.gz",
                       "chain/hg19ToHg38.over.chain.gz",
                       os.path.expandvars("$EXTERNAL_ROOT/chain/hg19ToHg38.over.chain.gz"))
    lo_exe = find_first(args.liftover_bin, "external/bin/liftOver",
                        "bin/liftOver", shutil.which("liftOver"))
    if args.hg19:
        print(f"lifting to hg38: {', '.join(args.hg19)}")
        print(f"  chain    {chain}")
        print(f"  liftOver {lo_exe}")
    _self_id()

    ours = (pd.read_parquet(args.peaks) if args.peaks.endswith(".parquet")
            else pd.read_csv(args.peaks, sep="\t", low_memory=False))
    if "is_canonical" not in ours.columns:
        sys.exit("our peak table needs is_canonical")
    cc = pick(ours.columns, CHROM_NAMES, "chromosome")
    pp = pick(ours.columns, POS_NAMES, "position")
    ss = pick(ours.columns, STRAND_NAMES, "strand")
    q = pd.DataFrame({
        "chrom": ours[cc].astype(str), "pos": ours[pp].astype(int),
        "strand": ours[ss].astype(str),
        "gene": ours["gene"] if "gene" in ours.columns else "NA",
        "is_canonical": ours["is_canonical"].astype(bool)}).reset_index(drop=True)
    n_can = int(q["is_canonical"].sum())
    print(f"ours: {len(q):,} peaks, {n_can:,} canonical, {len(q)-n_can:,} "
          f"competitor, {q['gene'].nunique():,} genes")

    tols = [int(x) for x in args.tolerances.split(",")]
    ext_all, summary = [], []
    for spec in args.external:
        if "=" not in spec:
            sys.exit(f"expected name=path, got {spec}")
        name, path = spec.split("=", 1)
        print(f"\n{name}: {path}")
        if path.endswith(".bw"):
            ext = call_peaks_from_bw(path, args.bw_min_value, args.bw_top_n)
            how = (f"top {args.bw_top_n:,} by signal" if args.bw_top_n
                   else f"value > {args.bw_min_value}")
            print(f"    called {len(ext):,} peaks ({how})")
        else:
            ext = load_table(path)
            print(f"    {len(ext):,} sites")
        if name in args.hg19:
            ext = liftover_df(ext, chain, lo_exe, name)
        if len(ext) == 0:
            sys.exit(f"{name} has no usable sites")
        ext["dataset"] = name
        ext_all.append(ext)

        d = nearest(q, ext)
        q[f"dist_{name}"] = d
        d_rev_can = nearest(ext, q[q["is_canonical"]].reset_index(drop=True))
        d_rev_com = nearest(ext, q[~q["is_canonical"]].reset_index(drop=True))
        for t in tols:
            hit = d <= t
            q[f"hit_{name}_{t}"] = hit
            for lab, m in (("canonical", q["is_canonical"].to_numpy()),
                           ("competitor", ~q["is_canonical"].to_numpy())):
                rev = (d_rev_can if lab == "canonical" else d_rev_com) <= t
                summary.append(dict(
                    dataset=name, tolerance=t, klass=lab,
                    n_ours=int(m.sum()), n_hit=int((m & hit).sum()),
                    frac_ours_hit=float((m & hit).sum() / max(m.sum(), 1)),
                    n_external=len(ext),
                    n_external_hit=int(rev.sum()),
                    frac_external_hit=float(rev.mean()),
                    genes_hit=int(q.loc[m & hit, "gene"].nunique())))

    s = pd.DataFrame(summary)
    mid = tols[len(tols) // 2]
    TOL_MAIN = mid

    print("\nfraction of OUR peaks with a match")
    print(s.pivot_table(index=["dataset", "tolerance"], columns="klass",
                        values="frac_ours_hit").round(4).to_string())

    print("\nfraction of EXTERNAL sites landing on our peaks "
          "(not capped by our set size)")
    print(s.pivot_table(index=["dataset", "tolerance"], columns="klass",
                        values="frac_external_hit").round(4).to_string())

    # competitor-only, with a displaced-control null
    rng = np.random.default_rng(0)
    comp_q = q[~q["is_canonical"]].reset_index(drop=True)
    print(f"\n{'='*72}")
    print("COMPETITOR OVERLAP".center(72))
    print(f"{'='*72}")
    print(f"n competitors considered: {len(comp_q):,} "
          f"in {comp_q['gene'].nunique():,} genes")
    print(f"null: each external set displaced by a random 10-100 kb offset, "
          f"{args.shuffle} rounds")
    print("'% of dataset' is the fraction of that dataset's own sites that fall "
          "on a competitor\n")
    hdr = (f"{'dataset':<11}{'tol':>5}{'sites':>9}{'in competitors':>16}"
           f"{'% of dataset':>14}{'95% CI':>16}{'fold':>7}{'p':>10}")
    print(hdr); print("-" * len(hdr))
    crows = []
    for name, ext in zip([e["dataset"].iloc[0] for e in ext_all], ext_all):
        ext = ext.reset_index(drop=True)
        d = nearest(comp_q, ext)
        d_rev = nearest(ext, comp_q)
        for t in tols:
            obs = int((d <= t).sum())
            k_ext = int((d_rev <= t).sum())
            f_ext = k_ext / len(ext)
            lo_e, hi_e = wilson(k_ext, len(ext))
            null = np.empty(args.shuffle)
            for i in range(args.shuffle):
                off = int(rng.integers(10_000, 100_000)) * int(rng.choice([-1, 1]))
                e2 = ext.copy()
                e2["pos"] = e2["pos"] + off
                null[i] = (nearest(comp_q, e2) <= t).sum()
            exp = float(null.mean())
            fold = obs / exp if exp > 0 else float("inf")
            pval = (1 + int((null >= obs).sum())) / (args.shuffle + 1)
            ci = f"{100*lo_e:.2f}-{100*hi_e:.2f}"
            print(f"{name:<11}{t:>5}{len(ext):>9,}{k_ext:>16,}"
                  f"{100*f_ext:>13.2f}%{ci:>16}{fold:>7.1f}{pval:>10.4g}")
            crows.append(dict(dataset=name, tolerance=t, n_external=len(ext),
                              n_competitors=len(comp_q),
                              n_external_in_competitors=k_ext,
                              pct_of_dataset=100 * f_ext,
                              ci_lo=100 * lo_e, ci_hi=100 * hi_e,
                              n_competitors_hit=obs,
                              pct_of_competitors=100 * obs / len(comp_q),
                              expected_shuffled=exp, fold_enrichment=fold,
                              p_empirical=pval,
                              p_floor=1 / (args.shuffle + 1)))
    pd.DataFrame(crows).to_csv(f"{args.out}_competitor.tsv", sep="\t", index=False)
    print(f"\np is empirical and floors at {1/(args.shuffle+1):.3g} "
          f"with {args.shuffle} rounds; raise --shuffle for a finer floor.")

    if len(ext_all) > 1:
        print(f"\nagreement between the external sets, tolerance {mid} nt")
        print("(fraction of the row set landing on the column set)")
        ours_c = q[~q["is_canonical"]][["chrom", "pos", "strand"]].copy()
        ours_c["dataset"] = "competitor"
        ours_k = q[q["is_canonical"]][["chrom", "pos", "strand"]].copy()
        ours_k["dataset"] = "canonical"
        ext_all = ext_all + [ours_c, ours_k]
        names = [e["dataset"].iloc[0] for e in ext_all]
        M = pd.DataFrame(index=names, columns=names, dtype=float)
        pair_rows = []
        for a, ea in zip(names, ext_all):
            for b, eb in zip(names, ext_all):
                if a == b:
                    M.loc[a, b] = 1.0
                    continue
                dd = nearest(ea.reset_index(drop=True), eb.reset_index(drop=True))
                k = int((dd <= mid).sum())
                f = k / len(ea)
                f0 = float((dd == 0).mean())
                lo, hi = wilson(k, len(ea))
                M.loc[a, b] = f
                pair_rows.append(dict(row=a, col=b, tolerance=mid,
                                      n_row=len(ea), n_col=len(eb), n_hit=k,
                                      frac_row_in_col=f, ci_lo=lo, ci_hi=hi,
                                      frac_exact_match=f0))
        print(M.round(4).to_string())
        pd.DataFrame(pair_rows).to_csv(f"{args.out}_external_pairwise.tsv",
                                       sep="\t", index=False)
        print("\nexact-coordinate agreement (distance 0), same layout")
        M0 = pd.DataFrame(index=names, columns=names, dtype=float)
        for r in pair_rows:
            M0.loc[r["row"], r["col"]] = r["frac_exact_match"]
        for n_ in names:
            M0.loc[n_, n_] = 1.0
        print(M0.round(4).to_string())
        print("\nA large gap between the two matrices means peak-boundary")
        print("jitter, not disagreement: 3'-seq peaks are ~40 nt wide, so the")
        print("same site gets different midpoints in different peak calls.")
    print(f"\n{'='*66}")
    print(f"HEADLINE, tolerance {TOL_MAIN} nt".center(66))
    print(f"{'='*66}")
    print("Are external sites more likely to land on a competitor than on a")
    print("canonical end? Odds ratio > 1 means yes.\n")
    hdr = (f"{'dataset':<10}{'sites':>8}{'%canon':>9}{'%compet':>9}"
           f"{'odds ratio':>12}{'95% CI':>18}{'p':>11}")
    print(hdr); print("-" * len(hdr))
    enr = []
    for name in [e["dataset"].iloc[0] for e in ext_all]:
        r = s[(s.dataset == name) & (s.tolerance == TOL_MAIN)]
        if len(r) != 2:
            continue
        rc = r[r.klass == "canonical"].iloc[0]
        rk = r[r.klass == "competitor"].iloc[0]
        a, b = rk.n_hit, rk.n_ours - rk.n_hit
        c_, d = rc.n_hit, rc.n_ours - rc.n_hit
        try:
            from scipy.stats import fisher_exact
            odds, pv = fisher_exact([[a, b], [c_, d]])
        except Exception:
            odds, pv = float("nan"), float("nan")
        if min(a, b, c_, d) > 0:
            se = np.sqrt(1 / a + 1 / b + 1 / c_ + 1 / d)
            lo, hi = np.exp(np.log(odds) - 1.96 * se), np.exp(np.log(odds) + 1.96 * se)
            ci = f"{lo:.2f} - {hi:.2f}"
        else:
            lo = hi = float("nan"); ci = "undefined (zero cell)"
        print(f"{name:<10}{rk.n_external:>8,}{100*rc.frac_ours_hit:>8.2f}%"
              f"{100*rk.frac_ours_hit:>8.2f}%{odds:>12.2f}{ci:>18}{pv:>11.2g}")
        enr.append(dict(dataset=name, tolerance=TOL_MAIN, n_external=rk.n_external,
                        n_canonical_hit=c_, n_competitor_hit=a,
                        pct_canonical=100 * rc.frac_ours_hit,
                        pct_competitor=100 * rk.frac_ours_hit,
                        odds_ratio=odds, ci_lo=lo, ci_hi=hi, p=pv))
    pd.DataFrame(enr).to_csv(f"{args.out}_enrichment.tsv", sep="\t", index=False)
    print("\nOR > 1: the set is iCPA-specific and agrees with your competitor class.")
    print("OR < 1: the set sits on canonical 3' ends instead.")
    print("Compare two conditions of one experiment by the ratio of their ORs.")

    print("\nRead each row across: competitor rate above canonical rate means")
    print("the external set is iCPA-specific and agrees with your classes.")
    print("Similar rates mean it is finding 3' ends generically.")

    s.to_csv(f"{args.out}_summary.tsv", sep="\t", index=False)
    q.to_csv(f"{args.out}_per_peak.tsv", sep="\t", index=False)
    pd.concat(ext_all, ignore_index=True).to_csv(
        f"{args.out}_external.tsv", sep="\t", index=False)
    print(f"\nwrote {args.out}_summary.tsv, _per_peak.tsv, _external.tsv"
          + ", _enrichment.tsv, _competitor.tsv"
          + (", _external_pairwise.tsv" if len(ext_all) > 1 else ""))


if __name__ == "__main__":
    main()