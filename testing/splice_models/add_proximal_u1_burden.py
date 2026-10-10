#!/usr/bin/env python3
# standalone telescripting-feature experiment
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-9
TRAIN_CHROMS = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                "chr10", "chr11", "chr12", "chr14", "chr15", "chr16", "chr17"}
VAL_CHROMS = {"chr3", "chr13"}
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
TAUS = (200, 400)
PROX_NT = 1000
PAD = 12


class GenomeFasta:
    def __init__(self, path):
        try:
            import pysam
            self.fa = pysam.FastaFile(path); self.engine = "pysam"
            self.refs = set(self.fa.references)
        except Exception:
            from pyfaidx import Fasta
            self.fa = Fasta(path, sequence_always_upper=True); self.engine = "pyfaidx"
            self.refs = set(self.fa.keys())

    def fetch(self, chrom, start, end, strand="+"):
        c = chrom if chrom in self.refs else chrom.lstrip("chr")
        if c not in self.refs:
            return None
        start = max(0, start)
        seq = (self.fa.fetch(c, start, end).upper() if self.engine == "pysam"
               else str(self.fa[c][start:end]).upper())
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        return seq


def load_u1_lookup(path):
    arr = np.load(path, allow_pickle=True)
    return arr.item() if arr.dtype == object else dict(arr)


def scan_u1(seq, u1_lookup):
    n = len(seq) - 9 + 1
    if n <= 0:
        return np.array([], dtype=np.float32)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        k = seq[i:i + 9]
        if "N" in k:
            continue
        v = u1_lookup.get(k)
        if v is not None:
            out[i] = -float(v)
    return out


def fetch_upstream(fa, chrom, center, strand, up=PROX_NT, pad=PAD):
    if strand == "+":
        gstart, gend = center - up, center + pad
    else:
        gstart, gend = center - pad, center + up
    return fa.fetch(chrom, gstart, gend, strand=strand)


def burden_features(seq, u1_lookup, up=PROX_NT, taus=TAUS):
    out = {f"prox_u1_sum_t{t}": np.nan for t in taus}
    out["prox_u1_max"] = np.nan
    if seq is None or len(seq) < up:
        return out
    u1 = scan_u1(seq, u1_lookup)
    if u1.size == 0:
        return out
    p = np.arange(u1.size)
    upstream = p < up
    d = (up - p[upstream]).astype(float)
    vals = u1[upstream]
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    d, vals = d[finite], vals[finite]
    bg = np.percentile(vals, 10)
    excess = np.maximum(vals - bg, 0.0)
    for t in taus:
        w = np.exp(-d / float(t))
        out[f"prox_u1_sum_t{t}"] = float(np.sum(excess * w))
    out["prox_u1_max"] = float(np.max(excess * np.exp(-d / 300.0)))
    return out


def read_pc_genes(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["chrom", "start", "end", "name", "score", "strand"])
    df["gene"] = df["name"].astype(str)
    if df["gene"].str.contains("_").any():
        df["gene"] = df["gene"].str.split("_").str[-1]
    return df.drop_duplicates("gene", keep="first")


def gene_baseline(fa, u1_lookup, row, n_samples, rng, up=PROX_NT):
    margin = up + PAD + 50
    lo, hi = int(row["start"]) + margin, int(row["end"]) - margin
    if hi <= lo:
        return None
    keys = [f"prox_u1_sum_t{t}" for t in TAUS] + ["prox_u1_max"]
    acc = {k: [] for k in keys}
    for pos in rng.integers(lo, hi, size=n_samples):
        seq = fetch_upstream(fa, row["chrom"], int(pos), row["strand"], up=up)
        f = burden_features(seq, u1_lookup, up=up)
        for k in keys:
            if np.isfinite(f[k]):
                acc[k].append(f[k])
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}


def p_at_1(df, prob):
    d = df.copy(); d["p"] = prob
    hits = n = 0
    for _, g in d.groupby("gene", sort=False):
        y = g["y"].to_numpy().astype(int)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        n += 1
        hits += int(y[int(np.argmax(g["p"].to_numpy()))] == 1)
    return (hits / n if n else np.nan), n


def fit_eval(Xtr, ytr, Xte, dte, seed=42, class_weight="balanced"):
    clf = HistGradientBoostingClassifier(
        max_iter=800, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
        class_weight=class_weight, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=seed)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    y = dte["y"].to_numpy().astype(int)
    pa, npaired = p_at_1(dte, p)
    return {"AUPRC": average_precision_score(y, p), "P@1": pa,
            "AUROC": roc_auc_score(y, p), "n_paired": npaired}


def slope(y, x):
    ok = np.isfinite(x)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return np.nan
    xs = (x[ok] - x[ok].mean()) / (x[ok].std() + EPS)
    return float(LogisticRegression(C=1e6, max_iter=2000).fit(xs.reshape(-1, 1), y[ok]).coef_[0, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--out_tsv", required=True, help="features with the new burden columns added")
    ap.add_argument("--n_baseline_samples", type=int, default=40)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=2000)
    args = ap.parse_args()
    cw = None if args.class_weight == "none" else "balanced"

    df = pd.read_csv(args.features_tsv, sep="\t")
    need = {"gene", "chrom", "center", "strand", "y"}
    if not need.issubset(df.columns):
        sys.exit(f"features_tsv missing required columns: {need - set(df.columns)}")
    fa = GenomeFasta(args.reference_fasta)
    u1 = load_u1_lookup(args.u1_lookup)
    genes = read_pc_genes(args.pc_genes_bed).set_index("gene")

    burden_keys = [f"prox_u1_sum_t{t}" for t in TAUS] + ["prox_u1_max"]

    print(f"[burden] scoring {len(df):,} labeled sites", file=sys.stderr)
    t0 = time.time()
    site_feats = []
    for i, r in enumerate(df.itertuples(index=False)):
        seq = fetch_upstream(fa, r.chrom, int(r.center), r.strand)
        site_feats.append(burden_features(seq, u1))
        if (i + 1) % args.log_every == 0:
            print(f"[burden]   {i+1:,}/{len(df):,}  {(time.time()-t0)/60:.1f}m", file=sys.stderr)
    for k in burden_keys:
        df[k] = [f[k] for f in site_feats]

    rng = np.random.default_rng(args.seed)
    uniq = df["gene"].unique()
    print(f"[burden] computing baselines for {len(uniq):,} genes "
          f"({args.n_baseline_samples} samples each)", file=sys.stderr)
    base = {}
    for j, g in enumerate(uniq):
        if g in genes.index:
            base[g] = gene_baseline(fa, u1, genes.loc[g], args.n_baseline_samples, rng)
        if (j + 1) % 500 == 0:
            print(f"[burden]   {j+1:,}/{len(uniq):,} genes  {(time.time()-t0)/60:.1f}m", file=sys.stderr)
    for k in burden_keys:
        bvals = df["gene"].map(lambda g: (base.get(g) or {}).get(k, np.nan))
        df[f"{k}__wg"] = df[k] - bvals

    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[burden] wrote {args.out_tsv}", file=sys.stderr)

    df["chrom_n"] = df["chrom"].astype(str).str.strip().apply(
        lambda c: ("chr" + c[3:].upper()) if c.lower().startswith("chr") else "chr" + c.upper())
    tr = df["chrom_n"].isin(TRAIN_CHROMS)
    te = df["chrom_n"].isin(TEST_CHROMS)
    ytr = df.loc[tr, "y"].to_numpy().astype(int)
    dte = df.loc[te].reset_index(drop=True)

    existing = [c for c in df.columns
                if (c.startswith("bin") and (c.endswith("_mean") or c.endswith("_max")))
                or c.endswith("__wg")]
    existing = [c for c in existing if not c.startswith("prox_u1")]
    raw_b = burden_keys
    wg_b = [f"{k}__wg" for k in burden_keys]

    feature_sets = {
        "base": existing,
        "+raw_burden": existing + raw_b,
        "+wg_burden": existing + wg_b,
        "+both": existing + raw_b + wg_b,
    }
    print(f"\n[burden] test-chromosome performance (native NaN, class_weight={cw}):", file=sys.stderr)
    print(f"  {'model':<14s} {'AUPRC':>7s} {'P@1':>7s} {'AUROC':>7s} {'n_paired':>9s}", file=sys.stderr)
    results = []
    for name, cols in feature_sets.items():
        m = fit_eval(df.loc[tr, cols].to_numpy(), ytr, dte[cols].to_numpy(), dte, args.seed, cw)
        results.append({"model": name, **m})
        print(f"  {name:<14s} {m['AUPRC']:>7.4f} {m['P@1']:>7.4f} {m['AUROC']:>7.4f} {m['n_paired']:>9d}",
              file=sys.stderr)
    pd.DataFrame(results).to_csv(args.out_tsv.replace(".tsv", "_metrics.tsv"), sep="\t", index=False)

    print(f"\n[burden] sign of the proximal burden (expected within < 0 = telescripting):", file=sys.stderr)
    yte = dte["y"].to_numpy().astype(int)
    for k in burden_keys:
        x = dte[k].to_numpy(float)
        gm = dte.groupby("gene")[k].transform("mean").to_numpy(float)
        wg = dte[f"{k}__wg"].to_numpy(float)
        print(f"  {k:<16s} pooled={slope(yte, x):+.3f}  between={slope(yte, gm):+.3f}  "
              f"within={slope(yte, x-gm):+.3f}  wg={slope(yte, wg):+.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()