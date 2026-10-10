#!/usr/bin/env python
# 20 gene exemplars, real end against a competitor of the same gene, with 3'ss

import os
import sys
import csv
import math
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

WIN = 2000
U1_BIN = 50
U1_THR = 4.0
ME_THR = 8.0
U1_CELL = "up_500|density_per_kb@4"
NPICK = 20
SEED = 42


def log(m):
    print(m, file=sys.stderr, flush=True)


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


_G = {}


def _init(fasta, u1_lookup):
    from extract_splicing_features import init_maxentscan, GenomeFasta, load_u1_lookup
    o = init_maxentscan()
    _G["s5"], _G["s3"] = o[0], o[1]
    _G["fa"] = GenomeFasta(fasta)
    u1 = load_u1_lookup(u1_lookup)
    L = len(next(iter(u1)))
    MAP = np.full(256, -1, np.int64)
    for i, ch in enumerate("ACGT"):
        MAP[ord(ch)] = i
    dense = np.full(4 ** L, np.nan, np.float32)
    powv = (4 ** np.arange(L - 1, -1, -1)).astype(np.int64)
    for kmer, val in u1.items():
        idx, bad = 0, False
        for k, ch in enumerate(kmer):
            c = MAP[ord(ch)]
            if c < 0:
                bad = True
                break
            idx += int(c) * int(powv[k])
        if not bad:
            dense[idx] = -val
    _G.update(MAP=MAP, U1LEN=L, powv=powv, dense=dense)


def _u1_full(seq):
    MAP, L, powv, dense = _G["MAP"], _G["U1LEN"], _G["powv"], _G["dense"]
    arr = MAP[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]
    n = len(arr)
    if n < L:
        return np.empty(0)
    codes = np.zeros(n - L + 1, np.int64)
    bad = np.zeros(n - L + 1, bool)
    for k in range(L):
        c = arr[k:k + (n - L + 1)]
        codes += c * powv[k]
        bad |= (c < 0)
    return np.where(bad, np.nan, dense[np.clip(codes, 0, len(dense) - 1)])


def scan_peak(chrom, center, strand):
    L = _G["U1LEN"]
    if strand == "+":
        glo, ghi = center - WIN, center + L
    else:
        glo, ghi = center - L, center + WIN
    if glo < 0:
        return [], [], []
    seq = _G["fa"].fetch(chrom, glo, ghi, strand=strand)
    if not seq or len(seq) < 60:
        return [], [], []
    seq = seq.upper()
    n = len(seq)
    arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    s3, s5 = _G["s3"], _G["s5"]
    acc, don = [], []
    for i in np.where((arr[:-1] == 65) & (arr[1:] == 71))[0]:
        p = int(i) + 2
        if p - 20 < 0 or p + 3 > n:
            continue
        try:
            x = s3(seq[p - 20:p + 3])
            x = float(np.nanmax(np.asarray(x, float))) if not np.isscalar(x) else float(x)
        except Exception:
            continue
        if np.isfinite(x) and x >= ME_THR:
            d = -WIN + p
            if -WIN <= d <= 0:
                acc.append((d, x))
    for i in np.where((arr[:-1] == 71) & (arr[1:] == 84))[0]:
        p = int(i)
        if p - 3 < 0 or p + 6 > n:
            continue
        try:
            x = s5(seq[p - 3:p + 6])
            x = float(np.nanmax(np.asarray(x, float))) if not np.isscalar(x) else float(x)
        except Exception:
            continue
        if np.isfinite(x) and x >= ME_THR:
            d = -WIN + p
            if -WIN <= d <= 0:
                don.append((d, x))
    v = _u1_full(seq)
    d = -WIN + np.arange(v.size, dtype=float)
    ok = np.isfinite(v) & (d >= -WIN) & (d <= 0)
    d, v = d[ok], v[ok]
    edges = np.arange(-WIN, 1, U1_BIN)
    b = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    strong = (v >= U1_THR).astype(float)
    cnt = np.bincount(b, weights=strong, minlength=len(edges) - 1)
    tot = np.bincount(b, minlength=len(edges) - 1).astype(float)
    dens = np.where(tot > 0, cnt / np.maximum(tot, 1) * 1000.0, np.nan)
    u1b = [((edges[j] + edges[j + 1]) / 2.0, float(dens[j])) for j in range(len(dens))
           if np.isfinite(dens[j])]
    return acc, don, u1b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--u1_cache", default="u1_grid_cache.npz")
    ap.add_argument("--fasta", default="../train/data/hg38.fa")
    ap.add_argument("--u1_lookup", default="../train/splice_train/data/u1_lookup.npy")
    a = ap.parse_args()
    for p in (a.pos, a.u1_cache, a.fasta, a.u1_lookup):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)

    with open(a.pos, newline="") as fh:
        R = list(csv.DictReader(fh, delimiter="\t"))
    genes = [r["gene"] for r in R]
    y = np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8)
    anchor = np.array([fnum(r["d_anchor_to_pa"]) for r in R])

    z = np.load(a.u1_cache, allow_pickle=True)
    ucols = [str(c) for c in z["cols"]]
    if list(z["gene"]) != genes:
        sys.exit("row alignment mismatch: %s vs %s" % (a.u1_cache, a.pos))
    u1 = z["M"][:, ucols.index(U1_CELL)].astype(float)

    # per-gene contrast
    by = defaultdict(lambda: ([], []))
    for i in range(len(genes)):
        by[genes[i]][0 if y[i] == 1 else 1].append(i)
    rows = []
    for g, (ci, ki) in by.items():
        if not ci or not ki:
            continue
        cu = np.nanmean(u1[ci])
        ku = np.nanmean(u1[ki])
        if not (np.isfinite(cu) and np.isfinite(ku)):
            continue
        rows.append(dict(gene=g, contrast=float(ku - cu), n_can=len(ci), n_comp=len(ki),
                         can_idx=ci, comp_idx=ki))
    rows.sort(key=lambda r: r["gene"])
    contrasts = np.array([r["contrast"] for r in rows])
    log("paired genes %d; contrast (competitor minus real end, sites/kb) "
        "min %.1f  p25 %.1f  median %.1f  p75 %.1f  max %.1f"
        % (len(rows), contrasts.min(), np.percentile(contrasts, 25),
           np.median(contrasts), np.percentile(contrasts, 75), contrasts.max()))
    log("  genes with a positive contrast: %d (%.1f%%)"
        % (int((contrasts > 0).sum()), 100 * (contrasts > 0).mean()))

    # eligibility: clean architecture, everything inside the drawn window
    def eligible(r):
        if r["n_can"] != 1 or r["n_comp"] < 1:
            return None
        ci = r["can_idx"][0]
        if not (np.isfinite(anchor[ci]) and 0 < anchor[ci] <= WIN):
            return None
        ks = [k for k in r["comp_idx"]
              if np.isfinite(anchor[k]) and 0 < anchor[k] <= WIN and np.isfinite(u1[k])]
        if not ks:
            return None
        target = float(np.nanmean(u1[r["comp_idx"]]))
        k = min(ks, key=lambda i: (abs(u1[i] - target), genes[i], i))
        return ci, k

    elig = [(j, e) for j, e in ((j, eligible(r)) for j, r in enumerate(rows))
            if e is not None]
    log("eligible genes %d of %d paired" % (len(elig), len(rows)))
    if len(elig) < NPICK:
        sys.exit("only %d eligible genes, need %d" % (len(elig), NPICK))
    rng = np.random.default_rng(SEED)
    draw = sorted(rng.choice(len(elig), size=NPICK, replace=False).tolist())
    picked = []
    for di in draw:
        j, (ci, k) = elig[di]
        emp = float((contrasts < contrasts[j]).mean() * 100)
        picked.append(dict(gene=rows[j]["gene"], actual_pct=emp,
                           contrast=contrasts[j],
                           n_comp=rows[j]["n_comp"], can_idx=ci, comp_idx=k,
                           can_u1=float(u1[ci]), comp_u1=float(u1[k]),
                           comp_mean_u1=float(np.nanmean(u1[rows[j]["comp_idx"]]))))
    log("drew %d genes with seed %d" % (len(picked), SEED))

    with open("exemplar_genes.tsv", "w") as fh:
        k = ["gene", "label", "seed", "actual_pct", "contrast",
             "n_competitors", "canonical_u1", "competitor_u1", "competitor_mean_u1",
             "drawn_pair_contrast",
             "canonical_center", "competitor_center", "chrom", "strand"]
        fh.write("\t".join(k) + "\n")
        for p in picked:
            ci, kk = p["can_idx"], p["comp_idx"]
            p["label"] = p["gene"]
            fh.write("\t".join(str(v) for v in [
                p["gene"], p["label"], SEED, "%.1f" % p["actual_pct"],
                "%.3f" % p["contrast"], p["n_comp"],
                "%.3f" % p["can_u1"], "%.3f" % p["comp_u1"], "%.3f" % p["comp_mean_u1"],
                "%.3f" % (p["comp_u1"] - p["can_u1"]),
                R[ci]["center"], R[kk]["center"], R[ci]["chrom"], R[ci]["strand"]]) + "\n")
    log("wrote exemplar_genes.tsv")

    # scan the 24 drawn peaks
    _init(a.fasta, a.u1_lookup)
    trk, ub = [], []
    for p in picked:
        for cls, idx in (("Real end", p["can_idx"]), ("Competitor", p["comp_idx"])):
            r = R[idx]
            acc, don, u1b = scan_peak(r["chrom"], int(float(r["center"])), r["strand"])
            for d, s in acc:
                trk.append((p["gene"], p["label"], p["actual_pct"], cls, "acceptor", d, s))
            for d, s in don:
                trk.append((p["gene"], p["label"], p["actual_pct"], cls, "donor", d, s))
            for d, s in u1b:
                ub.append((p["gene"], p["label"], p["actual_pct"], cls, d, s))
            log("  %-14s %-11s 3'ss>=8 %3d  5'ss>=8 %3d  U1 bins %d"
                % (p["label"], cls, len(acc), len(don), len(u1b)))
    with open("exemplar_tracks.tsv", "w") as fh:
        fh.write("gene\tlabel\tpct\tclass\tkind\tpos\tscore\n")
        for t in trk:
            fh.write("%s\t%s\t%.1f\t%s\t%s\t%g\t%.4f\n" % t)
    with open("exemplar_u1.tsv", "w") as fh:
        fh.write("gene\tlabel\tpct\tclass\tpos\tu1_density\n")
        for t in ub:
            fh.write("%s\t%s\t%.1f\t%s\t%g\t%.4f\n" % t)
    log("wrote exemplar_tracks.tsv (%d marks), exemplar_u1.tsv (%d bins)" % (len(trk), len(ub)))

    print("=" * 104)
    print("EXEMPLAR SELECTION  %d genes drawn at random, seed %d" % (NPICK, SEED))
    print("=" * 104)
    print("  eligibility: 1 canonical, >=1 competitor, anchor within %d nt of the pA" % WIN)
    print("  drawn competitor: the one whose upstream U1 is nearest the gene's competitor mean")
    print("  contrast = mean competitor upstream U1 minus real-end upstream U1")
    print("             (pA-500 -> pA, sites/kb, negated dG >= %.1f), the panel F/G/J cell;" % U1_THR)
    print("             it is reported for audit only and played no part in the draw")
    print()
    print("  %-14s %10s %8s %7s %9s %9s"
          % ("gene", "contrast", "pct", "n_comp", "U1 real", "U1 comp"))
    for p in picked:
        print("  %-14s %10.2f p%-7.1f %7d %9.1f %9.1f"
              % (p["gene"], p["contrast"], p["actual_pct"],
                 p["n_comp"], p["can_u1"], p["comp_u1"]))
    n_pos = sum(1 for p in picked if p["contrast"] > 0)
    n_drawn = sum(1 for p in picked if p["comp_u1"] > p["can_u1"])
    print()
    print("  gene contrast positive (competitor carries more U1): %d/%d selected"
          % (n_pos, len(picked)))
    print("  drawn pair positive (the competitor actually on the panel):  %d/%d"
          % (n_drawn, len(picked)))
    print("  population base: %d/%d paired genes (%.1f%%) have a positive contrast."
          % (int((contrasts > 0).sum()), contrasts.size, 100 * (contrasts > 0).mean()))
    print("  A random draw of %d from a %.1f%% base is expected to land near %.0f positive;"
          % (NPICK, 100 * (contrasts > 0).mean(), NPICK * (contrasts > 0).mean()))
    print("  sampling noise on 20 genes is wide, so the count is not itself evidence.")


if __name__ == "__main__":
    main()
