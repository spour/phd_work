#!/usr/bin/env python3
# find where the pas hexamer actually is

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HEX2 = ("AATAAA", "ATTAAA")
HEX12 = ["AATAAA", "ATTAAA", "TATAAA", "AGTAAA", "AATACA", "CATAAA",
         "GATAAA", "AATATA", "AATAGA", "AAAAAG", "ACTAAA", "AAGAAA"]
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class Genome:
    def __init__(self, path):
        fai = path + ".fai"
        if not os.path.exists(fai):
            sys.exit("need " + fai + "  (samtools faidx)")
        self.idx = {}
        for line in open(fai):
            f = line.split("\t")
            self.idx[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self.fh = open(path, "rb")
        self.n_missing = 0

    def fetch(self, c, a, b):
        if c not in self.idx:
            self.n_missing += 1
            return ""
        ln, off, lb, lw = self.idx[c]
        a, b = max(0, a), min(b, ln)
        if b <= a:
            return ""
        f = lambda p: off + p // lb * lw + p % lb
        self.fh.seek(f(a))
        return self.fh.read(f(b) - f(a)).decode().replace("\n", "").upper()


def liftover(df, chain, exe):
    tmp = tempfile.mkdtemp(prefix="lo_")
    src, dst, unm = (os.path.join(tmp, x) for x in ("i", "o", "u"))
    df = df.reset_index(drop=True)
    with open(src, "w") as fh:
        for i, r in df.iterrows():
            fh.write(f"{r.chrom}\t{int(r.start)}\t{int(r.end)}\t{i}\t0\t"
                     f"{r.strand if r.strand in '+-' else '+'}\n")
    subprocess.run([exe, "-minMatch=0.95", src, chain, dst, unm],
                   capture_output=True, text=True)
    o = pd.read_csv(dst, sep="\t", header=None,
                    names=["chrom", "start", "end", "idx", "sc", "strand"])
    shutil.rmtree(tmp, ignore_errors=True)
    o = o.loc[~o.idx.duplicated(keep=False)]
    k = df.loc[o.idx.to_numpy()].copy()
    k["chrom"] = o.chrom.to_numpy()
    k["start"] = o.start.to_numpy()
    k["end"] = o.end.to_numpy()
    print("  lifted " + f"{len(k):,}" + "/" + f"{len(df):,}")
    return k.reset_index(drop=True)


def anchor_positions(df, mode, flip):
    s = df["start"].to_numpy()
    e = df["end"].to_numpy()
    st = df["strand"].to_numpy().astype(str)
    if flip:
        st = np.where(st == "+", "-", np.where(st == "-", "+", st))
    plus = st != "-"
    if mode == "midpoint":
        pos = (s + e) // 2
    elif mode == "three_prime":
        pos = np.where(plus, e, s)
    elif mode == "five_prime":
        pos = np.where(plus, s, e)
    elif mode == "start":
        pos = s
    elif mode == "end":
        pos = e
    else:
        raise ValueError(mode)
    return pos.astype(int), st


def hex_rate(gen, chrom, pos, strand, offset, win_lo, win_hi, hexes,
             ignore_strand=False):
    hit = tot = 0
    for c, p, s in zip(chrom, pos, strand):
        p = int(p) + offset
        if ignore_strand or s not in "+-":
            seqs = [gen.fetch(c, p + win_lo, p + win_hi),
                    gen.fetch(c, p - win_hi, p - win_lo).translate(RC)[::-1]]
        elif s == "-":
            seqs = [gen.fetch(c, p - win_hi, p - win_lo).translate(RC)[::-1]]
        else:
            seqs = [gen.fetch(c, p + win_lo, p + win_hi)]
        seqs = [x for x in seqs if len(x) >= (win_hi - win_lo) - 2]
        if not seqs:
            continue
        tot += 1
        if any(h in x for x in seqs for h in hexes):
            hit += 1
    return (100.0 * hit / tot if tot else np.nan), tot


def profile(gen, chrom, pos, strand, lo=-100, hi=60, step=5, hexes=HEX2):
    counts = np.zeros(int((hi - lo) / step))
    n = 0
    for c, p, s in zip(chrom, pos, strand):
        p = int(p)
        if s == "-":
            seq = gen.fetch(c, p - hi, p - lo).translate(RC)[::-1]
        else:
            seq = gen.fetch(c, p + lo, p + hi)
        if len(seq) < (hi - lo) - 2:
            continue
        n += 1
        for h in hexes:
            i = seq.find(h)
            while i >= 0:
                off = lo + i
                b = int((off - lo) / step)
                if 0 <= b < len(counts):
                    counts[b] += 1
                i = seq.find(h, i + 1)
    return counts, np.arange(lo, hi, step), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None)
    ap.add_argument("--sites", default=None,
                    help="already-prepped table with chrom/pos/strand")
    ap.add_argument("--chrom", default="seqnames")
    ap.add_argument("--start", default="start")
    ap.add_argument("--end", default="end")
    ap.add_argument("--strand", default="strand")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--chain", default=None)
    ap.add_argument("--liftover-bin", default="external/bin/liftOver")
    ap.add_argument("--build", default="hg38")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="diag_anchor")
    a = ap.parse_args()

    gen = Genome(a.genome)
    if a.sites:
        d = pd.read_csv(a.sites, sep="\t")
        df = pd.DataFrame({"chrom": d.chrom.astype(str),
                           "start": d.pos.astype(int),
                           "end": d.pos.astype(int) + 1,
                           "strand": d.strand.astype(str)})
    else:
        raw = pd.read_csv(a.table, sep="\t", low_memory=False,
                          compression="gzip" if a.table.endswith(".gz") else None)
        df = pd.DataFrame({"chrom": raw[a.chrom].astype(str),
                           "start": pd.to_numeric(raw[a.start], errors="coerce"),
                           "end": pd.to_numeric(raw[a.end], errors="coerce"),
                           "strand": raw[a.strand].astype(str)}).dropna()
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
    df["chrom"] = df.chrom.where(df.chrom.str.startswith("chr"), "chr" + df.chrom)
    print("input " + f"{len(df):,}" + " intervals, median width "
          + f"{np.median(df.end - df.start):.0f}" + " nt")
    print("strand values: " + str(df.strand.value_counts().to_dict()))

    if a.build == "hg19":
        if not os.path.exists(a.liftover_bin) or not a.chain:
            sys.exit("hg19 input needs --chain and liftOver")
        df = liftover(df, a.chain, a.liftover_bin)

    if len(df) > a.n:
        df = df.sample(a.n, random_state=0).reset_index(drop=True)
    print("scanning " + f"{len(df):,}" + " sites\n")

    print("hexamer AATAAA/ATTAAA in [-40,-10) relative to anchor+offset")
    print("anchor        flip  offset   %hex   n")
    print("-" * 44)
    rows = []
    best = None
    for mode in ("midpoint", "three_prime", "five_prime", "start", "end"):
        for flip in (False, True):
            pos, st = anchor_positions(df, mode, flip)
            for off in (-40, -20, -10, 0, 10, 20, 40):
                r, n = hex_rate(gen, df.chrom, pos, st, off, -40, -10, HEX2)
                rows.append(dict(anchor=mode, flip=flip, offset=off,
                                 pct_hex=r, n=n))
                if np.isfinite(r) and (best is None or r > best["pct_hex"]):
                    best = rows[-1]
    R = pd.DataFrame(rows)
    top = R.sort_values("pct_hex", ascending=False).head(12)
    for _, x in top.iterrows():
        print(f"{x.anchor:<13} {str(x.flip):<5} {x.offset:>6}  "
              f"{x.pct_hex:>5.1f}  {x.n:>5,}")
    R.to_csv(a.out + "_grid.tsv", sep="\t", index=False)

    pos, st = anchor_positions(df, best["anchor"], best["flip"])
    ctl, _ = hex_rate(gen, df.chrom, pos + 5000 + best["offset"], st, 0,
                      -40, -10, HEX2)
    print("\nbest: anchor=" + best["anchor"] + " flip=" + str(best["flip"]) +
          " offset=" + str(best["offset"]) +
          "   " + f"{best['pct_hex']:.1f}" + "% vs displaced " +
          f"{ctl:.1f}" + "%")
    mid = R[(R.anchor == "midpoint") & (~R.flip) & (R.offset == 0)]
    if len(mid):
        print("current pipeline setting (midpoint, no flip, offset 0): "
              + f"{mid.pct_hex.iloc[0]:.1f}" + "%")
        print("difference: " + f"{best['pct_hex'] - mid.pct_hex.iloc[0]:+.1f}"
              + " percentage points")

    print("\nwindow sensitivity at the best anchor (12 hexamer variants too)")
    print("window          %AATAAA/ATTAAA   %12 variants")
    for lo, hi in ((-30, -15), (-40, -10), (-60, 0), (-100, 0), (-100, 40)):
        r2, _ = hex_rate(gen, df.chrom, pos + best["offset"], st, 0, lo, hi, HEX2)
        r12, _ = hex_rate(gen, df.chrom, pos + best["offset"], st, 0, lo, hi, HEX12)
        print(f"[{lo:>5},{hi:>4})      {r2:>8.1f}       {r12:>8.1f}")

    cnt, edges, n = profile(gen, df.chrom, pos + best["offset"], st)
    P = pd.DataFrame({"offset": edges, "count": cnt,
                      "per_site": cnt / max(n, 1)})
    P.to_csv(a.out + "_profile.tsv", sep="\t", index=False)
    print("\nhexamer start-position profile (per site, 5 nt bins)")
    mx = P.per_site.max()
    for _, r in P.iterrows():
        bar = "#" * int(round(40 * r.per_site / mx)) if mx > 0 else ""
        print(f"{int(r.offset):>5}  {r.per_site:>6.3f}  {bar}")
    pk = int(P.loc[P.per_site.idxmax(), "offset"])
    print("\npeak at " + str(pk) + " nt. A real cleavage site puts the hexamer "
          "start near -21.")
    if gen.n_missing:
        print("WARNING: " + str(gen.n_missing) + " fetches hit a contig absent "
              "from the FASTA")
    print("\nwrote " + a.out + "_grid.tsv and " + a.out + "_profile.tsv")


if __name__ == "__main__":
    main()