#!/usr/bin/env python3
# the terminal-vs-internal exon splice-site reference
import argparse, sys, random
import numpy as np, pandas as pd
from collections import defaultdict
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts_dir", default=None)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--max_tx", type=int, default=40000, help="cap transcripts for speed")
    ap.add_argument("--per_class", type=int, default=60000, help="cap exons per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exon_ref")
    a = ap.parse_args()
    if a.scripts_dir: sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import GenomeFasta, init_maxentscan
    out = init_maxentscan(); score5, score3 = out[0], out[1]
    fa = GenomeFasta(a.reference_fasta)

    # parse gtf: collect exons per transcript (protein_coding only)
    tx_exons = defaultdict(list)
    tx_ok = {}
    print("[gtf] parsing...", file=sys.stderr)
    with open(a.gtf) as fh:
        for line in fh:
            if line.startswith("#"): continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon": continue
            attr = f[8]
            if 'transcript_type "protein_coding"' not in attr and \
               'transcript_biotype "protein_coding"' not in attr: continue
            tid = None
            for kv in attr.split(";"):
                kv = kv.strip()
                if kv.startswith("transcript_id"):
                    tid = kv.split('"')[1]; break
            if tid is None: continue
            chrom, start, end, strand = f[0], int(f[3])-1, int(f[4]), f[6]
            tx_exons[tid].append((start, end, strand, chrom))
            if len(tx_exons) > a.max_tx and tid not in tx_exons: break

    rng = random.Random(a.seed)
    tids = list(tx_exons.keys())
    if len(tids) > a.max_tx:
        tids = rng.sample(tids, a.max_tx)
    print(f"[gtf] {len(tids):,} transcripts", file=sys.stderr)

    # classify exons, collect splice-site coordinates
    def score_acceptor(chrom, pos, strand):
        if strand == "+":
            s = fa.fetch(chrom, pos-20, pos+3, strand="+")
        else:
            s = fa.fetch(chrom, pos-2, pos+21, strand="-")
        if not s or len(s) != 23: return np.nan
        try:
            v = score3(s.upper()); return float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
        except Exception: return np.nan
    def score_donor(chrom, pos, strand):
        if strand == "+":
            s = fa.fetch(chrom, pos-2, pos+7, strand="+")
        else:
            s = fa.fetch(chrom, pos-6, pos+3, strand="-")
        if not s or len(s) != 9: return np.nan
        try:
            v = score5(s.upper()); return float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
        except Exception: return np.nan

    rows = []
    def nearest_dn_donor(chrom, acc_pos, strand, maxscan=1500, thr=8.0):
        if strand == "+":
            seg = fa.fetch(chrom, acc_pos, acc_pos+maxscan, strand="+")
        else:
            seg = fa.fetch(chrom, acc_pos-maxscan, acc_pos+1, strand="-")
        if not seg or len(seg) < 10: return np.nan
        seg = seg.upper()
        arr = np.frombuffer(seg.encode("ascii","replace"), np.uint8)
        gts = np.where((arr[:-1]==71) & (arr[1:]==84))[0]
        best_d = np.nan
        for p in gts:
            p = int(p)
            if p-3 < 0 or p+6 > len(seg): continue
            try:
                v = score5(seg[p-3:p+6])
                v = float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
            except Exception: continue
            if np.isfinite(v) and v >= thr:
                best_d = float(p); break
        return best_d
    for tid in tqdm(tids, desc="exons"):
        exs = tx_exons[tid]
        if len(exs) < 2: continue
        strand = exs[0][2]; chrom = exs[0][3]
        exs = sorted(exs, key=lambda e: e[0])
        if strand == "-": exs = exs[::-1]
        n = len(exs)
        for i, (st, en, sd, ch) in enumerate(exs):
            first = (i == 0); last = (i == n-1)
            if first: cls = "first"
            elif last: cls = "terminal"
            else: cls = "internal"
            if strand == "+":
                acc_pos, don_pos = st, en-1
            else:
                acc_pos, don_pos = en-1, st
            a3 = np.nan if first else score_acceptor(ch, acc_pos, strand)
            d5 = np.nan if last  else score_donor(ch, don_pos, strand)
            dn = np.nan
            if not first:
                dn = nearest_dn_donor(ch, acc_pos, strand)
            rows.append(dict(tx=tid, cls=cls, three=a3, five=d5, dn_donor_dist=dn))

    D = pd.DataFrame(rows)
    parts = []
    for c, gg in D.groupby("cls"):
        parts.append(gg.sample(min(len(gg), a.per_class), random_state=a.seed))
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(f"{a.out}_scores.tsv", sep="\t", index=False)

    print("[summary] median splice-site scores by exon class:", file=sys.stderr)
    for c in ("internal","terminal","first"):
        sub = D[D.cls==c]
        m3 = sub.three.dropna(); m5 = sub.five.dropna(); dd = sub.dn_donor_dist.dropna()
        has_dn = (sub.dn_donor_dist.notna()).mean() if len(sub) else 0
        print(f"    {c:9} n={len(sub):>6}  "
              f"3'ss median={m3.median():.2f}  "
              f"5'ss median={m5.median() if len(m5) else float('nan'):.2f}  "
              f"nearest-dn-donor median={dd.median() if len(dd) else float('nan'):.0f}nt "
              f"(has one within 1.5kb: {has_dn:.0%})", file=sys.stderr)
    print(f"[out] {a.out}_scores.tsv", file=sys.stderr)

if __name__ == "__main__":
    main()