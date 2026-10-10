#!/usr/bin/env python3
# terminal-exon-vs-not dataset built straight from the gencode gtf
import argparse, os, re, gzip
import numpy as np, pandas as pd

RF_TRAIN = {"chr1","chr2","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
            "chr11","chr12","chr14","chr15","chr16","chr17"}
RF_VAL  = {"chr3","chr13"}
RF_TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}

_GID = re.compile(r'gene_id "([^"]+)"')
_TID = re.compile(r'transcript_id "([^"]+)"')
_GT  = re.compile(r'gene_type "([^"]+)"')
_TT  = re.compile(r'transcript_type "([^"]+)"')
_EN  = re.compile(r'exon_number "?(\d+)"?')

def split_of(c):
    c = c if c.startswith("chr") else "chr"+c
    if c in RF_TRAIN: return "train"
    if c in RF_VAL:  return "val"
    if c in RF_TEST: return "test"
    return None

def revcomp(s):
    return s.translate(str.maketrans("ACGTNacgtn","TGCANtgcan"))[::-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default="data/annotation/gencode.v49.annotation.gtf.gz")
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--extra_neg_bed", default=None,
                    help="optional hard negatives (e.g. CPCCPA 3-way intragenic BED)")
    ap.add_argument("--canonical_only", action="store_true",
                    help="keep only MANE_Select / Ensembl_canonical transcripts (~1 per gene)")
    ap.add_argument("--internal_neg_only", action="store_true",
                    help="negatives = internal exons only (drop first exons)")
    ap.add_argument("--include_noncoding", action="store_true",
                    help="lift the protein_coding gene/transcript filter")
    ap.add_argument("--upstream",   type=int, default=1000)
    ap.add_argument("--downstream", type=int, default=500)
    ap.add_argument("--neg_ratio",  type=int, default=10)
    ap.add_argument("--min_sep",    type=int, default=50)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    import pysam
    fa = pysam.FastaFile(a.reference_fasta); refs = set(fa.references)
    W = a.upstream + a.downstream

    op = gzip.open if a.gtf.endswith(".gz") else open
    tx = {}
    n_lines = 0
    print(f"[parse] {a.gtf} ...")
    with op(a.gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            feat = f[2]
            if feat not in ("transcript", "exon"):
                continue
            attr = f[8]
            tid_m = _TID.search(attr)
            if tid_m is None:
                continue
            tid = tid_m.group(1)
            n_lines += 1
            if feat == "transcript":
                gt = _GT.search(attr); tt = _TT.search(attr)
                keep = True
                if not a.include_noncoding:
                    keep = (gt and gt.group(1) == "protein_coding" and
                            tt and tt.group(1) == "protein_coding")
                if a.canonical_only:
                    keep = keep and (('tag "MANE_Select"' in attr) or
                                     ('tag "Ensembl_canonical"' in attr))
                gid = _GID.search(attr)
                tx[tid] = {"gene": gid.group(1) if gid else tid,
                           "chrom": f[0], "strand": f[6], "keep": keep, "exons": []}
            else:
                t = tx.get(tid)
                if t is None:
                    continue
                en_m = _EN.search(attr)
                if en_m is None:
                    continue
                en = int(en_m.group(1))
                end3 = int(f[4]) if f[6] == "+" else int(f[3])
                t["exons"].append((en, end3))
            if n_lines % 2_000_000 == 0:
                print(f"  {n_lines:,} feature lines", flush=True)

    # classify exons per transcript
    pos_rows, neg_rows = [], []
    for tid, t in tx.items():
        if not t["keep"] or not t["exons"]:
            continue
        ens = [e for e, _ in t["exons"]]
        max_en = max(ens)
        for en, end3 in t["exons"]:
            is_terminal = (en == max_en)
            is_first = (en == 1)
            if is_terminal:
                pos_rows.append((t["chrom"], end3, t["strand"]))
            else:
                if a.internal_neg_only and is_first:
                    continue
                neg_rows.append((t["chrom"], end3, t["strand"]))

    pos = pd.DataFrame(pos_rows, columns=["chrom","end3","strand"]).drop_duplicates()
    neg = pd.DataFrame(neg_rows, columns=["chrom","end3","strand"]).drop_duplicates()
    print(f"[ends] terminal (pos) {len(pos):,} | non-terminal (neg, pre-filter) {len(neg):,}")

    # drop negatives within min_sep of a positive (ambiguous coords)
    keep_mask = np.ones(len(neg), dtype=bool)
    neg = neg.reset_index(drop=True)
    for (chrom, strand), gp in pos.groupby(["chrom","strand"]):
        parr = np.sort(gp.end3.to_numpy())
        sel = (neg.chrom.values == chrom) & (neg.strand.values == strand)
        idx = np.where(sel)[0]
        if len(idx) == 0 or len(parr) == 0:
            continue
        q = neg.end3.values[idx]
        ins = np.searchsorted(parr, q)
        for j, i in enumerate(idx):
            p = ins[j]; best = np.inf
            if p < len(parr): best = min(best, abs(parr[p] - q[j]))
            if p > 0:        best = min(best, abs(parr[p-1] - q[j]))
            if best <= a.min_sep:
                keep_mask[i] = False
    neg = neg[keep_mask]
    print(f"[ends] negatives after dropping those within {a.min_sep} nt of a positive: {len(neg):,}")

    # subsample negatives to neg_ratio x n_pos
    n_target = a.neg_ratio * len(pos)
    if len(neg) > n_target:
        neg = neg.sample(n=n_target, random_state=a.seed)
    print(f"[ends] negatives subsampled to {len(neg):,} (ratio {a.neg_ratio}x)")

    # fetch + write fastas split by chrom
    def fetch(chrom, center, strand):
        c = chrom if chrom in refs else chrom.lstrip("chr")
        if c not in refs:
            return None
        if strand == "+": s = center - a.upstream; e = center + a.downstream
        else:             s = center - a.downstream; e = center + a.upstream
        if s < 0:
            return None
        try:
            seq = fa.fetch(c, s, e).upper()
        except Exception:
            return None
        if len(seq) != W:
            return None
        return revcomp(seq) if strand == "-" else seq

    handles = {}; counts = {}
    def H(cls, sp):
        k = (cls, sp)
        if k not in handles:
            handles[k] = open(os.path.join(a.out_dir, f"{cls}_{sp}.fa"), "w")
        return handles[k]
    def emit(cls, chrom, center, strand, tag):
        sp = split_of(chrom)
        if sp is None:
            return
        seq = fetch(chrom, center, strand)
        if seq is None:
            return
        H(cls, sp).write(f">{tag}|{chrom}:{center}:{strand}|{sp}\n{seq}\n")
        counts[(cls, sp)] = counts.get((cls, sp), 0) + 1

    for i, r in enumerate(pos.itertuples(index=False)):
        emit("pos", r.chrom, int(r.end3), r.strand, f"TE_{i}")
    for i, r in enumerate(neg.itertuples(index=False)):
        emit("neg", r.chrom, int(r.end3), r.strand, f"NT_{i}")

    if a.extra_neg_bed and os.path.exists(a.extra_neg_bed):
        n_extra = 0
        for line in open(a.extra_neg_bed):
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            chrom = p[0]; center = (int(p[1]) + int(p[2])) // 2
            strand = p[5] if len(p) > 5 else "+"
            emit("neg", chrom, center, strand, f"HARDNEG_{n_extra}"); n_extra += 1
        print(f"[neg] + folded {n_extra:,} hard negatives from {a.extra_neg_bed}")

    for h in handles.values(): h.close()
    print(f"[done] window={W} nt (up {a.upstream}/down {a.downstream}); files in {a.out_dir}")
    for sp in ("train","val","test"):
        print(f"  {sp}: pos={counts.get(('pos',sp),0):,}  neg={counts.get(('neg',sp),0):,}")

if __name__ == "__main__":
    main()