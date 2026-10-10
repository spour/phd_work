#!/usr/bin/env python3
# extract 3'utr + last-exon bed from a gtf
import argparse
import gzip
import os
import re
import subprocess
from collections import defaultdict


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def parse_gtf(gtf_path):
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    wanted = {"exon", "CDS", "three_prime_UTR", "three_prime_utr", "UTR"}
    with open_maybe_gz(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] not in wanted:
                continue
            m = tx_re.search(p[8])
            if not m:
                continue
            yield p[2], p[0], int(p[3]) - 1, int(p[4]), m.group(1), p[6]


def sort_bed(path):
    subprocess.run(f"sort -k1,1 -k2,2n {path} -o {path}", shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--downstream_slop", type=int, default=1000,
                    help="extend the 3' zone downstream of the gene end by N nt "
                         "to catch readthrough CPAs (default 1000)")
    ap.add_argument("--bedtools", default="bedtools")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] Parsing GTF {args.gtf} ...")
    exons = defaultdict(list)
    cds   = defaultdict(list)
    utr3  = defaultdict(list)
    for feat, c, s, e, tx, strand in parse_gtf(args.gtf):
        if feat == "exon":
            exons[tx].append((c, s, e, strand))
        elif feat == "CDS":
            cds[tx].append((c, s, e, strand))
        elif feat in ("three_prime_UTR", "three_prime_utr"):
            utr3[tx].append((c, s, e, strand))
    print(f"  transcripts: exon={len(exons):,} CDS={len(cds):,} "
          f"3UTR(explicit)={len(utr3):,}")

    have_explicit_utr = len(utr3) > 0
    if have_explicit_utr:
        print("  -> using explicit three_prime_UTR features (GENCODE/Ensembl)")
    else:
        print("  -> computing 3'UTR as exonic regions downstream of CDS "
              "(UCSC/RefSeq style)")

    # emit terminal exons (last exon per transcript)
    term_path = os.path.join(args.out_dir, "terminal_exons.bed")
    print(f"[2/4] Writing terminal exons -> {term_path}")
    with open(term_path, "w") as f:
        for tx, exs in exons.items():
            strand = exs[0][3]
            if strand == "+":
                last = max(exs, key=lambda e: e[2])
            else:
                last = min(exs, key=lambda e: e[1])
            f.write(f"{last[0]}\t{last[1]}\t{last[2]}\t{tx}\t.\t{strand}\n")
    sort_bed(term_path)

    # emit 3'utr
    utr_path = os.path.join(args.out_dir, "three_prime_utr.bed")
    print(f"[3/4] Writing 3'UTR -> {utr_path}")
    n_utr = 0
    with open(utr_path, "w") as f:
        if have_explicit_utr:
            for tx, regs in utr3.items():
                for c, s, e, strand in regs:
                    f.write(f"{c}\t{s}\t{e}\t{tx}\t.\t{strand}\n")
                    n_utr += 1
        else:
            for tx, exs in exons.items():
                if tx not in cds:
                    continue
                strand = exs[0][3]
                if strand == "+":
                    stop = max(c[2] for c in cds[tx])
                    for c, s, e, st in exs:
                        ns = max(s, stop)
                        if ns < e:
                            f.write(f"{c}\t{ns}\t{e}\t{tx}\t.\t{st}\n")
                            n_utr += 1
                else:
                    stop = min(c[1] for c in cds[tx])
                    for c, s, e, st in exs:
                        ne = min(e, stop)
                        if s < ne:
                            f.write(f"{c}\t{s}\t{ne}\t{tx}\t.\t{st}\n")
                            n_utr += 1
    sort_bed(utr_path)
    print(f"  3'UTR intervals: {n_utr:,}")

    # build the 3' zone: union of (3'utr + terminal exon + downstream slop)
    zone_path = os.path.join(args.out_dir, "three_prime_zone.bed")
    print(f"[4/4] Building 3' exclusion zone (3'UTR + terminal exon + "
          f"{args.downstream_slop} nt downstream) -> {zone_path}")
    slop_path = os.path.join(args.out_dir, "_downstream_slop.bed")
    with open(slop_path, "w") as f:
        for tx, exs in exons.items():
            strand = exs[0][3]
            chrom = exs[0][0]
            if strand == "+":
                end_pos = max(e[2] for e in exs)
                f.write(f"{chrom}\t{end_pos}\t{end_pos + args.downstream_slop}"
                        f"\t{tx}_dslop\t.\t{strand}\n")
            else:
                start_pos = min(e[1] for e in exs)
                ns = max(0, start_pos - args.downstream_slop)
                f.write(f"{chrom}\t{ns}\t{start_pos}\t{tx}_dslop\t.\t{strand}\n")
    sort_bed(slop_path)

    cat_path = os.path.join(args.out_dir, "_zone_unmerged.bed")
    with open(cat_path, "w") as out:
        for p in (utr_path, term_path, slop_path):
            with open(p) as f:
                out.write(f.read())
    sort_bed(cat_path)
    tmp = zone_path + ".tmp"
    with open(tmp, "w") as out:
        subprocess.run([args.bedtools, "merge", "-s", "-c", "6",
                        "-o", "distinct", "-i", cat_path],
                       stdout=out, check=True)
    with open(tmp) as f, open(zone_path, "w") as out:
        for line in f:
            c, s, e, st = line.rstrip("\n").split("\t")
            out.write(f"{c}\t{s}\t{e}\tthree_prime_zone\t.\t{st}\n")
    os.remove(tmp); os.remove(cat_path); os.remove(slop_path)
    sort_bed(zone_path)

    def count(p): return sum(1 for _ in open(p))
    print("\nDone.")
    print(f"  terminal_exons.bed:    {count(term_path):,}")
    print(f"  three_prime_utr.bed:   {count(utr_path):,}")
    print(f"  three_prime_zone.bed:  {count(zone_path):,}  <-- use this as --exclude_bed")


if __name__ == "__main__":
    main()