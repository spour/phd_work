#!/usr/bin/env python3
# partition the old curated cryptic set into: (a) recovered by 3-way intragenic

import argparse
import os
import subprocess
import sys
import tempfile


def n_lines(path):
    if not path or not os.path.exists(path):
        return 0
    return sum(1 for _ in open(path) if _.strip() and not _.startswith("#"))


def intersect_count(a_bed, b_bed, strand=True, invert=False):
    flag = "-v" if invert else "-u"
    cmd = ["bedtools", "intersect", flag]
    if strand:
        cmd.append("-s")
    cmd += ["-a", a_bed, "-b", b_bed]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True)
    return sum(1 for line in res.stdout.splitlines() if line.strip())


def intersect_to_file(a_bed, b_bed, out_path, strand=True, invert=False):
    flag = "-v" if invert else "-u"
    cmd = ["bedtools", "intersect", flag]
    if strand:
        cmd.append("-s")
    cmd += ["-a", a_bed, "-b", b_bed]
    with open(out_path, "w") as fout:
        subprocess.run(cmd, stdout=fout, check=True)
    return n_lines(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_cryptic_bed",
                     default="data/Cryptic_CPA_sites.hg38.bed",
                     help="The OLDER curated cryptic-CPA BED")
    ap.add_argument("--three_way_bed",
                     default="figures/cpccpa_3way_rflrcnn_intragenic.bed")
    ap.add_argument("--three_prime_zone_bed",
                     default="data/annotation/three_prime_zone.bed")
    ap.add_argument("--pc_genes_bed",
                     default="data/pc_genes.bed")
    ap.add_argument("--figures_dir",
                     default="figures",
                     help="Directory containing cpccpa_tier*.bed")
    ap.add_argument("--no_strand", action="store_true",
                     help="Disable strand check (default: strand-aware)")
    ap.add_argument("--keep_tmp", action="store_true",
                     help="Keep intermediate BEDs for inspection")
    args = ap.parse_args()

    strand = not args.no_strand
    print(f"Strand check: {strand}")
    print()

    for path, label in [(args.old_cryptic_bed, "old cryptic"),
                        (args.three_way_bed,   "3-way intragenic"),
                        (args.three_prime_zone_bed, "3'UTR zone"),
                        (args.pc_genes_bed,    "pc_genes")]:
        ok = "OK " if os.path.exists(path) else "MISSING"
        n = n_lines(path) if os.path.exists(path) else 0
        print(f"  {ok}  {label:<22} {path}  ({n:,} lines)")
    print()

    if not os.path.exists(args.old_cryptic_bed):
        print(f"ERROR: --old_cryptic_bed not found")
        sys.exit(1)

    total = n_lines(args.old_cryptic_bed)
    print(f"=== Total old cryptic sites: {total:,} ===\n")

    tmpdir = tempfile.mkdtemp(prefix="classify_") if not args.keep_tmp \
              else "/tmp/classify_old_cryptic"
    if args.keep_tmp:
        os.makedirs(tmpdir, exist_ok=True)
    print(f"Intermediate BEDs: {tmpdir}\n")

    in_3way = os.path.join(tmpdir, "old_in_3way.bed")
    not_3way = os.path.join(tmpdir, "old_not_in_3way.bed")
    n_in_3way = intersect_to_file(args.old_cryptic_bed, args.three_way_bed,
                                    in_3way, strand=strand)
    n_not_3way = intersect_to_file(args.old_cryptic_bed, args.three_way_bed,
                                     not_3way, strand=strand, invert=True)
    print(f"(a) In 3-way intragenic BED:           {n_in_3way:>7,}  "
          f"({n_in_3way/total*100:5.1f}%)")
    print(f"    NOT in 3-way intragenic BED:        {n_not_3way:>7,}  "
          f"({n_not_3way/total*100:5.1f}%)\n")
    print(f"    -> the {n_not_3way:,} 'misses' are partitioned below.\n")

    if n_not_3way == 0:
        print("Nothing missing! All old-cryptic sites are in the 3-way BED.")
        return

    if os.path.exists(args.three_prime_zone_bed):
        in_3utr = os.path.join(tmpdir, "miss_in_3utr.bed")
        outside_3utr = os.path.join(tmpdir, "miss_outside_3utr.bed")
        n_in_3utr = intersect_to_file(not_3way, args.three_prime_zone_bed,
                                         in_3utr, strand=strand)
        n_outside_3utr = intersect_to_file(not_3way, args.three_prime_zone_bed,
                                             outside_3utr, strand=strand,
                                             invert=True)
        print(f"(b) Misses overlapping 3'UTR zone:    {n_in_3utr:>7,}  "
              f"({n_in_3utr/total*100:5.1f}% of total, "
              f"{n_in_3utr/n_not_3way*100:5.1f}% of misses)")
        print(f"    -> excluded by intragenic-only definition")
        print()
        misses_outside_3utr = outside_3utr
        n_misses_outside_3utr = n_outside_3utr
    else:
        print(f"(b) skipped — no three_prime_zone_bed")
        misses_outside_3utr = not_3way
        n_misses_outside_3utr = n_not_3way

    if os.path.exists(args.pc_genes_bed):
        in_pc = os.path.join(tmpdir, "miss_intragenic_in_pc.bed")
        outside_pc = os.path.join(tmpdir, "miss_outside_pc.bed")
        n_in_pc = intersect_to_file(misses_outside_3utr, args.pc_genes_bed,
                                       in_pc, strand=strand)
        n_outside_pc = intersect_to_file(misses_outside_3utr, args.pc_genes_bed,
                                            outside_pc, strand=strand,
                                            invert=True)
        print(f"(c) Intragenic misses outside pc_genes: "
              f"{n_outside_pc:>5,}  ({n_outside_pc/total*100:5.1f}% of total, "
              f"{n_outside_pc/n_misses_outside_3utr*100:5.1f}% of intragenic-misses)")
        print(f"    -> no parquet existed; no model could have called a peak here")
        print()
        misses_in_pc = in_pc
        n_misses_in_pc = n_in_pc
    else:
        print(f"(c) skipped — no pc_genes_bed")
        misses_in_pc = misses_outside_3utr
        n_misses_in_pc = n_misses_outside_3utr

    tiers = [
        ("tier4_all_four",      "all 4 models agree"),
        ("tier3_exactly_three", "exactly 3 models agree"),
        ("tier2_exactly_two",   "exactly 2 models agree"),
        ("tier1_exactly_one",   "exactly 1 model agrees"),
    ]

    print(f"(d) Of the {n_misses_in_pc:,} remaining (intragenic, in a pc_gene):")
    n_classified_to_tier = 0
    for tname, descr in tiers:
        tier_bed = os.path.join(args.figures_dir,
                                 f"cpccpa_{tname}_no_3utr.bed")
        if not os.path.exists(tier_bed):
            continue
        n = intersect_count(misses_in_pc, tier_bed, strand=strand)
        n_classified_to_tier += n
        marker = ""
        if "tier4" in tname or "tier3" in tname:
            if "tier4" in tname and n > 0:
                marker = "  <-- sanity: tier4 should be IN the 3-way BED. " \
                          "Check that 3-way includes APARENT2-required sites."
            if "tier3" in tname and n > 0:
                marker = "  <-- maybe 3-way of a DIFFERENT 3-model combo " \
                          "(e.g. RF+LR+APARENT2 not RF+LR+CNN)"
        print(f"      {descr:<28} {n:>6,}  "
              f"({n/total*100:5.1f}% of total){marker}")
    print()

    n_no_model = n_misses_in_pc - n_classified_to_tier
    print(f"(e) Intragenic, in pc_gene, NO model agrees (no peak at all): "
          f"{n_no_model:>5,}  ({n_no_model/total*100:5.1f}% of total)")
    print(f"    -> these are sites in the old set that the current 4-model")
    print(f"       ensemble does NOT call. Either the old set used a")
    print(f"       different threshold, a different method, or these are")
    print(f"       false positives of the old set.\n")

    print(f"{'=' * 60}")
    print(f"SUMMARY (old cryptic, n={total:,})")
    print(f"{'=' * 60}")
    print(f"  (a) recovered by 3-way intragenic BED:    {n_in_3way:>7,}  "
          f"({n_in_3way/total*100:5.1f}%)")
    if os.path.exists(args.three_prime_zone_bed):
        print(f"  (b) lost to 3'UTR-zone exclusion:         {n_in_3utr:>7,}  "
              f"({n_in_3utr/total*100:5.1f}%)")
    if os.path.exists(args.pc_genes_bed):
        print(f"  (c) outside any pc_gene (not scanned):    {n_outside_pc:>7,}  "
              f"({n_outside_pc/total*100:5.1f}%)")
    print(f"  (d) intragenic + in pc_gene, partial:     {n_classified_to_tier:>7,}  "
          f"({n_classified_to_tier/total*100:5.1f}%) "
          f"-- only 1 or 2 models agree")
    print(f"  (e) intragenic + in pc_gene, no support:  {n_no_model:>7,}  "
          f"({n_no_model/total*100:5.1f}%)")
    print(f"  {'=' * 56}")
    s_check = (n_in_3way
               + (n_in_3utr if os.path.exists(args.three_prime_zone_bed) else 0)
               + (n_outside_pc if os.path.exists(args.pc_genes_bed) else 0)
               + n_classified_to_tier
               + n_no_model)
    print(f"  total accounted for:                      {s_check:>7,}")
    if s_check != total:
        print(f"  WARN: total {total:,} != sum {s_check:,} "
               f"(diff {total - s_check:,})")

    if not args.keep_tmp:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f"\nintermediate BEDs left in {tmpdir}")


if __name__ == "__main__":
    main()