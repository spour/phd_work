#!/usr/bin/env python3
# the terminal-exon frame
import sys
import csv
import numpy as np

from clean_descriptor import build_pairs, wilcoxon_signed_rank
from geometry_test import decided

FEAT = "fig3_new_positional_features.tsv"
GRID = "gridsearch_cache.npz"
U1C = "u1_grid_cache.npz"
ME_THR = 8.0
U1_THR = 4.0
MIN_L = 50


def line(name, r):
    a, d = r["all"], r["decided"]
    print("%-44s %8.4f %8.4f %10.2e | %8.4f %10.2e %7.3f %8d"
          % (name, a["gene_avg"], a["pair_wt"], a["p"],
             d["gene_avg"], d["p"], r["tie_frac"], r["n_both"]))


def header(title, note=None):
    print()
    print("=" * 108)
    print(title)
    if note:
        print(note)
    print("=" * 108)
    print("%-44s %8s %8s %10s | %8s %10s %7s %8s"
          % ("test (competitor higher = 1)", "gene-avg", "pair-wt",
             "Wilcoxon p", "decided", "p", "tie", "n pairs"))


def main():
    raw = np.genfromtxt(FEAT, delimiter="\t", names=True, dtype=None,
                        encoding="utf-8",
                        usecols=("gene", "is_canonical", "d_anchor_to_pa",
                                 "first5_dn_dist", "first5_dn_present"))
    gene = raw["gene"].astype(str)
    y = raw["is_canonical"].astype(int)
    L = raw["d_anchor_to_pa"].astype(float)
    dd = raw["first5_dn_dist"].astype(float)
    p5 = raw["first5_dn_present"].astype(float)

    zg = np.load(GRID, allow_pickle=True)
    if not np.array_equal(zg["gene"].astype(str), gene):
        sys.exit("gene order differs between %s and %s" % (FEAT, GRID))
    gc = [str(x) for x in zg["cols"]]
    G = zg["M"]
    zu = np.load(U1C, allow_pickle=True)
    if not np.array_equal(zu["gene"].astype(str), gene):
        sys.exit("gene order differs between %s and %s" % (FEAT, U1C))
    uc = [str(x) for x in zu["cols"]]
    U = zu["M"]

    def gcol(n):
        if n not in gc:
            sys.exit("gridsearch cache lacks %s" % n)
        return G[:, gc.index(n)].astype(float)

    def ucol(n):
        if n not in uc:
            sys.exit("u1 cache lacks %s" % n)
        return U[:, uc.index(n)].astype(float)

    d5_pa = -gcol("MaxEnt|donor|interv|nearest_dist@%g" % ME_THR)
    du_pa = -ucol("interv|nearest_dist@%g" % U1_THR)
    n5 = gcol("MaxEnt|donor|interv|count_above@%g" % ME_THR)
    nu = ucol("interv|count_above@%g" % U1_THR)
    dens5 = gcol("MaxEnt|donor|interv|density_per_kb@%g" % ME_THR)
    densu = ucol("interv|density_per_kb@%g" % U1_THR)

    short = L < MIN_L
    Ls = np.where(short, np.nan, L)
    f5_near = 1.0 - d5_pa / Ls
    fu_near = 1.0 - du_pa / Ls
    f5_first = dd / Ls
    f5_first[p5 == 0] = np.nan

    ci, ki, slot, n_slot, _ = build_pairs(gene, y)
    print("[in] %d peaks, %d genes with both classes, %d pairs"
          % (gene.size, n_slot, ci.size), file=sys.stderr)
    print("[in] terminal exon L: median canonical %.0f nt, competitor %.0f nt; "
          "%d peaks with L<%d excluded from fractions"
          % (np.median(L[y == 1]), np.median(L[y == 0]), int(short.sum()), MIN_L),
          file=sys.stderr)
    for tag, v in (("strong 5'ss in interval", n5), ("strong U1 in interval", nu)):
        print("[in] %s: canonical %.4f, competitor %.4f of peaks have >=1"
              % (tag, (v[y == 1] > 0).mean(), (v[y == 0] > 0).mean()),
              file=sys.stderr)

    # fractional position
    header("1. FRACTIONAL POSITION within the terminal exon (0 = anchor, 1 = pA)",
           "conditioned on both members of the pair carrying such a site inside "
           "the interval")
    line("5'ss nearest the pA, frac position", decided(f5_near, ci, ki, slot, n_slot, True))
    line("5'ss first past the anchor, frac position", decided(f5_first, ci, ki, slot, n_slot, True))
    line("U1 site nearest the pA, frac position", decided(fu_near, ci, ki, slot, n_slot, True))

    for tag, f in (("5'ss nearest pA", f5_near), ("5'ss first from anchor", f5_first),
                   ("U1 nearest pA", fu_near)):
        b = np.isfinite(f[ci]) & np.isfinite(f[ki])
        print("    %-24s median frac: canonical %.3f, competitor %.3f, "
              "median paired difference %+.4f"
              % (tag, np.median(f[ci][b]), np.median(f[ki][b]),
                 np.median(f[ki][b] - f[ci][b])))

    one = (n5 == 1) & np.isfinite(f5_near)
    b = one[ci] & one[ki]
    if b.sum() > 500:
        r = decided(np.where(one, f5_near, np.nan), ci, ki, slot, n_slot, True)
        print()
        print("    exactly one strong 5'ss in the interval in BOTH members "
              "(%d pairs): here the nearest-to-pA site IS the mean position" % b.sum())
        line("    single-site frac position", r)

    # density
    header("2. DENSITY PER KB OF TERMINAL EXON (count in interval / L)")
    line("5'ss density per kb of exon", decided(dens5, ci, ki, slot, n_slot, True))
    line("U1 density per kb of exon", decided(densu, ci, ki, slot, n_slot, True))
    line("5'ss raw count in interval", decided(n5, ci, ki, slot, n_slot, True))
    line("U1 raw count in interval", decided(nu, ci, ki, slot, n_slot, True))

    # length
    header("3. TERMINAL EXON LENGTH L = anchor -> pA")
    line("competitor exon LONGER than its real end", decided(L, ci, ki, slot, n_slot, True))
    b = np.isfinite(L[ci]) & np.isfinite(L[ki])
    print("    median L: canonical %.0f nt, competitor %.0f nt, "
          "median paired difference %+.0f nt"
          % (np.median(L[ci][b]), np.median(L[ki][b]),
             np.median(L[ki][b] - L[ci][b])))


if __name__ == "__main__":
    main()
