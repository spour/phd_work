#!/usr/bin/env python3
# distance, not presence
import sys
import csv
import numpy as np

from clean_descriptor import build_pairs, win_rate, wilcoxon_signed_rank

GRID = "gridsearch_cache.npz"
UCUBE = "fig3_new_ucube.npz"
META = "fig3_ucube_meta.tsv"
ME_THR = 8.0
DIST_CAP = 2000
OUT_DIST = "geometry_dist_pooled.tsv"


def decided(v, ci, ki, slot, n_slot, comp_hi):
    c, k = v[ci], v[ki]
    ok = np.isfinite(c) & np.isfinite(k)
    hi, lo = (k, c) if comp_hi else (c, k)
    w_all = np.where(hi > lo, 1.0, np.where(hi == lo, 0.5, 0.0))
    res = {}
    for tag, m in (("all", ok), ("decided", ok & (c != k))):
        num = np.bincount(slot, weights=w_all * m, minlength=n_slot)
        den = np.bincount(slot, weights=m.astype(float), minlength=n_slot)
        g = den > 0
        rates = num[g] / den[g]
        p, _ = wilcoxon_signed_rank(rates - 0.5)
        res[tag] = dict(gene_avg=float(rates.mean()),
                        pair_wt=float(num.sum() / den.sum()),
                        p=p, npair=int(den.sum()), ngene=int(g.sum()))
    res["tie_frac"] = float((c[ok] == k[ok]).mean())
    res["n_both"] = int(ok.sum())
    return res


def line(name, r, comp_hi):
    d = r["decided"]
    a = r["all"]
    print("%-38s %8.4f %8.4f %10.2e | %8.4f %10.2e %8.3f %8d"
          % (name, a["gene_avg"], a["pair_wt"], a["p"],
             d["gene_avg"], d["p"], r["tie_frac"], r["n_both"]))


def main():
    met = list(csv.DictReader(open(META), delimiter="\t"))
    gene = np.array([r["gene"] for r in met])
    y = np.array([int(r["is_canonical"]) for r in met])

    zg = np.load(GRID, allow_pickle=True)
    if not np.array_equal(zg["gene"].astype(str), gene):
        sys.exit("gene order differs between %s and %s" % (GRID, META))
    gcols = [str(x) for x in zg["cols"]]
    G = zg["M"]

    def col(name):
        if name not in gcols:
            sys.exit("cache lacks column %s" % name)
        return G[:, gcols.index(name)].astype(float)

    d5 = -col("MaxEnt|donor|up_%d|nearest_dist@%g" % (DIST_CAP, ME_THR))
    d3 = -col("MaxEnt|acceptor|up_%d|nearest_dist@%g" % (DIST_CAP, ME_THR))

    zu = np.load(UCUBE, allow_pickle=True)
    me5 = zu["me5cube"]
    edges = zu["edges"]
    thrs = list(zu["me5_thrs"])
    ti = thrs.index(ME_THR)
    cnt5 = me5[:, :, 3 + ti].astype(float)
    lo_edge = edges[:-1]

    def band5(w):
        m = (lo_edge >= -w) & (lo_edge < 0)
        return cnt5[:, m].sum(axis=1)

    ci, ki, slot, n_slot, _ = build_pairs(gene, y)

    print("[in] %d peaks, %d canonical, %d competitor; %d genes with both, "
          "%d pairs" % (gene.size, int((y == 1).sum()), int((y == 0).sum()),
                        n_slot, ci.size), file=sys.stderr)
    print("[in] strong 5'ss present within %d nt upstream of the pA: "
          "canonical %.4f, competitor %.4f"
          % (DIST_CAP, np.isfinite(d5[y == 1]).mean(),
             np.isfinite(d5[y == 0]).mean()), file=sys.stderr)
    print("[in] strong 3'ss present: canonical %.4f, competitor %.4f"
          % (np.isfinite(d3[y == 1]).mean(), np.isfinite(d3[y == 0]).mean()),
          file=sys.stderr)

    # distances
    print()
    print("=" * 104)
    print("WITHIN-GENE DISTANCE  --  is the competitor's nearest strong site "
          "closer to the pA?")
    print("presence held constant: only pairs where both members have such a "
          "site inside %d nt" % DIST_CAP)
    print("=" * 104)
    print("%-38s %8s %8s %10s | %8s %10s %8s %8s"
          % ("test", "gene-avg", "pair-wt", "Wilcoxon p",
             "decided", "p", "tie", "n pairs"))
    r5 = decided(-d5, ci, ki, slot, n_slot, comp_hi=True)
    r3 = decided(-d3, ci, ki, slot, n_slot, comp_hi=True)
    line("5'ss nearer the pA in competitor", r5, True)
    line("3'ss nearer the pA in competitor", r3, True)

    for tag, d in (("5'ss", d5), ("3'ss", d3)):
        both = np.isfinite(d[ci]) & np.isfinite(d[ki])
        diff = d[ki][both] - d[ci][both]
        print("    %s median paired distance, canonical %.0f nt, competitor "
              "%.0f nt, median difference %+.0f nt"
              % (tag, np.median(d[ci][both]), np.median(d[ki][both]),
                 np.median(diff)))

    # bands
    print()
    print("=" * 104)
    print("WITHIN-GENE DENSITY  --  does the competitor carry MORE strong "
          "sites near the pA?")
    print("=" * 104)
    print("%-38s %8s %8s %10s | %8s %10s %8s %8s"
          % ("test", "gene-avg", "pair-wt", "Wilcoxon p",
             "decided", "p", "tie", "n pairs"))
    for w in (100, 200, 300):
        r = decided(band5(w), ci, ki, slot, n_slot, comp_hi=True)
        line("5'ss count in pA-%-4d (exact bands)" % w, r, True)
    for w in (100, 250, 500):
        v = col("MaxEnt|donor|up_%d|count_above@%g" % (w, ME_THR))
        line("5'ss count in pA-%-4d (cached)" % w,
             decided(v, ci, ki, slot, n_slot, comp_hi=True), True)
        v = col("MaxEnt|acceptor|up_%d|count_above@%g" % (w, ME_THR))
        line("3'ss count in pA-%-4d (cached)" % w,
             decided(v, ci, ki, slot, n_slot, comp_hi=True), True)

    # pooled dist
    BW = 50
    bins = np.arange(0, DIST_CAP + BW, BW)
    rows = []
    for cls, m in (("Canonical", y == 1), ("Competitor", y == 0)):
        v = d5[m]
        f = np.isfinite(v)
        h, _ = np.histogram(v[f], bins=bins)
        for lo, n in zip(bins[:-1], h):
            rows.append(dict(cls=cls, dist_lo=int(lo), dist_hi=int(lo + BW),
                             n=int(n), frac=float(n / f.sum()),
                             n_with_site=int(f.sum()), n_total=int(m.sum()),
                             frac_with_site=float(f.mean())))
    with open(OUT_DIST, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        wtr.writeheader()
        wtr.writerows(rows)
    print()
    print("pooled pA-to-nearest-strong-5'ss, peaks that have one within %d nt:"
          % DIST_CAP)
    for cls, m in (("Canonical", y == 1), ("Competitor", y == 0)):
        v = d5[m][np.isfinite(d5[m])]
        print("  %-11s n=%6d  median %5.0f  q25 %5.0f  q75 %5.0f  "
              "frac within 300 nt %.4f"
              % (cls, v.size, np.median(v), np.percentile(v, 25),
                 np.percentile(v, 75), (v <= 300).mean()))
    print("[out] %s" % OUT_DIST)


if __name__ == "__main__":
    main()
