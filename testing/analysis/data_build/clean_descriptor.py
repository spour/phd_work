#!/usr/bin/env python3
# which second descriptor separates real ends from competitors?
import sys
import numpy as np

FEAT = "fig3_new_positional_features.tsv"
U1_CACHE = "u1_grid_cache.npz"
U1_CELL = "up_500|density_per_kb@4"
ACC_THR = 8.0


# statistics
def wilcoxon_signed_rank(d):
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    n = nz.size
    if n < 10:
        return float("nan"), n
    a = np.abs(nz)
    order = np.argsort(a, kind="mergesort")
    a_s = a[order]
    rank = np.empty(n, float)
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and a_s[j + 1] == a_s[i]:
            j += 1
        rank[i:j + 1] = 0.5 * (i + j) + 1.0
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    r = np.empty(n, float)
    r[order] = rank
    w_pos = r[nz > 0].sum()
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return float("nan"), n
    z = (w_pos - mu) / np.sqrt(var)
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2.0))), n


def build_pairs(genes, y):
    order = np.argsort(genes, kind="mergesort")
    gs, ys = genes[order], y[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    ci, ki, slot, keep = [], [], [], []
    n_slot = 0
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        idx = order[lo:hi]
        c = idx[ys[lo:hi] == 1]
        k = idx[ys[lo:hi] == 0]
        if c.size == 0 or k.size == 0:
            continue
        cc, kk = np.meshgrid(c, k, indexing="ij")
        ci.append(cc.ravel())
        ki.append(kk.ravel())
        slot.append(np.full(cc.size, n_slot))
        keep.append(gs[lo])
        n_slot += 1
    return (np.concatenate(ci), np.concatenate(ki),
            np.concatenate(slot), n_slot, np.array(keep))


def win_rate(v, ci, ki, slot, n_slot):
    c, k = v[ci], v[ki]
    ok = np.isfinite(c) & np.isfinite(k)
    w = np.where(c > k, 1.0, np.where(c == k, 0.5, 0.0)) * ok
    num = np.bincount(slot, weights=w, minlength=n_slot)
    den = np.bincount(slot, weights=ok.astype(float), minlength=n_slot)
    m = den > 0
    return num[m] / den[m], float(num.sum() / den.sum()), int(den.sum())


# load
def main():
    raw = np.genfromtxt(FEAT, delimiter="\t", names=True, dtype=None,
                        encoding="utf-8", usecols=("gene", "is_canonical",
                                                   "acc", "first5_dn_present"))
    gene = raw["gene"].astype(str)
    y = raw["is_canonical"].astype(int)
    acc = raw["acc"].astype(float)
    dn5 = raw["first5_dn_present"].astype(float)

    z = np.load(U1_CACHE, allow_pickle=True)
    ucols = list(z["cols"])
    if U1_CELL not in ucols:
        sys.exit("u1 cache lacks %s" % U1_CELL)
    if not np.array_equal(z["gene"].astype(str), gene):
        sys.exit("gene order differs between %s and %s" % (FEAT, U1_CACHE))
    u1 = z["M"][:, ucols.index(U1_CELL)].astype(float)

    n = gene.size
    print("[in] %d peaks, %d canonical, %d competitor"
          % (n, int((y == 1).sum()), int((y == 0).sum())), file=sys.stderr)

    # the u1 cut
    u1_med_all = float(np.nanmedian(u1))
    strong = acc >= ACC_THR
    u1_med_strong = float(np.nanmedian(u1[strong]))
    print("[u1] median over all peaks %.2f sites/kb; over acc>=8 peaks %.2f"
          % (u1_med_all, u1_med_strong), file=sys.stderr)

    no5 = dn5 == 0
    lowu1 = u1 < u1_med_all

    # pooled
    print()
    print("=" * 74)
    print("POOLED  --  peaks with anchor acceptor MaxEnt >= %.0f only" % ACC_THR)
    print("=" * 74)
    nc = int((strong & (y == 1)).sum())
    nk = int((strong & (y == 0)).sum())
    print("strong-acceptor peaks: %d canonical (%.1f%% of canonical), "
          "%d competitor (%.1f%% of competitor)"
          % (nc, 100 * nc / (y == 1).sum(), nk, 100 * nk / (y == 0).sum()))
    print()
    print("%-34s %9s %9s %8s %8s" % ("descriptor", "canonical", "competitor",
                                     "ratio", "diff"))
    pooled = {}
    for name, mask in (("A  no strong intervening 5'ss", no5),
                       ("B  low PAS-proximal U1", lowu1)):
        fc = float(mask[strong & (y == 1)].mean())
        fk = float(mask[strong & (y == 0)].mean())
        pooled[name[0]] = (fc, fk)
        print("%-34s %9.4f %9.4f %8.3f %+8.4f"
              % (name, fc, fk, fc / fk if fk else float("nan"), fc - fk))

    clean_A = (strong & no5).astype(float)
    clean_B = (strong & lowu1).astype(float)

    ci, ki, slot, n_slot, gkeep = build_pairs(gene, y)
    print()
    print("=" * 74)
    print("WITHIN-GENE  --  is the real end 'clean' more often than its own "
          "competitors?")
    print("=" * 74)
    print("%d genes with both classes, %d canonical/competitor pairs"
          % (n_slot, ci.size))
    print()
    print("%-40s %9s %9s %11s" % ("descriptor", "gene-avg", "pair-wt",
                                  "Wilcoxon p"))
    wg = {}
    for name, v in (("A  strong 3'ss + no 5'ss", clean_A),
                    ("B  strong 3'ss + low U1", clean_B)):
        rates, pooled_rate, npair = win_rate(v, ci, ki, slot, n_slot)
        p, neff = wilcoxon_signed_rank(rates - 0.5)
        wg[name[0]] = (float(rates.mean()), pooled_rate, p)
        print("%-40s %9.4f %9.4f %11.2e   (n=%d informative genes)"
              % (name, rates.mean(), pooled_rate, p, neff))

    pA, pB = pooled["A"], pooled["B"]
    dA, dB = pA[0] - pA[1], pB[0] - pB[1]
    wA, wB = wg["A"][0], wg["B"][0]
    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("pooled:      %s separates more (%+.4f vs %+.4f canonical-minus-"
          "competitor)" % ("+ no 5'ss" if abs(dA) > abs(dB) else "+ low U1",
                           dA, dB))
    print("within-gene: %s separates more (%.4f vs %.4f gene-averaged "
          "win-rate)" % ("+ no 5'ss" if abs(wA - .5) > abs(wB - .5) else
                         "+ low U1", wA, wB))


if __name__ == "__main__":
    main()
