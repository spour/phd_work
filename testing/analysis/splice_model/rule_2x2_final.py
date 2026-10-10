#!/usr/bin/env python
# mechanistic decision-rule 2x2 from existing maxent outputs

import os
import sys
import csv
import math
import argparse

STRONG_ACC = 8.0
STRICT_ACC = 10.0
IEL_LO, IEL_HI = 50.0, 300.0
PUBLISHED_IEL = {"Canonical": 0.163755, "Competitor": 0.194855}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        cols = set(rdr.fieldnames or [])
        need = {"is_canonical", "acc", "nearest_dn"}
        missing = need - cols
        if missing:
            sys.exit("%s lacks required columns: %s\nHas: %s"
                     % (path, sorted(missing), sorted(cols)))
        rows = []
        for r in rdr:
            y = fnum(r["is_canonical"])
            a = fnum(r["acc"])
            d = fnum(r["nearest_dn"])
            if math.isnan(y) or math.isnan(a):
                continue
            rows.append((int(y), a, d))
    return rows


def pct(x):
    return "%.1f%%" % (100.0 * x)


def cell_stats(rows, keep, base):
    sel = [r for r in rows if keep(r)]
    n = len(sel)
    nc = sum(1 for r in sel if r[0] == 1)
    nk = n - nc
    p = nc / n if n else float("nan")
    return n, nc, nk, p, (p / base if n and base else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_peak_features.tsv")
    ap.add_argument("--indir", default=".")
    args = ap.parse_args()

    path = args.table if os.path.isfile(args.table) \
        else os.path.join(args.indir, args.table)
    if not os.path.isfile(path):
        sys.exit("missing input: %s" % path)

    rows = load(path)
    N = len(rows)
    NC = sum(1 for r in rows if r[0] == 1)
    NK = N - NC
    base = NC / float(N)

    def strong(r, thr=STRONG_ACC):
        return r[1] >= thr

    def has_dn5(r):
        return math.isfinite(r[2]) and IEL_LO <= r[2] <= IEL_HI

    n_nd_missing = sum(1 for r in rows if not math.isfinite(r[2]))

    print("=" * 74)
    print("Decision-rule 2x2 from existing MaxEnt outputs")
    print("=" * 74)
    print("source            : %s" % path)
    print("peaks             : %d  (%d canonical / %d competitor)" % (N, NC, NK))
    print("base rate         : %.5f  (%s)" % (base, pct(base)))
    print("strong_acc        : acc >= %.1f" % STRONG_ACC)
    print("has_dn5           : %.0f <= nearest_dn <= %.0f" % (IEL_LO, IEL_HI))
    print("nearest_dn absent : %d peaks (%s), counted as no_dn5"
          % (n_nd_missing, pct(n_nd_missing / float(N))))
    print()

    # definition check against the published class fractions
    print("CHECK: has_dn5 vs published internal_exon_like fractions")
    worst = 0.0
    for cls, lab in ((1, "Canonical"), (0, "Competitor")):
        idx = [r for r in rows if r[0] == cls]
        got = sum(1 for r in idx if has_dn5(r)) / float(len(idx))
        pub = PUBLISHED_IEL[lab]
        worst = max(worst, abs(got - pub))
        print("  %-11s derived %.6f   published %.6f   diff %+.6f"
              % (lab, got, pub, got - pub))
    if worst > 1e-4:
        print("  MISMATCH. The published fractions come from first5_from_anchor")
        print("  (distance from the anchor 3'ss, both directions), which carries no")
        print("  per-peak key and cannot be joined to acc. nearest_dn is distance")
        print("  downstream of the peak's own site. Cell counts below will not equal")
        print("  fig3_internal_exon_like.tsv. The direction of the result is robust")
        print("  to this (see the sensitivity block at the end); the exact percentages")
        print("  are not. Do not quote these against the published fractions.")
    else:
        print("  match")
    print()

    # the 2x2
    print("2x2  (rows: acceptor; cols: downstream 5'ss at %.0f-%.0f nt)"
          % (IEL_LO, IEL_HI))
    print("  %-24s %8s %8s %8s %10s %8s"
          % ("cell", "n", "canon", "comp", "P(canon)", "fold"))
    cells = [
        ("strong_acc & no_dn5", lambda r: strong(r) and not has_dn5(r)),
        ("strong_acc & has_dn5", lambda r: strong(r) and has_dn5(r)),
        ("weak_acc   & no_dn5", lambda r: not strong(r) and not has_dn5(r)),
        ("weak_acc   & has_dn5", lambda r: not strong(r) and has_dn5(r)),
    ]
    for nm, fn in cells:
        n, nc, nk, p, fold = cell_stats(rows, fn, base)
        print("  %-24s %8d %8d %8d %10.4f %7.2fx" % (nm, n, nc, nk, p, fold))
    print()

    # rule evaluation
    def evaluate(nm, fn):
        tp = sum(1 for r in rows if fn(r) and r[0] == 1)
        fp = sum(1 for r in rows if fn(r) and r[0] == 0)
        fn_ = NC - tp
        tn = NK - fp
        prec = tp / float(tp + fp) if tp + fp else float("nan")
        rec = tp / float(NC) if NC else float("nan")
        spec = tn / float(tn + fp) if tn + fp else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")
        print("RULE: %s" % nm)
        print("  confusion    TP %6d   FP %6d" % (tp, fp))
        print("               FN %6d   TN %6d" % (fn_, tn))
        print("  precision  P(canonical | rule positive) : %.4f  (%s)" % (prec, pct(prec)))
        print("  recall     fraction of ALL canonicals   : %.4f  (%s)" % (rec, pct(rec)))
        print("  specificity                             : %.4f  (%s)" % (spec, pct(spec)))
        print("  F1                                      : %.4f" % f1)
        print("  enrichment over base rate               : %.2fx" % (prec / base))
        print("  rule positives                          : %d (%s of all peaks)"
              % (tp + fp, pct((tp + fp) / float(N))))
        print()
        return prec, rec, prec / base

    p1, r1, f1x = evaluate("strong_acc (>= %.1f) AND NOT has_dn5" % STRONG_ACC,
                           lambda r: strong(r) and not has_dn5(r))
    p2, r2, f2x = evaluate("strict: acc >= %.1f AND NOT has_dn5" % STRICT_ACC,
                           lambda r: strong(r, STRICT_ACC) and not has_dn5(r))

    # sensitivity: does the conclusion survive the has_dn5 ambiguity?
    print("SENSITIVITY of the main rule to the has_dn5 window")
    print("  %-18s %8s %10s %10s %8s" % ("window (nt)", "pos", "precision", "recall", "fold"))
    for lo, hi in ((50, 300), (50, 500), (70, 300), (100, 300), (50, 200)):
        fn = lambda r, lo=lo, hi=hi: strong(r) and not (
            math.isfinite(r[2]) and lo <= r[2] <= hi)
        tp = sum(1 for r in rows if fn(r) and r[0] == 1)
        fp = sum(1 for r in rows if fn(r) and r[0] == 0)
        pr = tp / float(tp + fp) if tp + fp else float("nan")
        rc = tp / float(NC)
        print("  %-18s %8d %10.4f %10.4f %7.2fx"
              % ("[%d,%d]" % (lo, hi), tp + fp, pr, rc, pr / base))
    print()

    print("-" * 74)
    print("strong acceptor + no downstream 5'ss captures %s of true ends at %s "
          "precision (%.2f-fold over base rate)." % (pct(r1), pct(p1), f1x))
    print("-" * 74)


if __name__ == "__main__":
    main()
