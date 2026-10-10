#!/usr/bin/env python
# step 1 only: compute, print, decide

import os
import sys
import csv
import math
import time
import argparse
import warnings

import numpy as np

STRONG_ME = 8.0
SA_THRESHOLDS = (0.2, 0.5, 0.8)
UP = 2000
DN = 2000
Z95 = 1.959963984540054
GATE_RATIO, GATE_P = 1.5, 1e-10
SA_CACHE = "sa_donor_presence_cache.tsv"
BW = {("donor", "+"): "spliceAiDonorPlus.bw", ("donor", "-"): "spliceAiDonorMinus.bw"}


def log(m):
    print(m, file=sys.stderr, flush=True)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_tsv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if not (p > 1e-300):
        return "< 1e-300"
    return "%.1e" % p


def chisq_2x2(a, b, c, d):
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    if min(r1, r2, c1, c2) == 0:
        return 0.0, 1.0
    chi = 0.0
    for obs, e in ((a, r1 * c1 / n), (b, r1 * c2 / n), (c, r2 * c1 / n), (d, r2 * c2 / n)):
        chi += (obs - e) ** 2 / e
    return chi, math.erfc(math.sqrt(chi / 2.0))


def logistic_or(flag, y):
    X = np.column_stack([np.ones(flag.size), flag.astype(float)])
    beta = np.zeros(2)
    for _ in range(60):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-10)
        z = eta + (y - mu) / w
        XtW = X.T * w
        new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-11:
            beta = new
            break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    w = np.maximum(mu * (1 - mu), 1e-10)
    se = np.sqrt(np.diag(np.linalg.inv((X.T * w) @ X)))
    b, s = beta[1], se[1]
    return math.exp(b), math.exp(b - Z95 * s), math.exp(b + Z95 * s)


# spliceai donor presence


def scan_donor_presence(rows, bw_dir, threads):
    try:
        import pyBigWig
    except ImportError:
        return None, "pyBigWig not installed"
    paths = {}
    for k, nm in BW.items():
        p = os.path.join(bw_dir, nm)
        if not os.path.isfile(p):
            return None, "missing %s" % p
        paths[k] = p
    from concurrent.futures import ThreadPoolExecutor
    peaks = [(r["chrom"], int(float(r["center"])), r["strand"]) for r in rows]
    n = len(peaks)
    out = np.full((n, 2), np.nan)

    def run(idxs):
        h = {}
        try:
            for i in idxs:
                chrom, center, strand = peaks[i]
                ent = h.get(strand)
                if ent is None:
                    bw = pyBigWig.open(paths[("donor", strand)])
                    ent = h[strand] = (bw, bw.chroms())
                bw, sizes = ent
                clen = sizes.get(chrom)
                if clen is None:
                    continue
                if strand == "+":
                    up = (center - UP, center)
                    dn = (center, center + DN)
                else:
                    up = (center, center + UP)
                    dn = (center - DN, center)
                for j, (lo, hi) in enumerate((up, dn)):
                    lo_c, hi_c = max(0, lo), min(hi, clen)
                    if hi_c <= lo_c:
                        continue
                    try:
                        v = np.asarray(bw.values(chrom, lo_c, hi_c), dtype=np.float32)
                    except (RuntimeError, OverflowError):
                        continue
                    v = v[~np.isnan(v)]
                    if v.size:
                        out[i, j] = float(v.max())
        finally:
            for bw, _ in h.values():
                bw.close()
        return len(idxs)

    order = sorted(range(n), key=lambda i: (peaks[i][0], peaks[i][1]))
    jobs = [order[k::threads] for k in range(threads)]
    jobs = [j for j in jobs if j]
    log("scanning donor bigWigs for per-region maxima (%d peaks) ..." % n)
    t0 = time.time()
    done = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            for got in ex.map(run, jobs):
                done += got
                log("  scanned %d/%d" % (done, n))
    log("scan done in %.1f min" % ((time.time() - t0) / 60.0))
    return out, None


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--indir", default=".")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--no-spliceai", action="store_true")
    args = ap.parse_args()

    tbl = args.table if os.path.isfile(args.table) else os.path.join(args.indir, args.table)
    for p in (tbl,):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)
    R = read_tsv(tbl)
    cols = set(R[0].keys())
    y = np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8)
    n_can, n_comp = int((y == 1).sum()), int((y == 0).sum())
    log("peaks %d (%d real ends / %d competitors)" % (len(R), n_can, n_comp))

    # what the spec asked for vs what exists
    log("")
    log("FIELD AVAILABILITY")
    have_iv = "interv_donor" in cols
    have_nd = "nearest_dn" in cols
    log("  interv_donor        : %s" % ("present" if have_iv else "ABSENT"))
    log("  nearest_dn          : %s" % ("present" if have_nd else "ABSENT"))
    log("  dn5_nearest_strong  : ABSENT  (named in the spec, exists in no pipeline output)")
    log("  anchor position     : ABSENT  (no column gives the anchor 3'ss coordinate,")
    log("                        so 'between anchor and pA' cannot be delimited de novo)")

    rows = []

    def add(scorer, region, flag, note):
        a = int(((y == 1) & flag).sum())
        b = n_can - a
        c = int(((y == 0) & flag).sum())
        d = n_comp - c
        fc, fk = a / n_can, c / n_comp
        ratio = (fk / fc) if fc > 0 else float("nan")
        chi, p = chisq_2x2(a, b, c, d)
        orr, lo, hi = logistic_or(flag, y.astype(float))
        strong = (ratio >= GATE_RATIO) and (p < GATE_P)
        rows.append({
            "scorer": scorer, "region": region,
            "frac_canonical_present": fc, "frac_competitor_present": fk,
            "ratio_comp_over_canon": ratio, "chisq": chi, "p": p,
            "or_canonical_given_present": orr, "or_lo": lo, "or_hi": hi,
            "n_canonical_present": a, "n_competitor_present": c,
            "n_total": len(R),
            "verdict": "STRONG" if strong else "WEAK (supplement-grade)",
            "note": note,
        })

    # maxent
    if have_iv:
        iv = np.array([fnum(r["interv_donor"]) for r in R])
        add("MaxEnt", "intervening (anchor to pA)",
            np.isfinite(iv) & (iv >= STRONG_ME),
            "interv_donor >= %.1f; NaN (%d) counted as absent"
            % (STRONG_ME, int(np.isnan(iv).sum())))
    if have_nd:
        nd = np.array([fnum(r["nearest_dn"]) for r in R])
        add("MaxEnt", "past pA (proxy: any strong donor downstream)",
            np.isfinite(nd),
            "nearest_dn finite; reference point is the anchor, not the pA, so this "
            "is a proxy and not strictly 'past pA'")
        add("MaxEnt", "internal-exon-like (50-300nt from anchor)",
            np.isfinite(nd) & (nd >= 50) & (nd <= 300),
            "nearest_dn in [50,300]")
    if have_iv and have_nd:
        add("MaxEnt", "either region",
            (np.isfinite(iv) & (iv >= STRONG_ME)) | np.isfinite(nd), "union of the two above")

    sa_note = None
    if args.no_spliceai:
        sa_note = "skipped by --no-spliceai"
    else:
        cache_p = os.path.join(args.indir, SA_CACHE)
        sa = None
        if os.path.isfile(cache_p):
            C = read_tsv(cache_p)
            if len(C) == len(R):
                sa = np.column_stack([[fnum(r["don_max_up"]) for r in C],
                                      [fnum(r["don_max_dn"]) for r in C]])
                log("reused %s" % SA_CACHE)
        if sa is None:
            log("")
            log("sa_peak_scalars_final.tsv holds only sa_acc_max and sa_don_sum, with no")
            log("position, so donor PRESENCE cannot be derived from it. Falling back to")
            log("re-reading the donor bigWigs (the documented fallback).")
            if not {"chrom", "center", "strand"} <= cols:
                sa_note = "peak table lacks chrom/center/strand"
            else:
                sa, err = scan_donor_presence(R, args.bw_dir, max(1, args.threads))
                if sa is None:
                    sa_note = err
                else:
                    with open(cache_p, "w") as fh:
                        fh.write("gene\tdon_max_up\tdon_max_dn\n")
                        for i, r in enumerate(R):
                            fh.write("%s\t%.6g\t%.6g\n" % (r["gene"], sa[i, 0], sa[i, 1]))
                    log("cached -> %s" % cache_p)
        if sa is not None:
            for t in SA_THRESHOLDS:
                add("SpliceAI", "upstream 2 kb, donor >= %.1f" % t,
                    np.nan_to_num(sa[:, 0], nan=-1) >= t,
                    "SUPERSET of 'intervening': without an anchor coordinate the "
                    "upstream window cannot be cut at the anchor")
                add("SpliceAI", "downstream 2 kb (past pA), donor >= %.1f" % t,
                    np.nan_to_num(sa[:, 1], nan=-1) >= t,
                    "strictly past the pA site")
    if sa_note:
        log("SpliceAI rows NOT COMPUTED: %s" % sa_note)

    out = os.path.join(args.indir, "downstream5_stats.tsv")
    keys = ["scorer", "region", "frac_canonical_present", "frac_competitor_present",
            "ratio_comp_over_canon", "chisq", "p", "or_canonical_given_present",
            "or_lo", "or_hi", "n_canonical_present", "n_competitor_present",
            "n_total", "verdict", "note"]
    with open(out, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(
                ("%.6g" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    log("")
    log("=" * 108)
    log("PRESENCE OF A STRONG 5'ss, BY CLASS   (gate: ratio >= %.1fx AND p < %g)"
        % (GATE_RATIO, GATE_P))
    log("=" * 108)
    log("  %-9s %-46s %8s %8s %7s %11s %8s  %s"
        % ("scorer", "region", "canon", "comp", "ratio", "p", "OR", "verdict"))
    for r in rows:
        log("  %-9s %-46s %8.4f %8.4f %7.3f %11s %8.3f  %s"
            % (r["scorer"], r["region"], r["frac_canonical_present"],
               r["frac_competitor_present"], r["ratio_comp_over_canon"],
               p_label(r["p"]), r["or_canonical_given_present"], r["verdict"]))
    log("")
    n_strong = sum(1 for r in rows if r["verdict"] == "STRONG")
    log("DECISION: %d of %d rows clear the gate." % (n_strong, len(rows)))
    if n_strong == 0:
        log("  No row reaches %.1fx. STEP 2 is not justified on these numbers."
            % GATE_RATIO)
    log("wrote %s" % out)


if __name__ == "__main__":
    main()
