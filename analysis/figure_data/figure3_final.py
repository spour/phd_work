#!/usr/bin/env python
# compute side of figure3_final: maxent vs spliceai on canonical-vs-competitor

import os
import sys
import csv
import time
import hashlib
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import pyBigWig
except ImportError:
    sys.exit("pyBigWig not installed. Run: pip install pyBigWig")

BASE_URL = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/"
BW_NAMES = {
    ("acceptor", "+"): "spliceAiAcceptorPlus.bw",
    ("acceptor", "-"): "spliceAiAcceptorMinus.bw",
    ("donor", "+"): "spliceAiDonorPlus.bw",
    ("donor", "-"): "spliceAiDonorMinus.bw",
}
BIGWIGS = {}
SIGNALS = ("acceptor", "donor")

REQUIRED_TSVS = (
    "ratio.tsv",
    "bcshow_ratio.tsv",
    "bcshow_auroc.tsv",
    "acc_x_interv.tsv",
    "metagene.tsv",
    "class_counts.tsv",
)

UP = 2000
DN = 500
FULL = UP + DN
WINDOWS = (250, 500, 1000, 1500, 2000)

MG_EDGES = np.arange(-500, 501, 40)
MG_CTRS = (MG_EDGES[:-1] + MG_EDGES[1:]) / 2.0

EXPECT_ACC_MAX = 0.740
EXPECT_DON_SUM = 0.675
TOL = 0.02

N_THREADS = 12
MIN_CELL = 200
N_DECILES = 10
HOLDOUT_SEED = 0
TERTILE_LABELS = ("weak", "medium", "strong")


def log(m):
    print(m, file=sys.stderr, flush=True)


# auroc


def auroc(scores, labels):
    ok = ~np.isnan(scores)
    s, y = scores[ok], labels[ok]
    n = s.size
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    newgrp = np.r_[True, ss[1:] != ss[:-1]]
    grp = np.cumsum(newgrp)
    starts = np.r_[np.nonzero(newgrp)[0], n]
    avg = (starts[:-1] + starts[1:] + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[grp - 1]
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), n


def direction(a):
    if np.isnan(a):
        return "undefined"
    return "canonical" if a > 0.5 else "COMPETITOR"


def read_peaks(path):
    rows = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        need = {"gene", "chrom", "center", "strand", "is_canonical"}
        missing = need - set(rdr.fieldnames or [])
        if missing:
            sys.exit("%s missing required columns: %s" % (path, sorted(missing)))
        for r in rdr:
            try:
                rows.append((r["gene"], r["chrom"], int(float(r["center"])),
                             r["strand"], int(float(r["is_canonical"]))))
            except (ValueError, TypeError):
                continue
    return rows


def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def open_bw(url, tries=4):
    for t in range(tries):
        try:
            return pyBigWig.open(url)
        except (RuntimeError, OSError):
            if t == tries - 1:
                return None
            time.sleep(2 ** t)
    return None


def oriented(bw, chrom, center, strand, clen):
    full = np.full(FULL, np.nan, dtype=np.float32)
    lo, hi = (center - UP, center + DN) if strand == "+" else (center - DN, center + UP)
    lo_c, hi_c = max(0, lo), min(hi, clen)
    if hi_c <= lo_c:
        return full
    try:
        raw = np.asarray(bw.values(chrom, lo_c, hi_c), dtype=np.float32)
    except (RuntimeError, OverflowError):
        return full
    if strand == "+":
        m_lo = center + DN - hi_c
        full[m_lo:m_lo + raw.size] = raw[::-1]
    else:
        m_lo = lo_c - center + DN
        full[m_lo:m_lo + raw.size] = raw
    return full


N_WIN = len(WINDOWS)
COL_ACC_MAX = {W: i for i, W in enumerate(WINDOWS)}
COL_DON_SUM = {W: N_WIN + i for i, W in enumerate(WINDOWS)}
COL_DON_MAX = 2 * N_WIN
N_COLS = 2 * N_WIN + 1


def make_worker(peaks, feats, mg_sum, mg_cnt):
    def run(job):
        slot, idxs = job
        handles = {}
        try:
            for k, i in enumerate(idxs):
                _g, chrom, center, strand, cls = peaks[i]
                for s_i, sig in enumerate(SIGNALS):
                    ent = handles.get((sig, strand))
                    if ent is None:
                        bw = open_bw(BIGWIGS[(sig, strand)])
                        if bw is None:
                            continue
                        ent = handles[(sig, strand)] = (bw, bw.chroms())
                    bw, sizes = ent
                    clen = sizes.get(chrom)
                    if clen is None:
                        continue
                    full = oriented(bw, chrom, center, strand, clen)
                    up = full[DN:]
                    for W in WINDOWS:
                        seg = up[:W]
                        cov = ~np.isnan(seg)
                        if not cov.any():
                            continue
                        v = seg[cov]
                        if sig == "acceptor":
                            feats[i, COL_ACC_MAX[W]] = v.max()
                        else:
                            feats[i, COL_DON_SUM[W]] = v.sum()
                            if W == 2000:
                                feats[i, COL_DON_MAX] = v.max()
                    good = ~np.isnan(full)
                    mg_sum[slot, s_i, cls][good] += full[good]
                    mg_cnt[slot, s_i, cls][good] += 1
                if k and k % 5000 == 0:
                    log("    slot %d: %d/%d" % (slot, k, len(idxs)))
        finally:
            for bw, _ in handles.values():
                bw.close()
        return len(idxs)
    return run


def scan(peaks):
    n = len(peaks)
    feats = np.full((n, N_COLS), np.nan, dtype=np.float64)
    order = sorted(range(n), key=lambda i: (peaks[i][1], peaks[i][2]))
    jobs = [(k, order[k::N_THREADS]) for k in range(N_THREADS)]
    jobs = [(k, c) for k, c in jobs if c]
    mg_sum = np.zeros((len(jobs), len(SIGNALS), 2, FULL), dtype=np.float64)
    mg_cnt = np.zeros((len(jobs), len(SIGNALS), 2, FULL), dtype=np.int64)
    done = 0
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for got in ex.map(make_worker(peaks, feats, mg_sum, mg_cnt), jobs):
            done += got
            log("  scanned %d/%d" % (done, n))
    return feats, mg_sum.sum(axis=0), mg_cnt.sum(axis=0)


def gene_split(genes):
    cache, out = {}, np.empty(len(genes), dtype=np.int8)
    for i, g in enumerate(genes):
        h = cache.get(g)
        if h is None:
            d = hashlib.md5(("%d:%s" % (HOLDOUT_SEED, g)).encode()).hexdigest()
            h = cache[g] = int(d[:8], 16) & 1
        out[i] = h
    return out


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--indir", default=".")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--prefix", default="fig3")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--threads", type=int, default=N_THREADS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    globals()["N_THREADS"] = max(1, args.threads)
    outdir = args.outdir or args.indir

    # dependency contract, checked before any work
    missing = []
    for name in REQUIRED_TSVS:
        p = os.path.join(args.indir, "%s_%s" % (args.prefix, name))
        if not os.path.isfile(p):
            missing.append(p)
    if not os.path.isfile(args.table):
        missing.append(args.table)
    for key, nm in BW_NAMES.items():
        local = os.path.join(args.bw_dir, nm)
        if args.remote:
            BIGWIGS[key] = BASE_URL + nm
        elif os.path.isfile(local):
            BIGWIGS[key] = local
        else:
            missing.append(local)
    if missing:
        print("MISSING REQUIRED INPUTS (%d):" % len(missing), file=sys.stderr)
        for m in missing:
            print("  %s" % m, file=sys.stderr)
        print("Run figure3_data.py first, or fetch the bigWigs into %s"
              % args.bw_dir, file=sys.stderr)
        sys.exit(1)
    log("dependency contract satisfied (%d MaxEnt tsvs, 4 bigWigs%s)"
        % (len(REQUIRED_TSVS), ", remote" if args.remote else ", local"))

    # maxent read-only
    bca = read_tsv(os.path.join(args.indir, "%s_bcshow_auroc.tsv" % args.prefix))
    cc = read_tsv(os.path.join(args.indir, "%s_class_counts.tsv" % args.prefix))
    n_can = n_comp = None
    for r in cc:
        if r.get("cls") == "Canonical":
            n_can = int(float(r["n"]))
        elif r.get("cls") == "Competitor":
            n_comp = int(float(r["n"]))
    don_rows = [r for r in bca if r["mech"] == "donor"]
    me_don = max((float(r["auroc"]) for r in don_rows), default=float("nan"))
    log("MaxEnt donor peak (oriented) = %.4f" % me_don)

    peaks = read_peaks(args.table)
    if args.limit and args.limit < len(peaks):
        step = len(peaks) / float(args.limit)
        peaks = [peaks[int(i * step)] for i in range(args.limit)]
    y = np.array([p[4] for p in peaks], dtype=np.int8)
    split = gene_split([p[0] for p in peaks])
    tr, te = split == 0, split == 1
    log("%d peaks (%d canonical / %d competitor), scanning SpliceAI once ..."
        % (len(peaks), int((y == 1).sum()), int((y == 0).sum())))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats, mg_sum, mg_cnt = scan(peaks)
    log("scan done in %.1f min" % ((time.time() - t0) / 60.0))

    sa_acc_max = feats[:, COL_ACC_MAX[2000]]
    sa_don_sum = feats[:, COL_DON_SUM[2000]]
    sa_don_max = feats[:, COL_DON_MAX]

    a_acc_te, _ = auroc(sa_acc_max[te], y[te])
    a_dsum_te, _ = auroc(sa_don_sum[te], y[te])
    a_dmax_te, _ = auroc(sa_don_max[te], y[te])
    a_acc_all, _ = auroc(sa_acc_max, y)

    log("SANITY")
    log("  train %d (%d canon) / test %d (%d canon)"
        % (int(tr.sum()), int((y[tr] == 1).sum()),
           int(te.sum()), int((y[te] == 1).sum())))
    log("  dropped all-NaN: acceptor %d, donor %d"
        % (int(np.isnan(sa_acc_max).sum()), int(np.isnan(sa_don_sum).sum())))
    log("  sa_acc_max AUROC: pooled %.4f, test %.4f  (expect ~%.3f)"
        % (a_acc_all, a_acc_te, EXPECT_ACC_MAX))
    log("  sa_don_sum AUROC: test %.4f              (expect ~%.3f)"
        % (a_dsum_te, EXPECT_DON_SUM))
    off = min(abs(a_acc_all - EXPECT_ACC_MAX), abs(a_acc_te - EXPECT_ACC_MAX))
    if off > TOL:
        msg = ("acceptor AUROC (pooled %.4f / test %.4f) is off %.3f by more than %.2f"
               % (a_acc_all, a_acc_te, EXPECT_ACC_MAX, TOL))
        if args.limit:
            log("  WARNING (subsampled --limit %d): %s" % (args.limit, msg))
        else:
            sys.exit("STOP: %s. Orientation or strand map is wrong." % msg)
    else:
        log("  -> orientation reproduces the locked encodings.")
    if abs(a_dsum_te - EXPECT_DON_SUM) > TOL:
        log("  WARNING: donor sum test AUROC %.4f drifts from %.3f."
            % (a_dsum_te, EXPECT_DON_SUM))

    p1 = os.path.join(outdir, "sa_auroc_window_final.tsv")
    with open(p1, "w") as fh:
        fh.write("signal\treduction\twin_far\tauroc_test\tn\n")
        for sig, red, colmap in (("acceptor_max", "max", COL_ACC_MAX),
                                 ("donor_sum", "sum", COL_DON_SUM)):
            for W in WINDOWS:
                a, n = auroc(feats[te, colmap[W]], y[te])
                fh.write("%s\t%s\t%d\t%.4f\t%d\n" % (sig, red, W, a, n))

    b = n_can / float(n_can + n_comp)
    ok = ~np.isnan(sa_acc_max) & ~np.isnan(sa_don_sum)

    def decile(v):
        edges = np.unique(np.quantile(v, np.linspace(0, 1, N_DECILES + 1)))
        idx = np.clip(np.searchsorted(edges, v, side="right") - 1,
                      0, max(0, len(edges) - 2))
        return idx, edges

    ab, aed = decile(sa_acc_max[ok])
    db, ded = decile(sa_don_sum[ok])
    yo = y[ok]
    p2 = os.path.join(outdir, "sa_acc_x_don_final.tsv")
    n_cells = 0
    with open(p2, "w") as fh:
        fh.write("acc_bin\tacc_lo\tacc_hi\tdon_bin\tdon_lo\tdon_hi\t"
                 "n\tn_canonical\tp_canonical\tlog2_or\n")
        for i in range(max(1, len(aed) - 1)):
            for j in range(max(1, len(ded) - 1)):
                m = (ab == i) & (db == j)
                n = int(m.sum())
                if n < MIN_CELL:
                    continue
                nc = int((yo[m] == 1).sum())
                p = min(max(nc / float(n), 1e-4), 1 - 1e-4)
                fh.write("%d\t%.6g\t%.6g\t%d\t%.6g\t%.6g\t%d\t%d\t%.4f\t%.4f\n"
                         % (i + 1, aed[i], aed[i + 1], j + 1, ded[j], ded[j + 1],
                            n, nc, nc / float(n),
                            np.log2((p / (1 - p)) / (b / (1 - b)))))
                n_cells += 1

    q = np.nanquantile(sa_acc_max[tr], [1 / 3.0, 2 / 3.0])
    tert = np.digitize(sa_acc_max, q)
    p3 = os.path.join(outdir, "sa_tertile_final.tsv")
    tert_rows = []
    with open(p3, "w") as fh:
        fh.write("acc_tertile\tacc_lo\tacc_hi\tdonor_reduction\tauroc_test\t"
                 "direction\tn\tn_canonical\n")
        for t_i, tlab in enumerate(TERTILE_LABELS):
            sel = te & (tert == t_i) & ~np.isnan(sa_acc_max)
            lo = -np.inf if t_i == 0 else q[t_i - 1]
            hi = np.inf if t_i == 2 else q[t_i]
            for red, vals in (("sum", sa_don_sum), ("max", sa_don_max)):
                a, n = auroc(vals[sel], y[sel])
                nc = int((y[sel] == 1).sum())
                fh.write("%s\t%.6g\t%.6g\t%s\t%.4f\t%s\t%d\t%d\n"
                         % (tlab, lo, hi, red, a, direction(a), n, nc))
                tert_rows.append((tlab, red, a, direction(a), n, nc))

    log("")
    log("CROSSOVER STATISTIC: donor AUROC within acceptor tertiles (TEST, raw)")
    log("  %-8s %-6s %9s  %-11s %8s %7s"
        % ("tertile", "donor", "auroc", "direction", "n", "n_canon"))
    for tlab, red, a, d, n, nc in tert_rows:
        log("  %-8s %-6s %9.4f  %-11s %8d %7d" % (tlab, red, a, d, n, nc))
    strong_sum = [r for r in tert_rows if r[0] == "strong" and r[1] == "sum"]
    if strong_sum:
        a = strong_sum[0][2]
        if a < 0.5:
            log("  -> strong-acceptor donor AUROC %.4f < 0.5: donor marks the "
                "COMPETITOR. Claim holds." % a)
        else:
            log("  -> WARNING: strong-acceptor donor AUROC %.4f >= 0.5. The "
                "'at high acceptor' claim does NOT hold as stated." % a)

    p4 = os.path.join(outdir, "sa_marginal_final.tsv")
    with open(p4, "w") as fh:
        fh.write("metric\tauroc_test\tdirection\n")
        for nm, a in (("acceptor_max", a_acc_te), ("donor_sum", a_dsum_te),
                      ("donor_max", a_dmax_te)):
            fh.write("%s\t%.4f\t%s\n" % (nm, a, direction(a)))
        fh.write("maxent_acceptor\t0.6060\tcanonical\n")
        fh.write("maxent_donor_oriented\t%.4f\tcompetitor\n" % me_don)

    pos_of_m = DN - np.arange(FULL)
    keep = (pos_of_m >= MG_EDGES[0]) & (pos_of_m < MG_EDGES[-1])
    binidx = np.clip(((pos_of_m[keep] - MG_EDGES[0]) // 40).astype(np.int64),
                     0, len(MG_CTRS) - 1)
    p5 = os.path.join(outdir, "sa_metagene_final.tsv")
    with open(p5, "w") as fh:
        fh.write("cls\tpos\tsa_acc_mean\tsa_don_mean\n")
        for cls, nm in ((1, "Canonical"), (0, "Competitor")):
            v = {}
            for s_i, sig in enumerate(SIGNALS):
                s = np.bincount(binidx, weights=mg_sum[s_i, cls][keep],
                                minlength=len(MG_CTRS))
                c = np.bincount(binidx, weights=mg_cnt[s_i, cls][keep],
                                minlength=len(MG_CTRS))
                v[sig] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            for k, ctr in enumerate(MG_CTRS):
                fh.write("%s\t%.1f\t%.6g\t%.6g\n"
                         % (nm, ctr, v["acceptor"][k], v["donor"][k]))

    log("")
    log("WROTE")
    for p in (p1, p2, p3, p4, p5):
        log("  %s" % p)
    log("panel G cells kept (>=%d): %d ; base rate b = %.5f (%d/%d)"
        % (MIN_CELL, n_cells, b, n_can, n_comp))


if __name__ == "__main__":
    main()
