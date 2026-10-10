#!/usr/bin/env python
# does spliceai's context-aware acceptor score beat the hand-built maxent acceptor marginal

import os
import sys
import csv
import math
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

try:
    import pyBigWig
except ImportError:
    sys.exit("pyBigWig not installed. Run: pip install pyBigWig")

BASE = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/"
FILES = {
    ("acc", "+"): "spliceAiAcceptorPlus.bw",
    ("acc", "-"): "spliceAiAcceptorMinus.bw",
    ("don", "+"): "spliceAiDonorPlus.bw",
    ("don", "-"): "spliceAiDonorMinus.bw",
}
BIGWIGS = {}

WINDOW = 2000
BASELINE = 0.606
N_THREADS = 8


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# statistics


def _midranks(xs):
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auroc(scores, labels):
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _midranks(scores)
    rsum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rsum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    rx, ry = _midranks(xs), _midranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def quantile(sorted_xs, q):
    if not sorted_xs:
        return float("nan")
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def read_peaks(path):
    rows = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        need = {"chrom", "center", "strand", "is_canonical", "acc"}
        missing = need - set(rdr.fieldnames or [])
        if missing:
            sys.exit("missing required columns: %s" % sorted(missing))
        for r in rdr:
            try:
                rows.append(
                    {
                        "chrom": r["chrom"],
                        "center": int(float(r["center"])),
                        "strand": r["strand"],
                        "is_canonical": int(float(r["is_canonical"])),
                        "acc": float(r["acc"]),
                    }
                )
            except (ValueError, TypeError):
                continue
    return rows


def window_for(center, strand):
    if strand == "+":
        return max(0, center - WINDOW), center
    return center + 1, center + 1 + WINDOW


def open_bw(url, tries=4):
    for t in range(tries):
        try:
            return pyBigWig.open(url)
        except (RuntimeError, OSError):
            if t == tries - 1:
                return None
            time.sleep(2 ** t)
    return None


def make_worker(kind):

    def run(chunk):
        handles = {}
        out = []
        try:
            for idx, row in chunk:
                strand = row["strand"]
                url = BIGWIGS.get((kind, strand))
                if url is None:
                    out.append((idx, None))
                    continue
                ent = handles.get(strand)
                if ent is None:
                    bw = open_bw(url)
                    if bw is None:
                        out.append((idx, None))
                        continue
                    ent = handles[strand] = (bw, bw.chroms())
                bw, chrom_sizes = ent
                chrom = row["chrom"]
                clen = chrom_sizes.get(chrom)
                if clen is None:
                    out.append((idx, None))
                    continue
                start, end = window_for(row["center"], strand)
                end = min(end, clen)
                if end <= start:
                    out.append((idx, None))
                    continue
                try:
                    v = bw.stats(chrom, start, end, type="max")
                    v = v[0] if v else None
                except (RuntimeError, OverflowError):
                    v = None
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    out.append((idx, None))
                else:
                    out.append((idx, float(v)))
        finally:
            for bw, _ in handles.values():
                bw.close()
        return out

    return run


def score_all(rows, kind):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["chrom"], rows[i]["center"]))
    chunks = [[] for _ in range(N_THREADS)]
    for n, i in enumerate(order):
        chunks[n % N_THREADS].append((i, rows[i]))
    chunks = [c for c in chunks if c]

    vals = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        for res in ex.map(make_worker(kind), chunks):
            for idx, v in res:
                vals[idx] = v
            done += len(res)
            log("  [%s] %d/%d scored" % (kind, done, len(rows)))
    return vals


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: evenly sample N peaks across the table")
    ap.add_argument("--threads", type=int, default=N_THREADS,
                    help="concurrent bigWig readers")
    ap.add_argument("--bw-dir", default="spliceai_bw",
                    help="dir holding local .bw copies; missing ones read remotely")
    ap.add_argument("--remote", action="store_true",
                    help="force remote reads even if local copies exist")
    args = ap.parse_args()
    globals()["N_THREADS"] = max(1, args.threads)

    log("SpliceAI bigWigs (UCSC spliceAIWt, hg38); remote base %s" % BASE)
    n_local = 0
    for key, name in sorted(FILES.items()):
        local = os.path.join(args.bw_dir, name)
        if not args.remote and os.path.isfile(local):
            BIGWIGS[key] = local
            n_local += 1
            src = "local"
        else:
            BIGWIGS[key] = BASE + name
            src = "remote"
        log("  %s %s -> %s (%s)" % (key[0], key[1], name, src))
    if n_local < len(FILES):
        log("NOTE: %d/%d read remotely. UCSC drops connections long before a full"
            % (len(FILES) - n_local, len(FILES)))
        log("      70k-peak run completes; to fetch local copies (~870MB each):")
        log("      mkdir -p %s && cd %s && for f in %s; do wget -c %s$f; done"
            % (args.bw_dir, args.bw_dir, " ".join(sorted(FILES.values())), BASE))

    log("reading %s ..." % args.table)
    rows = read_peaks(args.table)
    if args.limit and args.limit < len(rows):
        step = len(rows) / float(args.limit)
        rows = [rows[int(i * step)] for i in range(args.limit)]
    log("%d peaks" % len(rows))

    log("querying acceptor bigWigs (2000nt upstream max, strand-aware) ...")
    acc_vals = score_all(rows, "acc")
    log("querying donor bigWigs ...")
    don_vals = score_all(rows, "don")

    n_total = len(rows)
    n_canon = sum(1 for r in rows if r["is_canonical"] == 1)
    n_comp = n_total - n_canon

    keep = [i for i in range(n_total) if acc_vals[i] is not None]
    n_drop = n_total - len(keep)

    s_acc = [acc_vals[i] for i in keep]
    y = [rows[i]["is_canonical"] for i in keep]
    my_acc = [rows[i]["acc"] for i in keep]

    a_acc = auroc(s_acc, y)

    keep_d = [i for i in range(n_total) if don_vals[i] is not None]
    s_don = [don_vals[i] for i in keep_d]
    y_don = [rows[i]["is_canonical"] for i in keep_d]
    a_don = auroc(s_don, y_don)

    rho = spearman(s_acc, my_acc)
    ss = sorted(s_acc)

    print("=" * 68)
    print("SpliceAI acceptor anchor check (hg38, UCSC spliceAIWt wildtype)")
    print("=" * 68)
    print()
    print("COUNTS")
    print("  peaks total          : %d" % n_total)
    print("  canonical (pos)      : %d" % n_canon)
    print("  competitor (neg)     : %d" % n_comp)
    print("  dropped, no coverage : %d (%.2f%%)"
          % (n_drop, 100.0 * n_drop / n_total if n_total else 0.0))
    print("  scored for AUROC     : %d" % len(keep))
    print()
    print("DISTRIBUTION of spliceai_acc_max (window max, %dnt upstream)" % WINDOW)
    print("  min    : %.4f" % (ss[0] if ss else float("nan")))
    print("  median : %.4f" % quantile(ss, 0.50))
    print("  p90    : %.4f" % quantile(ss, 0.90))
    print("  max    : %.4f" % (ss[-1] if ss else float("nan")))
    print()
    print("EXAMPLE ROWS")
    print("  %-6s %-11s %-6s %-5s %-10s %s"
          % ("chrom", "center", "strand", "canon", "spliceai", "maxent_acc"))
    for i in keep[:5]:
        r = rows[i]
        print("  %-6s %-11d %-6s %-5d %-10.4f %.4f"
              % (r["chrom"], r["center"], r["strand"], r["is_canonical"],
                 acc_vals[i], r["acc"]))
    print()
    print("COORDINATE SANITY")
    print("  Spearman(spliceai_acc_max, maxent acc) = %.4f" % rho)
    print("  (clearly positive = windows aligned; ~0 = coordinate/strand bug)")
    print()
    print("AUROC vs is_canonical (positives = canonical)")
    print("  SpliceAI acceptor-max : %.4f" % a_acc)
    print("  MaxEnt baseline       : %.4f" % BASELINE)
    print("  delta                 : %+.4f" % (a_acc - BASELINE))
    print()
    print("BONUS: SpliceAI donor-max (donor marks the competitor, so 1-AUROC too)")
    print("  scored                : %d (dropped %d)" % (len(keep_d), n_total - len(keep_d)))
    print("  donor-max AUROC       : %.4f" % a_don)
    print("  1 - AUROC             : %.4f" % (1.0 - a_don))
    print()

    if math.isnan(a_acc):
        verdict = "UNDEFINED (need both classes present)"
    elif a_acc > BASELINE + 0.005:
        verdict = "BEATS"
    elif a_acc < BASELINE - 0.005:
        verdict = "BELOW"
    else:
        verdict = "TIES"
    print("SpliceAI acceptor-max AUROC = %.3f vs MaxEnt baseline %.3f -> [%s]"
          % (a_acc, BASELINE, verdict))


if __name__ == "__main__":
    main()
