#!/usr/bin/env python3
# item 1: recalibrate the u1 "strong site" threshold with an empirical null + fdr
import sys, csv
import numpy as np

FASTA   = "../train/data/hg38.fa"
U1NPY   = "../train/splice_train/data/u1_lookup.npy"
META    = "fig3_ucube_meta.tsv"
WIN     = 2000
OLD_THR = 4.0
NULL_N  = 10000
SEED    = 42

HEDGES  = np.arange(-6.0, 12.0001, 0.05)
SWEEP   = np.round(np.arange(1.0, 10.0001, 0.25), 3)

def log(*a): print(*a, file=sys.stderr, flush=True)

# lookup -> dense score array indexed by base-4 9-mer code (a=0 c=1 g=2 t=3)
B = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): B[ch] = v
POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)

def encode_kmers(byte_mat):
    codes = B[byte_mat]
    bad = (codes < 0).any(1)
    idx = codes @ POW
    idx[bad] = -1
    return idx

log("[load] U1 lookup ...")
arr = np.load(U1NPY, allow_pickle=True)
d = arr.item() if arr.dtype == object else dict(arr)
keys = np.frombuffer("".join(d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
kcode = encode_kmers(keys)
SCORE = np.empty(4 ** 9, np.float64)
SCORE[kcode] = -np.fromiter(d.values(), np.float64, len(d))
log(f"[load]   {len(d):,} 9-mers ; score(-dG) min {SCORE.min():.3f} max {SCORE.max():.3f} "
    f"mean {SCORE.mean():.4f} sd {SCORE.std():.4f}")

lookup_hist, _ = np.histogram(SCORE, bins=HEDGES)
def tail_frac(counts, edges, t):
    ctr = (edges[:-1] + edges[1:]) / 2
    return counts[ctr >= t].sum() / max(counts.sum(), 1)
P_uniform = np.array([tail_frac(lookup_hist, HEDGES, t) for t in SWEEP])

# dinucleotide-preserving shuffle (altschul-erikson
def dinuc_shuffle(tok, rng):
    chars, toks = np.unique(tok, return_inverse=True)
    nxt = []
    for t in range(len(chars)):
        inds = np.where(toks[:-1] == t)[0] + 1
        nxt.append(inds)
    for t in range(len(chars)):
        m = len(nxt[t])
        if m > 1:
            perm = np.arange(m)
            perm[:-1] = rng.permutation(m - 1)
            nxt[t] = nxt[t][perm]
    counters = [0] * len(chars)
    ind = 0
    out = np.empty_like(toks)
    out[0] = toks[0]
    for j in range(1, len(toks)):
        t = toks[ind]
        ind = nxt[t][counters[t]]
        counters[t] += 1
        out[j] = toks[ind]
    return chars[out]

# scan a token array, return scores at valid 9-mer starts
from numpy.lib.stride_tricks import sliding_window_view
def scores_of(tok):
    if tok.shape[0] < 9: return np.empty(0)
    sw = sliding_window_view(tok, 9)
    good = (sw >= 0).all(1)
    if not good.any(): return np.empty(0)
    return SCORE[sw[good] @ POW]

# peaks
log("[load] meta + genome ...")
M = []
with open(META) as fh:
    r = csv.DictReader(fh, delimiter="\t")
    for row in r:
        M.append((row["chrom"], int(row["center"]), row["strand"], int(row["is_canonical"])))
log(f"[load]   {len(M):,} peaks")

sys.path.insert(0, "scripts")
from extract_splicing_features import GenomeFasta
G = GenomeFasta(FASTA)

rng = np.random.default_rng(SEED)
sample = set(rng.choice(len(M), size=min(NULL_N, len(M)), replace=False).tolist())

obs_hist  = np.zeros(len(HEDGES) - 1, np.int64)
null_hist = np.zeros(len(HEDGES) - 1, np.int64)
obs_s_hist = np.zeros(len(HEDGES) - 1, np.int64)
n_scored = n_null = 0
n_skip_N = 0
COMP = str.maketrans("ACGT", "TGCA")

for i, (chrom, center, strand, isc) in enumerate(M):
    if strand == "-":
        seq = G.fetch(chrom, center, center + WIN, "-")
    else:
        seq = G.fetch(chrom, center - WIN, center, "+")
    if not seq: continue
    tok = B[np.frombuffer(seq.encode("ascii"), np.uint8)]
    sc = scores_of(tok)
    if sc.size:
        obs_hist += np.histogram(sc, bins=HEDGES)[0]
        n_scored += sc.size
    if i in sample:
        if (tok < 0).any():
            n_skip_N += 1; continue
        obs_s_hist += np.histogram(sc, bins=HEDGES)[0]
        shuf = dinuc_shuffle(tok, rng)
        scs = scores_of(shuf.astype(np.int64))
        if scs.size:
            null_hist += np.histogram(scs, bins=HEDGES)[0]
            n_null += scs.size
    if (i + 1) % 10000 == 0:
        log(f"  {i+1:,}/{len(M):,}")

log(f"[scan] observed positions {n_scored:,} ; null positions {n_null:,} "
    f"(sample {len(sample):,}, skipped-N {n_skip_N})")

def tail_rate(counts, t):
    ctr = (HEDGES[:-1] + HEDGES[1:]) / 2
    return counts[ctr >= t].sum() / max(counts.sum(), 1)

obs_rate    = np.array([tail_rate(obs_s_hist, t) for t in SWEEP])
null_rate_a = np.array([tail_rate(null_hist,  t) for t in SWEEP])
fdr_a = np.where(obs_rate > 0, null_rate_a / obs_rate, np.nan)
fdr_b = np.where(obs_rate > 0, P_uniform   / obs_rate, np.nan)

def thr_at(fdr, target):
    ok = np.where(fdr <= target)[0]
    return float(SWEEP[ok[0]]) if ok.size else float("nan")
targets = [0.10, 0.05, 0.01]
chosen = {q: thr_at(fdr_a, q) for q in targets}
CHOSEN = chosen[0.05]

obs_ctr = (HEDGES[:-1] + HEDGES[1:]) / 2
def hits_ge(t): return int(obs_hist[obs_ctr >= t].sum())
h_old = hits_ge(OLD_THR); h_new = hits_ge(CHOSEN)
NP = len(M)

ctr = (HEDGES[:-1] + HEDGES[1:]) / 2
with open("u1_score_hist.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["score", "lookup", "observed", "dinuc_null"])
    lo = lookup_hist / max(lookup_hist.sum(), 1)
    oo = obs_hist   / max(obs_hist.sum(), 1)
    no = null_hist  / max(null_hist.sum(), 1)
    for c, a1, a2, a3 in zip(ctr, lo, oo, no):
        w.writerow([f"{c:.3f}", f"{a1:.8f}", f"{a2:.8f}", f"{a3:.8f}"])

with open("u1_threshold.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["threshold", "z", "tail_frac_uniform", "obs_rate", "null_rate_dinuc",
                "fdr_dinuc", "fdr_uniform"])
    mu, sd = SCORE.mean(), SCORE.std()
    for j, t in enumerate(SWEEP):
        w.writerow([f"{t:.3f}", f"{(t-mu)/sd:.4f}", f"{P_uniform[j]:.8f}",
                    f"{obs_rate[j]:.8f}", f"{null_rate_a[j]:.8f}",
                    f"{fdr_a[j]:.6f}", f"{fdr_b[j]:.6f}"])
    w.writerow([])
    w.writerow(["# chosen thresholds (dinuc-shuffle FDR)"])
    for q in targets: w.writerow([f"# FDR<={q:.2f}", f"thr={chosen[q]:.3f}"])
    w.writerow([f"# OLD_THR={OLD_THR}", f"NEW_THR(FDR0.05)={CHOSEN}"])
    w.writerow([f"# hits_old={h_old}", f"hits_new={h_new}",
                f"fold_reduction={h_old/max(h_new,1):.3f}"])
    w.writerow([f"# mean_hits_per_peak_old={h_old/NP:.3f}",
                f"mean_hits_per_peak_new={h_new/NP:.3f}"])
    w.writerow([f"# score_mean={mu:.4f}", f"score_sd={sd:.4f}",
                f"win_upstream={WIN}", f"n_peaks={NP}"])

mu, sd = SCORE.mean(), SCORE.std()
print("\n================  U1 THRESHOLD CALIBRATION  ================")
print(f"score(-dG) over all {len(d):,} 9-mers : mean {mu:.4f}  sd {sd:.4f}")
print(f"window: {WIN} nt upstream of site ; {NP:,} peaks ; {n_scored:,} scored 9-mer positions")
print(f"dinuc-shuffle null on {len(sample)-n_skip_N:,} peaks ({n_null:,} positions)\n")
print(f"{'thr':>6} {'z':>6} {'unif_tail':>10} {'obs_rate':>10} {'null_rate':>10} {'FDR_dinuc':>10} {'FDR_unif':>9}")
for j, t in enumerate(SWEEP):
    if t < 2.0 or t > 9.0: continue
    mark = ""
    if abs(t - OLD_THR) < 1e-9: mark = "  <- OLD"
    if abs(t - CHOSEN) < 1e-9: mark += "  <- NEW(FDR.05)"
    print(f"{t:6.2f} {(t-mu)/sd:6.2f} {P_uniform[j]:10.5f} {obs_rate[j]:10.5f} "
          f"{null_rate_a[j]:10.5f} {fdr_a[j]:10.4f} {fdr_b[j]:9.4f}{mark}")
print("\nchosen (dinuc-shuffle FDR):")
for q in targets: print(f"  FDR <= {q:.2f}  ->  threshold {chosen[q]:.2f}")
print(f"\nOLD threshold {OLD_THR}:  {h_old:,} hits  ({h_old/NP:.2f} per peak)")
print(f"NEW threshold {CHOSEN} (FDR0.05):  {h_new:,} hits  ({h_new/NP:.2f} per peak)")
print(f"fold reduction: {h_old/max(h_new,1):.2f}x")
print(f"\nat OLD 4.0 the dinuc-FDR is {fdr_a[np.argmin(np.abs(SWEEP-OLD_THR))]:.3f} "
      f"(fraction of hits explainable by composition)")
print("\n[out] u1_score_hist.tsv ; u1_threshold.tsv")
print("[note] nothing downstream changed. Awaiting sign-off before propagating.")
