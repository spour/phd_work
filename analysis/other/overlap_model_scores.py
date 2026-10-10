#!/usr/bin/env python3
# score every external site with the baseline models
import sys, csv, gzip
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
ROOT = "/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal"
WIN = 500

# gene intervals
GEN = {}
with open(f"{ROOT}/score_genome/pc_genes.bed") as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) < 6: continue
        GEN.setdefault(p[0], []).append((int(p[1]), int(p[2]), p[3]))
for c in GEN: GEN[c].sort()
GS = {c: np.array([g[0] for g in v]) for c, v in GEN.items()}
log(f"[genes] {sum(len(v) for v in GEN.values()):,} protein-coding gene intervals on "
    f"{len(GEN)} contigs")
def gene_at(chrom, pos):
    v = GEN.get(chrom)
    if not v: return None
    i = int(np.searchsorted(GS[chrom], pos, "right"))
    for j in range(max(0, i - 40), min(len(v), i + 1)):
        if v[j][0] <= pos <= v[j][1]: return v[j][2]
    return None

# candidate windows
import pyarrow.parquet as pq
P = pq.read_table("cpa_peaks_final.parquet").to_pandas()
R = pq.read_table(f"{ROOT}/clean/rerank_genomewide_scored.parquet").to_pandas()
Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
def index_by_chrom(chrom, pos, val):
    d = {}
    for c, x, v in zip(chrom, pos, val): d.setdefault(c, []).append((int(x), float(v)))
    out = {}
    for c, v in d.items():
        v.sort(); out[c] = (np.array([x for x, _ in v]), np.array([y for _, y in v]))
    return out
RFI = index_by_chrom(R["chrom"], R["center"], R["rf_prob"])
CNI = index_by_chrom(Z["chrom"], Z["center"], Z["logit"])
log(f"[windows] forest {len(R):,} windows on {len(RFI)} contigs; "
    f"CNN {len(Z['logit']):,} windows on {len(CNI)} contigs")
RFALL = np.asarray(R["rf_prob"], float); CNALL = np.asarray(Z["logit"], float)
RFS = np.sort(RFALL[np.isfinite(RFALL)]); CNS = np.sort(CNALL[np.isfinite(CNALL)])
def pct_of(v, srt):
    return np.nan if not np.isfinite(v) else 100.0 * np.searchsorted(srt, v, "left") / len(srt)

def max_over(idx, chrom, pos):
    e = idx.get(chrom)
    if e is None: return np.nan, 0
    p, v = e
    lo, hi = np.searchsorted(p, pos - WIN, "left"), np.searchsorted(p, pos + WIN, "right")
    if hi <= lo: return np.nan, 0
    w = v[lo:hi]
    return (np.nanmax(w) if np.any(np.isfinite(w)) else np.nan), hi - lo

# the site sets
SETS = {"atlas_3seq": "3'-seq IpA atlas (Singh + Lee)",
        "oh_u1amo": "U1 AMO (telescripting loss)",
        "pcf11_dep": "PCF11-dependent premature CPA"}
S = {}
for k in SETS:
    rows = []
    with open(f"hiconf_out/sites/{k}.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append((r["chrom"], int(r["pos"]), r.get("strand", ".")))
    S[k] = rows
S["model_fp"] = [(c, int(x), st) for c, x, st, k in
                 zip(P["chrom"], P["center"], P["strand"], P["is_canonical"]) if not k]
S["canonical"] = [(c, int(x), st) for c, x, st, k in
                  zip(P["chrom"], P["center"], P["strand"], P["is_canonical"]) if k]
SETS["model_fp"] = "internal competitors"; SETS["canonical"] = "canonical CPA"
for k in SETS: log(f"[set] {k:<12} {len(S[k]):>7,} sites")

# genes, and the shared set
GA = {}
for k in SETS:
    g = [gene_at(c, p) for c, p, _ in S[k]]
    GA[k] = g
    log(f"[gene] {k:<12} {sum(x is not None for x in g):>7,} of {len(g):,} sites fall in a "
        f"protein-coding gene ({100*sum(x is not None for x in g)/len(g):.0f}%)")
GSET = {k: set(x for x in GA[k] if x) for k in SETS}
SHARED = set.intersection(*GSET.values())
log(f"\n[shared] genes per set: " + ", ".join(f"{k} {len(GSET[k]):,}" for k in SETS))
log(f"[shared] {len(SHARED):,} genes are covered by ALL {len(SETS)} sets")
with open("overlap_shared_genes.tsv", "w") as fh:
    fh.write("gene\n")
    for g in sorted(SHARED): fh.write(g + "\n")

out = []
for k in SETS:
    for (c, p, st), g in zip(S[k], GA[k]):
        a, na = max_over(RFI, c, p); b, nb = max_over(CNI, c, p)
        out.append((k, c, p, st, g or "", int(g in SHARED) if g else 0, a, b, na, nb))
    log(f"[score] {k} done")
PA = np.array([o[6] for o in out], float); PB = np.array([o[7] for o in out], float)
rfp = np.array([pct_of(v, RFS) for v in PA]); cnp = np.array([pct_of(v, CNS) for v in PB])
with open("overlap_scores_sites.tsv", "w") as fh:
    fh.write("set\tchrom\tpos\tstrand\tgene\tin_shared\trf_max\tcnn_max\t"
             "n_rf_windows\tn_cnn_windows\trf_pct\tcnn_pct\n")
    for o, x, y in zip(out, rfp, cnp):
        fh.write("\t".join(str(v) if not isinstance(v, float) else
                           ("" if not np.isfinite(v) else f"{v:.5f}") for v in o)
                 + f"\t{'' if not np.isfinite(x) else f'{x:.2f}'}"
                 + f"\t{'' if not np.isfinite(y) else f'{y:.2f}'}\n")
log("[out] overlap_scores_sites.tsv")

with open("overlap_scores_summary.tsv", "w") as fh:
    fh.write("set\tlabel\tscope\tn_sites\tn_rf_scored\tn_cnn_scored\trf_median\t"
             "rf_mean_pct\tcnn_median\tcnn_mean_pct\tmedian_rf_windows\n")
    log(f"\n{'set':<12} {'scope':<8} {'n':>7} {'scored':>7} {'RF med':>8} {'RF pct':>7} "
        f"{'CNN med':>8} {'CNN pct':>8}")
    for k in SETS:
        for scope in ("all", "shared"):
            m = np.array([o[0] == k and (scope == "all" or o[5] == 1) for o in out])
            a = PA[m]; b = PB[m]; ra = rfp[m]; rb = cnp[m]
            ok = np.isfinite(a)
            nw = np.array([o[8] for o in out])[m]
            okb = np.isfinite(b)
            fh.write(f"{k}\t{SETS[k]}\t{scope}\t{m.sum()}\t{ok.sum()}\t{okb.sum()}\t"
                     f"{np.nanmedian(a):.5f}\t{np.nanmean(ra):.2f}\t{np.nanmedian(b):.5f}\t"
                     f"{np.nanmean(rb):.2f}\t{np.median(nw):.0f}\n")
            log(f"{k:<12} {scope:<8} {m.sum():>7,} {ok.sum():>7,} {np.nanmedian(a):>8.3f} "
                f"{np.nanmean(ra):>7.1f} {np.nanmedian(b):>8.3f} {np.nanmean(rb):>8.1f}")
log("[out] overlap_scores_summary.tsv overlap_shared_genes.tsv")
