#!/usr/bin/env python3
# genome-wide: where does the 500 nt baseline put a gene's 3' end
import sys, os, gzip, collections
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
SG   = "/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome"
PGD  = os.path.join(SG, "out_scores", "per_gene_dir")
MAXB = os.path.join(SG, "per_gene_max_windows.bed")
HALF = 250

P = np.load("cnn_wg_scores.npz", allow_pickle=True)
gene = np.array([str(x) for x in P["gene"]]); chrom = np.array([str(x) for x in P["chrom"]])
center = P["center"].astype(np.int64); y = P["y"].astype(int); logit = P["logit"].astype(float)
log(f"[peaks] {len(gene):,} peaks, {int((y==1).sum()):,} canonical, {int((y==0).sum()):,} competitors")

# step 1: the baseline's own argmax, genome wide
amax = {}
for line in open(MAXB):
    p = line.rstrip("\n").split("\t")
    if len(p) < 7: continue
    g = p[3].split("|")[0]
    amax[g] = (p[0], (int(p[1])+int(p[2]))//2, float(p[4]))
log(f"[scan] {len(amax):,} genes carry a genome-wide argmax window")

canon = {}
for i in np.flatnonzero(y == 1): canon[gene[i]] = (chrom[i], int(center[i]))
shared = [g for g in canon if g in amax]
log(f"[both] {len(shared):,} genes have a canonical 3' end and a scanned argmax")

rows_g = []
for g in shared:
    ch, cc = canon[g]; ac_ch, ac, asc = amax[g]
    d = abs(ac-cc)
    rows_g.append(dict(gene=g, chrom=ch, canonical=cc, argmax=ac, argmax_score=asc,
                       dist=d, hit=int(d <= HALF)))
nhit = sum(r["hit"] for r in rows_g)
log(f"[step 1] baseline argmax lands ON the real 3' end in {nhit:,} of {len(rows_g):,} genes "
    f"({100*nhit/len(rows_g):.1f}%)")

# step 2: per-peak baseline score, streamed
by_gene = collections.defaultdict(list)
for i in range(len(gene)): by_gene[gene[i]].append(i)
rf = np.full(len(gene), np.nan); nwin = np.zeros(len(gene), int)
todo = [g for g in by_gene if g in amax]
log(f"[stream] reading {len(todo):,} per-gene scans for per-peak baseline scores")
miss = 0
for n, g in enumerate(sorted(todo), 1):
    ch = chrom[by_gene[g][0]]
    fp = os.path.join(PGD, ch, f"{g}.bed6.gz")
    if not os.path.exists(fp): miss += 1; continue
    idx = by_gene[g]; cs = center[idx]
    lo = cs.min()-HALF-1; hi = cs.max()+HALF+1
    best = np.full(len(idx), -np.inf); cnt = np.zeros(len(idx), int)
    with gzip.open(fp, "rt") as fh:
        for line in fh:
            p = line.split("\t")
            wc = (int(p[1])+int(p[2]))//2
            if wc < lo or wc > hi: continue
            v = float(p[4])
            m = np.abs(cs-wc) <= HALF
            if m.any():
                cnt += m
                np.maximum(best, np.where(m, v, -np.inf), out=best)
    for k, i in enumerate(idx):
        if np.isfinite(best[k]): rf[i] = best[k]; nwin[i] = cnt[k]
    if n % 500 == 0: log(f"   {n:,}/{len(todo):,} genes")
log(f"[stream] done; {miss:,} gene files absent; {int(np.isfinite(rf).sum()):,} peaks scored")

with open("baseline_rescue_peaks.tsv", "w") as f:
    f.write("gene\tchrom\tcenter\tstrand\ty\tcnn_logit\trf_max\tn_windows\n")
    for i in range(len(gene)):
        f.write(f"{gene[i]}\t{chrom[i]}\t{center[i]}\t{str(P['strand'][i])}\t{y[i]}\t"
                f"{logit[i]:.6f}\t{'' if np.isnan(rf[i]) else f'{rf[i]:.6f}'}\t{nwin[i]}\n")

# step 3: who wins within each gene
def top_is_canonical(vals, lab):
    if len(vals) < 2 or not np.isfinite(vals).all(): return None
    return int(vals[np.argmax(vals)] == vals[lab == 1].max() and lab[np.argmax(vals)] == 1)
res = []
for g, idx in by_gene.items():
    idx = np.array(idx); lab = y[idx]
    if lab.sum() != 1 or len(idx) < 2: continue
    r_rf = rf[idx]; r_cn = logit[idx]
    if not np.isfinite(r_rf).all(): continue
    res.append(dict(gene=g, n_peaks=len(idx), n_comp=int((lab == 0).sum()),
                    rf_top=int(lab[np.argmax(r_rf)] == 1),
                    cnn_top=int(lab[np.argmax(r_cn)] == 1),
                    rf_win=float((r_rf[lab == 1][0] > r_rf[lab == 0]).mean()
                                 + 0.5*(r_rf[lab == 1][0] == r_rf[lab == 0]).mean()),
                    cnn_win=float((r_cn[lab == 1][0] > r_cn[lab == 0]).mean()
                                  + 0.5*(r_cn[lab == 1][0] == r_cn[lab == 0]).mean())))
R = res
log(f"[step 3] {len(R):,} genes with one canonical peak, at least one competitor, and full baseline coverage")
def frac(k, sub=None):
    d = R if sub is None else sub
    return sum(r[k] for r in d)/len(d) if d else float("nan")
fail = [r for r in R if r["rf_top"] == 0]
log(f"   baseline puts the real 3' end top in {sum(r['rf_top'] for r in R):,} of {len(R):,} genes "
    f"({100*frac('rf_top'):.1f}%)")
log(f"   CNN      puts the real 3' end top in {sum(r['cnn_top'] for r in R):,} of {len(R):,} genes "
    f"({100*frac('cnn_top'):.1f}%)")
log(f"   ON THE {len(fail):,} GENES THE BASELINE GETS WRONG, the CNN puts it top in "
    f"{sum(r['cnn_top'] for r in fail):,} ({100*frac('cnn_top',fail):.1f}%)")
log(f"   within-gene win rate, baseline {frac('rf_win'):.4f}   CNN {frac('cnn_win'):.4f}")
log(f"   on the failure genes,  baseline {frac('rf_win',fail):.4f}   CNN {frac('cnn_win',fail):.4f}")
with open("baseline_rescue_genes.tsv", "w") as f:
    f.write("gene\tn_peaks\tn_competitors\trf_top\tcnn_top\trf_winrate\tcnn_winrate\n")
    for r in R: f.write(f"{r['gene']}\t{r['n_peaks']}\t{r['n_comp']}\t{r['rf_top']}\t{r['cnn_top']}\t"
                        f"{r['rf_win']:.6f}\t{r['cnn_win']:.6f}\n")
with open("baseline_rescue_summary.tsv", "w") as f:
    f.write("quantity\tvalue\tn\n")
    f.write(f"genes_scanned_genomewide\t{len(amax)}\t{len(amax)}\n")
    f.write(f"genes_with_canonical_and_scan\t{len(rows_g)}\t{len(rows_g)}\n")
    f.write(f"baseline_argmax_on_real_3prime_end\t{nhit/len(rows_g):.6f}\t{nhit}\n")
    f.write(f"genes_eval\t{len(R)}\t{len(R)}\n")
    f.write(f"baseline_top1\t{frac('rf_top'):.6f}\t{sum(r['rf_top'] for r in R)}\n")
    f.write(f"cnn_top1\t{frac('cnn_top'):.6f}\t{sum(r['cnn_top'] for r in R)}\n")
    f.write(f"baseline_failures\t{len(fail)}\t{len(fail)}\n")
    f.write(f"cnn_top1_on_baseline_failures\t{frac('cnn_top',fail):.6f}\t{sum(r['cnn_top'] for r in fail)}\n")
    f.write(f"baseline_winrate\t{frac('rf_win'):.6f}\t{len(R)}\n")
    f.write(f"cnn_winrate\t{frac('cnn_win'):.6f}\t{len(R)}\n")
    f.write(f"baseline_winrate_on_failures\t{frac('rf_win',fail):.6f}\t{len(fail)}\n")
    f.write(f"cnn_winrate_on_failures\t{frac('cnn_win',fail):.6f}\t{len(fail)}\n")
with open("baseline_rescue_argmax.tsv", "w") as f:
    f.write("gene\tchrom\tcanonical\targmax\targmax_score\tdist\thit\n")
    for r in rows_g: f.write("\t".join(str(r[k]) for k in
                             ("gene","chrom","canonical","argmax","argmax_score","dist","hit"))+"\n")
log("[out] baseline_rescue_peaks.tsv / _genes.tsv / _summary.tsv / _argmax.tsv")
