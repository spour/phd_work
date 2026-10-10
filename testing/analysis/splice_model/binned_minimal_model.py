#!/usr/bin/env python3
# prune the binned splice model to the smallest feature set that holds the full model's
import sys
import numpy as np
import binned_common as K
from binned_common import log

ADD_MIN, TOL = 0.005, 0.005
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]; NBIN = int(Z["nbin"]); BINLO = Z["bin_lo"]
NCH = P.shape[1]
ACC = K.ACC
assert len(P) == len(K.y)

FULL = np.column_stack([P.reshape(len(P), -1), ACC])
log(f"[input] {NCH} channels x {NBIN} bins = {NCH*NBIN} profile features + 1 acceptor "
    f"scalar = {FULL.shape[1]} total")
log(f"[input] channels: {', '.join(CH)}")
log(f"[input] bins: {NBIN} x 100 nt over [{int(BINLO[0])}, {int(BINLO[-1])+100})")

LAM_FULL, IV_FULL = K.pick_lambda(FULL, "full binned + acceptor")
sc, _ = K.fit_score(FULL, K.trv, K.te, LAM_FULL)
R_FULL = K.suite(sc)
log(f"[full] test within-gene {R_FULL['winrate']:.4f} "
    f"[{R_FULL['winrate_lo']:.4f},{R_FULL['winrate_hi']:.4f}]  inner-val {IV_FULL:.4f}")
assert abs(R_FULL["winrate"] - 0.7101) < 5e-4, \
    f"full model does not reproduce the committed 0.7101 (got {R_FULL['winrate']:.4f})"
log("[full] reproduces the committed 0.7101, so the pruning below uses the same machinery")
log(f"[target] a set qualifies at inner-val >= {IV_FULL - TOL:.4f} "
    f"(full inner-val {IV_FULL:.4f} minus {TOL})\n")

def blk(ch): return P[:, CH.index(ch), :]

BLOCKS = {c: blk(c) for c in CH}
BLOCKS["acceptor_scalar"] = ACC[:, None]
BNAMES = list(BLOCKS)
log("A. CHANNEL selection (a channel enters with all 25 bins, or not at all)")
selA, curA = [], 0.5
while len(selA) < len(BNAMES):
    best = (-1.0, None)
    for nm in BNAMES:
        if nm in selA: continue
        M = np.column_stack([BLOCKS[k] for k in selA + [nm]])
        w = K.inner_wr(M)
        if w > best[0]: best = (w, nm)
    gain = best[0] - curA
    if gain < ADD_MIN:
        log(f"[A] stop: best remaining block {best[1]} adds only {gain:+.4f} (< {ADD_MIN})")
        break
    selA.append(best[1]); curA = best[0]
    log(f"[A] +{best[1]:16s} inner-val {curA:.4f} ({gain:+.4f})  "
        f"[{sum(BLOCKS[k].shape[1] for k in selA)} features]")
CHSEL = [c for c in selA if c != "acceptor_scalar"]
ACC_IN = "acceptor_scalar" in selA
log(f"[A] surviving channels: {', '.join(CHSEL) if CHSEL else 'none'}"
    f"{' + acceptor scalar' if ACC_IN else ''}  "
    f"({sum(BLOCKS[k].shape[1] for k in selA)} features, inner-val {curA:.4f})\n")

log("B(i). BIN selection over the surviving channels, one (channel, bin) cell at a time")
CELLS = [(c, b) for c in CHSEL for b in range(NBIN)]
def cellmat(cells, acc):
    cols = [P[:, CH.index(c), b] for c, b in cells]
    if acc: cols.append(ACC)
    return np.column_stack(cols) if cols else None
selB, curB, traceB = [], 0.5, []
if ACC_IN:
    curB = K.inner_wr(cellmat([], True))
    log(f"[Bi] acceptor scalar alone, inner-val {curB:.4f}")
while len(selB) < 30:
    best = (-1.0, None)
    for cell in CELLS:
        if cell in selB: continue
        w = K.inner_wr(cellmat(selB + [cell], ACC_IN))
        if w > best[0]: best = (w, cell)
    gain = best[0] - curB
    if gain < ADD_MIN:
        log(f"[Bi] stop: best remaining cell {best[1][0]}@{int(BINLO[best[1][1]])} "
            f"adds only {gain:+.4f} (< {ADD_MIN})")
        break
    selB.append(best[1]); curB = best[0]
    traceB.append((best[1], curB, gain, len(selB) + int(ACC_IN)))
    log(f"[Bi] +{best[1][0]:10s} @ {int(BINLO[best[1][1]]):+5d} nt   inner-val {curB:.4f} "
        f"({gain:+.4f})  [{len(selB)+int(ACC_IN)} features]")
log(f"[Bi] {len(selB)+int(ACC_IN)} features reach inner-val {curB:.4f}\n")

# b(ii)
REGIONS = {
 "R2": [(-2000, -500), (-500, 500)],
 "R3": [(-2000, -1000), (-1000, 0), (0, 500)],
 "R4": [(-2000, -1000), (-1000, -300), (-300, 0), (0, 500)],
 "R5": [(-2000, -1000), (-1000, -500), (-500, -200), (-200, 0), (0, 500)],
 "R6": [(-2000, -1200), (-1200, -700), (-700, -300), (-300, -100), (-100, 100), (100, 500)],
 "R8": [(-2000, -1500), (-1500, -1000), (-1000, -700), (-700, -400), (-400, -200),
        (-200, 0), (0, 200), (200, 500)],
}
def pool(ch, lo, hi):
    j = np.flatnonzero((BINLO >= lo) & (BINLO < hi))
    v = P[:, CH.index(ch), :][:, j]
    return np.nanmax(v, 1) if ch.endswith("_max") else np.nanmean(v, 1)
log("B(ii). REGION pooling over the surviving channels")
RES_R = {}
for tag, regs in REGIONS.items():
    cols = [pool(c, lo, hi) for c in CHSEL for lo, hi in regs]
    names = [f"{c}[{lo},{hi})" for c in CHSEL for lo, hi in regs]
    if ACC_IN: cols.append(ACC); names.append("spl_upstream_max_me3")
    M = np.column_stack(cols)
    w = K.inner_wr(M)
    RES_R[tag] = (w, M, names, regs)
    log(f"[Bii] {tag} ({len(regs)} regions x {len(CHSEL)} channels"
        f"{' + acceptor' if ACC_IN else ''} = {M.shape[1]} features)  inner-val {w:.4f}"
        f"{'   QUALIFIES' if w >= IV_FULL - TOL else ''}")

# the minimal qualifying set
CAND = []
for tag, (w, M, names, _) in RES_R.items():
    CAND.append((M.shape[1], w, tag, M, names))
for cell, w, g, n in traceB:
    idx = traceB.index((cell, w, g, n)) + 1
    cells = selB[:idx]
    names = [f"{c}@[{int(BINLO[b])},{int(BINLO[b])+100})" for c, b in cells]
    if ACC_IN: names = names + ["spl_upstream_max_me3"]
    CAND.append((n, w, f"bins-{idx}", cellmat(cells, ACC_IN), names))
QUAL = [c for c in CAND if c[1] >= IV_FULL - TOL]
log(f"\n[minimal] {len(QUAL)}/{len(CAND)} candidate sets reach inner-val "
    f">= {IV_FULL - TOL:.4f}")
if not QUAL:
    best_iv = max(CAND, key=lambda c: c[1])
    log(f"[minimal] none qualify at the {TOL} tolerance; the closest is {best_iv[2]} with "
        f"{best_iv[0]} features at inner-val {best_iv[1]:.4f} "
        f"({best_iv[1]-IV_FULL:+.4f} against the full model)")
    QUAL = [best_iv]
    NOTE = "no set met the 0.005 tolerance; reporting the closest"
else:
    NOTE = "smallest set within 0.005 of the full model on inner-val"
QUAL.sort(key=lambda c: (c[0], -c[1]))
NF, IVW, TAG, MM, NAMES = QUAL[0]
log(f"[minimal] chosen: {TAG}, {NF} features, inner-val {IVW:.4f} ({NOTE})")

REPORT = {}
def report(M, nm, lam=None):
    if lam is None: lam, ivw = K.pick_lambda(M, nm)
    else: ivw = K.inner_wr(M, lam)
    sc, b = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r["_lam"] = lam; r["_iv"] = ivw; r["_nf"] = M.shape[1]
    REPORT[nm] = r
    log(f"[{nm:30s}] p={M.shape[1]:3d}  inner-val {ivw:.4f}  test within-gene "
        f"{r['winrate']:.4f} [{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  "
        f"P@1 {r['p_at_1']:.4f} [{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  "
        f"AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}")
    return r

log("")
report(FULL, "full binned + acceptor", LAM_FULL)
CHONLY = np.column_stack([BLOCKS[k] for k in selA])
report(CHONLY, "channel-selected (A)")
report(MM, f"MINIMAL ({TAG})")
for tag in ("R2", "R3", "R4", "R5", "R6", "R8"):
    if tag in RES_R: report(RES_R[tag][1], f"regions {tag}")
if selB:
    for k in (3, 5, 8):
        if k <= len(selB):
            report(cellmat(selB[:k], ACC_IN), f"top-{k} bins + acceptor")
report(ACC[:, None], "acceptor alone")
from fastparquet import ParquetFile
_F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
_SPL = [c for c in _F.columns if c.startswith("spl_")]
_key = {(g, c, int(ct), s_): i for i, (g, c, ct, s_) in
        enumerate(zip(_F.gene, _F.chrom, _F.center, _F.strand))}
_row = np.array([_key.get((g, c, int(ct), s_), -1) for g, c, ct, s_ in
                 zip(K.C0["gene"][K.ok], K.C0["chrom"][K.ok],
                     K.C0["center"][K.ok], K.C0["strand"][K.ok])])
_h = _row >= 0
_X = np.full((len(K.y), len(_SPL)), np.nan)
_X[_h] = _F.loc[_row[_h], _SPL].to_numpy(float)
_S3 = [_SPL.index(c) for c in ("spl_pas_prox_u1", "spl_upstream_max_me3", "spl_comp_up_3ss")]
_r3 = report(_X[:, _S3], "scalar splice model (3)", 1.0)
assert abs(_r3["winrate"] - 0.66557) < 5e-4, "scalar baseline does not reproduce 0.6656"
log("[check] scalar baseline reproduces its committed 0.6656")

def paired(a, b, tag):
    da = REPORT[a]["_wg"][K.BS].mean(1) - REPORT[b]["_wg"][K.BS].mean(1)
    d = REPORT[a]["winrate"] - REPORT[b]["winrate"]
    lo, hi = np.percentile(da, [2.5, 97.5])
    p = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    log(f"[paired] {tag:44s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {p:.4f}"
        f"{'' if lo*hi > 0 else '   (CI spans 0)'}")
    return d, lo, hi, p
log("")
MINNM = f"MINIMAL ({TAG})"
PR = {"minimal minus full": paired(MINNM, "full binned + acceptor", "minimal minus full binned"),
      "minimal minus acceptor": paired(MINNM, "acceptor alone", "minimal minus acceptor alone"),
      "channel-selected minus full": paired("channel-selected (A)", "full binned + acceptor",
                                            "channel-selected (51) minus full (126)"),
      "channel-selected minus minimal": paired("channel-selected (A)", MINNM,
                                               "channel-selected (51) minus minimal (9)"),
      "minimal minus scalar": paired(MINNM, "scalar splice model (3)",
                                     "minimal (9) minus scalar model (3)"),
      "channel-selected minus scalar": paired("channel-selected (A)", "scalar splice model (3)",
                                              "channel-selected (51) minus scalar model (3)"),
      "regions R5 minus scalar": paired("regions R5", "scalar splice model (3)",
                                        "regions R5 (11) minus scalar model (3)")}

REF = {"scalar splice model (3 feat)": 0.66557, "splice ceiling (26 scalars)": 0.69686,
       "full binned + acceptor": R_FULL["winrate"], "sequence CNN": 0.82110}
log("")
for k, v in REF.items(): log(f"[reference] {k:30s} within-gene {v:.4f}")
rm = REPORT[MINNM]
log(f"\n[minimal] {NF} features, test within-gene {rm['winrate']:.4f} "
    f"[{rm['winrate_lo']:.4f},{rm['winrate_hi']:.4f}]  vs full {R_FULL['winrate']:.4f} "
    f"({rm['winrate']-R_FULL['winrate']:+.4f}) and scalar 0.6656 "
    f"({rm['winrate']-0.66557:+.4f})")
log(f"[overfit check] minimal inner-val {rm['_iv']:.4f} vs test {rm['winrate']:.4f} "
    f"(test - inner-val {rm['winrate']-rm['_iv']:+.4f}); full model "
    f"{IV_FULL:.4f} vs {R_FULL['winrate']:.4f} ({R_FULL['winrate']-IV_FULL:+.4f})")
log("[minimal] features: " + " | ".join(NAMES))

with open("binned_minimal_metrics.tsv", "w") as fh:
    fh.write("model\tmetric\tvalue\tci_lo\tci_hi\tn_features\tlambda\tinner_val\t"
             "n_genes\tn_peaks\n")
    for nm, r in REPORT.items():
        for m in ("winrate", "p_at_1", "auroc", "auprc"):
            fh.write(f"{nm}\t{m}\t{r[m]:.5f}\t{r[m+'_lo']:.5f}\t{r[m+'_hi']:.5f}\t"
                     f"{r['_nf']}\t{r['_lam']:g}\t{r['_iv']:.5f}\t"
                     f"{len(K.GRP_TE)}\t{len(K.te)}\n")
    for k, v in REF.items():
        fh.write(f"reference: {k}\twinrate\t{v:.5f}\t\t\t\t\t\t{len(K.GRP_TE)}\t{len(K.te)}\n")
    for t, (d, lo, hi, p) in PR.items():
        fh.write(f"paired: {t}\tdelta_winrate\t{d:.5f}\t{lo:.5f}\t{hi:.5f}\t\t\t{p:.4f}\t"
                 f"{len(K.GRP_TE)}\t{len(K.te)}\n")
    for tag, (w, M, names, _) in RES_R.items():
        fh.write(f"region sweep: {tag}\tinner_val\t{w:.5f}\t\t\t{M.shape[1]}\t\t\t"
                 f"{len(K.GRP_IV)}\t{len(K.iv)}\n")
with open("binned_minimal_features.tsv", "w") as fh:
    fh.write("rank\tfeature\tselection_stage\n")
    for i, n in enumerate(NAMES, 1): fh.write(f"{i}\t{n}\t{TAG}\n")
    fh.write(f"\t--- channel selection order ---\tA\n")
    for i, n in enumerate(selA, 1): fh.write(f"{i}\t{n}\tA\n")
    fh.write(f"\t--- bin selection order ---\tB(i)\n")
    for i, (cell, w, g, n) in enumerate(traceB, 1):
        fh.write(f"{i}\t{cell[0]}@[{int(BINLO[cell[1]])},{int(BINLO[cell[1]])+100})\t"
                 f"B(i) inner-val {w:.4f}\n")
log("[out] binned_minimal_metrics.tsv binned_minimal_features.tsv")
