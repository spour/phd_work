#!/usr/bin/env python3
# p@1 across the whole gene set, baseline alone and after reranking baseline candidates
import sys, csv, time
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
PAD = 40
CODE = np.full(256, 4, np.int64)
for i, ch in enumerate("ACGT"): CODE[ord(ch)] = i

# gene structure from the labelled table
LAB = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
tb = lambda x: x in ("True", "true", "1")
gcan, gcomp, gcont = {}, {}, {}
for r in LAB:
    g = r["gene"]
    gcan[g] = gcan.get(g, 0) + tb(r["is_canonical"])
    gcomp[g] = gcomp.get(g, 0) + (not tb(r["is_canonical"]))
    gcont[g] = gcont.get(g, False) or tb(r["contested"])
ALLG = np.array(sorted(gcan))
assert all(gcan[g] == 1 for g in ALLG)
CONT = np.array(sorted(g for g in ALLG if gcomp[g] >= 1))
UNC = np.array(sorted(g for g in ALLG if gcomp[g] == 0))
assert len(ALLG) == 15560 and len(CONT) == 6583 and len(UNC) == 8977
assert all(gcont[g] for g in CONT) and not any(gcont[g] for g in UNC)
log(f"[set] all genes {len(ALLG)} = {len(UNC)} uncontested (0 competitors) "
    f"+ {len(CONT)} contested (>=1 competitor)")
log(f"[set] 'all genes with a true end and >=1 competitor' is the {len(CONT)} CONTESTED genes, "
    f"where the baseline is 0 by construction; the whole-genome denominator keeps all "
    f"{len(ALLG)}")

# contested-gene peaks and their baseline probabilities
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
NP_ = len(M); assert NP_ == 70723
gene = np.array([r["gene"] for r in M]); chrom = np.array([r["chrom"] for r in M])
cen = np.array([int(r["center"]) for r in M]); strand = np.array([r["strand"] for r in M])
y = np.array([int(float(r["is_canonical"])) for r in M])
assert set(gene) == set(CONT) and y.sum() == 6583
from fastparquet import ParquetFile
F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
row = np.array([key[(gene[i], chrom[i], int(cen[i]), strand[i])] for i in range(NP_)])
BP = {c: F[c].to_numpy(float)[row] for c in ("rf_prob", "lr_prob", "cnn_prob")}
ACC = F["spl_upstream_max_me3"].to_numpy(float)[row]
log(f"[peaks] {NP_} peaks in contested genes; baseline probabilities and acceptor joined")

# sequence scan
G = L.G_["G"]
t0 = time.time()
GCB = np.full((NP_, NBIN), np.nan); U1B = np.full((NP_, 2 * NBIN), np.nan)
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bb = (o9 + UP) // BIN; kk = (bb >= 0) & (bb < NBIN); bk = bb[kk]
nb9 = np.bincount(bk, minlength=NBIN).astype(float)
for i in range(NP_):
    c = int(cen[i]); st = strand[i]
    a, b = (c - UP - PAD, c + DN + PAD) if st == "+" else (c - DN - PAD + 1, c + UP + PAD + 1)
    s = G.fetch(chrom[i], a, b, st)
    if not s or len(s) != UP + DN + 2 * PAD: continue
    tok = CODE[np.frombuffer(s.encode("ascii"), np.uint8)]
    flat = tok[PAD:PAD + UP + DN]
    core = flat.reshape(NBIN, BIN)
    nok = (core < 4).sum(1).astype(float)
    GCB[i] = np.where(nok > 0, ((core == 1) | (core == 2)).sum(1) / np.maximum(nok, 1), np.nan)
    sw = np.lib.stride_tricks.sliding_window_view(flat, 9)
    good = (sw < 4).all(1)
    u = np.nan_to_num(np.where(good, L.U1[np.where(good, sw @ L.POW, 0)], np.nan), nan=-99.0)
    for ci, thr in enumerate((4.0, 7.0)):
        ss = np.bincount(bk, weights=(u >= thr)[kk].astype(float), minlength=NBIN)
        U1B[i, ci * NBIN:(ci + 1) * NBIN] = np.where(nb9 > 0, ss / np.maximum(nb9, 1) * 1000.0,
                                                     np.nan)
    if (i + 1) % 20000 == 0: log(f"  scan {i+1}/{NP_}  {time.time()-t0:.0f}s")
ok = np.isfinite(GCB).all(1) & np.isfinite(U1B).all(1) & np.isfinite(ACC)
log(f"[scan] {NP_} peaks in {time.time()-t0:.0f}s; {int(ok.sum())} fully scored")

Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CHN = [str(c) for c in Z["channels"]]
AG = K.C0["gene"][K.ok]; AC = K.C0["chrom"][K.ok]
ACE = K.C0["center"][K.ok].astype(int); AST = K.C0["strand"][K.ok]
pos = {(gene[i], chrom[i], int(cen[i]), strand[i]): i for i in range(NP_)}
sub = np.array([pos[(AG[j], AC[j], int(ACE[j]), AST[j])] for j in range(len(AG))])
ref = np.column_stack([P[:, CHN.index("u1_dens4"), :], P[:, CHN.index("u1_dens7"), :]])
d = np.nanmax(np.abs(U1B[sub].astype(np.float32) - ref.astype(np.float32)))
assert d == 0.0, f"fresh scan does not reproduce binned_profiles.npz in float32 ({d})"
log("[check] the analysis-set subset of the fresh scan reproduces binned_profiles.npz exactly")

# machinery over contested peaks
GI = {g: np.flatnonzero(gene == g) for g in CONT}
PAIR = {g: (ix[y[ix] == 1], ix[y[ix] == 0]) for g, ix in GI.items()}
ASET = set(AG)

def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]

def centre(Mx):
    out = Mx.astype(float).copy()
    o = np.argsort(gene, kind="stable"); gs = gene[o]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        ix = o[cut[j]:cut[j + 1]]
        mu = np.nanmean(out[ix], 0)
        out[ix] = out[ix] - np.where(np.isfinite(mu), mu, 0.0)
    return out

def fit_eval(Mx, Cg, fit_i, ev_i, lam):
    mu = np.nanmean(Cg[fit_i], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Cg[fit_i], 0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Zm = (np.where(np.isfinite(Cg), Cg, mu) - mu) / sd
    bta = K.fit_ridge(np.column_stack([np.ones(len(fit_i)), Zm[fit_i]]), y[fit_i].astype(float), lam)
    return np.column_stack([np.ones(len(ev_i)), Zm[ev_i]]) @ bta

CHS = sorted(set(chrom))
def loco(Mx, tag):
    Cg = centre(Mx); sc = np.full(NP_, np.nan)
    for ch in CHS:
        te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
        trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(K.SEED)
        sh = trg.copy(); rs.shuffle(sh); ivg = set(sh[:int(round(0.25 * len(sh)))])
        iv = tr[np.array([g in ivg for g in gene[tr]])]
        it = tr[np.array([g not in ivg for g in gene[tr]])]
        gi = groups(iv); best = (-1.0, 1.0)
        for lam in K.LAMBDAS:
            w = K.winrate(fit_eval(Mx, Cg, it, iv, lam), gi)
            if w > best[0]: best = (w, lam)
        sc[te] = fit_eval(Mx, Cg, tr, te, best[1])
    assert np.isfinite(sc).all()
    log(f"[loco] {tag}: {len(CHS)} folds done")
    return sc

def p1(sc, genes):
    hit = np.empty(len(genes)); tie = 0
    for j, g in enumerate(genes):
        r, c = PAIR[g]; m = sc[r].max(); n = sc[c].max()
        hit[j] = float(m > n); tie += int(m == n)
    return hit, tie

def ci(h, nb=2000):
    bs = np.random.default_rng(K.SEED).integers(0, len(h), size=(nb, len(h)))
    v = h[bs].mean(1); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

SPL = np.column_stack([U1B, ACC])
SGC = np.column_stack([GCB, ACC, U1B])
S_SPL, S_SGC = loco(SPL, "splice 51"), loco(SGC, "splice+GC 76")
S_RER = loco(np.column_stack([BP["rf_prob"], S_SGC]), "rerank = rf_prob + splice+GC (stacked)")

SCORES = [("Baseline alone (rf_prob)", BP["rf_prob"]),
          ("Baseline NEGATED (circularity probe)", -BP["rf_prob"]),
          ("Splice model (51)", S_SPL),
          ("Splice + GC per bin (76)", S_SGC),
          ("RERANK: baseline + splice+GC [CIRCULAR]", S_RER)]
HIT = {}
log("")
for nm, sc in SCORES:
    for dn, gs in (("contested (6,583)", CONT),
                   ("analysis-set (4,847)", np.array([g for g in CONT if g in ASET]))):
        h, tie = p1(sc, gs); lo, hi = ci(h)
        HIT[(nm, dn)] = h
        log(f"[{dn:20s}] {nm:30s} P@1 {h.mean():.4f} [{lo:.4f},{hi:.4f}] "
            f"over {len(gs)} genes, {tie} ties")
    h = np.r_[HIT[(nm, "contested (6,583)")], np.ones(len(UNC))]
    HIT[(nm, "all genes (15,560)")] = h
    lo, hi = ci(h)
    log(f"[{'all genes (15,560)':20s}] {nm:30s} P@1 {h.mean():.4f} [{lo:.4f},{hi:.4f}] "
        f"over {len(h)} genes\n")

BASE = "Baseline alone (rf_prob)"
for nm in ("rf_prob", "lr_prob", "cnn_prob"):
    h, _ = p1(BP[nm], CONT)
    log(f"[baseline] {nm}: contested P@1 {h.mean():.4f}; whole-genome "
        f"{(h.sum()+len(UNC))/len(ALLG):.4f}")
log("")
RES = []
for nm, _ in SCORES:
    for dn in ("all genes (15,560)", "contested (6,583)", "analysis-set (4,847)"):
        h = HIT[(nm, dn)]; lo, hi = ci(h)
        RES.append((dn, nm, len(h), h.mean(), lo, hi))
    if nm == BASE: continue
    for dn in ("all genes (15,560)", "contested (6,583)"):
        a, b = HIT[(nm, dn)], HIT[(BASE, dn)]
        bs = np.random.default_rng(K.SEED).integers(0, len(a), size=(2000, len(a)))
        da = a[bs].mean(1) - b[bs].mean(1)
        lo, hi = np.percentile(da, [2.5, 97.5])
        log(f"[lift] {dn:20s} {nm:30s} minus baseline {a.mean()-b.mean():+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]")
        RES.append((dn + " LIFT vs baseline", nm, len(a), a.mean() - b.mean(), lo, hi))
with open("wholegenome_rerank.tsv", "w") as fh:
    fh.write("denominator\tmodel\tn_genes\tvalue\tci_lo\tci_hi\n")
    for r in RES: fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:+.5f}\t{r[4]:+.5f}\t{r[5]:+.5f}\n")
log("")
NEG = HIT[("Baseline NEGATED (circularity probe)", "contested (6,583)")].mean()
RER = HIT[("RERANK: baseline + splice+GC [CIRCULAR]", "contested (6,583)")].mean()
log(f"[CIRCULAR] Negating the baseline probability alone scores {NEG:.4f} on contested genes.")
log(f"[CIRCULAR] That is the selection rule read backwards: a competitor IS a window whose")
log(f"[CIRCULAR] baseline probability beats the canonical's, so low baseline probability means")
log(f"[CIRCULAR] canonical by construction. The reranker ({RER:.4f}) is a worse version of the")
log(f"[CIRCULAR] same trick, so its lift is an artefact of how candidates were chosen, not")
log(f"[CIRCULAR] evidence that splice features rescue baseline errors. It is reported but must")
log(f"[CIRCULAR] NOT be quoted as a result. Any reranker fed the baseline score inherits this.")
log("")
SG = HIT[("Splice + GC per bin (76)", "all genes (15,560)")].mean()
SP = HIT[("Splice model (51)", "all genes (15,560)")].mean()
log(f"[headline] The defensible whole-genome statement uses sequence features only, with no")
log(f"[headline] baseline score as an input: over {len(ALLG)} genes, P@1 rises from "
    f"{HIT[(BASE,'all genes (15,560)')].mean():.4f} (baseline) to {SP:.4f} (splice, 51) and")
log(f"[headline] {SG:.4f} (splice + positional GC, 76).")
log(f"[headline] All of the lift comes from the {len(CONT)} contested genes, diluted by "
    f"{len(UNC)} uncontested")
log(f"[headline] genes where one candidate exists and no model can change the answer.")
log("[out] wholegenome_rerank.tsv")
