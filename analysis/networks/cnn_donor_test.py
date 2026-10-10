#!/usr/bin/env python3
# a readout built to find a donor/u1 motif if one is there
import sys, os, time, argparse
import numpy as np
import torch
from cnn_interp_lib import (NT, consensus_pwm, pwm_match, match_threshold, write_meme,
                            dinuc_shuffle)

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--n_ism", type=int, default=384)
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
rng = np.random.default_rng(A.seed); torch.manual_seed(A.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
W, UP, BINW, NBINS = 19, 2000, 100, 25
IC_FLOOR, CONTRIB_Q = 5.0, 95

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; split = D["split"][ok]
te = np.flatnonzero(split == "test")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def onehot(a): return TBLt[torch.from_numpy(a.astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(o, bs=256):
    out = []
    with torch.no_grad():
        for k in range(0, len(o), bs): out.append(net(o[k:k+bs]).float().cpu())
    return torch.cat(out).numpy()

REFS_RAW = {
    "donor_5ss_AGGTAAGT":  "AGGTAAGT",
    "GU_rich":             "GTGTGTGTGTGT",
    "acceptor_3ss_pyAG":   "TTTTTTTTTTNCAGG",
    "PAS_AATAAA":          "AATAAA",
    "PAS_ATTAAA":          "ATTAAA",
    "U_rich":              "TTTTTTTTTTTT",
    "branchpoint_CTRAY":   "CTGAC",
}
DONOR_REFS = {"donor_5ss_AGGTAAGT", "GU_rich"}

def filter_pwms(n_seq=4000, frac=0.3):
    sub = rng.choice(te, min(n_seq, len(te)), replace=False)
    NF = net.c1.out_channels
    counts = np.zeros((NF, W, 4)); nsite = np.zeros(NF, int); maxact = np.zeros(NF)
    for k in range(0, len(sub), 128):
        with torch.no_grad(): act = net.first(onehot(X[sub[k:k+128]])).cpu().numpy()
        maxact = np.maximum(maxact, act.max((0, 2)))
    thr = frac * maxact
    for k in range(0, len(sub), 128):
        idx = sub[k:k+128]
        with torch.no_grad(): act = net.first(onehot(X[idx])).cpu().numpy()
        for f in range(NF):
            b, p = np.where(act[:, f, :] >= thr[f])
            for bi, pi in zip(b, p):
                s = pi - W // 2
                if s < 0 or s + W > X.shape[1]: continue
                seq = X[idx[bi], s:s+W]; g = seq < 4
                counts[f, np.arange(W)[g], seq[g]] += 1; nsite[f] += 1
    pwm = np.where(counts.sum(2, keepdims=True) > 0,
                   counts / np.maximum(counts.sum(2, keepdims=True), 1), 0.25)
    ic = np.nansum(pwm * np.log2(np.maximum(pwm, 1e-9) / 0.25), (1, 2))
    return pwm, nsite, maxact, ic

pwm, nsite, maxact, ic = filter_pwms()
NF = len(pwm)
write_meme(pwm, nsite, "filter_pwms_readout.meme", W=W)
refs = {k: consensus_pwm(v) for k, v in REFS_RAW.items()}
THR = match_threshold(refs, rng)
log("[filters] random-PWM null, 95th pct of r: " + "  ".join(f"{k} {v:.2f}" for k, v in THR.items()))

C = np.load("cnn_posmap_contrib.npz")["contrib"]
CT = float(np.percentile(np.abs(C), CONTRIB_Q))
log(f"[posmap] contribution threshold = {CONTRIB_Q}th percentile of |contrib| = {CT:.5f}")

recs = []
for f in range(NF):
    best_by_ref = {k: pwm_match(pwm[f], q) for k, q in refs.items()}
    bk = max(best_by_ref, key=lambda k: best_by_ref[k][0])
    br, bs = best_by_ref[bk]
    passes_null = br >= THR[bk]
    dn = max(DONOR_REFS, key=lambda k: best_by_ref[k][0])
    dn_r, dn_s = best_by_ref[dn]
    bb = int(np.argmax(C[f])); cmax = float(C[f, bb])
    recs.append(dict(f=f, name=f"filter{f:02d}", ic=float(ic[f]), n=int(nsite[f]),
                     best=bk if passes_null else "none", r=float(br), strand=bs,
                     cons="".join(NT[i] for i in pwm[f].argmax(1)),
                     donor_ref=dn, donor_r=float(dn_r),
                     donor_passes_null=bool(dn_r >= THR[dn]),
                     best_bin=bb, best_bin_start=bb*BINW-UP, contrib_max=cmax,
                     c1=bool(ic[f] >= IC_FLOOR),
                     c2=bool(passes_null and bk in DONOR_REFS),
                     c3=bool(cmax > 0 and cmax > CT)))
for r in recs: r["donor_call"] = bool(r["c1"] and r["c2"] and r["c3"])
with open("filter_matches_readout.tsv", "w") as fh:
    fh.write("filter\tn_sites\tinfo_content_bits\tbest_match\tr\tstrand\tconsensus\t"
             "best_donor_ref\tdonor_r\tdonor_beats_null\tbest_bin\tbest_bin_start\t"
             "contrib_max\tcrit1_ic\tcrit2_donor_match\tcrit3_contrib\tdonor_call\n")
    for r in recs:
        fh.write(f"{r['name']}\t{r['n']}\t{r['ic']:.2f}\t{r['best']}\t{r['r']:.3f}\t{r['strand']}\t"
                 f"{r['cons']}\t{r['donor_ref']}\t{r['donor_r']:.3f}\t{int(r['donor_passes_null'])}\t"
                 f"{r['best_bin']}\t{r['best_bin_start']}\t{r['contrib_max']:.6f}\t"
                 f"{int(r['c1'])}\t{int(r['c2'])}\t{int(r['c3'])}\t{int(r['donor_call'])}\n")
hits = {}
for r in recs:
    if r["best"] != "none": hits.setdefault(r["best"], []).append(r)
log("[filters] best match beating the random-PWM null: " +
    "  ".join(f"{k} x{len(v)}" for k, v in sorted(hits.items())))
log(f"[filters] IC >= {IC_FLOOR} bits: {sum(r['c1'] for r in recs)}/{NF}")
log(f"[filters] criterion 2 (best match IS a donor reference above null): "
    f"{sum(r['c2'] for r in recs)}/{NF}")
dcand = [r for r in recs if r["donor_passes_null"]]
log(f"[filters] filters whose DONOR-reference r alone clears the null (criterion 2 partial, "
    f"the known AT artifact): {len(dcand)} -> " +
    ", ".join(f"{r['name']}(r={r['donor_r']:.2f},IC={r['ic']:.1f},{r['cons']})" for r in dcand))
DONOR_CALLS = [r for r in recs if r["donor_call"]]
log(f"[filters] PASSING ALL THREE CRITERIA: "
    + (", ".join(r["name"] for r in DONOR_CALLS) if DONOR_CALLS else "NONE"))
acc_hi = [r for r in recs if r["best"] in ("acceptor_3ss_pyAG", "U_rich") and r["c1"]]
log(f"[filters] acceptor/U-rich filters at >= {IC_FLOOR} bits: "
    + ", ".join(f"{r['name']}(r={r['r']:.2f},IC={r['ic']:.1f})" for r in acc_hi))

GIA = {"donor_AGGTAAGT": "AGGTAAGT", "donor_CAGGTAAGT": "CAGGTAAGT",
       "donor_perm_ctrl_TAGAGTAG": "TAGAGTAG",
       "GU_rich_GTGTGTGT": "GTGTGTGT",
       "acceptor_pyAG": "TTTTTTTTTTTTTCAGG",
       "PAS_AATAAA": "AATAAA", "PAS_perm_ctrl_AAAATA": "AAAATA"}
def implant(seqs, motif, pos):
    m = np.array([NT.index(c) for c in motif], np.uint8)
    out = seqs.copy(); s = pos - len(m) // 2
    if s < 0 or s + len(m) > out.shape[1]: return None
    out[:, s:s+len(m)] = m
    return out
bg = np.stack([dinuc_shuffle(X[i], rng) for i in rng.choice(te, A.gia_bg, replace=False)])
bg_real = X[rng.choice(te[y[te] == 0], A.gia_bg, replace=False)]
positions = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
gia_best = {}
with open("gia_donor_test.tsv", "w") as fh:
    fh.write("background\tmotif\tposition\tdelta_logit\tse\tn\n")
    for bgname, B in (("dinuc_shuffled", bg), ("real_competitor", bg_real)):
        base = fwd(onehot(B))
        log(f"[GIA] {bgname} background logit mean {base.mean():+.3f} sd {base.std():.3f}")
        for name, m in GIA.items():
            best = (None, -1e9)
            for pos in positions:
                seqs = implant(B, m, pos + UP)
                if seqs is None: continue
                d = fwd(onehot(seqs)) - base
                fh.write(f"{bgname}\t{name}\t{pos}\t{d.mean():.5f}\t"
                         f"{d.std(ddof=1)/np.sqrt(len(d)):.5f}\t{len(d)}\n")
                if d.mean() > best[1]: best = (pos, float(d.mean()))
            gia_best[(bgname, name)] = best
            log(f"[GIA] {bgname:16s} {name:26s} max delta-logit {best[1]:+.4f} at {best[0]:+d} nt")
for bgn in ("dinuc_shuffled", "real_competitor"):
    d = gia_best[(bgn, "donor_AGGTAAGT")][1]; c = gia_best[(bgn, "donor_perm_ctrl_TAGAGTAG")][1]
    log(f"[GIA] {bgn}: donor AGGTAAGT {d:+.4f} vs its composition-matched permutation "
        f"{c:+.4f}  ->  difference {d-c:+.4f}")

def ism(seqs, bs=2048):
    L = seqs.shape[1]; out = np.zeros((len(seqs), L), np.float32)
    wt = fwd(onehot(seqs))
    for i in range(len(seqs)):
        s = seqs[i]
        V = np.array([(p, b) for b in range(4) for p in np.flatnonzero((s != b) & (s < 4))])
        d = np.zeros(L); n = np.zeros(L)
        for k in range(0, len(V), bs):
            ch = V[k:k+bs]
            Q = np.repeat(s[None, :], len(ch), 0)
            Q[np.arange(len(ch)), ch[:, 0]] = ch[:, 1].astype(np.uint8)
            lg = fwd(onehot(Q), bs=bs)
            np.add.at(d, ch[:, 0], lg - wt[i]); np.add.at(n, ch[:, 0], 1.0)
        out[i] = np.where(n > 0, d / np.maximum(n, 1), 0.0)
    return out

def occ(seqs, motif):
    code = np.array([NT.index(c) for c in motif], np.uint8)
    w = np.lib.stride_tricks.sliding_window_view(seqs, len(code), 1)
    return (w == code).all(2)

def wilcoxon(dv):
    dv = dv[dv != 0]; n = len(dv)
    if n < 6: return float("nan"), float("nan")
    a = np.abs(dv); o = np.argsort(a, kind="stable"); a = a[o]; sg = np.sign(dv[o])
    r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j+1] == a[i]: j += 1
        r[i:j+1] = (i + j + 2) / 2.0; i = j + 1
    Wp = float(r[sg > 0].sum()); mu = n * (n + 1) / 4.0
    _, cnt = np.unique(a, return_counts=True); tie = float((cnt**3 - cnt).sum())
    sd = np.sqrt(n*(n+1)*(2*n+1)/24.0 - tie/48.0)
    z = (Wp - mu - 0.5*np.sign(Wp - mu)) / sd
    from math import erfc, sqrt
    return z, erfc(abs(z)/sqrt(2))

ri = rng.choice(te, A.n_ism, replace=False)
r_seq = X[ri]; s_seq = np.stack([dinuc_shuffle(s, rng) for s in r_seq])
log(f"[ISM] {A.n_ism} real + {A.n_ism} paired dinucleotide shuffles, "
    f"{3*X.shape[1]:,} substitutions each")
t0 = time.time(); ar = ism(r_seq); an = ism(s_seq)
log(f"[ISM] done in {time.time()-t0:.0f}s")
MOT = {"donor_GTAAG": "GTAAG", "donor_AGGTAAG": "AGGTAAG", "GU_GTGTGT": "GTGTGT",
       "PAS_AATAAA": "AATAAA", "acceptor_TTTTTT": "TTTTTT", "ctrl_GGCGGC": "GGCGGC"}
irows = []
for nm, mo in MOT.items():
    mr = occ(r_seq, mo); ms = occ(s_seq, mo)
    wr = np.lib.stride_tricks.sliding_window_view(np.abs(ar), len(mo), 1).mean(2)
    ws = np.lib.stride_tricks.sliding_window_view(np.abs(an), len(mo), 1).mean(2)
    pr = np.array([wr[i][mr[i]].mean() if mr[i].any() else np.nan for i in range(len(r_seq))])
    ps = np.array([ws[i][ms[i]].mean() if ms[i].any() else np.nan for i in range(len(s_seq))])
    pm = np.array([ws[i][mr[i]].mean() if mr[i].any() else np.nan for i in range(len(r_seq))])
    posr = np.array([np.flatnonzero(mr[i]).mean() - UP if mr[i].any() else np.nan
                     for i in range(len(r_seq))])
    poss = np.array([np.flatnonzero(ms[i]).mean() - UP if ms[i].any() else np.nan
                     for i in range(len(s_seq))])
    both = np.isfinite(pr) & np.isfinite(ps)
    bm = np.isfinite(pr) & np.isfinite(pm)
    z, p = wilcoxon(pr[both] - ps[both])
    zm, pmw = wilcoxon(pr[bm] - pm[bm])
    ratio = float(np.nanmean(pr[both]) / np.nanmean(ps[both])) if both.sum() else np.nan
    ratio_pm = float(np.nanmean(pr[bm]) / np.nanmean(pm[bm])) if bm.sum() else np.nan
    irows.append((nm, mo, int(mr.sum()), int(ms.sum()), int(both.sum()),
                  float(np.nanmean(pr[both])), float(np.nanmean(ps[both])), ratio, z, p,
                  ratio_pm, zm, pmw, int(bm.sum()),
                  float(np.nanmean(posr)), float(np.nanmean(poss))))
    log(f"[ISM] {nm:16s} n_occ real {int(mr.sum()):6d} shuf {int(ms.sum()):6d}  "
        f"paired seqs {int(both.sum()):3d}  |ISM| real {np.nanmean(pr[both]):.5f} "
        f"shuf {np.nanmean(ps[both]):.5f}  ratio {ratio:.3f}  z {z:+.2f}  p {p:.3g}")
    log(f"[ISM-pos] {nm:16s} mean occurrence position real {np.nanmean(posr):+8.1f} nt  "
        f"shuffled {np.nanmean(poss):+8.1f} nt   |   POSITION-MATCHED null: ratio "
        f"{ratio_pm:.3f}  z {zm:+.2f}  p {pmw:.3g}")
def holm_correct(col):
    ps = [(i, irows[i][col]) for i in range(len(irows)) if np.isfinite(irows[i][col])]
    ps.sort(key=lambda z: z[1]); out = {}; m = len(ps); run = 0.0
    for rank, (i, pv) in enumerate(ps):
        run = max(run, min(1.0, (m - rank) * pv)); out[i] = run
    return out, m
holm, m = holm_correct(9)
holm_pm, _ = holm_correct(12)
with open("ism_donor_test.tsv", "w") as fh:
    fh.write("motif\tconsensus\tn_occ_real\tn_occ_shuffled\tn_paired_seqs\tmean_abs_ism_real\t"
             "mean_abs_ism_shuffled\tratio\twilcoxon_z\twilcoxon_p\tratio_position_matched\t"
             "z_position_matched\tp_position_matched\tn_paired_position_matched\t"
             "mean_pos_real\tmean_pos_shuffled\tholm_p\tholm_p_position_matched\tn_motifs_tested\n")
    for i, r in enumerate(irows):
        fh.write("\t".join(str(x) if isinstance(x, int) else (x if isinstance(x, str) else f"{x:.6g}")
                           for x in r)
                 + f"\t{holm.get(i, float('nan')):.6g}"
                 + f"\t{holm_pm.get(i, float('nan')):.6g}\t{m}\n")
for i, r in enumerate(irows):
    if np.isfinite(r[9]):
        log(f"[ISM-holm] {r[0]:16s} raw p {r[9]:.4g}  Holm p {holm[i]:.4g}  "
            f"{'SURVIVES' if holm[i] < 0.05 else 'ns'};  position-matched raw p "
            f"{r[12]:.4g}  Holm p {holm_pm.get(i, float('nan')):.4g}  "
            f"{'SURVIVES' if holm_pm.get(i, 1) < 0.05 else 'ns'}")

dn_ism = [r for r in irows if r[0].startswith("donor") or r[0].startswith("GU")]
verdict = "IS" if DONOR_CALLS else "IS NOT"
with open("donor_verdict.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    fh.write(f"verdict\ta donor/U1 motif {verdict} recovered\n")
    fh.write(f"n_filters_passing_all_three\t{len(DONOR_CALLS)}\n")
    fh.write(f"filters_passing_all_three\t{'|'.join(r['name'] for r in DONOR_CALLS) or 'none'}\n")
    fh.write(f"n_filters_ic_ge_{IC_FLOOR:.0f}\t{sum(r['c1'] for r in recs)}\n")
    fh.write(f"n_filters_donor_ref_beats_null\t{len(dcand)}\n")
    fh.write(f"contrib_threshold_p{CONTRIB_Q}\t{CT:.6f}\n")
    for bgn in ("dinuc_shuffled", "real_competitor"):
        fh.write(f"gia_{bgn}_donor_AGGTAAGT\t{gia_best[(bgn,'donor_AGGTAAGT')][1]:.5f}\n")
        fh.write(f"gia_{bgn}_donor_AGGTAAGT_pos\t{gia_best[(bgn,'donor_AGGTAAGT')][0]}\n")
        fh.write(f"gia_{bgn}_donor_perm_ctrl\t{gia_best[(bgn,'donor_perm_ctrl_TAGAGTAG')][1]:.5f}\n")
        fh.write(f"gia_{bgn}_acceptor\t{gia_best[(bgn,'acceptor_pyAG')][1]:.5f}\n")
        fh.write(f"gia_{bgn}_PAS_AATAAA\t{gia_best[(bgn,'PAS_AATAAA')][1]:.5f}\n")
        fh.write(f"gia_{bgn}_PAS_perm_ctrl\t{gia_best[(bgn,'PAS_perm_ctrl_AAAATA')][1]:.5f}\n")
    for i, r in enumerate(irows):
        fh.write(f"ism_{r[0]}_ratio\t{r[7]:.5f}\nism_{r[0]}_p\t{r[9]:.5g}\n"
             f"ism_{r[0]}_holm_p\t{holm.get(i, float('nan')):.5g}\n"
             f"ism_{r[0]}_ratio_position_matched\t{r[10]:.5f}\n"
             f"ism_{r[0]}_p_position_matched\t{r[12]:.5g}\n"
             f"ism_{r[0]}_holm_p_position_matched\t{holm_pm.get(i, float('nan')):.5g}\n"
             f"ism_{r[0]}_mean_pos_real\t{r[14]:.1f}\n"
             f"ism_{r[0]}_mean_pos_shuffled\t{r[15]:.1f}\n")
    fh.write("position_matched_note\tA raw real-vs-shuffle ISM ratio is positionally confounded: "
             "a motif positionally enriched in real sequence lands in high-importance positions "
             "while its shuffled copies are scattered. The position-matched null holds position "
             "fixed and changes only content, and is the one to read.\n")
    fh.write("pas_ctrl_caveat\tAATAAA is A5T1, so every permutation is one base from itself; "
             "the PAS permutation control is weak by construction. The donor control (A3G3T2) is not.\n")
log(f"\n[VERDICT] a donor/U1 motif {verdict} recovered by this readout.")
log(f"[out] filter_pwms_readout.meme filter_matches_readout.tsv gia_donor_test.tsv "
    f"ism_donor_test.tsv donor_verdict.tsv")
