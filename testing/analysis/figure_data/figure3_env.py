#!/usr/bin/env python3
# figure 3 -- environment of competitors vs canonical sites
import csv, sys, numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
WINS = [1, 2, 5, 10, 20]

# master
R = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
NP = len(R)
gene = np.array([r["gene"] for r in R])
chrom = np.array([r["chrom"] for r in R]); center = np.array([int(r["center"]) for r in R])
strand = np.array([r["strand"] for r in R])
iscanon = np.array([r["is_canonical"] == "True" for r in R])
contested = np.array([r["contested"] == "True" for r in R])
grp = np.where(~iscanon, "COMPETITOR", np.where(contested, "FAILURE", "SUCCESS"))
GROUPS = ["SUCCESS", "FAILURE", "COMPETITOR"]
gmask = {G: (grp == G) for G in GROUPS}
log("group counts: " + "  ".join(f"{G} {int(gmask[G].sum())}" for G in GROUPS))

ug, gid = np.unique(gene, return_inverse=True)

# annotation block (aligned to master)
A = np.load("env_annot_all.npz")
comp = A["comp"].astype(int); fracpos = A["fracpos"]; tss_dist = A["tss_dist"]
atlas_dist = A["atlas_dist"]; atlas_frac = A["atlas_frac"]; atlas_nsamp = A["atlas_nsamp"].astype(float)
padb_dist = A["padb_dist"]; padb_rpm = A["padb_rpm"]; padb_pse = A["padb_pse"]

# scan block: acc, asym, u1 (all groups)
acc = np.full(NP, np.nan); u1 = np.full(NP, np.nan)
asy3 = {w: np.full(NP, np.nan) for w in WINS}
n5up2 = np.full(NP, np.nan)

P = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
assert len(P) == 70723
posacc = np.array([float(r["acc"]) if r["acc"] not in ("", "nan") else np.nan for r in P])
AR = np.load("asym_real_cache.npz"); Z = np.load("u1_window_cache.npz")
u1c = Z["dens_5.25"]
ARu = {w: AR[f"n3ss_up_{w}"] for w in WINS}; ARd = {w: AR[f"n3ss_dn_{w}"] for w in WINS}
AR_n5up2 = AR["n5ss_up_2"]
cmap = {(P[i]["gene"], P[i]["chrom"], int(P[i]["center"]), P[i]["strand"]): i for i in range(len(P))}
nmatch = 0
for i in range(NP):
    if not contested[i]: continue
    row = cmap.get((gene[i], chrom[i], int(center[i]), strand[i]))
    if row is None: continue
    nmatch += 1
    acc[i] = posacc[row]; u1[i] = u1c[row]; n5up2[i] = AR_n5up2[row]
    for w in WINS: asy3[w][i] = ARu[w][row] - ARd[w][row]
log(f"contested peaks joined to cache: {nmatch} / {int((contested).sum())}")

S = np.load("success_scan_cache.npz", allow_pickle=True)
succ_idx = np.where(grp == "SUCCESS")[0]
assert len(succ_idx) == len(S["acc"]), (len(succ_idx), len(S["acc"]))
assert (S["gene"] == gene[succ_idx]).all() and (S["center"].astype(int) == center[succ_idx]).all()
acc[succ_idx] = S["acc"]; u1[succ_idx] = S["u1_525"]; n5up2[succ_idx] = S["n5ss_up_2"]
for w in WINS: asy3[w][succ_idx] = S[f"n3ss_up_{w}"] - S[f"n3ss_dn_{w}"]
log("SUCCESS scan block assigned")

# decoy block (panel a reference)
DA = np.load("env_annot_decoy.npz"); DS = np.load("decoy_set.npz")
dcomp = DS["comp"].astype(int)
dgene = np.array([P[c]["gene"] for c in dcomp])
_, dgid = np.unique(dgene, return_inverse=True)
d_atlas_dist = DA["atlas_dist"]; d_atlas_frac = DA["atlas_frac"]; d_atlas_nsamp = DA["atlas_nsamp"].astype(float)
d_padb_dist = DA["padb_dist"]; d_comp = DA["comp"].astype(int)

RNG = np.random.default_rng(11)
def _members(ids, n):
    m = defaultdict(list)
    for i in range(n): m[ids[i]].append(i)
    return [np.array(v) for v in m.values()]

def cboot(vals, ids, fn=None, B=1000):
    v = np.asarray(vals, float); fin = np.isfinite(v)
    o = np.argsort(ids, kind="mergesort"); ids_s = ids[o]; v_s = v[o]; fin_s = fin[o]
    cut = np.r_[0, np.flatnonzero(ids_s[1:] != ids_s[:-1]) + 1, len(ids_s)]
    ng = len(cut) - 1; sg = np.empty(ng); cg = np.empty(ng)
    for j in range(ng):
        seg = slice(cut[j], cut[j + 1]); f = fin_s[seg]
        sg[j] = v_s[seg][f].sum(); cg[j] = f.sum()
    tot = cg.sum(); obs = sg.sum() / tot if tot > 0 else np.nan
    idx = RNG.integers(0, ng, size=(B, ng))
    S = sg[idx].sum(1); N = cg[idx].sum(1)
    draws = np.where(N > 0, S / np.where(N > 0, N, 1.0), np.nan)
    lo, hi = np.percentile(draws[np.isfinite(draws)], [2.5, 97.5])
    return obs, lo, hi, draws

def frac_fn(thr_fn):
    return lambda v: float(np.mean(thr_fn(v)))
def nanmean(v):
    v = v[np.isfinite(v)]; return float(v.mean()) if len(v) else np.nan
def nanmedian(v):
    v = v[np.isfinite(v)]; return float(np.median(v)) if len(v) else np.nan

def diff_ci(drA, drB, obsA, obsB):
    d = drA - drB
    return obsA - obsB, *np.percentile(d[np.isfinite(d)], [2.5, 97.5])

def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); sv = s[order]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[order] = rk; npos = yv.sum(); nneg = n - npos
    return np.nan if npos == 0 or nneg == 0 else (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def _memof(ids):
    o = np.argsort(ids, kind="mergesort"); ids_s = ids[o]
    cut = np.r_[0, np.flatnonzero(ids_s[1:] != ids_s[:-1]) + 1, len(ids_s)]
    return [o[cut[j]:cut[j + 1]] for j in range(len(cut) - 1)]

def auroc_between(xa, ida, xb, idb, B=400):
    ma = np.isfinite(xa); mb = np.isfinite(xb)
    xa, ida = xa[ma], ida[ma]; xb, idb = xb[mb], idb[mb]
    def auf(a, b):
        b2 = np.sort(b); nB = len(b2)
        lo = np.searchsorted(b2, a, "left"); hi = np.searchsorted(b2, a, "right")
        return float(np.mean(lo + 0.5 * (hi - lo)) / nB)
    obs = auf(xa, xb)
    mema = _memof(ida); memb = _memof(idb); na = len(mema); nb = len(memb)
    dr = np.empty(B)
    for k in range(B):
        a = xa[np.concatenate([mema[p] for p in RNG.integers(0, na, na)])]
        b = xb[np.concatenate([memb[p] for p in RNG.integers(0, nb, nb)])]
        dr[k] = auf(a, b)
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return obs, lo, hi

def within_gene(val, higher_canon=True):
    rr = []
    gmap2 = defaultdict(lambda: {"c": [], "k": []})
    for i in range(NP):
        if not contested[i]: continue
        (gmap2[gene[i]]["c" if iscanon[i] else "k"]).append(i)
    for g, d in gmap2.items():
        ci = np.array(d["c"]); ki = np.array(d["k"])
        if len(ci) != 1 or len(ki) < 1: continue
        cv = val[ci]; kv = val[ki]; m = np.isfinite(kv)
        if not np.isfinite(cv[0]) or m.sum() < 1: continue
        kv = kv[m]; c = cv[0]
        w = ((c > kv).sum() + 0.5 * (c == kv).sum()) if higher_canon else ((kv > c).sum() + 0.5 * (kv == c).sum())
        rr.append(w / len(kv))
    rr = np.array(rr); obs = rr.mean(); ng = len(rr)
    bs = rr[RNG.integers(0, ng, size=(2000, ng))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(wr=obs, exc=obs - 0.5, lo=lo, hi=hi, ng=ng)

with open("fig3env_groupn.tsv", "w") as fh:
    fh.write("group\tn_peaks\tn_genes\n")
    for G in GROUPS:
        fh.write(f"{G}\t{int(gmask[G].sum())}\t{len(np.unique(gene[gmask[G]]))}\n")
    fh.write(f"DECOY\t{len(dcomp)}\t{len(np.unique(dgene))}\n")

def any_atlas(dist_a, dist_p): return (dist_a <= 250) | (dist_p <= 250)
rowsA = []; rowsAcmp = []
grp_any = {}
for G in GROUPS:
    m = gmask[G]
    aa = any_atlas(atlas_dist[m], padb_dist[m]).astype(float)
    ids = gid[m]
    o, lo, hi, dr = cboot(aa, ids, lambda v: float(v.mean()))
    grp_any[G] = (aa, ids, o, dr)
    hit = atlas_dist[m] <= 250
    med_frac = nanmedian(atlas_frac[m][hit]); med_ns = nanmedian(atlas_nsamp[m][hit])
    f10 = float(np.mean((atlas_dist[m] <= 250) & (atlas_frac[m] >= 0.10)))
    f25 = float(np.mean((atlas_dist[m] <= 250) & (atlas_frac[m] >= 0.25)))
    rowsA.append((G, int(m.sum()), o, lo, hi, med_frac, med_ns, f10, f25, int(hit.sum())))
d_aa = any_atlas(d_atlas_dist, d_padb_dist).astype(float)
o, lo, hi, dr_dec = cboot(d_aa, dgid, lambda v: float(v.mean()))
d_hit = d_atlas_dist <= 250
rowsA.append(("DECOY", len(d_aa), o, lo, hi,
              nanmedian(d_atlas_frac[d_hit]), nanmedian(d_atlas_nsamp[d_hit]),
              float(np.mean((d_atlas_dist <= 250) & (d_atlas_frac >= 0.10))),
              float(np.mean((d_atlas_dist <= 250) & (d_atlas_frac >= 0.25))), int(d_hit.sum())))
grp_any["DECOY"] = (d_aa, dgid, o, dr_dec)
for a, b in [("COMPETITOR", "DECOY"), ("COMPETITOR", "FAILURE"), ("FAILURE", "SUCCESS")]:
    va, ia, oa, dra = grp_any[a]; vb, ib, ob, drb = grp_any[b]
    d, dlo, dhi = diff_ci(dra, drb, oa, ob)
    rowsAcmp.append((f"{a} - {b}", "any_atlas_frac_diff", d, dlo, dhi))
with open("fig3env_panelA.tsv", "w") as fh:
    fh.write("group\tn\tany_atlas\tlo\thi\tmed_usage\tmed_nsamp\tfrac_ge10\tfrac_ge25\tn_hit\n")
    for r in rowsA: fh.write("\t".join(str(x) for x in r) + "\n")
with open("fig3env_panelA_cmp.tsv", "w") as fh:
    fh.write("comparison\tmetric\teffect\tlo\thi\n")
    for r in rowsAcmp: fh.write("\t".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in r) + "\n")

LABC = {0: "TERM", 1: "INTERNAL", 2: "INTRON", 3: "OTHER"}
with open("fig3env_panelB.tsv", "w") as fh:
    fh.write("group\tcompartment\tfrac\tlo\thi\n")
    for G in GROUPS + ["DECOY"]:
        if G == "DECOY":
            cc = d_comp; ids = dgid; n = len(cc)
        else:
            m = gmask[G]; cc = comp[m]; ids = gid[m]; n = int(m.sum())
        for k in range(4):
            ind = (cc == k).astype(float)
            o, lo, hi, _ = cboot(ind, ids, lambda v: float(v.mean()))
            fh.write(f"{G}\t{LABC[k]}\t{o:.5f}\t{lo:.5f}\t{hi:.5f}\n")
with open("fig3env_panelB_cmp.tsv", "w") as fh:
    fh.write("comparison\tmetric\teffect\tlo\thi\n")
    def comp_frac(G, k):
        if G == "DECOY": cc = d_comp; ids = dgid
        else: cc = comp[gmask[G]]; ids = gid[gmask[G]]
        return cboot((cc == k).astype(float), ids, lambda v: float(v.mean()))
    for k, nm in [(2, "intron"), (0, "term")]:
        for a, b in [("COMPETITOR", "FAILURE"), ("SUCCESS", "FAILURE")]:
            oa, _, _, dra = comp_frac(a, k); ob, _, _, drb = comp_frac(b, k)
            d, dlo, dhi = diff_ci(dra, drb, oa, ob)
            fh.write(f"{a} - {b}\tfrac_{nm}_diff\t{d:.5f}\t{dlo:.5f}\t{dhi:.5f}\n")

with open("fig3env_panelC_vals.tsv", "w") as fh:
    fh.write("group\tfracpos\ttss_dist\n")
    for G in GROUPS:
        m = gmask[G]
        for fp, td in zip(fracpos[m], tss_dist[m]):
            if np.isfinite(fp): fh.write(f"{G}\t{fp:.5f}\t{td:.1f}\n")
with open("fig3env_panelC_stats.tsv", "w") as fh:
    fh.write("group\tmean_fracpos\tlo\thi\tmedian_fracpos\tmedian_tss_kb\n")
    mfp = {}
    for G in GROUPS:
        m = gmask[G]
        o, lo, hi, dr = cboot(fracpos[m], gid[m])
        mfp[G] = (o, dr)
        fh.write(f"{G}\t{o:.4f}\t{lo:.4f}\t{hi:.4f}\t{nanmedian(fracpos[m]):.4f}\t{nanmedian(tss_dist[m])/1000:.2f}\n")
with open("fig3env_panelC_cmp.tsv", "w") as fh:
    fh.write("comparison\tmetric\teffect\tlo\thi\n")
    for a, b in [("COMPETITOR", "FAILURE"), ("SUCCESS", "FAILURE")]:
        oa, dra = mfp[a]; ob, drb = mfp[b]
        d, dlo, dhi = diff_ci(dra, drb, oa, ob)
        fh.write(f"{a} - {b}\tmean_fracpos_diff\t{d:.4f}\t{dlo:.4f}\t{dhi:.4f}\n")
        au, alo, ahi = auroc_between(fracpos[gmask[a]], gid[gmask[a]], fracpos[gmask[b]], gid[gmask[b]])
        fh.write(f"{a} > {b}\tAUROC_fracpos\t{au:.4f}\t{alo:.4f}\t{ahi:.4f}\n")

with open("fig3env_panelD_vals.tsv", "w") as fh:
    fh.write("group\tacc\n")
    for G in GROUPS:
        for v in acc[gmask[G]]:
            if np.isfinite(v): fh.write(f"{G}\t{v:.4f}\n")
with open("fig3env_panelD_stats.tsv", "w") as fh:
    fh.write("group\tmean_acc\tlo\thi\tmedian_acc\n")
    for G in GROUPS:
        m = gmask[G]
        o, lo, hi, _ = cboot(acc[m], gid[m], nanmean)
        fh.write(f"{G}\t{o:.4f}\t{lo:.4f}\t{hi:.4f}\t{nanmedian(acc[m]):.4f}\n")
with open("fig3env_panelD_cmp.tsv", "w") as fh:
    fh.write("comparison\tmetric\teffect\tlo\thi\n")
    for a, b in [("FAILURE", "COMPETITOR"), ("SUCCESS", "FAILURE"), ("SUCCESS", "COMPETITOR")]:
        au, alo, ahi = auroc_between(acc[gmask[a]], gid[gmask[a]], acc[gmask[b]], gid[gmask[b]])
        fh.write(f"{a} > {b}\tAUROC_acc\t{au:.4f}\t{alo:.4f}\t{ahi:.4f}\n")
    wgD = within_gene(acc, higher_canon=True)
    fh.write(f"FAILURE vs own-COMPETITOR\twin_rate_acc\t{wgD['wr']:.4f}\t{wgD['lo']:.4f}\t{wgD['hi']:.4f}\n")
lol = []
cby = defaultdict(lambda: {"c": [], "k": []})
for i in range(NP):
    if not contested[i]: continue
    cby[gene[i]]["c" if iscanon[i] else "k"].append(i)
for g, d in cby.items():
    ci = np.array(d["c"]); ki = np.array(d["k"])
    if len(ci) != 1: continue
    cv = acc[ci[0]]; kv = acc[ki]; kv = kv[np.isfinite(kv)]
    if not np.isfinite(cv) or len(kv) == 0: continue
    lol.append(cv - kv.mean())
lol = np.sort(np.array(lol))
frac_pos = float((lol > 0).mean())
with open("fig3env_panelD_lolli.tsv", "w") as fh:
    fh.write("rank\tdiff\n")
    for r, v in enumerate(lol): fh.write(f"{r}\t{v:.4f}\n")
with open("fig3env_panelD_lolli_meta.tsv", "w") as fh:
    fh.write("n_genes\tfrac_above0\tmean_diff\n")
    fh.write(f"{len(lol)}\t{frac_pos:.4f}\t{lol.mean():.4f}\n")

with open("fig3env_panelE_vals.tsv", "w") as fh:
    fh.write("group\tasy5k\n")
    for G in GROUPS:
        for v in asy3[5][gmask[G]]:
            if np.isfinite(v): fh.write(f"{G}\t{v:.1f}\n")
with open("fig3env_panelE_stats.tsv", "w") as fh:
    fh.write("group\tmean_asy5k\tlo\thi\tmedian\n")
    for G in GROUPS:
        m = gmask[G]
        o, lo, hi, _ = cboot(asy3[5][m], gid[m], nanmean)
        fh.write(f"{G}\t{o:.3f}\t{lo:.3f}\t{hi:.3f}\t{nanmedian(asy3[5][m]):.2f}\n")
with open("fig3env_panelE_cmp.tsv", "w") as fh:
    fh.write("comparison\tmetric\teffect\tlo\thi\n")
    for a, b in [("FAILURE", "COMPETITOR"), ("SUCCESS", "COMPETITOR"), ("SUCCESS", "FAILURE")]:
        au, alo, ahi = auroc_between(asy3[5][gmask[a]], gid[gmask[a]], asy3[5][gmask[b]], gid[gmask[b]])
        fh.write(f"{a} > {b}\tAUROC_asy5k\t{au:.4f}\t{alo:.4f}\t{ahi:.4f}\n")
with open("fig3env_panelE_sweep.tsv", "w") as fh:
    fh.write("window_kb\twin_rate\texc\tlo\thi\tng\n")
    for w in WINS:
        r = within_gene(asy3[w], higher_canon=True)
        fh.write(f"{w}\t{r['wr']:.4f}\t{r['exc']:+.4f}\t{r['lo']:.4f}\t{r['hi']:.4f}\t{r['ng']}\n")

with open("fig3env_supp.tsv", "w") as fh:
    fh.write("group\tmetric\tmean\tlo\thi\tmedian\n")
    for G in GROUPS:
        m = gmask[G]
        for nm, arr in [("u1_dens500", u1), ("n5ss_up2kb", n5up2), ("padb_rpm", padb_rpm), ("padb_pse", padb_pse)]:
            o, lo, hi, _ = cboot(arr[m], gid[m], nanmean)
            fh.write(f"{G}\t{nm}\t{o:.4f}\t{lo:.4f}\t{hi:.4f}\t{nanmedian(arr[m]):.4f}\n")

log("wrote fig3env_*.tsv")
print("\n========== FIGURE 3 SUMMARY ==========")
print("n per group:", {G: int(gmask[G].sum()) for G in GROUPS}, "decoy", len(dcomp))
print("\nPanel A  fraction with any atlas site in 500nt window (+/-CI):")
for r in rowsA: print(f"  {r[0]:<11} {r[2]:.3f} [{r[3]:.3f},{r[4]:.3f}]  med_usage {r[5]:.3f}  med_nsamp {r[6]:.1f}  >=10% {r[7]:.3f}  >=25% {r[8]:.3f}  (n={r[1]})")
print("  key: COMPETITOR - DECOY any-atlas diff =", f"{rowsAcmp[0][2]:+.3f} [{rowsAcmp[0][3]:+.3f},{rowsAcmp[0][4]:+.3f}]")
print("\nPanel D  acc mean & rank effects:")
print("  within-gene FAILURE>own-competitor win-rate:", f"{wgD['wr']:.4f} [{wgD['lo']:.4f},{wgD['hi']:.4f}]  lollipop frac>0 {frac_pos:.3f} ({len(lol)}g)")
print("\nPanel E  asymmetry within-gene scale sweep (FAILURE vs competitors):")
for w in WINS:
    r = within_gene(asy3[w], higher_canon=True)
    print(f"  {w:>2}kb  win-rate {r['wr']:.4f} exc {r['exc']:+.4f} [{r['lo']:.4f},{r['hi']:.4f}] ({r['ng']}g)")
print("======================================")
