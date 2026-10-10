#!/usr/bin/env python3
# independent test of u1-dependent usage of competitor sites
import os, sys, subprocess, collections, numpy as np, pyBigWig
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); Z = np.load(os.path.join(ROOT, "data/occupancy_profiles_decoy.npz"), allow_pickle=True)
gene = np.concatenate([D["gene"], Z["gene"]]); ch = np.concatenate([D["chrom"], Z["chrom"]]); cen = np.concatenate([D["center"].astype(np.int64), Z["center"].astype(np.int64)])
cls = np.concatenate([np.where(D["y"].astype(int) == 1, "real 3' end", "competitor"), np.full(len(Z["gene"]), "decoy")])
tmp = os.path.join(ROOT, "data/_lift"); os.makedirs(tmp, exist_ok=True)
with open(os.path.join(tmp, "in.bed"), "w") as fh:
    for i in range(len(cen)): fh.write(f"{ch[i]}\t{cen[i]}\t{cen[i]+1}\t{i}\n")
subprocess.run([os.path.join(PUB, "external/bin/liftOver"), os.path.join(tmp, "in.bed"), os.path.join(PUB, "external/chain/hg38ToHg19.over.chain.gz"),
                os.path.join(tmp, "out.bed"), os.path.join(tmp, "unmapped.bed")], check=True, capture_output=True)
h19 = {}
for l in open(os.path.join(tmp, "out.bed")):
    f = l.split("\t"); h19[int(f[3])] = (f[0], int(f[1]))
log(f"lifted {len(h19):,} of {len(cen):,}")
G = os.path.join(PUB, "external/geo/GSE103252"); W = 50
sig = {}
for cond, fn in (("control", "GSE103252_cAMO_15nmol_odT_selection.polyA_sites.bw"), ("U1 AMO", "GSE103252_U1AMO_15nmol_odT_selection.polyA_sites.bw")):
    bw = pyBigWig.open(os.path.join(G, fn)); sz = bw.chroms(); v = np.full(len(cen), np.nan)
    for i, (c, p) in h19.items():
        if c in sz and p - W >= 0 and p + W + 1 <= sz[c]: v[i] = np.nansum(np.nan_to_num(np.asarray(bw.values(c, p - W, p + W + 1), float)))
    sig[cond] = v; log(cond, "genome sum", bw.header()["sumData"])
ok = np.isfinite(sig["control"]) & np.isfinite(sig["U1 AMO"])
genes = sorted(set(D["gene"])); gix = {g: k for k, g in enumerate(genes)}; NG = len(genes); kg = np.array([gix[g] for g in gene])
def gmean(v, m):
    s = np.zeros(NG); n = np.zeros(NG); np.add.at(s, kg[m], v[m]); np.add.at(n, kg[m], 1); return s / np.maximum(n, 1), n > 0
RNG = np.random.default_rng(0)
def boot(fn, keep):
    k = np.flatnonzero(keep); e = fn(k); b = np.array([fn(k[RNG.integers(0, len(k), len(k))]) for _ in range(2000)]); b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5]); return e, lo, hi, max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1 / 2000), len(k)
OUT = []
def out(*a): OUT.append(a); log("  ".join(str(x) for x in a))
GM = {}
for c in ("real 3' end", "competitor", "decoy"):
    m = ok & (cls == c)
    for cond in ("control", "U1 AMO"):
        GM[(c, cond, "sig")] = gmean(sig[cond], m); GM[(c, cond, "hit")] = gmean((sig[cond] > 0).astype(float), m)
        out(c, cond, "sites", int(m.sum()), "fraction with signal", f"{(sig[cond][m] > 0).mean():.4f}", "mean signal", f"{sig[cond][m].mean():.4f}")
for kind, lab in (("sig", "mean signal"), ("hit", "fraction of sites with signal")):
    f = lambda c, k: np.log2(GM[(c, "U1 AMO", kind)][0][k].sum() / GM[(c, "control", kind)][0][k].sum())
    for c in ("real 3' end", "competitor", "decoy"):
        e = boot(lambda k: f(c, k), GM[(c, "control", kind)][1]); out(lab, c, "log2 U1 AMO / control", f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"P {e[3]:.3g}", e[4])
    both = GM[("competitor", "control", kind)][1] & GM[("decoy", "control", kind)][1]
    e = boot(lambda k: f("competitor", k) - f("decoy", k), both); out(lab, "competitor minus decoy", "difference in log2 U1 AMO / control", f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"P {e[3]:.3g}", e[4])
    for cond in ("control", "U1 AMO"):
        e = boot(lambda k: np.log2(GM[("competitor", cond, kind)][0][k].sum() / GM[("decoy", cond, kind)][0][k].sum()), both)
        out(lab, "competitor over decoy", cond, f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"P {e[3]:.3g}", e[4])
with open(os.path.join(ROOT, "data/oh2017_fraction.tsv"), "w") as fh:
    fh.write("class\tcondition\tsites\tsites_with_signal\tfraction\tlo\thi\n")
    for c in ("real 3' end", "competitor", "decoy"):
        m = ok & (cls == c)
        for cond in ("control", "U1 AMO"):
            hit = (sig[cond] > 0).astype(float); s_ = np.zeros(NG); n_ = np.zeros(NG); np.add.at(s_, kg[m], hit[m]); np.add.at(n_, kg[m], 1)
            e = boot(lambda k: s_[k].sum() / n_[k].sum(), n_ > 0)
            fh.write(f"{c}\t{cond}\t{int(m.sum())}\t{int(hit[m].sum())}\t{e[0]:.5f}\t{e[1]:.5f}\t{e[2]:.5f}\n")
with open(os.path.join(ROOT, "data/oh2017_sites.tsv"), "w") as fh:
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
np.savez_compressed(os.path.join(ROOT, "data/oh2017_per_site.npz"), control=sig["control"], u1amo=sig["U1 AMO"], cls=cls, gene=gene)
log("[out] data/oh2017_sites.tsv")
