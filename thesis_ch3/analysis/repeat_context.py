#!/usr/bin/env python3
# which repeats do competitor sites and their windows lie in?
import os, sys, math, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
src = open(os.path.join(HERE, "stats_ch3.py")).read(); cut = src.index("# ---------------------------------------------------------------- data")
ns = {"__file__": os.path.join(HERE, "stats_ch3.py")}; exec(src[:cut], ns); wilcoxon = ns["wilcoxon"]
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
need = set(ch)
GROUP = lambda cls, fam: ("Alu" if fam == "Alu" else "other SINE (MIR)" if cls == "SINE" else "L1" if fam == "L1" else
                          "L2 and other LINE" if cls == "LINE" else "LTR" if cls.startswith("LTR") else "DNA transposon" if cls.startswith("DNA")
                          else "simple repeat or low complexity" if cls in ("Simple_repeat", "Low_complexity") else "other")
GROUPS = ["Alu", "other SINE (MIR)", "L1", "L2 and other LINE", "LTR", "DNA transposon", "simple repeat or low complexity", "other"]
R = collections.defaultdict(list)
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[5] not in need: continue
        s_, e_ = int(f[6]), int(f[7]); strand = f[9]; cls, fam = f[11], f[12]
        cstart = int(f[13]) if strand == "+" else int(f[15]); cend = int(f[14])
        R[f[5]].append((s_, e_, strand, f[10], GROUP(cls, fam), cstart, cend))
for k in R: R[k].sort()
S = {k: np.array([r[0] for r in v]) for k, v in R.items()}; E = {k: np.array([r[1] for r in v]) for k, v in R.items()}
MAXLEN = {k: int((E[k] - S[k]).max()) for k in R}
log(f"[rmsk] {sum(len(v) for v in R.values()):,} elements on {len(R)} chromosomes")
def overlapping(c, a, b):
    lo = np.searchsorted(S[c], a - MAXLEN[c]); hi = np.searchsorted(S[c], b)
    idx = np.arange(lo, hi); return idx[E[c][idx] > a]
N = len(y); cov = np.zeros((N, len(GROUPS)), np.float32)
site_grp = np.array(["none"] * N, dtype=object); site_sense = np.zeros(N, np.int8); site_name = np.array([""] * N, dtype=object)
alu_pos = np.full(N, np.nan)
for i in range(N):
    c = ch[i]; p = int(cen[i]); s = st[i]
    a, b = (p - 2000, p) if s == "+" else (p + 1, p + 2001)
    mask = {gname: np.zeros(2000, bool) for gname in GROUPS}
    for j in overlapping(c, a, b):
        e = R[c][j]; lo, hi = max(a, e[0]) - a, min(b, e[1]) - a; mask[e[4]][lo:hi] = True
    cov[i] = [mask[gname].mean() for gname in GROUPS]
    hit = overlapping(c, p, p + 1)
    if len(hit):
        e = R[c][hit[0]]; site_grp[i] = e[4]; site_name[i] = e[3]; site_sense[i] = 1 if e[2] == s else -1
        if e[4] == "Alu":
            frac = (p - e[0]) / max(1, e[1] - e[0]); alu_pos[i] = e[5] + frac * (e[6] - e[5]) if e[2] == "+" else e[6] - frac * (e[6] - e[5])
    if i % 10000 == 0: log(f"  {i:,}")
with open(os.path.join(ROOT, "data/repeat_sites.tsv"), "w") as fh:
    fh.write("idx\tsite_group\tsite_name\tsite_orientation\talu_consensus_pos\t" + "\t".join("cov_" + x.replace(" ", "_") for x in GROUPS) + "\n")
    for i in range(N):
        fh.write(f"{i}\t{site_grp[i]}\t{site_name[i]}\t{site_sense[i]}\t{alu_pos[i]:.1f}\t" + "\t".join(f"{v:.4f}" for v in cov[i]) + "\n")
OUT = []
def out(*a): OUT.append(a); log("  ".join(str(x) for x in a))
for gname in GROUPS + ["none"]:
    for lab, m in (("real", y == 1), ("competitor", y == 0)):
        sel = site_grp[m] == gname
        out("site", gname, lab, f"{sel.mean():.4f}", f"sense {np.mean(site_sense[m][sel] == 1):.3f}" if sel.any() else "")
am = (site_grp == "Alu") & (y == 0)
h, edges = np.histogram(alu_pos[am & (site_sense == 1)], bins=np.arange(0, 330, 30))
out("alu_position_sense_competitors", " ".join(f"{int(a)}-{int(b)}:{n}" for a, b, n in zip(edges[:-1], edges[1:], h)))
h, edges = np.histogram(alu_pos[am & (site_sense == -1)], bins=np.arange(0, 330, 30))
out("alu_position_antisense_competitors", " ".join(f"{int(a)}-{int(b)}:{n}" for a, b, n in zip(edges[:-1], edges[1:], h)))
gidx = collections.defaultdict(list)
for i, gg in enumerate(g): gidx[gg].append(i)
gidx = [np.array(v) for v in gidx.values()]
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); L = C["logit"].astype(float)
Lc = L.copy(); covc = cov.astype(float).copy()
for ii in gidx: Lc[ii] -= L[ii].mean(); covc[ii] -= cov[ii].mean(0)
for j, gname in enumerate(GROUPS):
    d = np.array([cov[ii, j][y[ii] == 1][0] - cov[ii, j][y[ii] == 0].mean() for ii in gidx]); z, p, n = wilcoxon(d)
    r2 = np.corrcoef(covc[:, j], Lc)[0, 1] ** 2
    out("window", gname, f"real {cov[y==1, j].mean():.4f}", f"competitor {cov[y==0, j].mean():.4f}", f"within-gene mean diff {d.mean():+.4f}, z = {z:.1f}, r = {abs(z)/math.sqrt(max(n,1)):.3f}, P = {p:.2g}",
        f"network logit R^2 {r2:.3f}, slope sign {np.sign(np.corrcoef(covc[:, j], Lc)[0,1]):+.0f}")
Mb = np.column_stack([np.ones(N), covc]); beta, *_ = np.linalg.lstsq(Mb, Lc, rcond=None)
out("window", "all families jointly", f"network logit R^2 {1 - ((Lc - Mb @ beta)**2).sum() / ((Lc - Lc.mean())**2).sum():.3f}")
with open(os.path.join(ROOT, "data/repeat_context.tsv"), "w") as fh:
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/repeat_sites.tsv data/repeat_context.tsv")
# orientation background for l1
ant = 0; tot = 0
for i in np.flatnonzero(y == 0):
    c = ch[i]; p = int(cen[i]); s = st[i]; a, b = (p - 2000, p) if s == "+" else (p + 1, p + 2001)
    for j in overlapping(c, a, b):
        e = R[c][j]
        if e[4] != "L1": continue
        L = min(b, e[1]) - max(a, e[0]); tot += L; ant += L * (e[2] != s)
m = (y == 0) & (site_grp == "L1")
line = f"L1 orientation: antisense among competitor-site L1 elements {np.mean(site_sense[m] == -1):.3f} (n {m.sum()}); among all L1 bases in competitor upstream windows {ant/tot:.3f}"
log(line); open(os.path.join(ROOT, "data/repeat_context.tsv"), "a").write("orientation\t" + line + "\n")
