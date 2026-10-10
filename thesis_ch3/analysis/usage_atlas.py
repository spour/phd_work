#!/usr/bin/env python3
# are the competitor sites identified from sequence used as poly(a) sites?
import os, sys, glob, subprocess, collections, numpy as np, pyBigWig
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, ".."); G = os.path.join(PUB, "external/geo")
sys.path.insert(0, "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/17.0.0/lib/python3.11/site-packages")
import pyarrow.parquet as pq
def log(*a): print(*a, file=sys.stderr, flush=True)
T = pq.read_table(os.path.join(PUB, "cpa_peaks_final.parquet")).to_pydict()
gene = np.array(T["gene"]).astype(str); ch = np.array(T["chrom"]).astype(str); cen = np.array(T["center"]).astype(np.int64); st = np.array(T["strand"]).astype(str)
can = np.array(T["is_canonical"]); ana = np.array(T["in_analysis_set"]); apa = np.array(T["flag_apa_annotated"])
# intronic controls
GTF = os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf"); ex = collections.defaultdict(list); keep = set(gene)
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] == "exon" and 'transcript_type "protein_coding"' in f[8]:
            gn = f[8].split('gene_name "')[1].split('"')[0]
            if gn in keep: ex[(gn, f[0], f[6])].append((int(f[3]) - 1, int(f[4])))
rng = np.random.default_rng(0); gi = collections.defaultdict(list)
for i, g_ in enumerate(gene): gi[g_].append(i)
cg, cc, cp, cs = [], [], [], []
for g_, ii in gi.items():
    ii = np.array(ii); r = ii[can[ii]]; comp = ii[~can[ii]]
    if len(r) != 1 or len(comp) == 0: continue
    k = (g_, ch[r[0]], st[r[0]])
    if k not in ex: continue
    iv = sorted(ex[k]); m = []
    for a, b in iv:
        if m and a <= m[-1][1]: m[-1][1] = max(m[-1][1], b)
        else: m.append([a, b])
    intr = [(m[j][1], m[j + 1][0]) for j in range(len(m) - 1) if m[j + 1][0] - m[j][1] > 200]
    end = cen[r[0]]; intr = [(a, min(b, end - 500)) for a, b in intr] if st[r[0]] == "+" else [(max(a, end + 500), b) for a, b in intr]
    intr = [(a + 100, b - 100) for a, b in intr if b - a > 400]
    if not intr: continue
    L = np.array([b - a for a, b in intr]); cum = np.cumsum(L); got = 0; tries = 0; cand = cen[ii]
    while got < 3 * len(comp) and tries < 30 * len(comp) + 100:
        tries += 1; u = rng.integers(0, cum[-1]); j = np.searchsorted(cum, u, side="right"); p = intr[j][0] + (u - (cum[j - 1] if j else 0))
        if np.abs(cand - p).min() < 500: continue
        cg.append(g_); cc.append(k[1]); cp.append(int(p)); cs.append(k[2]); got += 1
ncand = len(gene)
gene = np.concatenate([gene, cg]); ch = np.concatenate([ch, cc]); cen = np.concatenate([cen, np.array(cp, np.int64)]); st = np.concatenate([st, cs])
cls = np.concatenate([np.where(can, "real 3' end", "competitor"), np.full(len(cg), "intronic control")]); N = len(cen)
inana = np.concatenate([ana, np.zeros(len(cg), bool)]); isapa = np.concatenate([apa, np.zeros(len(cg), bool)])
log(f"sites {N:,}: competitors {(cls=='competitor').sum():,}, controls {len(cg):,}")
# atlas
A = collections.defaultdict(list)
with open(os.path.join(PUB, "../train/data/atlas.clusters.2.0.GRCh38.96.tsv")) as fh:
    fh.readline()
    for line in fh:
        f = line.rstrip("\n").split("\t"); ns = sum(1 for x in f[13:] if x not in ("0", "", "NA") and float(x) > 0)
        A[("chr" + f[0], f[5])].append((int(f[6]), ns))
AP = {k: np.array(sorted(v)) for k, v in A.items()}
dist = np.full(N, 10 ** 9); nlib = np.zeros(N, int)
for i in range(N):
    a = AP.get((ch[i], st[i]))
    if a is None: continue
    j = np.searchsorted(a[:, 0], cen[i]); c_ = [x for x in (j - 1, j) if 0 <= x < len(a)]
    dist[i] = min(abs(a[x, 0] - cen[i]) for x in c_)
    lo, hi = np.searchsorted(a[:, 0], cen[i] - 50), np.searchsorted(a[:, 0], cen[i] + 50, side="right")
    if hi > lo: nlib[i] = a[lo:hi, 1].max()
# cells
tmp = os.path.join(ROOT, "data/_lift"); os.makedirs(tmp, exist_ok=True)
with open(os.path.join(tmp, "in2.bed"), "w") as fh:
    for i in range(N): fh.write(f"{ch[i]}\t{cen[i]}\t{cen[i]+1}\t{i}\n")
subprocess.run([os.path.join(PUB, "external/bin/liftOver"), os.path.join(tmp, "in2.bed"), os.path.join(PUB, "external/chain/hg38ToHg19.over.chain.gz"),
                os.path.join(tmp, "out2.bed"), os.path.join(tmp, "unmapped2.bed")], check=True, capture_output=True)
h19c = np.array([""] * N, dtype=object); h19p = np.full(N, -1)
for l in open(os.path.join(tmp, "out2.bed")):
    f = l.split("\t"); h19c[int(f[3])] = f[0]; h19p[int(f[3])] = int(f[1])
F = lambda pat: sorted(glob.glob(os.path.join(G, pat)))
EXP = [("HeLa (Oh 2017)", "U1 AMO", "hg19", "u50", F("GSE103252/*cAMO_15nmol*polyA_sites.bw"), F("GSE103252/*U1AMO_15nmol*polyA_sites.bw")),
       ("HeLa (Feng 2025)", "U1 AMO", "hg38", "u", F("GSE250342/*_Control-[ab]_1*"), F("GSE250342/*_U1-[ab]_1*")),
       ("HEK293 (Feng 2025)", "U1 AMO", "hg38", "u", F("GSE250342/*_293-ctr-*"), F("GSE250342/*_293-U1-*")),
       ("SW480 (Feng 2025)", "U1 AMO", "hg38", "u", F("GSE250342/*SW480-NC-*"), F("GSE250342/*SW480-U1-*")),
       ("HeLa (Feng 2024)", "U1 AMO", "hg38", "u", F("GSE262383/*HeLa-ctr-*"), F("GSE262383/*HeLa-U1-*")),
       ("HEK293T (Soles 2026)", "U1 AMO", "hg19", "u", F("GSE275154/*_c25_*"), F("GSE275154/*_u1_*")),
       ("HeLa (Kamieniarz-Gdula 2019)", "PCF11 knockdown", "hg19", "s", [os.path.join(G, "GSE123105/siLUC")], [os.path.join(G, "GSE123105/siPCF11")])]
for e in EXP: assert e[4] and e[5], e
jobs = sorted({(f, e[2], e[3]) for e in EXP for f in e[4] + e[5]})
def work(job):
    f, build, mode = job; v = np.full(N, np.nan, np.float32)
    bws = {"+": pyBigWig.open(f + ".fwd.bw"), "-": pyBigWig.open(f + ".rev.bw")} if mode == "s" else None
    bw = None if mode == "s" else pyBigWig.open(f); sz = (bws["+"] if bws else bw).chroms(); nochr = "chr1" not in sz
    for i in range(N):
        c, p = (ch[i], int(cen[i])) if build == "hg38" else (h19c[i], int(h19p[i]))
        if p < 0 or not c: continue
        if nochr: c = c[3:]
        if c not in sz: continue
        if mode == "u50": a, b = p - 50, p + 51
        else: a, b = (p - 100, p + 21) if st[i] == "+" else (p - 20, p + 101)
        if a < 0 or b > sz[c]: continue
        v[i] = np.nansum(np.nan_to_num(np.asarray((bws[st[i]] if bws else bw).values(c, a, b), np.float32)))
    return f, v
if __name__ == "__main__":
    with Pool(6) as pool: S = dict(pool.map(work, jobs))
    genes = sorted(set(gene)); gix = {g_: k for k, g_ in enumerate(genes)}; NG = len(genes); kg = np.array([gix[g_] for g_ in gene]); RNG = np.random.default_rng(1)
    def frac(hit, m):
        s_ = np.zeros(NG); n_ = np.zeros(NG); np.add.at(s_, kg[m], hit[m].astype(float)); np.add.at(n_, kg[m], 1)
        e = s_.sum() / n_.sum(); b = []
        for _ in range(1000):
            k = RNG.integers(0, NG, NG); b.append(s_[k].sum() / max(n_[k].sum(), 1))
        return e, *np.percentile(b, [2.5, 97.5]), int(m.sum()), int(hit[m].sum())
    SETS = {"real 3' end": cls == "real 3' end", "all competitors": cls == "competitor", "competitors, analysis set": (cls == "competitor") & inana,
            "intronic control": cls == "intronic control"}
    with open(os.path.join(ROOT, "data/usage_atlas.tsv"), "w") as fh:
        fh.write("source\tmeasure\tclass\tfraction\tlo\thi\tsites\thits\n")
        for name, hit in (("within 50 nt of an atlas site", dist <= 50), ("within 500 nt of an atlas site", dist <= 500),
                          ("atlas site within 50 nt, in at least 5 libraries", nlib >= 5), ("atlas site within 50 nt, in at least 20 libraries", nlib >= 20)):
            for c, m in SETS.items():
                e = frac(hit, m); fh.write(f"PolyASite 2.0\t{name}\t{c}\t{e[0]:.5f}\t{e[1]:.5f}\t{e[2]:.5f}\t{e[3]}\t{e[4]}\n"); log(f"atlas  {name:52s} {c:26s} {e[0]:.4f} [{e[1]:.4f},{e[2]:.4f}]  {e[4]}/{e[3]}")
        for lab, trt, build, mode, fc, ft in EXP:
            okk = np.all([np.isfinite(S[f]) for f in fc + ft], 0)
            for cond, fs in (("control", fc), (trt, ft)):
                hit = np.any([np.nan_to_num(S[f]) > 0 for f in fs], 0)
                for c, m in SETS.items():
                    e = frac(hit, m & okk); fh.write(f"{lab}\t{cond}\t{c}\t{e[0]:.5f}\t{e[1]:.5f}\t{e[2]:.5f}\t{e[3]}\t{e[4]}\n"); log(f"{lab:30s} {cond:16s} {c:26s} {e[0]:.4f} [{e[1]:.4f},{e[2]:.4f}]  {e[4]}/{e[3]}")
    np.savez_compressed(os.path.join(ROOT, "data/usage_sites.npz"), gene=gene, chrom=ch, center=cen, strand=st, cls=cls, analysis=inana, atlas_dist=dist, atlas_libraries=nlib,
                        files=np.array([os.path.basename(f) for f, _, _ in jobs]), signal=np.stack([S[f] for f, _, _ in jobs]))
    log("[out] data/usage_atlas.tsv data/usage_sites.npz")
