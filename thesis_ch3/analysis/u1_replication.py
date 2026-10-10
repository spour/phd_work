#!/usr/bin/env python3
# use of real 3' ends, competitors and matched intronic decoys after inhibition of u1
import os, sys, glob, numpy as np, pyBigWig
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, ".."); G = os.path.join(PUB, "external/geo")
def log(*a): print(*a, file=sys.stderr, flush=True)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); Z = np.load(os.path.join(ROOT, "data/occupancy_profiles_decoy.npz"), allow_pickle=True)
gene = np.concatenate([D["gene"], Z["gene"]]); ch = np.concatenate([D["chrom"], Z["chrom"]]).astype(str); cen = np.concatenate([D["center"].astype(np.int64), Z["center"].astype(np.int64)])
st = np.concatenate([D["strand"], Z["strand"]]).astype(str)
cls = np.concatenate([np.where(D["y"].astype(int) == 1, "real 3' end", "competitor"), np.full(len(Z["gene"]), "decoy")]); N = len(cen)
h19c = np.array([""] * N, dtype=object); h19p = np.full(N, -1)
for l in open(os.path.join(ROOT, "data/_lift/out.bed")):
    f = l.split("\t"); h19c[int(f[3])] = f[0]; h19p[int(f[3])] = int(f[1])
F = lambda pat: sorted(glob.glob(os.path.join(G, pat)))
EXP = [
    ("GSE250342", "HeLa", "U1 AMO", "hg38", F("GSE250342/*_Control-[ab]_1*"), F("GSE250342/*_U1-[ab]_1*")),
    ("GSE250342", "HeLa", "U6 AMO", "hg38", F("GSE250342/*_Control-[ab]_1*"), F("GSE250342/*_U6-[ab]_1*")),
    ("GSE250342", "HeLa", "U2 AMO", "hg38", F("GSE250342/*_Control-[ab]_1*"), F("GSE250342/*HeLa-U2-*")),
    ("GSE250342", "HEK293", "U1 AMO", "hg38", F("GSE250342/*_293-ctr-*"), F("GSE250342/*_293-U1-*")),
    ("GSE250342", "HEK293", "U6 AMO", "hg38", F("GSE250342/*_293-ctr-*"), F("GSE250342/*_293-U6-*")),
    ("GSE250342", "HEK293", "U2 AMO", "hg38", F("GSE250342/*_293-ctr-*"), F("GSE250342/*_293-U2-*")),
    ("GSE250342", "SW480", "U1 AMO", "hg38", F("GSE250342/*SW480-NC-*"), F("GSE250342/*SW480-U1-*")),
    ("GSE250342", "SW480", "U6 AMO", "hg38", F("GSE250342/*SW480-NC-*"), F("GSE250342/*SW480-U6-*")),
    ("GSE262383", "HeLa", "U1 AMO", "hg38", F("GSE262383/*HeLa-ctr-*"), F("GSE262383/*HeLa-U1-*")),
    ("GSE262383", "HeLa", "U4 AMO", "hg38", F("GSE262383/*HeLa-ctr-*"), F("GSE262383/*HeLa-U4-*")),
    ("GSE275154", "HEK293T", "U1 AMO", "hg19", F("GSE275154/*_c25_*"), F("GSE275154/*_u1_*")),
    ("GSE275154", "HEK293T", "U2 AMO", "hg19", F("GSE275154/*_c50_*"), F("GSE275154/*_u2_*")),
    ("GSE275154", "HEK293T", "pladienolide B", "hg19", F("GSE275154/*_dmso_*"), F("GSE275154/*_pb_*"))]
for e in EXP: assert len(e[4]) >= 2 and len(e[5]) >= 2, e
files = sorted({(f, e[3]) for e in EXP for f in e[4] + e[5]})
def work(job):
    f, build = job; bw = pyBigWig.open(f); sz = bw.chroms(); pre = "" if "chr1" in sz else None; v = np.full(N, np.nan, np.float32)
    for i in range(N):
        c, p = (ch[i], int(cen[i])) if build == "hg38" else (h19c[i], int(h19p[i]))
        if p < 0 or not c: continue
        if pre is None: c = c[3:]
        if c not in sz: continue
        a, b = (p - 100, p + 21) if st[i] == "+" else (p - 20, p + 101)
        if a < 0 or b > sz[c]: continue
        v[i] = np.nansum(np.nan_to_num(np.asarray(bw.values(c, a, b), np.float32)))
    return f, v
if __name__ == "__main__":
    with Pool(6) as pool: S = dict(pool.map(work, files))
    genes = sorted(set(D["gene"])); gix = {g: k for k, g in enumerate(genes)}; NG = len(genes); kg = np.array([gix[g] for g in gene])
    RNG = np.random.default_rng(0)
    def boot(fn):
        k = np.arange(NG); e = fn(k); b = np.array([fn(RNG.integers(0, NG, NG)) for _ in range(2000)]); b = b[np.isfinite(b)]
        lo, hi = np.percentile(b, [2.5, 97.5]); return e, lo, hi, max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1 / 2000)
    fr = open(os.path.join(ROOT, "data/u1_replication_fraction.tsv"), "w"); fr.write("dataset\tcell_line\ttreatment\tcondition\tclass\tsites\tsites_with_reads\tfraction\tlo\thi\n")
    sto = open(os.path.join(ROOT, "data/u1_replication_stats.tsv"), "w"); sto.write("dataset\tcell_line\ttreatment\tquantity\testimate\tlo\thi\tP\n")
    for ds, cell, trt, build, fc, ft in EXP:
        okk = np.all([np.isfinite(S[f]) for f in fc + ft], 0)
        hit = {"control": np.any([S[f] > 0 for f in fc], 0).astype(float), trt: np.any([S[f] > 0 for f in ft], 0).astype(float)}
        A = {}
        for c in ("real 3' end", "competitor", "decoy"):
            m = okk & (cls == c)
            for cond in ("control", trt):
                s_ = np.zeros(NG); n_ = np.zeros(NG); np.add.at(s_, kg[m], hit[cond][m]); np.add.at(n_, kg[m], 1); A[(c, cond)] = (s_, n_)
                e = boot(lambda k: s_[k].sum() / n_[k].sum())
                fr.write(f"{ds}\t{cell}\t{trt}\t{'control' if cond == 'control' else 'treated'}\t{c}\t{int(m.sum())}\t{int(hit[cond][m].sum())}\t{e[0]:.5f}\t{e[1]:.5f}\t{e[2]:.5f}\n")
        frac = lambda c, cond, k: A[(c, cond)][0][k].sum() / A[(c, cond)][1][k].sum()
        lfc = lambda c, k: np.log2(frac(c, trt, k) / frac(c, "control", k))
        Q = [("competitor over decoy, control (log2)", lambda k: np.log2(frac("competitor", "control", k) / frac("decoy", "control", k))),
             ("competitor over decoy, treated (log2)", lambda k: np.log2(frac("competitor", trt, k) / frac("decoy", trt, k))),
             ("log2 fold change, real 3' end", lambda k: lfc("real 3' end", k)), ("log2 fold change, competitor", lambda k: lfc("competitor", k)),
             ("log2 fold change, decoy", lambda k: lfc("decoy", k)), ("log2 fold change, competitor minus decoy", lambda k: lfc("competitor", k) - lfc("decoy", k))]
        for name, fn in Q:
            e = boot(fn); sto.write(f"{ds}\t{cell}\t{trt}\t{name}\t{e[0]:+.3f}\t{e[1]:+.3f}\t{e[2]:+.3f}\t{e[3]:.3g}\n")
            log(f"{ds} {cell:8s} {trt:15s} {name:45s} {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] P {e[3]:.3g}")
    fr.close(); sto.close()
    np.savez_compressed(os.path.join(ROOT, "data/u1_replication_per_site.npz"), files=np.array([os.path.basename(f) for f, _ in files]), signal=np.stack([S[f] for f, _ in files]), cls=cls, gene=gene)
    log("[out] data/u1_replication_fraction.tsv data/u1_replication_stats.tsv")
