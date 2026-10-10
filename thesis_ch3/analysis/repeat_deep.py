#!/usr/bin/env python3
# why do so many competitor sites lie in repeat elements
import os, sys, math, collections, numpy as np, RNA, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
sys.path.insert(0, "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/17.0.0/lib/python3.11/site-packages")
import pyarrow.parquet as pq
def log(*a): print(*a, file=sys.stderr, flush=True)
src = open(os.path.join(HERE, "stats_ch3.py")).read(); cut = src.index("# ---------------------------------------------------------------- data")
ns = {"__file__": os.path.join(HERE, "stats_ch3.py")}; exec(src[:cut], ns); welch = ns["welch"]
RNG = np.random.default_rng(0); OUT = []
def out(*a): OUT.append(a); log("  ".join(str(x) for x in a))
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); X = D["X"]; y = D["y"].astype(int); g = D["gene"]
RS = list(__import__("csv").DictReader(open(os.path.join(ROOT, "data/repeat_sites.tsv")), delimiter="\t"))
sg = np.array([r["site_group"] for r in RS]); so = np.array([int(r["site_orientation"]) for r in RS])
grp = np.full(len(y), "", dtype=object); grp[y == 1] = "real 3' end"
c = y == 0
grp[c & (sg == "none")] = "competitor, no repeat"
grp[c & (sg == "L1") & (so == -1)] = "competitor, L1 antisense"; grp[c & (sg == "L1") & (so == 1)] = "competitor, L1 sense"
grp[c & (sg == "LTR")] = "competitor, LTR"; grp[c & (sg == "Alu")] = "competitor, Alu"; grp[c & (sg == "DNA transposon")] = "competitor, DNA transposon"
grp[c & (grp == "")] = "competitor, other repeat"
GROUPS = ["real 3' end", "competitor, no repeat", "competitor, L1 antisense", "competitor, L1 sense", "competitor, LTR",
          "competitor, Alu", "competitor, DNA transposon", "competitor, other repeat"]
for G in GROUPS: out("n", G, int((grp == G).sum()))
NT = np.array(list("ACGTN")); S = lambda i, a, b: "".join(NT[np.minimum(X[i][2000 + a:2000 + b], 4)])
PAS12 = ["AATAAA", "ATTAAA", "AGTAAA", "TATAAA", "CATAAA", "GATAAA", "AATATA", "AATACA", "AATAGA", "AAAAAG", "ACTAAA", "AATGAA"]
# 1 cleavage elements
prof = {G: np.zeros(81) for G in GROUPS}
for G in GROUPS:
    idx = np.flatnonzero(grp == G); f = collections.defaultdict(list)
    for i in idx:
        s = S(i, -100, 50)
        for k in range(40, 121):
            h = s[k:k + 6]
            if h in ("AATAAA", "ATTAAA"): prof[G][k - 40] += 1
        pas = s[65:90]
        f["canonical PAS (AATAAA or ATTAAA) in -35 to -10"].append(any(h in pas for h in ("AATAAA", "ATTAAA")))
        f["AATAAA in -35 to -10"].append("AATAAA" in pas)
        f["any of 12 PAS variants in -35 to -10"].append(any(h in pas for h in PAS12))
        f["TGTA in -100 to -20"].append(s[0:80].count("TGTA"))
        dn = s[101:141]; f["T fraction, +1 to +40"].append(dn.count("T") / 40); f["GT dinucleotides, +1 to +40"].append(dn.count("GT"))
        f["A fraction, -30 to +30"].append(s[70:130].count("A") / 60); f["run of >= 8 A within -30 to +30"].append("AAAAAAAA" in s[70:130])
    for k, v in f.items(): out("elements", G, k, f"{np.mean(v):.3f}")
    prof[G] /= max(1, len(idx))
with open(os.path.join(ROOT, "data/repeat_deep_pas_profile.tsv"), "w") as fh:
    fh.write("offset\t" + "\t".join(GROUPS) + "\n")
    for k in range(81): fh.write(f"{k - 60}\t" + "\t".join(f"{prof[G][k]:.4f}" for G in GROUPS) + "\n")
# 2 where in the element (l1, ltr)
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
want = {(str(ch[i]), int(cen[i])) for i in np.flatnonzero(np.isin(grp, ["competitor, L1 antisense", "competitor, L1 sense", "competitor, LTR", "competitor, Alu"]))}
bych = collections.defaultdict(list)
for k in want: bych[k[0]].append(k[1])
bych = {k: np.array(sorted(v)) for k, v in bych.items()}
hit = {}
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        P = bych.get(f[5])
        if P is None: continue
        s_, e_ = int(f[6]), int(f[7]); a = np.searchsorted(P, s_); b = np.searchsorted(P, e_)
        if a == b: continue
        strand = f[9]; cs, ce = (int(f[13]), int(f[14])) if strand == "+" else (int(f[15]), int(f[14]))
        cons_len = ce + (-int(f[15]) if strand == "+" else -int(f[13]))
        for p in P[a:b]:
            frac = (p - s_) / max(1, e_ - s_)
            pos = cs + frac * (ce - cs) if strand == "+" else ce - frac * (ce - cs)
            hit[(f[5], int(p))] = (f[10], f[12], pos, cons_len)
for G in ("competitor, L1 antisense", "competitor, L1 sense", "competitor, LTR", "competitor, Alu"):
    idx = np.flatnonzero(grp == G); rec = [hit.get((str(ch[i]), int(cen[i]))) for i in idx]; rec = [r for r in rec if r]
    names = collections.Counter(r[0] for r in rec).most_common(6)
    rel = np.array([r[2] / max(1, r[3]) for r in rec])
    h, e = np.histogram(rel, bins=np.linspace(0, 1, 11))
    out("element", G, f"n {len(rec)}", "top subfamilies " + ", ".join(f"{n} {k}" for n, k in names),
        "relative consensus position (deciles 5'->3') " + " ".join(str(int(v)) for v in h))
# 3 secondary structure
sys.path.insert(0, PUB)
from cnn_interp_lib import dinuc_shuffle as _dsh
_TOK = {c: i for i, c in enumerate("ACGT")}
def dinuc_shuffle(s, rng):
    return "".join("ACGT"[k] for k in _dsh(np.array([_TOK[c] for c in s], np.int64), rng))
def struct(seq, pas_at):
    fc = RNA.fold_compound(seq.replace("T", "U")); _, mfe = fc.mfe(); fc.exp_params_rescale(mfe); fc.pf()
    bpp = np.array(fc.bpp()); pp = bpp + bpp.T; unp = 1 - pp.sum(1)[1:]
    return mfe / len(seq), (unp[pas_at:pas_at + 6].mean() if pas_at is not None else np.nan)
SMF = {}
for G in GROUPS:
    idx = np.flatnonzero(grp == G); idx = RNG.choice(idx, min(1500, len(idx)), replace=False)
    m, u, ms = [], [], []
    for i in idx:
        s = S(i, -100, 100)
        if "N" in s: continue
        pas = s[65:90]; at = None
        for h in ("AATAAA", "ATTAAA"):
            k = pas.find(h)
            if k >= 0: at = 65 + k; break
        a, b = struct(s, at); m.append(a); u.append(b); ms.append(struct(dinuc_shuffle(s, RNG), None)[0])
    SMF[G] = (np.array(m), np.array(u), np.array(ms))
    out("structure", G, f"n {len(m)}", f"MFE per nt {np.mean(m):.4f}", f"shuffled {np.mean(ms):.4f}", f"real minus shuffled {np.mean(np.array(m)-np.array(ms)):+.4f}",
        f"PAS unpaired probability {np.nanmean(u):.3f} (n with PAS {np.sum(~np.isnan(u))})")
for G in GROUPS[1:]:
    d, t, df, p = welch(SMF[G][0], SMF["competitor, no repeat"][0]) if G != "competitor, no repeat" else welch(SMF[G][0], SMF["real 3' end"][0])
    ref = "no-repeat competitors" if G != "competitor, no repeat" else "real 3' ends"
    out("structure_test", G, f"MFE per nt minus {ref} {d:+.4f}", f"Welch t = {t:.2f}", f"P = {p:.2g}")
# 4 usage over the whole shortlist
W = np.load(os.path.join(PUB, "cnn_wg_scores.npz"), allow_pickle=True)
T = pq.read_table(os.path.join(PUB, "cpa_peaks_final.parquet")).to_pydict()
key = lambda gg, cc, x, s: (str(gg), str(cc), int(x), str(s))
row = {key(*t): i for i, t in enumerate(zip(T["gene"], T["chrom"], T["center"], T["strand"]))}
ix = np.array([row[key(*t)] for t in zip(W["gene"], W["chrom"], W["center"], W["strand"])])
apa = np.array(T["flag_apa_annotated"])[ix]; dn = np.array(T["flag_downstream_of_canonical"])[ix]; can = np.array(T["is_canonical"])[ix]
sel = (~can) & (~dn)
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa"))
inrep = np.array([fa.fetch(str(W["chrom"][i]), int(W["center"][i]) - 10, int(W["center"][i]) + 10).islower() if sel[i] else False for i in range(len(sel))])
for lab, m_ in (("site in repeat (all 20 nt soft-masked)", sel & inrep), ("site not in repeat", sel & ~inrep)):
    out("usage", lab, f"competitors {int(m_.sum())}", f"within 500 nt of an annotated poly(A) site {apa[m_].mean():.3f}")
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); L = C["logit"].astype(float)
gi = collections.defaultdict(list)
for i, gg in enumerate(g): gi[gg].append(i)
Lc = np.full(len(L), np.nan)
for ii in gi.values():
    ii = np.array(ii); cc = ii[y[ii] == 0]; Lc[cc] = L[cc] - L[cc].mean()
for G in GROUPS[1:]:
    v = Lc[grp == G]; out("network", G, f"gene-centred competitor logit {np.nanmean(v):+.3f}", f"n {np.sum(~np.isnan(v))}")
pasp = np.array([any(h in S(i, -35, -10) for h in ("AATAAA", "ATTAAA")) for i in range(len(y))])
arun = np.array(["AAAAAAAA" in S(i, -30, 30) for i in range(len(y))])
for G in ("competitor, no repeat", "competitor, L1 antisense", "competitor, LTR", "competitor, Alu"):
    m_ = grp == G
    for lab, f_ in (("canonical PAS present", pasp), ("A-run present", arun)):
        a, b = Lc[m_ & f_], Lc[m_ & ~f_]
        if len(a) > 30 and len(b) > 30:
            d, t, df, p = welch(a, b); out("network_within", G, lab, f"difference {d:+.3f}", f"n {len(a)} vs {len(b)}", f"P = {p:.2g}")
with open(os.path.join(ROOT, "data/repeat_deep.tsv"), "w") as fh:
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/repeat_deep.tsv data/repeat_deep_pas_profile.tsv")
