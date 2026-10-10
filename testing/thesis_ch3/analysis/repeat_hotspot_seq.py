#!/usr/bin/env python3
# the rna around competitor sites in each repeat hotspot, and where the poly(a) signal sits
import os, json, collections, numpy as np, pysam
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
sites = collections.defaultdict(list)
for i in np.flatnonzero(y == 0): sites[str(ch[i])].append((int(cen[i]), str(st[i]), int(i)))
for k in sites: sites[k].sort()
SP = {k: np.array([p for p, _, _ in v]) for k, v in sites.items()}
fam = lambda cls, f: "L1" if f == "L1" else "LTR" if cls.startswith("LTR") else "Alu" if f == "Alu" else None
HOT = {
    "LTR_sense_end":      ("LTR", "sense", "rel", 33 / 40, 38 / 40),
    "Alu_sense_start":    ("Alu", "sense", "rel", 0, 2 / 40),
    "Alu_sense_middle":   ("Alu", "sense", "rel", 17 / 40, 19 / 40),
    "Alu_antisense_Urun": ("Alu", "antisense", "rel", 0, 5 / 40),
    "Alu_antisense_middle": ("Alu", "antisense", "rel", 25 / 40, 28 / 40),
    "L1_antisense_400":   ("L1", "antisense", "nt", 400, 500),
}
members = collections.defaultdict(list); in_rep = set()
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[5] not in SP: continue
        c_ = f[5]; s_, e_, strand = int(f[6]), int(f[7]), f[9]
        a, b = np.searchsorted(SP[c_], s_), np.searchsorted(SP[c_], e_)
        if a == b: continue
        for k in range(a, b): in_rep.add(sites[c_][k][2])
        F = fam(f[11], f[12])
        if F is None: continue
        cs, ce = (int(f[13]), int(f[14])) if strand == "+" else (int(f[15]), int(f[14]))
        L = max(1, ce + (-int(f[15]) if strand == "+" else -int(f[13])))
        for k in range(a, b):
            p, gs, idx = sites[c_][k]; o = "sense" if gs == strand else "antisense"
            frac = (p - s_) / max(1, e_ - s_); pos = cs + frac * (ce - cs) if strand == "+" else ce - frac * (ce - cs)
            rel = pos / L if o == "sense" else 1 - pos / L
            for name, (F_, o_, ax, lo, hi) in HOT.items():
                if F_ == F and o_ == o and lo <= (rel if ax == "rel" else L - pos) < hi: members[name].append((c_, p, gs))
members["competitor_no_repeat"] = [(str(ch[i]), int(cen[i]), str(st[i])) for i in np.flatnonzero(y == 0) if int(i) not in in_rep]
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa")); tr = str.maketrans("ACGTN", "TGCAN")
UP, DN = 45, 15
def rna(c_, p, gs):
    s = fa.fetch(c_, p - UP, p + DN + 1).upper() if gs == "+" else fa.fetch(c_, p - DN, p + UP + 1).upper().translate(tr)[::-1]
    return s.replace("T", "U")
PAS12 = ["AAUAAA", "AUUAAA", "AGUAAA", "UAUAAA", "CAUAAA", "GAUAAA", "AAUAUA", "AAUACA", "AAUAGA", "AAAAAG", "ACUAAA", "AAUGAA"]
OUT = {}
for name, M in members.items():
    N = UP + DN + 1; cnt = np.zeros((N, 4)); pas = np.zeros(N - 5); anyc = 0; any12 = 0; hexc = collections.Counter(); n = 0
    for c_, p, gs in M:
        s = rna(c_, p, gs)
        if len(s) != N: continue
        n += 1
        for i, x in enumerate(s):
            if x in "ACGU": cnt[i, "ACGU".index(x)] += 1
        for k in range(N - 5):
            if s[k:k + 6] in ("AAUAAA", "AUUAAA"): pas[k] += 1
        w = s[UP - 35:UP - 10 + 6]
        anyc += any(h in w for h in ("AAUAAA", "AUUAAA")); any12 += any(h in w for h in PAS12)
        for h in PAS12:
            if h in w: hexc[h] += 1
    fr = cnt / cnt.sum(1, keepdims=True)
    OUT[name] = {"n": n, "offset0": -UP, "top": "".join("ACGU"[j] for j in fr.argmax(1)), "topfreq": fr.max(1).round(3).tolist(),
                 "freq": fr.round(3).tolist(), "pas_pos": (pas / n).round(4).tolist(), "pas_canonical": round(anyc / n, 3),
                 "pas_any12": round(any12 / n, 3), "top_hex": [(h, round(c / n, 3)) for h, c in hexc.most_common(4)]}
    k = int(np.argmax(pas)); print(f"{name:22s} n {n:6d}  canonical PAS {anyc/n:.2f}  any12 {any12/n:.2f}  peak PAS start {k-UP:+d} ({pas[k]/n:.2f})  {OUT[name]['top_hex']}\n   {OUT[name]['top'][:UP]}|{OUT[name]['top'][UP:]}")
json.dump(OUT, open(os.path.join(ROOT, "data/repeat_hotspot_seq.json"), "w"))
print("[out] data/repeat_hotspot_seq.json")
