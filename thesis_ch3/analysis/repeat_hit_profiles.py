#!/usr/bin/env python3
# where along ltr, alu and l1 elements the competitor sites fall
import os, json, collections, numpy as np, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
comp = np.flatnonzero(y == 0); NB = 40
fam = lambda cls, f: "L1" if f == "L1" else "LTR" if cls.startswith("LTR") else "Alu" if f == "Alu" else None
sites = collections.defaultdict(list); win = collections.defaultdict(list)
for i in comp:
    c = int(cen[i]); s = str(st[i]); sites[str(ch[i])].append((c, s))
    win[str(ch[i])].append(((c - 2000, c) if s == "+" else (c + 1, c + 2001)) + (s,))
for k in sites: sites[k].sort()
for k in win: win[k].sort()
SP = {k: np.array([p for p, _ in v]) for k, v in sites.items()}; W0 = {k: np.array([w[0] for w in v]) for k, v in win.items()}
rel = collections.defaultdict(list); bg = collections.defaultdict(list); hits = collections.defaultdict(list)
rng = np.random.default_rng(0)
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t"); F = fam(f[11], f[12])
        if F is None or f[5] not in SP: continue
        c_ = f[5]; s_, e_, strand = int(f[6]), int(f[7]), f[9]
        cs, ce = (int(f[13]), int(f[14])) if strand == "+" else (int(f[15]), int(f[14]))
        L = max(1, ce + (-int(f[15]) if strand == "+" else -int(f[13])))
        def relpos(p):
            frac = (p - s_) / max(1, e_ - s_)
            return (cs + frac * (ce - cs) if strand == "+" else ce - frac * (ce - cs)) / L
        a, b = np.searchsorted(SP[c_], s_), np.searchsorted(SP[c_], e_)
        for k in range(a, b):
            p, gs = sites[c_][k]; o = "sense" if gs == strand else "antisense"; r = relpos(p)
            rel[(F, o)].append(r if o == "sense" else 1 - r); hits[(F, o)].append((c_, p, gs))
        V = win[c_]; j = np.searchsorted(W0[c_], s_ - 2001)
        while j < len(V) and V[j][0] < e_:
            wa, wb, gs = V[j]; lo, hi = max(wa, s_), min(wb, e_)
            if hi > lo:
                o = "sense" if gs == strand else "antisense"; r = relpos(int(rng.integers(lo, hi)))
                bg[(F, o)].append(r if o == "sense" else 1 - r)
            j += 1
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa"))
comp_tab = str.maketrans("ACGTN", "TGCAN")
def rna(c_, p, gs):
    s = fa.fetch(c_, p - 50, p + 21).upper() if gs == "+" else fa.fetch(c_, p - 20, p + 51).upper().translate(comp_tab)[::-1]
    return s.replace("T", "U")
def cons(name, anti):
    s = [r for r in json.load(open(os.path.join(ROOT, f"data/dfam/{name}.json")))["results"] if r["name"] == name][0]["consensus_sequence"].upper()
    if anti: s = s.translate(comp_tab)[::-1]
    return s.replace("T", "U")
OUT = {}
for F, o, cname in (("LTR", "sense", "MLT1C"), ("Alu", "sense", "AluSx"), ("Alu", "antisense", "AluSx"), ("L1", "antisense", None), ("L1", "sense", None), ("LTR", "antisense", "MLT1C")):
    h, _ = np.histogram(rel[(F, o)], bins=np.linspace(0, 1, NB + 1)); b_, _ = np.histogram(bg[(F, o)], bins=np.linspace(0, 1, NB + 1))
    cnt = np.zeros((71, 4))
    for c_, p, gs in hits[(F, o)]:
        for i, x in enumerate(rna(c_, p, gs)):
            if x in "ACGU": cnt[i, "ACGU".index(x)] += 1
    OUT[f"{F}_{o}"] = {"n": int(h.sum()), "hist": (h / h.sum()).round(4).tolist(), "bg": (b_ / b_.sum()).round(4).tolist(),
                       "logo": (cnt / cnt.sum(1, keepdims=True)).round(3).tolist(), "consensus": cons(cname, o == "antisense") if cname else None, "consensus_name": cname}
    print(F, o, h.sum(), "peak bin", int(h.argmax()), "U at +1..+20", round(float(OUT[f"{F}_{o}"]["logo"][51][3]), 3))
json.dump(OUT, open(os.path.join(ROOT, "data/repeat_hit_profiles.json"), "w"))
print("[out] data/repeat_hit_profiles.json")
