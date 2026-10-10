#!/usr/bin/env python3
# one row per repeat family x orientation for every analysis-set competitor whose site lies
import os, json, collections, numpy as np, pysam
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); assert (C["gene"] == g).all()
L = C["logit"].astype(float); Lc = L.copy()
gi = collections.defaultdict(list)
for i, gg in enumerate(g): gi[gg].append(i)
for ii in gi.values(): ii = np.array(ii); Lc[ii] -= L[ii][y[ii] == 0].mean() if (y[ii] == 0).any() else 0
GROUP = lambda cls, fam: ("Alu" if fam == "Alu" else "MIR and other SINE" if cls == "SINE" else "L1" if fam == "L1" else
                          "L2 and other LINE" if cls == "LINE" else "LTR" if cls.startswith("LTR") else "DNA transposon" if cls.startswith("DNA")
                          else "simple or low complexity" if cls in ("Simple_repeat", "Low_complexity") else "other")
N = len(y); NB = 20
pts = collections.defaultdict(list); win = collections.defaultdict(list)
for i in range(N):
    c = int(cen[i]); s = str(st[i]); pts[str(ch[i])].append((c, s, i))
    if y[i] == 0: win[str(ch[i])].append(((c - 2000, c) if s == "+" else (c + 1, c + 2001)) + (s,))
for k in pts: pts[k].sort()
for k in win: win[k].sort()
PP = {k: np.array([p for p, _, _ in v]) for k, v in pts.items()}; W0 = {k: np.array([w[0] for w in v]) for k, v in win.items()}
best = {}
bg = collections.defaultdict(list); rng = np.random.default_rng(0)
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t"); c_ = f[5]
        if c_ not in PP: continue
        s_, e_, strand = int(f[6]), int(f[7]), f[9]; G = GROUP(f[11], f[12])
        cs, ce = (int(f[13]), int(f[14])) if strand == "+" else (int(f[15]), int(f[14]))
        Lcons = max(1, ce + (-int(f[15]) if strand == "+" else -int(f[13])))
        def rel(p, o):
            frac = (p - s_) / max(1, e_ - s_); r = (cs + frac * (ce - cs) if strand == "+" else ce - frac * (ce - cs)) / Lcons
            return min(max(r if o == "sense" else 1 - r, 0), 0.999999)
        a, b = np.searchsorted(PP[c_], s_), np.searchsorted(PP[c_], e_)
        for k in range(a, b):
            p, gs, idx = pts[c_][k]; o = "sense" if gs == strand else "antisense"
            if idx not in best or s_ < best[idx][0]: best[idx] = (s_, G, o, rel(p, o), f[10])
        V = win[c_]; j = np.searchsorted(W0[c_], s_ - 2001)
        while j < len(V) and V[j][0] < e_:
            wa, wb, gs = V[j]; lo, hi = max(wa, s_), min(wb, e_)
            if hi > lo:
                o = "sense" if gs == strand else "antisense"; bg[(G, o)].append(rel(int(rng.integers(lo, hi)), o))
            j += 1
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa")); tr = str.maketrans("ACGTN", "TGCAN")
def rna(i):
    p = int(cen[i]); c_ = str(ch[i])
    s = fa.fetch(c_, p - 45, p + 21).upper() if st[i] == "+" else fa.fetch(c_, p - 20, p + 46).upper().translate(tr)[::-1]
    return s.replace("T", "U")
def summarise(idx):
    pas = u = 0
    for i in idx:
        s = rna(i); w = s[10:41]
        pas += ("AAUAAA" in w) or ("AUUAAA" in w); u += s[46:66].count("U") / 20
    return pas / len(idx), u / len(idx), float(Lc[idx].mean())
comp = np.flatnonzero(y == 0); real = np.flatnonzero(y == 1); ncomp = len(comp); nreal = len(real)
rows = []
keys = collections.defaultdict(list)
for i, v in best.items(): keys[(v[1], v[2])].append(i)
for (G, o), idx in keys.items():
    ci = np.array([i for i in idx if y[i] == 0]); ri = [i for i in idx if y[i] == 1]
    if len(ci) == 0: continue
    r = np.array([best[i][3] for i in ci])
    h, _ = np.histogram(r, bins=np.linspace(0, 1, NB + 1)); hb, _ = np.histogram(bg[(G, o)], bins=np.linspace(0, 1, NB + 1))
    hf = h / h.sum(); bf = hb / max(1, hb.sum())
    hot = float(hf[(bf > 0) & (hf / np.maximum(bf, 1e-9) >= 2)].sum()) if G not in ("simple or low complexity", "other") else None
    pas, u, net = summarise(ci)
    sub = collections.Counter(best[i][4] for i in ci).most_common(3)
    rows.append({"group": G, "orientation": o, "n": int(len(ci)), "share": len(ci) / ncomp, "real_share": len(ri) / nreal,
                 "hist": hf.round(4).tolist(), "bg": bf.round(4).tolist(), "hotspot_share": hot, "pas": round(pas, 3),
                 "u_after": round(u, 3), "network": round(net, 3), "top_subfamilies": sub})
    print(f"{G:26s} {o:9s} n {len(ci):5d}  share {len(ci)/ncomp:.3f}  real {len(ri)/nreal:.4f}  hotspot {hot if hot is None else round(hot,2)}  PAS {pas:.2f}  U {u:.2f}  net {net:+.2f}  {sub}")
nonrep = np.array([i for i in comp if i not in best]); pas, u, net = summarise(nonrep)
ref = {"competitor_no_repeat": {"n": int(len(nonrep)), "share": len(nonrep) / ncomp, "pas": round(pas, 3), "u_after": round(u, 3), "network": round(net, 3)}}
pas, u, net = summarise(real); ref["real_end"] = {"n": int(nreal), "pas": round(pas, 3), "u_after": round(u, 3), "network": round(net, 3)}
print("non-repeat", ref["competitor_no_repeat"]); print("real", ref["real_end"])
print("repeat competitors total", sum(r["n"] for r in rows), "of", ncomp)
json.dump({"rows": sorted(rows, key=lambda r: -r["n"]), "ref": ref, "n_comp": int(ncomp)}, open(os.path.join(ROOT, "data/repeat_all_families.json"), "w"))
print("[out] data/repeat_all_families.json")
