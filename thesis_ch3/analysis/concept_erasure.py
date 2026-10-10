#!/usr/bin/env python3
# which sequence properties does the network represent, and which does it rely on?
import sys, os, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def log(*a): print(*a, file=sys.stderr, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
dev = "cuda" if torch.cuda.is_available() else "cpu"; log("device", dev)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); X = D["X"]; y = D["y"].astype(int); sp = D["split"]; gene = D["gene"]
ch = D["chrom"].astype(str); cen = D["center"].astype(np.int64); st = D["strand"].astype(str); N = len(y)
P = np.load(os.path.join(ROOT, "data/splice_positional.npz"))
z0 = 2000; frac = lambda b, a, c: (X[:, z0 + a:z0 + c] == b).mean(1)
def count(motif, a, c):
    m = np.ones((N, c - a - len(motif) + 1), bool)
    for j, b in enumerate(motif): m &= X[:, z0 + a + j:z0 + c - len(motif) + 1 + j] == "ACGT".index(b)
    return m.sum(1).astype(float)
C = collections.OrderedDict()
C["U fraction, -300 to 0"] = frac(3, -300, 0); C["C fraction, -300 to 0"] = frac(1, -300, 0); C["A fraction, -300 to 0"] = frac(0, -300, 0); C["G fraction, -300 to 0"] = frac(2, -300, 0)
C["UGUA count, -300 to 0"] = count("TGTA", -300, 0)
C["poly(A) signal, -35 to -10"] = ((count("AATAAA", -35, -4) + count("ATTAAA", -35, -4)) > 0).astype(float)
C["GU dinucleotides, +1 to +100"] = count("GT", 1, 101)
C["U1 sites per kb, -500 to 0"] = P["u1_4"][:, 15:20].mean(1).astype(float); C["strong 5' splice sites, -500 to 0"] = P["d5_8"][:, 15:20].mean(1).astype(float)
C["best 3' splice site, -500 to 0"] = P["a3_max"][:, 15:20].max(1).astype(float); C["best 3' splice site, -2,000 to -126"] = P["acc"].astype(float)
GTF = os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf"); tx = collections.defaultdict(list)
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] == "exon" and 'transcript_type "protein_coding"' in f[8]: tx[f[8].split('transcript_id "')[1].split('"')[0]].append((f[0], int(f[3]) - 1, int(f[4]), f[6]))
S5 = collections.defaultdict(set); S3 = collections.defaultdict(set)
for ex in tx.values():
    if len(ex) < 2: continue
    s_ = ex[0][3]; ex = sorted(ex, key=lambda e: e[1], reverse=(s_ == "-"))
    for k, (c, a, b, _) in enumerate(ex):
        if k > 0: S3[(c, s_)].add(a if s_ == "+" else b)
        if k < len(ex) - 1: S5[(c, s_)].add(b if s_ == "+" else a)
def near(Sd, lo=60, hi=1000):
    A = {k: np.array(sorted(v)) for k, v in Sd.items()}; out = np.zeros(N)
    for i in range(N):
        a = A.get((ch[i], st[i]))
        if a is None: continue
        l, h = (cen[i] - hi, cen[i] - lo) if st[i] == "+" else (cen[i] + lo, cen[i] + hi)
        out[i] = float(np.searchsorted(a, h, side="right") > np.searchsorted(a, l))
    return out
C["annotated 5' splice site within 1 kb upstream"] = near(S5); C["annotated 3' splice site within 1 kb upstream"] = near(S3)
log({k: round(float(v.mean()), 3) for k, v in C.items()})
WITHIN = len(sys.argv) > 1 and sys.argv[1] == "within"
if WITHIN:
    trm = (sp == "train") | (sp == "val")
    for k in C:
        v = C[k].astype(float).copy()
        for c_ in (0, 1): v[y == c_] -= v[trm & (y == c_)].mean()
        C[k] = v
TBL = torch.zeros(256, 4); TBL[255] = 0.25
for k in range(4): TBL[k, k] = 1.0
TBL = TBL.to(dev)
class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False); s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        s.do = nn.Dropout(0.4); s.fc1 = nn.Linear(96 * 25, 64); s.fc2 = nn.Linear(64, 1)
    def conv(s, x):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), 10)
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5)
        return F.max_pool1d(F.relu(s.b3(s.c3(h))), 2).flatten(1)
tr = np.flatnonzero((sp == "train") | (sp == "val")); te = np.flatnonzero(sp == "test")
gl = collections.defaultdict(list)
for i in te: gl[gene[i]].append(i)
GL = [np.array(v) for v in gl.values() if (y[v] == 1).sum() == 1 and len(v) > 1]
def perf(s):
    w = []; p = []
    for ii in GL:
        t = s[ii][y[ii] == 1][0]; o = s[ii][y[ii] == 0]; w.append((t > o).mean() + 0.5 * (t == o).mean()); p.append(float((o < t).all()))
    return float(np.mean(p)), float(np.mean(w))
def eraser(H, z):
    mu = H.mean(0); Hc = H - mu; S = Hc.T @ Hc / len(H) + 1e-4 * torch.eye(H.shape[1], device=H.device, dtype=H.dtype)
    ev, U = torch.linalg.eigh(S); ev = ev.clamp_min(1e-8); Wm = U @ torch.diag(ev.rsqrt()) @ U.T; Wp = U @ torch.diag(ev.sqrt()) @ U.T
    c = Hc.T @ (z - z.mean()) / len(H); d = Wm @ c; d = d / d.norm().clamp_min(1e-12)
    return lambda h: h - ((h - mu) @ Wm @ d)[:, None] * (Wp @ d)[None, :]
def r2(H_tr, z_tr, H_te, z_te, lam=10.0):
    mu = H_tr.mean(0); sd = H_tr.std(0) + 1e-6; A = (H_tr - mu) / sd; B = (H_te - mu) / sd; zm = z_tr.mean()
    w = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=A.device, dtype=A.dtype), A.T @ (z_tr - zm)); p = B @ w + zm
    return float(1 - ((z_te - p) ** 2).sum() / ((z_te - z_te.mean()) ** 2).sum())
nets = []
for seed in (42, 43, 44, 45, 46):
    n = V().to(dev); n.load_state_dict(torch.load(os.path.join(PUB, f"acc_arch_base_s{seed}.pt"), map_location=dev)["state"]); n.eval(); nets.append(n)
@torch.no_grad()
def acts(n, idx, bs=2048):
    out = []
    for k in range(0, len(idx), bs):
        xb = TBL[torch.from_numpy(X[idx[k:k + bs]].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous(); out.append(n.conv(xb).double())
    return torch.cat(out)
comp_te = torch.from_numpy(y[te] == 0).to(dev); rng = np.random.default_rng(0); NPERM = 20
RES = collections.defaultdict(lambda: collections.defaultdict(list)); BASE = []; LOG = {}
with torch.no_grad():
    for ni, n in enumerate(nets):
        A_tr = acts(n, tr); A_te = acts(n, te); W1, b1, W2, b2 = n.fc1.weight.double(), n.fc1.bias.double(), n.fc2.weight.double(), n.fc2.bias.double()
        head = {"convolutional output (2,400)": lambda a: (F.relu(a @ W1.T + b1) @ W2.T + b2).squeeze(1), "dense layer (64)": lambda h: (h @ W2.T + b2).squeeze(1)}
        rep = {"convolutional output (2,400)": (A_tr, A_te), "dense layer (64)": (F.relu(A_tr @ W1.T + b1), F.relu(A_te @ W1.T + b1))}
        s0 = np.zeros(N); s0[te] = head["dense layer (64)"](rep["dense layer (64)"][1]).cpu().numpy(); BASE.append(s0.copy())
        for layer, (Htr, Hte) in rep.items():
            for cname, z in C.items():
                ztr = torch.from_numpy(z[tr]).to(dev).double(); zte = torch.from_numpy(z[te]).to(dev).double()
                RES[(layer, cname)]["r2"].append(r2(Htr, ztr, Hte, zte)); RES[(layer, cname)]["r2_comp"].append(r2(Htr, ztr, Hte[comp_te], zte[comp_te]))
                s = np.zeros(N); s[te] = head[layer](eraser(Htr, ztr)(Hte)).cpu().numpy(); RES[(layer, cname)]["erased"].append(s)
                ctl = []
                for _ in range(NPERM if layer.startswith("dense") else 5):
                    zp = ztr[torch.from_numpy(rng.permutation(len(tr))).to(dev)]; s = np.zeros(N); s[te] = head[layer](eraser(Htr, zp)(Hte)).cpu().numpy(); ctl.append(s)
                RES[(layer, cname)]["control"].append(ctl)
        log(f"network {ni + 1}/5 done")
base = np.mean(BASE, 0); p0, w0 = perf(base); log(f"intact: P@1 {p0:.3f} win-rate {w0:.3f}")
RB = np.random.default_rng(1)
def pg(s): return np.array([float((s[ii][y[ii] == 0] < s[ii][y[ii] == 1][0]).all()) for ii in GL])
pb = pg(base)
with open(os.path.join(ROOT, "data/concept_erasure_within.tsv" if WITHIN else "data/concept_erasure.tsv"), "w") as fh:
    fh.write("layer\tconcept\tr2_heldout\tr2_heldout_competitors\tp_at_1_intact\tp_at_1_erased\tp_at_1_control\tdelta_vs_control\tdelta_lo\tdelta_hi\twin_rate_intact\twin_rate_erased\twin_rate_control\n")
    for (layer, cname), r in RES.items():
        se = np.mean(r["erased"], 0); pe, we = perf(se); nper = len(r["control"][0])
        ctl = [np.mean([r["control"][ni][k] for ni in range(5)], 0) for k in range(nper)]; pc = np.mean([pg(s) for s in ctl], 0); wc = np.mean([perf(s)[1] for s in ctl])
        d = pg(se) - pc; bt = [d[RB.integers(0, len(d), len(d))].mean() for _ in range(2000)]; lo, hi = np.percentile(bt, [2.5, 97.5])
        fh.write(f"{layer}\t{cname}\t{np.mean(r['r2']):.3f}\t{np.mean(r['r2_comp']):.3f}\t{p0:.3f}\t{pe:.3f}\t{pc.mean():.3f}\t{d.mean():+.4f}\t{lo:+.4f}\t{hi:+.4f}\t{w0:.3f}\t{we:.3f}\t{wc:.3f}\n")
        log(f"{layer:30s} {cname:42s} R2 {np.mean(r['r2']):.2f} (comp {np.mean(r['r2_comp']):.2f})  P@1 erased {pe:.3f} control {pc.mean():.3f}  delta {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
log("[out] data/concept_erasure.tsv")
