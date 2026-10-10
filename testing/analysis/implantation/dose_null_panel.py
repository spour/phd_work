#!/usr/bin/env python3
# the dose series against four negatives, every unit on one protocol
import sys, argparse, itertools
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nshuf", type=int, default=50); P.add_argument("--pos", type=int, default=-230)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; UP = 2000
UNITS = [("model_me3",      "ACGGTGTACTTTTGTTGCAGGTG", "3'ss the model picks, score3 10.04",  7),
         ("robust_agdep",   "CCATGCCTTTGATCCTGCAGATG", "3'ss that needs its AG, score3 10.38", 7),
         ("maxent_optimum", "TTTCCTCTCTCTCCCCGCAGGTG", "3'ss MaxEnt optimum, score3 17.17",   7),
         ("t_rich_noC",     "TTTTTTTTTGTGTGTTTTAGGTT", "3'ss T-rich, score3 14.38",           7),
         ("u1_donor",       "CAGGTGAGT",               "5'ss U1 optimum, score5 10.86",      14),
         ("free_max",       "GCCGACGGAGGACCGACCGATTC", "no constraint, score3 -24.79",        7)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; sp = D["split"][ok]
te = np.flatnonzero((sp == "test") & (X.max(1) <= 3))
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=4096):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
OBSMAX = float(fwd(X[te]).max())
B = X[te[y[te] == 1]]; base = fwd(B)
log(f"[bg] {len(B):,} real 3' end windows; fixed position {A.pos}; "
    f"model's observed maximum on held-out sequence {OBSMAX:+.2f}")
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def raw_logits(s):
    m = tok(s); s0 = A.pos+UP-len(m)//2
    q = B.copy(); q[:, s0:s0+len(m)] = m; return fwd(q)
def deltas(seqs, chunk=32):
    out = np.empty(len(seqs))
    for i in range(0, len(seqs), chunk):
        cs = seqs[i:i+chunk]
        q = np.empty((len(cs)*len(B), B.shape[1]), B.dtype)
        for j, s in enumerate(cs):
            m = tok(s); s0 = A.pos+UP-len(m)//2
            blk = B.copy(); blk[:, s0:s0+len(m)] = m
            q[j*len(B):(j+1)*len(B)] = blk
        out[i:i+chunk] = (fwd(q).reshape(len(cs), len(B))-base[None, :]).mean(1)
    return out
SHIFT = [{a: NT[(NT.index(a)+k) % 4] for a in NT} for k in (1, 2, 3)]
DERANGE = [dict(zip(NT, p)) for p in itertools.permutations(NT) if all(a != b for a, b in zip(NT, p))]
def amap(s, m): return "".join(m[c] for c in s)
rows = []
for name, unit, desc, maxn in UNITS:
    log(f"\n{name}  {unit}  {desc}")
    for n in range(1, maxn+1):
        ins = unit*n
        L = raw_logits(ins); raw = float((L-base).mean())
        past = float((L > OBSMAX).mean())*100
        pmed = float(1/(1+np.exp(-np.median(L))))
        sh = deltas([amap(ins, m) for m in SHIFT])
        dg = deltas([amap(ins, m) for m in DERANGE])
        r = np.random.default_rng(A.seed+n)
        sf = deltas(["".join(r.permutation(list(ins))) for _ in range(A.nshuf)])
        z = (raw-float(dg.mean()))/float(dg.std(ddof=1))
        rows.append((name, unit, desc, n, len(ins), raw, float(sh.mean()), float(dg.mean()),
                     float(dg.std(ddof=1)), float(sf.mean()), raw-float(sh.mean()),
                     raw-float(dg.mean()), raw-float(sf.mean()), z, past, pmed))
        log(f"  n={n:>2} len={len(ins):3d}  raw {raw:+8.4f}  -shift {raw-sh.mean():+8.4f}  "
            f"-derang {raw-dg.mean():+8.4f}  -shuf {raw-sf.mean():+8.4f}  z {z:+7.1f}  "
            f"past-max {past:5.1f}%  p {pmed:.4f}")
with open("dose_null_panel.tsv", "w") as f:
    f.write("name\tunit\tdescription\tn_copies\tinsert_len\traw\tshift_mean\tderange_mean\t"
            "derange_sd\tshuffle_mean\tminus_shift\tminus_derange\tminus_shuffle\tz_vs_derange\t"
            "pct_past_observed_max\tmedian_prob\tobserved_max\tn_bg\n")
    for r_ in rows: f.write("\t".join(str(x) for x in r_)+f"\t{OBSMAX}\t{len(B)}\n")
log("\n[out] dose_null_panel.tsv")
