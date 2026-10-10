#!/usr/bin/env python3
# where does the tandem series stop meaning anything?
import sys, argparse, itertools
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--maxn", type=int, default=24); P.add_argument("--pos", type=int, default=-230)
A = P.parse_args()
NT = "ACGT"; UP = 2000
UNITS = [("model_me3", "ACGGTGTACTTTTGTTGCAGGTG"), ("robust_agdep", "CCATGCCTTTGATCCTGCAGATG"),
         ("maxent_optimum", "TTTCCTCTCTCTCCCCGCAGGTG"), ("t_rich_noC", "TTTTTTTTTGTGTGTTTTAGGTT"),
         ("u1_donor", "CAGGTGAGT"), ("free_max", "GCCGACGGAGGACCGACCGATTC")]
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
B = X[te[y[te] == 1]]; base = fwd(B); W = B.shape[1]
DERANGE = [dict(zip(NT, p)) for p in itertools.permutations(NT) if all(a != b for a, b in zip(NT, p))]
def amap(s, m): return "".join(m[c] for c in s)
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def logits(s):
    m = tok(s); s0 = A.pos+UP-len(m)//2
    if s0 < 0 or s0+len(m) > W: return None
    q = B.copy(); q[:, s0:s0+len(m)] = m; return fwd(q)
log(f"[bg] {len(B):,} real 3' end windows; fixed position {A.pos}; ceiling {OBSMAX:+.2f}; "
    f"window {W} nt")
rows = []
for name, unit in UNITS:
    log(f"\n{name}  {unit}  ({len(unit)} nt per copy)")
    log(f"  {'n':>3}{'nt':>6}{'raw':>10}{'marginal':>10}{'-derang':>10}{'med p':>9}"
        f"{'ceiling':>9}")
    prev = 0.0
    for n in range(1, A.maxn+1):
        ins = unit*n
        L = logits(ins)
        if L is None:
            log(f"  {n:>3}{len(ins):>6}   insert no longer fits the window, stopping"); break
        raw = float((L-base).mean()); marg = raw-prev; prev = raw
        dg = np.array([float((logits(amap(ins, m))-base).mean()) for m in DERANGE])
        pmed = float(1/(1+np.exp(-np.median(L)))); ceil = float((L > OBSMAX).mean())*100
        rows.append((name, unit, n, len(ins), raw, marg, float(dg.mean()), raw-float(dg.mean()),
                     pmed, ceil))
        log(f"  {n:>3}{len(ins):>6}{raw:>+10.3f}{marg:>+10.3f}{raw-dg.mean():>+10.3f}"
            f"{pmed:>9.5f}{ceil:>8.1f}%")
with open("dose_saturation.tsv", "w") as f:
    f.write("name\tunit\tn_copies\tinsert_len\traw\tmarginal\tderange_mean\tminus_derange\t"
            "median_prob\tpct_past_ceiling\tobserved_max\tn_bg\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+f"\t{OBSMAX}\t{len(B)}\n")
log(f"\nFIRST DOSE AT WHICH EACH DIAGNOSTIC FAILS")
log(f"  {'unit':<16}{'med p > .999':>14}{'ceiling > 5%':>14}{'marginal falls':>16}{'peak raw at':>13}")
import collections
G = collections.defaultdict(list)
for r in rows: G[r[0]].append(r)
for name, rs in G.items():
    p999 = next((r[2] for r in rs if r[8] > .999), None)
    c5 = next((r[2] for r in rs if r[9] > 5), None)
    half = next((r[2] for r in rs[1:] if r[5] < rs[0][5]/2), None)
    pk = max(rs, key=lambda r: r[4])[2]
    f = lambda v: str(v) if v else "never"
    log(f"  {name:<16}{f(p999):>14}{f(c5):>14}{f(half):>16}{pk:>13}")
log("[out] dose_saturation.tsv")
