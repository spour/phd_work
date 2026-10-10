#!/usr/bin/env python3
# panel b in probability, on a background with room to move both ways
import sys, argparse
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nshuf", type=int, default=10); P.add_argument("--pos", type=int, default=-230)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; UP = 2000
UNITS = [("acc_model", "ACGGTGTACTTTTGTTGCAGGTG", "3' splice site the model picks (score3 10.04)", 4),
         ("acc_agdep", "CCATGCCTTTGATCCTGCAGATG", "3' splice site needing its AG (score3 10.38)", 4),
         ("acc_best",  "TTTCCTCTCTCTCCCCGCAGGTG", "3' splice site, MaxEnt optimum (score3 17.17)", 4),
         ("acc_noC",   "TTTTTTTTTGTGTGTTTTAGGTT", "3' splice site, T-rich (score3 14.38)", 4),
         ("donor",     "CAGGTGAGT",               "5' splice site, U1 optimum (score5 10.86)", 8)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
BG = {"real_competitor": X[te[y[te] == 0]], "real_canonical": X[te[y[te] == 1]]}
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=8192):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def prob(s, B):
    m = tok(s); s0 = A.pos+UP-len(m)//2
    q = B.copy(); q[:, s0:s0+len(m)] = m
    return 1/(1+np.exp(-fwd(q)))
rows = []
for bn, B in BG.items():
    p0 = 1/(1+np.exp(-fwd(B)))
    log(f"\n=== {bn}  {len(B):,} windows, baseline median probability {np.median(p0):.4f} ===")
    log(f"  {'unit':<11}{'n':>3}{'median p':>11}{'shuffle p':>11}{'motif-base':>12}"
        f"{'shuf-base':>11}{'motif-shuf':>12}")
    for name, unit, desc, maxn in UNITS:
        for n in range(1, maxn+1):
            ins = unit*n
            pm = np.median(prob(ins, B))
            r = np.random.default_rng(A.seed+n)
            ps = np.array([np.median(prob("".join(r.permutation(list(ins))), B))
                           for _ in range(A.nshuf)])
            rows.append((bn, name, unit, desc, n, len(ins), float(np.median(p0)), pm,
                         float(ps.mean()), float(ps.std(ddof=1)/np.sqrt(A.nshuf)),
                         pm-float(np.median(p0)), float(ps.mean())-float(np.median(p0)),
                         pm-float(ps.mean()), len(B)))
            log(f"  {name:<11}{n:>3}{pm:>11.4f}{ps.mean():>11.4f}"
                f"{pm-np.median(p0):>+12.4f}{ps.mean()-np.median(p0):>+11.4f}"
                f"{pm-ps.mean():>+12.4f}")
with open("panelB_prob.tsv", "w") as f:
    f.write("background\tname\tunit\tdescription\tn_copies\tinsert_len\tbase_median_p\tmotif_p\t"
            "shuffle_p\tshuffle_se\tmotif_minus_base\tshuffle_minus_base\tmotif_minus_shuffle\tn_bg\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
log("\n[out] panelB_prob.tsv")
