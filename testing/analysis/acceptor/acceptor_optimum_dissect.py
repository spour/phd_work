#!/usr/bin/env python3
# localise the model's favourite acceptor
import sys, argparse
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--pos", type=int, default=-230); A = P.parse_args()
NT = "ACGT"; UP = 2000
WIN = "ACGGTGTACTTTTGTTGCAGGTG"
REF = {"model_me3": WIN, "maxent_optimum": "TTTCCTCTCTCTCCCCGCAGGTG",
       "t_rich_noC": "TTTTTTTTTGTGTGTTTTAGGTT"}
dev = "cuda" if torch.cuda.is_available() else "cpu"
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return float("nan")
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
B = X[te[y[te] == 1]]
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=2048):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
base = fwd(B)
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def d_at(s, pos):
    m = tok(s); s0 = pos+UP-len(m)//2
    q = B.copy(); q[:, s0:s0+len(m)] = m
    return float((fwd(q)-base).mean())
rows = []
GRID = list(range(-1900, 500, 40))
log(f"[bg] {len(B):,} real 3' end windows")
log(f"\nBEST OF {len(GRID)} POSITIONS, comparable to the dose table")
for k, s in REF.items():
    v = [d_at(s, p) for p in GRID]
    j = int(np.argmax(v))
    rows.append(("best_of_grid", k, s, GRID[j], max(v), s3(s)))
    log(f"  {k:<16} {s}  score3 {s3(s):+6.2f}  best {max(v):+.4f} at {GRID[j]:+5d}  "
        f"(fixed {A.pos}: {d_at(s,A.pos):+.4f})")
log(f"\nSATURATION MUTAGENESIS of {WIN} at fixed position {A.pos}")
d0 = d_at(WIN, A.pos)
log(f"  intact {d0:+.4f}")
M = np.zeros((len(WIN), 4))
for i in range(len(WIN)):
    for bi, b in enumerate(NT):
        if b == WIN[i]: M[i, bi] = 0.0; continue
        t = WIN[:i]+b+WIN[i+1:]
        M[i, bi] = d_at(t, A.pos)-d0
        rows.append(("saturation", f"pos{i+1}_{WIN[i]}to{b}", t, A.pos, d0+M[i, bi], s3(t)))
log(f"  {'pos':>4} {'wt':>3} " + "".join(f"{b:>9}" for b in NT) + f"{'worst':>9}")
for i in range(len(WIN)):
    log(f"  {i+1:>4} {WIN[i]:>3} " + "".join(f"{M[i,bi]:>+9.3f}" for bi in range(4))
        + f"{M[i].min():>+9.3f}")
imp = -M.min(1)
log(f"\n  most costly positions (drop when mutated to the worst base):")
for i in np.argsort(-imp)[:8]:
    log(f"    position {i+1:>2}  {WIN[i]}  -{imp[i]:.3f}   context {WIN[max(0,i-4):i]}[{WIN[i]}]{WIN[i+1:i+5]}")
log(f"\nTARGETED KNOCKOUTS")
KO = {"TGTAC->TGCAC (pos 5-9)": WIN[:4]+"TGCAC"+WIN[9:],
      "TGTAC->CACGT scramble":  WIN[:4]+"CACGT"+WIN[9:],
      "TGTAC->AAAAA":           WIN[:4]+"AAAAA"+WIN[9:],
      "AG->AC (pos 19-20)":     WIN[:18]+"AC"+WIN[20:],
      "reversed":               WIN[::-1]}
for k, t in KO.items():
    v = d_at(t, A.pos); rows.append(("knockout", k, t, A.pos, v, s3(t)))
    log(f"  {k:<26} {t}  {v:+.4f}   ({v-d0:+.4f} vs intact, score3 {s3(t):+6.2f})")
with open("acceptor_optimum_dissect.tsv", "w") as f:
    f.write("kind\tname\tsequence\tposition\tdelta_logit\tscore3\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
np.savetxt("acceptor_optimum_satmut.tsv", M, delimiter="\t", header="\t".join(NT), comments="")
log("[out] acceptor_optimum_dissect.tsv / acceptor_optimum_satmut.tsv")
