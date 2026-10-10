#!/usr/bin/env python3
# does a c tract flip sign with length?
import sys, argparse
import numpy as np, torch
from cnn_interp_lib import NT
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP = 2000
POS = [-1500, -800, -400, -150, -20, 100]
LEN = [6, 12, 23, 46, 92, 161]
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=1024):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
B = X[te[y[te] == 1]]
log(f"[bg] {len(B):,} real 3' end windows; fixed positions {POS}; lengths {LEN}; no maximum")
base = fwd(B); V = {}; rows = []
for b in "ACGT":
    for L in LEN:
        m = np.full(L, NT.index(b), np.uint8)
        for p_ in POS:
            s0 = p_+UP-L//2
            q = B.copy(); q[:, s0:s0+L] = m
            d = float((fwd(q)-base).mean()); V[(b, L, p_)] = d; rows.append((b, L, p_, d))
with open("homopolymer_length.tsv", "w") as f:
    f.write("base\tlength\tposition\tdelta_logit\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
log(f"\nMEAN over the six fixed positions, by base and length")
log(f"  {'base':<6}" + "".join(f"{L:>10}" for L in LEN))
for b in "ACGT":
    log(f"  {b:<6}" + "".join(f"{np.mean([V[(b,L,p)] for p in POS]):>+10.4f}" for L in LEN))
log(f"\nAT THE -150 POSITION alone")
log(f"  {'base':<6}" + "".join(f"{L:>10}" for L in LEN))
for b in "ACGT":
    log(f"  {b:<6}" + "".join(f"{V[(b,L,-150)]:>+10.4f}" for L in LEN))
import os
for fn in ("cnn_kmer_scan.tsv", "kmer_scan.tsv"):
    if os.path.exists(fn):
        log(f"\n[6-mer scan] the four homopolymers as read from {fn}")
        hdr = None
        for line in open(fn):
            p = line.rstrip("\n").split("\t")
            if hdr is None: hdr = p; continue
            if p[0] in ("AAAAAA","CCCCCC","GGGGGG","TTTTTT"):
                log("   " + "  ".join(f"{h}={v}" for h, v in zip(hdr, p)))
        break
log("[out] homopolymer_length.tsv")
