#!/usr/bin/env python3
# resolve the flagged anomaly
import sys, time, argparse
import numpy as np, torch
from cnn_interp_lib import NT
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt"); P.add_argument("--ncomp", type=int, default=4096)
P.add_argument("--seed", type=int, default=42); P.add_argument("--copies", type=int, default=1)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
POS = [-1500, -800, -400, -150, -20, 100]
rng = np.random.default_rng(A.seed)
UNITS = {"acc_ppt": ("TTTTTTTTTTTTTCAGG", 14),
         "acc_noC": ("TTTTTTTTTGTGTGTTTTAGGTT", 18),
         "acc_best": ("TTTCCTCTCTCTCCCCGCAGGTG", 18)}
def t2c(s, j): return s[:j].replace("T", "C")+s[j:]
def agko(s, j): return s[:j]+"AC"+s[j+2:]
SEQ = []
for k, (s, j) in UNITS.items():
    SEQ += [(k, "wt", s), (k, "t2c", t2c(s, j)), (k, "ag_ko", agko(s, j))]
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
log(f"[bg] copies={A.copies}; {len(B):,} real 3' end windows; fixed positions {POS}, NO maximum taken")
base = fwd(B); V = {}; rows = []
for k, arm, s_ in SEQ:
    m = np.array([NT.index(c) for c in s_*A.copies], np.uint8)
    for p_ in POS:
        s0 = p_+UP-len(m)//2
        q = B.copy(); q[:, s0:s0+len(m)] = m
        d = float((fwd(q)-base).mean()); V[(k, arm, p_)] = d
        rows.append((k, arm, p_, d))
with open("acceptor_fixedpos_check_n%d.tsv" % A.copies, "w") as f:
    f.write("unit\tarm\tposition\tdelta_logit\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
log(f"\nRAW effect at each FIXED position (no maximum)")
log(f"  {'unit':<9}{'arm':<7}" + "".join(f"{p:>+10}" for p in POS))
for k in UNITS:
    for arm in ("wt", "t2c", "ag_ko"):
        log(f"  {k:<9}{arm:<7}" + "".join(f"{V[(k,arm,p)]:>+10.4f}" for p in POS))
log(f"\nWT MINUS ITS VARIANT at each fixed position (positive = intact scores higher)")
log(f"  {'unit':<9}{'control':<9}" + "".join(f"{p:>+10}" for p in POS) + f"{'mean':>10}")
for k in UNITS:
    for arm in ("t2c", "ag_ko"):
        d = [V[(k,"wt",p)]-V[(k,arm,p)] for p in POS]
        log(f"  {k:<9}{arm:<9}" + "".join(f"{x:>+10.4f}" for x in d) + f"{np.mean(d):>+10.4f}")
log(f"\n[compare] the dose table, which takes a MAXIMUM over 60 positions, reported")
log(f"          acc_best minus its T-to-C variant as -0.064 at one copy.")
d = np.mean([V[("acc_best","wt",p)]-V[("acc_best","t2c",p)] for p in POS])
log(f"          At fixed positions the same quantity is {d:+.4f}.")
log("[out] acceptor_fixedpos_check_n%d.tsv" % A.copies)
