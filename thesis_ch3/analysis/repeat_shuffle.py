#!/usr/bin/env python3
# does the network recognise repeat sequence by its arrangement or by its composition?
import os, sys, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
sys.path.insert(0, PUB); from cnn_interp_lib import dinuc_shuffle
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); X = D["X"]; y = D["y"].astype(int); split = D["split"]
RS = list(csv.DictReader(open(os.path.join(ROOT, "data/repeat_sites.tsv")), delimiter="\t"))
sg = np.array([r["site_group"] for r in RS]); so = np.array([int(r["site_orientation"]) for r in RS])
te = split == "test"
GR = {"real 3' end": te & (y == 1), "competitor, no repeat": te & (y == 0) & (sg == "none"),
      "competitor, L1 antisense": te & (y == 0) & (sg == "L1") & (so == -1), "competitor, L1 sense": te & (y == 0) & (sg == "L1") & (so == 1),
      "competitor, LTR sense": te & (y == 0) & (sg == "LTR") & (so == 1), "competitor, LTR antisense": te & (y == 0) & (sg == "LTR") & (so == -1),
      "competitor, Alu": te & (y == 0) & (sg == "Alu"), "competitor, DNA transposon": te & (y == 0) & (sg == "DNA transposon"),
      "competitor, LTR": te & (y == 0) & (sg == "LTR")}
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
    def forward(s, x):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), 10)
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5); h = F.max_pool1d(F.relu(s.b3(s.c3(h))), 2)
        return s.fc2(s.do(F.relu(s.fc1(s.do(h.flatten(1)))))).squeeze(1)
nets = []
for seed in (42, 43, 44, 45, 46):
    n = V().to(dev); n.load_state_dict(torch.load(os.path.join(PUB, f"acc_arch_base_s{seed}.pt"), map_location=dev)["state"]); n.eval(); nets.append(n)
@torch.no_grad()
def score(Xa, bs=4096):
    o = np.zeros(len(Xa), np.float32)
    for k in range(0, len(Xa), bs):
        xb = TBL[torch.from_numpy(Xa[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
        o[k:k+bs] = np.mean([n(xb).float().cpu().numpy() for n in nets], 0)
    return o
rng = np.random.default_rng(0); out = open(os.path.join(ROOT, "data/repeat_shuffle.tsv"), "w")
out.write("group\tn\tdelta_logit\tlo\thi\n")
for G, m in GR.items():
    idx = np.flatnonzero(m); base = score(X[idx]); d = np.zeros(len(idx))
    for _ in range(3):
        A = X[idx].copy()
        for r in range(len(idx)):
            for a, b in ((1700, 1960), (2040, 2300)):
                seg = A[r, a:b].astype(np.int64)
                if (seg < 4).all(): A[r, a:b] = dinuc_shuffle(seg, rng).astype(A.dtype)
        d += (score(A) - base) / 3
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    out.write(f"{G}\t{len(idx)}\t{d.mean():.4f}\t{lo:.4f}\t{hi:.4f}\n"); out.flush()
    print(f"{G:28s} n {len(idx):5d}  shuffling flanks changes logit by {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
