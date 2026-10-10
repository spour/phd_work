#!/usr/bin/env python3
# does the implantation result depend on where the construct is written?
import sys, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def log(*a): print(*a, file=sys.stderr, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
dev = "cuda" if torch.cuda.is_available() else "cpu"; log("device", dev)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
X = D["X"]; y = D["y"].astype(int); s = D["split"]
comp = np.flatnonzero((s == "test") & (y == 0)); Xc = X[comp]; log(f"{len(comp):,} competitor windows")
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
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5)
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), 2)
        return s.fc2(s.do(F.relu(s.fc1(s.do(h.flatten(1)))))).squeeze(1)
nets = []
for seed in (42, 43, 44, 45, 46):
    n = V().to(dev); n.load_state_dict(torch.load(os.path.join(PUB, f"acc_arch_base_s{seed}.pt"), map_location=dev)["state"]); n.eval(); nets.append(n)
@torch.no_grad()
def score(Xa, bs=4096):
    out = np.zeros(len(Xa), np.float32)
    for k in range(0, len(Xa), bs):
        xb = TBL[torch.from_numpy(Xa[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
        out[k:k+bs] = np.mean([n(xb).float().cpu().numpy() for n in nets], 0)
    return out
CODE = {c: i for i, c in enumerate("ACGT")}
enc = lambda q: np.array([CODE[c] for c in q], np.uint8)
CON = [("acceptor", "TTTAACTTTCTTTTTTACAGACG", 1), ("acceptor", "TTTAACTTTCTTTTTTACAGACG", 4),
       ("donor", "CAGGTGAGT", 1), ("donor", "CAGGTGAGT", 8)]
OFFS = sorted(set(list(range(-1900, 401, 100)) + [-230]))
rng = np.random.default_rng(0); t0 = time.time()
base = score(Xc); pbase = 1 / (1 + np.exp(-base))
out = open(os.path.join(ROOT, "data/implant_position.tsv"), "w")
out.write("construct\tcopies\toffset\tdelta_construct\tdelta_control\tdiff\tdiff_lo\tdiff_hi\tmedian_p_construct\tmedian_p_base\n")
for name, unit, k in CON:
    ins = enc(unit * k); L = len(ins); shuf = [rng.permutation(ins) for _ in range(3)]
    for off in OFFS:
        a = off + 2000
        if a < 0 or a + L > 2500: continue
        A = Xc.copy(); A[:, a:a+L] = ins; dc = score(A) - base
        dctl = np.zeros(len(Xc), np.float32)
        for sq in shuf:
            B = Xc.copy(); B[:, a:a+L] = sq; dctl += (score(B) - base) / len(shuf)
        d = dc - dctl; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.write(f"{name}\t{k}\t{off}\t{dc.mean():.4f}\t{dctl.mean():.4f}\t{d.mean():.4f}\t{lo:.4f}\t{hi:.4f}\t"
                  f"{np.median(1/(1+np.exp(-(base+dc)))):.4f}\t{np.median(pbase):.4f}\n"); out.flush()
    log(f"  {name} x{k} done  {time.time()-t0:.0f}s")
out.close(); log("[out] data/implant_position.tsv")
