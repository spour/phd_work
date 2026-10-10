#!/usr/bin/env python3
# how much of the network's within-gene ranking depends on splice sites?
import sys, os, time
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
PUB = os.path.join(ROOT, "..")
UP = 2000; CORE = (UP - 40, UP + 40)
T3, T5 = 6.5, 8.0
NT = np.array(list("ACGTN")); A_, C_, G_, T_ = 0, 1, 2, 3
M3 = M5 = None

def _init():
    global M3, M5
    from maxentpy.maxent import load_matrix3, load_matrix5
    M3 = load_matrix3(); M5 = load_matrix5()

def strong_sites(w, lo=0, hi=None):
    from maxentpy import maxent
    L = len(w); hi = L if hi is None else hi; out = []
    s = "".join(NT[np.minimum(w, 4)])
    for j in range(max(0, lo - 22), min(L - 23, hi - 1) + 1):
        if w[j+18] == A_ and w[j+19] == G_ and "N" not in s[j:j+23]:
            if maxent.score3(s[j:j+23], matrix=M3) >= T3: out.append((j, 23, "acc"))
    for j in range(max(0, lo - 8), min(L - 9, hi - 1) + 1):
        if w[j+3] == G_ and w[j+4] == T_ and "N" not in s[j:j+9]:
            if maxent.score5(s[j:j+9], matrix=M5) >= T5: out.append((j, 9, "don"))
    return out

def merge(iv):
    iv = sorted(iv); m = []
    for a, b in iv:
        if m and a <= m[-1][1]: m[-1][1] = max(m[-1][1], b)
        else: m.append([a, b])
    return [tuple(x) for x in m]

MODE = "permute"
def scramble(w, iv, rng):
    w0 = w; w = w.copy()
    acgt = w0[w0 < 4]; p = np.bincount(acgt, minlength=4) / max(1, len(acgt))
    for a, b in iv:
        for _ in range(50):
            if MODE == "permute": w[a:b] = rng.permutation(w[a:b])
            else: w[a:b] = rng.choice(4, size=b - a, p=p).astype(w.dtype)
            if not strong_sites(w, a, b): break
        else:
            return w, False
    return w, True

def control_intervals(w, iv, taken, rng):
    L = len(w); busy = np.zeros(L, bool)
    for a, b in taken: busy[a:b] = True
    busy[CORE[0]:CORE[1]] = True
    out = []
    for a, b in iv:
        n = b - a
        for _ in range(500):
            s = int(rng.integers(0, L - n))
            if not busy[s:s+n].any():
                busy[s:s+n] = True; out.append((s, s + n)); break
    return out

def _one(i):
    w = XT[i]; rng = np.random.default_rng(1000 + i)
    S = [(a, a + n, k) for a, n, k in strong_sites(w)
         if a + n <= CORE[0] or a >= CORE[1]]
    acc = merge([(a, b) for a, b, k in S if k == "acc"])
    don = merge([(a, b) for a, b, k in S if k == "don"])
    both = merge(acc + don)
    allS = [(a, b) for a, b, _ in S]
    out = {"n_acc": len([1 for *_, k in S if k == "acc"]),
           "n_don": len([1 for *_, k in S if k == "don"]),
           "bp_both": sum(b - a for a, b in both)}
    ok = True
    for name, iv in (("acc", acc), ("don", don), ("both", both)):
        out[name], o1 = scramble(w, iv, rng)
        civ = control_intervals(w, iv, allS, rng)
        out["ctrl_" + name], o2 = scramble(w, civ, rng)
        ok &= o1 and o2 and len(civ) == len(iv)
    out["ok"] = ok
    return out

def do_scan(ncpu):
    global XT
    from multiprocessing import Pool
    D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); ok = D["ok"]
    X = D["X"][ok]; split = D["split"][ok]
    ti = np.flatnonzero(split == "test"); XT = X[ti]
    log(f"[scan] {len(ti):,} held-out windows on {ncpu} CPUs")
    t0 = time.time()
    with Pool(ncpu, initializer=_init) as P:
        R = P.map(_one, range(len(ti)), chunksize=16)
    log(f"[scan] done in {time.time()-t0:.0f}s")
    arms = ["acc", "don", "both", "ctrl_acc", "ctrl_don", "ctrl_both"]
    np.savez_compressed(os.path.join(ROOT, f"data/splice_{TAG}_windows.npz"), test_idx=ti,
        **{a: np.stack([r[a] for r in R]) for a in arms},
        n_acc=np.array([r["n_acc"] for r in R]), n_don=np.array([r["n_don"] for r in R]),
        bp_both=np.array([r["bp_both"] for r in R]), ok=np.array([r["ok"] for r in R]))
    na = np.array([r["n_acc"] for r in R]); nd = np.array([r["n_don"] for r in R])
    log(f"[scan] strong acceptors per window: mean {na.mean():.2f}, median {np.median(na):.0f}; "
        f"donors: mean {nd.mean():.2f}, median {np.median(nd):.0f}; "
        f"bases scrambled (both): mean {np.mean([r['bp_both'] for r in R]):.0f} of 2,500; "
        f"windows fully verified {np.mean([r['ok'] for r in R]):.4f}")

def do_score():
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); ok = D["ok"]
    y = D["y"][ok].astype(int); gene = D["gene"][ok]; X = D["X"][ok]
    W = np.load(os.path.join(ROOT, f"data/splice_{TAG}_windows.npz"))
    ti = W["test_idx"]; yt = y[ti]; gt = gene[ti]
    TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
    for i in range(4): TBL[i, i] = 1.0
    TBLt = torch.from_numpy(TBL)
    class V(nn.Module):
        def __init__(s, pools=(10, 5, 2)):
            super().__init__()
            s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False); s.b1 = nn.BatchNorm1d(32)
            s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
            s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
            s.p = pools; s.do = nn.Dropout(0.4)
            s.fc1 = nn.Linear(96 * 25, 64); s.fc2 = nn.Linear(64, 1)
        def forward(s, x):
            h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), s.p[0])
            h = F.max_pool1d(F.relu(s.b2(s.c2(h))), s.p[1])
            h = F.max_pool1d(F.relu(s.b3(s.c3(h))), s.p[2])
            return s.fc2(s.do(F.relu(s.fc1(s.do(h.flatten(1)))))).squeeze(1)
    def score(net, Xa, bs=1024):
        o = np.empty(len(Xa), np.float32)
        with torch.no_grad():
            for k in range(0, len(Xa), bs):
                xb = TBLt[torch.from_numpy(Xa[k:k+bs].astype(np.int64))].permute(0, 2, 1)
                o[k:k+bs] = net(xb).numpy()
        return o
    def metrics(sc):
        by = {}
        for k in range(len(sc)):
            r, c = by.setdefault(gt[k], ([], []))
            (r if yt[k] == 1 else c).append(sc[k])
        wr, p1 = [], []
        for r, c in by.values():
            if not r or not c: continue
            r = np.array(r); c = np.array(c)
            wr.append(((r[:, None] > c[None, :]).sum() + 0.5 * (r[:, None] == c[None, :]).sum()) / (len(r) * len(c)))
            p1.append(float(r.max() > c.max()))
        return float(np.mean(wr)), float(np.mean(p1)), len(wr)
    ref = {int(r.split("\t")[1]): float(r.split("\t")[7]) for r in open(os.path.join(PUB, "acceptor_arch_train.tsv"))
           if r.startswith("base\t")}
    arms = {"intact": X[ti]}; arms.update({a: W[a] for a in ["acc", "don", "both", "ctrl_acc", "ctrl_don", "ctrl_both"]})
    rows = []
    for seed in (42, 43, 44, 45, 46):
        ck = torch.load(os.path.join(PUB, f"acc_arch_base_s{seed}.pt"), map_location="cpu")
        net = V(); net.load_state_dict(ck["state"]); net.eval()
        for a, Xa in arms.items():
            wr, p1, ng = metrics(score(net, Xa))
            rows.append((seed, a, wr, p1, ng))
            log(f"[seed {seed}] {a:10s} within-gene {wr:.4f}  P@1 {p1:.4f}  ({ng} genes)")
            if a == "intact":
                assert abs(wr - ref[seed]) < 0.002, (wr, ref[seed])
    with open(os.path.join(ROOT, f"data/splice_{TAG}.tsv"), "w") as fh:
        fh.write("seed\tarm\ttest_within_gene\ttest_p_at_1\tn_genes\n")
        for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
    log(f"[out] data/splice_{TAG}.tsv")

if __name__ == "__main__":
    if "resample" in sys.argv: MODE = "resample"
    TAG = "scramble" if MODE == "permute" else "resample"
    if sys.argv[1] == "scan": do_scan(int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8)
    elif sys.argv[1] == "score": do_score()
    elif sys.argv[1] == "test":
        global_ = globals()
        D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); ok = D["ok"]
        ti = np.flatnonzero(D["split"][ok] == "test")[:6]; global_["XT"] = D["X"][ok][ti]
        _init(); t0 = time.time()
        for i in range(len(ti)):
            r = _one(i)
            ch = {a: int((r[a] != XT[i]).sum()) for a in ["acc", "don", "both", "ctrl_both"]}
            comp = MODE == "resample" or all(np.array_equal(np.sort(r[a]), np.sort(XT[i])) for a in ["acc", "don", "both", "ctrl_acc", "ctrl_don", "ctrl_both"])
            left = [k for *_, k in strong_sites(r["both"]) ]
            log(f"  win {i}: acc {r['n_acc']} don {r['n_don']}  bases changed {ch}  "
                f"composition kept {comp}  strong sites left in 'both' {len(left)} "
                f"(core-overlapping allowed)  ok {r['ok']}")
        log(f"  {time.time()-t0:.1f}s for {len(ti)} windows")
