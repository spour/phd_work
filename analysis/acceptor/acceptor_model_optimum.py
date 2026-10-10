#!/usr/bin/env python3
# what 3' splice site does the model like?
import sys, argparse, math
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nbg_search", type=int, default=256)
P.add_argument("--nperm", type=int, default=500)
P.add_argument("--restarts", type=int, default=3)
P.add_argument("--iters", type=int, default=40)
P.add_argument("--floor", type=float, default=10.0)
P.add_argument("--pos", type=int, default=-230)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; L = 23; AGI = 18
UP = 2000
dev = "cuda" if torch.cuda.is_available() else "cpu"

from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def score3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return float("nan")

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
B_all = X[te[y[te] == 1]]
rng = np.random.default_rng(A.seed)
B_srch = B_all[rng.choice(len(B_all), min(A.nbg_search, len(B_all)), replace=False)]
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=2048):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
BASE_S = fwd(B_srch); BASE_A = fwd(B_all)
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def implant(bg, m, pos):
    s0 = pos+UP-len(m)//2
    q = bg.copy(); q[:, s0:s0+len(m)] = m; return q
def delta(s, bg, base, pos):
    return float((fwd(implant(bg, tok(s), pos))-base).mean())
def delta_many(seqs, bg, base, pos, bs=64):
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i+bs]
        q = np.concatenate([implant(bg, tok(s), pos) for s in chunk], 0)
        v = fwd(q).reshape(len(chunk), len(bg))
        out.extend((v-base[None,:]).mean(1).tolist())
    return out

def ok_arm(s, arm):
    if arm == "free": return True
    if s[AGI:AGI+2] != "AG": return False
    if arm == "ag": return True
    return score3(s) >= A.floor

def seed_seq(arm, r):
    for _ in range(4000):
        s = "".join(r.choice(list(NT), L))
        if arm != "free": s = s[:AGI]+"AG"+s[AGI+2:]
        if ok_arm(s, arm): return s
    return "TTTCCTCTCTCTCCCCGCAGGTG"

trace = []
winners = {}
for arm in ("free", "ag", "me3"):
    best_overall = None
    for rs in range(A.restarts):
        r = np.random.default_rng(A.seed+1000*rs+hash(arm) % 997)
        s = seed_seq(arm, r); v = delta(s, B_srch, BASE_S, A.pos)
        trace.append((arm, rs, 0, s, v, score3(s)))
        for it in range(1, A.iters+1):
            cand = []
            for i in range(L):
                if arm != "free" and i in (AGI, AGI+1): continue
                for b in NT:
                    if b == s[i]: continue
                    t = s[:i]+b+s[i+1:]
                    if ok_arm(t, arm): cand.append(t)
            if not cand: break
            vals = delta_many(cand, B_srch, BASE_S, A.pos)
            j = int(np.argmax(vals))
            if vals[j] <= v + 1e-6: break
            s, v = cand[j], vals[j]
            trace.append((arm, rs, it, s, v, score3(s)))
        log(f"[{arm}] restart {rs}: {s}  delta {v:+.4f}  score3 {score3(s):+.2f}  ({it} steps)")
        if best_overall is None or v > best_overall[1]: best_overall = (s, v)
    winners[arm] = best_overall[0]
    log(f"[{arm}] WINNER {best_overall[0]}  search delta {best_overall[1]:+.4f}")

with open("acceptor_model_optimum.tsv", "w") as f:
    f.write("arm\trestart\tstep\tsequence\tdelta_search\tscore3\n")
    for t in trace: f.write("\t".join(str(x) for x in t)+"\n")

# permutation battery on the full background set, at the same fixed position
REF = {"maxent_optimum":"TTTCCTCTCTCTCCCCGCAGGTG","t_rich_noC":"TTTTTTTTTGTGTGTTTTAGGTT"}
ALL = {f"model_{k}": v for k, v in winners.items()}; ALL.update(REF)
rows = []; perm_rows = []
log(f"\n[perm] {A.nperm} permutations of each winner's own bases; {len(B_all):,} backgrounds; "
    f"fixed position {A.pos}")
for name, s in ALL.items():
    d = delta(s, B_all, BASE_A, A.pos)
    pr = np.random.default_rng(A.seed)
    perms = []
    seen = set()
    for _ in range(A.nperm):
        t = "".join(pr.permutation(list(s)))
        perms.append(t); seen.add(t)
    pv = delta_many(perms, B_all, BASE_A, A.pos)
    pv = np.array(pv)
    ge = int((pv >= d).sum())
    z = (d-pv.mean())/pv.std(ddof=1)
    ident = sum(1 for t in perms if t == s)
    ag = s[:AGI]+"AC"+s[AGI+2:] if s[AGI:AGI+2] == "AG" else None
    d_ag = delta(ag, B_all, BASE_A, A.pos) if ag else float("nan")
    d_rev = delta(s[::-1], B_all, BASE_A, A.pos)
    rows.append((name, s, score3(s), d, float(pv.mean()), float(pv.std(ddof=1)), z,
                 (ge+1)/(A.nperm+1), ident, float(pv.max()), d_ag, d_rev))
    for t, v in zip(perms, pv): perm_rows.append((name, t, float(v)))
    log(f"  {name:<18} {s}  score3 {score3(s):+6.2f}  intact {d:+.4f}  "
        f"perm mean {pv.mean():+.4f} sd {pv.std(ddof=1):.4f}  z {z:+.2f}  "
        f"p {(ge+1)/(A.nperm+1):.4f}  identical {ident}")
with open("acceptor_model_optimum_perm.tsv", "w") as f:
    f.write("name\tpermuted_sequence\tdelta_logit\n")
    for r in perm_rows: f.write("\t".join(str(x) for x in r)+"\n")
with open("acceptor_model_optimum_summary.tsv", "w") as f:
    f.write("name\tsequence\tscore3\tdelta_intact\tperm_mean\tperm_sd\tz_vs_perm\tp_perm\t"
            "n_identical_perms\tperm_max\tdelta_AG_knockout\tdelta_reversed\tn_bg\tposition\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+f"\t{len(B_all)}\t{A.pos}\n")
log(f"\n{'name':<18}{'intact':>9}{'perm mean':>11}{'z':>8}{'p':>9}{'AG knockout':>13}{'reversed':>10}")
for r in rows:
    log(f"{r[0]:<18}{r[3]:>+9.4f}{r[4]:>+11.4f}{r[6]:>+8.2f}{r[7]:>9.4f}{r[10]:>+13.4f}{r[11]:>+10.4f}")
log("[out] acceptor_model_optimum.tsv / _perm.tsv / _summary.tsv")
