#!/usr/bin/env python3
# search for a 3' splice site that behaves like one
import sys, argparse
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nbg_search", type=int, default=128)
P.add_argument("--kshuf", type=int, default=6)
P.add_argument("--nperm", type=int, default=500)
P.add_argument("--restarts", type=int, default=3)
P.add_argument("--iters", type=int, default=40)
P.add_argument("--floor", type=float, default=10.0)
P.add_argument("--wg", type=float, default=2.0)
P.add_argument("--pos", type=int, default=-230)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; L = 23; AGI = 18; UP = 2000
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
B_all = X[te[y[te] == 1]]
rng0 = np.random.default_rng(A.seed)
B_s = B_all[rng0.choice(len(B_all), min(A.nbg_search, len(B_all)), replace=False)]
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=4096):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
BS = fwd(B_s); BA = fwd(B_all)
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def deltas(seqs, bg, base, pos, chunk=48):
    out = np.empty(len(seqs))
    for i in range(0, len(seqs), chunk):
        cs = seqs[i:i+chunk]
        q = np.empty((len(cs)*len(bg), bg.shape[1]), bg.dtype)
        for j, s in enumerate(cs):
            m = tok(s); s0 = pos+UP-len(m)//2
            blk = bg.copy(); blk[:, s0:s0+len(m)] = m
            q[j*len(bg):(j+1)*len(bg)] = blk
        v = fwd(q).reshape(len(cs), len(bg))
        out[i:i+chunk] = (v-base[None, :]).mean(1)
    return out
def agko(s): return s[:AGI]+"AC"+s[AGI+2:]
def shuffles(s, r, k): return ["".join(r.permutation(list(s))) for _ in range(k)]
def score_batch(cands, r):
    pack, idx = [], []
    for s in cands:
        base_i = len(pack); pack.append(s); pack.append(agko(s))
        pack.extend(shuffles(s, r, A.kshuf)); idx.append(base_i)
    v = deltas(pack, B_s, BS, A.pos)
    Aa = np.array([v[i] for i in idx])
    Gg = Aa-np.array([v[i+1] for i in idx])
    Ss = Aa-np.array([v[i+2:i+2+A.kshuf].mean() for i in idx])
    return Aa, Ss, Gg
def ok3(s): return s[AGI:AGI+2] == "AG" and s3(s) >= A.floor
def seed_seq(r):
    for _ in range(20000):
        s = "".join(r.choice(list(NT), L)); s = s[:AGI]+"AG"+s[AGI+2:]
        if ok3(s): return s
    return "TTTCCTCTCTCTCCCCGCAGGTG"
def obj(arm, Aa, Ss, Gg):
    return Gg if arm == "agdep" else Aa+Ss+A.wg*Gg
trace, winners = [], {}
for arm in ("agdep", "joint"):
    best = None
    for rs in range(A.restarts):
        r = np.random.default_rng(A.seed+7919*rs+(0 if arm == "agdep" else 31))
        s = seed_seq(r)
        a0, s0_, g0 = score_batch([s], r); cur = float(obj(arm, a0, s0_, g0)[0])
        trace.append((arm, rs, 0, s, float(a0[0]), float(s0_[0]), float(g0[0]), cur, s3(s)))
        it = 0
        for it in range(1, A.iters+1):
            cand = [s[:i]+b+s[i+1:] for i in range(L) if i not in (AGI, AGI+1)
                    for b in NT if b != s[i]]
            cand = [c for c in cand if ok3(c)]
            if not cand: break
            Aa, Ss, Gg = score_batch(cand, r); J = obj(arm, Aa, Ss, Gg)
            j = int(np.argmax(J))
            if J[j] <= cur+1e-6: break
            s, cur = cand[j], float(J[j])
            trace.append((arm, rs, it, s, float(Aa[j]), float(Ss[j]), float(Gg[j]), cur, s3(s)))
        aa, ss, gg = score_batch([s], r)
        log(f"[{arm}] restart {rs}: {s}  A {aa[0]:+.3f}  S {ss[0]:+.3f}  G {gg[0]:+.3f}  "
            f"score3 {s3(s):+.2f}  ({it} steps)")
        if best is None or cur > best[1]: best = (s, cur)
    winners[arm] = best[0]
    log(f"[{arm}] WINNER {best[0]}")
with open("acceptor_agdep_search.tsv", "w") as f:
    f.write("arm\trestart\tstep\tsequence\tA_activation\tS_vs_shuffle\tG_ag_dependence\tobjective\tscore3\n")
    for t in trace: f.write("\t".join(str(x) for x in t)+"\n")

# full evaluation on all backgrounds
CAND = {f"search_{k}": v for k, v in winners.items()}
CAND["prev_model_me3"] = "ACGGTGTACTTTTGTTGCAGGTG"
CAND["maxent_optimum"] = "TTTCCTCTCTCTCCCCGCAGGTG"
CAND["t_rich_noC"] = "TTTTTTTTTGTGTGTTTTAGGTT"
log(f"\n[final] {len(B_all):,} backgrounds, {A.nperm} permutations, fixed position {A.pos}")
rows, prows = [], []
for name, s in CAND.items():
    pr = np.random.default_rng(A.seed)
    perms = shuffles(s, pr, A.nperm)
    variants = {"intact": s, "ag_to_ac": agko(s), "ag_to_ct": s[:AGI]+"CT"+s[AGI+2:],
                "ag_to_ga": s[:AGI]+"GA"+s[AGI+2:], "ag_deleted": s[:AGI]+s[AGI+2:],
                "reversed": s[::-1]}
    vv = deltas(list(variants.values()), B_all, BA, A.pos)
    dv = dict(zip(variants.keys(), vv))
    pv = deltas(perms, B_all, BA, A.pos)
    d = dv["intact"]; z = (d-pv.mean())/pv.std(ddof=1); ge = int((pv >= d).sum())
    agworst = max(dv["ag_to_ac"], dv["ag_to_ct"], dv["ag_to_ga"], dv["ag_deleted"])
    rows.append((name, s, s3(s), d, float(pv.mean()), z, (ge+1)/(A.nperm+1),
                 dv["ag_to_ac"], dv["ag_to_ct"], dv["ag_to_ga"], dv["ag_deleted"],
                 dv["reversed"], d-float(pv.mean()), d-agworst))
    for t, v in zip(perms, pv): prows.append((name, t, float(v)))
with open("acceptor_agdep_final.tsv", "w") as f:
    f.write("name\tsequence\tscore3\tintact\tperm_mean\tz_vs_perm\tp_perm\tag_to_ac\tag_to_ct\t"
            "ag_to_ga\tag_deleted\treversed\tdrop_on_shuffle\tdrop_on_best_AG_disruption\tn_bg\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+f"\t{len(B_all)}\n")
with open("acceptor_agdep_perm.tsv", "w") as f:
    f.write("name\tpermuted_sequence\tdelta_logit\n")
    for r in prows: f.write("\t".join(str(x) for x in r)+"\n")
log(f"\n{'name':<16}{'score3':>7}{'intact':>9}{'shuffle':>9}{'z':>8}{'AG>AC':>8}{'AG>CT':>8}"
    f"{'AG>GA':>8}{'AGdel':>8}{'dropShuf':>10}{'dropAG':>8}")
for r in rows:
    log(f"{r[0]:<16}{r[2]:>+7.2f}{r[3]:>+9.3f}{r[4]:>+9.3f}{r[5]:>+8.1f}{r[7]:>+8.3f}"
        f"{r[8]:>+8.3f}{r[9]:>+8.3f}{r[10]:>+8.3f}{r[12]:>+10.3f}{r[13]:>+8.3f}")
log("[out] acceptor_agdep_search.tsv / _final.tsv / _perm.tsv")
