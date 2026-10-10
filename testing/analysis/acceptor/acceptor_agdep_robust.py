#!/usr/bin/env python3
# an ag requirement the search cannot game
import sys, argparse, itertools
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nbg_search", type=int, default=128); P.add_argument("--kshuf", type=int, default=6)
P.add_argument("--nperm", type=int, default=500); P.add_argument("--restarts", type=int, default=3)
P.add_argument("--iters", type=int, default=40); P.add_argument("--floor", type=float, default=10.0)
P.add_argument("--wg", type=float, default=2.0); P.add_argument("--pos", type=int, default=-230)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; L = 23; AGI = 18; UP = 2000
DIS = ["AC", "CT", "GA", "TT", "CC"]
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
r0 = np.random.default_rng(A.seed)
B_s = B_all[r0.choice(len(B_all), min(A.nbg_search, len(B_all)), replace=False)]
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
        out[i:i+chunk] = (fwd(q).reshape(len(cs), len(bg))-base[None, :]).mean(1)
    return out
def disrupt(s): return [s[:AGI]+d+s[AGI+2:] for d in DIS]+[s[:AGI]+s[AGI+2:]]
NDIS = len(DIS)+1
def shuf(s, r, k): return ["".join(r.permutation(list(s))) for _ in range(k)]
def score_batch(cands, r):
    pack, idx = [], []
    for s in cands:
        idx.append(len(pack)); pack.append(s); pack.extend(disrupt(s)); pack.extend(shuf(s, r, A.kshuf))
    v = deltas(pack, B_s, BS, A.pos)
    Aa = np.array([v[i] for i in idx])
    Gr = Aa-np.array([v[i+1:i+1+NDIS].max() for i in idx])
    Ss = Aa-np.array([v[i+1+NDIS:i+1+NDIS+A.kshuf].mean() for i in idx])
    return Aa, Ss, Gr
def ok3(s): return s[AGI:AGI+2] == "AG" and s3(s) >= A.floor
def seed_seq(r):
    for _ in range(20000):
        s = "".join(r.choice(list(NT), L)); s = s[:AGI]+"AG"+s[AGI+2:]
        if ok3(s): return s
    return "TTTCCTCTCTCTCCCCGCAGGTG"
trace, winners = [], {}
for arm in ("grob", "joint_rob"):
    best = None
    for rs in range(A.restarts):
        r = np.random.default_rng(A.seed+7919*rs+(0 if arm == "grob" else 31))
        s = seed_seq(r)
        aa, ss, gg = score_batch([s], r)
        cur = float(gg[0] if arm == "grob" else aa[0]+ss[0]+A.wg*gg[0])
        trace.append((arm, rs, 0, s, float(aa[0]), float(ss[0]), float(gg[0]), cur, s3(s)))
        it = 0
        for it in range(1, A.iters+1):
            cand = [c for c in (s[:i]+b+s[i+1:] for i in range(L) if i not in (AGI, AGI+1)
                                for b in NT if b != s[i]) if ok3(c)]
            if not cand: break
            Aa, Ss, Gr = score_batch(cand, r)
            J = Gr if arm == "grob" else Aa+Ss+A.wg*Gr
            j = int(np.argmax(J))
            if J[j] <= cur+1e-6: break
            s, cur = cand[j], float(J[j])
            trace.append((arm, rs, it, s, float(Aa[j]), float(Ss[j]), float(Gr[j]), cur, s3(s)))
        aa, ss, gg = score_batch([s], r)
        log(f"[{arm}] restart {rs}: {s}  A {aa[0]:+.3f}  S {ss[0]:+.3f}  Grob {gg[0]:+.3f}  "
            f"score3 {s3(s):+.2f}  ({it} steps)")
        if best is None or cur > best[1]: best = (s, cur)
    winners[arm] = best[0]; log(f"[{arm}] WINNER {best[0]}")
with open("acceptor_agdep_robust_search.tsv", "w") as f:
    f.write("arm\trestart\tstep\tsequence\tA\tS\tGrob\tobjective\tscore3\n")
    for t in trace: f.write("\t".join(str(x) for x in t)+"\n")
CAND = {f"robust_{k}": v for k, v in winners.items()}
CAND["prev_agdep"] = "TGCCGCCGGTGTCCCGACAGATT"
CAND["prev_model_me3"] = "ACGGTGTACTTTTGTTGCAGGTG"
CAND["maxent_optimum"] = "TTTCCTCTCTCTCCCCGCAGGTG"
CAND["t_rich_noC"] = "TTTTTTTTTGTGTGTTTTAGGTT"
DN = ["".join(p) for p in itertools.product(NT, repeat=2)]
log(f"\n[final] {len(B_all):,} backgrounds, {A.nperm} permutations, all 16 dinucleotides at 19-20")
rows, drows = [], []
for name, s in CAND.items():
    pr = np.random.default_rng(A.seed)
    perms = shuf(s, pr, A.nperm)
    dseqs = [s[:AGI]+d+s[AGI+2:] for d in DN]+[s[:AGI]+s[AGI+2:], s[::-1]]
    dv = deltas(dseqs, B_all, BA, A.pos)
    dmap = dict(zip(DN, dv[:16])); ddel = dv[16]; drev = dv[17]
    pv = deltas(perms, B_all, BA, A.pos)
    d = dmap["AG"]; z = (d-pv.mean())/pv.std(ddof=1); ge = int((pv >= d).sum())
    alt = [dmap[k] for k in DN if k != "AG"]+[ddel]
    rows.append((name, s, s3(s), d, float(pv.mean()), z, (ge+1)/(A.nperm+1),
                 d-float(pv.mean()), float(np.mean(alt)), float(np.max(alt)),
                 d-float(np.max(alt)), int(sum(1 for a in alt if a >= d)), ddel, drev))
    for k in DN: drows.append((name, k, float(dmap[k])))
    drows.append((name, "deleted", float(ddel)))
    log(f"  {name:<16} {s}  intact {d:+.3f}  best alternative {max(alt):+.3f}  "
        f"worst-case AG gain {d-max(alt):+.3f}  ({sum(1 for a in alt if a>=d)}/16 beat AG)")
with open("acceptor_agdep_robust_final.tsv", "w") as f:
    f.write("name\tsequence\tscore3\tintact_AG\tperm_mean\tz_vs_perm\tp_perm\tdrop_on_shuffle\t"
            "alt_dinuc_mean\talt_dinuc_best\tworst_case_AG_gain\tn_alts_beating_AG\tdeleted\treversed\tn_bg\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+f"\t{len(B_all)}\n")
with open("acceptor_agdep_dinuc.tsv", "w") as f:
    f.write("name\tdinucleotide\tdelta_logit\n")
    for r in drows: f.write("\t".join(str(x) for x in r)+"\n")
log(f"\n{'name':<16}{'score3':>7}{'AG':>8}{'shuffle':>9}{'z':>7}{'altMean':>9}{'altBest':>9}"
    f"{'AGgain':>8}{'beatAG':>8}")
for r in rows:
    log(f"{r[0]:<16}{r[2]:>+7.2f}{r[3]:>+8.3f}{r[4]:>+9.3f}{r[5]:>+7.1f}{r[8]:>+9.3f}"
        f"{r[9]:>+9.3f}{r[10]:>+8.3f}{r[11]:>6d}/16")
log("[out] acceptor_agdep_robust_search.tsv / _final.tsv / acceptor_agdep_dinuc.tsv")
