#!/usr/bin/env python3
# evolve the 5' splice site itself, and build real pwms for what the model wants at fixed
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nrep", type=int, default=64)
P.add_argument("--nbg", type=int, default=64)
P.add_argument("--steps", type=int, default=40)
P.add_argument("--pos", type=int, default=-150)
P.add_argument("--bs", type=int, default=768)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
rng = np.random.default_rng(A.seed)

sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
POW = np.array([int(L.POW[p]) for p in range(9)])
U1 = np.asarray(L.U1)
def u1dg(t): return float(U1[int(np.dot(np.asarray(t, np.int64), POW))])
U1X = "".join("ACGT"[(int(np.nanargmax(U1)) // int(POW[p])) % 4] for p in range(9))
assert U1X == "CAGGTGAGT"
SEED9 = np.array([NT.index(c) for c in U1X], np.uint8)
log(f"[seed] the strongest U1 duplex of all 262,144 9-mers is {U1X}, -dG {u1dg(SEED9):.2f}")

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3) & (y == 0))
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a):
    return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def fwd(a):
    out = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), A.bs):
            out[k:k+A.bs] = net(oh(a[k:k+A.bs])).float().cpu().numpy()
    return out
def txt(t): return "".join(NT[i] for i in np.asarray(t))

def run(rep, Lk, seed_seq, pinned, tag):
    r = np.random.default_rng([A.seed, sum(map(ord, tag)), rep])
    B = X[r.choice(te, A.nbg, replace=False)]
    base = fwd(B); s0 = A.pos + UP - Lk // 2
    free = [p for p in range(Lk) if p not in pinned]
    def score(cands):
        M = np.repeat(B[None], len(cands), 0).reshape(len(cands), A.nbg, WIN)
        for j, c in enumerate(cands): M[j, :, s0:s0+Lk] = c
        return (fwd(M.reshape(-1, WIN)).reshape(len(cands), A.nbg) - base[None]).mean(1)
    s = (seed_seq.copy() if seed_seq is not None
         else r.integers(0, 4, Lk).astype(np.uint8))
    for p, b in pinned.items(): s[p] = b
    cur = float(score([s])[0]); tr = [(0, txt(s), cur)]
    for t in range(1, A.steps + 1):
        cands = []
        for p in free:
            for b in range(4):
                c = s.copy(); c[p] = b; cands.append(c)
        v = score(cands); j = int(v.argmax())
        if v[j] <= cur + 1e-9: break
        s, cur = cands[j], float(v[j]); tr.append((t, txt(s), cur))
    return s, cur, tr

ARMS = [("rand23", 23, None, {}, "random 23-mer start"),
        ("rand9",   9, None, {}, "random 9-mer start"),
        ("u1seed",  9, SEED9, {}, "seeded from CAGGTGAGT, unconstrained"),
        ("u1gt",    9, SEED9, {3: 2, 4: 3}, "seeded from CAGGTGAGT, GT pinned at intron 1-2")]
TR = []; FIN = []; t0 = time.time()
for nm, Lk, sd, pin, desc in ARMS:
    log(f"\n[arm] {nm}: {desc}, {A.nrep} replicates, each on its own draw of {A.nbg} backgrounds")
    for rep in range(A.nrep):
        s, cur, tr = run(rep, Lk, sd, pin, nm)
        for st, sq, v in tr:
            TR.append((nm, rep, st, sq, v, u1dg([NT.index(c) for c in sq]) if Lk == 9 else np.nan,
                       int((np.frombuffer(sq.encode(), np.uint8) !=
                            np.frombuffer(U1X.encode(), np.uint8)).sum()) if Lk == 9 else -1))
        FIN.append((nm, rep, txt(s), cur, len(tr) - 1))
    d = np.array([f[3] for f in FIN if f[0] == nm])
    log(f"[arm] {nm}: gain {d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.4f}, "
        f"range {d.min():+.4f} to {d.max():+.4f}, {time.time()-t0:.0f}s")

with open("evolve_seeded_traj.tsv", "w") as fh:
    fh.write("arm\trep\tstep\tsequence\tdelta_logit\tu1_dg\thamming_to_u1\n")
    for r in TR:
        fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]:.5f}\t"
                 f"{'' if not np.isfinite(r[5]) else f'{r[5]:.4f}'}\t{r[6]}\n")
with open("evolve_seeded_final.tsv", "w") as fh:
    fh.write("arm\trep\tsequence\tdelta_logit\tn_steps\tu1_dg\thamming_to_u1\n")
    for nm, rep, sq, v, ns in FIN:
        nine = len(sq) == 9
        fh.write(f"{nm}\t{rep}\t{sq}\t{v:.5f}\t{ns}\t"
                 f"{f'{u1dg([NT.index(c) for c in sq]):.4f}' if nine else ''}\t"
                 f"{sum(a!=b for a,b in zip(sq,U1X)) if nine else ''}\n")
with open("evolve_seeded_pwm.tsv", "w") as fh:
    fh.write("arm\toffset\tA\tC\tG\tT\tic\n")
    for nm, Lk, _, _, _ in ARMS:
        S = np.array([[NT.index(c) for c in f[2]] for f in FIN if f[0] == nm])
        p = np.stack([(S == i).mean(0) for i in range(4)])
        ic = 2.0 + np.where(p > 0, p * np.log2(np.clip(p, 1e-12, None)), 0).sum(0)
        for q in range(Lk):
            fh.write(f"{nm}\t{q}\t" + "\t".join(f"{p[i,q]:.4f}" for i in range(4)) + f"\t{ic[q]:.4f}\n")
log("\n[out] evolve_seeded_traj.tsv evolve_seeded_final.tsv evolve_seeded_pwm.tsv")

log("\n[U1] what happens to the splice site when the model is allowed to rewrite it")
for nm, desc in (("u1seed", "unconstrained"), ("u1gt", "GT pinned")):
    F = [f for f in FIN if f[0] == nm]
    dg = np.array([u1dg([NT.index(c) for c in f[2]]) for f in F])
    hm = np.array([sum(a != b for a, b in zip(f[2], U1X)) for f in F])
    v = np.array([f[3] for f in F])
    st = np.array([f[4] for f in F])
    z0 = np.array([t[4] for t in TR if t[0] == nm and t[2] == 0])
    log(f"  {nm:>7} ({desc}): starts {z0.mean():+.4f} at position {A.pos} and ends "
        f"{v.mean():+.4f}, after {st.mean():.1f} substitutions of 9")
    log(f"          U1 duplex -dG falls {u1dg(SEED9):.2f} -> {dg.mean():.2f} "
        f"(range {dg.min():.2f} to {dg.max():.2f}); "
        f"{hm.mean():.1f} of 9 positions changed")
    uq = {}
    for f in F: uq[f[2]] = uq.get(f[2], 0) + 1
    log(f"          {len(uq)} distinct endpoints; most common: " +
        ", ".join(f"{k} x{v_}" for k, v_ in sorted(uq.items(), key=lambda x: -x[1])[:4]))
for nm in ("rand23", "rand9"):
    F = [f for f in FIN if f[0] == nm]
    uq = {}
    for f in F: uq[f[2]] = uq.get(f[2], 0) + 1
    v = np.array([f[3] for f in F])
    log(f"\n[pwm] {nm}: {len(uq)} distinct endpoints over {len(F)} replicates, "
        f"gain {v.mean():+.4f}; most common: " +
        ", ".join(f"{k} x{c}" for k, c in sorted(uq.items(), key=lambda x: -x[1])[:3]))
    S = np.array([[NT.index(c) for c in f[2]] for f in F])
    p = np.stack([(S == i).mean(0) for i in range(4)])
    ic = 2.0 + np.where(p > 0, p * np.log2(np.clip(p, 1e-12, None)), 0).sum(0)
    log(f"       consensus {''.join(NT[i] for i in p.argmax(0))}  "
        f"mean IC {ic.mean():.2f} bits of 2, max {ic.max():.2f}")
