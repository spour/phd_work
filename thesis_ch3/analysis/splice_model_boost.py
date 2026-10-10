#!/usr/bin/env python3
# can the 26-feature splice model be improved with more splice features
import os, sys, math, collections, numpy as np, torch
torch.set_num_threads(6)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
F = np.load(os.path.join(ROOT, "data/splice_positional.npz")); D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
ok = D["ok"]; y = D["y"][ok].astype(int); gene = np.array([str(x) for x in D["gene"][ok]]); chrom = np.array([str(x) for x in D["chrom"][ok]])
LAM = float(np.load(os.path.join(PUB, "splice_model_26.npz"), allow_pickle=True)["lam"]); N = len(y)
gi = collections.defaultdict(list)
for i, g in enumerate(gene): gi[g].append(i)
GL = [np.array(v) for v in gi.values()]; gof = np.zeros(N, int)
for k, ii in enumerate(GL): gof[ii] = k
acc = F["acc"][:, None].astype(float)
SETS = collections.OrderedDict([
    ("26 features: U1 sites (25 bins) + upstream acceptor", ["u1_4", "acc"]),
    ("U1 sites in 50 nt bins (50) + acceptor", ["u1_4_50", "acc"]),
    ("+ strong U1 sites and mean U1 score per bin", ["u1_4", "u1_7", "u1_mean", "acc"]),
    ("+ MaxEnt 5' splice sites per bin", ["u1_4", "d5_8", "d5_max", "acc"]),
    ("+ MaxEnt 3' splice sites per bin", ["u1_4", "a3_max", "a3_65", "acc"]),
    ("all splice features (176)", ["u1_4", "u1_7", "u1_mean", "d5_8", "d5_max", "a3_max", "a3_65", "acc"])])
def mat(keys): return np.column_stack([acc if k == "acc" else F[k].astype(float) for k in keys])
def centre(M):
    M = M.copy()
    for ii in GL: M[ii] -= M[ii].mean(0, keepdims=True)
    return M
def irls(Xd, yy, lam, it=40):
    Z = np.column_stack([np.ones(len(Xd)), Xd]); bt = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(Z @ bt, -30, 30))); w = np.clip(p * (1 - p), 1e-6, None); z = Z @ bt + (yy - p) / w
        R = np.eye(Z.shape[1]) * lam; R[0, 0] = 0; bt = np.linalg.solve(Z.T @ (Z * w[:, None]) + R, Z.T @ (w * z))
    return bt
def per_gene(s, genes=None):
    w, p = [], []
    for ii in (GL if genes is None else genes):
        t = s[ii][y[ii] == 1][0]; o = s[ii][y[ii] == 0]; w.append((t > o).mean() + 0.5 * (t == o).mean()); p.append(float((o < t).all()))
    return np.array(w), np.array(p)
chroms = sorted(set(chrom)); rng0 = np.random.default_rng(0)
FOLD = []
for c in chroms:
    te = chrom == c; trg = np.unique(gof[~te]); hold = rng0.permutation(trg)[: len(trg) // 4]
    va = np.isin(gof, hold) & ~te; FOLD.append((te, ~te & ~va, va, [GL[k] for k in hold]))
def run_ridge(M):
    Mc = centre(M); s = np.zeros(N)
    for te, tr, va, vg in FOLD:
        mu = Mc[tr].mean(0); sd = Mc[tr].std(0); sd[sd < 1e-9] = 1; Z = (Mc - mu) / sd; best = None
        for lam in (LAM, 10 * LAM, 100 * LAM, 1000 * LAM):
            bt = irls(Z[tr], y[tr].astype(float), lam); wv = per_gene(Z @ bt[1:], vg)[0].mean()
            if best is None or wv > best[0]: best = (wv, lam)
        bt = irls(Z[tr | va], y[tr | va].astype(float), best[1]); s[te] = Z[te] @ bt[1:]
    return s
class Lin(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.f = torch.nn.Linear(d, 1)
    def forward(s, x): return s.f(x).squeeze(-1)
class MLP(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(s, x): return s.f(x).squeeze(-1)
class Conv(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.c = (d - 1) // 25
        s.cv = torch.nn.Sequential(torch.nn.Conv1d(s.c, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv1d(32, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Dropout(0.3))
        s.f = torch.nn.Sequential(torch.nn.Linear(16 * 25 + 1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(s, x):
        h = s.cv(x[:, :-1].reshape(len(x), s.c, 25)).flatten(1); return s.f(torch.cat([h, x[:, -1:]], 1)).squeeze(-1)
def run_listwise(M, cls, seeds=(0, 1, 2)):
    Mc = centre(M); s = np.zeros(N)
    for fi, (te, tr, va, vg) in enumerate(FOLD):
        mu = Mc[tr].mean(0); sd = Mc[tr].std(0); sd[sd < 1e-9] = 1; Z = torch.tensor((Mc - mu) / sd, dtype=torch.float32)
        trg = [ii for ii in GL if tr[ii[0]]]; acc_s = np.zeros(N)
        for seed in seeds:
            torch.manual_seed(seed); r = np.random.default_rng(seed); net = cls(M.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
            best, bad, state = -1, 0, None
            for ep in range(60):
                net.train(); order = r.permutation(len(trg))
                for b in range(0, len(order), 64):
                    loss = 0
                    for k in order[b:b + 64]:
                        ii = trg[k]; o = net(Z[ii]); loss = loss - torch.log_softmax(o, 0)[int(np.flatnonzero(y[ii] == 1)[0])]
                    opt.zero_grad(); (loss / 64).backward(); opt.step()
                net.eval()
                with torch.no_grad(): sv = net(Z).numpy()
                wv = per_gene(sv, vg)[0].mean()
                if wv > best + 1e-4: best, bad, state = wv, 0, {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 6: break
            net.load_state_dict(state); net.eval()
            with torch.no_grad(): o = net(Z).numpy()
            for ii in GL:
                if te[ii[0]]: acc_s[ii] += (o[ii] - o[ii].mean()) / (o[ii].std() + 1e-9)
        s[te] = acc_s[te]
        if fi % 6 == 0: log(f"      fold {fi + 1}/23")
    return s
RNG = np.random.default_rng(1)
def ci(v): return np.percentile([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(2000)], [2.5, 97.5])
def mcnemar(a, b):
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    if n10 + n01 == 0: return 1.0
    x = (abs(n10 - n01) - 1) ** 2 / (n10 + n01); return math.erfc(math.sqrt(x / 2))
OUT = []; SC = {}; base = None
def report(name, kind, s):
    global base
    w, p = per_gene(s); lo, hi = ci(p); lw, hw = ci(w)
    if base is None: base = p
    d = p.mean() - base.mean(); pm = mcnemar(p, base) if p is not base else float("nan")
    OUT.append((name, kind, f"{p.mean():.3f}", f"[{lo:.3f}, {hi:.3f}]", f"{w.mean():.3f}", f"[{lw:.3f}, {hw:.3f}]", f"{d:+.3f}", f"{pm:.2g}"))
    log(f"  P@1 {p.mean():.3f} [{lo:.3f},{hi:.3f}]  win-rate {w.mean():.3f}  vs 26-feature {d:+.3f} (McNemar P {pm:.2g})   {kind}: {name}"); SC[f"{kind}|{name}"] = s
for name, keys in SETS.items(): report(name, "ridge", run_ridge(mat(keys)))
allk = SETS["all splice features (176)"]
for kind, cls, keys, nm in (("listwise linear", Lin, ["u1_4", "acc"], "26 features"), ("listwise linear", Lin, allk, "all splice features (176)"),
                            ("listwise MLP", MLP, allk, "all splice features (176)"), ("listwise convolution over bins", Conv, allk, "all splice features (176)")):
    log(f"[{kind}] {nm}"); report(nm, kind, run_listwise(mat(keys), cls))
cnn = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True)["logit"].astype(float); report("sequence network (reference)", "network", cnn)
with open(os.path.join(ROOT, "data/splice_model_boost.tsv"), "w") as fh:
    fh.write("features\tmodel\tp_at_1\tci95\twin_rate\twin_rate_ci95\tdiff_vs_26_feature\tmcnemar_P\n")
    for r in OUT: fh.write("\t".join(r) + "\n")
np.savez_compressed(os.path.join(ROOT, "data/splice_boost_scores.npz"), **{k.replace("|", "__").replace(" ", "_"): v for k, v in SC.items()})
log("[out] data/splice_model_boost.tsv")
