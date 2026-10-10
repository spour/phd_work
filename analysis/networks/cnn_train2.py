#!/usr/bin/env python3
# retrain sweep for the first-layer activation / regularisation
import sys, os, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)

P = argparse.ArgumentParser()
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
P.add_argument("--out", default="cnn_model_v2")
P.add_argument("--only", default=None,
               help="restrict the sweep to configs whose tag contains this string")
A = P.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device {dev}  torch {torch.__version__}  cuda {torch.version.cuda}")

D = np.load(A.data, allow_pickle=True)
ok = D["ok"]; X = D["X"][ok]; y = D["y"][ok].astype(np.float32)
gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
log("  " + "  ".join(f"{s} {len(idx[s])} ({int(y[idx[s]].sum())} real)" for s in idx))

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBL = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
def batch_onehot(ii): return TBL[Xt[ii].to(dev)].permute(0, 2, 1).contiguous()

npos = float(y[idx["train"]].sum()); nneg = len(idx["train"]) - npos
POSW = nneg / npos
log(f"pos_weight {POSW:.2f}")

def within_gene(s, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r, c = by.setdefault(g[k], ([], []))
        (r if yy[k] == 1 else c).append(s[k])
    v = []
    for r, c in by.values():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        v.append(((r[:, None] > c[None, :]).sum() + 0.5 * (r[:, None] == c[None, :]).sum())
                 / (len(r) * len(c)))
    return float(np.mean(v)), len(v)

def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt) + cs + 1) / 2.0
    return float((rk[inv][:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

# calibration, both kinds, inline (lbfgs pulls in a broken sympy here)
def bce_np(z, yy): return float(np.mean(np.logaddexp(0, np.clip(z, -60, 60)) - yy * z))

def fit_temperature(s, yy):
    lo, hi = -3.0, 12.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if bce_np(s / np.exp(m1), yy) < bce_np(s / np.exp(m2), yy): hi = m2
        else: lo = m1
    return float(np.exp((lo + hi) / 2))

def fit_platt(s, yy):
    Xd = np.column_stack([s, np.ones(len(s))]); b = np.array([0.5, -1.0]); cur = bce_np(Xd @ b, yy)
    for _ in range(200):
        z = np.clip(Xd @ b, -30, 30); p = 1 / (1 + np.exp(-z)); W = np.maximum(p * (1 - p), 1e-6)
        g = Xd.T @ (p - yy) / len(yy); H = (Xd * W[:, None]).T @ Xd / len(yy) + 1e-8 * np.eye(2)
        d = np.linalg.solve(H, g); t = 1.0
        while t > 1e-10 and bce_np(Xd @ (b - t * d), yy) > cur - 1e-12: t *= 0.5
        b = b - t * d; new = bce_np(Xd @ b, yy)
        if cur - new < 1e-12: break
        cur = new
    return float(b[0]), float(b[1]), cur

# one run
from cnn_model_def2 import Net2

def run(cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    net = Net2(A.dropout, cfg["variant"], cfg["bn_head"], cfg["logscale"]).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POSW, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=cfg["wd"])
    l1 = cfg.get("l1_act", 0.0)

    def scores(ii, bs=512):
        net.eval(); out = np.empty(len(ii), np.float32)
        with torch.no_grad():
            for k in range(0, len(ii), bs):
                out[k:k+bs] = net(batch_onehot(ii[k:k+bs])).float().cpu().numpy()
        return out

    best = (-1, -1, None); tr = idx["train"]
    for ep in range(1, A.epochs + 1):
        net.train(); perm = np.random.permutation(tr)
        for k in range(0, len(perm), A.batch):
            j = perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            o, act = net(batch_onehot(j), want_act=True)
            loss = crit(o, yt[j].to(dev))
            if l1: loss = loss + l1 * act.abs().mean()
            loss.backward(); opt.step()
        sv = scores(idx["val"]); wg, _ = within_gene(sv, idx["val"])
        if wg > best[0]:
            best = (wg, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        elif ep - best[1] >= A.patience: break
    net.load_state_dict(best[2])

    sv = scores(idx["val"]).astype(np.float64); yv = y[idx["val"]].astype(np.float64)
    T = fit_temperature(sv, yv); pa, pb, pl = fit_platt(sv, yv)
    st = scores(idx["test"]).astype(np.float64)
    wg_t, ng = within_gene(st, idx["test"])
    res = dict(seed=seed, epoch=best[1], val_wg=best[0], test_wg=wg_t, n_genes=ng,
               T_only=T, platt_scale=pa, platt_bias=pb, platt_bce=pl,
               logit_sd=float(st.std()), logit_min=float(st.min()), logit_max=float(st.max()),
               test_auroc=auroc(st[y[idx["test"]] == 1], st[y[idx["test"]] == 0]),
               act_max=float(np.abs(np.concatenate([sv, st])).max()))
    with torch.no_grad():
        oh = batch_onehot(idx["test"][:256]); a = net.first(oh)
        res["first_act_max"] = float(a.max()); res["first_act_mean"] = float(a.mean())
    return res, net, best[1]

CFGS = [
    dict(tag="V0_reference",   variant="exp",      bn_head=False, logscale=False, wd=1e-6),
    dict(tag="V1_exp_wd1e-4",  variant="exp",      bn_head=True,  logscale=True,  wd=1e-4),
    dict(tag="V1_exp_wd3e-4",  variant="exp",      bn_head=True,  logscale=True,  wd=3e-4),
    dict(tag="V1_exp_wd1e-3",  variant="exp",      bn_head=True,  logscale=True,  wd=1e-3),
    dict(tag="V2_sp_l1_1e-6",  variant="softplus", bn_head=True,  logscale=False, wd=1e-4, l1_act=1e-6),
    dict(tag="V2_sp_l1_1e-5",  variant="softplus", bn_head=True,  logscale=False, wd=1e-4, l1_act=1e-5),
    dict(tag="V2_sp_l1_1e-4",  variant="softplus", bn_head=True,  logscale=False, wd=1e-4, l1_act=1e-4),
]

if __name__ == "__main__":
    if A.only: CFGS = [c for c in CFGS if A.only in c["tag"]]
    t0 = time.time(); rows = []; keep = {}
    for cfg in CFGS:
        rs = []
        for sd in A.seeds:
            r, net, ep = run(cfg, sd); r["tag"] = cfg["tag"]; rs.append(r); rows.append(r)
            keep[(cfg["tag"], sd)] = ({k: v.clone() for k, v in net.state_dict().items()}, ep)
            log(f"  {cfg['tag']:16s} seed {sd}  ep {ep:2d}  val_wg {r['val_wg']:.4f}  "
                f"test_wg {r['test_wg']:.4f}  T_only {r['T_only']:.3g}  "
                f"Platt scale {r['platt_scale']:.3f} bias {r['platt_bias']:+.3f}  "
                f"logit sd {r['logit_sd']:.2f}  first-layer act max {r['first_act_max']:.3g}")
        v = np.array([x["val_wg"] for x in rs]); t = np.array([x["test_wg"] for x in rs])
        log(f"[{cfg['tag']:16s}] val {v.mean():.4f} +/- {v.std():.4f}   "
            f"test {t.mean():.4f} +/- {t.std():.4f}   "
            f"T_only {np.median([x['T_only'] for x in rs]):.4g}   "
            f"Platt T {1/np.median([x['platt_scale'] for x in rs]):.3f}\n")

    with open("cnn_retrain_sweep.tsv", "w") as fh:
        ks = ["tag","seed","epoch","val_wg","test_wg","test_auroc","T_only","platt_scale",
              "platt_bias","platt_bce","logit_sd","logit_min","logit_max","first_act_max",
              "first_act_mean","n_genes"]
        fh.write("\t".join(ks) + "\n")
        for r in rows: fh.write("\t".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")

    # selection: best mean val within-gene among the non-reference variants
    by = {}
    for r in rows: by.setdefault(r["tag"], []).append(r)
    ref = float(np.mean([x["val_wg"] for x in by["V0_reference"]])) if "V0_reference" in by else float("nan")
    cand = {k: float(np.mean([x["val_wg"] for x in v])) for k, v in by.items() if k != "V0_reference"} \
           or {k: float(np.mean([x["val_wg"] for x in v])) for k, v in by.items()}
    win = max(cand, key=cand.get)
    log(f"[select] on VAL within-gene: {win} ({cand[win]:.4f}) vs reference V0 ({ref:.4f}); "
        f"candidates " + "  ".join(f"{k} {v:.4f}" for k, v in sorted(cand.items(), key=lambda z: -z[1])))
    bs = max(by[win], key=lambda r: r["val_wg"])["seed"]
    state, ep = keep[(win, bs)]
    cfg = [c for c in CFGS if c["tag"] == win][0]
    net = Net2(A.dropout, cfg["variant"], cfg["bn_head"], cfg["logscale"]).to(dev)
    net.load_state_dict(state); net.eval()

    def sc_all(bs_=512):
        out = np.empty(len(y), np.float32); ii = np.arange(len(y))
        with torch.no_grad():
            for k in range(0, len(ii), bs_):
                out[k:k+bs_] = net(batch_onehot(ii[k:k+bs_])).float().cpu().numpy()
        return out
    s_all = sc_all().astype(np.float64)
    sv = s_all[idx["val"]]; yv = y[idx["val"]].astype(np.float64)
    T = fit_temperature(sv, yv); pa, pb, _ = fit_platt(sv, yv)
    np.savez_compressed(A.out + "_scores.npz", logit=s_all, logit_cal=s_all * pa + pb,
                        y=y, gene=gene, split=split, temperature=T,
                        platt_scale=pa, platt_bias=pb, tag=win, seed=bs)
    torch.save(dict(state=net.state_dict(), args=vars(A), cfg=cfg, temperature=T,
                    platt=(pa, pb), epoch=ep, seed=bs), A.out + ".pt")
    log(f"[chosen] {win} seed {bs} epoch {ep}: T_only {T:.4g}, Platt scale {pa:.4f} "
        f"(T={1/pa:.3f}) bias {pb:+.4f}")
    log(f"[out] cnn_retrain_sweep.tsv {A.out}.pt {A.out}_scores.npz  ({time.time()-t0:.0f}s)")
