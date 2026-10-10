#!/usr/bin/env python3
# de novo motif discovery on the evolved sequences
import sys, argparse
import numpy as np
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--widths", default="6,7,8,9,10")
P.add_argument("--nseed", type=int, default=8)
P.add_argument("--nnull", type=int, default=5)
P.add_argument("--iters", type=int, default=150)
P.add_argument("--tol", type=float, default=0.05)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
WID = [int(x) for x in A.widths.split(",")]

def load(path, arm):
    import csv
    return np.array([[NT.index(c) for c in r["sequence"]]
                     for r in csv.DictReader(open(path), delimiter="\t") if r["arm"] == arm])

def tcm(S, w, P0, bg, iters):
    W = np.lib.stride_tricks.sliding_window_view(S, w, axis=1)
    jj = np.arange(w)[None, None, :]
    bgll = np.log(bg)[W].sum(2)
    Pm, lam = P0.copy(), 0.02
    for _ in range(iters):
        lp = np.log(np.clip(Pm, 1e-9, None))
        a = np.log(lam) + lp[W, jj].sum(2); b = np.log1p(-lam) + bgll
        m = np.maximum(a, b)
        z = np.exp(a - m) / (np.exp(a - m) + np.exp(b - m))
        zf = z.ravel()
        cnt = np.stack([np.bincount(W[:, :, j].ravel(), weights=zf, minlength=4)
                        for j in range(w)], 1)
        Pn = (cnt + 0.25) / (cnt.sum(0) + 1.0)
        ln = float(z.mean())
        done = np.abs(Pn - Pm).max() < 1e-5 and abs(ln - lam) < 1e-7
        Pm, lam = Pn, ln
        if done: break
    re = float((Pm * (np.log2(np.clip(Pm, 1e-12, None)) - np.log2(bg)[:, None])).sum())
    return Pm, lam, re

def topk(S, w, n):
    r = np.random.default_rng(A.seed)
    def cnt(T):
        c = np.zeros(4 ** w, np.int64); p4 = 4 ** np.arange(w - 1, -1, -1)
        for x in T:
            np.add.at(c, np.lib.stride_tricks.sliding_window_view(x.astype(np.int64), w) @ p4, 1)
        return c
    obs = cnt(S); nul = cnt(np.array([r.permutation(x) for x in S]))
    sc = (obs + 1) / (nul + 1)
    return [np.array([(i // 4 ** (w - 1 - j)) % 4 for j in range(w)])
            for i in np.argsort(-sc * (obs >= 5))[:n]]

def seedpwm(k, w):
    p = np.full((4, w), 0.08)
    for j in range(w): p[k[j], j] = 0.76
    return p / p.sum(0, keepdims=True)

def discover(S, tag):
    bg = np.array([(S == i).mean() for i in range(4)])
    log(f"\n[{tag}] {S.shape[0]} sequences x {S.shape[1]} nt; background composition "
        + " ".join(f"{NT[i]} {bg[i]:.3f}" for i in range(4)))
    rows = []; best = {}
    r = np.random.default_rng(A.seed)
    SH = [np.array([r.permutation(x) for x in S]) for _ in range(A.nnull)]
    for w in WID:
        cand = [tcm(S, w, seedpwm(k, w), bg, A.iters) for k in topk(S, w, A.nseed)]
        Pm, lam, re = max(cand, key=lambda t: t[2])
        nul = [tcm(T, w, seedpwm(topk(T, w, 1)[0], w), bg, A.iters)[2] for T in SH]
        best[w] = (Pm, lam, re)
        rows.append((w, re, float(np.mean(nul)), float(np.std(nul, ddof=1)), lam))
        log(f"[{tag}] w={w:2d}  relative entropy {re:6.2f} bits   shuffled null "
            f"{np.mean(nul):5.2f} +/- {np.std(nul,ddof=1):.2f}   "
            f"consensus {''.join(NT[i] for i in Pm.argmax(0))}   occurrence weight {lam:.4f}")
    mx = max(r_[1] for r_ in rows)
    W = min(r_[0] for r_ in rows if r_[1] >= (1 - A.tol) * mx)
    Pm, lam, re = best[W]
    ic = 2.0 + np.where(Pm > 0, Pm * np.log2(np.clip(Pm, 1e-12, None)), 0).sum(0)
    log(f"[{tag}] CHOSEN w={W} (smallest within {A.tol:.0%} of the best total relative entropy "
        f"{mx:.2f}); consensus {''.join(NT[i] for i in Pm.argmax(0))}, "
        f"{re:.2f} bits against a shuffled null of {rows[WID.index(W)][2]:.2f}, "
        f"per-column information content up to {ic.max():.2f} of 2, mean {ic.mean():.2f}")
    return W, Pm, lam, re, ic, rows

UP = load("evolve_seqs.tsv", "site")
DN = load("evolve_min_seqs.tsv", "site")
OUT = {}
for tag, S in (("activating", UP), ("repressive", DN)):
    OUT[tag] = discover(S, tag)
with open("evolved_motif_pwm.tsv", "w") as fh:
    fh.write("set\twidth\toffset\tA\tC\tG\tT\tic\n")
    for tag, (W, Pm, lam, re, ic, rows) in OUT.items():
        for j in range(W):
            fh.write(f"{tag}\t{W}\t{j}\t" + "\t".join(f"{Pm[i,j]:.4f}" for i in range(4))
                     + f"\t{ic[j]:.4f}\n")
with open("evolved_motif_fit.tsv", "w") as fh:
    fh.write("set\twidth\trelative_entropy\tnull_mean\tnull_sd\tocc_weight\tchosen\n")
    for tag, (W, Pm, lam, re, ic, rows) in OUT.items():
        for w, r_, nm, ns, lm in rows:
            fh.write(f"{tag}\t{w}\t{r_:.4f}\t{nm:.4f}\t{ns:.4f}\t{lm:.6f}\t{int(w==W)}\n")
log("\n[out] evolved_motif_pwm.tsv evolved_motif_fit.tsv")
