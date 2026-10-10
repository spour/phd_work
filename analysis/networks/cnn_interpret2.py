#!/usr/bin/env python3
# interpretability of the retrained cnn, at the retrained logit scale
import sys, os, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from cnn_interp_lib import (NT, REF_MOTIFS, consensus_pwm, pwm_match, match_threshold,
                            write_meme, dinuc_shuffle)

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model_v2.pt")
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--legacy", action="store_true")
P.add_argument("--n_seq", type=int, default=400)
P.add_argument("--n_ism", type=int, default=64)
P.add_argument("--n_shuf", type=int, default=3)
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--ig_steps", type=int, default=32)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--suffix", default="_retrained")
A = P.parse_args()
rng = np.random.default_rng(A.seed); torch.manual_seed(A.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
SF = A.suffix

ck = torch.load(A.model, map_location=dev, weights_only=False)
if A.legacy:
    from cnn_model_def import Net
    net = Net(ck["args"]["dropout"]).to(dev); tag = "legacy_exp"
else:
    from cnn_model_def2 import Net2
    c = ck["cfg"]; net = Net2(ck["args"]["dropout"], c["variant"], c["bn_head"], c["logscale"]).to(dev)
    tag = c["tag"]
net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
log(f"model {A.model}  variant {tag}  epoch {ck['epoch']}  T_only {ck['temperature']:.4g}"
    + (f"  Platt scale {ck['platt'][0]:.4f} bias {ck['platt'][1]:+.4f}" if "platt" in ck else ""))

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; split = D["split"][ok]; UP = int(D["up"])
te = np.flatnonzero(split == "test")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def onehot(a): return TBLt[torch.from_numpy(a.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def fwd(oh, bs=256):
    out = []
    with torch.no_grad():
        for k in range(0, len(oh), bs): out.append(net(oh[k:k+bs]).float().cpu())
    return torch.cat(out).numpy()

W = 19
def filter_pwms(n_seq=4000, frac=0.3):
    sub = rng.choice(te, min(n_seq, len(te)), replace=False)
    counts = np.zeros((32, W, 4)); nsite = np.zeros(32, int); maxact = np.zeros(32)
    for k in range(0, len(sub), 128):
        with torch.no_grad(): act = net.first(onehot(X[sub[k:k+128]])).cpu().numpy()
        maxact = np.maximum(maxact, act.max((0, 2)))
    thr = frac * maxact
    for k in range(0, len(sub), 128):
        idx = sub[k:k+128]
        with torch.no_grad(): act = net.first(onehot(X[idx])).cpu().numpy()
        for f in range(32):
            b, p = np.where(act[:, f, :] >= thr[f])
            for bi, pi in zip(b, p):
                s = pi - W // 2
                if s < 0 or s + W > X.shape[1]: continue
                seq = X[idx[bi], s:s + W]; g = seq < 4
                counts[f, np.arange(W)[g], seq[g]] += 1; nsite[f] += 1
    pwm = np.where(counts.sum(2, keepdims=True) > 0,
                   counts / np.maximum(counts.sum(2, keepdims=True), 1), 0.25)
    ic = np.nansum(pwm * np.log2(np.maximum(pwm, 1e-9) / 0.25), (1, 2))
    return pwm, nsite, maxact, ic

def saliency(oh):
    oh = oh.clone().requires_grad_(True)
    net.zero_grad(); net(oh).sum().backward()
    return oh.grad.detach()
def integrated_gradients(oh, steps):
    base = torch.full_like(oh, 0.25); tot = torch.zeros_like(oh)
    for a in torch.linspace(1.0 / steps, 1.0, steps, device=oh.device):
        x = (base + a * (oh - base)).requires_grad_(True)
        net.zero_grad(); net(x).sum().backward()
        tot += x.grad.detach()
    return (oh - base) * tot / steps

_DL = None
try:
    from captum.attr import DeepLift
    _DL = DeepLift(net)
    log("[attr] Captum DeepLift available -- CAVEAT: every non-linearity in this network is a "
        "FUNCTIONAL call (torch.exp / F.softplus / F.relu), not an nn.Module, so Captum cannot "
        "attach its rescale rule to any of them. DeepLift here degenerates to gradient x input "
        "and is NOT an independent method; ISM is the one that is.")
except Exception as e:
    log(f"[attr] Captum DeepLift unavailable ({type(e).__name__}: {e}); skipped")

def deeplift(oh):
    return _DL.attribute(oh.clone().requires_grad_(True),
                         baselines=torch.full_like(oh, 0.25)).detach()

def attribution_raw(seqs, method, bs=32):
    outs = []
    for k in range(0, len(seqs), bs):
        oh = onehot(seqs[k:k+bs])
        if method == "saliency": g = saliency(oh)
        elif method == "ig":     g = integrated_gradients(oh, A.ig_steps)
        else:                    g = deeplift(oh)
        outs.append((g * oh).sum(1).cpu().numpy())
    return np.concatenate(outs)

def ism(seqs, bs=1024):
    L = seqs.shape[1]; out = np.zeros((len(seqs), L), np.float32)
    wt = fwd(onehot(seqs))
    for i in range(len(seqs)):
        s = seqs[i]; d = np.zeros(L, np.float64); n = np.zeros(L)
        variants = []; where = []
        for b in range(4):
            m = (s != b) & (s < 4)
            pos = np.flatnonzero(m)
            for p in pos: variants.append((p, b))
        V = np.array(variants)
        for k in range(0, len(V), bs):
            chunk = V[k:k+bs]
            Q = np.repeat(s[None, :], len(chunk), 0)
            Q[np.arange(len(chunk)), chunk[:, 0]] = chunk[:, 1].astype(np.uint8)
            lg = fwd(onehot(Q), bs=bs)
            np.add.at(d, chunk[:, 0], lg - wt[i]); np.add.at(n, chunk[:, 0], 1.0)
        out[i] = np.where(n > 0, d / np.maximum(n, 1), 0.0)
    return out

STATS = (("mean_abs", lambda a: np.abs(a).mean()),
         ("p99.9", lambda a: np.quantile(np.abs(a), 0.999)),
         ("max_per_seq", lambda a: np.abs(a).max(1).mean()),
         ("top20_per_seq", lambda a: np.sort(np.abs(a), 1)[:, -20:].mean()))

GIA_MOTIFS = {"AATAAA": "AATAAA", "ATTAAA": "ATTAAA",
              "acceptor_pyAG": "TTTTTTTTTTTTTCAGG", "donor_GT": "CAGGTAAGT",
              "control_shuffled_AATAAA": "ATAAAT"}
def implant(seqs, motif, pos):
    m = np.array([NT.index(c) for c in motif], np.uint8)
    out = seqs.copy(); s = pos - len(m) // 2
    if s < 0 or s + len(m) > out.shape[1]: return None
    out[:, s:s+len(m)] = m
    return out

if __name__ == "__main__":
    t0 = time.time()
    # filters
    pwm, nsite, maxact, ic = filter_pwms()
    write_meme(pwm, nsite, f"filter_pwms{SF}.meme")
    refs = {k: consensus_pwm(v) for k, v in REF_MOTIFS.items()}
    THR = match_threshold(refs, rng)
    log("[filters] random-PWM null, 95th pct of r per reference: "
        + "  ".join(f"{k} {v:.2f}" for k, v in THR.items()))
    hits = {}
    with open(f"filter_matches{SF}.tsv", "w") as fh:
        fh.write("filter\tn_sites\tmax_activation\tinfo_content_bits\tbest_match\tr\tstrand\tconsensus\n")
        for f in range(len(pwm)):
            cands = [(pwm_match(pwm[f], q)[0], pwm_match(pwm[f], q)[1], k) for k, q in refs.items()]
            best = max(cands)
            cons = "".join(NT[i] for i in pwm[f].argmax(1))
            passes = best[0] >= THR.get(best[2], 1.0)
            fh.write(f"filter{f:02d}\t{nsite[f]}\t{maxact[f]:.3f}\t{ic[f]:.2f}\t"
                     f"{best[2] if passes else 'none'}\t{best[0]:.3f}\t{best[1]}\t{cons}\n")
            if passes: hits.setdefault(best[2], []).append((f, best[0], ic[f], cons))
    log("[filters] beating the random-PWM null: " +
        ("  ".join(f"{k} x{len(v)}" for k, v in hits.items()) if hits else "NONE"))
    for k, v in hits.items():
        for f, r, i_, cons in v:
            log(f"[filters]   filter{f:02d} -> {k:20s} r={r:.3f}  IC {i_:.1f} bits  "
                f"n {nsite[f]}  {cons}")
    log(f"[filters] IC over 19 positions: mean {ic.mean():.2f}  min {ic.min():.2f}  "
        f"max {ic.max():.2f}  >=5 bits {int((ic>=5).sum())}/32")

    # attribution + ism vs dinuc null
    sub = rng.choice(te, min(A.n_seq, len(te)), replace=False)
    real = X[sub]
    shuf = np.stack([dinuc_shuffle(s, rng) for s in np.repeat(real, A.n_shuf, 0)])
    log(f"[null] {len(shuf)} dinucleotide shuffles of {len(real)} test sequences "
        f"(GC real {float(((real==1)|(real==2)).mean()):.4f} vs shuffled "
        f"{float(((shuf==1)|(shuf==2)).mean()):.4f})")
    methods = ["saliency", "ig"] + (["deeplift"] if _DL is not None else [])
    rows = []
    for m in methods:
        ar = attribution_raw(real, m); an = attribution_raw(shuf, m)
        for nm, fn in STATS:
            r_, n_ = float(fn(ar)), float(fn(an))
            rows.append((m, nm, r_, n_, len(real), len(shuf)))
            log(f"[null] {m:9s} {nm:14s} real {r_:.4g}  shuffled {n_:.4g}  ratio {r_/n_:.3f}")
        np.savez_compressed(f"attr_{m}{SF}.npz", real=ar.astype(np.float32), idx=sub)

    ri = rng.choice(te, A.n_ism, replace=False)
    r_ism = X[ri]; s_ism = np.stack([dinuc_shuffle(s, rng) for s in r_ism])
    log(f"[ism] {A.n_ism} real + {A.n_ism} shuffled sequences, "
        f"{3*X.shape[1]:,} single-base substitutions each")
    ar = ism(r_ism); an = ism(s_ism)
    for nm, fn in STATS:
        r_, n_ = float(fn(ar)), float(fn(an))
        rows.append(("ism", nm, r_, n_, len(r_ism), len(s_ism)))
        log(f"[null] {'ism':9s} {nm:14s} real {r_:.4g}  shuffled {n_:.4g}  ratio {r_/n_:.3f}")
    np.savez_compressed(f"attr_ism{SF}.npz", real=ar.astype(np.float32),
                        shuffled=an.astype(np.float32), idx=ri)
    nb = X.shape[1] // 100
    with open(f"ism_profile{SF}.tsv", "w") as fh:
        fh.write("offset\treal\tshuffled\tratio\n")
        pr = np.abs(ar[:, :nb*100]).reshape(len(ar), nb, 100).mean((0, 2))
        pn = np.abs(an[:, :nb*100]).reshape(len(an), nb, 100).mean((0, 2))
        for j in range(nb):
            fh.write(f"{-UP+j*100+50}\t{pr[j]:.6g}\t{pn[j]:.6g}\t{pr[j]/max(pn[j],1e-12):.4f}\n")
    with open(f"attribution_over_null{SF}.tsv", "w") as fh:
        fh.write("method\tstatistic\treal\tshuffled\tratio\tn_real\tn_shuffled\n")
        for m, nm, r_, n_, a_, b_ in rows:
            fh.write(f"{m}\t{nm}\t{r_:.6g}\t{n_:.6g}\t{r_/n_:.4f}\t{a_}\t{b_}\n")
    best_ratio = max(r_ / n_ for _, _, r_, n_, _, _ in rows)
    log(f"[null] LARGEST real/shuffled ratio over every method and statistic: {best_ratio:.3f}")

    # gia
    bg = np.stack([dinuc_shuffle(X[i], rng) for i in rng.choice(te, A.gia_bg, replace=False)])
    bg_real = X[rng.choice(te[y[te] == 0], A.gia_bg, replace=False)]
    positions = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
    with open(f"gia{SF}.tsv", "w") as fh:
        fh.write("background\tmotif\tposition\tdelta_logit\tse\tn\n")
        for bgname, B in (("dinuc_shuffled", bg), ("real_competitor", bg_real)):
            base = fwd(onehot(B))
            log(f"[GIA] {bgname} background logit mean {base.mean():+.3f} sd {base.std():.3f}")
            for name, mo in GIA_MOTIFS.items():
                best = (None, -1e9)
                for pos in positions:
                    seqs = implant(B, mo, pos + UP)
                    if seqs is None: continue
                    d = fwd(onehot(seqs)) - base
                    fh.write(f"{bgname}\t{name}\t{pos}\t{d.mean():.5f}\t"
                             f"{d.std(ddof=1)/np.sqrt(len(d)):.5f}\t{len(d)}\n")
                    if d.mean() > best[1]: best = (pos, d.mean())
                log(f"[GIA] {bgname:16s} {name:24s} max delta-logit {best[1]:+.3f} at {best[0]:+d} nt")
    log(f"[out] filter_pwms{SF}.meme filter_matches{SF}.tsv attribution_over_null{SF}.tsv "
        f"ism_profile{SF}.tsv gia{SF}.tsv  ({time.time()-t0:.0f}s)")
