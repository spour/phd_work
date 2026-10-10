#!/usr/bin/env python3
# what the cnn reads
import sys, os, json, argparse, time
import numpy as np
import torch, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--n_seq", type=int, default=400)
P.add_argument("--n_shuf", type=int, default=3)
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--ig_steps", type=int, default=32)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
rng = np.random.default_rng(A.seed); torch.manual_seed(A.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
NT = "ACGT"

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
log(f"model epoch {ck['epoch']}, temperature {ck['temperature']:.3f}, device {dev}")

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; split = D["split"][ok]; UP = int(D["up"])
te = np.flatnonzero(split == "test")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def onehot(idx_arr): return TBLt[torch.from_numpy(idx_arr.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
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
        oh = onehot(X[sub[k:k+128]])
        with torch.no_grad(): act = net.first(oh).cpu().numpy()
        maxact = np.maximum(maxact, act.max((0, 2)))
    thr = frac * maxact
    for k in range(0, len(sub), 128):
        idx = sub[k:k+128]; oh = onehot(X[idx])
        with torch.no_grad(): act = net.first(oh).cpu().numpy()
        for f in range(32):
            b, p = np.where(act[:, f, :] >= thr[f])
            for bi, pi in zip(b, p):
                s = pi - W // 2
                if s < 0 or s + W > X.shape[1]: continue
                seq = X[idx[bi], s:s + W]
                g = seq < 4
                counts[f, np.arange(W)[g], seq[g]] += 1
                nsite[f] += 1
    pwm = np.where(counts.sum(2, keepdims=True) > 0,
                   counts / np.maximum(counts.sum(2, keepdims=True), 1), 0.25)
    ic = np.nansum(pwm * np.log2(np.maximum(pwm, 1e-9) / 0.25), (1, 2))
    return pwm, nsite, maxact, ic

REF_MOTIFS = {
    "PAS_AATAAA": "AATAAA",
    "PAS_ATTAAA": "ATTAAA",
    "acceptor_3ss_pyAG": "TTTTTTTTTTNCAGG",
    "donor_5ss": "CAGGTAAGT",
    "CA_cleavage": "CA",
}
def consensus_pwm(s):
    m = np.full((len(s), 4), 0.05)
    for i, c in enumerate(s):
        if c == "N": m[i] = 0.25
        else: m[i] = 0.05; m[i, NT.index(c)] = 0.85
    return m / m.sum(1, keepdims=True)

def pwm_match(p, q, min_overlap=6):
    def sc(a, b):
        best = -1.0
        for off in range(-len(b) + min_overlap, len(a) - min_overlap + 1):
            i0, i1 = max(0, off), min(len(a), off + len(b))
            j0, j1 = i0 - off, i1 - off
            if i1 - i0 < min_overlap: continue
            u = a[i0:i1].ravel(); v = b[j0:j1].ravel()
            if u.std() < 1e-9 or v.std() < 1e-9: continue
            best = max(best, float(np.corrcoef(u, v)[0, 1]))
        return best
    rc = q[::-1, ::-1]
    f, r = sc(p, q), sc(p, rc)
    return (f, "+") if f >= r else (r, "-")

def match_threshold(refs, rng, n=300, w=19, q=0.95):
    out = {}
    for k, ref in refs.items():
        v = [pwm_match(rng.dirichlet(np.ones(4) * 0.35, w), ref)[0] for _ in range(n)]
        out[k] = float(np.quantile(v, q))
    return out

def write_meme(pwm, ic, nsite, path="filter_pwms.meme"):
    with open(path, "w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                 "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for f in range(len(pwm)):
            fh.write(f"MOTIF filter{f:02d}\nletter-probability matrix: alength= 4 w= {W} "
                     f"nsites= {max(nsite[f],1)} E= 0\n")
            for r in pwm[f]: fh.write(" " + " ".join(f"{x:.6f}" for x in r) + "\n")
            fh.write("\n")

def dinuc_shuffle(tok, rng):
    t = tok[tok < 4]
    if len(t) < 3: return tok.copy()
    last = t[-1]; edges = [[] for _ in range(4)]
    for i in range(len(t) - 1): edges[t[i]].append(t[i + 1])
    for _ in range(100):
        pick = {}
        okv = True
        for v in range(4):
            if not edges[v] or v == last: continue
            pick[v] = edges[v][rng.integers(len(edges[v]))]
        for v in list(pick):
            seen = set(); u = v
            while u != last and u in pick and u not in seen:
                seen.add(u); u = pick[u]
            if u != last: okv = False; break
        if okv: break
    else:
        return tok.copy()
    order = []
    for v in range(4):
        e = list(edges[v])
        if v in pick:
            e.remove(pick[v]); rng.shuffle(e); e.append(pick[v])
        else: rng.shuffle(e)
        order.append(e)
    ptr = [0, 0, 0, 0]; out = [t[0]]; u = t[0]
    for _ in range(len(t) - 1):
        if ptr[u] >= len(order[u]): break
        nxt = order[u][ptr[u]]; ptr[u] += 1; out.append(nxt); u = nxt
    out = np.array(out, tok.dtype)
    if len(out) != len(t): return tok.copy()
    res = tok.copy(); res[tok < 4] = out
    return res

def saliency(oh):
    oh = oh.clone().requires_grad_(True)
    net.zero_grad(); net(oh).sum().backward()
    return oh.grad.detach()
def integrated_gradients(oh, steps=32):
    base = torch.full_like(oh, 0.25)
    tot = torch.zeros_like(oh)
    for a in torch.linspace(1.0 / steps, 1.0, steps, device=oh.device):
        x = (base + a * (oh - base)).requires_grad_(True)
        net.zero_grad(); net(x).sum().backward()
        tot += x.grad.detach()
    return (oh - base) * tot / steps

def attribution_raw(seqs, method, bs=32):
    outs = []
    for k in range(0, len(seqs), bs):
        oh = onehot(seqs[k:k+bs])
        g = saliency(oh) if method == "saliency" else integrated_gradients(oh, A.ig_steps)
        outs.append((g * oh).sum(1).cpu().numpy())
    return np.concatenate(outs)

GIA_MOTIFS = {"AATAAA": "AATAAA", "ATTAAA": "ATTAAA",
              "acceptor_pyAG": "TTTTTTTTTTTTTCAGG", "donor_GT": "CAGGTAAGT",
              "control_shuffled_AATAAA": "ATAAAT"}
def implant(seqs, motif, pos):
    m = np.array([NT.index(c) for c in motif], np.uint8)
    out = seqs.copy(); s = pos - len(m) // 2
    if s < 0 or s + len(m) > out.shape[1]: return None
    out[:, s:s + len(m)] = m
    return out

if __name__ == "__main__":
    t0 = time.time()
    pwm, nsite, maxact, ic = filter_pwms()
    write_meme(pwm, ic, nsite)
    refs = {k: consensus_pwm(v) for k, v in REF_MOTIFS.items()}
    THR = match_threshold(refs, rng)
    log("[filters] random-PWM null, 95th percentile of r per reference: "
        + "  ".join(f"{k} {v:.2f}" for k, v in THR.items()))
    with open("filter_matches.tsv", "w") as fh:
        fh.write("filter\tn_sites\tmax_activation\tinfo_content_bits\tbest_match\tr\tstrand\tconsensus\n")
        hits = {}
        for f in range(len(pwm)):
            best = max(((pwm_match(pwm[f], q)[0], pwm_match(pwm[f], q)[1], k) for k, q in refs.items()))
            cons = "".join(NT[i] for i in pwm[f].argmax(1))
            passes = best[0] >= THR.get(best[2], 1.0)
            fh.write(f"filter{f:02d}\t{nsite[f]}\t{maxact[f]:.3f}\t{ic[f]:.2f}\t"
                     f"{best[2] if passes else 'none'}\t{best[0]:.3f}\t{best[1]}\t{cons}\n")
            if passes: hits.setdefault(best[2], []).append((f, best[0]))
    log("[filters] filters whose best match beats the random-PWM null: " +
        ("  ".join(f"{k} x{len(v)}" for k, v in hits.items()) if hits else "NONE"))
    log(f"[filters] mean information content {ic.mean():.2f} bits over {W} positions; "
        f"filters with IC > 4 bits: {int((ic>4).sum())}/32")

    sub = rng.choice(te, min(A.n_seq, len(te)), replace=False)
    real = X[sub]
    shuf = np.stack([dinuc_shuffle(s, rng) for s in np.repeat(real, A.n_shuf, 0)])
    log(f"[null] {len(shuf)} dinucleotide shuffles of {len(real)} test sequences "
        f"(GC real {float(((real==1)|(real==2)).mean()):.4f} vs shuffled "
        f"{float(((shuf==1)|(shuf==2)).mean()):.4f})")
    rows = []
    HEX = {"AATAAA": "AATAAA", "ATTAAA": "ATTAAA", "TATAAA": "TATAAA",
           "CTGCAG_ctrl": "CTGCAG", "GGCGGC_ctrl": "GGCGGC"}
    def hexpos(seqs, h):
        code = np.array([NT.index(c) for c in h], np.uint8)
        w = np.lib.stride_tricks.sliding_window_view(seqs, len(code), 1)
        return (w == code).all(2)
    for method in ("saliency", "ig"):
        ar = attribution_raw(real, method); an = attribution_raw(shuf, method)
        stat = {}
        for nm, fn in (("mean_abs", lambda a: np.abs(a).mean()),
                       ("p99.9", lambda a: np.quantile(np.abs(a), 0.999)),
                       ("max_per_seq", lambda a: np.abs(a).max(1).mean()),
                       ("top20_per_seq", lambda a: np.sort(np.abs(a), 1)[:, -20:].mean())):
            stat[nm] = (float(fn(ar)), float(fn(an)))
        rows.append((method, stat))
        for nm, (r_, n_) in stat.items():
            log(f"[null] {method:8s} {nm:14s} real {r_:.4g}  shuffled {n_:.4g}  ratio {r_/n_:.3f}")
        with open(f"hexamer_attribution_{method}.tsv", "w") as fh:
            fh.write("hexamer\tmean_attr_real\tn_real\tmean_attr_shuffled\tn_shuffled\tratio\n")
            for nm, h in HEX.items():
                mr = hexpos(real, h); ms = hexpos(shuf, h)
                sr = np.lib.stride_tricks.sliding_window_view(ar, len(h), 1).sum(2)[mr]
                ss = np.lib.stride_tricks.sliding_window_view(an, len(h), 1).sum(2)[ms]
                if len(sr) < 10 or len(ss) < 10: continue
                fh.write(f"{nm}\t{sr.mean():.6g}\t{len(sr)}\t{ss.mean():.6g}\t{len(ss)}\t"
                         f"{sr.mean()/ss.mean() if ss.mean()!=0 else float('nan'):.4f}\n")
                log(f"[hex] {method:8s} {nm:12s} attr real {sr.mean():+.4g} (n {len(sr)})  "
                    f"shuffled {ss.mean():+.4g} (n {len(ss)})")
        nb = X.shape[1] // 100
        pr = np.abs(ar[:, :nb*100]).reshape(len(ar), nb, 100).mean((0, 2))
        pn = np.abs(an[:, :nb*100]).reshape(len(an), nb, 100).mean((0, 2))
        with open(f"attribution_profile_{method}.tsv", "w") as fh:
            fh.write("offset\treal\tshuffled\tratio\n")
            for j in range(nb):
                off = -UP + j * 100 + 50
                fh.write(f"{off}\t{pr[j]:.6g}\t{pn[j]:.6g}\t{pr[j]/max(pn[j],1e-12):.4f}\n")
        np.savez_compressed(f"attr_{method}.npz", real=ar.astype(np.float32), idx=sub)
    with open("attribution_over_null.tsv", "w") as fh:
        fh.write("method\tstatistic\treal\tshuffled\tratio\tn_real\tn_shuffled\n")
        for m, st in rows:
            for nm, (r_, n_) in st.items():
                fh.write(f"{m}\t{nm}\t{r_:.6g}\t{n_:.6g}\t{r_/n_:.4f}\t{len(real)}\t{len(shuf)}\n")

    # 4 gia
    bg_src = rng.choice(te, A.gia_bg, replace=False)
    bg = np.stack([dinuc_shuffle(X[i], rng) for i in bg_src])
    comp_pool = te[y[te] == 0]
    bg_real = X[rng.choice(comp_pool, A.gia_bg, replace=False)]
    positions = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
    with open("gia_results.tsv", "w") as fh:
        fh.write("background\tmotif\tposition\tdelta_logit\tse\tn\n")
        for bgname, B in (("dinuc_shuffled", bg), ("real_competitor", bg_real)):
            base_logit = fwd(onehot(B))
            log(f"[GIA] {bgname} background logit mean {base_logit.mean():+.3f} "
                f"sd {base_logit.std():.3f} (model logit range on test is about -9 to +18)")
            for name, m in GIA_MOTIFS.items():
                best = (None, -1e9)
                for pos in positions:
                    seqs = implant(B, m, pos + UP)
                    if seqs is None: continue
                    d = fwd(onehot(seqs)) - base_logit
                    fh.write(f"{bgname}\t{name}\t{pos}\t{d.mean():.5f}\t"
                             f"{d.std(ddof=1)/np.sqrt(len(d)):.5f}\t{len(d)}\n")
                    if d.mean() > best[1]: best = (pos, d.mean())
                log(f"[GIA] {bgname:16s} {name:24s} max delta-logit {best[1]:+.3f} at {best[0]:+d} nt")
    # 5 seqlets
    ar = np.load("attr_ig.npz")["real"]; K = 12
    sc = np.abs(ar); w = np.lib.stride_tricks.sliding_window_view(sc, K, 1).sum(2)
    seqlets = []
    for i in range(len(ar)):
        for p in np.argsort(-w[i])[:3]:
            s = real[i, p:p + K]
            if (s < 4).all(): seqlets.append((w[i, p], s, p - UP))
    seqlets.sort(key=lambda z: -z[0]); seqlets = seqlets[:2000]
    M = np.stack([s for _, s, _ in seqlets])
    oh = np.eye(4)[M]
    cl = []
    used = np.zeros(len(M), bool)
    for i in range(len(M)):
        if used[i]: continue
        sim = (oh * oh[i]).sum((1, 2)) / K
        mem = np.flatnonzero((sim >= 0.55) & ~used)
        if len(mem) < 20: continue
        used[mem] = True
        cl.append((len(mem), oh[mem].mean(0), np.median([seqlets[j][2] for j in mem])))
        if len(cl) >= 10: break
    with open("seqlet_clusters.tsv", "w") as fh:
        fh.write("cluster\tn_seqlets\tmedian_position\tconsensus\tbest_match\tr\n")
        for ci, (n, p, pos) in enumerate(cl):
            best = max((pwm_match(p, q)[0], k) for k, q in refs.items())
            pass_null = best[0] >= THR.get(best[1], 1.0)
            cons = "".join(NT[i] for i in p.argmax(1))
            fh.write(f"{ci}\t{n}\t{pos:.0f}\t{cons}\t{best[1] if pass_null else 'none'}\t{best[0]:.3f}\n")
            log(f"[seqlet] cluster {ci}: n={n} median pos {pos:+.0f} {cons} -> "
                f"{best[1]+' r='+format(best[0],'.2f') if pass_null else 'no match above the random-PWM null'}")
    log(f"[note] TF-MoDISco unavailable offline (modiscolite not in the CC wheelhouse); "
        f"seqlet_clusters.tsv is a documented substitute, not TF-MoDISco")
    log(f"[out] filter_pwms.meme filter_matches.tsv attribution_over_null.tsv "
        f"attribution_profile_*.tsv gia_results.tsv seqlet_clusters.tsv  ({time.time()-t0:.0f}s)")
