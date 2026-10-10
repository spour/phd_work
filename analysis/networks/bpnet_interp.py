#!/usr/bin/env python3
# the interpretability suite applied to the bpnet variant
import sys, os, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from cnn_interp_lib import NT, REF_MOTIFS, consensus_pwm, pwm_match, match_threshold, write_meme

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="bpnet_model.pt")
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--n_pwm", type=int, default=4000)
P.add_argument("--posmap_genes", type=int, default=600)
P.add_argument("--batch", type=int, default=256)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--stages", default="1,2,3")
A = P.parse_args()
STAGES = set(A.stages.split(","))
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    log(f"[env] cpu, {torch.get_num_threads()} torch threads")
NBINS, BINW, UP = 25, 100, 2000
rng = np.random.default_rng(A.seed)

from bpnet_model_def import BPNet, DIL
ck = torch.load(A.model, map_location=dev, weights_only=False)
W = ck["args"]["width"]; NF = ck["args"]["nf"]
net = BPNet(ck["args"]["dropout"], NF, W).to(dev)
net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
RF = BPNet.receptive_field(W)
log(f"[model] {A.model}  L1 {ck['l1']:g}  epoch {ck['epoch']}  {NF} conv1 filters of width {W}")
log(f"[arch] dilations {DIL}; receptive field per output bin = {W} + 2*{sum(DIL)} + {BINW-1} "
    f"= {RF} nt (reported model 378 nt, matched deliberately)")

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
te = np.flatnonzero(split == "test"); yte = y[te]
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def oh(codes):
    return TBLt[torch.from_numpy(codes.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def groups(ii):
    by = {}
    for k, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k] == 1 else c).append(k)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def winrate(s, grp):
    return float(np.mean([((s[r][:, None] > s[c][None, :]).sum()
                           + 0.5 * (s[r][:, None] == s[c][None, :]).sum()) / (len(r) * len(c))
                          for r, c in grp]))
def fwd(codes, bs=None):
    bs = bs or A.batch; out = np.empty(len(codes), np.float32)
    with torch.no_grad():
        for k in range(0, len(codes), bs):
            out[k:k+bs] = net(oh(codes[k:k+bs])).float().cpu().numpy()
    return out

if "1" in STAGES:
    t0 = time.time()
    sub = rng.choice(te, min(A.n_pwm, len(te)), replace=False)
    qs = sub[:400]; acc = []
    for k in range(0, len(qs), 128):
        with torch.no_grad(): acc.append(net.first(oh(X[qs[k:k+128]])).cpu().numpy())
    a0 = np.concatenate(acc, 0)
    thr = np.quantile(a0.transpose(1, 0, 2).reshape(NF, -1), 0.999, axis=1)
    del a0, acc
    counts = np.zeros((NF, W, 4)); nsite = np.zeros(NF, int); maxact = np.zeros(NF)
    for k in range(0, len(sub), 128):
        ii = sub[k:k+128]
        with torch.no_grad(): a = net.first(oh(X[ii])).cpu().numpy()
        maxact = np.maximum(maxact, a.max((0, 2)))
        for f in range(NF):
            b, p = np.where(a[:, f, :] >= thr[f])
            for bi, pi in zip(b, p):
                s_ = pi - W // 2
                if s_ < 0 or s_ + W > X.shape[1]: continue
                sq = X[ii[bi], s_:s_+W]; g = sq < 4
                counts[f, np.arange(W)[g], sq[g]] += 1; nsite[f] += 1
    PWM = np.where(counts.sum(2, keepdims=True) > 0,
                   counts / np.maximum(counts.sum(2, keepdims=True), 1), 0.25)
    IC = np.nansum(PWM * np.log2(np.maximum(PWM, 1e-9) / 0.25), (1, 2))
    ICC = IC / W
    refs = {k_: consensus_pwm(v) for k_, v in REF_MOTIFS.items()}
    thr95 = match_threshold(refs, np.random.default_rng(A.seed))
    write_meme(PWM, nsite, "bpnet_pwms_matched.meme", W=W)
    rows = []
    with open("bpnet_filters_all.tsv", "w") as fh:
        fh.write("filter\tic_bits\tic_per_col\tn_sites\tmax_activation\tbest_match\tr\tstrand\t"
                 "beats_null\tconsensus\tfrac_AT\twidth\n")
        for f in range(NF):
            cands = [(pwm_match(PWM[f], q)[0], pwm_match(PWM[f], q)[1], k_)
                     for k_, q in refs.items()]
            r_, st, k_ = max(cands)
            beats = r_ >= thr95[k_]
            cons = "".join(NT[i] for i in PWM[f].argmax(1))
            fat = float(PWM[f][:, [0, 3]].sum() / W)
            rows.append((f, ICC[f], nsite[f], k_ if beats else "none", r_, cons, fat))
            fh.write(f"filter{f:02d}\t{IC[f]:.3f}\t{ICC[f]:.4f}\t{nsite[f]}\t{maxact[f]:.3f}\t"
                     f"{k_ if beats else 'none'}\t{r_:.3f}\t{st}\t{int(beats)}\t{cons}\t"
                     f"{fat:.4f}\t{W}\n")
    with open("bpnet_logos_all.tsv", "w") as fh:
        fh.write("filter\tposition\tbase\tprob\theight\tic_col\n")
        for f in range(NF):
            for p_ in range(W):
                q = np.clip(PWM[f, p_], 1e-9, 1)
                icc = 2.0 + float((q * np.log2(q)).sum())
                for b_, base in enumerate("ACGT"):
                    fh.write(f"filter{f:02d}\t{p_+1}\t{base}\t{PWM[f,p_,b_]:.6f}\t"
                             f"{PWM[f,p_,b_]*icc:.6f}\t{icc:.6f}\n")
    nm = sum(1 for r in rows if r[3] != "none")
    log(f"\n[1] {NF} conv1 filters, width {W}: IC {ICC.min():.3f} to {ICC.max():.3f} bits/column "
        f"(median {np.median(ICC):.3f}); {nm}/{NF} beat the random-PWM null")
    import collections
    log("[1] what the matching filters match: " + ", ".join(
        f"{k_} x{v}" for k_, v in collections.Counter(r[3] for r in rows if r[3] != "none").items()))
    at_m = np.mean([r[6] for r in rows if r[3] != "none"])
    at_n = np.mean([r[6] for r in rows if r[3] == "none"])
    log(f"[1] A+T share of the PWM mass: matching filters {at_m:.3f}, non-matching {at_n:.3f} "
        f"(0.5 would be unbiased); a high value means the match is AT composition, not a motif")
    for f, ic_, ns_, m_, r_, c_, fat in sorted(rows, key=lambda t: -t[1])[:8]:
        log(f"    filter{f:02d}  IC {ic_:.3f}  sites {ns_:6d}  A+T {fat:.2f}  "
            f"match {m_:>18} r {r_:.3f}  {c_}")
    log(f"[out] bpnet_filters_all.tsv bpnet_pwms_matched.meme bpnet_logos_all.tsv "
        f"({time.time()-t0:.0f}s)")

if "2" in STAGES:
    t0 = time.time()
    GTE = groups(te); L = X.shape[1]
    base = fwd(X[te]); wg0 = winrate(base, GTE)
    log(f"\n[2] intact within-gene {wg0:.5f} on {len(GTE)} test genes")
    starts = list(range(0, L - BINW + 1, 50))
    with open("bpnet_profile.tsv", "w") as fh:
        fh.write("model\tstart\tend\tcentre\twinrate\td_winrate\td_frac_of_lift\t"
                 "mean_abs_dlogit\tintact_winrate\tn_genes\n")
        for a_ in starts:
            b_ = a_ + BINW
            cd = X[te]
            with torch.no_grad():
                out = np.empty(len(te), np.float32)
                for k in range(0, len(te), A.batch):
                    v = oh(cd[k:k+A.batch]); v[:, :, a_:b_] = 0.25
                    out[k:k+A.batch] = net(v).float().cpu().numpy()
            w = winrate(out, GTE); d = wg0 - w
            fh.write(f"BPNet variant\t{a_-UP}\t{b_-UP}\t{(a_+b_)/2-UP:.0f}\t{w:.6f}\t{d:.6f}\t"
                     f"{d/(wg0-0.5):.6f}\t{np.abs(out-base).mean():.6f}\t{wg0:.6f}\t{len(GTE)}\n")
    log(f"[out] bpnet_profile.tsv ({len(starts)} windows, {time.time()-t0:.0f}s)")

if "3" in STAGES:
    t0 = time.time()
    def head_from_first(a):
        h = net.bn_post(a)
        for blk in net.blocks: h = h + blk(h)
        h = F.avg_pool1d(h, net.pool).flatten(1)
        return net.fc2(F.relu(net.fc1(h))).squeeze(1)
    with torch.no_grad():
        x = oh(X[te[:64]]); a = net.first(x)
        assert float((head_from_first(a) - net(x)).abs().max()) < 1e-3, \
            "conv1 re-entry does not reproduce the full forward"
    log("[3] conv1 re-entry reproduces the full forward")

    gl = sorted(set(gene[te])); rng2 = np.random.default_rng(A.seed)
    keep = set(rng2.choice(gl, min(A.posmap_genes, len(gl)), replace=False))
    sel = np.array([i for i, g in enumerate(gene[te]) if g in keep])
    GSEL = groups(te[sel]); ysel = yte[sel]
    log(f"[3] {len(sel)} sites in {len(GSEL)} genes; {NF} filters x {NBINS} bins = "
        f"{NF*NBINS} occlusions, no pooling shortcut available")
    ACT = []
    with torch.no_grad():
        for k in range(0, len(sel), A.batch):
            ACT.append(net.first(oh(X[te[sel][k:k+A.batch]])))
    with torch.no_grad():
        b0 = np.concatenate([head_from_first(a).float().cpu().numpy() for a in ACT])
    wg0s = winrate(b0, GSEL)
    log(f"[3] intact within-gene on the subsample {wg0s:.5f}")
    with open("bpnet_posmap.tsv", "w") as fh:
        fh.write("filter\tbin\tbin_start\tbin_end\tcontrib\tmean_d_real\tmean_d_comp\t"
                 "winrate\td_winrate\tic_per_col\tbest_match\tn_sites\tn_genes\n")
        FI = np.genfromtxt("bpnet_filters_all.tsv", names=True, delimiter="\t", dtype=None,
                           encoding="utf-8")
        for f in range(NF):
            for b in range(NBINS):
                out = []
                with torch.no_grad():
                    for a in ACT:
                        a2 = a.clone(); a2[:, f, b*BINW:(b+1)*BINW] = 0.0
                        out.append(head_from_first(a2).float().cpu().numpy())
                s = np.concatenate(out); dl = b0 - s
                cr, cc = dl[ysel == 1].mean(), dl[ysel == 0].mean()
                w = winrate(s, GSEL)
                fh.write(f"filter{f:02d}\t{b}\t{b*BINW-UP}\t{(b+1)*BINW-UP}\t{cr-cc:.6f}\t"
                         f"{cr:.6f}\t{cc:.6f}\t{w:.6f}\t{wg0s-w:.6f}\t"
                         f"{FI['ic_per_col'][f]:.4f}\t{FI['best_match'][f]}\t{len(sel)}\t"
                         f"{len(GSEL)}\n")
            fh.flush()
            el = time.time() - t0
            log(f"  {f+1}/{NF} filters  {el/60:.1f} min  "
                f"eta {(el/(f+1)*(NF-f-1))/60:.1f} min")
    log(f"[out] bpnet_posmap.tsv ({(time.time()-t0)/60:.1f} min)")
