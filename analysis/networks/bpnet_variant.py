#!/usr/bin/env python3
# does a motif-friendly architecture buy sharper first-layer filters without losing
import sys, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import binned_common as K
from binned_common import log
from cnn_interp_lib import (NT, REF_MOTIFS, consensus_pwm, pwm_match, match_threshold,
                            write_meme, dinuc_shuffle)

P = argparse.ArgumentParser()
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--l1", default="1e-4,1e-3,1e-2")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--nf", type=int, default=48)
P.add_argument("--width", type=int, default=25)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--n_pwm", type=int, default=4000)
P.add_argument("--ntrain", type=int, default=0, help="smoke test: subsample training rows")
A = P.parse_args()
L1S = [float(x) for x in A.l1.split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] device {dev}  torch {torch.__version__}"
    + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))

D = np.load(A.data, allow_pickle=True)
ok = D["ok"]; X = D["X"][ok]; y = D["y"][ok].astype(np.float32)
gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
log("[data] " + "  ".join(f"{s} {len(idx[s])} ({int(y[idx[s]].sum())} real)" for s in idx))
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
def onehot(a):
    t = a if torch.is_tensor(a) else torch.from_numpy(a.astype(np.int64))
    return TBLt[t.to(dev)].permute(0, 2, 1).contiguous()

from bpnet_model_def import BPNet
RF_NEW, RF_OLD = BPNet.receptive_field(A.width), 378
log(f"[arch] BPNet receptive field {RF_NEW} nt per output bin; reported model {RF_OLD} nt")

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
GVAL = groups(idx["val"])

# l1 sweep
def train_one(l1):
    torch.manual_seed(A.seed); np.random.seed(A.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    net = BPNet(A.dropout, A.nf, A.width).to(dev)
    tr = idx["train"][:A.ntrain] if A.ntrain else idx["train"]
    npos = float(y[tr].sum()); nneg = len(tr) - npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)
    def sc(ii, bs=256):
        net.eval(); out = np.empty(len(ii), np.float32)
        with torch.no_grad():
            for k in range(0, len(ii), bs):
                out[k:k+bs] = net(onehot(Xt[ii[k:k+bs]])).float().cpu().numpy()
        return out
    best = (-1.0, -1, None); hist = []; t0 = time.time()
    for ep in range(1, A.epochs + 1):
        net.train(); perm = np.random.permutation(tr); tot = 0.0; act = 0.0; nb = 0
        for k in range(0, len(perm), A.batch):
            j = perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            o, a = net(onehot(Xt[j]), return_first=True)
            sp = a.abs().mean()
            loss = crit(o, yt[j].to(dev)) + l1 * sp
            loss.backward(); opt.step()
            tot += float(loss.detach()); act += float(sp.detach()); nb += 1
        w = winrate(sc(idx["val"]), GVAL)
        hist.append((ep, tot/nb, act/nb, w))
        if w > best[0]:
            best = (w, ep, {k_: v.detach().cpu().clone() for k_, v in net.state_dict().items()})
        elif ep - best[1] >= A.patience:
            break
    net.load_state_dict(best[2])
    log(f"[sweep] L1 {l1:g}: best epoch {best[1]}, val within-gene {best[0]:.4f}, "
        f"mean |conv1| {hist[best[1]-1][2]:.4f}, {len(hist)} epochs, {time.time()-t0:.0f}s")
    return net, best, hist

import os
RES = []
for l1 in L1S:
    ckpt = f"bpnet_arm_{l1:g}.pt"
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        net = BPNet(A.dropout, A.nf, A.width).to(dev); net.load_state_dict(c["state"])
        RES.append(dict(l1=l1, net=net, val_wg=c["val_wg"], epoch=c["epoch"],
                        hist=c["hist"], act=c["act"]))
        log(f"[sweep] L1 {l1:g}: restored from {ckpt}, val within-gene {c['val_wg']:.4f}")
        continue
    net, best, hist = train_one(l1)
    r = dict(l1=l1, net=net, val_wg=best[0], epoch=best[1], hist=hist, act=hist[best[1]-1][2])
    RES.append(r)
    tmp = ckpt + ".tmp"
    torch.save(dict(state=net.state_dict(), val_wg=r["val_wg"], epoch=r["epoch"],
                    hist=r["hist"], act=r["act"], l1=l1), tmp)
    os.replace(tmp, ckpt)
    log(f"[sweep] L1 {l1:g}: checkpointed to {ckpt}")
BEST = max(RES, key=lambda r: r["val_wg"])
net = BEST["net"]; net.eval()
for p in net.parameters(): p.requires_grad_(False)
log(f"[sweep] selected L1 {BEST['l1']:g} on validation within-gene {BEST['val_wg']:.4f}")
with open("bpnet_sweep.tsv", "w") as fh:
    fh.write("l1\tbest_epoch\tval_winrate\tmean_abs_conv1\tn_epochs\tselected\n")
    for r in RES:
        fh.write(f"{r['l1']:g}\t{r['epoch']}\t{r['val_wg']:.5f}\t{r['act']:.5f}\t"
                 f"{len(r['hist'])}\t{int(r is BEST)}\n")
log("[out] bpnet_sweep.tsv")

def scores(ii, bs=256):
    out = np.empty(len(ii), np.float32)
    with torch.no_grad():
        for k in range(0, len(ii), bs):
            out[k:k+bs] = net(onehot(Xt[ii[k:k+bs]])).float().cpu().numpy()
    return out

sv = torch.from_numpy(scores(idx["val"])).to(dev); yv = yt[idx["val"]].to(dev)
logT = torch.zeros(1, device=dev, requires_grad=True)
optT = torch.optim.LBFGS([logT], lr=0.1, max_iter=100); bce = nn.BCEWithLogitsLoss()
def closure():
    optT.zero_grad(); l = bce(sv / torch.exp(logT), yv); l.backward(); return l
optT.step(closure); Temp = float(torch.exp(logT.detach()))
torch.save(dict(state=net.state_dict(), args=vars(A), l1=BEST["l1"], temperature=Temp,
                epoch=BEST["epoch"]), "bpnet_model.pt")
log(f"[fit] temperature {Temp:.4f}; saved bpnet_model.pt")

SC = scores(K.te)
R = K.suite(SC.astype(np.float64))
REP = K.suite(np.load("cnn_model_scores.npz", allow_pickle=True)["logit"]
              .astype(np.float64)[K.ok][K.te])
assert abs(REP["winrate"] - 0.82110) < 5e-5, "reported-model anchor moved"
d = R["_wg"] - REP["_wg"]; db = d[K.BS].mean(1)
log(f"\n[1] BPNet within-gene {R['winrate']:.5f} [{R['winrate_lo']:.5f},{R['winrate_hi']:.5f}]  "
    f"P@1 {R['p_at_1']:.5f} [{R['p_at_1_lo']:.5f},{R['p_at_1_hi']:.5f}]  "
    f"AUROC {R['auroc']:.5f}  AUPRC {R['auprc']:.5f}")
log(f"[1] reported model {REP['winrate']:.5f} / P@1 {REP['p_at_1']:.5f}; paired delta "
    f"{d.mean():+.5f} [{np.percentile(db,2.5):+.5f},{np.percentile(db,97.5):+.5f}]")
DLO, DHI = float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))
PERF = ("improved" if (d.mean() > 0.02 and DLO > 0) else
        "degraded" if (d.mean() < -0.02 and DHI < 0) else "unchanged")
HOLDS = PERF == "unchanged"
log(f"[1] performance vs the reported model: {PERF.upper()} "
    f"({d.mean():+.5f} [{DLO:+.5f},{DHI:+.5f}], threshold 0.02)")
with open("bpnet_metrics.tsv", "w") as fh:
    fh.write("model\twithin_gene\twg_lo\twg_hi\tp_at_1\tp1_lo\tp1_hi\tauroc\tauprc\tn_genes\n")
    for nm, r in (("BPNet variant", R), ("Reported CNN", REP)):
        fh.write(f"{nm}\t{r['winrate']:.5f}\t{r['winrate_lo']:.5f}\t{r['winrate_hi']:.5f}\t"
                 f"{r['p_at_1']:.5f}\t{r['p_at_1_lo']:.5f}\t{r['p_at_1_hi']:.5f}\t"
                 f"{r['auroc']:.5f}\t{r['auprc']:.5f}\t{len(K.GRP_TE)}\n")
    fh.write(f"paired_delta\t{d.mean():.5f}\t{np.percentile(db,2.5):.5f}\t"
             f"{np.percentile(db,97.5):.5f}\t\t\t\t\t\t{len(K.GRP_TE)}\n")
log("[out] bpnet_metrics.tsv")

rng = np.random.default_rng(A.seed)
def filter_pwms(model, W, NF, first, n_seq=A.n_pwm, frac=0.3):
    sub = rng.choice(K.te, min(n_seq, len(K.te)), replace=False)
    counts = np.zeros((NF, W, 4)); nsite = np.zeros(NF, int); maxact = np.zeros(NF)
    for k in range(0, len(sub), 128):
        with torch.no_grad(): a = first(onehot(X[sub[k:k+128]])).cpu().numpy()
        maxact = np.maximum(maxact, a.max((0, 2)))
    thr = frac * maxact
    for k in range(0, len(sub), 128):
        ii = sub[k:k+128]
        with torch.no_grad(): a = first(onehot(X[ii])).cpu().numpy()
        for f in range(NF):
            b, p = np.where(a[:, f, :] >= thr[f])
            for bi, pi in zip(b, p):
                s = pi - W // 2
                if s < 0 or s + W > X.shape[1]: continue
                sq = X[ii[bi], s:s+W]; g = sq < 4
                counts[f, np.arange(W)[g], sq[g]] += 1; nsite[f] += 1
    pwm = np.where(counts.sum(2, keepdims=True) > 0,
                   counts / np.maximum(counts.sum(2, keepdims=True), 1), 0.25)
    ic = np.nansum(pwm * np.log2(np.maximum(pwm, 1e-9) / 0.25), (1, 2))
    return pwm, nsite, maxact, ic

def filter_pwms_q(model, W, NF, first, n_seq=A.n_pwm, q=0.999, n_sub=400):
    sub = rng.choice(K.te, min(n_seq, len(K.te)), replace=False)
    qs = sub[:min(n_sub, len(sub))]
    acc = []
    for k in range(0, len(qs), 128):
        with torch.no_grad(): acc.append(first(onehot(X[qs[k:k+128]])).cpu().numpy())
    a0 = np.concatenate(acc, 0)
    thr = np.quantile(a0.transpose(1, 0, 2).reshape(NF, -1), q, axis=1)
    del a0, acc
    counts = np.zeros((NF, W, 4)); nsite = np.zeros(NF, int)
    for k in range(0, len(sub), 128):
        ii = sub[k:k+128]
        with torch.no_grad(): a = first(onehot(X[ii])).cpu().numpy()
        for f in range(NF):
            b, p = np.where(a[:, f, :] >= thr[f])
            for bi, pi in zip(b, p):
                s_ = pi - W // 2
                if s_ < 0 or s_ + W > X.shape[1]: continue
                sq = X[ii[bi], s_:s_+W]; g = sq < 4
                counts[f, np.arange(W)[g], sq[g]] += 1; nsite[f] += 1
    pwm = np.where(counts.sum(2, keepdims=True) > 0,
                   counts / np.maximum(counts.sum(2, keepdims=True), 1), 0.25)
    ic = np.nansum(pwm * np.log2(np.maximum(pwm, 1e-9) / 0.25), (1, 2))
    return pwm, nsite, ic

from cnn_model_def import Net as OldNet
ock = torch.load("cnn_model.pt", map_location=dev, weights_only=False)
onet = OldNet(ock["args"]["dropout"]).to(dev); onet.load_state_dict(ock["state"]); onet.eval()
for p in onet.parameters(): p.requires_grad_(False)

PW, NS, MA, IC = filter_pwms(net, A.width, A.nf, net.first)
ICC = IC / A.width
OPW, ONS, OMA, OIC = filter_pwms(onet, 19, 32, onet.first)
OICC = OIC / 19.0
PUB = np.genfromtxt("filter_matches_reference.tsv", names=True, delimiter="\t", dtype=None,
                    encoding="utf-8")
dmax = float(np.abs(np.sort(OIC) - np.sort(PUB["info_content_bits"])).max())
log(f"[2] reported model re-derived here vs filter_matches_reference.tsv: max |IC diff| "
    f"{dmax:.3f} bits over 32 filters (same procedure, different random sequence sample)")
PWq, NSq, ICq = filter_pwms_q(net, A.width, A.nf, net.first)
OPWq, ONSq, OICq = filter_pwms_q(onet, 19, 32, onet.first)
ICCq, OICCq = ICq / A.width, OICq / 19.0
log(f"[2] count-matched sites per filter: BPNet median {int(np.median(NSq))}, "
    f"reported median {int(np.median(ONSq))} (30%-of-max: {int(np.median(NS))} and "
    f"{int(np.median(ONS))})")
refs = {k: consensus_pwm(v) for k, v in REF_MOTIFS.items()}
thr95 = match_threshold(refs, np.random.default_rng(A.seed))
write_meme(PW, NS, "bpnet_pwms.meme", W=A.width)
OLD = PUB
rows = []
with open("bpnet_filters.tsv", "w") as fh:
    fh.write("filter\tn_sites\tmax_activation\tic_bits\tic_per_col\tic_per_col_matched\t"
             "n_sites_matched\tbest_match\tr\tstrand\tbeats_null\tconsensus\twidth\n")
    for f in range(A.nf):
        cands = [(pwm_match(PW[f], q)[0], pwm_match(PW[f], q)[1], k) for k, q in refs.items()]
        r_, st, k_ = max(cands)
        beats = r_ >= thr95[k_]
        cons = "".join(NT[i] for i in PW[f].argmax(1))
        rows.append((f, ICC[f], NS[f], k_ if beats else "none", r_, cons))
        fh.write(f"filter{f:02d}\t{NS[f]}\t{MA[f]:.3f}\t{IC[f]:.2f}\t{ICC[f]:.4f}\t"
                 f"{ICCq[f]:.4f}\t{NSq[f]}\t"
                 f"{k_ if beats else 'none'}\t{r_:.3f}\t{st}\t{int(beats)}\t{cons}\t{A.width}\n")
log(f"\n[2] 30%-of-max threshold   BPNet {ICC.min():.3f}-{ICC.max():.3f} "
    f"(median {np.median(ICC):.3f}) bits/col over {A.nf} filters of width {A.width}; "
    f"reported {OICC.min():.3f}-{OICC.max():.3f} (median {np.median(OICC):.3f}) over 32 of 19")
log(f"[2] count-matched (99.9th pct)  BPNet {ICCq.min():.3f}-{ICCq.max():.3f} "
    f"(median {np.median(ICCq):.3f}); reported {OICCq.min():.3f}-{OICCq.max():.3f} "
    f"(median {np.median(OICCq):.3f})")
SHARPER = np.median(ICCq) > np.median(OICCq) and ICCq.max() > OICCq.max()
log(f"[2] sharper, on the count-matched comparison: {'YES' if SHARPER else 'NO'}  "
    f"(median {np.median(ICCq)-np.median(OICCq):+.3f}, "
    f"max {ICCq.max()-OICCq.max():+.3f} bits/col)")
nm_ = sum(1 for r in rows if r[3] != "none")
log(f"[2] {nm_}/{A.nf} beat the random-PWM null against a reference motif "
    f"(reported model: {int((OLD['best_match']!='none').sum())}/32)")
for f, ic_, ns_, m_, r_, c_ in sorted(rows, key=lambda t: -t[1])[:5]:
    log(f"    filter{f:02d}  IC {ic_:.3f} b/col  sites {ns_:6d}  match {m_:>18} r {r_:.3f}  {c_}")
log("[out] bpnet_filters.tsv bpnet_pwms.meme")

GIA = {"donor_AGGTAAGT": "AGGTAAGT", "donor_perm_ctrl_TAGAGTAG": "TAGAGTAG",
       "GU_rich_GTGTGTGT": "GTGTGTGT", "acceptor_pyAG": "TTTTTTTTTTTTTCAGG",
       "PAS_AATAAA": "AATAAA", "PAS_perm_ctrl_AAAATA": "AAAATA"}
UP = 2000
POS = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
rb = np.random.default_rng(A.seed)
BG = np.stack([dinuc_shuffle(X[i], rb) for i in rb.choice(K.te, A.gia_bg, replace=False)])
BGR = X[rb.choice(K.te[K.y[K.te] == 0], A.gia_bg, replace=False)]
def fwd(m, a, bs=256):
    out = []
    with torch.no_grad():
        for k in range(0, len(a), bs): out.append(m(onehot(a[k:k+bs])).float().cpu().numpy())
    return np.concatenate(out)
def implant(seqs, motif, pos):
    m = np.array([NT.index(c) for c in motif], np.uint8)
    o = seqs.copy(); s = pos - len(m) // 2
    if s < 0 or s + len(m) > o.shape[1]: return None
    o[:, s:s+len(m)] = m
    return o

BESTG = {}
with open("bpnet_gia.tsv", "w") as fh:
    fh.write("model\tbackground\tmotif\tposition\tdelta_logit\tse\tn\n")
    for mn, m in (("BPNet variant", net), ("Reported CNN", onet)):
        for bn, B in (("dinuc_shuffled", BG), ("real_competitor", BGR)):
            base = fwd(m, B)
            for name, mo in GIA.items():
                bst = (None, -1e9)
                for p_ in POS:
                    s_ = implant(B, mo, p_ + UP)
                    if s_ is None: continue
                    dd = fwd(m, s_) - base
                    fh.write(f"{mn}\t{bn}\t{name}\t{p_}\t{dd.mean():.5f}\t"
                             f"{dd.std(ddof=1)/np.sqrt(len(dd)):.5f}\t{len(dd)}\n")
                    if dd.mean() > bst[1]: bst = (p_, float(dd.mean()))
                BESTG[(mn, bn, name)] = bst
log("\n[3] implantation, both models on the SAME backgrounds and position grid")
for mn in ("Reported CNN", "BPNet variant"):
    for bn in ("dinuc_shuffled", "real_competitor"):
        log(f"  {mn} / {bn}")
        for name in GIA:
            p_, v = BESTG[(mn, bn, name)]
            log(f"    {name:>26} {v:+.4f} at {p_:+5d}")
chk = BESTG[("Reported CNN", "dinuc_shuffled", "acceptor_pyAG")][1]
log(f"[3] check: reported model acceptor in shuffled background {chk:+.4f} here against "
    f"+0.27169 in donor_verdict.tsv (different background draw, same procedure)")
if A.gia_bg >= 128:
    assert abs(chk - 0.27169) < 0.08, "the implantation procedure does not reproduce the earlier run"
else:
    log(f"[3] reproduction check skipped: {A.gia_bg} backgrounds is too few to compare")

NEW = []
for bn in ("dinuc_shuffled", "real_competitor"):
    for name, ctrl in (("donor_AGGTAAGT", "donor_perm_ctrl_TAGAGTAG"),
                       ("PAS_AATAAA", "PAS_perm_ctrl_AAAATA"), ("GU_rich_GTGTGTGT", None)):
        v = BESTG[("BPNet variant", bn, name)][1]
        c = BESTG[("BPNet variant", bn, ctrl)][1] if ctrl else 0.0
        o = BESTG[("Reported CNN", bn, name)][1]
        oc = BESTG[("Reported CNN", bn, ctrl)][1] if ctrl else 0.0
        if v - c > 0.05 and (v - c) > 2 * (o - oc):
            NEW.append((bn, name, v - c, o - oc))
        log(f"[3] {bn:16s} {name:20s} BPNet {v:+.4f} over control {v-c:+.4f}; "
            f"reported {o:+.4f} over control {o-oc:+.4f}")
log(f"[3] motifs registering on the BPNet variant but not the reported model: "
    f"{'none' if not NEW else NEW}")

# verdict
acc_new = BESTG[("BPNet variant", "dinuc_shuffled", "acceptor_pyAG")][1]
with open("bpnet_verdict.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    fh.write(f"selected_l1\t{BEST['l1']:g}\nreceptive_field_new\t{RF_NEW}\n"
             f"receptive_field_reported\t{RF_OLD}\n"
             f"winrate_new\t{R['winrate']:.5f}\nwinrate_reported\t{REP['winrate']:.5f}\n"
             f"winrate_delta\t{d.mean():.5f}\nperformance\t{PERF}\n"
             f"p_at_1_new\t{R['p_at_1']:.5f}\np_at_1_reported\t{REP['p_at_1']:.5f}\n"
             f"ic_per_col_new_min\t{ICC.min():.4f}\nic_per_col_new_max\t{ICC.max():.4f}\n"
             f"ic_per_col_new_median\t{np.median(ICC):.4f}\n"
             f"ic_matched_new_median\t{np.median(ICCq):.4f}\n"
             f"ic_matched_new_max\t{ICCq.max():.4f}\n"
             f"ic_matched_old_median\t{np.median(OICCq):.4f}\n"
             f"ic_matched_old_max\t{OICCq.max():.4f}\n"
             f"ic_per_col_old_min\t{OICC.min():.4f}\nic_per_col_old_max\t{OICC.max():.4f}\n"
             f"ic_per_col_old_median\t{np.median(OICC):.4f}\nfilters_sharper\t{int(SHARPER)}\n"
             f"n_matched_new\t{nm_}\nn_matched_old\t{int((OLD['best_match']!='none').sum())}\n"
             f"acceptor_new\t{acc_new:.5f}\nnew_motifs\t{'none' if not NEW else len(NEW)}\n")
    if PERF == "improved" and not SHARPER and not NEW:
        v = (f"The motif-friendly architecture ranks BETTER: within-gene {R['winrate']:.4f} "
             f"against {REP['winrate']:.4f} ({d.mean():+.4f} [{DLO:+.4f},{DHI:+.4f}]), but its "
             f"filters are not sharper count-matched ({np.median(ICCq):.3f} against "
             f"{np.median(OICCq):.3f} bits/column) and no motif beyond the acceptor registers. "
             f"The gain is in the trunk, not a motif code.")
    elif PERF == "improved" and SHARPER and not NEW:
        v = (f"The variant both ranks better ({d.mean():+.4f}) and carries sharper filters "
             f"({np.median(ICCq):.3f} against {np.median(OICCq):.3f} bits/column).")
    elif HOLDS and SHARPER and not NEW:
        v = ("A motif-friendly architecture gives cleaner filters but recovers the same elements; "
             "no new motif registers. Cosmetic: keep the reported model.")
    elif NEW:
        v = f"A motif absent from the reported model registers on the variant: {NEW}"
    elif PERF == "degraded":
        v = (f"The architecture is not free: within-gene {R['winrate']:.4f} against "
             f"{REP['winrate']:.4f}, a change of {d.mean():+.4f}.")
    else:
        v = (f"Performance holds but filters are not sharper (median {np.median(ICCq):.3f} against "
             f"{np.median(OICCq):.3f} bits/column, count-matched); no reason to switch.")
    fh.write(f"verdict\t{v}\n")
log(f"\n[VERDICT] {v}")
log("[note] bpnet_verdict.py recomputes this from the TSVs in seconds and is authoritative")
log("[out] bpnet_verdict.tsv bpnet_gia.tsv bpnet_model.pt")
