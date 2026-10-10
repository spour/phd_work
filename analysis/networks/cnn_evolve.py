#!/usr/bin/env python3
# in-silico evolution against the reported cnn (cnn_model.pt)
import sys, time, argparse, itertools, math
import numpy as np
import torch
from cnn_interp_lib import NT, dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nevolve", type=int, default=64)
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--ins_bg", type=int, default=128)
P.add_argument("--steps_whole", type=int, default=400)
P.add_argument("--steps_site", type=int, default=150)
P.add_argument("--steps_ins", type=int, default=150)
P.add_argument("--topk", type=int, default=192)
P.add_argument("--nrestart", type=int, default=3)
P.add_argument("--nscram", type=int, default=10)
P.add_argument("--patience", type=int, default=40)
P.add_argument("--bs", type=int, default=768)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--direction", choices=("max", "min"), default="max")
P.add_argument("--start", choices=("competitor", "canonical"), default="competitor")
P.add_argument("--suffix", default="")
A = P.parse_args()
SGN = 1.0 if A.direction == "max" else -1.0
SUF = A.suffix
WORD = "beats" if A.direction == "max" else "falls below"
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
SITE = (UP - 250, UP)
rng = np.random.default_rng(A.seed)
log(f"[env] device {dev}  torch {torch.__version__}")

# model and data
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
assert X.shape[1] == WIN, X.shape
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)

def oh(a):
    return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()

def fwd(a, bs=None):
    bs = bs or A.bs; out = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs):
            out[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return out

def grad(a, bs=128):
    G = np.empty((len(a), 4, WIN), np.float32)
    for k in range(0, len(a), bs):
        v = oh(a[k:k+bs]).clone().requires_grad_(True)
        net(v).sum().backward()
        G[k:k+bs] = v.grad.detach().float().cpu().numpy()
    return G

# shuffles and composition
def mono(s, key):
    return np.random.default_rng(key).permutation(np.asarray(s))

def binshuf(s, key, nbin=25):
    r = np.random.default_rng(key); o = np.asarray(s).copy(); w = len(s) // nbin
    for b in range(nbin): o[b*w:(b+1)*w] = r.permutation(o[b*w:(b+1)*w])
    return o

def dinuc(s, key):
    return dinuc_shuffle(np.asarray(s, np.uint8), np.random.default_rng(key))

def frac(s):
    s = np.asarray(s); return np.array([(s == i).mean() for i in range(4)])
def at(s): f = frac(s); return float(f[0] + f[3])
def ent(s):
    f = frac(s); f = f[f > 0]; return float(-(f * np.log2(f)).sum())
def runs(s):
    s = np.asarray(s); brk = np.flatnonzero(np.diff(s) != 0)
    L = np.diff(np.concatenate(([-1], brk, [len(s)-1])))
    return float(L.max()), float(L[L >= 4].sum() / len(s))
def txt(s): return "".join(NT[i] for i in np.asarray(s))

# the evolution loop
def evolve(seqs, region, steps, exact, tag):
    a, b = region; L = b - a; N = len(seqs)
    cur = np.ascontiguousarray(seqs.copy()); start = cur.copy()
    base = fwd(cur); score = base.copy()
    traj = [(0, float(score.mean()), float(score.mean() - base.mean()))]
    if exact:
        pp0 = a + np.repeat(np.arange(L), 4); bb0 = np.tile(np.arange(4), L); nc = 4 * L
    else:
        nc = A.topk
    stall = 0; t0 = time.time()
    for t in range(1, steps + 1):
        if exact:
            M = np.repeat(cur, nc, axis=0)
            M[np.arange(N * nc), np.tile(pp0, N)] = np.tile(bb0, N).astype(np.uint8)
        else:
            g = grad(cur)[:, :, a:b]
            own = np.take_along_axis(g, cur[:, None, a:b].astype(np.int64), 1)
            gain = (g - own).reshape(N, -1)
            sel = np.argpartition(-SGN * gain, nc - 1, axis=1)[:, :nc]
            pp = a + (sel % L); bb = (sel // L).astype(np.uint8)
            M = np.repeat(cur, nc, axis=0)
            M[np.arange(N * nc), pp.ravel()] = bb.ravel()
        s = fwd(M).reshape(N, nc)
        j = (SGN * s).argmax(1); nw = s[np.arange(N), j]
        gained = SGN * nw > SGN * score + 1e-9
        if exact: pmut, bmut = a + (j // 4), (j % 4).astype(np.uint8)
        else:     pmut, bmut = pp[np.arange(N), j], bb[np.arange(N), j]
        idx = np.flatnonzero(gained)
        cur[idx, pmut[idx]] = bmut[idx]; score[idx] = nw[idx]
        traj.append((t, float(score.mean()), float((score - base).mean())))
        stall = 0 if gained.any() else stall + 1
        if t % 25 == 0 or stall >= A.patience:
            log(f"[{tag}] step {t:4d}  logit {score.mean():+.3f}  gain {np.mean(score-base):+.3f}  "
                f"moved {gained.sum():3d}/{N}  {time.time()-t0:.0f}s")
        if stall >= A.patience:
            log(f"[{tag}] converged: no sequence improved for {A.patience} steps"); break
    return cur, start, base, score, traj

# arms 1 and 2
SC = 1 if A.start == "canonical" else 0
BG = X[rng.choice(te[y[te] == SC], A.nevolve, replace=False)]
log(f"\n[start] {A.direction}imising the logit from {A.nevolve} held-out real "
    f"{A.start} windows, mean logit {fwd(BG).mean():+.3f}")
log(f"[arm] whole : 2500 nt, gradient-proposed {A.topk} candidates scored exactly, "
    f"<= {A.steps_whole} steps")
Wc, Ws, Wb, Wsc, Wtr = evolve(BG, (0, WIN), A.steps_whole, False, "whole")
log(f"[arm] site  : [-250, 0], EXHAUSTIVE {4*250} substitutions per step, <= {A.steps_site} steps")
Sc, Ss, Sb, Ssc, Str = evolve(BG, SITE, A.steps_site, True, "site")

def shuffle_arm(evolved, startseq, base, region, kinds, tag):
    a, b = region; out = {}
    for kind, fn in kinds:
        acc = np.zeros((A.nscram, len(evolved)))
        for j in range(A.nscram):
            M = startseq.copy()
            for i in range(len(evolved)):
                M[i, a:b] = fn(evolved[i, a:b], [A.seed, sum(map(ord, tag + kind)), i, j])
            acc[j] = fwd(M) - base
        out[kind] = acc
    return out

log(f"\n[control] {A.nscram} shuffles per evolved sequence, replaced in its own background")
WSH = shuffle_arm(Wc, Ws, Wb, (0, WIN), [("mono", mono), ("binshuf", binshuf), ("dinuc", dinuc)], "whole")
SSH = shuffle_arm(Sc, Ss, Sb, SITE, [("mono", mono), ("dinuc", dinuc)], "site")

rows = []
for tag, ev, bs_, sc, sh in (("whole", Wc, Wb, Wsc, WSH), ("site", Sc, Sb, Ssc, SSH)):
    g = sc - bs_
    log(f"\n[{tag}] evolved gain {g.mean():+.4f} +/- {g.std(ddof=1)/np.sqrt(len(g)):.4f} "
        f"(per-sequence range {g.min():+.3f} to {g.max():+.3f})")
    for kind, acc in sh.items():
        c = acc.mean(0); d = g - c
        log(f"[{tag}] its own {kind:>7} shuffle {c.mean():+.4f}   "
            f"evolved minus shuffle {d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.4f}   "
            f"{WORD} it in {(SGN*d>0).mean()*100:.0f}% of sequences")
        rows.append((tag, kind, float(g.mean()), float(c.mean()), float(d.mean()),
                     float(d.std(ddof=1)/np.sqrt(len(d))), float((d > 0).mean())))

a, b = SITE
EVW = Sc[:, a:b]; STW = Ss[:, a:b]
log(f"\n[what] site window [-250, 0], {len(EVW)} sequences x {b-a} nt")
for nm, W in (("starting", STW), ("evolved ", EVW)):
    f = np.stack([frac(w) for w in W]).mean(0); e = np.mean([ent(w) for w in W])
    r = np.array([runs(w) for w in W]).mean(0)
    log(f"  {nm}  A {f[0]:.3f} C {f[1]:.3f} G {f[2]:.3f} T {f[3]:.3f}   A+T {f[0]+f[3]:.3f}   "
        f"entropy {e:.3f} bits   longest run {r[0]:.1f}   in runs>=4 {r[1]*100:.0f}%")
fw = np.stack([frac(w) for w in Wc]).mean(0)
log(f"  whole-window evolved, all 2500 nt: A {fw[0]:.3f} C {fw[1]:.3f} G {fw[2]:.3f} T {fw[3]:.3f}"
    f"   A+T {fw[0]+fw[3]:.3f}   entropy {np.mean([ent(w) for w in Wc]):.3f} bits")

PWM = np.stack([(EVW == i).mean(0) for i in range(4)])
IC = 2.0 + np.where(PWM > 0, PWM * np.log2(np.clip(PWM, 1e-12, None)), 0).sum(0)
with open(f"evolve{SUF}_pwm.tsv", "w") as fh:
    fh.write("offset\tA\tC\tG\tT\tic\n")
    for p in range(b - a):
        fh.write(f"{p-(b-a)}\t" + "\t".join(f"{PWM[i,p]:.4f}" for i in range(4)) + f"\t{IC[p]:.4f}\n")
k = 30; s_ = int(np.argmax(np.convolve(IC, np.ones(k), "valid")))
log(f"\n[logo] highest-information {k} nt of the site window, at offset "
    f"{s_-(b-a)} to {s_+k-(b-a)}, mean IC {IC[s_:s_+k].mean():.2f} bits of 2")
cons = "".join(NT[i] for i in PWM[:, s_:s_+k].argmax(0))
log(f"  consensus {cons}")
log("  " + "".join("ACGT"[i] if PWM[i, s_+q] > 0.5 else "." for q, i in
                   enumerate(PWM[:, s_:s_+k].argmax(0))) + "   (. = no base above 50%)")
for i in range(4):
    log(f"  {NT[i]} " + "".join("#" if PWM[i, s_+q] > .5 else "+" if PWM[i, s_+q] > .3
                                else "-" if PWM[i, s_+q] > .15 else " " for q in range(k)))
log(f"  IC" + "".join("#" if IC[s_+q] > 1.0 else "+" if IC[s_+q] > .5 else
                      "-" if IC[s_+q] > .2 else " " for q in range(k)))

K = 6
def kcount(W):
    c = np.zeros(4 ** K, np.int64)
    P4 = 4 ** np.arange(K - 1, -1, -1)
    for w in W:
        v = np.lib.stride_tricks.sliding_window_view(w.astype(np.int64), K) @ P4
        np.add.at(c, v, 1)
    return c
CE, CS = kcount(EVW), kcount(STW)
CM = kcount(np.stack([mono(w, [A.seed, 77, i]) for i, w in enumerate(EVW)]))
def nm_(v): return (v + 0.5) / (v.sum() + 0.5 * 4 ** K)
MINC = 20
with open(f"evolve{SUF}_kmers.tsv", "w") as fh:
    fh.write("kmer\tn_evolved\tn_starting\tn_evolved_shuffled\tlog2_vs_starting\tlog2_vs_shuffled\n")
    for i in np.argsort(-CE):
        if CE[i] == 0: break
        km = "".join(NT[(i // 4 ** (K-1-j)) % 4] for j in range(K))
        fh.write(f"{km}\t{CE[i]}\t{CS[i]}\t{CM[i]}\t"
                 f"{math.log2(nm_(CE)[i]/nm_(CS)[i]):.3f}\t"
                 f"{math.log2(nm_(CE)[i]/nm_(CM)[i]):.3f}\n")
log(f"\n[kmers] top {K}-mers seen at least {MINC} times in the evolved site windows "
    f"(count, log2 vs starting, log2 vs a shuffle OF THE EVOLVED windows -- the second column is "
    f"the arrangement test, since the shuffle holds composition)")
for i in [q for q in np.argsort(-CE) if CE[q] >= MINC][:15]:
    km = "".join(NT[(i // 4 ** (K-1-j)) % 4] for j in range(K))
    log(f"  {km}  {CE[i]:6d}  vs starting {math.log2(nm_(CE)[i]/nm_(CS)[i]):+6.2f}"
        f"  vs own shuffle {math.log2(nm_(CE)[i]/nm_(CM)[i]):+6.2f}")
KNOWN = {"AATAAA": "canonical PAS", "ATTAAA": "variant PAS", "TGTA": "CFIm UGUA",
         "GTAAGT": "U1 5' splice site", "TTTTTT": "poly-T", "TGTGTG": "GU-rich DSE",
         "CAG": "3' splice site AG", "GGGG": "G-quadruplex-like"}
log("[kmers] known elements, occurrences per evolved site window vs per starting window:")
for m, nm2 in KNOWN.items():
    ce = np.mean([txt(w).count(m) for w in EVW]); cs = np.mean([txt(w).count(m) for w in STW])
    log(f"  {m:<7} {nm2:<18} evolved {ce:6.2f}   starting {cs:6.2f}")

with open(f"evolve{SUF}_seqs.tsv", "w") as fh:
    fh.write("arm\ti\tstart_logit\tfinal_logit\tgain\tat\tentropy\tsequence\n")
    for tag, ev, bs_, sc, reg in (("whole", Wc, Wb, Wsc, (0, WIN)), ("site", Sc, Sb, Ssc, SITE)):
        for i in range(len(ev)):
            w = ev[i, reg[0]:reg[1]]
            fh.write(f"{tag}\t{i}\t{bs_[i]:.4f}\t{sc[i]:.4f}\t{sc[i]-bs_[i]:.4f}\t{at(w):.4f}\t"
                     f"{ent(w):.4f}\t{txt(w)}\n")
with open(f"evolve{SUF}_traj.tsv", "w") as fh:
    fh.write("arm\tstep\tmean_logit\tmean_gain\n")
    for tag, tr in (("whole", Wtr), ("site", Str)):
        for s in tr: fh.write(f"{tag}\t{s[0]}\t{s[1]:.5f}\t{s[2]:.5f}\n")

IBG = X[rng.choice(te[y[te] == SC], A.ins_bg, replace=False)]
IB = fwd(IBG); P0 = UP - 150
def evolve_insert(L, tag):
    best = (-SGN * 1e9, None)
    for r in range(A.nrestart):
        rr = np.random.default_rng([A.seed, L, r])
        s = rr.integers(0, 4, L).astype(np.uint8)
        def sc(cands):
            M = np.repeat(IBG, len(cands), axis=0).reshape(len(IBG), len(cands), WIN)
            for j, c in enumerate(cands): M[:, j, P0-L//2:P0-L//2+L] = c
            return (fwd(M.reshape(-1, WIN)).reshape(len(IBG), len(cands))
                    - IB[:, None]).mean(0)
        cur = float(sc([s])[0]); stall = 0
        for t in range(1, A.steps_ins + 1):
            cands = [s.copy() for _ in range(4 * L)]
            for p in range(L):
                for bq in range(4): cands[4*p+bq][p] = bq
            v = sc(cands); j = int((SGN * v).argmax())
            if SGN * v[j] > SGN * cur + 1e-9: s, cur, stall = cands[j], float(v[j]), 0
            else: stall += 1
            if stall >= 3: break
        log(f"[insert L={L}] restart {r}: {txt(s)}  gain {cur:+.4f}  ({t} steps)")
        if SGN * cur > SGN * best[0]: best = (cur, s.copy())
    log(f"[insert L={L}] {tag} winner {txt(best[1])}  gain {best[0]:+.4f}  A+T {at(best[1]):.2f}")
    return best[1]

log(f"\n[arm] insert: shared inserts evolved on {A.ins_bg} backgrounds at -150, "
    f"{A.nrestart} restarts, then implanted by the motif protocol")
EV23 = evolve_insert(23, "length-matched to the MaxEnt-optimal acceptor")
EV6 = evolve_insert(6, "length-matched to AATAAA")

UNITS = {"evolved_23": txt(EV23), "evolved_6": txt(EV6), "AATAAA": "AATAAA",
         "acc_best": "TTTCCTCTCTCTCCCCGCAGGTG", "acc_ppt": "TTTTTTTTTTTTTCAGG",
         "donor": "CAGGTGAGT"}
BGC = X[rng.choice(te[y[te] == 1], A.gia_bg, replace=False)]
BGX = X[rng.choice(te[y[te] == 0], A.gia_bg, replace=False)]
POS = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def scan(B, base, m):
    bst = (None, -SGN * 1e9, 0.0)
    for p in POS:
        s0 = p + UP - len(m) // 2
        if s0 < 0 or s0 + len(m) > WIN: continue
        q = B.copy(); q[:, s0:s0+len(m)] = m
        d = fwd(q) - base
        mm = float(d.mean())
        if SGN * mm > SGN * bst[1]: bst = (p, mm, float(d.std(ddof=1) / np.sqrt(len(d))))
    return bst
log(f"\n[contrast] {len(UNITS)} units x {len(POS)} positions x 2 real backgrounds x {A.gia_bg}, "
    f"each with {A.nscram} mononucleotide-shuffle controls")
CT = []
for bn, B in (("real_canonical", BGC), ("real_competitor", BGX)):
    base = fwd(B)
    for k_, s_ in UNITS.items():
        m = tok(s_); p, v, se = scan(B, base, m)
        cs = np.array([scan(B, base, mono(m, [A.seed, 555, sum(map(ord, k_)), j]))[1]
                       for j in range(A.nscram)])
        CT.append((bn, k_, s_, len(s_), at(m), p, v, se, float(cs.mean()),
                   float(cs.std(ddof=1)), v - float(cs.mean())))
with open(f"evolve{SUF}_contrast.tsv", "w") as fh:
    fh.write("background\tunit\tsequence\tlength\tat\tbest_position\tdelta_logit\tse\t"
             "mono_shuffle_mean\tmono_shuffle_sd\tminus_composition\n")
    for r in CT:
        fh.write("\t".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in r) + "\n")
with open(f"evolve{SUF}_summary.tsv", "w") as fh:
    fh.write("arm\tcontrol\tevolved_gain\tcontrol_gain\tdifference\tse\tfrac_beating\n")
    for r in rows: fh.write("\t".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in r) + "\n")

ORD = ["acc_best", "acc_ppt", "donor", "AATAAA", "evolved_6", "evolved_23"]
for bn, bl in (("real_canonical", "Real 3' end background"),
               ("real_competitor", "Real competitor background")):
    log(f"\n[{bl}]  {'most positive' if SGN>0 else 'most negative'} effect over the position "
        f"grid, and that effect minus the unit's own base composition")
    log(f"  {'unit':>11} {'len':>3} {'A+T':>5} {'pos':>6} {'raw':>9} {'composition':>12} {'raw-comp':>10}")
    for k_ in ORD:
        r = [x for x in CT if x[0] == bn and x[1] == k_][0]
        log(f"  {k_:>11} {r[3]:>3} {r[4]:>5.2f} {r[5]:>6} {r[6]:>+9.4f} {r[8]:>+9.4f}+/-{r[9]:<4.2f}"
            f" {r[10]:>+10.4f}")
MB = [x[6] for x in CT if x[1] in ("acc_best", "acc_ppt", "donor", "AATAAA")]
log(f"\n[read] whole-window and site-local changes are in the same logit units as the table "
    f"above; the real motifs reach {(max if SGN>0 else min)(MB):+.3f}.")
log("[out] " + " ".join(f"evolve{SUF}_{f}.tsv" for f in
    ("traj", "seqs", "pwm", "kmers", "contrast", "summary")))
