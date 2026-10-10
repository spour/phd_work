#!/usr/bin/env python3
# integrated gradients positional profile, as a cross-check on the occlusion map
import sys, os, time, argparse
import numpy as np
from cnn_interp_lib import dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)

P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--data",  default="cnn_data.npz")
P.add_argument("--steps", type=int, default=128)
P.add_argument("--batch", type=int, default=256)
P.add_argument("--device", default="auto")
P.add_argument("--smooth", type=int, default=51, help="moving-average width, odd")
P.add_argument("--seed", type=int, default=42)
P.add_argument("--nsite", type=int, default=0, help="smoke test: keep every Nth test site")
A = P.parse_args()
assert A.smooth % 2 == 1

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
UP, L = int(D["up"]), X.shape[1]
te = np.flatnonzero(split == "test")
XTE = np.ascontiguousarray(X[te]); YTE = y[te]; GTE = gene[te]; NTE = len(te)

by = {}
for k, g in enumerate(GTE):
    rr, cc = by.setdefault(g, ([], []))
    (rr if YTE[k] == 1 else cc).append(k)
GNAME = [g for g, (rr, cc) in by.items() if rr and cc]
GRP = [(np.array(by[g][0]), np.array(by[g][1])) for g in GNAME]
NG = len(GRP)
RANKB = [(1, 1, "rank 1"), (2, 3, "rank 2-3"), (4, 10, "rank 4-10"), (11, 10**9, "rank >10")]

def winrate(s):
    return float(np.mean([((s[rr][:, None] > s[cc][None, :]).sum()
                           + 0.5 * (s[rr][:, None] == s[cc][None, :]).sum())
                          / (len(rr) * len(cc)) for rr, cc in GRP]))

import torch
DEV = torch.device(A.device if A.device != "auto"
                   else ("cuda" if torch.cuda.is_available() else "cpu"))
log(f"[device] {DEV}" + (f"  {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))
from cnn_model_def import Net
ck = torch.load(A.model, map_location="cpu", weights_only=False)
NET = Net(ck["args"]["dropout"]).to(DEV); NET.load_state_dict(ck["state"]); NET.eval()
for prm in NET.parameters(): prm.requires_grad_(False)
log(f"[model] {A.model} epoch {ck['epoch']} (cnn_model_def.Net)")

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(DEV)
def onehot(codes):
    return TBLt[torch.from_numpy(codes.astype(np.int64)).to(DEV)].permute(0, 2, 1).contiguous()

BASE = np.empty(NTE)
with torch.inference_mode():
    for k in range(0, NTE, 2048):
        BASE[k:k + 2048] = NET(onehot(XTE[k:k + 2048])).float().cpu().numpy()
REF = np.load("cnn_model_scores.npz", allow_pickle=True)["logit"][ok][te].astype(np.float64)
WG0, WGR = winrate(BASE), winrate(REF)
log(f"[check] re-scored vs cnn_model_scores.npz: rms {np.sqrt(((BASE-REF)**2).mean()):.2e} "
    f"r {np.corrcoef(BASE, REF)[0,1]:.8f}; within-gene fresh {WG0:.6f} cached {WGR:.6f}")
assert abs(WG0 - WGR) < 1e-6 and abs(WG0 - 0.82110) < 5e-5, \
    "this is not the model that produced cnn_model_scores.npz"

SITE_STRATUM = np.full(NTE, -1, int)
for si, (lo, hi, lab) in enumerate(RANKB):
    for i, (rr, cc) in enumerate(GRP):
        t = BASE[rr[0]]
        rk = 1 + int((BASE[cc] > t).sum()) + int(np.ceil(0.5 * (BASE[cc] == t).sum()))
        if lo <= rk <= hi: SITE_STRATUM[np.r_[rr, cc]] = si
assert (SITE_STRATUM >= 0).all()
log(f"[set] {NTE} test sites, {NG} genes; stratum sizes " +
    ", ".join(f"{RANKB[s][2]}={int((SITE_STRATUM==s).sum())}" for s in range(4)))

ALPH = (np.arange(A.steps) + 0.5) / A.steps

def run(null):
    rows = np.arange(NTE)[::A.nsite] if A.nsite else np.arange(NTE)
    acc = np.zeros((6, L)); nrow = np.zeros(6); ce, ct = [], []
    t0 = time.time()
    for bi, k in enumerate(range(0, len(rows), A.batch)):
        idx = rows[k:k + A.batch]
        cd = XTE[idx]
        sh = np.stack([dinuc_shuffle(cd[i], np.random.default_rng([A.seed, int(r), 0]))
                       for i, r in enumerate(idx)])
        if null:
            cd2 = np.stack([dinuc_shuffle(cd[i], np.random.default_rng([A.seed, int(r), 1]))
                            for i, r in enumerate(idx)])
            x, xb = onehot(cd2), onehot(sh)
        else:
            x, xb = onehot(cd), onehot(sh)
        diff = x - xb
        g = torch.zeros_like(x)
        for a in ALPH:
            z = (xb + float(a) * diff).requires_grad_(True)
            (gr,) = torch.autograd.grad(NET(z).sum(), z)
            g += gr
        attr = (diff * (g / len(ALPH))).sum(1).cpu().numpy()
        with torch.inference_mode():
            f1 = NET(x).float().cpu().numpy(); f0 = NET(xb).float().cpu().numpy()
        ce.append(np.abs(attr.sum(1) - (f1 - f0))); ct.append(np.abs(f1 - f0))
        acc[0] += attr.sum(0);         nrow[0] += len(idx)
        acc[1] += np.abs(attr).sum(0); nrow[1] += len(idx)
        if not null:
            for s in range(4):
                m = SITE_STRATUM[idx] == s
                if m.any(): acc[2 + s] += np.abs(attr[m]).sum(0); nrow[2 + s] += int(m.sum())
        if (bi + 1) % 10 == 0:
            done = k + len(idx); el = time.time() - t0
            log(f"  {'null' if null else 'real'} {done}/{len(rows)} sites  {el/60:.1f} min  "
                f"eta {(el/done*(len(rows)-done))/60:.1f} min")
    ce, ct = np.concatenate(ce), np.concatenate(ct)
    log(f"[ig] {'null' if null else 'real'} completeness: median |sum attr - (f(x)-f(x'))| "
        f"{np.median(ce):.4f} against median |f(x)-f(x')| {np.median(ct):.4f} "
        f"({100*np.median(ce)/max(np.median(ct),1e-9):.2f}%)")
    return acc / np.maximum(nrow[:, None], 1)

log(f"[plan] {A.steps} IG steps x {NTE} sites x 2 (real + composition null)")
REAL = run(False)
NULL = run(True)

def smooth(v, w):
    return np.convolve(np.pad(v, w // 2, mode="edge"), np.ones(w) / w, mode="valid")

pos = np.arange(L) - UP
COLS = dict(ig_signed=REAL[0], ig_abs=REAL[1],
            ig_abs_rank1=REAL[2], ig_abs_rank23=REAL[3],
            ig_abs_rank410=REAL[4], ig_abs_rankgt10=REAL[5],
            ig_null_signed=NULL[0], ig_null_abs=NULL[1])
keys = list(COLS); SM = {k: smooth(v, A.smooth) for k, v in COLS.items()}
with open("ig_positional.tsv", "w") as fh:
    fh.write("pos\t" + "\t".join(keys) + "\t" + "\t".join(k + "_smooth" for k in keys) +
             "\tsmooth_nt\tn_steps\tn_test\tn_genes\n")
    for i in range(L):
        fh.write(f"{pos[i]}\t" + "\t".join(f"{COLS[k][i]:.6e}" for k in keys) + "\t" +
                 "\t".join(f"{SM[k][i]:.6e}" for k in keys) +
                 f"\t{A.smooth}\t{A.steps}\t{NTE}\t{NG}\n")
log("[out] ig_positional.tsv")

# agreement with occlusion
def bin100(v): return v.reshape(-1, 100).sum(1)
BC = np.arange(-2000, 500, 100) + 50
excess = bin100(REAL[1]) - bin100(NULL[1])
log("\n[IG] departure from the composition null, |IG| summed in 100 nt bins:")
log(f"[IG] total real {bin100(REAL[1]).sum():.3f} vs null {bin100(NULL[1]).sum():.3f} "
    f"(ratio {bin100(REAL[1]).sum()/max(bin100(NULL[1]).sum(),1e-9):.3f})")
o = np.argsort(-excess)
log("[IG] top 3 bins by excess over null: " + "  ".join(
    f"[{BC[i]-50:+d},{BC[i]+50:+d}) {excess[i]:+.4f}" for i in o[:3]))
try:
    T = np.genfromtxt("occlusion_multiscale.tsv", names=True, delimiter="\t")
    m = T["width"] == 100
    c, d, z = T["centre"][m], T["d_winrate"][m], T["z"][m]
    om = np.argsort(c)
    di, zi = np.interp(BC, c[om], d[om]), np.interp(BC, c[om], z[om])
    for nm, v in (("|IG|", bin100(REAL[1])), ("|IG| excess over null", excess)):
        log(f"[IG] {nm:22s} vs occlusion d_winrate r = {float(np.corrcoef(v, di)[0,1]):+.3f};  "
            f"vs occlusion z r = {float(np.corrcoef(v, zi)[0,1]):+.3f}")
    log(f"[IG] occlusion peak (100 nt, z) {BC[int(np.argmax(zi))]:+d}; "
        f"IG peak {BC[int(np.argmax(bin100(REAL[1])))]:+d}; "
        f"IG-excess peak {BC[int(np.argmax(excess))]:+d}")
except Exception as e:
    log(f"[IG] occlusion comparison skipped ({e})")
