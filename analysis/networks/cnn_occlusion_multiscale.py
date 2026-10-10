#!/usr/bin/env python3
# is the positional concentration in the occlusion map real
import sys, os, time, argparse
import numpy as np
import multiprocessing as mp
from cnn_interp_lib import dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)

P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--data",  default="cnn_data.npz")
P.add_argument("--widths", default="25,50,100,200,400")
P.add_argument("--nrep", type=int, default=10, help="dinucleotide shuffles per window")
P.add_argument("--pool_sd", type=int, default=2, help="+/- windows pooled into the null sd")
P.add_argument("--batch", type=int, default=2048)
P.add_argument("--device", default="auto")
P.add_argument("--shuf_workers", type=int,
               default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1))
P.add_argument("--seed", type=int, default=42)
P.add_argument("--limit", type=int, default=0, help="smoke test: keep every Nth window")
A = P.parse_args()
WIDTHS = [int(w) for w in A.widths.split(",")]
RANKB  = [(1, 1, "rank 1"), (2, 3, "rank 2-3"), (4, 10, "rank 4-10"), (11, 10**9, "rank >10")]

# receptive field
r, j = 1, 1
for k, s in [(19, 1), (10, 10), (7, 1), (5, 5), (5, 1), (2, 2)]:
    r, j = r + (k - 1) * j, j * s
RF, STRIDE = r, j
assert (RF, STRIDE) == (378, 100), f"architecture changed: rf {RF} stride {STRIDE}"
log(f"[arch] final-bin receptive field {RF} nt, stride {STRIDE} nt -- a {STRIDE} nt mask lies "
    f"inside the receptive field of ~{RF//STRIDE} neighbouring bins")

# data
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
GRP   = [(np.array(by[g][0]), np.array(by[g][1])) for g in GNAME]
NG    = len(GRP)
log(f"[set] {NTE} test sites, {NG} test genes with both classes, "
    f"{int(YTE.sum())} real / {int((YTE == 0).sum())} competitor")

def winrate(s, idx=None):
    G = GRP if idx is None else [GRP[i] for i in idx]
    return float(np.mean([((s[rr][:, None] > s[cc][None, :]).sum()
                           + 0.5 * (s[rr][:, None] == s[cc][None, :]).sum())
                          / (len(rr) * len(cc)) for rr, cc in G]))

# shuffle pool: forked before any cuda context exists
def _shuf(args):
    a, b, rep, lo, hi = args
    out = np.empty((hi - lo, b - a), XTE.dtype)
    for i in range(lo, hi):
        out[i - lo] = dinuc_shuffle(XTE[i, a:b], np.random.default_rng([A.seed, b - a, a, rep, i]))
    return lo, out

SHUF = None
if A.shuf_workers > 0:
    SHUF = mp.get_context("fork").Pool(A.shuf_workers)
    log(f"[shuffle] {A.shuf_workers} numpy workers, forked before CUDA init")
CHUNKS = [(k, min(k + 512, NTE)) for k in range(0, NTE, 512)]

def shuffled(a, b, rep):
    out = np.empty((NTE, b - a), XTE.dtype)
    jobs = [(a, b, rep, lo, hi) for lo, hi in CHUNKS]
    for lo, blk in (SHUF.imap_unordered(_shuf, jobs) if SHUF else map(_shuf, jobs)):
        out[lo:lo + len(blk)] = blk
    return out

# model, on the gpu
import torch
DEV = torch.device(A.device if A.device != "auto"
                   else ("cuda" if torch.cuda.is_available() else "cpu"))
log(f"[device] {DEV}" + (f"  {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))
from cnn_model_def import Net
ck = torch.load(A.model, map_location="cpu", weights_only=False)
NET = Net(ck["args"]["dropout"]).to(DEV); NET.load_state_dict(ck["state"]); NET.eval()
log(f"[model] {A.model} epoch {ck['epoch']} seed {ck['args']['seed']} (cnn_model_def.Net)")

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(DEV)
OH = torch.from_numpy(np.ascontiguousarray(TBL[XTE].transpose(0, 2, 1))).to(DEV)
log(f"[mem] one-hot {tuple(OH.shape)} {OH.element_size()*OH.nelement()/1e9:.2f} GB on {DEV}")

def score_all(mask=None, sub=None):
    out = np.empty(NTE, np.float64)
    a, b = mask if mask is not None else (sub[:2] if sub is not None else (0, 0))
    with torch.inference_mode():
        for k in range(0, NTE, A.batch):
            j = min(k + A.batch, NTE); v = OH[k:j]
            if mask is None and sub is None:
                out[k:j] = NET(v).float().cpu().numpy(); continue
            keep = v[:, :, a:b].clone()
            if mask is not None:
                v[:, :, a:b] = 0.25
            else:
                v[:, :, a:b] = TBLt[torch.from_numpy(sub[2][k:j].astype(np.int64)).to(DEV)] \
                                   .permute(0, 2, 1)
            out[k:j] = NET(v).float().cpu().numpy()
            v[:, :, a:b] = keep
    return out

t0 = time.time(); BASE = score_all(); TPASS = time.time() - t0
REF = np.load("cnn_model_scores.npz", allow_pickle=True)["logit"][ok][te].astype(np.float64)
WG0, WGR = winrate(BASE), winrate(REF)
log(f"[check] re-scored vs cnn_model_scores.npz: rms {np.sqrt(((BASE-REF)**2).mean()):.2e} "
    f"max {np.abs(BASE-REF).max():.2e} r {np.corrcoef(BASE, REF)[0,1]:.8f}; "
    f"within-gene fresh {WG0:.6f} cached {WGR:.6f}")
assert abs(WG0 - WGR) < 1e-6 and np.corrcoef(BASE, REF)[0, 1] > 0.9999, \
    "this is not the model that produced cnn_model_scores.npz"
assert abs(WG0 - 0.82110) < 5e-5, f"reported CNN is within-gene 0.82110, got {WG0:.5f}"
log(f"[intact] within-gene {WG0:.5f}  logit sd {BASE.std():.3f}  ({TPASS:.2f}s / full pass)")

RANK = np.empty(NG, int); NSITE = np.empty(NG, int)
for i, (rr, cc) in enumerate(GRP):
    t = BASE[rr[0]]
    RANK[i] = 1 + int((BASE[cc] > t).sum()) + int(np.ceil(0.5 * (BASE[cc] == t).sum()))
    NSITE[i] = len(rr) + len(cc)
STRATA = []
for lo, hi, lab in RANKB:
    idx = np.flatnonzero((RANK >= lo) & (RANK <= hi))
    rows = np.concatenate([np.r_[GRP[i][0], GRP[i][1]] for i in idx]) if len(idx) else np.array([], int)
    STRATA.append(dict(lab=lab, lo=lo, idx=idx, rows=rows,
                       wg0=winrate(BASE, idx) if len(idx) else np.nan,
                       nsite=float(NSITE[idx].mean()) if len(idx) else np.nan))
    log(f"[rank] {lab:9s} {len(idx):5d} genes ({100*len(idx)/NG:4.1f}%)  "
        f"mean sites/gene {STRATA[-1]['nsite']:5.1f}  within-gene {STRATA[-1]['wg0']:.4f}")
assert sum(len(s["idx"]) for s in STRATA) == NG

try:
    C0 = np.load("analysis_columns.npz", allow_pickle=True)
    ACC = C0["spl_upstream_max_me3"][ok][te]
    for s in STRATA:
        rr = np.array([GRP[i][0][0] for i in s["idx"]])
        cc = np.concatenate([GRP[i][1] for i in s["idx"]]) if len(s["idx"]) else np.array([], int)
        s["acc_true"] = float(np.nanmean(ACC[rr])) if len(rr) else np.nan
        s["acc_comp"] = float(np.nanmean(ACC[cc])) if len(cc) else np.nan
        log(f"[acc]  {s['lab']:9s} mean upstream MaxEnt 3'ss  true {s['acc_true']:6.3f}  "
            f"competitor {s['acc_comp']:6.3f}  delta {s['acc_true']-s['acc_comp']:+.3f}")
except Exception as e:
    log(f"[acc]  skipped ({e})")
    for s in STRATA: s["acc_true"] = s["acc_comp"] = np.nan

TASKS = []
for w in WIDTHS:
    step = w if w <= 25 else w // 2
    for a in range(0, L - w + 1, step):
        TASKS.append((w, a))
if A.limit: TASKS = TASKS[::A.limit]; log(f"[SMOKE] every {A.limit}th window kept")
log(f"[plan] {len(TASKS)} windows over {len(WIDTHS)} scales, {A.nrep} shuffles each "
    f"-> {len(TASKS)*(1+A.nrep)} forward passes, ~{len(TASKS)*(1+A.nrep)*TPASS/60:.1f} min of GPU")

RES = []; t0 = time.time()
for n, (w, a) in enumerate(TASKS, 1):
    b = a + w
    s = score_all(mask=(a, b)); d = s - BASE
    out = dict(w=w, a=a, b=b, wg=winrate(s), mabs=float(np.abs(d).mean()),
               dreal=float(d[YTE == 1].mean()), dcomp=float(d[YTE == 0].mean()))
    shw, shm = [], []
    for rep in range(A.nrep):
        ss = score_all(sub=(a, b, shuffled(a, b, rep)))
        shw.append(winrate(ss)); shm.append(float(np.abs(ss - BASE).mean()))
    out["shuf_wg"] = np.asarray(shw, float)
    out["shuf_wg_mean"] = float(np.mean(shw)); out["shuf_wg_sd"] = float(np.std(shw, ddof=1))
    out["shuf_mabs_mean"] = float(np.mean(shm))
    if w == 100: out["per_site"] = d.astype(np.float32)
    RES.append(out)
    if n % 20 == 0 or n == len(TASKS):
        el = time.time() - t0
        log(f"  {n}/{len(TASKS)} windows  {el/60:.1f} min  eta {(el/n*(len(TASKS)-n))/60:.1f} min")
if SHUF: SHUF.close(); SHUF.join()
RES.sort(key=lambda r: (r["w"], r["a"]))
log(f"[scan] done in {(time.time()-t0)/60:.1f} min")

for w in WIDTHS:
    rw = [r for r in RES if r["w"] == w]
    var = np.array([r["shuf_wg"].var(ddof=1) for r in rw])
    for i, r in enumerate(rw):
        lo, hi = max(0, i - A.pool_sd), min(len(rw), i + A.pool_sd + 1)
        r["shuf_sd_pooled"] = float(np.sqrt(var[lo:hi].mean()))
        r["sd_df"] = (hi - lo) * (A.nrep - 1)
log(f"[null] sd pooled over +/-{A.pool_sd} windows, up to "
    f"{(2*A.pool_sd+1)*(A.nrep-1)} df, against {A.nrep-1} df unpooled")

with open("occlusion_multiscale.tsv", "w") as fh:
    fh.write("width\tstart\tend\tcentre\twinrate\td_winrate\td_winrate_per100\t"
             "shuf_winrate\td_winrate_shuf\tshuf_sd\tshuf_sd_pooled\tsd_df\tz\t"
             "mean_abs_dlogit\tmean_abs_dlogit_shuf\tmean_dlogit_real\tmean_dlogit_comp\t"
             "intact_winrate\trf_nt\tn_rep\tn_test\tn_genes\n")
    for r in RES:
        dobs = WG0 - r["wg"]
        dsh  = WG0 - r["shuf_wg_mean"]
        z    = (dobs - dsh) / max(r["shuf_sd_pooled"], 1e-9)
        fh.write(f"{r['w']}\t{r['a']-UP}\t{r['b']-UP}\t{(r['a']+r['b'])/2-UP:.0f}\t"
                 f"{r['wg']:.6f}\t{dobs:.6f}\t{dobs/(r['w']/100.0):.6f}\t"
                 f"{r['shuf_wg_mean']:.6f}\t{dsh:.6f}\t{r['shuf_wg_sd']:.6f}\t"
                 f"{r['shuf_sd_pooled']:.6f}\t{r['sd_df']}\t{z:.4f}\t"
                 f"{r['mabs']:.6f}\t{r['shuf_mabs_mean']:.6f}\t"
                 f"{r['dreal']:.6f}\t{r['dcomp']:.6f}\t"
                 f"{WG0:.6f}\t{RF}\t{A.nrep}\t{NTE}\t{NG}\n")
log("[out] occlusion_multiscale.tsv")

GRID = np.arange(-2000, 501, 25.0)
def prof(w, key):
    rr = [r for r in RES if r["w"] == w]
    c  = np.array([(r["a"] + r["b"]) / 2 - UP for r in rr])
    if key == "z":
        v = np.array([((WG0 - r["wg"]) - (WG0 - r["shuf_wg_mean"]))
                      / max(r["shuf_sd_pooled"], 1e-9) for r in rr])
    elif key == "d100":
        v = np.array([(WG0 - r["wg"]) / (r["w"] / 100.0) for r in rr])
    else:
        v = np.array([WG0 - r["shuf_wg_mean"] for r in rr])
    return c, v, np.interp(GRID, c, v)

log("\n[A] MULTI-SCALE VERDICT")
log(f"{'scale':>6} {'n_win':>6} {'peak z':>8} {'max z':>8} {'peak drop/100nt':>16} {'centroid z':>11}")
PK = {}
for w in WIDTHS:
    c, v, gz = prof(w, "z")
    _, d100, gd = prof(w, "d100")
    pz = GRID[int(np.argmax(gz))]; pd = GRID[int(np.argmax(gd))]
    pos = np.maximum(gz, 0); cen = float((GRID * pos).sum() / max(pos.sum(), 1e-9))
    PK[w] = dict(pz=pz, pd=pd, cen=cen, gz=gz, maxz=float(gz.max()))
    log(f"{w:>6} {len(c):>6} {pz:>+8.0f} {gz.max():>8.2f} {pd:>+16.0f} {cen:>+11.0f}")

MINZ = 3.0
AD = [w for w in WIDTHS if PK[w]["maxz"] >= MINZ] or WIDTHS
if len(AD) < len(WIDTHS):
    log(f"[A] scales excluded for max z < {MINZ} (probe too narrow to register): "
        + ", ".join(f"{w}nt max z {PK[w]['maxz']:.2f}" for w in WIDTHS if w not in AD))
pzs = np.array([PK[w]["pz"] for w in AD]); cens = np.array([PK[w]["cen"] for w in AD])
Zs  = np.vstack([PK[w]["gz"] for w in AD])
IU  = np.triu_indices(len(AD), 1)
CCV = np.corrcoef(Zs)[IU] if len(AD) > 1 else np.array([1.0])
pk_txt = ", ".join("%dnt %+.0f" % (w, PK[w]["pz"]) for w in AD)
log(f"[A] peak z position across scales: {pk_txt}")
log(f"[A] spread of peak positions: {pzs.max()-pzs.min():.0f} nt "
    f"(one final-bin receptive field is {RF} nt)")
log(f"[A] positive-mass centroid across scales: {cens.min():+.0f} to {cens.max():+.0f} "
    f"(spread {cens.max()-cens.min():.0f} nt)")
log(f"[A] pairwise correlation of the {len(AD)} adequate z profiles: min {CCV.min():.3f} "
    f"median {np.median(CCV):.3f}")
agree = (pzs.max() - pzs.min()) <= RF and CCV.min() > 0.5
log(f"[A] VERDICT: {'CONCENTRATION REAL' if agree else 'SCALE-DEPENDENT (probe artifact)'} "
    f"-- peak spread {'<=' if (pzs.max()-pzs.min())<=RF else '>'} one receptive field, "
    f"min profile correlation {CCV.min():.2f}")

c100, v100, _ = prof(100, "d100")
pv = np.maximum(v100, 0); o = np.argsort(-pv)
k = int(np.searchsorted(np.cumsum(pv[o]), 0.5 * pv.sum())) + 1
log(f"[A] at 100 nt, {k} of {len(c100)} windows carry half the summed positive drop "
    f"(uniform would be {len(c100)//2}); the largest single window is "
    f"[{c100[o[0]]-50:+.0f},{c100[o[0]]+50:+.0f}) at {100*pv[o[0]]/pv.sum():.1f}% of the total")

R100 = [r for r in RES if r["w"] == 100]
assert all("per_site" in r for r in R100), "per-site deltas missing at the 100 nt scale"
with open("occlusion_by_rank.tsv", "w") as fh:
    fh.write("stratum\tn_genes\tmean_sites_per_gene\tstratum_intact_winrate\tacc_true\tacc_comp\t"
             "start\tend\tcentre\twinrate\td_winrate\tmean_abs_dlogit\tmean_dlogit_true\t"
             "mean_dlogit_comp\tshare_of_drop\twidth\tn_test\tn_genes_total\n")
    for s in STRATA:
        if not len(s["idx"]): continue
        dw = np.array([s["wg0"] - winrate(BASE + r["per_site"], s["idx"]) for r in R100])
        pos = np.maximum(dw, 0); tot = max(pos.sum(), 1e-12)
        trows = np.array([GRP[i][0][0] for i in s["idx"]])
        crows = np.concatenate([GRP[i][1] for i in s["idx"]])
        for r, d, sh in zip(R100, dw, pos / tot):
            ps = r["per_site"]
            fh.write(f"{s['lab']}\t{len(s['idx'])}\t{s['nsite']:.2f}\t{s['wg0']:.6f}\t"
                     f"{s['acc_true']:.4f}\t{s['acc_comp']:.4f}\t"
                     f"{r['a']-UP}\t{r['b']-UP}\t{(r['a']+r['b'])/2-UP:.0f}\t"
                     f"{s['wg0']-d:.6f}\t{d:.6f}\t{np.abs(ps[s['rows']]).mean():.6f}\t"
                     f"{ps[trows].mean():.6f}\t{ps[crows].mean():.6f}\t{sh:.6f}\t"
                     f"100\t{len(s['rows'])}\t{NG}\n")
        s["dw"] = dw
log("[out] occlusion_by_rank.tsv")

log("\n[B] RANK-STRATIFIED PROFILE")
log("[B] note: the rank-1 stratum has intact within-gene 1.0000 BY CONSTRUCTION (rank 1 means "
    "the true site beats every competitor), so its drop is one-sided; the other strata can move "
    "either way")
cen100 = np.array([(r["a"] + r["b"]) / 2 - UP for r in R100])
AGG = np.array([WG0 - r["wg"] for r in R100])
AGGp = np.maximum(AGG, 0) / max(np.maximum(AGG, 0).sum(), 1e-12)
for s in STRATA:
    if "dw" not in s: continue
    p = np.maximum(s["dw"], 0); p = p / max(p.sum(), 1e-12)
    pk = float(cen100[int(np.argmax(s["dw"]))])
    log(f"  {s['lab']:9s} n={len(s['idx']):5d}  peak {pk:+6.0f}  "
        f"max drop {s['dw'].max():+.4f}  summed positive drop {np.maximum(s['dw'],0).sum():.4f}  "
        f"shape r vs aggregate {float(np.corrcoef(p, AGGp)[0,1]):+.3f}")
log("[B] d_winrate per stratum is confounded: the strata are defined BY the model's ranking, so "
    "rank 1 starts at 1.0000 and can only fall while rank 2-3 starts at 0.4812 and every mask "
    "RAISES it. occlusion_verdict.py redoes part B on mean_dlogit_true - mean_dlogit_comp, which "
    "is bounded by neither, and that is the reading to quote.")
