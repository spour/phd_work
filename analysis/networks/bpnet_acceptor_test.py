#!/usr/bin/env python3
# does the bpnet variant read the 3' splice site, where the reported cnn does not?
import sys, math, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--bpnet", default="bpnet_model.pt")
P.add_argument("--cnn", default="cnn_model.pt")
P.add_argument("--npair", type=int, default=50)
P.add_argument("--ncomp", type=int, default=4096)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN, LFR = 2000, 2500, 120
POS = [-1500, -1000, -800, -600, -400, -200, -50, 0]
UPPOS = [p for p in POS if p <= -400]
rng = np.random.default_rng(A.seed)

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; chrom = D["chrom"][ok]; cen = D["center"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
log(f"[data] {len(te):,} held-out test peaks with a strict ACGT window "
    f"({int((y[te]==1).sum()):,} real 3' ends, {int((y[te]==0).sum()):,} competitors)")

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def loader(path, which):
    ck = torch.load(path, map_location=dev, weights_only=False)
    if which == "bpnet":
        from bpnet_model_def import BPNet
        a = ck["args"]; net = BPNet(a["dropout"], a["nf"], a["width"]).to(dev)
    else:
        from cnn_model_def import Net
        net = Net(ck["args"]["dropout"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net
NETS = {"BPNet": loader(A.bpnet, "bpnet"), "CNN": loader(A.cnn, "cnn")}
def fwd(net, a, bs=768):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o

# test 1
import pyarrow.parquet as pq
Pq = pq.read_table("cpa_peaks_final.parquet").to_pandas()
acc = {(g, c, int(x)): v for g, c, x, v in
       zip(Pq["gene"], Pq["chrom"], Pq["center"], Pq["spl_upstream_max_me3"])}
av = np.array([acc.get((gene[i], chrom[i], int(cen[i])), np.nan) for i in te])
keep = np.isfinite(av)
sub = te[keep]; av = av[keep]
cls = (y[sub] == 1); gg = gene[sub]
log(f"[test1] {len(sub):,} test peaks carry a real upstream acceptor score "
    f"({cls.sum():,} real ends, {(~cls).sum():,} competitors)")
SC = {k: fwd(v, X[sub]) for k, v in NETS.items()}

def pear(x, y_):
    n = len(x); r = float(np.corrcoef(x, y_)[0,1])
    z = 0.5*math.log((1+r)/(1-r)); se = 1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se)
def rk(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); r[o] = np.arange(len(v))
    s = v[o]; i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]: j += 1
        if j > i: r[o[i:j+1]] = (i+j)/2
        i = j+1
    return r
rows = []
log(f"\nTEST 1  score against REAL upstream acceptor strength, held-out test peaks")
log(f"  {'model':<7} {'scope':<26} {'n':>7} {'Pearson r':>10} {'95% CI':>20} {'Spearman':>9}")
uq, inv = np.unique(gg, return_inverse=True); cnt = np.bincount(inv)
for mdl in ("CNN", "BPNet"):
    L = SC[mdl]
    for nm, m in (("all test peaks", np.ones(len(L), bool)),
                  ("real 3' ends only", cls), ("competitors only", ~cls)):
        r, lo, hi = pear(av[m], L[m]); s, _, _ = pear(rk(av[m]), rk(L[m]))
        log(f"  {mdl:<7} {nm:<26} {m.sum():>7,} {r:>+10.4f} "
            f"{f'[{lo:+.4f}, {hi:+.4f}]':>20} {s:>+9.4f}")
        rows.append((mdl, nm, int(m.sum()), r, lo, hi, s))
    mA = np.bincount(inv, av)/cnt; mL = np.bincount(inv, L)/cnt
    k2 = cnt[inv] >= 2
    r, lo, hi = pear(av[k2]-mA[inv[k2]], L[k2]-mL[inv[k2]])
    s, _, _ = pear(rk(av[k2]-mA[inv[k2]]), rk(L[k2]-mL[inv[k2]]))
    log(f"  {mdl:<7} {'within gene':<26} {k2.sum():>7,} {r:>+10.4f} "
        f"{f'[{lo:+.4f}, {hi:+.4f}]':>20} {s:>+9.4f}")
    rows.append((mdl, "within gene", int(k2.sum()), r, lo, hi, s))
with open("bpnet_acceptor_test.tsv", "w") as f:
    f.write("model\tscope\tn\tpearson_r\tci_lo\tci_hi\tspearman\n")
    for r_ in rows: f.write("\t".join(str(x) for x in r_) + "\n")

log(f"\nTEST 1  mean score by decile of the real upstream acceptor score, WITHIN real 3' ends")
q = np.quantile(av[cls], np.linspace(0, 1, 11))
log(f"  {'decile':>6} {'acceptor':>13} {'n':>6} " + " ".join(f"{m:>10}" for m in ("CNN", "BPNet")))
for i in range(10):
    m = (av >= q[i]) & (av <= q[i+1]) & cls if i == 9 else (av >= q[i]) & (av < q[i+1]) & cls
    if m.sum() < 5: continue
    log(f"  {i+1:>6} {f'{q[i]:.2f}-{q[i+1]:.2f}':>13} {m.sum():>6,} "
        + " ".join(f"{SC[k][m].mean():>+10.3f}" for k in ("CNN", "BPNet")))

# test 5
FA = "/scratch/spour98/jobs/hg38.fa"
GTF = ("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome/"
       "GRCh38.gencode.v43.annotation.gtf")
FAI = {}
for line in open(FA + ".fai"):
    n_, ln, off, lb, wb = line.split()[:5]; FAI[n_] = (int(ln), int(off), int(lb), int(wb))
fh = open(FA, "rb")
def fetch(c, s, e):
    if c not in FAI: return None
    ln, off, lb, wb = FAI[c]
    if s < 0 or e > ln or e <= s: return None
    fh.seek(off + s//lb*wb + s%lb)
    return fh.read((e-s) + (e-s)//lb + 2).decode().replace("\n", "")[:e-s].upper()
def rc(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0
TX = {}
with open(GTF) as g:
    for line in g:
        if line[0] == "#": continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        i = f[8].find('transcript_id "'); tid = f[8][i+15:f[8].find('"', i+15)]
        TX.setdefault(tid, (f[0], f[6], []))[2].append((int(f[3]), int(f[4])))
ACC = []; seen = set()
for tid, (c, st, ex) in TX.items():
    if c not in FAI or len(ex) < 2: continue
    ex.sort()
    for k in range(len(ex)-1):
        istart, iend = ex[k][1]+1, ex[k+1][0]-1
        if iend-istart+1 < 300: continue
        key = (c, st, ex[k+1][0] if st == "+" else ex[k][1])
        if key in seen: continue
        s = (fetch(c, ex[k+1][0]-21, ex[k+1][0]+2) if st == "+"
             else (lambda z: rc(z) if z else None)(fetch(c, ex[k][1]-3, ex[k][1]+20)))
        if not s or len(s) != 23 or set(s)-set("ACGT"): continue
        seen.add(key)
        if s[18:20] != "AG": continue
        ACC.append((c, st, key[2], istart, iend, s3(s)))
ACC.sort(key=lambda a: -a[5])
def okq(z): return z and len(z) == LFR and not (set(z)-set("ACGT"))
PAIRS = []
for c, st, pos, istart, iend, sc in ACC:
    x = (fetch(c, pos-1-(LFR-3), pos+2) if st == "+"
         else (lambda z: rc(z) if z else None)(fetch(c, pos-3, pos-3+LFR)))
    m = (istart+iend)//2
    yq = None
    if m-istart >= 200 and iend-m >= 200:
        z = fetch(c, m-LFR//2, m-LFR//2+LFR); yq = z if st == "+" else (rc(z) if z else None)
    if okq(x) and okq(yq): PAIRS.append((x, yq))
    if len(PAIRS) >= A.npair: break
log(f"\n[pairs] {len(PAIRS)} introns giving both an acceptor fragment and an acceptor-free control")
def mono(s, k): return "".join(np.random.default_rng(k).permutation(list(s)))
SEQ = []
for i, (x, yq) in enumerate(PAIRS):
    SEQ += [(i, "acceptor", "motif", x), (i, "acceptor", "mono", mono(x, [A.seed, 1, i])),
            (i, "intron", "motif", yq), (i, "intron", "mono", mono(yq, [A.seed, 2, i]))]
BGC = X[te[y[te] == 1]]
cx = te[y[te] == 0]; BGX = X[rng.choice(cx, min(A.ncomp, len(cx)), replace=False)]
log(f"[bg] {len(BGC):,} real 3' ends and {len(BGX):,} competitors")
V = {}; prows = []; t0 = time.time()
for mdl in ("BPNet", "CNN"):
    net = NETS[mdl]
    for bn, B in (("real_canonical", BGC), ("real_competitor", BGX)):
        base = fwd(net, B)
        for i, c_, arm, s_ in SEQ:
            m = np.array([NT.index(ch) for ch in s_], np.uint8)
            for p_ in POS:
                s0 = p_ + UP - LFR//2
                q = B.copy(); q[:, s0:s0+LFR] = m
                d = float((fwd(net, q) - base).mean())
                V[(mdl, bn, c_, arm, p_, i)] = d
                prows.append((mdl, bn, c_, arm, p_, i, d))
        log(f"[run] {mdl} {bn} done  {time.time()-t0:.0f}s")
with open("bpnet_acceptor_positional.tsv", "w") as f:
    f.write("model\tbackground\tclass\tarm\tposition\tfragment\tdelta_logit\n")
    for r_ in prows: f.write("\t".join(str(x) for x in r_) + "\n")
n = len(PAIRS)
def vec(mdl, bn, c_, arm, p): return np.array([V[(mdl, bn, c_, arm, p, i)] for i in range(n)])
log(f"\nTEST 5  pooled {UPPOS[0]} to {UPPOS[-1]}, the {n} fragments as the independent units")
for mdl in ("CNN", "BPNet"):
    for bn in ("real_canonical", "real_competitor"):
        Aa = np.mean([vec(mdl,bn,"acceptor","motif",p)-vec(mdl,bn,"acceptor","mono",p) for p in UPPOS],0)
        Ii = np.mean([vec(mdl,bn,"intron","motif",p)-vec(mdl,bn,"intron","mono",p) for p in UPPOS],0)
        for nm, d in (("acceptor vs own shuffle", Aa), ("intron vs own shuffle", Ii),
                      ("acceptor MINUS intron", Aa-Ii)):
            se = d.std(ddof=1)/math.sqrt(n); z = d.mean()/se
            log(f"  {mdl:<6} {bn:<16} {nm:<24} {d.mean():>+8.4f} +/- {se:.4f} "
                f"[{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}] z = {z:>+5.1f}"
                f"{'' if abs(z)>=2 else '  ns'}")
log("[out] bpnet_acceptor_test.tsv bpnet_acceptor_positional.tsv")
