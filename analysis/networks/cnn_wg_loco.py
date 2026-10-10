#!/usr/bin/env python3
# out-of-fold cnn scores over the whole-genome candidate set
import sys, os, csv, time, argparse
import numpy as np
import torch, torch.nn as nn

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--inner_frac", type=float, default=0.25)
P.add_argument("--folds", default="")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, DN, PAD = 2000, 500, 40
log(f"[env] device {dev}  torch {torch.__version__}"
    + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))

# contested candidate set
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
wg_gene = np.array([r["gene"] for r in M])
wg_chrom = np.array([r["chrom"] for r in M])
wg_cen = np.array([int(r["center"]) for r in M])
wg_strand = np.array([r["strand"] for r in M])
wg_y = np.array([int(float(r["is_canonical"])) for r in M])
NPK = len(M)
CONT = np.array(sorted(set(wg_gene)))
log(f"[wg] {NPK} contested-gene peaks in {len(CONT)} genes")

LAB = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
tb = lambda x: x in ("True", "true", "1")
gcomp = {}
for r in LAB:
    gcomp[r["gene"]] = gcomp.get(r["gene"], 0) + (not tb(r["is_canonical"]))
NALL = len(gcomp); NUNC = sum(1 for v in gcomp.values() if v == 0)
log(f"[wg] {NALL} genes genome-wide, {NUNC} uncontested (one candidate, rank 1 for any model), "
    f"{NALL-NUNC} contested; baseline P@1 = {NUNC}/{NALL} = {NUNC/NALL:.5f}")
assert NALL == 15560 and NUNC == 8977, f"expected 15,560 / 8,977, got {NALL} / {NUNC}"

CODE = np.full(256, 4, np.int64)
for i, ch in enumerate("ACGT"): CODE[ord(ch)] = i
CACHE = "cnn_wg_tokens.npy"
if os.path.exists(CACHE):
    TOK = np.load(CACHE)
    assert TOK.shape == (NPK, UP + DN), f"stale cache {TOK.shape}"
    log(f"[wg] token cache reused: {CACHE}")
else:
    sys.path.insert(0, "scripts")
    import u1_5ss_landscape_data as L
    G = L.G_["G"]; t0 = time.time()
    TOK = np.empty((NPK, UP + DN), np.uint8)
    for i in range(NPK):
        c = int(wg_cen[i]); st = wg_strand[i]
        a, b = (c - UP - PAD, c + DN + PAD) if st == "+" else (c - DN - PAD + 1, c + UP + PAD + 1)
        s = G.fetch(wg_chrom[i], a, b, st)
        TOK[i] = CODE[np.frombuffer(s.encode("ascii"), np.uint8)][PAD:PAD + UP + DN]
        if (i + 1) % 20000 == 0: log(f"  fetch {i+1}/{NPK}  {time.time()-t0:.0f}s")
    np.save(CACHE, TOK)
    log(f"[wg] fetched {NPK} windows in {time.time()-t0:.0f}s -> {CACHE}")
TOK = np.where(TOK == 4, 255, TOK).astype(np.uint8)

# analysis set, for training
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(np.float32)
gene = D["gene"][ok]; chrom = D["chrom"][ok].astype(str)
CHS = sorted(set(chrom))
assert set(CHS) == set(np.unique(wg_chrom)), "chromosome sets differ"

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def onehot(codes):
    return TBLt[torch.from_numpy(codes.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
from cnn_model_def import Net

def groups(g_, y_):
    by = {}
    for k, gg in enumerate(g_):
        r, c = by.setdefault(gg, ([], []))
        (r if y_[k] == 1 else c).append(k)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def winrate(s, grp):
    return float(np.mean([((s[r][:, None] > s[c][None, :]).sum()
                           + 0.5 * (s[r][:, None] == s[c][None, :]).sum()) / (len(r) * len(c))
                          for r, c in grp]))

SC = np.full(NPK, np.nan)
WANT = [c for c in CHS if not A.folds or c in A.folds.split(",")]
t00 = time.time()
for fi, ch in enumerate(WANT, 1):
    tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(A.seed)
    sh = trg.copy(); rs.shuffle(sh)
    ivg = set(sh[:int(round(A.inner_frac * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    GIV = groups(gene[iv], y[iv])
    torch.manual_seed(A.seed); np.random.seed(A.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    net = Net(A.dropout).to(dev)
    npos = float(y[it].sum()); nneg = len(it) - npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)
    def score(codes, bs=1024):
        net.eval(); out = np.empty(len(codes), np.float32)
        with torch.no_grad():
            for k in range(0, len(codes), bs):
                out[k:k+bs] = net(onehot(codes[k:k+bs])).float().cpu().numpy()
        return out
    best = (-1.0, -1, None); t0 = time.time()
    for ep in range(1, A.epochs + 1):
        net.train(); perm = np.random.permutation(it)
        for k in range(0, len(perm), A.batch):
            j = perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            loss = crit(net(onehot(X[j])), torch.from_numpy(y[j]).to(dev))
            loss.backward(); opt.step()
        w = winrate(score(X[iv]), GIV)
        if w > best[0]:
            best = (w, ep, {k_: v.detach().cpu().clone() for k_, v in net.state_dict().items()})
        elif ep - best[1] >= A.patience:
            break
    net.load_state_dict(best[2])
    hold = np.flatnonzero(wg_chrom == ch)
    SC[hold] = score(TOK[hold])
    log(f"  [{fi:2d}/{len(WANT)}] {ch:>5}  train {len(it):5d}  inner-val {best[0]:.4f} ep {best[1]:2d}"
        f"  scored {len(hold):5d} contested peaks  {time.time()-t0:.0f}s"
        f"  [{(time.time()-t00)/60:.1f} min]")

if not A.folds:
    assert np.isfinite(SC).all(), "LOCO left contested peaks unscored"
np.savez_compressed("cnn_wg_scores.npz", logit=SC, gene=wg_gene, chrom=wg_chrom, y=wg_y,
                    center=wg_cen, strand=wg_strand)
log(f"[out] cnn_wg_scores.npz  ({(time.time()-t00)/60:.1f} min)")

# p@1 ladder
GI = {g: np.flatnonzero(wg_gene == g) for g in CONT}
PAIR = {g: (ix[wg_y[ix] == 1], ix[wg_y[ix] == 0]) for g, ix in GI.items()}
PAIR = {g: v for g, v in PAIR.items() if len(v[0]) and len(v[1])}
GL = np.array(sorted(PAIR))
hit = np.array([float((SC[PAIR[g][1]] >= SC[PAIR[g][0]][0]).sum() == 0) for g in GL])
rs = np.random.default_rng(A.seed).integers(0, len(GL), size=(2000, len(GL)))
lo, hi = np.percentile(hit[rs].mean(1), [2.5, 97.5])
wg = (NUNC + hit.sum()) / NALL
wlo, whi = (NUNC + hit[rs].mean(1) * len(GL)) / NALL, None
wl, wh = np.percentile((NUNC + hit[rs].sum(1)) / NALL, [2.5, 97.5])
log(f"\n[CNN] contested {len(GL)} genes: P@1 {hit.mean():.5f} [{lo:.5f},{hi:.5f}]")
log(f"[CNN] genome-wide {NALL} genes: P@1 {wg:.5f} [{wl:.5f},{wh:.5f}] "
    f"({NUNC} uncontested counted correct for every model)")

SM = {r["metric"]: r for r in csv.DictReader(open("splice_model_26_metrics.tsv"), delimiter="\t")}
sp_c = float(SM["p_at_1_contested"]["value"]); sp_w = float(SM["p_at_1_whole_genome"]["value"])
base = NUNC / NALL
log(f"[ladder] genome-wide P@1  baseline {base:.5f}  splice {sp_w:.5f}  CNN {wg:.5f}")
log(f"[ladder] contested only   baseline 0.00000  splice {sp_c:.5f}  CNN {hit.mean():.5f}")
with open("cnn_wholegenome_p1.tsv", "w") as fh:
    fh.write("model\tscope\tn_genes\tp_at_1\tci_lo\tci_hi\tbaseline\tnote\n")
    fh.write(f"Baseline\tgenome-wide\t{NALL}\t{base:.5f}\t\t\t{base:.5f}\t"
             f"{NUNC} uncontested genes have one candidate and are rank 1 for any model\n")
    fh.write(f"Splice model (26 features)\tgenome-wide\t{NALL}\t{sp_w:.5f}\t"
             f"{SM['p_at_1_whole_genome']['ci_lo']}\t{SM['p_at_1_whole_genome']['ci_hi']}\t"
             f"{base:.5f}\tLOCO over contested genes, from splice_model_26_metrics.tsv\n")
    fh.write(f"Sequence CNN\tgenome-wide\t{NALL}\t{wg:.5f}\t{wl:.5f}\t{wh:.5f}\t{base:.5f}\t"
             f"LOCO, all {NPK} contested candidates scored out of fold\n")
    fh.write(f"Baseline\tcontested\t{len(GL)}\t0.00000\t\t\t0.00000\t"
             f"a contested gene is one the baseline does not already rank first\n")
    fh.write(f"Splice model (26 features)\tcontested\t{len(GL)}\t{sp_c:.5f}\t"
             f"{SM['p_at_1_contested']['ci_lo']}\t{SM['p_at_1_contested']['ci_hi']}\t0.00000\t\n")
    fh.write(f"Sequence CNN\tcontested\t{len(GL)}\t{hit.mean():.5f}\t{lo:.5f}\t{hi:.5f}\t"
             f"0.00000\t\n")
log("[out] cnn_wholegenome_p1.tsv")
