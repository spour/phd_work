#!/usr/bin/env python3
# the decisive control: does the cnn still separate real 3' ends from competitors when
import sys, time
import numpy as np
import torch
from cnn_model_def import Net
from cnn_interpret import dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("cnn_model.pt", map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
te = np.flatnonzero(split == "test")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def score(seqs, bs=512):
    out = np.empty(len(seqs), np.float32)
    with torch.no_grad():
        for k in range(0, len(seqs), bs):
            oh = TBLt[torch.from_numpy(seqs[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1)
            out[k:k+bs] = net(oh.contiguous()).float().cpu().numpy()
    return out

def mono_shuffle(t, rng):
    s = t.copy(); m = s < 4; v = s[m]; rng.shuffle(v); s[m] = v; return s

def metrics(sc):
    by = {}
    for k, g in enumerate(gene[te]):
        r, c = by.setdefault(g, ([], []))
        (r if y[te][k] == 1 else c).append(sc[k])
    wg, p1 = [], []
    for r, c in by.values():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        wg.append(((r[:, None] > c[None, :]).sum() + 0.5 * (r[:, None] == c[None, :]).sum())
                  / (len(r) * len(c)))
        p1.append(float(r.max() > c.max()) + 0.5 * float(r.max() == c.max()))
    pos = sc[y[te] == 1]; neg = sc[y[te] == 0]
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt) + cs + 1) / 2.0
    au = (rk[inv][:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    rs = np.random.default_rng(42); w = np.array(wg)
    b = w[rs.integers(0, len(w), size=(1000, len(w)))].mean(1)
    return (float(w.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
            float(np.mean(p1)), float(au), len(w))

Xte = X[te]
rows = []
t0 = time.time()
variants = {"real": Xte,
            "dinuc_shuffled": np.stack([dinuc_shuffle(s, rng) for s in Xte]),
            "mono_shuffled": np.stack([mono_shuffle(s, rng) for s in Xte])}
log(f"shuffles built in {time.time()-t0:.0f}s")
for nm, S in variants.items():
    gc = float(((S == 1) | (S == 2)).mean())
    sc = score(S); m = metrics(sc)
    rows.append((nm, gc, sc.mean(), *m))
    log(f"[{nm:15s}] GC {gc:.4f}  mean logit {sc.mean():+.3f}  within-gene {m[0]:.4f} "
        f"[{m[1]:.4f},{m[2]:.4f}]  P@1 {m[3]:.4f}  pooled AUROC {m[4]:.4f}  ({m[5]} genes)")
with open("cnn_shuffle_control.tsv", "w") as fh:
    fh.write("sequence\tGC\tmean_logit\twinrate\tci_lo\tci_hi\tp_at_1\tpooled_auroc\tn_genes\n")
    for r in rows:
        fh.write(f"{r[0]}\t{r[1]:.5f}\t{r[2]:.4f}\t{r[3]:.5f}\t{r[4]:.5f}\t{r[5]:.5f}\t"
                 f"{r[6]:.5f}\t{r[7]:.5f}\t{r[8]}\n")
f = {r[0]: r for r in rows}
surv = (f["dinuc_shuffled"][3] - 0.5) / (f["real"][3] - 0.5) * 100
log(f"\n[headline] within-gene {f['real'][3]:.4f} on real sequence -> "
    f"{f['dinuc_shuffled'][3]:.4f} with every motif destroyed and composition preserved "
    f"({surv:.0f}% of the lift survives), -> {f['mono_shuffled'][3]:.4f} on mononucleotide shuffles")
log("[out] cnn_shuffle_control.tsv")
