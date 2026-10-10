#!/usr/bin/env python3
# fit the interpretable readout on top of the frozen current model
import sys, time, argparse
import numpy as np
import torch, torch.nn as nn

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--epochs", type=int, default=40)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=3e-3)
P.add_argument("--lam", type=float, default=1e-3, help="L2 pull of pos_w toward 1")
P.add_argument("--patience", type=int, default=8)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

from cnn_model_def import Net
from cnn_readout_def import NetReadout, NBINS, BINW
ck = torch.load(A.model, map_location=dev, weights_only=False)
base = Net(ck["args"]["dropout"]).to(dev); base.load_state_dict(ck["state"]); base.eval()
net = NetReadout(base).to(dev); net.freeze_base()
ntr = sum(p.numel() for p in net.parameters() if p.requires_grad)
log(f"device {dev}; readout parameters {ntr} (base frozen: "
    f"{sum(p.numel() for p in base.parameters())} params)")

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(np.float32); gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBL = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
def oh(ii): return TBL[Xt[ii].to(dev)].permute(0, 2, 1).contiguous()

def within_gene(s, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r, c = by.setdefault(g[k], ([], []))
        (r if yy[k] == 1 else c).append(s[k])
    v = [((np.array(r)[:, None] > np.array(c)[None, :]).sum()
          + 0.5 * (np.array(r)[:, None] == np.array(c)[None, :]).sum()) / (len(r) * len(c))
         for r, c in by.values() if r and c]
    return float(np.mean(v)), len(v)

def scores(model, ii, bs=512):
    model.eval(); out = np.empty(len(ii), np.float32)
    with torch.no_grad():
        for k in range(0, len(ii), bs):
            out[k:k+bs] = model(oh(ii[k:k+bs])).float().cpu().numpy()
    return out

# identity check: the readout at initialisation must be the base model
chk = idx["test"][:512]
with torch.no_grad():
    d = float((net(oh(chk)) - base(oh(chk))).abs().max())
assert d < 1e-4, f"readout is not the identity at init (max |diff| {d})"
log(f"[check] readout == base at initialisation, max |diff| {d:.2e}")
base_test, ng = within_gene(scores(net, idx["test"]), idx["test"])
base_val, _ = within_gene(scores(net, idx["val"]), idx["val"])
log(f"[base] within-gene val {base_val:.4f}  test {base_test:.4f} ({ng} genes)")

npos = float(y[idx["train"]].sum()); nneg = len(idx["train"]) - npos
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, device=dev))
opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=A.lr)

best = (-1.0, 0, None)
tr = idx["train"]
for ep in range(1, A.epochs + 1):
    net.train(); net.freeze_base()
    perm = np.random.permutation(tr); tot = 0.0; t0 = time.time()
    for k in range(0, len(perm), A.batch):
        j = perm[k:k+A.batch]
        opt.zero_grad(set_to_none=True)
        loss = crit(net(oh(j)), yt[j].to(dev)) + A.lam * ((net.pos_w - 1.0) ** 2).sum()
        loss.backward(); opt.step(); tot += float(loss) * len(j)
    wg, _ = within_gene(scores(net, idx["val"]), idx["val"])
    log(f"  ep {ep:3d}  loss {tot/len(perm):.4f}  val within-gene {wg:.4f}  "
        f"beta {float(net.beta):+.4f}  pos_w in [{float(net.pos_w.min()):.3f},"
        f"{float(net.pos_w.max()):.3f}]  {time.time()-t0:.0f}s")
    if wg > best[0]:
        best = (wg, ep, {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items() if "base." not in k})
    elif ep - best[1] >= A.patience:
        log(f"  early stop at {ep}; best epoch {best[1]} (val {best[0]:.4f})"); break
net.load_state_dict(best[2], strict=False)
log(f"restored epoch {best[1]}, val within-gene {best[0]:.4f} (frozen base val {base_val:.4f})")

st = scores(net, idx["test"]); wg_t, ng = within_gene(st, idx["test"])
delta = wg_t - base_test
log(f"[readout] test within-gene {wg_t:.4f} vs base {base_test:.4f}  (delta {delta:+.4f}; "
    f"tolerance 0.02 -> {'KEEP the learned readout' if abs(delta) <= 0.02 else 'REVERT to post-hoc'})")

# emit pos_w
pw = net.pos_w.detach().cpu().numpy()
with open("cnn_posmap.tsv", "w") as fh:
    fh.write("filter\tbin\tbin_start\tbin_end\tlearned_pos_w\n")
    for f in range(pw.shape[0]):
        for b in range(pw.shape[1]):
            fh.write(f"filter{f:02d}\t{b}\t{b*BINW-2000}\t{(b+1)*BINW-2000}\t{pw[f,b]:.6f}\n")
log(f"[posmap] learned pos_w range [{pw.min():.3f},{pw.max():.3f}], "
    f"mean |pos_w - 1| {np.abs(pw-1).mean():.4f}, "
    f"filters with any |pos_w-1| > 0.1: {int((np.abs(pw-1).max(1) > 0.1).sum())}/{pw.shape[0]}")

# attention track, canonical vs competitor, on test
te = idx["test"]; yte = y[te]
NB = 100
acc = {0: np.zeros(2500), 1: np.zeros(2500)}; cnt = {0: 0, 1: 0}
net.eval()
with torch.no_grad():
    for k in range(0, len(te), 512):
        j = te[k:k+512]
        _, al = net(oh(j), want_attn=True)
        al = al.float().cpu().numpy(); yy = yte[k:k+512]
        for c in (0, 1):
            m = yy == c
            if m.any(): acc[c] += al[m].sum(0); cnt[c] += int(m.sum())
prof = {c: acc[c] / max(cnt[c], 1) for c in (0, 1)}
W = 25
with open("cnn_attention_profile.tsv", "w") as fh:
    fh.write("offset\tcanonical\tcompetitor\tdiff\tn_canonical\tn_competitor\n")
    for i in range(0, 2500, W):
        a1 = prof[1][i:i+W].mean(); a0 = prof[0][i:i+W].mean()
        fh.write(f"{i - 2000 + W//2}\t{a1:.6f}\t{a0:.6f}\t{a1-a0:.6f}\t{cnt[1]}\t{cnt[0]}\n")
log(f"[attention] beta {float(net.beta):+.4f}; attention (mean 1) canonical range "
    f"[{prof[1].min():.3f},{prof[1].max():.3f}], competitor [{prof[0].min():.3f},{prof[0].max():.3f}]")
d = prof[1] - prof[0]; pk = int(np.argmax(np.abs(d)))
log(f"[attention] largest canonical-competitor gap {d[pk]:+.4f} at {pk-2000:+d} nt")

torch.save(dict(readout={k: v for k, v in net.state_dict().items() if "base." not in k},
                base_model=A.model, test_winrate=wg_t, base_winrate=base_test,
                epoch=best[1], args=vars(A)), "cnn_readout.pt")
with open("cnn_readout_metrics.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    fh.write(f"base_test_winrate\t{base_test:.5f}\nreadout_test_winrate\t{wg_t:.5f}\n"
             f"delta\t{delta:.5f}\ntolerance\t0.02\n"
             f"path\t{'learned_readout' if abs(delta)<=0.02 else 'posthoc_occlusion'}\n"
             f"best_epoch\t{best[1]}\nbeta\t{float(net.beta):.6f}\n"
             f"n_readout_params\t{ntr}\nn_test_genes\t{ng}\n")
log("[out] cnn_posmap.tsv cnn_attention_profile.tsv cnn_readout.pt cnn_readout_metrics.tsv")
