#!/usr/bin/env python3
# matched dinucleotide-shuffle benchmark of the cnn and the hand-built splice model
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, "scripts")

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--k", type=int, default=3, help="shuffles per site")
P.add_argument("--seed", type=int, default=42)
P.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 8))
P.add_argument("--out", default="shuffle_benchmark.tsv")
A = P.parse_args()

UP, DN, NB = 2000, 500, 8
ME5P = "../train/splice_train/data/me5_lookup.npy"
U1P  = "../train/splice_train/data/u1_lookup.npy"
NT = np.array(list("ACGT") + ["N"])
SEED, NBOOT = A.seed, 1000

# data
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(float); gene = D["gene"][ok]; split = D["split"][ok]
te = np.flatnonzero(split == "test")
Xte = X[te]; yte = y[te]; gte = gene[te]
log(f"test {len(te)} sites, {len(set(gte))} genes, {int(yte.sum())} real, window {X.shape[1]} nt")

by = {}
for k, g in enumerate(gte):
    r, c = by.setdefault(g, ([], []))
    (r if yte[k] == 1 else c).append(k)
GRP = [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def gene_wr(sc):
    return np.array([((sc[r][:, None] > sc[c][None, :]).sum()
                      + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                     for r, c in GRP])
def boot_ci(per_gene):
    rs = np.random.default_rng(SEED)
    B = rs.integers(0, len(per_gene), size=(NBOOT, len(per_gene)))
    v = per_gene[B].mean(1)
    return float(per_gene.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

# the shuffles, generated once
from cnn_interp_lib import dinuc_shuffle
rng = np.random.default_rng(SEED)
t0 = time.time()
SH = np.stack([np.stack([dinuc_shuffle(s, rng) for s in Xte]) for _ in range(A.k)])
log(f"generated {A.k} dinucleotide shuffles per site in {time.time()-t0:.0f}s -> {SH.shape}")
def gc(a): return float(((a == 1) | (a == 2)).mean())
def cpg(a):
    return float(((a[:, :-1] == 1) & (a[:, 1:] == 2)).mean())
log(f"[check] GC  real {gc(Xte):.5f}  shuffled {np.mean([gc(s) for s in SH]):.5f}")
log(f"[check] CpG real {cpg(Xte):.5f}  shuffled {np.mean([cpg(s) for s in SH]):.5f}")

# arm a: the cnn
import torch
from cnn_model_def import Net
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("cnn_model.pt", map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def cnn_score(seqs, bs=512):
    out = np.empty(len(seqs), np.float32)
    with torch.no_grad():
        for k in range(0, len(seqs), bs):
            oh = TBLt[torch.from_numpy(seqs[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
            out[k:k+bs] = net(oh).float().cpu().numpy()
    return out
cnn_real = gene_wr(cnn_score(Xte))
cnn_shuf = np.mean([gene_wr(cnn_score(SH[j])) for j in range(A.k)], 0)
log(f"[CNN] real {cnn_real.mean():.4f}   shuffled {cnn_shuf.mean():.4f}   "
    f"drop {cnn_real.mean()-cnn_shuf.mean():+.4f}")

# arm b: the splice model, features re-derived on the same shuffles
FIT = np.load("splice_model_fit.npz", allow_pickle=True)
SEL = [str(x) for x in FIT["sel_names"]]; BETA = FIT["beta"]; MU = FIT["mu"]; SD = FIT["sd"]
log(f"[splice] fitted model: {SEL}  beta {np.round(BETA,4)}")

_G = {}
def _init():
    from extract_splicing_features import init_maxentscan, load_me5_lookup, load_u1_lookup
    from compute_gene_baselines import derive_axes
    import splice_extra_features as XF
    from splice_common import compute_splice_row
    _, s3, _, m3 = init_maxentscan()
    _G.update(score3=s3, m3=m3, me5=load_me5_lookup(ME5P), u1=load_u1_lookup(U1P),
              derive_axes=derive_axes, XF=XF, crow=compute_splice_row)

def _one(tok):
    seq = "".join(NT[np.where(tok < 4, tok, 4)])
    r = _G["crow"](seq, UP, DN, NB, _G["me5"], _G["score3"], _G["m3"], _G["u1"],
                   _G["derive_axes"], _G["XF"])
    return [r[c] for c in SEL]

def centre_by_gene(M, g):
    out = M.copy()
    order = np.argsort(g, kind="stable"); gs = g[order]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        idx = order[cut[j]:cut[j+1]]
        mu = np.nanmean(out[idx], 0)
        out[idx] = out[idx] - np.where(np.isfinite(mu), mu, 0.0)
    return out

def splice_score(feat):
    Cg = centre_by_gene(feat, gte)
    Z = (np.where(np.isfinite(Cg), Cg, MU) - MU) / SD
    return np.column_stack([np.ones(len(Z)), Z]) @ BETA

from multiprocessing import Pool
def featurise(tokens, tag):
    t = time.time()
    with Pool(A.jobs, initializer=_init) as p:
        out = np.array(p.map(_one, list(tokens), chunksize=32), float)
    log(f"  featurised {tag}: {out.shape} in {time.time()-t:.0f}s "
        f"({100*np.isfinite(out).all(1).mean():.1f}% complete rows)")
    return out

if __name__ == "__main__":
    real_feat = featurise(Xte, "real")
    spl_real = gene_wr(splice_score(real_feat))
    shuf_wr = []
    for j in range(A.k):
        shuf_wr.append(gene_wr(splice_score(featurise(SH[j], f"shuffle {j+1}/{A.k}"))))
    spl_shuf = np.mean(shuf_wr, 0)
    log(f"[splice] real {spl_real.mean():.4f}   shuffled {spl_shuf.mean():.4f}   "
        f"drop {spl_real.mean()-spl_shuf.mean():+.4f}")

    rows = []
    for model, real, shuf in (("cnn", cnn_real, cnn_shuf), ("splice_selected", spl_real, spl_shuf)):
        mr, lr, hr = boot_ci(real); ms, ls, hs = boot_ci(shuf)
        dm, dl, dh = boot_ci(real - shuf)
        lift = mr - 0.5
        rows.append(dict(model=model, real=mr, real_lo=lr, real_hi=hr,
                         shuffled=ms, shuffled_lo=ls, shuffled_hi=hs,
                         drop=dm, drop_lo=dl, drop_hi=dh,
                         pct_of_lift_lost=100 * dm / lift if lift > 0 else np.nan,
                         pct_of_lift_kept=100 * (ms - 0.5) / lift if lift > 0 else np.nan))
        log(f"[{model:16s}] real {mr:.4f} [{lr:.4f},{hr:.4f}]  shuffled {ms:.4f} [{ls:.4f},{hs:.4f}]  "
            f"drop {dm:+.4f} [{dl:+.4f},{dh:+.4f}] = {100*dm/lift:.1f}% of the lift; "
            f"{100*(ms-0.5)/lift:.1f}% of the lift survives shuffling")
    with open(A.out, "w") as fh:
        fh.write("model\treal\treal_lo\treal_hi\tshuffled\tshuffled_lo\tshuffled_hi\tdrop\t"
                 "drop_lo\tdrop_hi\tpct_of_lift_lost\tpct_of_lift_kept\tn_genes\tn_sites\tk_shuffles\n")
        for r in rows:
            fh.write("\t".join(f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k])
                     for k in ("model","real","real_lo","real_hi","shuffled","shuffled_lo",
                               "shuffled_hi","drop","drop_lo","drop_hi","pct_of_lift_lost",
                               "pct_of_lift_kept"))
                     + f"\t{len(GRP)}\t{len(te)}\t{A.k}\n")
        fh.write(f"check\tGC_real\t{gc(Xte):.6f}\tGC_shuffled\t{np.mean([gc(s) for s in SH]):.6f}\t"
                 f"CpG_real\t{cpg(Xte):.6f}\tCpG_shuffled\t{np.mean([cpg(s) for s in SH]):.6f}\t"
                 f"\t\t\t\t{len(GRP)}\t{len(te)}\t{A.k}\n")
    log(f"[out] {A.out}")
