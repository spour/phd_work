#!/usr/bin/env python3
# why a feature worth auroc 0.62 alone adds 0.001 to the cnn
import sys, math
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
g_, c_, x_ = D["gene"][ok], D["chrom"][ok], D["center"][ok]
ACC = np.load("acceptor_scalar.npz")["acc"].astype(float)
Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
zk = {(a,b,int(v)): i for i,(a,b,v) in enumerate(zip(Z["gene"],Z["chrom"],Z["center"]))}
L = np.array([Z["logit"][zk[(a,b,int(v))]] if (a,b,int(v)) in zk else np.nan
              for a,b,v in zip(g_,c_,x_)])
te = np.flatnonzero((split=="test") & np.isfinite(L))
log(f"[data] {len(te):,} held-out test peaks in {len(np.unique(gene[te])):,} genes")

def per_gene(sc, ii):
    G = gene[ii]; Y = y[ii]; by = {}
    for k in range(len(ii)):
        r,c = by.setdefault(G[k], ([],[])); (r if Y[k]==1 else c).append(sc[k])
    out = {}
    for gg,(r,c) in by.items():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        out[gg] = float(((r[:,None]>c[None,:]).sum()+0.5*(r[:,None]==c[None,:]).sum())/(len(r)*len(c)))
    return out
WC = per_gene(L[te], te); WA = per_gene(ACC[te], te)
G = sorted(set(WC) & set(WA))
wc = np.array([WC[g] for g in G]); wa = np.array([WA[g] for g in G])
log(f"[genes] {len(G):,} genes with both a real 3' end and at least one competitor")
log(f"   CNN mean within-gene win rate      {wc.mean():.4f}")
log(f"   acceptor mean within-gene win rate {wa.mean():.4f}")
log(f"   correlation between the two, gene by gene: r = {np.corrcoef(wc,wa)[0,1]:+.4f}")

rc = wc == 1.0; ra = wa == 1.0
log(f"\n[2x2] genes where each scorer puts the real 3' end FIRST")
log(f"   {'':22}{'acceptor right':>16}{'acceptor wrong':>16}")
log(f"   {'CNN right':<22}{int((rc&ra).sum()):>16,}{int((rc&~ra).sum()):>16,}")
log(f"   {'CNN wrong':<22}{int((~rc&ra).sum()):>16,}{int((~rc&~ra).sum()):>16,}")
log(f"   CNN first in {100*rc.mean():.1f}% of genes, acceptor first in {100*ra.mean():.1f}%")
log(f"   of the {int(ra.sum()):,} genes the acceptor gets right, {100*(rc&ra).sum()/max(ra.sum(),1):.1f}% "
    f"are already right for the CNN")
log(f"   the acceptor rescues {int((~rc&ra).sum()):,} genes the CNN misses, "
    f"{100*(~rc&ra).sum()/max((~rc).sum(),1):.1f}% of the CNN's {int((~rc).sum()):,} failures; "
    f"chance for a scorer this good would be about {100*ra.mean():.1f}%")

log(f"\n[deciding] the acceptor WHERE THE CNN FAILS")
for nm, m in (("all genes", np.ones(len(G), bool)),
              ("genes where the CNN gets it right", rc),
              ("genes where the CNN FAILS", ~rc)):
    log(f"   acceptor within-gene win rate, {nm:<36} {wa[m].mean():.4f}  (n={int(m.sum()):,})")
log(f"\n[reverse] the CNN where the ACCEPTOR fails")
for nm, m in (("all genes", np.ones(len(G), bool)),
              ("genes where the acceptor gets it right", ra),
              ("genes where the acceptor FAILS", ~ra)):
    log(f"   CNN within-gene win rate, {nm:<41} {wc[m].mean():.4f}  (n={int(m.sum()):,})")
se = lambda v: v.std(ddof=1)/math.sqrt(len(v))
d = wa[~rc].mean() - 0.5
log(f"\n   the acceptor's win rate in the CNN's failures is {wa[~rc].mean():.4f} "
    f"+/- {se(wa[~rc]):.4f}, i.e. {d:+.4f} from chance (z = {d/se(wa[~rc]):+.1f})")
with open("acceptor_redundancy.tsv","w") as fh:
    fh.write("gene\tcnn_win\tacceptor_win\tcnn_first\tacceptor_first\n")
    for i,gg in enumerate(G):
        fh.write(f"{gg}\t{wc[i]:.4f}\t{wa[i]:.4f}\t{int(rc[i])}\t{int(ra[i])}\n")
log("[out] acceptor_redundancy.tsv")

def _sweep():
    va = np.flatnonzero((split == "val") & np.isfinite(L))
    def wgs(s, ii):
        G = gene[ii]; Y = y[ii]; by = {}
        for k in range(len(ii)):
            r, c = by.setdefault(G[k], ([], [])); (r if Y[k] == 1 else c).append(s[k])
        v = []
        for r, c in by.values():
            if not r or not c: continue
            r = np.array(r); c = np.array(c)
            v.append(((r[:,None]>c[None,:]).sum()+.5*(r[:,None]==c[None,:]).sum())/(len(r)*len(c)))
        return float(np.mean(v))
    def zin(v, ii):
        out = np.zeros(len(ii)); G = gene[ii]
        for gg in np.unique(G):
            m = G == gg; s = v[ii][m]
            out[m] = (s-s.mean())/(s.std()+1e-9) if len(s) > 1 else 0.0
        return out
    W = (0, .05, .1, .15, .2, .3, .4, .5, .75, 1.0)
    zl, za = zin(L, te), zin(ACC, te); zlv, zav = zin(L, va), zin(ACC, va)
    base = wgs(L[te], te)
    log(f"\n[combine] CNN alone {base:.4f}; sweeping the acceptor's weight")
    for w in W: log(f"   w = {w:<5} within-gene {wgs(zl+w*za, te):.4f}")
    wv = max(((wgs(zlv+w*zav, va), w) for w in W))[1]
    log(f"   weight chosen on VALIDATION = {wv}; held-out TEST {wgs(zl+wv*za, te):.4f} "
        f"({wgs(zl+wv*za, te)-base:+.4f})")
    log(f"   best achievable if tuned on test: {max(wgs(zl+w*za, te) for w in W):.4f}")
_sweep()
