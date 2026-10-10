#!/usr/bin/env python3
# figure3_v3_2 summary panel: descriptive stratification
import numpy as np, csv, math
from collections import defaultdict
def log(*a): import sys; print(*a,file=sys.stderr,flush=True)

R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t"))
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
acc=np.array([float(r["acc"]) if r["acc"] not in("","nan") else np.nan for r in R])
dn =np.array([float(r["first5_dn_present"]) for r in R])
d525=np.load("u1_window_cache.npz")["dens_5.25"]
assert np.array_equal(isc,np.load("u1_window_cache.npz")["isc"].astype(int))
BASE=float((isc==1).mean()); NC=int((isc==1).sum()); NK=int((isc==0).sum())

med=float(np.median(d525))
A=(acc>=8).astype(int)
Dp=(dn==1).astype(int)
Uh=(d525>med).astype(int)

def wilson(k,n,z=1.959963984540054):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

# output 1: 8-cell grid
def cell_label(a,dp,uh):
    return ("strong" if a else "weak")+" acc, "+("donor" if dp else "no donor")+", "+("hi" if uh else "lo")+" U1"
rows=[]
for a in (1,0):
    for dp in (0,1):
        for uh in (0,1):
            m=(A==a)&(Dp==dp)&(Uh==uh); n=int(m.sum()); k=int((m&(isc==1)).sum())
            p=k/n if n else float("nan"); lo,hi=wilson(k,n)
            rows.append(dict(acc=a,donor=dp,u1=uh,label=cell_label(a,dp,uh),n=n,n_canon=k,
                             p_real=p,enrich=p/BASE if n else float("nan"),ci_lo=lo,ci_hi=hi,
                             favourable=int(a==1 and dp==0 and uh==0)))
rows.sort(key=lambda r:-r["p_real"])
with open("v32_sum_grid8.tsv","w") as fh:
    fh.write("label\tacc\tdonor\tu1\tn\tn_canon\tp_real\tenrich\tci_lo\tci_hi\tfavourable\n")
    for r in rows:
        fh.write("{label}\t{acc}\t{donor}\t{u1}\t{n}\t{n_canon}\t{p_real:.5f}\t{enrich:.3f}\t{ci_lo:.5f}\t{ci_hi:.5f}\t{favourable}\n".format(**r))
log("OUTPUT 1  8-cell grid (sorted by P(real)):")
log(f"  base rate {BASE:.5f} ({NC} canonical / {NK} competitor)")
for r in rows:
    log(f"  {r['label']:34s} n={r['n']:6d} canon={r['n_canon']:5d} P(real)={r['p_real']:.4f} "
        f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] enrich={r['enrich']:.2f}x")

r4=[]
for a in (1,0):
    for dp in (0,1):
        m=(A==a)&(Dp==dp); n=int(m.sum()); k=int((m&(isc==1)).sum()); p=k/n if n else float("nan"); lo,hi=wilson(k,n)
        r4.append(dict(label=("strong" if a else "weak")+" acc, "+("no donor" if dp==0 else "donor"),
                       acc=a,donor=dp,n=n,n_canon=k,p_real=p,enrich=p/BASE,ci_lo=lo,ci_hi=hi,
                       favourable=int(a==1 and dp==0)))
r4.sort(key=lambda r:-r["p_real"])
with open("v32_sum_grid4.tsv","w") as fh:
    fh.write("label\tacc\tdonor\tn\tn_canon\tp_real\tenrich\tci_lo\tci_hi\tfavourable\n")
    for r in r4:
        fh.write("{label}\t{acc}\t{donor}\t{n}\t{n_canon}\t{p_real:.5f}\t{enrich:.3f}\t{ci_lo:.5f}\t{ci_hi:.5f}\t{favourable}\n".format(**r))
log("\nOUTPUT 1b  4-cell acceptor x donor:")
for r in r4:
    log(f"  {r['label']:22s} n={r['n']:6d} P(real)={r['p_real']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] enrich={r['enrich']:.2f}x")

# output 2: waterfall
steps=[("base rate (all peaks)",np.ones(len(R),bool)),
       ("+ strong acceptor",A==1),
       ("+ no intervening donor",(A==1)&(Dp==0)),
       ("+ low U1",(A==1)&(Dp==0)&(Uh==0))]
with open("v32_sum_waterfall.tsv","w") as fh:
    fh.write("step\tn\tn_canon\tp_real\tenrich\tci_lo\tci_hi\n")
    log("\nOUTPUT 2  waterfall (reading order):")
    for nm,m in steps:
        n=int(m.sum()); k=int((m&(isc==1)).sum()); p=k/n; lo,hi=wilson(k,n)
        fh.write(f"{nm}\t{n}\t{k}\t{p:.5f}\t{p/BASE:.3f}\t{lo:.5f}\t{hi:.5f}\n")
        log(f"  {nm:26s} n={n:6d} canon={k:5d} P(real)={p:.4f} [{lo:.4f},{hi:.4f}] enrich={p/BASE:.2f}x")

# output 3: within-gene
def within_gene_cell(mask,label,nboot=2000):
    by=defaultdict(list)
    for i in np.where(mask)[0]: by[gene[i]].append(i)
    genes=[g for g,idx in by.items() if len(idx)==1]
    hits=np.array([isc[by[g][0]]==1 for g in genes],dtype=float)
    frac=float(hits.mean()) if len(hits) else float("nan")
    rs=np.random.default_rng(42)
    bs=hits[rs.integers(0,len(hits),size=(nboot,len(hits)))].mean(1) if len(hits) else np.array([np.nan])
    lo,hi=np.percentile(bs,[2.5,97.5])
    return dict(label=label,n_genes=len(genes),frac_real=frac,ci_lo=lo,ci_hi=hi)
cells=[("acceptor only (strong)",A==1),
       ("acceptor + no donor",(A==1)&(Dp==0)),
       ("all favourable (strong acc, no donor, low U1)",(A==1)&(Dp==0)&(Uh==0))]
with open("v32_sum_withingene.tsv","w") as fh:
    fh.write("cell\tn_genes\tfrac_real\tci_lo\tci_hi\n")
    log("\nOUTPUT 3  within-gene (genes with exactly ONE peak in the cell; fraction where it is the real end):")
    for nm,m in cells:
        r=within_gene_cell(m,nm)
        fh.write(f"{r['label']}\t{r['n_genes']}\t{r['frac_real']:.5f}\t{r['ci_lo']:.5f}\t{r['ci_hi']:.5f}\n")
        log(f"  {nm:48s} n_genes={r['n_genes']:5d}  P(that peak is real)={r['frac_real']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")

# donor-u1 correlation
phi=float(np.corrcoef(Dp.astype(float),Uh.astype(float))[0,1])
pear=float(np.corrcoef(dn,d525)[0,1])
with open("v32_sum_meta.tsv","w") as fh:
    fh.write("key\tvalue\n")
    for k,v in [("base_rate",BASE),("n_canonical",NC),("n_competitor",NK),("u1_median",med),
                ("phi_donor_u1",phi),("pearson_first5dn_dens525",pear),("acc_strong_frac",float((A==1).mean()))]:
        fh.write(f"{k}\t{v}\n")
log(f"\nDONOR-U1 correlation (same 9-mer family): phi(donor,u1_high)={phi:.3f}, pearson(first5_dn_present,dens_5.25)={pear:.3f}")
print("[out] v32_sum_grid8.tsv v32_sum_grid4.tsv v32_sum_waterfall.tsv v32_sum_withingene.tsv v32_sum_meta.tsv")
