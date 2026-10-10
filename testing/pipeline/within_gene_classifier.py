#!/usr/bin/env python3
# (self-contained: no extract_splicing_features import) can within-gene splicing context
import argparse, os, sys, numpy as np, pandas as pd

_venv = os.environ.get("VIRTUAL_ENV")
if _venv:
    _sp = os.path.join(_venv, "lib",
                       f"python{sys.version_info.major}.{sys.version_info.minor}",
                       "site-packages")
    if os.path.isdir(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5, load_matrix3
except Exception as e:
    sys.exit(f"[fatal] cannot import maxentpy: {type(e).__name__}: {e}\n"
             f"        VIRTUAL_ENV={_venv}\n"
             f"        sys.path[:3]={sys.path[:3]}")


class GenomeFasta:
    def __init__(self, path):
        self.path = path
        try:
            import pysam
            self.fa = pysam.FastaFile(path); self.engine = "pysam"
            self.refs = set(self.fa.references)
        except Exception:
            from pyfaidx import Fasta
            self.fa = Fasta(path, sequence_always_upper=True); self.engine = "pyfaidx"
            self.refs = set(self.fa.keys())

    def fetch(self, chrom, start, end, strand="+"):
        c = chrom if chrom in self.refs else chrom.lstrip("chr")
        if c not in self.refs: return None
        if self.engine == "pysam":
            seq = self.fa.fetch(c, start, end).upper()
        else:
            seq = str(self.fa[c][start:end]).upper()
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        return seq


def init_maxentscan():
    m5 = load_matrix5(); m3 = load_matrix3()
    def score5(seq):
        if len(seq) != 9 or "N" in seq: return -99.0
        try: return float(maxent.score5(seq, matrix=m5))
        except Exception: return -99.0
    def score3(seq):
        if len(seq) != 23 or "N" in seq: return -99.0
        try: return float(maxent.score3(seq, matrix=m3))
        except Exception: return -99.0
    return score5, score3


def load_u1_lookup(path):
    arr = np.load(path, allow_pickle=True)
    return arr.item() if arr.dtype == object else dict(arr)

def scan_u1_max(seq, lo, hi, u1):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<9: return np.nan
    best=np.nan
    for i in range(lo,hi-9+1):
        k=seq[i:i+9].upper()
        if "N" in k: continue
        v=u1.get(k)
        if v is not None:
            val=-float(v)
            if np.isnan(best) or val>best: best=val
    return best

def scan_u1_sum(seq, lo, hi, u1):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<9: return np.nan
    tot=0.0; seen=False
    for i in range(lo,hi-9+1):
        k=seq[i:i+9].upper()
        if "N" in k: continue
        v=u1.get(k)
        if v is not None:
            tot+=-float(v); seen=True
    return tot if seen else np.nan

def scan_me5_max(seq, lo, hi, score5):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<9: return np.nan
    best=np.nan
    for i in range(lo,hi-9+1):
        w=seq[i:i+9].upper()
        if "N" in w or w[3:5]!="GT": continue
        try: v=float(score5(w))
        except Exception: continue
        if np.isnan(best) or v>best: best=v
    return best

def scan_3ss_max(seq, lo, hi, score3):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<23: return np.nan
    region=seq[lo:hi].upper()
    ra=np.frombuffer(region.encode("ascii","replace"),dtype=np.uint8)
    ag=np.where((ra[:-1]==65)&(ra[1:]==71))[0]+2
    best=np.nan
    for i in ag:
        p=lo+int(i); s0=p-20
        if s0<0 or p+3>len(seq): continue
        try:
            v=score3(seq[s0:p+3].upper())
            v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
        except Exception: continue
        if np.isnan(best) or v>best: best=v
    return best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shortlist",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True)
    ap.add_argument("--window",type=int,default=1000,help="proximal upstream window (nt)")
    ap.add_argument("--pad",type=int,default=10)
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--out",default="within_gene_clf")
    a=ap.parse_args()

    score5, score3 = init_maxentscan()
    fa=GenomeFasta(a.reference_fasta); u1=load_u1_lookup(a.u1_lookup)
    win=a.window; pad=a.pad

    df=pd.read_parquet(a.shortlist)
    if "n_internal" in df.columns: df=df[df.n_internal>0]
    df=df[df.groupby("gene")["is_canonical"].transform("any")]
    keep=df.groupby("gene")["is_canonical"].agg(["any","all"])
    good=keep[(keep["any"])&(~keep["all"])].index
    df=df[df.gene.isin(good)].copy()

    def feats(chrom,center,strand):
        if strand=="+": w0=max(0,center-win-pad); w1=center+pad
        else:           w0=max(0,center-pad); w1=center+win+pad
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: return (np.nan,np.nan,np.nan,np.nan)
        if not seq: return (np.nan,np.nan,np.nan,np.nan)
        L=len(seq); site=L-pad
        acc =scan_3ss_max(seq, max(0,site-win), site-pad, score3)
        don =scan_me5_max(seq, max(0,site-win), site, score5) if score5 else np.nan
        uu  =scan_u1_max (seq, max(0,site-win), site, u1)
        usum=scan_u1_sum (seq, max(0,site-win), site, u1)
        return (acc,don,uu,usum)

    print(f"[data] {df.gene.nunique():,} contested genes, {len(df):,} sites",file=sys.stderr)
    recs=[]
    for _,r in df.iterrows():
        acc,don,uu,usum=feats(str(r["chrom"]),int(r["center"]),r["strand"])
        recs.append(dict(gene=r["gene"], y=int(bool(r["is_canonical"])),
                         acc=acc, don=don, u1=uu, u1sum=usum))
    F=pd.DataFrame(recs).dropna(subset=["acc","u1"])
    for col in ["acc","don","u1","u1sum"]:
        if col in F: F[col+"__wg"]=F[col]-F.groupby("gene")[col].transform("mean")
    F=F.dropna(subset=["acc__wg","u1__wg"])
    print(f"[data] {len(F):,} sites with features, {F.gene.nunique():,} genes; "
          f"canonical={F.y.sum():,}",file=sys.stderr)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    feature_sets={
        "acceptor only":["acc__wg"],
        "donor only (5'ss+U1)":["don__wg","u1__wg","u1sum__wg"],
        "U1 only (max+sum)":["u1__wg","u1sum__wg"],
        "combined (acc+donor+U1)":["acc__wg","don__wg","u1__wg","u1sum__wg"],
    }
    groups=F["gene"].values; y=F["y"].values
    gkf=GroupKFold(n_splits=a.folds)
    oof={name:np.full(len(F),np.nan) for name in feature_sets}
    for name,cols in feature_sets.items():
        cols=[c for c in cols if c in F.columns]
        X=F[cols].fillna(0.0).values
        for tr,te in gkf.split(X,y,groups):
            m=LogisticRegression(max_iter=1000,class_weight="balanced")
            m.fit(X[tr],y[tr])
            oof[name][te]=m.predict_proba(X[te])[:,1]

    _genes=F["gene"].values
    _uniq=np.unique(_genes)
    _gidx={g:np.where(_genes==g)[0] for g in _uniq}
    def auroc_ci(scores, B=1000, seed=0):
        rng=np.random.default_rng(seed)
        base=roc_auc_score(y,scores)
        boot=[]
        for _ in range(B):
            gsamp=rng.choice(_uniq,size=len(_uniq),replace=True)
            idx=np.concatenate([_gidx[g] for g in gsamp])
            yy=y[idx]
            if yy.min()==yy.max(): continue
            boot.append(roc_auc_score(yy,scores[idx]))
        boot=np.array(boot)
        lo,hi=np.percentile(boot,[2.5,97.5])
        return base,lo,hi,boot

    print(f"\n=== within-gene classifier: canonical vs competitor (GroupKFold) ===",file=sys.stderr)
    res={}; boots={}
    for name in feature_sets:
        b,lo,hi,bt=auroc_ci(oof[name]); res[name]=(b,lo,hi); boots[name]=bt
        print(f"  {name:<26} AUROC={b:.3f} [{lo:.3f}-{hi:.3f}]",file=sys.stderr)

    best_single=max(["acceptor only","donor only (5'ss+U1)","U1 only (max+sum)"],
                    key=lambda k:res[k][0])
    cb=boots["combined (acc+donor+U1)"]; sb=boots[best_single]
    L=min(len(cb),len(sb)); delta=cb[:L]-sb[:L]
    dlo,dhi=np.percentile(delta,[2.5,97.5]); dmed=np.median(delta)
    comp = dlo>0
    print(f"\n  best single = {best_single} (AUROC {res[best_single][0]:.3f})",file=sys.stderr)
    print(f"  combined - best single: {dmed:+.3f} [{dlo:+.3f}-{dhi:+.3f}]  "
          f"=> {'COMPLEMENTARY (CI excludes 0)' if comp else 'not clearly additive'}",file=sys.stderr)

    F=F.assign(score=oof["combined (acc+donor+U1)"])
    p1=F.groupby("gene").apply(lambda g: int(g.loc[g.score.idxmax(),"y"]==1))
    p1_rand=float(np.mean(1.0/F.groupby("gene").size().values))
    print(f"\n  per-gene P@1 (canonical ranked #1 by combined): {p1.mean():.3f}  "
          f"(random {p1_rand:.3f}, n={len(p1):,} genes)",file=sys.stderr)

    out=pd.DataFrame([dict(model=k,auroc=res[k][0],lo=res[k][1],hi=res[k][2]) for k in res])
    out.to_csv(f"{a.out}_auroc.tsv",sep="\t",index=False)
    pd.DataFrame({"model_delta":["combined - "+best_single],"median":[dmed],"lo":[dlo],"hi":[dhi]}
                 ).to_csv(f"{a.out}_delta.tsv",sep="\t",index=False)
    F[["gene","y","acc__wg","don__wg","u1__wg","u1sum__wg","score"]].to_csv(
        f"{a.out}_sites.tsv",sep="\t",index=False)
    with open(f"{a.out}_p1.tsv","w") as fh:
        fh.write(f"metric\tvalue\nP@1_combined\t{p1.mean():.4f}\n"
                 f"P@1_random\t{p1_rand:.4f}\nn_genes\t{len(p1)}\n")
    print(f"\n[out] {a.out}_auroc.tsv, _delta.tsv, _sites.tsv, _p1.tsv",file=sys.stderr)

if __name__=="__main__": main()