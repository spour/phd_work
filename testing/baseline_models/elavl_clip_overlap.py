#!/usr/bin/env python3
# test whether suppressed internal cpa sites are enriched for elavl/hur clip binding vs
import argparse, sys, math
import numpy as np, pandas as pd

def load_clip(path):
    iv={}
    with open(path) as fh:
        for ln in fh:
            if ln.startswith(("#","track","browser")) or not ln.strip(): continue
            f=ln.split("\t")
            if len(f)<3: continue
            c=f[0]; s=int(f[1]); e=int(f[2])
            iv.setdefault(c,[]).append((s,e))
    for c in iv: iv[c]=sorted(iv[c])
    return iv

def overlaps(iv, chrom, s, e):
    arr=iv.get(chrom)
    if not arr: return False
    import bisect
    starts=[x[0] for x in arr]
    i=bisect.bisect_right(starts,e)-1
    j=max(0,i-3)
    for k in range(j,min(len(arr),i+2)):
        if arr[k][0]<e and arr[k][1]>s: return True
    return False

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def max_a_run(seq):
    if not seq: return 0
    s=seq.upper(); best=cur=0
    for ch in s:
        if ch=="A": cur+=1; best=max(best,cur)
        else: cur=0
    return best

def au_content(seq):
    if not seq: return np.nan
    s=seq.upper(); return (s.count("A")+s.count("T"))/len(s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels",required=True); ap.add_argument("--clip",required=True)
    ap.add_argument("--reference_fasta",required=True); ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--window",type=int,default=200,help="+-window around center for overlap & composition")
    ap.add_argument("--a_run_thresh",type=int,default=8,help="genomic A-run length flagged as priming risk")
    ap.add_argument("--n_perm",type=int,default=1000)
    ap.add_argument("--out",default="elavl_overlap")
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import GenomeFasta

    df=pd.read_parquet(a.labels)
    clip=load_clip(a.clip)
    fa=GenomeFasta(a.reference_fasta)
    print(f"[load] {len(df):,} sites, CLIP on {len(clip)} chroms",file=sys.stderr)

    recs=[]
    for _,r in df.iterrows():
        c=int(r["center"]); ch=r["chrom"]; st=r["strand"]
        s=c-a.window; e=c+a.window
        hit=overlaps(clip,ch,s,e)
        try: seq=fa.fetch(ch,max(0,s),e,strand=st)
        except Exception: seq=None
        recs.append(dict(label=int(r["label"]),clip_hit=int(hit),
                         au=au_content(seq),a_run=max_a_run(seq) if seq else 0,
                         compartment=r.get("compartment","NA")))
    F=pd.DataFrame(recs)
    F.to_parquet(f"{a.out}_sites.parquet",index=False)

    def rate(mask):
        n=mask.sum(); k=F[mask].clip_hit.sum()
        p,lo,hi=wilson(int(k),int(n)); return p,lo,hi,int(n)

    sup=F.label==0; use=F.label==1
    ps,los,his,ns=rate(sup); pu,lou,hiu,nu=rate(use)
    print(f"\n=== raw ELAVL/HuR CLIP overlap (window +-{a.window}) ===",file=sys.stderr)
    print(f"  suppressed: {ps:.3f} [{los:.3f},{his:.3f}] (n={ns})",file=sys.stderr)
    print(f"  used:       {pu:.3f} [{lou:.3f},{hiu:.3f}] (n={nu})",file=sys.stderr)
    print(f"  raw enrichment (supp-used) = {ps-pu:+.3f}",file=sys.stderr)

    keep=F.a_run<a.a_run_thresh
    print(f"\n=== after excluding A-run>={a.a_run_thresh} (internal-priming risk) ===",file=sys.stderr)
    print(f"  excluded {int((~keep).sum())} sites ({(~keep).mean()*100:.1f}%); "
          f"suppressed lose {int((~keep & sup).sum())}, used lose {int((~keep & use).sum())}",file=sys.stderr)
    ps2,_,_,ns2=rate(sup&keep); pu2,_,_,nu2=rate(use&keep)
    print(f"  suppressed: {ps2:.3f} (n={ns2})   used: {pu2:.3f} (n={nu2})   "
          f"enrichment = {ps2-pu2:+.3f}",file=sys.stderr)

    Fk=F[keep].copy()
    bins=np.linspace(0,1,11); Fk["aubin"]=np.digitize(Fk.au.fillna(0.5),bins)
    sup_k=Fk[Fk.label==0]; use_k=Fk[Fk.label==1]
    rng=np.random.default_rng(0); diffs=[]
    sup_rate=sup_k.clip_hit.mean()
    for _ in range(a.n_perm):
        samp=[]
        for b,grp in sup_k.groupby("aubin"):
            pool=use_k[use_k.aubin==b]
            if len(pool)==0: continue
            samp.append(pool.sample(len(grp),replace=True,random_state=rng.integers(1e9)))
        if not samp: continue
        matched_used_rate=pd.concat(samp).clip_hit.mean()
        diffs.append(sup_rate-matched_used_rate)
    diffs=np.array(diffs)
    print(f"\n=== composition-matched test (the decisive one) ===",file=sys.stderr)
    if len(diffs)==0:
        print(f"  [WARN] no AU-overlapping used sites to match; suppressed and used AU",file=sys.stderr)
        print(f"  distributions may not overlap. Inspect {a.out}_sites.parquet AU columns.",file=sys.stderr)
    else:
        p_emp=(np.sum(diffs<=0)+1)/(len(diffs)+1)
        print(f"  suppressed CLIP rate = {sup_rate:.3f}",file=sys.stderr)
        print(f"  AU-matched used CLIP rate = {sup_rate-diffs.mean():.3f} (mean over {len(diffs)} resamples)",file=sys.stderr)
        print(f"  mean enrichment over AU-matched used = {diffs.mean():+.3f} "
              f"[{np.percentile(diffs,2.5):+.3f},{np.percentile(diffs,97.5):+.3f}]",file=sys.stderr)
        print(f"  empirical p (enrichment<=0) = {p_emp:.4f}",file=sys.stderr)
    print(f"\n  enrichment survives AU-matching => REAL ELAVL signal, not composition artifact.",file=sys.stderr)
    print(f"  enrichment vanishes => it was A/U composition / eCLIP A-bias, not regulation.",file=sys.stderr)
    print(f"[out] {a.out}_sites.parquet",file=sys.stderr)

if __name__=="__main__": main()
