#!/usr/bin/env python3
# a battery of competitor-vs-canonical splicing tests
import argparse, sys, re
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# sr-protein ese and hnrnp ess motif sets (consensus pwms collapsed to k-mers)
ESE = ["GAAGAA","GGAGGA","AGGAGG","GAAGAT","CACCTG","TGGACG","AGGACA","GATGAA"]
ESS = ["TTTTTT","CTAGCT","TAGGGT","TGTATA","ATTTTA","TCTTTC","TAGTAG","GTAGTA"]

def auroc(pos, neg):
    if len(pos) < 20 or len(neg) < 20: return np.nan
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); r = ((cs - cnt) + cs + 1)[inv] / 2.0
    return float((r[y==1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def wilson(k, n, z=1.96):
    if n == 0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return p, max(0,c-h), min(1,c+h)

def paired_within_gene(df, col):
    hits=0; n=0; diffs=[]
    for g, gg in df.groupby("gene", sort=False):
        can = gg.loc[gg.is_canonical, col]; comp = gg.loc[~gg.is_canonical, col]
        can = can.dropna(); comp = comp.dropna()
        if len(can)==0 or len(comp)==0: continue
        cv=float(can.iloc[0]); kv=float(comp.mean())
        if not (np.isfinite(cv) and np.isfinite(kv)): continue
        hits += int(cv > kv); n += 1; diffs.append(cv - kv)
    if n==0: return dict(frac=np.nan, lo=np.nan, hi=np.nan, med_diff=np.nan, n=0)
    p,lo,hi = wilson(hits, n)
    return dict(frac=p, lo=lo, hi=hi, med_diff=float(np.median(diffs)), n=n)

def gc(seq):
    if not seq: return np.nan
    s=seq.upper(); g=s.count("G")+s.count("C"); return g/max(len(s),1)

def motif_density(seq, motifs):
    if not seq: return np.nan
    s=seq.upper(); L=max(len(s),1); c=0
    for m in motifs: c += s.count(m)
    return c / (L/100.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts_dir", default=None)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--feat", default=None, help="shortlist_feat_all.parquet to also mine precomputed features")
    ap.add_argument("--up_win", type=int, default=300, help="upstream window for local scans (nt)")
    ap.add_argument("--pas_win", type=int, default=50, help="window around site for PAS hexamer scan")
    ap.add_argument("--max_per_class", type=int, default=8000)
    ap.add_argument("--imbalanced", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fig3_alt")
    a = ap.parse_args()
    if a.scripts_dir: sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import GenomeFasta, init_maxentscan, load_u1_lookup, load_me5_lookup
    import threading
    _tls = threading.local()
    def _scorers():
        if not hasattr(_tls,"s3"):
            o=init_maxentscan(); _tls.s5,_tls.s3=o[0],o[1]
        return _tls.s5,_tls.s3
    _m=init_maxentscan(); score5,score3=_m[0],_m[1]
    fa = GenomeFasta(a.reference_fasta)
    u1 = load_u1_lookup(a.u1_lookup)
    U1LEN = len(next(iter(u1)))
    _MAP = np.full(256,-1,np.int64)
    for b,c in zip(b"ACGT",range(4)): _MAP[b]=c
    u1dense = np.full(4**U1LEN, np.nan, np.float32); _pow=(4**np.arange(U1LEN-1,-1,-1)).astype(np.int64)
    for kmer,val in u1.items():
        idx=0
        for ch in kmer.encode(): idx=idx*4+int(_MAP[ch])
        u1dense[idx]=val
    def u1_max(seq):
        arr=_MAP[np.frombuffer(seq.encode("ascii","replace"),np.uint8)]; n=len(arr)
        if n<U1LEN: return np.nan
        codes=np.zeros(n-U1LEN+1,np.int64); bad=np.zeros(n-U1LEN+1,bool)
        for k in range(U1LEN):
            cc=arr[k:k+(n-U1LEN+1)]; codes+=cc*_pow[k]; bad|=(cc<0)
        v=np.where(bad,np.nan,u1dense[np.clip(codes,0,len(u1dense)-1)])
        return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan

    df = pd.read_parquet(a.shortlist); df=df[df.n_internal>0]
    df["cls"]=np.where(df.is_canonical,"Canonical","Competitor")
    if a.imbalanced:
        parts=[gg.sample(min(len(gg),a.max_per_class),random_state=a.seed) for _,gg in df.groupby("cls")]
    else:
        nt=min(a.max_per_class,int(df.cls.value_counts().min()))
        parts=[gg.sample(nt,random_state=a.seed) if len(gg)>nt else gg for _,gg in df.groupby("cls")]
    df=pd.concat(parts,ignore_index=True)
    print(f"[data] {len(df):,} peaks, base {df.is_canonical.mean():.3f}", file=sys.stderr)

    def _scan(rec):
        score5, score3 = _scorers()
        c=int(rec["center"]); strand=rec["strand"]; chrom=str(rec["chrom"])
        pad=10; UW=a.up_win
        if strand=="+": w0,w1=max(0,c-UW-40),c+a.pas_win
        else:           w0,w1=max(0,c-a.pas_win),c+UW+40
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: return None
        if not seq or len(seq)<40: return None
        seq=seq.upper(); L=len(seq)
        site = (c-w0) if strand=="+" else (w1-c)
        site = int(np.clip(site,0,L-1))
        up = seq[max(0,site-UW):site]
        pasreg = seq[max(0,site-a.pas_win):min(L,site+a.pas_win)]
        r={}
        arr=np.frombuffer(up.encode("ascii","replace"),np.uint8)
        ags=np.where((arr[:-1]==65)&(arr[1:]==71))[0]+2
        best3=np.nan; bp=np.nan; ppt=np.nan; best3_pos=None
        for i in ags:
            p=int(i)
            if p-20<0 or p+3>len(up): continue
            try:
                v=score3(up[p-20:p+3]); v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
            except Exception: continue
            if np.isfinite(v) and (not np.isfinite(best3) or v>best3):
                best3=v; best3_pos=p
                bpreg=up[max(0,p-40):max(0,p-18)]
                bp = (bpreg.count("A")+bpreg.count("C"))/max(len(bpreg),1)
                pptreg=up[max(0,p-20):max(0,p-1)]
                ppt = (pptreg.count("C")+pptreg.count("T"))/max(len(pptreg),1)
        r["acceptor_3ss"]=best3; r["branchpoint"]=bp; r["ppt"]=ppt
        gts=np.where((arr[:-1]==71)&(arr[1:]==84))[0]
        best5=np.nan
        for i in gts:
            p=int(i)
            if p-3<0 or p+6>len(up): continue
            try:
                v=score5(up[p-3:p+6]); v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
            except Exception: continue
            if np.isfinite(v) and (not np.isfinite(best5) or v>best5): best5=v
        r["donor_5ss"]=best5
        iv_sum=np.nan; iv_n=0; iv_nstrong=0; iv_density=np.nan; iv_meanpos=np.nan
        if best3_pos is not None and best3_pos < len(up):
            seg = up[best3_pos:]
            seg_len = max(len(seg), 1)
            sarr=np.frombuffer(seg.encode("ascii","replace"),np.uint8)
            sgts=np.where((sarr[:-1]==71)&(sarr[1:]==84))[0]
            svals=[]
            for j in sgts:
                q=int(j)
                if q-3<0 or q+6>len(seg): continue
                try:
                    vv=score5(seg[q-3:q+6]); vv=float(np.nanmax(np.asarray(vv,float))) if not np.isscalar(vv) else float(vv)
                except Exception: continue
                if np.isfinite(vv) and vv>-99: svals.append(vv)
            if svals:
                svals=np.array(svals)
                iv_n=len(svals)
                iv_sum=float(np.sum(np.clip(svals,0,None)))
                iv_nstrong=int((svals>=6.5).sum())
                iv_density=float(iv_nstrong)/(seg_len/1000.0)
        r["iv_donor_sum"]=iv_sum
        r["iv_donor_n"]=float(iv_n)
        r["iv_donor_nstrong"]=float(iv_nstrong)
        r["iv_donor_density_perkb"]=iv_density
        r["u1_max"]=u1_max(up)
        r["gc_up"]=gc(up)
        r["ese_density"]=motif_density(up, ESE)
        r["ess_density"]=motif_density(up, ESS)
        pas=0
        for hexk in ("AATAAA","ATTAAA"):
            pas=max(pas, 2 if hexk in pasreg else 0)
        if pas==0:
            for hexk in ("AGTAAA","TATAAA","CATAAA","GATAAA","AATATA","AATACA","AATAGA","ACTAAA","AAGAAA","AATGAA"):
                if hexk in pasreg: pas=1; break
        r["pas_strength"]=float(pas)
        return (rec["gene"], bool(rec["is_canonical"]), r)

    recs=df.to_dict("records"); results=[]
    if a.n_jobs>1:
        print(f"[warn] --n_jobs {a.n_jobs} ignored: pysam is not thread-safe (corrupts "
              f"sequence fetches). Running serially; this is a one-pass battery so it's fast "
              f"enough. For parallelism use the process-based figure3_data.py pattern.",
              file=sys.stderr)
    if False:
        pass
    else:
        for rec in tqdm(recs,total=len(recs),desc="scan"):
            x=_scan(rec)
            if x: results.append(x)

    base=pd.DataFrame([{"gene":g,"is_canonical":ic,**r} for g,ic,r in results])
    SCAN_COLS=[c for c in base.columns if c not in ("gene","is_canonical")]
    base["asym_acc_minus_donor"]=base["acceptor_3ss"]-base["donor_5ss"]
    base["acc_minus_u1"]=base["acceptor_3ss"]-base["u1_max"]
    base["ese_minus_ess"]=base["ese_density"]-base["ess_density"]
    DERIVED=["asym_acc_minus_donor","acc_minus_u1","ese_minus_ess"]

    FEAT_COLS=[]
    if a.feat:
        fe=pd.read_parquet(a.feat)
        spl=[c for c in fe.columns if c.startswith("spl_")]
        key=["gene","chrom","center"] if all(k in fe.columns for k in ("gene","chrom","center")) else ["gene"]
        fe2=fe[["gene","is_canonical"]+spl].copy()
        FEAT_SRC=fe2; FEAT_COLS=spl
    else:
        FEAT_SRC=None

    # evaluate every signal
    rows=[]
    def eval_col(frame, col, source):
        sub=frame[["gene","is_canonical",col]].dropna(subset=[col])
        pc=sub.loc[sub.is_canonical,col].values; kc=sub.loc[~sub.is_canonical,col].values
        au=auroc(pc,kc)
        pr=paired_within_gene(sub.rename(columns={col:col}), col)
        rows.append(dict(signal=col, source=source, auroc=au,
                         auroc_disc=(abs(au-0.5) if np.isfinite(au) else np.nan),
                         paired_frac=pr["frac"], paired_lo=pr["lo"], paired_hi=pr["hi"],
                         paired_med_diff=pr["med_diff"], paired_n=pr["n"],
                         canon_mean=float(np.nanmean(pc)) if len(pc) else np.nan,
                         comp_mean=float(np.nanmean(kc)) if len(kc) else np.nan))
    for col in SCAN_COLS+DERIVED: eval_col(base, col, "scan")
    if FEAT_SRC is not None:
        for col in FEAT_COLS: eval_col(FEAT_SRC, col, "reranker_feat")

    S=pd.DataFrame(rows)
    S["paired_sig"]=((S.paired_lo>0.5)|(S.paired_hi<0.5)).astype(int)
    S=S.sort_values("auroc_disc",ascending=False)
    S.to_csv(f"{a.out}_summary.tsv",sep="\t",index=False)
    base.to_parquet(f"{a.out}_values.parquet",index=False)
    keep_cols=["gene","is_canonical"]+[c for c in base.columns if c not in ("gene","is_canonical")]
    base[keep_cols].to_csv(f"{a.out}_values.tsv",sep="\t",index=False)

    print("\n===== signal ranking (by |AUROC-0.5|) =====",file=sys.stderr)
    print(f"{'signal':30}{'src':14}{'AUROC':>7}{'paired%':>9}{'sig':>4}",file=sys.stderr)
    for _,r in S.head(25).iterrows():
        au=f"{r.auroc:.3f}" if np.isfinite(r.auroc) else "  nan"
        pf=f"{r.paired_frac:.3f}" if np.isfinite(r.paired_frac) else "  nan"
        print(f"{r.signal:30}{str(r.source):14}{au:>7}{pf:>9}{r.paired_sig:>4}",file=sys.stderr)
    print(f"\n[out] {a.out}_summary.tsv (all {len(S)} signals ranked), {a.out}_values.parquet",file=sys.stderr)

if __name__=="__main__":
    main()