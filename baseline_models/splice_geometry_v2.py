#!/usr/bin/env python3
# better test of the u1-relay / telescripting model on dominant vs cryptic cpa sites
import argparse, sys, time
import numpy as np, pandas as pd

TAUS=(200,500,1000,2000)
WINDOWS=(500,1000,2000,5000,10000)
WEAK_5SS=3.0; WEAK_3SS=3.0
PAD=30; ACCEPT_WIN=40

def load_bed(path):
    rows=[]
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip(): continue
            f=ln.rstrip("\n").split("\t")
            rows.append(dict(chrom=f[0],start=int(f[1]),end=int(f[2]),
                             name=f[3] if len(f)>3 else ".",
                             strand=f[5] if len(f)>5 else "+"))
    return pd.DataFrame(rows)

def auc_mw(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int); ok=np.isfinite(s); s=s[ok]; y=y[ok]
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank(method="average").to_numpy()
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def cohens_d(x,y):
    x=np.asarray(x,float); y=np.asarray(y,int); ok=np.isfinite(x); x=x[ok]; y=y[ok]
    a=x[y==1]; b=x[y==0]
    if len(a)<2 or len(b)<2: return np.nan
    return (a.mean()-b.mean())/(np.sqrt((a.var()+b.var())/2)+1e-9)

def load_internal_exon_3ends(gtf, max_n=4000):
    from collections import defaultdict
    tx=defaultdict(list)
    with open(gtf) as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            f=ln.split("\t")
            if len(f)<9 or f[2]!="exon": continue
            tid=None
            for kv in f[8].split(";"):
                kv=kv.strip()
                if kv.startswith("transcript_id"): tid=kv.split('"')[1]; break
            if tid: tx[tid].append((int(f[3]),int(f[4]),f[6],f[0]))
    rows=[]
    for tid,ex in tx.items():
        if len(ex)<3: continue
        ex=sorted(ex)
        strand=ex[0][2]; chrom=ex[0][3]
        internal=ex[1:-1]
        for (s,e,st,ch) in internal:
            pos = e if strand=="+" else s
            rows.append(dict(chrom=chrom,start=pos-1,end=pos+1,name=f"intexon_{tid}",
                             strand=strand,center=pos,label=2))
            if len(rows)>=max_n: break
        if len(rows)>=max_n: break
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--used",required=True); ap.add_argument("--cryptic",required=True)
    ap.add_argument("--scripts_dir",required=True); ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",default=None)
    ap.add_argument("--gtf",default=None,help="enable internal-exon positive control")
    ap.add_argument("--max_scan",type=int,default=10000)
    ap.add_argument("--stride",type=int,default=1,help="scan every Nth position (speed)")
    ap.add_argument("--weak_5ss",type=float,default=WEAK_5SS); ap.add_argument("--weak_3ss",type=float,default=WEAK_3SS)
    ap.add_argument("--max_sites",type=int,default=0)
    ap.add_argument("--log_every",type=int,default=2000)
    ap.add_argument("--out",default="splice_geom2")
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    import extract_splicing_features as esf
    fa=esf.GenomeFasta(a.reference_fasta)
    score5=score3=None
    if hasattr(esf,"init_maxentscan"):
        try:
            ms=esf.init_maxentscan()
            if isinstance(ms,(tuple,list)) and len(ms)>=2: score5,score3=ms[0],ms[1]
            elif isinstance(ms,dict): score5=ms.get("score5"); score3=ms.get("score3")
        except Exception as e: print(f"[warn] init_maxentscan: {e}",file=sys.stderr)
    for c in ["score5","score_5","maxent5"]:
        if score5 is None and hasattr(esf,c): score5=getattr(esf,c)
    for c in ["score3","score_3","maxent3"]:
        if score3 is None and hasattr(esf,c): score3=getattr(esf,c)
    if score5 is None or score3 is None:
        sys.exit("[fatal] could not resolve score5/score3; edit the import shim.")
    u1=None
    if a.u1_lookup:
        arr=np.load(a.u1_lookup,allow_pickle=True); u1=arr.item() if arr.dtype==object else dict(arr)

    used=load_bed(a.used); used["label"]=1; used["center"]=((used.start+used.end)//2).astype(int)
    cry=load_bed(a.cryptic); cry["label"]=0; cry["center"]=((cry.start+cry.end)//2).astype(int)
    if a.max_sites>0:
        used=used.sample(min(a.max_sites,len(used)),random_state=0)
        cry=cry.sample(min(a.max_sites,len(cry)),random_state=0)
    parts=[used,cry]
    if a.gtf:
        print("[ctrl] parsing internal-exon 3' ends (positive control)...",file=sys.stderr)
        ie=load_internal_exon_3ends(a.gtf,max_n=(a.max_sites or 4000))
        print(f"[ctrl] {len(ie):,} internal-exon controls",file=sys.stderr)
        parts.append(ie)
    df=pd.concat(parts,ignore_index=True)
    print(f"[geom] {len(df):,} sites ({int((df.label==1).sum())} used, "
          f"{int((df.label==0).sum())} cryptic, {int((df.label==2).sum())} ctrl)",file=sys.stderr)

    S=a.stride
    def scan5_dir(seq):
        ds=[]; sc=[]
        n=len(seq)-9+1
        for i in range(PAD, n, S):
            w=seq[i:i+9]
            if "N" in w or w[3:5]!="GT": continue
            try: v=score5(w)
            except Exception: continue
            ds.append(i-PAD); sc.append(v)
        return np.array(ds,float), np.array(sc,float)
    def scan3_dir(seq, center_idx):
        ds=[]; sc=[]
        n=len(seq)-23+1
        for i in range(0, min(center_idx,n), S):
            w=seq[i:i+23]
            if "N" in w or w[18:20]!="AG": continue
            try: v=score3(w)
            except Exception: continue
            ds.append(center_idx-i); sc.append(v)
        return np.array(ds,float), np.array(sc,float)
    def scan5_up(seq, center_idx):
        ds=[]; sc=[]
        n=len(seq)-9+1
        for i in range(0, min(center_idx,n), S):
            w=seq[i:i+9]
            if "N" in w or w[3:5]!="GT": continue
            try: v=score5(w)
            except Exception: continue
            ds.append(center_idx-i); sc.append(v)
        return np.array(ds,float), np.array(sc,float)
    def scanu1_ds(seq):
        if u1 is None: return np.array([]),np.array([])
        ds=[]; sc=[]; n=len(seq)-9+1
        for i in range(PAD,n,S):
            w=seq[i:i+9]
            if "N" in w: continue
            v=u1.get(w)
            if v is not None: ds.append(i-PAD); sc.append(-float(v))
        return np.array(ds,float),np.array(sc,float)
    def burden(dist,score,tau,thr=None):
        if dist.size==0: return 0.0
        s=score.copy()
        if thr is not None: s=np.where(s>=thr,s,0.0)
        s=np.maximum(s,0.0)
        return float(np.sum(s*np.exp(-dist/float(tau))))
    def nearest(dist,score,thr):
        m=score>=thr
        if not m.any(): return np.nan,np.nan
        k=int(np.argmin(dist[m]))
        return float(dist[m][k]), float(score[m][k])

    feats=[]; t0=time.time()
    for idx,r in enumerate(df.itertuples(index=False)):
        c=int(r.center); st=r.strand; ch=r.chrom
        if st=="+": ds_seq=fa.fetch(ch,c-PAD,c+a.max_scan,strand=st)
        else:       ds_seq=fa.fetch(ch,c-a.max_scan,c+PAD,strand=st)
        if st=="+": us_seq=fa.fetch(ch,c-a.max_scan,c+PAD,strand=st)
        else:       us_seq=fa.fetch(ch,c-PAD,c+a.max_scan,strand=st)
        row={}
        # downstream 5'ss (adaptive + burden + window sweep)
        if ds_seq:
            d5d,d5s=scan5_dir(ds_seq)
            nd,ns=nearest(d5d,d5s,a.weak_5ss)
            row["near_ds_5ss_dist"]=nd; row["near_ds_5ss_score"]=ns
            for T in TAUS: row[f"ds_5ss_burden_t{T}"]=burden(d5d,d5s,T,thr=a.weak_5ss)
            for W in WINDOWS:
                m=d5d<=W
                row[f"ds_5ss_max_{W}"]=float(d5s[m].max()) if m.any() and np.isfinite(d5s[m]).any() else np.nan
            if u1 is not None:
                ud,us_=scanu1_ds(ds_seq)
                for T in TAUS: row[f"ds_u1_burden_t{T}"]=burden(ud,us_,T)
        # upstream 3'ss + ppt
        if us_seq:
            ci=a.max_scan
            u3d,u3s=scan3_dir(us_seq,ci)
            nd,ns=nearest(u3d,u3s,a.weak_3ss)
            row["near_us_3ss_dist"]=nd; row["near_us_3ss_score"]=ns
            for T in TAUS: row[f"us_3ss_burden_t{T}"]=burden(u3d,u3s,T,thr=a.weak_3ss)
            acc=us_seq[ci-ACCEPT_WIN:ci] if len(us_seq)>=ci else us_seq
            row["ppt_strength"]=((acc.count("C")+acc.count("T"))/len(acc)) if acc else np.nan
            u5d,u5s=scan5_up(us_seq,ci)
            nd,ns=nearest(u5d,u5s,a.weak_5ss)
            row["near_us_5ss_dist"]=nd; row["near_us_5ss_score"]=ns
        feats.append(row)
        if (idx+1)%a.log_every==0:
            print(f"[geom]   {idx+1:,}/{len(df):,}  {(time.time()-t0)/60:.1f}m",file=sys.stderr)
    F=pd.DataFrame(feats); out=pd.concat([df.reset_index(drop=True),F],axis=1)
    out.to_parquet(f"{a.out}.parquet",index=False)
    print(f"[geom] wrote {a.out}.parquet  ({F.shape[1]} features)",file=sys.stderr)

    # analysis: used(1) vs cryptic(0) only
    uc=out[out.label.isin([0,1])]; y=uc["label"].to_numpy().astype(int)
    exp_sign={"near_ds_5ss_dist":+1,"near_ds_5ss_score":-1,"near_us_3ss_dist":-1,
              "near_us_3ss_score":+1,"ppt_strength":+1,"near_us_5ss_score":-1}
    for T in TAUS: exp_sign[f"ds_5ss_burden_t{T}"]=-1; exp_sign[f"us_3ss_burden_t{T}"]=+1; exp_sign[f"ds_u1_burden_t{T}"]=-1
    print(f"\n=== discrimination used(1) vs cryptic(0): AUC, cohen_d, matches U1-relay? ===",file=sys.stderr)
    rows=[]
    for f in F.columns:
        if f.startswith("ds_5ss_max_"): continue
        au=auc_mw(uc[f],y); d=cohens_d(uc[f],y)
        v=""
        if f in exp_sign and np.isfinite(d):
            ok=(d>0)==(exp_sign[f]>0); v="  match" if (ok and abs(d)>0.05) else ("  n.s." if abs(d)<=0.05 else "  OPPOSITE")
        rows.append(dict(feature=f,auc=au,cohens_d=d,matches=v.strip()))
        print(f"  {f:<22}{au:>8.3f}{d:>9.3f}{v}",file=sys.stderr)
    pd.DataFrame(rows).to_csv(f"{a.out}_discrim.tsv",sep="\t",index=False)

    # window sweep: does ds_5ss_max discriminate at any window?
    print(f"\n=== (C) WINDOW SWEEP: ds_5ss_max discrimination by window width ===",file=sys.stderr)
    print(f"  {'window':>8}{'AUC':>9}{'cohen_d':>10}",file=sys.stderr)
    for W in WINDOWS:
        col=f"ds_5ss_max_{W}"
        if col in uc: print(f"  {W:>8}{auc_mw(uc[col],y):>9.3f}{cohens_d(uc[col],y):>10.3f}",file=sys.stderr)
    print(f"  (flat ~0.5 across windows = v1 null was REAL, not a window artifact)",file=sys.stderr)

    if (out.label==2).any():
        print(f"\n=== (D) POSITIVE CONTROL: downstream 5'ss strength by group ===",file=sys.stderr)
        for W in (500,1000):
            col=f"ds_5ss_max_{W}"
            if col not in out: continue
            mu=out.loc[out.label==1,col].median(); mc=out.loc[out.label==0,col].median(); mi=out.loc[out.label==2,col].median()
            print(f"  median {col}:  used={mu:.2f}  cryptic={mc:.2f}  internal-exon-ctrl={mi:.2f}",file=sys.stderr)
        print(f"  (if ctrl >> used/cryptic, the scanner DETECTS downstream 5'ss when present,",file=sys.stderr)
        print(f"   so a flat used-vs-cryptic result is a real biological null, not a scan bug)",file=sys.stderr)
    print(f"\n[out] {a.out}.parquet, {a.out}_discrim.tsv",file=sys.stderr)

if __name__=="__main__": main()