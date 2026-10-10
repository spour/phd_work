#!/usr/bin/env python3
# test the u1-relay / telescripting model (yoon, quan, soles
import argparse, sys, time
import numpy as np, pandas as pd

# scan windows (transcribed orientation
SCAN_DS = 1000
SCAN_US = 1000
PAD     = 30
WEAK_5SS = 3.0

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
    s=np.asarray(s,float); y=np.asarray(y,int)
    ok=np.isfinite(s); s=s[ok]; y=y[ok]
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank(method="average").to_numpy()
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def cohens_d(x,y):
    x=np.asarray(x,float); y=np.asarray(y,int); ok=np.isfinite(x); x=x[ok]; y=y[ok]
    a=x[y==1]; b=x[y==0]
    if len(a)<2 or len(b)<2: return np.nan
    return (a.mean()-b.mean())/(np.sqrt((a.var()+b.var())/2)+1e-9)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--used",required=True); ap.add_argument("--cryptic",required=True)
    ap.add_argument("--scripts_dir",required=True); ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",default=None,help="optional {9mer:dG} npy for downstream U1 feature")
    ap.add_argument("--scan_ds",type=int,default=SCAN_DS); ap.add_argument("--scan_us",type=int,default=SCAN_US)
    ap.add_argument("--weak_5ss",type=float,default=WEAK_5SS)
    ap.add_argument("--max_sites",type=int,default=0,help="0 = all; else subsample per class for a quick run")
    ap.add_argument("--log_every",type=int,default=3000)
    ap.add_argument("--out",default="splice_geom")
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
        except Exception as e:
            print(f"[warn] init_maxentscan() failed: {e}",file=sys.stderr)
    for cand in ["score5","score_5","maxent5"]:
        if score5 is None and hasattr(esf,cand): score5=getattr(esf,cand)
    for cand in ["score3","score_3","maxent3"]:
        if score3 is None and hasattr(esf,cand): score3=getattr(esf,cand)
    if score5 is None or score3 is None:
        sys.exit("[fatal] could not resolve score5/score3 from extract_splicing_features; "
                 "edit the import shim at the top of main() to match your function names.")

    u1=None
    if a.u1_lookup:
        arr=np.load(a.u1_lookup,allow_pickle=True); u1=arr.item() if arr.dtype==object else dict(arr)

    used=load_bed(a.used); used["label"]=1
    cry=load_bed(a.cryptic); cry["label"]=0
    if a.max_sites>0:
        used=used.sample(min(a.max_sites,len(used)),random_state=0)
        cry=cry.sample(min(a.max_sites,len(cry)),random_state=0)
    df=pd.concat([used,cry],ignore_index=True)
    df["center"]=((df.start+df.end)//2).astype(int)
    print(f"[geom] {len(df):,} sites ({int((df.label==1).sum())} used, {int((df.label==0).sum())} cryptic)",file=sys.stderr)

    def fetch_window(chrom,center,strand,up,dn):
        if strand=="+": gs,ge=center-up,center+dn
        else:           gs,ge=center-dn,center+up
        return fa.fetch(chrom,gs,ge,strand=strand)

    def scan5(seq):
        n=len(seq)-9+1; out=np.full(max(n,0),np.nan,np.float32)
        for i in range(max(n,0)):
            w=seq[i:i+9]
            if "N" in w: continue
            if w[3:5]!="GT": continue
            try: out[i]=score5(w)
            except Exception: pass
        return out
    def scan3(seq):
        n=len(seq)-23+1; out=np.full(max(n,0),np.nan,np.float32)
        for i in range(max(n,0)):
            w=seq[i:i+23]
            if "N" in w: continue
            if w[18:20]!="AG": continue
            try: out[i]=score3(w)
            except Exception: pass
        return out
    def scanu1(seq):
        if u1 is None: return None
        n=len(seq)-9+1; out=np.full(max(n,0),np.nan,np.float32)
        for i in range(max(n,0)):
            w=seq[i:i+9]
            if "N" in w: continue
            v=u1.get(w)
            if v is not None: out[i]=-float(v)
        return out
    def ppt_score(seq_up, accept_win=40):
        s=seq_up[-accept_win:] if len(seq_up)>=accept_win else seq_up
        if not s: return np.nan
        return (s.count("C")+s.count("T"))/len(s)

    feats=[]; t0=time.time()
    for i,r in enumerate(df.itertuples(index=False)):
        c=int(r.center); st=r.strand; ch=r.chrom
        ds=fetch_window(ch,c,st,PAD,a.scan_ds)
        us=fetch_window(ch,c,st,a.scan_us,PAD)
        # downstream 5'ss (p1,p3)
        d5=scan5(ds) if ds else np.array([])
        ds_5ss_max=ds_5ss_max_dist=ds_5ss_near_dist=np.nan
        if d5.size:
            pos=np.arange(d5.size)
            dn_mask=pos>=PAD
            dd=pos[dn_mask]-PAD
            vv=d5[dn_mask]
            fin=np.isfinite(vv)
            if fin.any():
                dd2=dd[fin]; vv2=vv[fin]
                k=int(np.argmax(vv2)); ds_5ss_max=float(vv2[k]); ds_5ss_max_dist=float(dd2[k])
                strong=vv2>=a.weak_5ss
                if strong.any(): ds_5ss_near_dist=float(dd2[strong].min())
        ds_u1_max=np.nan
        if u1 is not None and ds:
            uu=scanu1(ds)
            if uu is not None and uu.size:
                pos=np.arange(uu.size); dn=pos>=PAD; vv=uu[dn]
                if np.isfinite(vv).any(): ds_u1_max=float(np.nanmax(vv))
        # upstream 3'ss + ppt (p2)
        u3=scan3(us) if us else np.array([])
        us_3ss_max=us_3ss_max_dist=np.nan
        if u3.size:
            pos=np.arange(u3.size); up_mask=pos< a.scan_us; uu=u3[up_mask]
            updist=a.scan_us-pos[up_mask]
            fin=np.isfinite(uu)
            if fin.any():
                uu2=uu[fin]; ud2=updist[fin]
                k=int(np.argmax(uu2)); us_3ss_max=float(uu2[k]); us_3ss_max_dist=float(ud2[k])
        ppt=ppt_score(us[:a.scan_us] if us else "")
        # upstream 5'ss (p5 degradation code)
        u5=scan5(us) if us else np.array([])
        us_5ss_max=np.nan
        if u5.size:
            pos=np.arange(u5.size); up_mask=pos< a.scan_us; uu=u5[up_mask]
            if np.isfinite(uu).any(): us_5ss_max=float(np.nanmax(uu))
        feats.append(dict(ds_5ss_max=ds_5ss_max,ds_5ss_max_dist=ds_5ss_max_dist,
                          ds_5ss_near_dist=ds_5ss_near_dist,ds_u1_max=ds_u1_max,
                          us_3ss_max=us_3ss_max,us_3ss_max_dist=us_3ss_max_dist,
                          ppt_strength=ppt,us_5ss_max=us_5ss_max))
        if (i+1)%a.log_every==0:
            print(f"[geom]   {i+1:,}/{len(df):,}  {(time.time()-t0)/60:.1f}m",file=sys.stderr)
    F=pd.DataFrame(feats); out=pd.concat([df.reset_index(drop=True),F],axis=1)
    out.to_parquet(f"{a.out}.parquet",index=False)
    print(f"[geom] wrote {a.out}.parquet",file=sys.stderr)

    y=out["label"].to_numpy().astype(int)
    print(f"\n=== U1-relay predictions: dominant(used)=1 vs cryptic=0 ===",file=sys.stderr)
    print(f"  {'feature':<18}{'AUC':>8}{'cohen_d':>9}   prediction",file=sys.stderr)
    preds={"ds_5ss_max":"cryptic higher (P1: downstream 5'ss suppresses) -> AUC<0.5",
           "ds_5ss_near_dist":"cryptic SMALLER dist (P1/P3: closer downstream 5'ss) -> AUC>0.5 for dist means used farther",
           "ds_u1_max":"cryptic higher (P1) -> AUC<0.5",
           "us_3ss_max":"used higher (P2: licensing 3'ss) -> AUC>0.5",
           "ppt_strength":"used higher (P2: acceptor PPT) -> AUC>0.5",
           "us_5ss_max":"cryptic higher (P5: degradation code) -> AUC<0.5"}
    exp_sign={"ds_5ss_max":-1,"ds_5ss_near_dist":+1,"ds_u1_max":-1,
              "us_3ss_max":+1,"ppt_strength":+1,"us_5ss_max":-1}
    rows=[]
    for f in ["ds_5ss_max","ds_5ss_max_dist","ds_5ss_near_dist","ds_u1_max",
              "us_3ss_max","us_3ss_max_dist","ppt_strength","us_5ss_max"]:
        au=auc_mw(out[f],y); d=cohens_d(out[f],y)
        verdict=""
        if f in exp_sign and np.isfinite(d):
            ok = (d>0)==(exp_sign[f]>0)
            verdict = "  <-- matches U1-relay" if ok and abs(d)>0.05 else ("  (n.s.)" if abs(d)<=0.05 else "  <-- OPPOSITE")
        rows.append(dict(feature=f,auc=au,cohens_d=d,expected_sign=exp_sign.get(f,0),
                         matches=( (np.isfinite(d) and f in exp_sign and (d>0)==(exp_sign[f]>0) and abs(d)>0.05) )))
        print(f"  {f:<18}{au:>8.3f}{d:>9.3f}   {preds.get(f,'')}{verdict}",file=sys.stderr)
    pd.DataFrame(rows).to_csv(f"{a.out}_discrim.tsv",sep="\t",index=False)

    print(f"\n=== P3 distance-dependence (downstream-5'ss suppression decays ~1kb) ===",file=sys.stderr)
    dd=out["ds_5ss_near_dist"].to_numpy(float)
    have=np.isfinite(dd)
    if have.sum()>100:
        q=np.quantile(dd[have],[0,.2,.4,.6,.8,1.0])
        print(f"  {'dist bin (nt)':<18}{'n':>7}{'used_frac':>11}",file=sys.stderr)
        for b in range(5):
            m=have & (dd>=q[b]) & (dd<=q[b+1])
            if m.sum()>0:
                print(f"  {f'{q[b]:.0f}-{q[b+1]:.0f}':<18}{int(m.sum()):>7}{y[m].mean():>11.3f}",file=sys.stderr)
        print(f"  (P3: used_frac should RISE with distance if downstream-5'ss suppresses)",file=sys.stderr)
    sentinel=out["ds_5ss_near_dist"].fillna(out["ds_5ss_near_dist"].max()+a.scan_ds)
    print(f"  AUC of ds_5ss_near_dist (NaN=far sentinel) = {auc_mw(sentinel,y):.3f}",file=sys.stderr)

    print(f"\n=== P4 combinatorial (upstream-3'ss x downstream-5'ss) ===",file=sys.stderr)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        def auc_of(cols):
            X=out[cols].to_numpy(float)
            X=np.nan_to_num(X,nan=np.nanmedian(X)) if np.isnan(X).any() else X
            p=cross_val_predict(LogisticRegression(max_iter=1000),X,y,cv=5,method="predict_proba")[:,1]
            return auc_mw(p,y)
        a_ds=auc_of(["ds_5ss_max","ds_5ss_near_dist"])
        a_us=auc_of(["us_3ss_max","ppt_strength"])
        a_both=auc_of(["ds_5ss_max","ds_5ss_near_dist","us_3ss_max","ppt_strength"])
        print(f"  downstream-5'ss only : AUC {a_ds:.3f}",file=sys.stderr)
        print(f"  upstream-3'ss only   : AUC {a_us:.3f}",file=sys.stderr)
        print(f"  combined (P4)        : AUC {a_both:.3f}  (should beat either alone)",file=sys.stderr)
    except Exception as e:
        print(f"  [skip] {e}",file=sys.stderr)
    print(f"\n[out] {a.out}.parquet, {a.out}_discrim.tsv",file=sys.stderr)

if __name__=="__main__": main()