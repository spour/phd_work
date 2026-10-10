#!/usr/bin/env python3
# local, distance-blind features for each candidate site
import argparse, sys
import numpy as np, pandas as pd

PAS_VARIANTS=["AATAAA","ATTAAA","AGTAAA","TATAAA","CATAAA","GATAAA",
              "AATATA","AATACA","AATAGA","ACTAAA","AAGAAA","AATGAA"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--choice_sites", required=True)
    ap.add_argument("--scripts_dir", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--up", type=int, default=3000)
    ap.add_argument("--dn", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup,
                                           init_maxentscan, extract_one_site)
    from compute_gene_baselines import derive_axes
    try:
        import splice_extra_features as XF; HAVE_XF=True
    except Exception: HAVE_XF=False

    fa=GenomeFasta(a.reference_fasta)
    u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    df=pd.read_parquet(a.choice_sites)

    def pas_features(seq, up):
        region=seq[max(0,up-40):up-5].upper()
        best=0; cnt=0
        for k,v in enumerate(PAS_VARIANTS):
            if v in region:
                cnt+=1
                best=max(best, len(PAS_VARIANTS)-k)
        dn_reg=seq[up+5:up+40].upper()
        u_rich=dn_reg.count("T")/max(1,len(dn_reg))
        gu_rich=(dn_reg.count("T")+dn_reg.count("G"))/max(1,len(dn_reg))
        ca=1.0 if seq[up:up+2].upper() in ("CA","TA") else 0.0
        return best, cnt, u_rich, gu_rich, ca

    rows=[]
    for i,r in enumerate(df.itertuples()):
        st=r.strand if r.strand in ("+","-") else "+"
        center=int(r.pos)
        if st=="+": w0=max(0,center-a.up); w1=center+a.dn
        else:       w0=max(0,center-a.dn); w1=center+a.up
        seq=fa.fetch(r.chrom,w0,w1,strand=st)
        rec=dict(gene=r.gene,chrom=r.chrom,pos=r.pos,strand=st,is_true=r.is_true,
                 rf_prob=getattr(r,"rf_prob",np.nan))
        if seq is None or len(seq)<a.up+a.dn-200:
            rows.append(rec); continue
        pas_best,pas_cnt,u_rich,gu_rich,ca=pas_features(seq,a.up)
        rec.update(pas_best=pas_best,pas_count=pas_cnt,dse_u=u_rich,dse_gu=gu_rich,cleavage_ca=ca)
        try:
            feat=extract_one_site(seq,a.up,a.dn,8,me5,score3,m3,u1)
            axes=derive_axes(feat,seq,a.up)
            for k in ("upstream_max_me3","term_3ss","term_ppt","local_me5",
                      "dn5ss_has_strong","pas_prox_u1","upstream_mean_u1","interv_me5_max"):
                v=axes.get(k); rec[k]=float(v) if v is not None else np.nan
        except Exception: pass
        if HAVE_XF:
            try:
                acc=XF._find_acceptor(seq,a.up,score3,m3)
                rec["bp_strength"]=XF.branchpoint_score(seq,a.up,acc_pos=acc)
                rec["u2af65_ppt"]=XF.u2af65_ppt(seq,a.up,acc_pos=acc)
                rec["u1_suppression_down"]=XF.u1_suppression_down(seq,a.up,a.dn)
            except Exception: pass
        rows.append(rec)
        if (i+1)%20000==0: print(f"  {i+1:,}/{len(df):,}",flush=True)
    out=pd.DataFrame(rows)
    feats=[c for c in out.columns if c not in ("gene","chrom","pos","strand","is_true")]
    print(f"[featurized] {len(out):,} sites, {len(feats)} features: {feats}")
    out.to_parquet(a.out,index=False)
    print(f"[done] {a.out}")

if __name__=="__main__":
    main()