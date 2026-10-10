#!/usr/bin/env python3
# (corrected to match the validated pipeline) one row per genome-wide rf peak
import argparse, os, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS={"chr3","chr13"}
AXES=["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt",
      "local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max"]

def find_peaks_nms(scores, positions, min_dist, max_peaks):
    order=np.argsort(scores)[::-1]; chosen=[]; cpos=[]
    for i in order:
        if not np.isfinite(scores[i]): continue
        p=positions[i]
        if any(abs(p-q)<=min_dist for q in cpos): continue
        chosen.append(int(i)); cpos.append(int(p))
        if len(chosen)>=max_peaks: break
    return chosen

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts_dir",required=True,
                    help="dir holding extract_splicing_features.py + compute_gene_baselines.py")
    ap.add_argument("--rf_peaks",default=None,help="_cache_peaks_rf parquet (optional; else NMS on per-gene rf)")
    ap.add_argument("--per_gene_dir",required=True)
    ap.add_argument("--splice_scan_dir",required=True)
    ap.add_argument("--pc_genes_bed",required=True)
    ap.add_argument("--gene_baselines",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True)
    ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--upstream_nt",type=int,default=3000)
    ap.add_argument("--downstream_nt",type=int,default=1000)
    ap.add_argument("--n_bins",type=int,default=8)
    ap.add_argument("--center_offset",type=int,default=249,
                    help="cleavage center = window_start_genomic +/- this (strand-aware)")
    ap.add_argument("--tolerance_nt",type=int,default=250)
    ap.add_argument("--min_peak_distance",type=int,default=250)
    ap.add_argument("--max_peaks",type=int,default=20)
    ap.add_argument("--force_canonical",dest="force_canonical",action="store_true",default=True,
                    help="Always include the canonical window as a candidate (default on). "
                         "This is the reranking denominator: canonical + internal RF competitors.")
    ap.add_argument("--no_force_canonical",dest="force_canonical",action="store_false",
                    help="Disable forcing the canonical in (candidate set = RF peaks only).")
    ap.add_argument("--limit_genes",type=int,default=0)
    ap.add_argument("--xf_test_only",action="store_true",
                    help="compute the new XF features only on test chroms (fast screen)")
    ap.add_argument("--out",default="peak_table.parquet")
    a=ap.parse_args()

    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup,
        load_me5_lookup, init_maxentscan, extract_one_site)
    from compute_gene_baselines import derive_axes
    try:
        import splice_extra_features as XF
        HAVE_XF=True
    except Exception as e:
        print(f"[warn] splice_extra_features not importable ({e}); new features skipped",file=sys.stderr)
        HAVE_XF=False

    fa=GenomeFasta(a.reference_fasta)
    u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()

    pc=pd.read_csv(a.pc_genes_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","score","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]
    pc=pc.drop_duplicates("gene"); pc["gene_length"]=pc["end"]-pc["start"]
    meta=pc.set_index("gene")
    base=pd.read_csv(a.gene_baselines,sep="\t").drop_duplicates("gene",keep="first").set_index("gene")
    have_base=set(c for c in base.columns)

    for label,p,isdir in [("per_gene_dir",a.per_gene_dir,True),
                          ("splice_scan_dir",a.splice_scan_dir,True),
                          ("reference_fasta",a.reference_fasta,False),
                          ("pc_genes_bed",a.pc_genes_bed,False)]:
        ok = os.path.isdir(p) if isdir else os.path.exists(p)
        if not ok:
            sys.exit(f"[err] --{label} not found: {p}\n"
                     f"      (check for an unreplaced '...' placeholder in the path)")
    if a.rf_peaks and not os.path.exists(a.rf_peaks):
        sys.exit(f"[err] --rf_peaks not found: {a.rf_peaks}")
    if "baseline_term_ppt" not in have_base:
        print("[warn] gene_baselines has no baseline_term_ppt column; "
              "term_ppt__wg will be absent",file=sys.stderr)

    if a.rf_peaks:
        rp=pd.read_parquet(a.rf_peaks); rp["gene"]=rp["gene"].astype(str)
        centers_by_gene={g:((d["start"]+d["end"])//2).to_numpy() for g,d in rp.groupby("gene")}
        gene_iter=sorted(centers_by_gene)
    else:
        centers_by_gene=None
        gene_iter=sorted(f[:-8] for f in os.listdir(a.per_gene_dir) if f.endswith(".parquet"))
    if a.limit_genes>0: gene_iter=gene_iter[:a.limit_genes]

    rows=[]
    cnt=dict(tried=0,not_in_meta=0,missing_files=0,read_fail=0,empty_merge=0,no_peaks=0,extract_fail=0)
    for gi,gene in enumerate(tqdm(gene_iter,desc='genes',unit='gene')):
        cnt['tried']+=1
        if gene not in meta.index: cnt['not_in_meta']+=1; continue
        gm=meta.loc[gene]; chrom=gm["chrom"]; strand=gm["strand"]; glen=int(gm["gene_length"])
        rfp=os.path.join(a.per_gene_dir,f"{gene}.parquet"); spp=os.path.join(a.splice_scan_dir,f"{gene}.parquet")
        if not (os.path.exists(rfp) and os.path.exists(spp)): cnt['missing_files']+=1; continue
        try:
            cols=["window_start_genomic","rf_prob","is_constitutive_cpa"]
            for _col in ("lr_prob","cnn_prob","aparent2_prob"): cols.append(_col)
            rf=pd.read_parquet(rfp,columns=cols)
            sp=pd.read_parquet(spp,columns=["window_start_genomic","splicing_prob"])
        except Exception:
            cnt['read_fail']+=1; continue
        m=rf.merge(sp,on="window_start_genomic",how="inner").dropna(subset=["rf_prob"]).reset_index(drop=True)
        if len(m)==0: cnt['empty_merge']+=1; continue
        pos=m["window_start_genomic"].to_numpy(); rfv=m["rf_prob"].to_numpy()
        canon=m.loc[m["is_constitutive_cpa"]==True,"window_start_genomic"].to_numpy()
        if centers_by_gene is not None and gene in centers_by_gene:
            idx=sorted({int(np.argmin(np.abs(pos-c))) for c in centers_by_gene[gene]})
        else:
            idx=find_peaks_nms(rfv,pos,a.min_peak_distance,a.max_peaks)
        if a.force_canonical and len(canon)>0:
            idx=set(idx)
            for cp in canon:
                idx.add(int(np.argmin(np.abs(pos-cp))))
            idx=sorted(idx)
        for j in idx:
            ws=int(pos[j])
            center = ws + a.center_offset if strand=="+" else ws - a.center_offset
            if strand=="+": w0=max(0,center-a.upstream_nt); w1=center+a.downstream_nt
            else:           w0=max(0,center-a.downstream_nt); w1=center+a.upstream_nt
            try:
                seq=fa.fetch(chrom,w0,w1,strand=strand)
            except Exception:
                cnt['extract_fail']+=1; continue
            if seq is None or len(seq)<(a.upstream_nt+a.downstream_nt-100): cnt['extract_fail']+=1; continue
            try:
                feat=extract_one_site(seq,a.upstream_nt,a.downstream_nt,a.n_bins,me5,score3,m3,u1)
                axes=derive_axes(feat,seq,a.upstream_nt)
            except Exception:
                cnt['extract_fail']+=1; continue
            in_zone=int(len(canon)>0 and np.min(np.abs(canon-ws))<=a.tolerance_nt)
            rec=dict(gene=gene,chrom=chrom,position=ws,
                     rf_prob=float(rfv[j]),
                     lr_prob=float(m["lr_prob"].iloc[j]) if "lr_prob" in m else np.nan,
                     cnn_prob=float(m["cnn_prob"].iloc[j]) if "cnn_prob" in m else np.nan,
                     aparent2_prob=float(m["aparent2_prob"].iloc[j]) if "aparent2_prob" in m else np.nan,
                     splicing_prob=float(m["splicing_prob"].iloc[j]),
                     in_zone=in_zone,gene_length=glen)
            for ax in AXES:
                v=axes.get(ax,np.nan); rec[ax]=float(v) if v is not None else np.nan
                bcol=f"baseline_{ax}"
                if gene in base.index and bcol in have_base and np.isfinite(v):
                    bv=base.at[gene,bcol]
                    if pd.notna(bv):
                        rec[f"{ax}__wg"]=float(v)-float(bv)
            # new interpretable features (literature-motivated)
            _c2 = chrom if str(chrom).startswith("chr") else "chr"+str(chrom)
            if HAVE_XF and (not a.xf_test_only or _c2 in TEST_CHROMS):
                try:
                    acc=XF._find_acceptor(seq,a.upstream_nt,score3,m3)
                    rec["bp_strength"]=XF.branchpoint_score(seq,a.upstream_nt,acc_pos=acc)
                    rec["u2af65_ppt"]=XF.u2af65_ppt(seq,a.upstream_nt,acc_pos=acc)
                    rec["u2af35_ag"]=XF.u2af35_ag(seq,a.upstream_nt,acc_pos=acc)
                    rec["u1_suppression_down"]=XF.u1_suppression_down(seq,a.upstream_nt,a.downstream_nt)
                    for nf in ("bp_strength","u2af65_ppt","u2af35_ag","u1_suppression_down"):
                        bcol=f"baseline_{nf}"
                        if gene in base.index and bcol in have_base and np.isfinite(rec.get(nf,np.nan)):
                            bv=base.at[gene,bcol]
                            if pd.notna(bv): rec[f"{nf}__wg"]=float(rec[nf])-float(bv)
                except Exception:
                    pass
            rows.append(rec)
        if a.limit_genes==0 and (gi+1)%2000==0:
            print(f"  {gi+1:,}/{len(gene_iter):,} genes, {len(rows):,} peaks, drops={cnt}",file=sys.stderr)

    print(f"[drops] tried={cnt['tried']} not_in_meta={cnt['not_in_meta']} "
          f"missing_files={cnt['missing_files']} read_fail={cnt['read_fail']} "
          f"empty_merge={cnt['empty_merge']} no_peaks={cnt['no_peaks']} "
          f"extract_fail={cnt['extract_fail']} -> kept {len(rows):,} peaks",file=sys.stderr)
    if len(rows)==0:
        ex=gene_iter[:5]
        print("[err] zero peaks collected. First 5 genes attempted and file checks:",file=sys.stderr)
        for g in ex:
            rfp=os.path.join(a.per_gene_dir,f"{g}.parquet"); spp=os.path.join(a.splice_scan_dir,f"{g}.parquet")
            print(f"   {g}: in_meta={g in meta.index} rf={os.path.exists(rfp)} splice={os.path.exists(spp)}",file=sys.stderr)
        sys.exit("[err] nothing to write. Fix the failing stage above "
                 "(most often a wrong --per_gene_dir/--splice_scan_dir path or a gene-name mismatch).")
    out=pd.DataFrame(rows)
    # collapse in_zone to one canonical candidate per gene
    if len(out):
        iz=out[out.in_zone==1]
        keep_idx=iz.loc[iz.groupby("gene")["rf_prob"].idxmax()].index
        drop_idx=set(iz.index)-set(keep_idx)
        n_before=len(out); out=out.drop(index=list(drop_idx)).reset_index(drop=True)
        print(f"[collapse] dropped {len(drop_idx):,} redundant in_zone windows "
              f"({n_before:,} -> {len(out):,} rows); one canonical per gene",file=sys.stderr)
    out.to_parquet(a.out,index=False)
    c2=out["chrom"].where(out["chrom"].str.startswith("chr"),"chr"+out["chrom"])
    split=np.where(c2.isin(TEST_CHROMS),"test",np.where(c2.isin(VAL_CHROMS),"val","train"))
    out["_split"]=split
    wg=[c for c in out.columns if c.endswith("__wg")]
    print(f"[done] {len(out):,} peaks, {out.gene.nunique():,} genes -> {a.out}",file=sys.stderr)
    print(f"[split] {pd.Series(split).value_counts().to_dict()}",file=sys.stderr)
    print(f"[features] {len(wg)} __wg cols: {wg}",file=sys.stderr)
    te=out[out._split=="test"]
    if len(te):
        p1=te.groupby("gene").apply(lambda g:int(g.iloc[int(np.argmax(g.rf_prob.to_numpy()))].in_zone==1)
                                    if g.in_zone.sum()>0 else 0).mean()
        print(f"[smoke] test cpa_only P@1 = {p1:.4f} (expect ~0.406)",file=sys.stderr)
        if "term_ppt__wg" in out.columns:
            can=te[te.in_zone==1]["term_ppt__wg"].dropna(); cry=te[te.in_zone==0]["term_ppt__wg"].dropna()
            if len(can) and len(cry):
                allv=np.concatenate([can.values,cry.values]); lab=np.concatenate([np.ones(len(can)),np.zeros(len(cry))])
                order=np.argsort(allv); ranks=np.empty(len(allv)); ranks[order]=np.arange(1,len(allv)+1)
                auc=(ranks[lab==1].sum()-len(can)*(len(can)+1)/2)/(len(can)*len(cry))
                print(f"[GATE] term_ppt__wg canonical-vs-internal AUROC = {auc:.3f} "
                      f"(canon mean {can.mean():+.3f} vs internal {cry.mean():+.3f}); "
                      f">0.55 = carries signal, ~0.50 = dead",file=sys.stderr)
    out.drop(columns=["_split"],inplace=True)

if __name__=="__main__":
    main()