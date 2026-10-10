#!/usr/bin/env python3
# quick per-candidate ranking test for cpa x splice interaction
import argparse, sys, os, tempfile
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

RF_TRAIN={"chr1","chr2","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
          "chr11","chr12","chr14","chr15","chr16","chr17"}
RF_TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--scripts_dir", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--pwms", required=True)
    ap.add_argument("--cpa_window", type=int, default=500,
                    help=">=500; if >500, tiled at --cpa_stride and max-pooled to one vector")
    ap.add_argument("--cpa_stride", type=int, default=250)
    ap.add_argument("--center_offset", type=int, default=249)
    ap.add_argument("--max_train_genes", type=int, default=4000)
    ap.add_argument("--max_test_genes", type=int, default=1200)
    ap.add_argument("--bg", nargs=4, type=float, default=[0.3,0.2,0.2,0.3])
    ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args()
    sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import GenomeFasta
    from featurization import read_pwms, featurize_fasta_to_df

    rng=np.random.default_rng(a.seed)
    df=pd.read_parquet(a.peak_table); df["chrom"]=df["chrom"].astype(str)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"])
    SPL=[c for c in ["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt",
        "local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max"] if c in df.columns]

    trg=df[df.chrom.isin(RF_TRAIN)].gene.unique()
    teg=df[df.chrom.isin(RF_TEST)].gene.unique()
    trg=rng.choice(trg,size=min(a.max_train_genes,len(trg)),replace=False)
    teg=rng.choice(teg,size=min(a.max_test_genes,len(teg)),replace=False)
    sub=df[df.gene.isin(set(trg)|set(teg))].copy().reset_index(drop=True)
    print(f"[mini] train genes {len(trg):,}  test genes {len(teg):,}  candidates {len(sub):,}", flush=True)

    pc=pd.read_csv(a.pc_genes_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","sc","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]
    strand=pc.drop_duplicates("gene").set_index("gene")["strand"].to_dict()
    sub["strand"]=sub["gene"].map(strand).fillna("+")

    fa=GenomeFasta(a.reference_fasta)
    pwms=read_pwms(a.pwms, max_len=20, bg=np.array(a.bg))
    half=a.cpa_window//2
    ids=[]; seqs=[]
    for row in sub.itertuples():
        st=row.strand; ws=int(row.position)
        center = ws+a.center_offset if st=="+" else ws-a.center_offset
        seq=fa.fetch(row.chrom, max(0,center-half), center+half, strand=st)
        ids.append(f"{row.gene}|{row.chrom}|{ws}")
        seqs.append(seq if seq else "N"*a.cpa_window)
    sub["cid"]=ids
    print(f"[mini] featurizing {len(ids):,} candidate windows ({a.cpa_window} nt)...", flush=True)
    tmp=tempfile.NamedTemporaryFile("w",suffix=".fa",delete=False)
    for cid,seq in zip(ids,seqs): tmp.write(f">{cid}\n{seq}\n")
    tmp.close()
    cpa_df=featurize_fasta_to_df(tmp.name, pwms, bg=np.array(a.bg))
    if a.cpa_window>500:
        cpa_df.index=[ix.rsplit("_",1)[0] if "_" in ix else ix for ix in cpa_df.index]
        cpa_df=cpa_df.groupby(level=0).max()
    os.unlink(tmp.name)
    cpa_cols=list(cpa_df.columns)
    cpa_df=cpa_df.reindex(sub["cid"].values).reset_index(drop=True)
    sub=pd.concat([sub, cpa_df], axis=1)

    def contested(grp):
        iz=grp.in_zone.to_numpy(); rf=grp.rf_prob.to_numpy()
        if iz.sum()==0: return False
        return bool((rf[iz==0]>=rf[np.argmax(iz)]).any())

    def eval_set(cols, trm, tem):
        Xtr=sub.loc[trm, cols].fillna(0).to_numpy(); ytr=sub.loc[trm,"in_zone"].to_numpy()
        clf=HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
            max_leaf_nodes=63, random_state=0)
        clf.fit(Xtr,ytr)
        te=sub.loc[tem].copy()
        te["s"]=clf.predict_proba(te[cols].fillna(0).to_numpy())[:,1]
        recs=[]
        for g,grp in te.groupby("gene",sort=False):
            iz=grp.in_zone.to_numpy()
            if iz.sum()==0: continue
            top=int(np.argmax(grp["s"].to_numpy()))
            recs.append((contested(grp), int(iz[top]==1), iz.sum()/len(grp)))
        return pd.DataFrame(recs,columns=["con","hit","rand"])

    trm=sub.chrom.isin(RF_TRAIN); tem=sub.chrom.isin(RF_TEST)
    sub["rf_prob"]=sub["rf_prob"].fillna(0.0)
    CPA = ["rf_prob"] + cpa_cols
    sets={"rfprob_only":["rf_prob"],
          "CPA(raw+rfprob)":CPA,
          "splice_only":SPL,
          "merged(CPA+splice)":CPA+SPL}
    print(f"\n{'model':<14}{'contested_P@1':>14}{'cont_random':>12}{'all_P@1':>9}{'noncon_P@1':>12}")
    res={}
    for name,cols in sets.items():
        R=eval_set(cols,trm,tem); res[name]=R
        con=R[R.con]; non=R[~R.con]
        print(f"{name:<14}{con.hit.mean():>14.3f}{con['rand'].mean():>12.3f}"
              f"{R.hit.mean():>9.3f}{non.hit.mean():>12.3f}", flush=True)
    cm=res["merged(CPA+splice)"][res["merged(CPA+splice)"].con].hit.mean()
    cs=res["splice_only"][res["splice_only"].con].hit.mean()
    cc=res["CPA(raw+rfprob)"][res["CPA(raw+rfprob)"].con].hit.mean()
    cr=res["rfprob_only"]
    print(f"\n[baseline check] rfprob_only NON-contested P@1 = {cr[~cr.con].hit.mean():.3f} "
          f"(should be ~1.0 if the real RF baseline is intact)")
    print(f"[interaction] merged contested P@1 = {cm:.3f}  vs  "
          f"strong-CPA baseline {cc:.3f}, splice_only {cs:.3f}")
    print("  merged > BOTH (esp > CPA-with-rfprob) => real CPA x splice interaction beyond")
    print("    the RF summary; worth a genome-wide run.")
    print("  merged ~= CPA-with-rfprob => the interaction was a weak-baseline artifact; stop.")
    print("  watch noncon_P@1 in merged: inflation = terminal-exon annotation circularity.")

if __name__=="__main__":
    main()