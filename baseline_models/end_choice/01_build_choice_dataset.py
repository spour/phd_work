#!/usr/bin/env python3
# usage-based, distance-blind choice dataset for 3' end selection
import argparse
import numpy as np, pandas as pd

def read_bed6(p):
    df=pd.read_csv(p,sep="\t",header=None,usecols=range(6),
                   names=["chrom","start","end","name","score","strand"])
    df["chrom"]=df["chrom"].astype(str); return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--cpccpa_intragenic", required=True)
    ap.add_argument("--pc_genes", required=True)
    ap.add_argument("--rf_peak_table", default=None)
    ap.add_argument("--floor", type=float, default=0.05,
                    help="min usage for an atlas COMPETITOR to count (default 0.05)")
    ap.add_argument("--out", required=True)
    a=ap.parse_args()

    genes=read_bed6(a.pc_genes).drop_duplicates("name")
    genes["gene"]=genes["name"].astype(str).str.split("_").str[-1]
    gi={ch:d[["start","end","gene","strand"]].sort_values("start").to_numpy()
        for ch,d in genes.groupby("chrom")}
    def assign(chrom,pos):
        arr=gi.get(chrom)
        if arr is None: return None,None
        hit=arr[(arr[:,0]<=pos)&(arr[:,1]>=pos)]
        if len(hit)==0: return None,None
        h=hit[np.argmin(hit[:,1]-hit[:,0])]; return h[2],h[3]

    # atlas: read full schema, parse usage + type
    at=pd.read_csv(a.atlas,sep="\t",header=None)
    at["chrom"]=at[0].astype(str)
    at["pos"]=((at[1]+at[2])//2).astype(int)
    at["usage"]=pd.to_numeric(at[4],errors="coerce").fillna(0.0)
    at["strand"]=at[5].astype(str)
    at["site_type"]=at[9].astype(str) if at.shape[1]>9 else "NA"
    g=[]; st=[]
    for r in at.itertuples():
        gg,ss=assign(r.chrom,int(r.pos)); g.append(gg); st.append(ss)
    at["gene"]=g; at["gstrand"]=st
    at=at.dropna(subset=["gene"]).copy()
    at["strand"]=at["gstrand"].where(at["gstrand"].isin(["+","-"]),at["strand"])

    # per gene: dominant site = max usage
    rows=[]
    for gene,grp in at.groupby("gene"):
        grp=grp.sort_values("usage",ascending=False)
        dom=grp.iloc[0]
        rows.append((gene,dom.chrom,int(dom.pos),dom.strand,1,float(dom.usage),dom.site_type))
        for r in grp.iloc[1:].itertuples():
            if r.usage>=a.floor:
                rows.append((gene,r.chrom,int(r.pos),r.strand,0,float(r.usage),r.site_type))
    atlas_sites=pd.DataFrame(rows,columns=["gene","chrom","pos","strand","is_true","usage","site_type"])

    # add intragenic cpccpa competitors (cpa-competent suppressed sites)
    cp=read_bed6(a.cpccpa_intragenic); cp["pos"]=((cp["start"]+cp["end"])//2).astype(int)
    crows=[]
    for r in cp.itertuples():
        gg,ss=assign(r.chrom,int(r.pos))
        if gg is None: continue
        crows.append((gg,r.chrom,int(r.pos),ss if ss in("+","-") else r.strand,0,np.nan,"CPCCPA"))
    cpccpa=pd.DataFrame(crows,columns=["gene","chrom","pos","strand","is_true","usage","site_type"])

    df=pd.concat([atlas_sites,cpccpa],ignore_index=True)
    df=df.sort_values(["gene","pos","is_true"],ascending=[True,True,False])
    df=df.drop_duplicates(["gene","pos"],keep="first").reset_index(drop=True)

    keep=[g for g,grp in df.groupby("gene") if grp.is_true.sum()==1 and (grp.is_true==0).sum()>=1]
    df=df[df.gene.isin(set(keep))].reset_index(drop=True)

    if a.rf_peak_table:
        pt=pd.read_parquet(a.rf_peak_table); pt["chrom"]=pt["chrom"].astype(str)
        rf=np.full(len(df),np.nan)
        ptg={g:(d["position"].to_numpy(),d["rf_prob"].to_numpy()) for g,d in pt.groupby("gene")}
        for i,r in enumerate(df.itertuples()):
            t=ptg.get(r.gene)
            if t is None: continue
            P,R=t; j=int(np.argmin(np.abs(P-r.pos)))
            if abs(int(P[j])-int(r.pos))<=250: rf[i]=R[j]
        df["rf_prob"]=rf

    ng=df.gene.nunique()
    print(f"[choice] genes {ng:,} | sites {len(df):,} | mean cand/gene {len(df)/ng:.1f}")
    print(f"[truth] dominant-site type distribution:")
    print(df[df.is_true==1]["site_type"].value_counts().to_string())
    print(f"[usage] dominant median usage {df[df.is_true==1]['usage'].median():.3f}")
    if "rf_prob" in df.columns:
        print(f"[rf] sites with rf_prob: {df['rf_prob'].notna().mean():.1%}")
    df.to_parquet(a.out,index=False); print(f"[done] {a.out}")

if __name__=="__main__":
    main()