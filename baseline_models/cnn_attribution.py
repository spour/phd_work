#!/usr/bin/env python3
# what did the cnn reranker learn that the hand-crafted splice features miss?
import argparse, sys
import numpy as np, pandas as pd

BASES=np.array(list("ACGT"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",required=True);  ap.add_argument("--test_meta",required=True)
    ap.add_argument("--splice_scored",required=True,help="rerank_test_scored.parquet (splice picks)")
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--width",type=int,default=64); ap.add_argument("--dropout",type=float,default=0.4)
    ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--n_examples",type=int,default=8,help="peaks for ISM motif zoom")
    ap.add_argument("--out",default="cnn_attr")
    a=ap.parse_args()
    import torch, torch.nn as nn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    Xtr=np.load(a.train_npz)["X"].astype(np.float32); mtr=pd.read_parquet(a.train_meta).reset_index(drop=True)
    Xte=np.load(a.test_npz)["X"].astype(np.float32);  mte=pd.read_parquet(a.test_meta).reset_index(drop=True)
    L=Xtr.shape[1]; cleave=a.upstream_nt

    class CNN(nn.Module):
        def __init__(s,w,p):
            super().__init__()
            s.net=nn.Sequential(
                nn.Conv1d(4,w,15,padding=7), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(p),
                nn.Conv1d(w,w,11,padding=5), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(p),
                nn.Conv1d(w,w,7,padding=3),  nn.BatchNorm1d(w), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                nn.Linear(w,w), nn.ReLU(), nn.Dropout(p), nn.Linear(w,1))
        def forward(s,x): return s.net(x).squeeze(-1)

    net=CNN(a.width,a.dropout).to(dev)
    ytr=mtr["is_canonical"].astype(int).to_numpy()
    pw=torch.tensor([(1-ytr.mean())/max(ytr.mean(),1e-6)],dtype=torch.float32,device=dev)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(Xtr).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
    n=len(Xt); rng=np.random.default_rng(0)
    for ep in range(a.epochs):
        net.train(); perm=rng.permutation(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; xb=Xt[idx].to(dev); yb=yt[idx].to(dev)
            opt.zero_grad(); lossf(net(xb),yb).backward(); opt.step()
    print(f"[train] done ({a.epochs} ep)",file=sys.stderr)

    net.eval()
    def score(X):
        out=[]
        with torch.no_grad():
            for i in range(0,len(X),1024): out.append(torch.sigmoid(net(X[i:i+1024].to(dev))).cpu().numpy())
        return np.concatenate(out)
    Xv=torch.tensor(Xte).permute(0,2,1); cnn_sc=score(Xv)
    mte=mte.copy(); mte["cnn"]=cnn_sc
    sp=pd.read_parquet(a.splice_scored)[["gene","center","splice_score"]]
    mte=mte.merge(sp,on=["gene","center"],how="left")
    def argmax_is_canon(df,col):
        i=int(np.argmax(df[col].to_numpy())); return bool(df["is_canonical"].to_numpy()[i])
    disc=[]
    for g,gg in mte.groupby("gene",sort=False):
        if gg["splice_score"].isna().any(): continue
        cnn_ok=argmax_is_canon(gg,"cnn"); spl_ok=argmax_is_canon(gg,"splice_score")
        if cnn_ok and not spl_ok: disc.append(g)
    print(f"[disc] {len(disc)} genes CNN-correct & splice-wrong",file=sys.stderr)
    sub=mte[mte.gene.isin(set(disc))].reset_index(drop=True)
    canon=sub[sub.is_canonical].reset_index(drop=True)
    ridx=[mte.index[(mte.gene==r.gene)&(mte.center==r.center)][0] for _,r in canon.iterrows()]
    Xc=Xte[ridx]

    Xcg=torch.tensor(Xc).permute(0,2,1).to(dev).requires_grad_(True)
    net.zero_grad(); out=net(Xcg).sum(); out.backward()
    gxi=(Xcg.grad*Xcg).sum(1).detach().cpu().numpy()
    pos_imp=np.abs(gxi).mean(0)

    rel=np.arange(L)-cleave
    prof=pd.DataFrame({"rel_pos":rel,"importance":pos_imp})
    prof.to_csv(f"{a.out}_position_profile.tsv",sep="\t",index=False)
    def region_mean(lo,hi): 
        m=(rel>=lo)&(rel<hi); return float(pos_imp[m].mean()) if m.any() else np.nan
    pas=region_mean(-40,40)
    up_splice=region_mean(-1000,-100)
    dn_splice=region_mean(60,500)
    print(f"\n=== positional importance (mean |grad x input|) ===",file=sys.stderr)
    print(f"  PAS/cleavage window (-40..40):   {pas:.4f}",file=sys.stderr)
    print(f"  upstream splice (-1000..-100):   {up_splice:.4f}",file=sys.stderr)
    print(f"  downstream/intronic (60..500):   {dn_splice:.4f}",file=sys.stderr)
    ratio=pas/np.nanmean([up_splice,dn_splice]) if up_splice and dn_splice else np.nan
    print(f"  PAS-to-splice importance ratio:  {ratio:.2f}",file=sys.stderr)
    print(f"  (ratio >> 1 => CNN keys on PAS grammar [RF's job]; ~1 or <1 => it keys",file=sys.stderr)
    print(f"   on splice-distance context = candidate signal the splice features miss)",file=sys.stderr)

    seqs=[]
    for k in range(min(a.n_examples,len(Xc))):
        oh=Xc[k]; hot=int(np.argmax(np.abs(gxi[k])))
        s="".join(BASES[oh[max(0,hot-6):hot+6].argmax(1)])
        seqs.append(dict(example=k,hot_rel_pos=int(rel[hot]),motif_context=s))
    pd.DataFrame(seqs).to_csv(f"{a.out}_motif_examples.tsv",sep="\t",index=False)
    print(f"\n[motif examples] hottest-position +-6nt context:",file=sys.stderr)
    for r in seqs: print(f"  rel {r['hot_rel_pos']:>6}: {r['motif_context']}",file=sys.stderr)
    print(f"\n[out] {a.out}_position_profile.tsv, {a.out}_motif_examples.tsv",file=sys.stderr)

if __name__=="__main__": main()