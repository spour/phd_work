#!/usr/bin/env python3
# cnn reranker: same task and same window as the hand-crafted splice reranker
import argparse, sys, math
import numpy as np, pandas as pd

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def p1(meta, scores):
    m=meta.copy(); m["s"]=scores; h=[]
    for g,gg in m.groupby("gene",sort=False):
        top=int(np.argmax(gg["s"].to_numpy()))
        h.append(int(bool(gg["is_canonical"].to_numpy()[top])))
    return np.array(h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",required=True);  ap.add_argument("--test_meta",required=True)
    ap.add_argument("--splice_p1",type=float,default=None,help="hand-crafted reranker P@1 for comparison")
    ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--width",type=int,default=64)
    ap.add_argument("--dropout",type=float,default=0.4); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--gene_split",action="store_true",
                    help="leak test: pool train+test, split by GENE, report held-out P@1")
    ap.add_argument("--val_frac",type=float,default=0.15,help="gene fraction for early stopping")
    ap.add_argument("--seeds",type=int,default=3,help="average over N seeds")
    ap.add_argument("--out",default="cnn_rerank")
    a=ap.parse_args()
    import torch, torch.nn as nn
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev="cuda" if torch.cuda.is_available() else "cpu"

    Xtr=np.load(a.train_npz)["X"].astype(np.float32); mtr=pd.read_parquet(a.train_meta).reset_index(drop=True)
    Xte=np.load(a.test_npz)["X"].astype(np.float32);  mte=pd.read_parquet(a.test_meta).reset_index(drop=True)
    L=Xtr.shape[1]

    class CNN(nn.Module):
        def __init__(s,w,p):
            super().__init__()
            s.net=nn.Sequential(
                nn.Conv1d(4,w,15,padding=7), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(p),
                nn.Conv1d(w,w,11,padding=5), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(p),
                nn.Conv1d(w,w,7,padding=3),  nn.BatchNorm1d(w), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                nn.Linear(w,w), nn.ReLU(), nn.Dropout(p), nn.Linear(w,1))
        def forward(s,x): return s.net(x).squeeze(-1)

    def train_eval(Xtr,ytr,mtr,Xte,mte,seed):
        torch.manual_seed(seed); np.random.seed(seed)
        rng=np.random.default_rng(seed)
        genes=mtr["gene"].to_numpy()
        ug=np.array(sorted(set(genes))); rng.shuffle(ug)
        nval=max(1,int(len(ug)*a.val_frac)); vg=set(ug[:nval])
        vmask=np.array([g in vg for g in genes]); tmask=~vmask
        net=CNN(a.width,a.dropout).to(dev)
        pw=torch.tensor([(1-ytr[tmask].mean())/max(ytr[tmask].mean(),1e-6)],dtype=torch.float32,device=dev)
        opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=1e-4)
        lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt=torch.tensor(Xtr).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
        tr_idx=np.nonzero(tmask)[0]
        def score(X):
            net.eval(); out=[]
            with torch.no_grad():
                for i in range(0,len(X),1024): out.append(torch.sigmoid(net(X[i:i+1024].to(dev))).cpu().numpy())
            return np.concatenate(out)
        best_vp=-1; best_test=None; best_train=None; patience=0
        for ep in range(a.epochs):
            net.train(); perm=rng.permutation(tr_idx)
            for i in range(0,len(perm),a.batch):
                idx=perm[i:i+a.batch]
                xb=Xt[idx].to(dev); yb=yt[idx].to(dev)
                opt.zero_grad(); loss=lossf(net(xb),yb); loss.backward(); opt.step()
            vsc=score(Xt[vmask]); vp1=p1(mtr[vmask],vsc).mean()
            if vp1>best_vp:
                best_vp=vp1; patience=0
                best_test=score(torch.tensor(Xte).permute(0,2,1))
                best_train=score(Xt[tmask])
                best_train_meta=mtr[tmask]
            else:
                patience+=1
                if patience>=6: break
        return best_test, best_train, best_train_meta, best_vp

    ytr=mtr["is_canonical"].astype(int).to_numpy()
    print(f"[data] train {Xtr.shape}, test {Xte.shape}; pos rate train={ytr.mean():.3f}",file=sys.stderr)

    if a.gene_split:
        Xall=np.concatenate([Xtr,Xte],0)
        mall=pd.concat([mtr,mte],ignore_index=True)
        yall=mall["is_canonical"].astype(int).to_numpy()
        ug=np.array(sorted(mall["gene"].unique())); np.random.default_rng(0).shuffle(ug)
        k=5; folds=np.array_split(ug,k); hits=[]
        for fi in range(k):
            te_g=set(folds[fi]); tr_g=set(np.concatenate([folds[j] for j in range(k) if j!=fi]))
            trm=mall["gene"].isin(tr_g).to_numpy(); tem=mall["gene"].isin(te_g).to_numpy()
            bt,_,_,_=train_eval(Xall[trm],yall[trm],mall[trm].reset_index(drop=True),
                                Xall[tem],mall[tem].reset_index(drop=True),seed=fi)
            hits.append(p1(mall[tem].reset_index(drop=True),bt))
        H=np.concatenate(hits); p,lo,hi=wilson(int(H.sum()),len(H))
        print(f"\n=== CNN GENE-SPLIT (leak test, 5-fold by gene, n={len(H)}) ===",file=sys.stderr)
        print(f"  held-out P@1 = {p:.3f} [{lo:.3f},{hi:.3f}]",file=sys.stderr)
        if a.splice_p1 is not None:
            print(f"  hand-crafted splice = {a.splice_p1:.3f}   CNN(gene-split) - splice = {p-a.splice_p1:+.3f}",file=sys.stderr)
            print(f"  (if this gain is ~0, the chromosome-split CNN edge was GENE-IDENTITY LEAK,",file=sys.stderr)
            print(f"   not real cis signal -- same failure mode as mean-centering)",file=sys.stderr)
        sys.exit(0)

    tests=[]; trains=[]; vps=[]
    for sd in range(a.seeds):
        bt,btr,btrm,vp=train_eval(Xtr,ytr,mtr,Xte,mte,seed=sd)
        tests.append(bt); trains.append((btr,btrm)); vps.append(vp)
        print(f"  seed {sd}: best val-P@1={vp:.3f}",file=sys.stderr)
    sc=np.mean(tests,0)
    h_te=p1(mte,sc); p,lo,hi=wilson(int(h_te.sum()),len(h_te))
    h_tr=p1(trains[-1][1],trains[-1][0])
    print(f"\n=== CNN reranker (de novo sequence, same window; {a.seeds}-seed avg, early-stopped) ===",file=sys.stderr)
    print(f"  test contested P@1 = {p:.3f} [{lo:.3f},{hi:.3f}]  (n={len(h_te)} genes)",file=sys.stderr)
    print(f"  train P@1 = {h_tr.mean():.3f}   gap={h_tr.mean()-p:+.3f}",file=sys.stderr)
    if a.splice_p1 is not None:
        print(f"  hand-crafted splice reranker = {a.splice_p1:.3f}   CNN - splice = {p-a.splice_p1:+.3f}",file=sys.stderr)
        print(f"  (run --gene_split to confirm any edge is not gene-identity leak)",file=sys.stderr)
    out=mte.copy(); out["cnn_score"]=sc; out.to_parquet(f"{a.out}_test_scored.parquet",index=False)
    print(f"[out] {a.out}_test_scored.parquet",file=sys.stderr)

if __name__=="__main__": main()