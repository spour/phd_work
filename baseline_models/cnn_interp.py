#!/usr/bin/env python3
# robust, interpretable-by-design cnn reranker
import argparse, sys, math
import numpy as np, pandas as pd
BASES=np.array(list("ACGT"))

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def p1(meta,scores):
    m=meta.copy(); m["s"]=scores; h=[]
    for g,gg in m.groupby("gene",sort=False):
        h.append(int(bool(gg["is_canonical"].to_numpy()[int(np.argmax(gg["s"].to_numpy()))])))
    return np.array(h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",required=True);  ap.add_argument("--test_meta",required=True)
    ap.add_argument("--splice_p1",type=float,default=None)
    ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--nfilt",type=int,default=48)
    ap.add_argument("--filt_len",type=int,default=19); ap.add_argument("--dropout",type=float,default=0.5)
    ap.add_argument("--wd",type=float,default=3e-4); ap.add_argument("--val_frac",type=float,default=0.15)
    ap.add_argument("--seeds",type=int,default=3); ap.add_argument("--gene_split",action="store_true")
    ap.add_argument("--depth",type=int,default=1,choices=[1,2],help="conv layers before pooling")
    ap.add_argument("--upstream_nt",type=int,default=1000)
    ap.add_argument("--out",default="cnn_interp")
    a=ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    dev="cuda" if torch.cuda.is_available() else "cpu"

    Xtr=np.load(a.train_npz)["X"].astype(np.float32); mtr=pd.read_parquet(a.train_meta).reset_index(drop=True)
    Xte=np.load(a.test_npz)["X"].astype(np.float32);  mte=pd.read_parquet(a.test_meta).reset_index(drop=True)
    L=Xtr.shape[1]

    class InterpCNN(nn.Module):
        def __init__(s,nf,fl,p,depth=1):
            super().__init__()
            s.depth=depth
            s.conv=nn.Conv1d(4,nf,fl,padding=fl//2)
            s.bn=nn.BatchNorm1d(nf)
            if depth>=2:
                s.conv2=nn.Conv1d(nf,nf,11,padding=5)
                s.bn2=nn.BatchNorm1d(nf)
                s.drop2=nn.Dropout(p)
            s.att=nn.Conv1d(nf,nf,1)
            s.drop=nn.Dropout(p)
            s.readout=nn.Linear(nf,1)
        def features(s,x):
            z=s.bn(s.conv(x))
            m=torch.exp(z.clamp(max=8))
            if s.depth>=2:
                h=torch.relu(s.bn2(s.conv2(m)))
                h=s.drop2(h)
                alpha=torch.softmax(s.att(h),dim=2)
                pooled=(h*alpha).sum(2)
            else:
                alpha=torch.softmax(s.att(z),dim=2)
                pooled=(m*alpha).sum(2)
            return pooled, alpha
        def forward(s,x):
            pooled,_=s.features(x)
            return s.readout(s.drop(pooled)).squeeze(-1)

    def train_eval(Xtr,ytr,mtr,Xte,mte,seed):
        torch.manual_seed(seed); np.random.seed(seed); rng=np.random.default_rng(seed)
        genes=mtr["gene"].to_numpy(); ug=np.array(sorted(set(genes))); rng.shuffle(ug)
        nval=max(1,int(len(ug)*a.val_frac)); vg=set(ug[:nval])
        vmask=np.array([g in vg for g in genes]); tmask=~vmask
        net=InterpCNN(a.nfilt,a.filt_len,a.dropout,depth=a.depth).to(dev)
        pw=torch.tensor([(1-ytr[tmask].mean())/max(ytr[tmask].mean(),1e-6)],dtype=torch.float32,device=dev)
        opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=a.wd)
        lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt=torch.tensor(Xtr).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
        tr_idx=np.nonzero(tmask)[0]
        def score(X):
            net.eval(); out=[]
            with torch.no_grad():
                for i in range(0,len(X),1024): out.append(torch.sigmoid(net(X[i:i+1024].to(dev))).cpu().numpy())
            return np.concatenate(out)
        best_vp=-1; best_test=None; best_net=None; patience=0
        for ep in range(a.epochs):
            net.train(); perm=rng.permutation(tr_idx)
            for i in range(0,len(perm),a.batch):
                idx=perm[i:i+a.batch]; opt.zero_grad()
                lossf(net(Xt[idx].to(dev)),yt[idx].to(dev)).backward(); opt.step()
            vp=p1(mtr[vmask],score(Xt[vmask])).mean()
            if vp>best_vp:
                best_vp=vp; patience=0; best_test=score(torch.tensor(Xte).permute(0,2,1))
                best_net={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            else:
                patience+=1
                if patience>=6: break
        return best_test, best_net, best_vp

    ytr=mtr["is_canonical"].astype(int).to_numpy()
    print(f"[data] train {Xtr.shape}, test {Xte.shape}; interpretable exp-CNN, "
          f"nfilt={a.nfilt} fl={a.filt_len} depth={a.depth}",file=sys.stderr)

    if a.gene_split:
        Xall=np.concatenate([Xtr,Xte],0); mall=pd.concat([mtr,mte],ignore_index=True)
        yall=mall["is_canonical"].astype(int).to_numpy()
        ug=np.array(sorted(mall["gene"].unique())); np.random.default_rng(0).shuffle(ug)
        k=5; folds=np.array_split(ug,k); hits=[]
        for fi in range(k):
            te_g=set(folds[fi]); tr_g=set(np.concatenate([folds[j] for j in range(k) if j!=fi]))
            trm=mall["gene"].isin(tr_g).to_numpy(); tem=mall["gene"].isin(te_g).to_numpy()
            bt,_,_=train_eval(Xall[trm],yall[trm],mall[trm].reset_index(drop=True),
                              Xall[tem],mall[tem].reset_index(drop=True),seed=fi)
            hits.append(p1(mall[tem].reset_index(drop=True),bt))
        H=np.concatenate(hits); p,lo,hi=wilson(int(H.sum()),len(H))
        print(f"\n=== interpretable CNN GENE-SPLIT (5-fold, n={len(H)}) ===",file=sys.stderr)
        print(f"  held-out P@1 = {p:.3f} [{lo:.3f},{hi:.3f}]",file=sys.stderr)
        if a.splice_p1: print(f"  splice={a.splice_p1:.3f}  interp-CNN - splice = {p-a.splice_p1:+.3f}",file=sys.stderr)
        sys.exit(0)

    tests=[]; nets=[]
    for sd in range(a.seeds):
        bt,bn,vp=train_eval(Xtr,ytr,mtr,Xte,mte,seed=sd); tests.append(bt); nets.append(bn)
        print(f"  seed {sd}: val-P@1={vp:.3f}",file=sys.stderr)
    sc=np.mean(tests,0); h=p1(mte,sc); p,lo,hi=wilson(int(h.sum()),len(h))
    print(f"\n=== interpretable CNN ({a.seeds}-seed avg, early-stopped) ===",file=sys.stderr)
    print(f"  test P@1 = {p:.3f} [{lo:.3f},{hi:.3f}] (n={len(h)} genes)",file=sys.stderr)
    if a.splice_p1: print(f"  splice={a.splice_p1:.3f}  interp-CNN - splice = {p-a.splice_p1:+.3f}",file=sys.stderr)

    # interpret: reload last seed, read motif weights + attention positions
    net=InterpCNN(a.nfilt,a.filt_len,a.dropout,depth=a.depth).to(dev); net.load_state_dict(nets[-1]); net.eval()
    w=net.readout.weight.detach().cpu().numpy().ravel()
    convw=net.conv.weight.detach().cpu().numpy()
    Xc=torch.tensor(Xte[mte.is_canonical.to_numpy().astype(bool)][:4000]).permute(0,2,1)
    cleave=a.upstream_nt
    with torch.no_grad():
        _,alpha=net.features(Xc.to(dev)); alpha=alpha.cpu().numpy()
    pos=np.arange(L)-cleave
    medpos=[]
    for f in range(a.nfilt):
        wsum=alpha[:,f,:].mean(0); medpos.append(int(pos[np.argmax(wsum)]))
    rows=[]
    for f in range(a.nfilt):
        pwm=convw[f]; cons="".join(BASES[pwm.argmax(0)])
        rows.append(dict(filter=f,readout_weight=float(w[f]),consensus=cons,att_peak_relpos=medpos[f]))
    D=pd.DataFrame(rows).reindex(np.argsort(-np.abs(w)))
    D.to_csv(f"{a.out}_motif_weights.tsv",sep="\t",index=False)
    print(f"\n=== top motifs by |readout weight| (signed contribution to true-end score) ===",file=sys.stderr)
    print(f"  {'filt':>4} {'weight':>8} {'att_pos':>8}  consensus",file=sys.stderr)
    for _,r in D.head(15).iterrows():
        fi=int(r["filter"])
        print(f"  {fi:>4} {r.readout_weight:>+8.3f} {int(r.att_peak_relpos):>8}  {r.consensus}",file=sys.stderr)
    print(f"\n  weight>0 => motif pushes toward TRUE END; att_pos = where it attends (rel cleavage)",file=sys.stderr)
    print(f"[out] {a.out}_motif_weights.tsv",file=sys.stderr)

if __name__=="__main__": main()