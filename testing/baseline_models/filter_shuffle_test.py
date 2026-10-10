#!/usr/bin/env python3
# for each top discriminative cnn filter
import argparse, sys
import numpy as np, pandas as pd
BASES=np.array(list("ACGT"))

def dinuc_shuffle(seq_idx, rng):
    n=len(seq_idx)
    if n<3: return seq_idx.copy()
    edges={b:[] for b in range(4)}
    for i in range(n-1): edges[seq_idx[i]].append(seq_idx[i+1])
    last=seq_idx[-1]
    for b in edges: rng.shuffle(edges[b])
    for _attempt in range(20):
        e={b:list(edges[b]) for b in edges}
        for b in e: rng.shuffle(e[b])
        out=[seq_idx[0]]; cur=seq_idx[0]; ok=True
        for _ in range(n-1):
            if not e[cur]: ok=False; break
            nxt=e[cur].pop(); out.append(nxt); cur=nxt
        if ok and len(out)==n: return np.array(out)
    return seq_idx.copy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",default=None); ap.add_argument("--test_meta",default=None)
    ap.add_argument("--filters",required=True,help="comma list of filter indices to test")
    ap.add_argument("--n_seqs",type=int,default=2000,help="top-firing seqs per filter")
    ap.add_argument("--n_shuffles",type=int,default=5)
    ap.add_argument("--width",type=int,default=64); ap.add_argument("--dropout",type=float,default=0.4)
    ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--filt_len",type=int,default=15)
    ap.add_argument("--out",default="filter_shuffle")
    a=ap.parse_args()
    import torch, torch.nn as nn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0); rng=np.random.default_rng(0)

    X=np.load(a.train_npz)["X"].astype(np.float32); m=pd.read_parquet(a.train_meta).reset_index(drop=True)
    if a.test_npz:
        X=np.concatenate([X,np.load(a.test_npz)["X"].astype(np.float32)],0)
        m=pd.concat([m,pd.read_parquet(a.test_meta)],ignore_index=True)
    L=X.shape[1]; W=a.width; fl=a.filt_len; half=fl//2

    class CNN(nn.Module):
        def __init__(s,w,p,flen):
            super().__init__()
            s.conv1=nn.Conv1d(4,w,flen,padding=flen//2); s.bn1=nn.BatchNorm1d(w)
            s.rest=nn.Sequential(nn.ReLU(),nn.MaxPool1d(4),nn.Dropout(p),
                nn.Conv1d(w,w,11,padding=5),nn.BatchNorm1d(w),nn.ReLU(),nn.MaxPool1d(4),nn.Dropout(p),
                nn.Conv1d(w,w,7,padding=3),nn.BatchNorm1d(w),nn.ReLU(),nn.AdaptiveMaxPool1d(1),nn.Flatten(),
                nn.Linear(w,w),nn.ReLU(),nn.Dropout(p),nn.Linear(w,1))
        def first(s,x): return s.bn1(s.conv1(x))
        def forward(s,x): return s.rest(torch.relu(s.first(x))).squeeze(-1)

    net=CNN(W,a.dropout,fl).to(dev)
    y=m["is_canonical"].astype(int).to_numpy()
    pw=torch.tensor([(1-y.mean())/max(y.mean(),1e-6)],dtype=torch.float32,device=dev)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(X).permute(0,2,1); yt=torch.tensor(y,dtype=torch.float32)
    n=len(Xt)
    for ep in range(a.epochs):
        net.train(); perm=rng.permutation(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; opt.zero_grad()
            lossf(net(Xt[idx].to(dev)),yt[idx].to(dev)).backward(); opt.step()
    print(f"[train] done",file=sys.stderr); net.eval()

    Wc=net.conv1.weight.detach().cpu().numpy()
    bc=net.conv1.bias.detach().cpu().numpy()
    filters=[int(x) for x in a.filters.split(",")]

    Xseq=X.argmax(2)
    with torch.no_grad():
        act_max=np.full((n,len(filters)),-1e9); act_pos=np.zeros((n,len(filters)),int)
        for i in range(0,n,512):
            xb=torch.tensor(X[i:i+512]).permute(0,2,1).to(dev)
            a1=net.conv1(xb).cpu().numpy()
            for fi,f in enumerate(filters):
                am=a1[:,f,:]; act_max[i:i+am.shape[0],fi]=am.max(1); act_pos[i:i+am.shape[0],fi]=am.argmax(1)

    def conv_score(subseq_idx, f):
        oh=np.zeros((4,fl),np.float32)
        for p,b in enumerate(subseq_idx): oh[b,p]=1.0
        return float((Wc[f]*oh).sum()+bc[f])

    rows=[]
    for fi,f in enumerate(filters):
        order=np.argsort(act_max[:,fi])[::-1][:a.n_seqs]
        real=[]; shuf=[]
        for si in order:
            p0=act_pos[si,fi]-half
            if p0<0 or p0+fl>L: continue
            sub=Xseq[si,p0:p0+fl]
            real.append(conv_score(sub,f))
            for _ in range(a.n_shuffles):
                ss=dinuc_shuffle(sub,rng); shuf.append(conv_score(ss,f))
        real=np.array(real); shuf=np.array(shuf)
        if len(real)<10: continue
        d=(real.mean()-shuf.mean())/(np.sqrt((real.var()+shuf.var())/2)+1e-9)
        retained=(real.mean()-shuf.mean())/ (abs(real.mean())+1e-9)
        rows.append(dict(filter=f,real_mean=real.mean(),shuf_mean=shuf.mean(),
                         cohens_d=d,frac_retained=retained,n=len(real)))
    R=pd.DataFrame(rows); R.to_csv(f"{a.out}.tsv",sep="\t",index=False)
    print(f"\n=== filter: real vs dinucleotide-shuffled activation ===",file=sys.stderr)
    print(f"  {'filt':>4} {'real':>8} {'shuf':>8} {'cohens_d':>9} {'retained':>9}",file=sys.stderr)
    for _,r in R.iterrows():
        fi=int(r["filter"])
        print(f"  {fi:>4} {r.real_mean:>8.2f} {r.shuf_mean:>8.2f} {r.cohens_d:>+9.2f} {r.frac_retained:>+9.2f}",file=sys.stderr)
    print(f"\n  large cohens_d / high retained => filter reads MOTIF ARRANGEMENT (real >> shuffled)",file=sys.stderr)
    print(f"  near 0 => filter reads COMPOSITION (dinuc-shuffle preserves it) => not worth chasing RBPs",file=sys.stderr)
    print(f"[out] {a.out}.tsv",file=sys.stderr)

if __name__=="__main__": main()