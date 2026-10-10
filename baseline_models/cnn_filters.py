#!/usr/bin/env python3
# read the cnn's first-layer conv filters as motif detectors (basset/deepbind-style)
import argparse, sys
import numpy as np, pandas as pd
BASES=np.array(list("ACGT"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",default=None); ap.add_argument("--test_meta",default=None)
    ap.add_argument("--width",type=int,default=64); ap.add_argument("--dropout",type=float,default=0.4)
    ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--filt_len",type=int,default=15)
    ap.add_argument("--upstream_nt",type=int,default=1000,help="cleavage-site index in the window")
    ap.add_argument("--top_act",type=int,default=500,help="top activating sites per filter for the PWM")
    ap.add_argument("--out",default="cnn_filters")
    a=ap.parse_args()
    import torch, torch.nn as nn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    X=np.load(a.train_npz)["X"].astype(np.float32); m=pd.read_parquet(a.train_meta).reset_index(drop=True)
    if a.test_npz:
        X=np.concatenate([X,np.load(a.test_npz)["X"].astype(np.float32)],0)
        m=pd.concat([m,pd.read_parquet(a.test_meta)],ignore_index=True)
    L=X.shape[1]; W=a.width

    class CNN(nn.Module):
        def __init__(s,w,p,fl):
            super().__init__()
            s.conv1=nn.Conv1d(4,w,fl,padding=fl//2)
            s.bn1=nn.BatchNorm1d(w)
            s.rest=nn.Sequential(
                nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(p),
                nn.Conv1d(w,w,11,padding=5), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(p),
                nn.Conv1d(w,w,7,padding=3),  nn.BatchNorm1d(w), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                nn.Linear(w,w), nn.ReLU(), nn.Dropout(p), nn.Linear(w,1))
        def first(s,x): return s.bn1(s.conv1(x))
        def forward(s,x): return s.rest(torch.relu(s.first(x))).squeeze(-1)

    net=CNN(W,a.dropout,a.filt_len).to(dev)
    y=m["is_canonical"].astype(int).to_numpy()
    pw=torch.tensor([(1-y.mean())/max(y.mean(),1e-6)],dtype=torch.float32,device=dev)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(X).permute(0,2,1); yt=torch.tensor(y,dtype=torch.float32)
    n=len(Xt); rng=np.random.default_rng(0)
    for ep in range(a.epochs):
        net.train(); perm=rng.permutation(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; opt.zero_grad()
            lossf(net(Xt[idx].to(dev)),yt[idx].to(dev)).backward(); opt.step()
    print(f"[train] done",file=sys.stderr); net.eval()

    # pass all seqs through first conv, collect per-filter activations
    fl=a.filt_len; half=fl//2
    nfilt=W
    pwm_counts=np.zeros((nfilt,fl,4))
    top_vals=[ -np.ones(a.top_act)*1e9 for _ in range(nfilt) ]
    top_seqs=[ [None]*a.top_act for _ in range(nfilt) ]
    peak_maxact=np.zeros((n,nfilt),dtype=np.float32)
    peak_argpos=np.zeros((n,nfilt),dtype=np.int32)
    with torch.no_grad():
        for i in range(0,n,512):
            xb=Xt[i:i+512].to(dev)
            act=net.first(xb).cpu().numpy()
            peak_maxact[i:i+act.shape[0]]=act.max(2)
            peak_argpos[i:i+act.shape[0]]=act.argmax(2)
            xnp=X[i:i+act.shape[0]]
            for f in range(nfilt):
                pos=act[:,f,:].argmax(1)
                vals=act[np.arange(act.shape[0]),f,pos]
                for b in range(act.shape[0]):
                    v=vals[b]
                    if v>top_vals[f].min():
                        j=int(top_vals[f].argmin())
                        p0=pos[b]-half
                        if 0<=p0 and p0+fl<=L:
                            top_vals[f][j]=v; top_seqs[f][j]=xnp[b,p0:p0+fl].argmax(1)
    info=[]
    with open(f"{a.out}_motifs.txt","w") as fh:
        for f in range(nfilt):
            S=[s for s in top_seqs[f] if s is not None]
            if len(S)<20: 
                info.append(dict(filter=f,ic=0.0,consensus="(weak)")); continue
            M=np.zeros((fl,4))
            for s in S:
                for p,b in enumerate(s): M[p,b]+=1
            P=M/M.sum(1,keepdims=True)
            ic=float((P*np.log2(P/0.25+1e-9)).sum())
            cons="".join(BASES[P.argmax(1)])
            info.append(dict(filter=f,ic=ic,consensus=cons))
            fh.write(f">filter_{f}  IC={ic:.2f}  consensus={cons}\n")
            for p in range(fl): fh.write("  "+" ".join(f"{P[p,k]:.2f}" for k in range(4))+"\n")

    # discrimination: mean max-activation canonical vs competitor
    isc=m["is_canonical"].to_numpy().astype(bool)
    rows=[]
    for f in range(nfilt):
        ac=peak_maxact[isc,f]; co=peak_maxact[~isc,f]
        d=ac.mean()-co.mean(); pooled=np.sqrt((ac.var()+co.var())/2)+1e-9
        rows.append(dict(filter=f,canon_act=ac.mean(),comp_act=co.mean(),
                         diff=d,cohens_d=d/pooled,
                         ic=info[f]["ic"],consensus=info[f]["consensus"]))
    D=pd.DataFrame(rows).sort_values("cohens_d",key=abs,ascending=False)
    D.to_csv(f"{a.out}_discrimination.tsv",sep="\t",index=False)

    print(f"\n=== top discriminative filters (canonical vs competitor activation) ===",file=sys.stderr)
    print(f"  {'filt':>4} {'cohens_d':>9} {'IC':>5}  consensus",file=sys.stderr)
    for _,r in D.head(15).iterrows():
        fi=int(r["filter"])
        print(f"  {fi:>4} {r.cohens_d:>+9.3f} {r.ic:>5.1f}  {r.consensus}",file=sys.stderr)
    print(f"\n  (positive cohens_d = motif STRONGER at canonical = marks true end;",file=sys.stderr)
    print(f"   match consensus vs AATAAA/ATTAAA [PAS], GT..AG [splice], TGTA [CFIm],",file=sys.stderr)
    print(f"   GU/U-rich [DSE]. PAS-like discriminative filters => finer PAS-quality",file=sys.stderr)
    print(f"   discrimination; splice-like => splice context.)",file=sys.stderr)
    # positional firing profile of the top discriminative filters
    cleave=a.upstream_nt
    isc=m["is_canonical"].to_numpy().astype(bool)
    topf=D.head(15)["filter"].astype(int).tolist()
    prof_rows=[]
    for f in topf:
        relpos=peak_argpos[isc,f].astype(int)-cleave
        med=int(np.median(relpos)); 
        frac_dse=float(((relpos>=13)&(relpos<=30)).mean())
        frac_up=float((relpos<-40).mean()); frac_core=float((np.abs(relpos)<=40).mean())
        frac_dn=float((relpos>40).mean())
        prof_rows.append(dict(filter=f,consensus=D.set_index("filter").loc[f,"consensus"],
            cohens_d=float(D.set_index("filter").loc[f,"cohens_d"]),
            median_relpos=med,frac_in_DSE_13_30=frac_dse,
            frac_upstream=frac_up,frac_core_pm40=frac_core,frac_downstream=frac_dn))
    P=pd.DataFrame(prof_rows); P.to_csv(f"{a.out}_filter_positions.tsv",sep="\t",index=False)
    print(f"\n=== where the top discriminative filters fire (relative to cleavage site) ===",file=sys.stderr)
    print(f"  {'filt':>4} {'cons':>16} {'cohd':>6} {'medpos':>7} {'DSE+13-30':>10} {'core':>6} {'down':>6}",file=sys.stderr)
    for _,r in P.iterrows():
        fi=int(r["filter"])
        print(f"  {fi:>4} {r.consensus:>16} {r.cohens_d:>+6.2f} {r.median_relpos:>7d} "
              f"{r.frac_in_DSE_13_30:>10.2f} {r.frac_core_pm40:>6.2f} {r.frac_downstream:>6.2f}",file=sys.stderr)
    print(f"\n  frac in DSE+13-30 high (and survives dinuc-shuffle) => Pol II elongation-coupling",file=sys.stderr)
    print(f"  GC element (Geisberg/co-transcriptional). Scattered/upstream => not that.",file=sys.stderr)
    print(f"\n[out] {a.out}_motifs.txt, {a.out}_discrimination.tsv, {a.out}_filter_positions.tsv",file=sys.stderr)

if __name__=="__main__": main()