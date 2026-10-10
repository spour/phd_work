#!/usr/bin/env python3
# train a cnn on used-vs-suppressed sequence, extract motifs (ig seqlets)
import argparse, sys, math
import numpy as np, pandas as pd
BASES=np.array(list("ACGT"))
def auc_mw(s,y):
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank(method="average").to_numpy()
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",required=True); ap.add_argument("--test_meta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000)
    ap.add_argument("--nfilt",type=int,default=48); ap.add_argument("--filt_len",type=int,default=19)
    ap.add_argument("--dropout",type=float,default=0.5); ap.add_argument("--wd",type=float,default=3e-4)
    ap.add_argument("--depth",type=int,default=2); ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--ig_steps",type=int,default=24); ap.add_argument("--seqlet_len",type=int,default=15)
    ap.add_argument("--n_motifs",type=int,default=12); ap.add_argument("--max_attr",type=int,default=6000)
    ap.add_argument("--attr_class",choices=["used","suppressed","both"],default="used",
                    help="which class to attribute: used=licensing motifs, suppressed=silencer motifs")
    ap.add_argument("--out",default="supp_modisco")
    a=ap.parse_args()
    import torch, torch.nn as nn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0); rng=np.random.default_rng(0)
    Xtr=np.load(a.train_npz)["X"].astype(np.float32); mtr=pd.read_parquet(a.train_meta).reset_index(drop=True)
    Xte=np.load(a.test_npz)["X"].astype(np.float32); mte=pd.read_parquet(a.test_meta).reset_index(drop=True)
    L=Xtr.shape[1]; cleave=a.upstream_nt
    class CNN(nn.Module):
        def __init__(s,nf,fl,p,depth):
            super().__init__(); s.depth=depth
            s.conv=nn.Conv1d(4,nf,fl,padding=fl//2); s.bn=nn.BatchNorm1d(nf)
            if depth>=2: s.conv2=nn.Conv1d(nf,nf,11,padding=5); s.bn2=nn.BatchNorm1d(nf); s.drop2=nn.Dropout(p)
            s.att=nn.Conv1d(nf,nf,1); s.drop=nn.Dropout(p); s.readout=nn.Linear(nf,1)
        def forward(s,x):
            z=s.bn(s.conv(x)); m=torch.exp(z.clamp(max=8))
            if s.depth>=2:
                h=torch.relu(s.bn2(s.conv2(m))); h=s.drop2(h)
                al=torch.softmax(s.att(h),2); pooled=(h*al).sum(2)
            else:
                al=torch.softmax(s.att(z),2); pooled=(m*al).sum(2)
            return s.readout(s.drop(pooled)).squeeze(-1)
    net=CNN(a.nfilt,a.filt_len,a.dropout,a.depth).to(dev)
    y=mtr["label"].to_numpy().astype(int)
    pw=torch.tensor([(1-y.mean())/max(y.mean(),1e-6)],dtype=torch.float32,device=dev)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=a.wd)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(Xtr).permute(0,2,1); yt=torch.tensor(y,dtype=torch.float32); n=len(Xt)
    for ep in range(a.epochs):
        net.train(); perm=rng.permutation(n)
        for i in range(0,n,a.batch):
            idx=perm[i:i+a.batch]; opt.zero_grad(); lf(net(Xt[idx].to(dev)),yt[idx].to(dev)).backward(); opt.step()
    net.eval()
    def score(X):
        o=[]
        with torch.no_grad():
            for i in range(0,len(X),1024): o.append(torch.sigmoid(net(torch.tensor(X[i:i+1024]).permute(0,2,1).to(dev))).cpu().numpy())
        return np.concatenate(o)
    sc=score(Xte); yte=mte["label"].to_numpy().astype(int)
    print(f"\n=== used-vs-suppressed CNN (TEST) ===",file=sys.stderr)
    print(f"  overall AUC = {auc_mw(sc,yte):.3f}",file=sys.stderr)
    for comp in ["TE","INTERNAL"]:
        mk=(mte["compartment"]==comp).to_numpy()
        if mk.sum()>50 and len(np.unique(yte[mk]))==2:
            print(f"  AUC within {comp:8s} (n={int(mk.sum())}) = {auc_mw(sc[mk],yte[mk]):.3f} "
                  f"<- if ~0.5, overall AUC was COMPARTMENT confound",file=sys.stderr)
    # ig attribution per chosen class
    def attribute_and_cluster(target_label, tag, sign):
        sel=(mte["label"]==target_label).to_numpy(); Xc=Xte[sel]
        if len(Xc)==0: 
            print(f"[{tag}] no sites",file=sys.stderr); return
        if len(Xc)>a.max_attr: Xc=Xc[rng.choice(len(Xc),a.max_attr,replace=False)]
        Xc_t=torch.tensor(Xc).permute(0,2,1); base=torch.zeros_like(Xc_t[:1])
        attrs=np.zeros((len(Xc),L),np.float32)
        for i in range(0,len(Xc_t),128):
            xb=Xc_t[i:i+128].to(dev); ig=torch.zeros_like(xb)
            for s_ in range(1,a.ig_steps+1):
                xi=(base.to(dev)+ (s_/a.ig_steps)*(xb-base.to(dev))).clone().requires_grad_(True)
                net.zero_grad(); (sign*net(xi)).sum().backward(); ig=ig+xi.grad.detach()
            ig=ig/a.ig_steps*(xb-base.to(dev)); attrs[i:i+xb.shape[0]]=ig.sum(1).cpu().numpy()
        sl=a.seqlet_len; half=sl//2; seqlets=[]; spos=[]
        for k in range(len(Xc)):
            c=int(np.argmax(np.convolve(np.abs(attrs[k]),np.ones(sl),mode="same"))); p0=c-half
            if p0<0 or p0+sl>L: continue
            sub=Xc[k,p0:p0+sl]
            if sub.sum()>=sl*0.8: seqlets.append(sub); spos.append(c-cleave)
        seqlets=np.array(seqlets); spos=np.array(spos)
        if len(seqlets)<40: print(f"[{tag}] too few seqlets",file=sys.stderr); return
        flat=seqlets.reshape(len(seqlets),-1).astype(np.float32)
        K=min(a.n_motifs,max(2,len(seqlets)//200)); ci=rng.choice(len(flat),K,replace=False); cen=flat[ci].copy()
        for _ in range(25):
            lab=(flat@cen.T).argmax(1)
            for c in range(K):
                mm=flat[lab==c]
                if len(mm)>0: cen[c]=mm.mean(0)
        rows=[]
        with open(f"{a.out}_{tag}_motifs.meme","w") as fh:
            fh.write("MEME version 4\n\nALPHABET= ACGT\n\n")
            for c in range(K):
                idx=np.nonzero(lab==c)[0]
                if len(idx)<20: continue
                M=seqlets[idx].sum(0).astype(float); P=M/M.sum(1,keepdims=True)
                ic=float((P*np.log2(P/0.25+1e-9)).sum()); cons="".join(BASES[P.argmax(1)])
                rows.append(dict(motif=c,n=len(idx),ic=round(ic,2),consensus=cons,medpos=int(np.median(spos[idx]))))
                fh.write(f"MOTIF {tag}_motif_{c} {cons}\nletter-probability matrix: alength= 4 w= {sl} nsites= {len(idx)}\n")
                for p in range(sl): fh.write(" "+" ".join(f"{P[p,b]:.4f}" for b in range(4))+"\n")
                fh.write("\n")
        R=pd.DataFrame(rows).sort_values("n",ascending=False); R.to_csv(f"{a.out}_{tag}_motif_summary.tsv",sep="\t",index=False)
        print(f"\n=== {tag} motifs (IG seqlets) ===\n  {'motif':>5}{'n':>6}{'IC':>6}{'medpos':>8}  consensus",file=sys.stderr)
        for _,r in R.iterrows(): print(f"  {int(r.motif):>5}{int(r.n):>6}{r.ic:>6.1f}{int(r.medpos):>8}  {r.consensus}",file=sys.stderr)

    if a.attr_class in ("used","both"):
        attribute_and_cluster(1,"used",+1.0)
    if a.attr_class in ("suppressed","both"):
        attribute_and_cluster(0,"suppressed",-1.0)
    print(f"\n  IC>8 = real motif; IC~3-4 = noise/composition. TOMTOM the .meme vs CISBP-RNA.",file=sys.stderr)
    print(f"[out] {a.out}_*_motifs.meme, {a.out}_*_motif_summary.tsv",file=sys.stderr)
if __name__=="__main__": main()
