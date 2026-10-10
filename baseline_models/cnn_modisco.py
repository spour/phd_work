#!/usr/bin/env python3
# extract what the 2-layer cnn learned as usable motifs: 1
import argparse, sys, math
import numpy as np, pandas as pd
BASES=np.array(list("ACGT"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",required=True);  ap.add_argument("--test_meta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000)
    ap.add_argument("--nfilt",type=int,default=48); ap.add_argument("--filt_len",type=int,default=19)
    ap.add_argument("--dropout",type=float,default=0.5); ap.add_argument("--wd",type=float,default=3e-4)
    ap.add_argument("--depth",type=int,default=2); ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--ig_steps",type=int,default=32,help="integrated-gradients steps")
    ap.add_argument("--seqlet_len",type=int,default=15); ap.add_argument("--n_motifs",type=int,default=12)
    ap.add_argument("--max_attr_peaks",type=int,default=6000,help="cap peaks for attribution speed")
    ap.add_argument("--out",default="modisco")
    a=ap.parse_args()
    import torch, torch.nn as nn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0); rng=np.random.default_rng(0)

    Xtr=np.load(a.train_npz)["X"].astype(np.float32); mtr=pd.read_parquet(a.train_meta).reset_index(drop=True)
    Xte=np.load(a.test_npz)["X"].astype(np.float32);  mte=pd.read_parquet(a.test_meta).reset_index(drop=True)
    L=Xtr.shape[1]; cleave=a.upstream_nt

    class InterpCNN(nn.Module):
        def __init__(s,nf,fl,p,depth):
            super().__init__()
            s.depth=depth
            s.conv=nn.Conv1d(4,nf,fl,padding=fl//2); s.bn=nn.BatchNorm1d(nf)
            if depth>=2:
                s.conv2=nn.Conv1d(nf,nf,11,padding=5); s.bn2=nn.BatchNorm1d(nf); s.drop2=nn.Dropout(p)
            s.att=nn.Conv1d(nf,nf,1); s.drop=nn.Dropout(p); s.readout=nn.Linear(nf,1)
        def forward(s,x):
            z=s.bn(s.conv(x)); m=torch.exp(z.clamp(max=8))
            if s.depth>=2:
                h=torch.relu(s.bn2(s.conv2(m))); h=s.drop2(h)
                alpha=torch.softmax(s.att(h),dim=2); pooled=(h*alpha).sum(2)
            else:
                alpha=torch.softmax(s.att(z),dim=2); pooled=(m*alpha).sum(2)
            return s.readout(s.drop(pooled)).squeeze(-1)

    net=InterpCNN(a.nfilt,a.filt_len,a.dropout,a.depth).to(dev)
    y=mtr["is_canonical"].astype(int).to_numpy()
    pw=torch.tensor([(1-y.mean())/max(y.mean(),1e-6)],dtype=torch.float32,device=dev)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=a.wd)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(Xtr).permute(0,2,1); yt=torch.tensor(y,dtype=torch.float32); n=len(Xt)
    for ep in range(a.epochs):
        net.train(); perm=rng.permutation(n)
        for i in range(0,n,a.batch):
            idx=perm[i:i+a.batch]; opt.zero_grad()
            lossf(net(Xt[idx].to(dev)),yt[idx].to(dev)).backward(); opt.step()
    torch.save(net.state_dict(), f"{a.out}_model.pt")
    print(f"[train] done, saved {a.out}_model.pt",file=sys.stderr); net.eval()

    isc=mte["is_canonical"].to_numpy().astype(bool)
    Xc=Xte[isc]
    if len(Xc)>a.max_attr_peaks:
        sel=rng.choice(len(Xc),a.max_attr_peaks,replace=False); Xc=Xc[sel]
    print(f"[attr] integrated gradients on {len(Xc):,} canonical peaks",file=sys.stderr)
    Xc_t=torch.tensor(Xc).permute(0,2,1)
    baseline=torch.zeros_like(Xc_t[:1])
    attrs=np.zeros((len(Xc),L),dtype=np.float32)
    bs=128
    for i in range(0,len(Xc_t),bs):
        xb=Xc_t[i:i+bs].to(dev)
        ig=torch.zeros_like(xb)
        for s_ in range(1,a.ig_steps+1):
            alpha=s_/a.ig_steps
            xi=(baseline.to(dev)+alpha*(xb-baseline.to(dev))).clone().requires_grad_(True)
            out=net(xi).sum(); net.zero_grad(); out.backward()
            ig=ig+xi.grad.detach()
        ig=ig/a.ig_steps*(xb-baseline.to(dev))
        a_pos=ig.sum(1).cpu().numpy()
        attrs[i:i+xb.shape[0]]=a_pos

    # extract seqlets: per peak, top-|attribution| windows
    sl=a.seqlet_len; half=sl//2
    seqlets=[]; seqlet_pos=[]
    for k in range(len(Xc)):
        ap_=np.abs(attrs[k])
        c=int(np.argmax(np.convolve(ap_,np.ones(sl),mode="same")))
        p0=c-half
        if p0<0 or p0+sl>L: continue
        sub=Xc[k,p0:p0+sl]
        if sub.sum()<sl*0.8: continue
        seqlets.append(sub); seqlet_pos.append(c-cleave)
    seqlets=np.array(seqlets); seqlet_pos=np.array(seqlet_pos)
    print(f"[seqlets] {len(seqlets):,} extracted",file=sys.stderr)

    # cluster seqlets into motifs
    used_modisco=False
    try:
        import modiscolite
        used_modisco=True
        print("[modisco] modisco-lite found; (manual clustering still run as fallback)",file=sys.stderr)
    except Exception:
        pass

    flat=seqlets.reshape(len(seqlets),-1).astype(np.float32)
    K=min(a.n_motifs,max(2,len(seqlets)//200))
    ci=rng.choice(len(flat),K,replace=False); centers=flat[ci].copy()
    for _ in range(25):
        sim=flat@centers.T
        lab=sim.argmax(1)
        for c in range(K):
            mem=flat[lab==c]
            if len(mem)>0: centers[c]=mem.mean(0)
    rows=[]; 
    with open(f"{a.out}_motifs.meme","w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\n")
        for c in range(K):
            mem_idx=np.nonzero(lab==c)[0]
            if len(mem_idx)<20: continue
            M=seqlets[mem_idx].sum(0).astype(float)
            P=M/M.sum(1,keepdims=True)
            ic=float((P*np.log2(P/0.25+1e-9)).sum())
            cons="".join(BASES[P.argmax(1)])
            medpos=int(np.median(seqlet_pos[mem_idx]))
            rows.append(dict(motif=c,n=len(mem_idx),ic=round(ic,2),consensus=cons,median_relpos=medpos))
            fh.write(f"MOTIF motif_{c} {cons}\n")
            fh.write(f"letter-probability matrix: alength= 4 w= {P.shape[0]} nsites= {len(mem_idx)}\n")
            for p in range(P.shape[0]):
                fh.write(" "+" ".join(f"{P[p,b]:.4f}" for b in range(4))+"\n")
            fh.write("\n")
    R=pd.DataFrame(rows).sort_values("n",ascending=False)
    R.to_csv(f"{a.out}_motif_summary.tsv",sep="\t",index=False)
    print(f"\n=== extracted motifs (IG seqlet clusters on canonical peaks) ===",file=sys.stderr)
    print(f"  {'motif':>5} {'n':>6} {'IC':>5} {'medpos':>7}  consensus",file=sys.stderr)
    for _,r in R.iterrows():
        print(f"  {int(r.motif):>5} {int(r.n):>6} {r.ic:>5.1f} {int(r.median_relpos):>7}  {r.consensus}",file=sys.stderr)
    print(f"\n  scan these PWMs over your shortlist peaks -> new reranker features.",file=sys.stderr)
    print(f"  match {a.out}_motifs.meme against CISBP-RNA / ATtRACT with TOMTOM for factor IDs.",file=sys.stderr)
    print(f"[out] {a.out}_motifs.meme, {a.out}_motif_summary.tsv, {a.out}_model.pt",file=sys.stderr)
    prof=pd.DataFrame({"rel_pos":np.arange(L)-cleave,"mean_abs_attr":np.abs(attrs).mean(0)})
    prof.to_csv(f"{a.out}_attr_profile.tsv",sep="\t",index=False)

if __name__=="__main__": main()