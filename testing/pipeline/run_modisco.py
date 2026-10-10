#!/usr/bin/env python3
# de-novo motif discovery (tf-modisco) for the canonical-vs-competitor sequence model
import argparse, sys
import numpy as np, pandas as pd

BASES = np.array(list("ACGT"))
_TRANS = np.full(256, 4, dtype=np.int8)
for _c, _i in {"A":0,"C":1,"G":2,"T":3,"a":0,"c":1,"g":2,"t":3}.items():
    _TRANS[ord(_c)] = _i
_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
def seq_to_idx(s): return _TRANS[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)].copy()
def revcomp(s): return s.translate(_COMP)[::-1]

def consensus_pwm(seq, p=0.85):
    idx = {"A":0,"C":1,"G":2,"T":3}; M = np.full((len(seq),4), (1-p)/3)
    for i,ch in enumerate(seq):
        if ch in idx: M[i]=(1-p)/3; M[i,idx[ch]]=p
        else: M[i]=0.25
    return M
KNOWN = {"5'ss donor":consensus_pwm("AGGTAAGT"),"3'ss acceptor":consensus_pwm("TTTTCAGG"),
         "PAS AAUAAA":consensus_pwm("AATAAA"),"PAS variant":consensus_pwm("ATTAAA"),
         "U-rich (DSE)":consensus_pwm("TTTTT"),"GU-rich (DSE)":consensus_pwm("TGTGTG"),
         "UGUA (CFIm)":consensus_pwm("TGTA"),"branchpoint":consensus_pwm("CTRAC")}
def rc_pwm(M): return M[::-1][:, [3,2,1,0]]
def match_pwm(f, known=KNOWN, mo=5):
    fw=f.shape[0]; best=("none",-1.0)
    for nm,M0 in known.items():
        for M,tag in ((M0,""),(rc_pwm(M0)," (rc)")):
            mw=M.shape[0]
            for off in range(-(mw-mo), fw-mo+1):
                m0=max(0,-off); f0=max(0,off); ov=min(mw-m0, fw-f0)
                if ov<mo: continue
                sim=float(np.mean(np.sum(M[m0:m0+ov]*f[f0:f0+ov],axis=1)))
                if sim>best[1]: best=(nm+tag,sim)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True); ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--upstream_nt", type=int, default=2000); ap.add_argument("--downstream_nt", type=int, default=500)
    ap.add_argument("--filters", type=int, default=64); ap.add_argument("--w1", type=int, default=19)
    ap.add_argument("--w2", type=int, default=7); ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--n_pos_bins", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seqs", type=int, default=3000, help="sequences fed to modisco")
    ap.add_argument("--modisco_window", type=int, default=500, help="crop +/- this many nt around the site")
    ap.add_argument("--max_seqlets", type=int, default=20000)
    ap.add_argument("--sliding_window", type=int, default=15); ap.add_argument("--flank", type=int, default=5)
    ap.add_argument("--out", default="cnn_interp_2b")
    a = ap.parse_args()

    import torch, torch.nn as nn
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # load shortlist + fetch sequence (identical to cnn_interpret.py)
    import pysam
    sl = pd.read_parquet(a.shortlist)
    sl = sl[sl["n_internal"] > 0].copy(); sl["is_canonical"] = sl["is_canonical"].astype(bool)
    sl = sl.reset_index(drop=True)
    L = a.upstream_nt + a.downstream_nt + 1; up, dn = a.upstream_nt, a.downstream_nt
    print(f"[data] {sl.gene.nunique():,} genes / {len(sl):,} peaks; window {L} nt", file=sys.stderr)
    fa = pysam.FastaFile(a.reference_fasta); reflens = {r: fa.get_reference_length(r) for r in fa.references}
    def oriented(chrom, gs, ge):
        cl = reflens.get(chrom)
        if cl is None: return "N"*(ge-gs)
        a_=max(0,gs); b_=min(cl,ge); core=fa.fetch(chrom,a_,b_) if b_>a_ else ""
        return ("N"*(a_-gs if gs<0 else 0))+core+("N"*(ge-b_ if ge>cl else 0))
    X = np.zeros((len(sl), L), dtype=np.int8)
    for i, r in enumerate(sl.itertuples(index=False)):
        c=int(r.center)
        s = revcomp(oriented(r.chrom, c-dn, c+up+1)) if r.strand=="-" else oriented(r.chrom, c-up, c+dn+1)
        X[i] = seq_to_idx(s[:L] if len(s)>=L else s+"N"*(L-len(s)))
    fa.close()
    y = sl["is_canonical"].to_numpy(); gene = sl["gene"].to_numpy()

    LUT = torch.zeros(5,4); LUT[0,0]=LUT[1,1]=LUT[2,2]=LUT[3,3]=1.0; LUT=LUT.to(dev)
    def onehot(t): return LUT[t].permute(0,2,1)
    def onehot_np(idx):
        oh=np.zeros((idx.shape[0],idx.shape[1],4),dtype=np.float32); m=idx<4
        rr,cc=np.nonzero(m); oh[rr,cc,idx[m]]=1.0; return oh
    Xt = torch.tensor(X, dtype=torch.long)

    class TwoBranchCNN(nn.Module):
        def __init__(s, nf, w1, w2, hidden, p, n_bins):
            super().__init__(); s.n_bins=n_bins
            s.conv1=nn.Conv1d(4,nf,w1,padding=w1//2); s.conv2=nn.Conv1d(nf,nf,w2,padding=w2//2); s.bn2=nn.BatchNorm1d(nf)
            s.att=nn.Conv1d(nf,1,1); s.pos_w=nn.Parameter(torch.zeros(nf,n_bins))
            s.fc1=nn.Linear(nf,hidden); s.drop=nn.Dropout(p); s.fc2=nn.Linear(hidden,1); s.bias=nn.Parameter(torch.zeros(1))
        def first(s,x): return torch.exp(s.conv1(x).clamp(max=40))
        def forward(s,x):
            a1=s.first(x); h=torch.relu(s.bn2(s.conv2(a1))); aw=torch.softmax(s.att(h),dim=-1)
            hl=(h*aw).sum(-1); local=s.fc2(s.drop(torch.relu(s.fc1(hl)))).squeeze(-1)
            binned=nn.functional.adaptive_max_pool1d(a1,s.n_bins); pos=(binned*s.pos_w.unsqueeze(0)).sum(dim=(1,2))
            return local+pos+s.bias

    # train one interpretation model (gene-held-out val, early stop on auprc)
    rng = np.random.default_rng(a.seed)
    ug = np.array(sorted(set(gene))); rng.shuffle(ug); vg = set(ug[:max(1,int(len(ug)*a.val_frac))])
    vmask = np.array([g in vg for g in gene]); tr = np.nonzero(~vmask)[0]; va = np.nonzero(vmask)[0]
    net = TwoBranchCNN(a.filters, a.w1, a.w2, a.hidden, a.dropout, a.n_pos_bins).to(dev)
    pos = y[tr].mean(); pw = torch.tensor([(1-pos)/max(pos,1e-6)], dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); yt = torch.tensor(y.astype(np.float32))
    from sklearn.metrics import average_precision_score
    def sc(rows, bs=512):
        net.eval(); o=[]
        with torch.no_grad():
            for i in range(0,len(rows),bs): o.append(torch.sigmoid(net(onehot(Xt[rows[i:i+bs]].to(dev)))).cpu().numpy())
        return np.concatenate(o)
    best=-1; bstate=None; bad=0
    for ep in range(a.epochs):
        net.train(); perm=rng.permutation(tr)
        for i in range(0,len(perm),a.batch):
            idx=perm[i:i+a.batch]; xb=onehot(Xt[idx].to(dev)); yb=yt[idx].to(dev)
            opt.zero_grad(); loss=lossf(net(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        m=average_precision_score(y[va], sc(va))
        if m>best: best=m; bad=0; bstate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            bad+=1
            if bad>=a.patience: break
    net.load_state_dict(bstate); torch.save(bstate, f"{a.out}_model.pt")
    print(f"[train] best val AUPRC {best:.3f}; saved {a.out}_model.pt", file=sys.stderr)

    # hypothetical attributions (raw gradient) on a sample, cropped around the site
    samp = rng.choice(len(X), min(a.n_seqs, len(X)), replace=False)
    def hyp_grad(idx_arr, batch=128):
        net.eval(); out=np.zeros((len(idx_arr), idx_arr.shape[1], 4), dtype=np.float32)
        for i in range(0,len(idx_arr),batch):
            xb=onehot(torch.tensor(idx_arr[i:i+batch],dtype=torch.long).to(dev)).clone().requires_grad_(True)
            yb=net(xb); net.zero_grad(); yb.sum().backward()
            out[i:i+batch]=xb.grad.detach().cpu().numpy().transpose(0,2,1)
        return out
    site=up; w=a.modisco_window; lo=max(0,site-w); hi=min(L,site+w)
    hyp = hyp_grad(X[samp])[:, lo:hi, :]
    oh  = onehot_np(X[samp])[:, lo:hi, :]
    hyp = np.ascontiguousarray(hyp.astype(np.float32)); oh = np.ascontiguousarray(oh.astype(np.float32))
    print(f"[modisco] input {hyp.shape} (seqs x {hi-lo} nt x 4), cropped to +/-{w} around site", file=sys.stderr)

    # run tf-modisco (robust to modisco-lite api)
    try:
        import modiscolite
    except ImportError:
        sys.exit("[err] modisco-lite not importable in this env")
    try:
        pos_p, neg_p = modiscolite.tfmodisco.TFMoDISco(
            hypothetical_contribs=hyp, one_hot=oh,
            max_seqlets_per_metacluster=a.max_seqlets,
            sliding_window_size=a.sliding_window, flank_size=a.flank,
            target_seqlet_fdr=0.05, n_leiden_runs=2)
    except TypeError as e:
        import inspect
        print(f"[modisco] TFMoDISco signature mismatch: {e}", file=sys.stderr)
        print("[modisco] actual signature:", inspect.signature(modiscolite.tfmodisco.TFMoDISco), file=sys.stderr)
        sys.exit("Re-run with matching kwargs (paste the signature above and I will pin it)")
    try:
        modiscolite.io.save_hdf5(f"{a.out}_modisco.h5", pos_p, neg_p, window_size=a.sliding_window)
    except Exception as e:
        print(f"[modisco] save_hdf5 note: {e}", file=sys.stderr)

    def pattern_pfm(pat):
        for attr in ("sequence", "seq", "ppm"):
            if hasattr(pat, attr):
                m = np.array(getattr(pat, attr), dtype=float)
                if m.ndim == 2 and m.shape[1] == 4: return m
                if m.ndim == 2 and m.shape[0] == 4: return m.T
        return None
    rows = []
    for lab, pats in (("pos", pos_p), ("neg", neg_p)):
        if pats is None: continue
        for pi, pat in enumerate(pats):
            pfm = pattern_pfm(pat)
            if pfm is None: continue
            pfm = pfm / np.clip(pfm.sum(axis=1, keepdims=True), 1e-9, None)
            ns = int(len(pat.seqlets)) if hasattr(pat, "seqlets") else 0
            mm, ms = match_pwm(pfm)
            for pp in range(pfm.shape[0]):
                for b in range(4):
                    rows.append(dict(metacluster=lab, motif=pi, pos=pp, base=BASES[b],
                                     prob=float(pfm[pp, b]), n_seqlets=ns, match=mm, match_score=round(ms, 3)))
    if not rows:
        print("[modisco] no patterns extracted (motif objects lacked a PFM attribute)", file=sys.stderr)
    else:
        pd.DataFrame(rows).to_csv(f"{a.out}_modisco_motifs.tsv", sep="\t", index=False)
        npos = len(pos_p) if pos_p is not None else 0; nneg = len(neg_p) if neg_p is not None else 0
        print(f"[modisco] wrote {a.out}_modisco_motifs.tsv ({npos} pos, {nneg} neg patterns)", file=sys.stderr)

if __name__ == "__main__":
    main()