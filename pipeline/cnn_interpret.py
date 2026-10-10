#!/usr/bin/env python3
# a sequence-only twin of the splice reranker
import argparse, sys, math
import numpy as np, pandas as pd

BASES = np.array(list("ACGT"))
_TRANS = np.full(256, 4, dtype=np.int8)
for _c, _i in {"A":0,"C":1,"G":2,"T":3,"a":0,"c":1,"g":2,"t":3}.items():
    _TRANS[ord(_c)] = _i
_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def seq_to_idx(s):
    return _TRANS[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)].copy()

def revcomp(s):
    return s.translate(_COMP)[::-1]

def dinuc_shuffle(s, rng, max_tries=25):
    s = np.asarray(s); n = len(s)
    if n < 3: return s.copy()
    syms = np.unique(s); first = s[0]; last = s[-1]
    from collections import defaultdict
    for _ in range(max_tries):
        out = defaultdict(list)
        for i in range(n - 1):
            out[s[i]].append(s[i + 1])
        last_edge = {}; ok = True
        for node in syms:
            if node == last: continue
            ch = out[node]
            if not ch: ok = False; break
            last_edge[node] = ch[int(rng.integers(len(ch)))]
        if not ok: continue
        good = True
        for node in syms:
            if node == last: continue
            seen = set(); cur = node
            while cur != last:
                if cur in seen or cur not in last_edge: good = False; break
                seen.add(cur); cur = last_edge[cur]
            if not good: break
        if not good: continue
        adj = {}
        for node in syms:
            lst = list(out[node])
            if node != last:
                lst.remove(last_edge[node]); rng.shuffle(lst); lst.append(last_edge[node])
            else:
                rng.shuffle(lst)
            adj[node] = lst
        ptr = {node: 0 for node in syms}; res = [first]; cur = first
        for _ in range(n - 1):
            nxt = adj[cur][ptr[cur]]; ptr[cur] += 1; res.append(nxt); cur = nxt
        return np.array(res, dtype=s.dtype)
    return s.copy()

# known motifs (consensus -> pwm)
def consensus_pwm(seq, p=0.85):
    idx = {"A":0,"C":1,"G":2,"T":3}
    M = np.full((len(seq), 4), (1 - p) / 3)
    for i, ch in enumerate(seq):
        if ch in idx: M[i] = (1 - p) / 3; M[i, idx[ch]] = p
        else: M[i] = 0.25
    return M
KNOWN_MOTIFS = {
    "5'ss donor":    consensus_pwm("AGGTAAGT"),
    "3'ss acceptor": consensus_pwm("TTTTCAGG"),
    "PAS AAUAAA":    consensus_pwm("AATAAA"),
    "PAS variant":   consensus_pwm("ATTAAA"),
    "U-rich (DSE)":  consensus_pwm("TTTTT"),
    "GU-rich (DSE)": consensus_pwm("TGTGTG"),
    "branchpoint":   consensus_pwm("CTRAC"),
}
def rc_pwm(M): return M[::-1][:, [3, 2, 1, 0]]
def match_pwm(filt, known=KNOWN_MOTIFS, min_overlap=5):
    fw = filt.shape[0]; best = ("none", -1.0)
    for name, M0 in known.items():
        for M, tag in ((M0, ""), (rc_pwm(M0), " (rc)")):
            mw = M.shape[0]
            for off in range(-(mw - min_overlap), fw - min_overlap + 1):
                m0 = max(0, -off); f0 = max(0, off); ov = min(mw - m0, fw - f0)
                if ov < min_overlap: continue
                sim = float(np.mean(np.sum(M[m0:m0 + ov] * filt[f0:f0 + ov], axis=1)))
                if sim > best[1]: best = (name + tag, sim)
    return best
def info_content(pwm, eps=1e-6):
    p = np.clip(pwm, eps, 1); return float(np.sum(p * np.log2(p / 0.25)))

# metrics
def wilson(k, n, z=1.96):
    if n == 0: return (0., 0., 0.)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)
def within_gene_p1(gene, is_can, score, rng):
    order = np.argsort(gene, kind="stable")
    g = np.asarray(gene)[order]; yc = np.asarray(is_can)[order]
    s = np.asarray(score)[order] + rng.random(len(score)) * 1e-9
    b = np.flatnonzero(g[1:] != g[:-1]) + 1
    st = np.concatenate([[0], b]); en = np.concatenate([b, [len(g)]])
    return np.array([int(bool(yc[a + int(np.argmax(s[a:c]))])) for a, c in zip(st, en)])

def balanced_chrom_folds(chroms, counts, k, seed=0):
    uniq = sorted(set(chroms)); tot = {c: 0 for c in uniq}
    for c, n in zip(chroms, counts): tot[c] += n
    load = [0] * k; assign = {}
    for c in sorted(uniq, key=lambda x: -tot[x]):
        j = int(np.argmin(load)); assign[c] = j; load[j] += tot[c]
    return assign

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True,
                    help="shortlist parquet with gene, chrom, strand, center, is_canonical, n_internal")
    ap.add_argument("--reference_fasta", required=True, help="genome FASTA (e.g. hg38.fa)")
    ap.add_argument("--upstream_nt", type=int, default=2000)
    ap.add_argument("--downstream_nt", type=int, default=500)
    ap.add_argument("--n_folds", type=int, default=4,
                    help="chromosome-held-out folds for leak-free P@1 (higher approaches true LOCO)")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--filters", type=int, default=32)
    ap.add_argument("--w1", type=int, default=19); ap.add_argument("--w2", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--activation", default="exp", choices=["exp", "relu"])
    ap.add_argument("--arch", default="simple", choices=["simple", "two_branch"],
                    help="'two_branch' adds attention pooling and an interpretable motif x position "
                         "weight map (local motif branch + coarse positional branch)")
    ap.add_argument("--n_pos_bins", type=int, default=50, help="position bins for the two_branch positional map")
    ap.add_argument("--dinuc_control", action="store_true",
                    help="also score dinucleotide-shuffled sequence; performance should collapse "
                         "if the model uses positioned motifs rather than composition")
    ap.add_argument("--act_threshold", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.15); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--modisco", action="store_true")
    ap.add_argument("--modisco_max_seqlets", type=int, default=20000)
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = use all peaks; else cap per class for speed")
    ap.add_argument("--out", default="cnn_interp")
    a = ap.parse_args()

    import torch, torch.nn as nn
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[warn] no GPU found; a sequence CNN on 2.5kb windows will be slow on CPU", file=sys.stderr)

    import pysam
    sl = pd.read_parquet(a.shortlist)
    need = ["gene", "chrom", "strand", "center", "is_canonical", "n_internal"]
    miss = [c for c in need if c not in sl.columns]
    if miss: sys.exit(f"[err] shortlist missing columns: {miss}")
    sl = sl[sl["n_internal"] > 0].copy()
    sl["is_canonical"] = sl["is_canonical"].astype(bool)
    if a.max_per_class > 0:
        parts = [g.sample(min(len(g), a.max_per_class), random_state=a.seed)
                 for _, g in sl.groupby("is_canonical")]
        sl = pd.concat(parts).sort_index()
    sl = sl.reset_index(drop=True)
    L = a.upstream_nt + a.downstream_nt + 1
    print(f"[data] {sl.gene.nunique():,} contested genes / {len(sl):,} peaks; "
          f"fetching {L} nt windows (site at index {a.upstream_nt})", file=sys.stderr)

    fa = pysam.FastaFile(a.reference_fasta)
    reflens = {r: fa.get_reference_length(r) for r in fa.references}
    def oriented(chrom, gstart, gend):
        clen = reflens.get(chrom)
        if clen is None: return "N" * (gend - gstart)
        a_ = max(0, gstart); b_ = min(clen, gend)
        core = fa.fetch(chrom, a_, b_) if b_ > a_ else ""
        return ("N" * (a_ - gstart if gstart < 0 else 0)) + core + ("N" * (gend - b_ if gend > clen else 0))
    up, dn = a.upstream_nt, a.downstream_nt
    X = np.zeros((len(sl), L), dtype=np.int8)
    for i, r in enumerate(sl.itertuples(index=False)):
        c = int(r.center)
        if r.strand == "-":
            s = revcomp(oriented(r.chrom, c - dn, c + up + 1))
        else:
            s = oriented(r.chrom, c - up, c + dn + 1)
        X[i] = seq_to_idx(s[:L] if len(s) >= L else s + "N" * (L - len(s)))
        if (i + 1) % 20000 == 0: print(f"    fetched {i+1:,}/{len(sl):,}", file=sys.stderr)
    fa.close()
    y = sl["is_canonical"].to_numpy()
    gene = sl["gene"].to_numpy(); chrom = sl["chrom"].to_numpy()

    LUT = torch.zeros(5, 4); LUT[0,0]=LUT[1,1]=LUT[2,2]=LUT[3,3]=1.0
    LUT = LUT.to(dev)
    def onehot(idx_t):
        return LUT[idx_t].permute(0, 2, 1)

    class InterpCNN(nn.Module):
        def __init__(s, nf, w1, w2, hidden, p, act):
            super().__init__(); s.act = act
            s.conv1 = nn.Conv1d(4, nf, w1, padding=w1 // 2); s.pool1 = nn.MaxPool1d(4)
            s.conv2 = nn.Conv1d(nf, nf, w2, padding=w2 // 2); s.bn2 = nn.BatchNorm1d(nf)
            s.gpool = nn.AdaptiveMaxPool1d(1)
            s.fc1 = nn.Linear(nf, hidden); s.drop = nn.Dropout(p); s.fc2 = nn.Linear(hidden, 1)
        def first(s, x):
            z = s.conv1(x); return torch.exp(z.clamp(max=40)) if s.act == "exp" else torch.relu(z)
        def forward(s, x):
            h = s.pool1(s.first(x)); h = torch.relu(s.bn2(s.conv2(h)))
            h = s.gpool(h).squeeze(-1); h = s.drop(torch.relu(s.fc1(h)))
            return s.fc2(h).squeeze(-1)

    class TwoBranchCNN(nn.Module):
        def __init__(s, nf, w1, w2, hidden, p, act, n_bins):
            super().__init__(); s.act = act; s.n_bins = n_bins
            s.conv1 = nn.Conv1d(4, nf, w1, padding=w1 // 2)
            s.conv2 = nn.Conv1d(nf, nf, w2, padding=w2 // 2); s.bn2 = nn.BatchNorm1d(nf)
            s.att = nn.Conv1d(nf, 1, 1)
            s.pos_w = nn.Parameter(torch.zeros(nf, n_bins))
            s.fc1 = nn.Linear(nf, hidden); s.drop = nn.Dropout(p); s.fc2 = nn.Linear(hidden, 1)
            s.bias = nn.Parameter(torch.zeros(1))
        def first(s, x):
            z = s.conv1(x); return torch.exp(z.clamp(max=40)) if s.act == "exp" else torch.relu(z)
        def _aw(s, h):
            return torch.softmax(s.att(h), dim=-1)            
        def forward(s, x):
            a1 = s.first(x)                              
            h = torch.relu(s.bn2(s.conv2(a1)))
            aw = s._aw(h); h_local = (h * aw).sum(-1)          
            local = s.fc2(s.drop(torch.relu(s.fc1(h_local)))).squeeze(-1)
            binned = nn.functional.adaptive_max_pool1d(a1, s.n_bins) 
            pos = (binned * s.pos_w.unsqueeze(0)).sum(dim=(1, 2))
            return local + pos + s.bias
        def attention(s, x):
            a1 = s.first(x); h = torch.relu(s.bn2(s.conv2(a1)))
            return s._aw(h).squeeze(1)

    def make_net():
        if a.arch == "two_branch":
            return TwoBranchCNN(a.filters, a.w1, a.w2, a.hidden, a.dropout, a.activation, a.n_pos_bins).to(dev)
        return InterpCNN(a.filters, a.w1, a.w2, a.hidden, a.dropout, a.activation).to(dev)

    Xt = torch.tensor(X, dtype=torch.long)
    def score_idx(net, rows, bs=512):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                xb = onehot(Xt[rows[i:i+bs]].to(dev))
                out.append(torch.sigmoid(net(xb)).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def train_on(train_rows, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        gtr = gene[train_rows]; ug = np.array(sorted(set(gtr))); rng.shuffle(ug)
        vg = set(ug[:max(1, int(len(ug) * a.val_frac))])
        vmask = np.array([g in vg for g in gtr]); tr = train_rows[~vmask]; va = train_rows[vmask]
        net = make_net()
        pos = y[tr].mean(); pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        yt = torch.tensor(y.astype(np.float32))
        try:
            from sklearn.metrics import average_precision_score as _aps
        except Exception:
            _aps = None
        best = -1; best_state = None; bad = 0
        for ep in range(a.epochs):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm), a.batch):
                idx = perm[i:i+a.batch]
                xb = onehot(Xt[idx].to(dev)); yb = yt[idx].to(dev)
                opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            vs = score_idx(net, va)
            metric = _aps(y[va], vs) if _aps is not None else -np.mean((vs - y[va]) ** 2)
            if metric > best:
                best = metric; bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience: break
        net.load_state_dict(best_state); return net

    counts = np.ones(len(sl))
    assign = balanced_chrom_folds(chrom, counts, a.n_folds, seed=a.seed)
    fold_of = np.array([assign[c] for c in chrom])
    all_rows = np.arange(len(sl))
    pooled = np.full(len(sl), np.nan); fold0_net = None; fold0_test = None
    for k in range(a.n_folds):
        test_rows = all_rows[fold_of == k]; train_rows = all_rows[fold_of != k]
        if len(test_rows) == 0: continue
        print(f"[cv] fold {k+1}/{a.n_folds}: train {len(train_rows):,} / test {len(test_rows):,} peaks", file=sys.stderr)
        net = train_on(train_rows, seed=a.seed + k)
        pooled[test_rows] = score_idx(net, test_rows)
        if k == 0: fold0_net = net; fold0_test = test_rows
    rng = np.random.default_rng(a.seed)
    hits = within_gene_p1(gene, y, pooled, rng); pP, pLO, pHI = wilson(int(hits.sum()), len(hits))
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y, pooled)); auprc = float(average_precision_score(y, pooled))
    except Exception:
        auroc = auprc = float("nan")
    print(f"[cv] leak-free within-gene P@1 = {pP:.3f} [{pLO:.3f},{pHI:.3f}] (n={len(hits):,} genes); "
          f"AUROC {auroc:.3f}  AUPRC {auprc:.3f}", file=sys.stderr)
    pd.DataFrame([dict(metric="p1", value=pP, ci_lo=pLO, ci_hi=pHI),
                  dict(metric="auroc", value=auroc), dict(metric="auprc", value=auprc),
                  dict(metric="n_genes", value=int(len(hits))),
                  dict(metric="n_folds", value=int(a.n_folds)),
                  dict(metric="pos_rate", value=float(y.mean()))]).to_csv(
        f"{a.out}_metrics.tsv", sep="\t", index=False)
    sl_out = sl[["gene", "chrom", "center", "is_canonical"]].copy(); sl_out["cnn_score"] = pooled
    sl_out.to_parquet(f"{a.out}_test_scored.parquet", index=False)

    # interpretation on fold-0 model + its held-out sequences
    net = fold0_net; test_rows = fold0_test
    if net is None:
        print("[interp] no fold-0 model; skipping interpretation", file=sys.stderr); return
    Xte = X[test_rows]; yte = y[test_rows]

    def onehot_np(idx_arr):
        oh = np.zeros((idx_arr.shape[0], idx_arr.shape[1], 4), dtype=np.float32)
        m = idx_arr < 4
        rows, cols = np.nonzero(m); oh[rows, cols, idx_arr[m]] = 1.0
        return oh

    with torch.no_grad():
        acts = []
        for i in range(0, len(Xte), 512):
            xb = onehot(torch.tensor(Xte[i:i+512], dtype=torch.long).to(dev))
            acts.append(net.first(xb).cpu().numpy())
        A = np.concatenate(acts, 0)
    Xoh = onehot_np(Xte); half = a.w1 // 2; nF = A.shape[1]; ycan = yte.astype(bool)
    filt_rows = []; meta_rows = []
    for f in range(nF):
        Af = A[:, f, :]; thr = a.act_threshold * Af.max()
        if thr <= 0:
            meta_rows.append(dict(filter=f, n_hits=0, class_pref=0.0, info_content=0.0, match="dead", match_score=0.0)); continue
        pwm = np.zeros((a.w1, 4)); nhit = 0
        hit = np.argwhere(Af >= thr)
        if len(hit) > 60000:
            hit = hit[np.random.default_rng(0).choice(len(hit), 60000, replace=False)]
        for si, pj in hit:
            s0 = pj - half
            if s0 < 0 or s0 + a.w1 > A.shape[2]: continue
            pwm += Xoh[si, s0:s0 + a.w1, :]; nhit += 1
        if nhit < 20:
            meta_rows.append(dict(filter=f, n_hits=nhit, class_pref=0.0, info_content=0.0, match="rare", match_score=0.0)); continue
        pwm = pwm / pwm.sum(axis=1, keepdims=True)
        per_seq_max = Af.max(axis=1)
        pref = float(per_seq_max[ycan].mean() - per_seq_max[~ycan].mean())
        ic = info_content(pwm); name, sc = match_pwm(pwm)
        for pos in range(a.w1):
            for b in range(4): filt_rows.append(dict(filter=f, pos=pos, base=BASES[b], prob=float(pwm[pos, b])))
        meta_rows.append(dict(filter=f, n_hits=int(nhit), class_pref=pref, info_content=ic, match=name, match_score=round(sc, 3)))
    pd.DataFrame(filt_rows).to_csv(f"{a.out}_filters_pwm.tsv", sep="\t", index=False)
    fmeta = pd.DataFrame(meta_rows).sort_values("class_pref", key=lambda c: c.abs(), ascending=False)
    fmeta.to_csv(f"{a.out}_filter_meta.tsv", sep="\t", index=False)
    print(f"[filters] {int((fmeta.n_hits>=20).sum())}/{nF} live; top by |class preference|:", file=sys.stderr)
    for r in fmeta.head(8).itertuples():
        print(f"    filter {r.filter:<3} pref={r.class_pref:+.2f} IC={r.info_content:.1f} -> {r.match} ({r.match_score})", file=sys.stderr)

    def attributions(idx_arr, batch=128):
        net.eval(); out = np.zeros((len(idx_arr), 4, idx_arr.shape[1]), dtype=np.float32)
        for i in range(0, len(idx_arr), batch):
            xb = onehot(torch.tensor(idx_arr[i:i+batch], dtype=torch.long).to(dev)).clone().requires_grad_(True)
            yb = net(xb); net.zero_grad(); yb.sum().backward()
            g = xb.grad.detach().cpu().numpy(); out[i:i+batch] = g * xb.detach().cpu().numpy()
        return out
    ste = score_idx(net, test_rows); n_each = max(1, a.n_examples // 2)
    ci = np.where(ycan)[0]; ki = np.where(~ycan)[0]
    sel = np.concatenate([ci[np.argsort(-ste[ci])][:n_each], ki[np.argsort(ste[ki])][:n_each]])
    attr = attributions(Xte[sel]); ex_rows = []
    for e, gi in enumerate(sel):
        oh = Xoh[gi]; contrib = attr[e].T
        per_pos = (contrib * oh).sum(axis=1)
        base_at = np.where(oh.sum(axis=1) > 0, BASES[oh.argmax(axis=1)], "N")
        for pos in range(len(per_pos)):
            ex_rows.append(dict(example=int(e), is_canonical=bool(ycan[gi]),
                                pos=pos - a.upstream_nt, base=str(base_at[pos]), contrib=float(per_pos[pos])))
    pd.DataFrame(ex_rows).to_csv(f"{a.out}_attr_examples.tsv", sep="\t", index=False)
    print(f"[attribution] wrote {len(sel)} example saliency tracks", file=sys.stderr)

    if a.arch == "two_branch":
        pw = net.pos_w.detach().cpu().numpy()
        bin_w = L / a.n_pos_bins
        centers = (np.arange(a.n_pos_bins) + 0.5) * bin_w - a.upstream_nt
        match_of = dict(zip(fmeta["filter"], fmeta["match"]))
        pref_of = dict(zip(fmeta["filter"], fmeta["class_pref"]))
        pw_rows = []
        for f in range(pw.shape[0]):
            for j in range(pw.shape[1]):
                pw_rows.append(dict(filter=f, bin=j, position_nt=float(centers[j]),
                                    weight=float(pw[f, j]),
                                    match=match_of.get(f, ""), class_pref=float(pref_of.get(f, 0.0))))
        pd.DataFrame(pw_rows).to_csv(f"{a.out}_pos_weights.tsv", sep="\t", index=False)
        with torch.no_grad():
            aw = net.attention(onehot(torch.tensor(Xte[sel], dtype=torch.long).to(dev))).cpu().numpy()
        at_rows = []
        for e, gi in enumerate(sel):
            for pos in range(aw.shape[1]):
                at_rows.append(dict(example=int(e), is_canonical=bool(ycan[gi]),
                                    pos=pos - a.upstream_nt, attention=float(aw[e, pos])))
        pd.DataFrame(at_rows).to_csv(f"{a.out}_attention_examples.tsv", sep="\t", index=False)
        print(f"[two_branch] wrote motif x position map ({pw.shape[0]}x{pw.shape[1]}) and attention tracks", file=sys.stderr)

    if a.dinuc_control:
        rng2 = np.random.default_rng(a.seed + 7)
        Xsh = np.stack([dinuc_shuffle(Xte[i], rng2) for i in range(len(Xte))])
        with torch.no_grad():
            ssh = []
            for i in range(0, len(Xsh), 512):
                xb = onehot(torch.tensor(Xsh[i:i+512], dtype=torch.long).to(dev))
                ssh.append(torch.sigmoid(net(xb)).cpu().numpy())
            ssh = np.concatenate(ssh)
        try:
            from sklearn.metrics import roc_auc_score
            au_real = float(roc_auc_score(yte, ste)); au_sh = float(roc_auc_score(yte, ssh))
        except Exception:
            au_real = au_sh = float("nan")
        pd.DataFrame([dict(metric="auroc_real_holdout", value=au_real),
                      dict(metric="auroc_dinuc_shuffled", value=au_sh)]).to_csv(
            f"{a.out}_dinuc_control.tsv", sep="\t", index=False)
        print(f"[dinuc] held-out AUROC real {au_real:.3f} vs dinucleotide-shuffled {au_sh:.3f} "
              f"(collapse toward 0.5 = positioned-motif signal)", file=sys.stderr)

    if a.modisco:
        try:
            import modiscolite
            samp = np.random.default_rng(0).choice(len(Xte), min(len(Xte), 4000), replace=False)
            attr_s = attributions(Xte[samp]).transpose(0, 2, 1); onehot_s = onehot_np(Xte[samp])
            pos_p, neg_p = modiscolite.tfmodisco.TFMoDISco(
                hypothetical_contribs=attr_s, one_hot=onehot_s,
                max_seqlets_per_metacluster=a.modisco_max_seqlets,
                sliding_window_size=15, flank_size=5, target_seqlet_fdr=0.05, n_leiden_runs=2)
            modiscolite.io.save_hdf5(f"{a.out}_modisco.h5", pos_p, neg_p, window_size=15)
            rows = []
            for lab, pats in (("pos", pos_p), ("neg", neg_p)):
                if pats is None: continue
                for pi, pat in enumerate(pats):
                    pfm = np.array(pat.sequence); pfm = pfm / np.clip(pfm.sum(axis=1, keepdims=True), 1e-9, None)
                    ns = int(len(pat.seqlets)) if hasattr(pat, "seqlets") else 0
                    mm, ms = match_pwm(pfm)
                    for pos in range(pfm.shape[0]):
                        for b in range(4):
                            rows.append(dict(metacluster=lab, motif=pi, pos=pos, base=BASES[b],
                                             prob=float(pfm[pos, b]), n_seqlets=ns, match=mm, match_score=round(ms, 3)))
            pd.DataFrame(rows).to_csv(f"{a.out}_modisco_motifs.tsv", sep="\t", index=False)
            print(f"[modisco] wrote {a.out}_modisco_motifs.tsv and _modisco.h5", file=sys.stderr)
        except ImportError:
            print("[modisco] modisco-lite not installed; skipping (filters + attribution unaffected)", file=sys.stderr)
        except Exception as e:
            print(f"[modisco] failed ({type(e).__name__}: {e}); skipping", file=sys.stderr)

    print(f"\n[out] {a.out}_metrics.tsv, {a.out}_filters_pwm.tsv, {a.out}_filter_meta.tsv, "
          f"{a.out}_attr_examples.tsv, {a.out}_test_scored.parquet", file=sys.stderr)

if __name__ == "__main__":
    main()