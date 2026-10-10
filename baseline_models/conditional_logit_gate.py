#!/usr/bin/env python3
# quantify how much splicing gates cpa, beyond the rf probability, the right way
import argparse
import numpy as np, pandas as pd
from scipy.optimize import minimize

RF_TRAIN = {"chr1","chr2","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
            "chr11","chr12","chr14","chr15","chr16","chr17"}
RF_VAL  = {"chr3","chr13"}
RF_TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}

PRIOR = {"term_3ss__wg": +1, "upstream_max_me3__wg": +1, "term_ppt__wg": +1,
         "rf_logit": +1, "u1_suppression_down__wg": -1, "dn5ss_has_strong__wg": -1,
         "pas_prox_u1__wg": +1, "upstream_mean_u1__wg": +1, "local_me5__wg": -1,
         "interv_me5_max__wg": -1, "bp_strength__wg": +1, "u2af65_ppt__wg": +1,
         "u2af35_ag__wg": +1, "u1_duplex_down__wg": +1, "u2_duplex_bp__wg": +1}

def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))

# group structure: rows must be sorted by gene (contiguous groups)
def group_index(genes):
    codes = pd.factorize(genes)[0]
    order = np.argsort(codes, kind="stable")
    return order, codes[order]

def seg_starts_counts(codes_sorted):
    starts = np.r_[0, np.flatnonzero(np.diff(codes_sorted)) + 1]
    counts = np.diff(np.r_[starts, len(codes_sorted)])
    return starts, counts

def seg_softmax_ll(eta, starts, counts, true_pos):
    gmax = np.maximum.reduceat(eta, starts)
    gmax_row = np.repeat(gmax, counts)
    ex = np.exp(eta - gmax_row)
    gsum = np.add.reduceat(ex, starts)
    p = ex / np.repeat(gsum, counts)
    ll = (eta[true_pos] - (np.log(gsum) + gmax))
    return p, ll

def fit_clogit(X, starts, counts, true_pos, y):
    k = X.shape[1]
    def nll_grad(b):
        eta = X @ b
        p, ll = seg_softmax_ll(eta, starts, counts, true_pos)
        nll = -ll.sum()
        grad = X.T @ (p - y)
        return nll, grad
    res = minimize(nll_grad, np.zeros(k), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500))
    return res.x

def prep(df, feats):
    df = df.sort_values("gene", kind="stable").reset_index(drop=True)
    codes = pd.factorize(df["gene"].to_numpy())[0]
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    counts = np.diff(np.r_[starts, len(df)])
    iz = df["in_zone"].to_numpy()
    keep_rows = []
    k_starts = []
    k_counts = []
    k_truepos = []
    cur = 0
    for s, c in zip(starts, counts):
        sub = iz[s:s+c]
        w = np.where(sub == 1)[0]
        if len(w) == 1 and c >= 2:
            keep_rows.extend(range(s, s+c))
            k_starts.append(cur)
            k_counts.append(c)
            k_truepos.append(cur + int(w[0]))
            cur += c
    keep_rows = np.array(keep_rows, dtype=int)
    dfk = df.iloc[keep_rows].reset_index(drop=True)
    starts = np.array(k_starts, dtype=int)
    counts = np.array(k_counts, dtype=int)
    true_pos = np.array(k_truepos, dtype=int)
    pos_in_group = list(zip(starts.tolist(), counts.tolist()))
    return dfk, starts, counts, true_pos, pos_in_group

def evaluate(df, feats, mean, sd, beta, pos_in_group, contested_mask_fn):
    X = ((df[feats].to_numpy() - mean) / sd)
    eta = X @ beta
    lls, p1, rnd, contested = [], [], [], []
    for (s, c) in pos_in_group:
        e = eta[s:s+c]; iz = df["in_zone"].to_numpy()[s:s+c]
        m = e.max(); lse = np.log(np.exp(e - m).sum()) + m
        ll = float(e[np.argmax(iz)] - lse)
        lls.append(ll)
        p1.append(int(np.argmax(e) == int(np.argmax(iz))))
        rnd.append(1.0 / c)
        contested.append(contested_mask_fn(df, s, c))
    return np.array(lls), np.array(p1), np.array(rnd), np.array(contested)

def contested_at(df, s, c):
    sub = df.iloc[s:s+c]
    iz = sub["in_zone"].to_numpy(); rf = sub["rf_prob"].to_numpy()
    crf = rf[np.argmax(iz)]
    return bool((rf[iz == 0] >= crf).any())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--splice_feats", nargs="+",
                    default=["term_3ss__wg", "upstream_max_me3__wg"])
    ap.add_argument("--include_val_in_train", action="store_true", default=True)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--n_shuffle", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr" + df["chrom"])
    df["rf_logit"] = logit(df["rf_prob"])
    feats0 = ["rf_logit"]
    feats1 = ["rf_logit"] + a.splice_feats
    allfeats = feats1
    for f in allfeats:
        df[f] = df[f].fillna(0.0)

    train_chroms = RF_TRAIN | (RF_VAL if a.include_val_in_train else set())
    tr = df[df.chrom.isin(train_chroms)].copy()
    te = df[df.chrom.isin(RF_TEST)].copy()

    trp, s_tr, c_tr, tp_tr, pig_tr = prep(tr, allfeats)
    tep, s_te, c_te, tp_te, pig_te = prep(te, allfeats)
    print(f"[data] train choice-genes {len(pig_tr):,} | test choice-genes {len(pig_te):,}")
    print(f"[features] M0 = {feats0}\n           M1 = {feats1}\n")

    mean = trp[allfeats].to_numpy().mean(0); sd = trp[allfeats].to_numpy().std(0); sd[sd == 0] = 1
    Xtr = (trp[allfeats].to_numpy() - mean) / sd
    ytr = np.zeros(len(trp)); ytr[tp_tr] = 1.0

    def fit_on(feature_list):
        cols = [allfeats.index(f) for f in feature_list]
        b = fit_clogit(Xtr[:, cols], s_tr, c_tr, tp_tr, ytr)
        full = np.zeros(len(allfeats)); 
        for bi, ci in zip(b, cols): full[ci] = bi
        return full, b, cols

    b0_full, b0, cols0 = fit_on(feats0)
    b1_full, b1, cols1 = fit_on(feats1)

    _, ll0_tr = seg_softmax_ll(Xtr @ b0_full, s_tr, c_tr, tp_tr)
    _, ll1_tr = seg_softmax_ll(Xtr @ b1_full, s_tr, c_tr, tp_tr)
    lr = 2 * (ll1_tr.sum() - ll0_tr.sum()); dfree = len(a.splice_feats)
    from scipy.stats import chi2
    p_lr = chi2.sf(lr, dfree)
    print(f"[LR test, in-sample] chi2={lr:.1f}  df={dfree}  p={p_lr:.2e}  "
          f"(splice adds information beyond RF)")

    ll0, p10, rnd0, con0 = evaluate(tep, allfeats, mean, sd, b0_full, pig_te, contested_at)
    ll1, p11, rnd1, con1 = evaluate(tep, allfeats, mean, sd, b1_full, pig_te, contested_at)
    con = con1.astype(bool)

    def report(mask, name):
        if mask.sum() == 0:
            return
        g0 = ll0[mask].mean(); g1 = ll1[mask].mean()
        gain_bits = (g1 - g0) / np.log(2)
        pr = np.exp(g1 - g0)
        print(f"\n[{name}]  n={int(mask.sum()):,}")
        print(f"  held-out mean log-prob on TRUE end:  M0={g0:.4f}  M1={g1:.4f} nats")
        print(f"  >>> info splice adds beyond RF = {gain_bits:+.4f} bits/gene")
        print(f"  >>> splice multiplies prob on the true end by {pr:.3f}x (geo-mean)")
        print(f"  P@1 (argmax of frozen model):  M0={p10[mask].mean():.3f}  "
              f"M1={p11[mask].mean():.3f}  random={rnd1[mask].mean():.3f}")
        print(f"      P@1 gain M1 over random = {p11[mask].mean()-rnd1[mask].mean():+.3f}")

    report(np.ones(len(ll1), bool), "ALL choice-genes")
    report(con, "CONTESTED (RF ties/loses, where splice can gate)")
    report(~con, "NON-CONTESTED (RF already wins)")

    idx = np.arange(len(ll1))
    boot_bits_all, boot_bits_con, boot_p1_con = [], [], []
    for _ in range(a.n_boot):
        bs = rng.choice(idx, size=len(idx), replace=True)
        boot_bits_all.append(((ll1[bs]-ll0[bs]).mean())/np.log(2))
        cb = bs[con[bs]]
        if len(cb) > 20:
            boot_bits_con.append(((ll1[cb]-ll0[cb]).mean())/np.log(2))
            boot_p1_con.append(p11[cb].mean()-rnd1[cb].mean())
    def ci(v): 
        v = np.array(v); return np.percentile(v,2.5), np.percentile(v,97.5)
    lo,hi = ci(boot_bits_all); print(f"\n[bootstrap] ALL info gain: {np.mean(boot_bits_all):+.4f} bits "
                                     f"[{lo:+.4f},{hi:+.4f}]")
    lo,hi = ci(boot_bits_con); print(f"[bootstrap] CONTESTED info gain: {np.mean(boot_bits_con):+.4f} bits "
                                     f"[{lo:+.4f},{hi:+.4f}]")
    lo,hi = ci(boot_p1_con);   print(f"[bootstrap] CONTESTED P@1 gain vs random: {np.mean(boot_p1_con):+.4f} "
                                     f"[{lo:+.4f},{hi:+.4f}]")

    print(f"\n[M1 coefficients (standardized) and direction vs prior]")
    for f, bi in zip(feats1, b1):
        pr = PRIOR.get(f, 0); ok = "n/a" if pr == 0 else ("ok" if np.sign(bi) == pr else "INVERTED")
        print(f"  {f:<24} beta={bi:+.3f}  {ok}")

    print(f"\n[shuffle null] permute the true end within gene, refit M1, "
          f"held-out info gain should -> ~0 bits")
    for sd_seed in range(a.n_shuffle):
        rr = np.random.default_rng(100 + sd_seed)
        y_sh = np.zeros(len(trp)); tp_sh = []
        for (s, c) in pig_tr:
            tp_sh.append(s + int(rr.integers(c)))
        tp_sh = np.array(tp_sh); y_sh[tp_sh] = 1.0
        b1s = fit_clogit(Xtr[:, cols1], s_tr, c_tr, tp_sh, y_sh)
        b1s_full = np.zeros(len(allfeats))
        for bi, ci_ in zip(b1s, cols1): b1s_full[ci_] = bi
        b0s = fit_clogit(Xtr[:, cols0], s_tr, c_tr, tp_sh, y_sh)
        b0s_full = np.zeros(len(allfeats))
        for bi, ci_ in zip(b0s, cols0): b0s_full[ci_] = bi
        ll0s,_,_,_ = evaluate(tep, allfeats, mean, sd, b0s_full, pig_te, contested_at)
        ll1s,_,_,_ = evaluate(tep, allfeats, mean, sd, b1s_full, pig_te, contested_at)
        gbits = ((ll1s-ll0s)[con].mean())/np.log(2)
        flag = "OK ~0" if abs(gbits) < 0.01 else "LEAK SUSPECT"
        print(f"  shuffle {sd_seed}: contested info gain = {gbits:+.4f} bits   {flag}")

    print(f"\nINTERPRETATION:")
    print(f"  - LR p-value: is the splice contribution statistically real.")
    print(f"  - CONTESTED info gain (bits): how much splicing resolves the true 3' end")
    print(f"    beyond CPA grammar, where CPA grammar is ambiguous. This is the number.")
    print(f"  - prob-ratio: how many times more probability splice puts on the true end.")
    print(f"  - shuffle null near 0 certifies the gain is signal, not artifact.")

if __name__ == "__main__":
    main()