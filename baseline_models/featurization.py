#!/usr/bin/env python3
# featurization for cpa prediction (pwm scores + engineered k-mer counts per 30 nt bin
import argparse
import os
from collections import deque

import numpy as np
import pandas as pd
from Bio import SeqIO


# pwm loading (u->t) & log-odds

def read_pwm_table_with_U(path):
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    if any(c.lower() == "pos" for c in df.columns):
        df = df.drop(columns=[c for c in df.columns if c.lower() == "pos"])
    lower = {c.lower(): c for c in df.columns}
    if "u" in lower and "t" not in lower:
        df = df.rename(columns={lower["u"]: "T"})
    colmap = {c.upper(): c for c in df.columns}
    df = df[[colmap[w] for w in ["A", "C", "G", "T"]]]
    arr = df.to_numpy(dtype=float).T
    colsum = arr.sum(axis=0, keepdims=True)
    colsum[colsum == 0.0] = 1.0
    return arr / colsum


def logodds_matrix(pwm_probs, bg=None, eps=1e-4):
    if bg is None:
        bg = np.full(4, 0.25, dtype=float)
    bg = np.asarray(bg, dtype=float)
    odds = (pwm_probs + eps) / bg[:, None]
    return 10.0 * np.log10(odds)


def read_pwms(pwms_list_file, max_len=20, bg=None):
    pwms = {}
    with open(pwms_list_file) as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            pwm_probs = read_pwm_table_with_U(path)
            L = pwm_probs.shape[1]
            if L > max_len:
                print(f"  WARNING: skipping {os.path.basename(path)} "
                      f"(length {L} > max_len {max_len})")
                continue
            name = os.path.basename(path)
            pwms[name] = logodds_matrix(pwm_probs, bg=bg)
    print(f"PWMs loaded: {len(pwms)}")
    return pwms


# one-hot & ops

_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.uint8)
    for i, ch in enumerate(seq.upper()):
        j = _IDX.get(ch)
        if j is not None:
            arr[i, j] = 1
    return arr


def correlate_pwm_over_seq(ohe, pwm_logodds):
    Lp = pwm_logodds.shape[1]
    out = np.zeros(ohe.shape[0] - Lp + 1, dtype=np.float64)
    for b in (0, 1, 2, 3):
        out += np.correlate(ohe[:, b], pwm_logodds[b, :], mode="valid")
    return out


def sliding_max_1d(x, w):
    n = x.shape[0]
    if w < 1 or w > n:
        raise ValueError("Invalid window size")
    dq = deque()
    out = np.empty(n - w + 1, dtype=x.dtype)
    for i in range(n):
        while dq and dq[0] <= i - w:
            dq.popleft()
        while dq and x[dq[-1]] <= x[i]:
            dq.pop()
        dq.append(i)
        if i >= w - 1:
            out[i - w + 1] = x[dq[0]]
    return out


def bins_index_matrix(n_macro, n_bins, bin_stride):
    return np.arange(n_macro)[:, None] + (np.arange(n_bins) * bin_stride)[None, :]


# engineered k-mer families

ENG_FAMILIES = ["CA_UA", "DSEs", "poly_U_kmers", "poly_U_kmer", "UGUA_kmers"]


def engineered_counts_per_macro_bins(macro_ohe_500):
    L, C = macro_ohe_500.shape
    micro_L, micro_stride = 30, 10
    n_bins = ((L - micro_L) // micro_stride) + 1

    s0, s1 = macro_ohe_500.strides
    bins = np.lib.stride_tricks.as_strided(
        macro_ohe_500,
        shape=(n_bins, micro_L, C),
        strides=(micro_stride * s0, s0, s1),
        writeable=False,
    )
    A = bins[:, :, 0].astype(bool)
    Cc = bins[:, :, 1].astype(bool)
    G = bins[:, :, 2].astype(bool)
    T = bins[:, :, 3].astype(bool)

    A_next = A[:, 1:]
    CA = (Cc[:, :-1] & A_next).sum(axis=1)
    TA = (T[:, :-1] & A_next).sum(axis=1)
    CA_UA = CA + TA

    G_count = G.sum(axis=1)
    T_count = T.sum(axis=1)
    GG = (G[:, :-1] & G[:, 1:]).sum(axis=1)
    GT = (G[:, :-1] & T[:, 1:]).sum(axis=1)
    TG = (T[:, :-1] & G[:, 1:]).sum(axis=1)
    TT = (T[:, :-1] & T[:, 1:]).sum(axis=1)
    TTT = (T[:, :-2] & T[:, 1:-1] & T[:, 2:]).sum(axis=1)
    TGG = (T[:, :-2] & G[:, 1:-1] & G[:, 2:]).sum(axis=1)
    TGT = (T[:, :-2] & G[:, 1:-1] & T[:, 2:]).sum(axis=1)
    GTG = (G[:, :-2] & T[:, 1:-1] & G[:, 2:]).sum(axis=1)
    DSEs = G_count + T_count + GG + GT + TG + TT + TTT + TGG + TGT + GTG

    def count_pattern(pat):
        Lp = len(pat)
        pos_max = micro_L - Lp + 1
        mask = np.ones((n_bins, pos_max), dtype=bool)
        for off, ch in enumerate(pat):
            if ch == 0:   mask &= A[:, off:off + pos_max]
            elif ch == 1: mask &= Cc[:, off:off + pos_max]
            elif ch == 2: mask &= G[:, off:off + pos_max]
            elif ch == 3: mask &= T[:, off:off + pos_max]
        return mask.sum(axis=1)

    poly_U_kmers = (count_pattern([3, 0, 3, 3, 3, 3])
                    + count_pattern([3, 2, 3, 3, 3, 3])
                    + count_pattern([3, 3, 3, 3, 3, 3]))
    poly_U_kmer = count_pattern([3, 3, 3, 3])
    UGUA_kmers = count_pattern([3, 2, 3, 0])

    return np.stack([CA_UA, DSEs, poly_U_kmers, poly_U_kmer, UGUA_kmers], axis=1)


# pwm → per-bin maxima

def pwm_max_affinity_per_bin_all_macros(ohe, pwm_logodds,
                                        micro_L=30, micro_stride=10,
                                        macro_L=500, macro_stride=1):
    Lp = pwm_logodds.shape[1]
    seq_len = ohe.shape[0]
    if seq_len < macro_L:
        return np.empty((0, (macro_L - micro_L) // micro_stride + 1))
    S = correlate_pwm_over_seq(ohe, pwm_logodds)
    win = micro_L - Lp + 1
    if win < 1:
        raise ValueError(f"PWM length {Lp} longer than micro window {micro_L}")
    M = sliding_max_1d(S, win)
    n_macro = (seq_len - macro_L) // macro_stride + 1
    n_bins = (macro_L - micro_L) // micro_stride + 1
    starts = bins_index_matrix(n_macro, n_bins, micro_stride)
    return M[starts]


def _feature_column_names(pwm_names, n_bins=48):
    cols = []
    for name in pwm_names:
        cols.extend([f"{name}_{i + 1}" for i in range(n_bins)])
    for fam in ENG_FAMILIES:
        cols.extend([f"{fam}_{i + 1}" for i in range(n_bins)])
    return cols


def featurize_fasta_to_df(fasta_path, pwms, bg=None, verbose=True):
    macro_L, macro_stride = 500, 1
    micro_L, micro_stride = 30, 10
    n_bins = (macro_L - micro_L) // micro_stride + 1

    pwm_names = list(pwms.keys())
    col_names = _feature_column_names(pwm_names, n_bins=n_bins)

    chunks, ids = [], []
    n_processed, n_skipped = 0, 0
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).upper()
        if len(seq) < macro_L:
            n_skipped += 1
            continue

        ohe = one_hot(seq)
        n_macro = (len(seq) - macro_L) // macro_stride + 1

        pwm_blocks = []
        for name in pwm_names:
            mm = pwm_max_affinity_per_bin_all_macros(ohe, pwms[name])
            pwm_blocks.append(mm)
        pwm_block = np.concatenate(pwm_blocks, axis=1) if pwm_blocks else np.empty((n_macro, 0))

        s0, s1 = ohe.strides
        macros = np.lib.stride_tricks.as_strided(
            ohe,
            shape=(n_macro, macro_L, 4),
            strides=(macro_stride * s0, s0, s1),
            writeable=False,
        )
        eng_block = np.stack(
            [engineered_counts_per_macro_bins(macros[i]) for i in range(n_macro)],
            axis=0,
        ).reshape(n_macro, n_bins * len(ENG_FAMILIES))

        chunks.append(np.concatenate([pwm_block, eng_block], axis=1).astype(np.float32))
        for i in range(n_macro):
            ids.append(rec.id if n_macro == 1 else f"{rec.id}_{i}")
        n_processed += 1

    if not chunks:
        raise ValueError(f"No sequences >= {macro_L} nt in {fasta_path}")

    X = np.concatenate(chunks, axis=0)
    expected = (len(pwm_names) + len(ENG_FAMILIES)) * n_bins
    if X.shape[1] != expected:
        raise ValueError(f"Feature count mismatch: got {X.shape[1]}, expected {expected}")

    if verbose:
        print(f"  processed: {n_processed:,} sequences  "
              f"(skipped {n_skipped:,} < {macro_L} nt)  "
              f"-> {X.shape[0]:,} rows x {X.shape[1]} features")

    return pd.DataFrame(X, index=ids, columns=col_names)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--fasta", required=True)
    p.add_argument("-pwms", "--pwms", required=True)
    p.add_argument("-out", "--outfile", required=True)
    p.add_argument("--bg", nargs=4, type=float, default=[0.3, 0.2, 0.2, 0.3])
    args = p.parse_args()

    bg = np.array(args.bg, dtype=float)
    pwms = read_pwms(args.pwms, max_len=20, bg=bg)
    df = featurize_fasta_to_df(args.fasta, pwms, bg=bg)

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    df.to_csv(args.outfile, sep="\t", index=True)
    print(f"Wrote {args.outfile}  shape={df.shape}")


if __name__ == "__main__":
    _cli()