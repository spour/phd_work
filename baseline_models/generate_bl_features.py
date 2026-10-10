# !/usr/bin/env python3
print("running")

import os
import re
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO

pd.options.display.precision = 16
pd.options.display.max_columns = None
print("imports successful")

# pwm loading (u->t) & log-odds

def read_pwm_table_with_U(path):
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    if any(c.lower() == "pos" for c in df.columns):
        df = df.drop(columns=[c for c in df.columns if c.lower() == "pos"])
    lower = {c.lower(): c for c in df.columns}
    if "u" in lower and "t" not in lower:
        df = df.rename(columns={lower["u"]: "T"})
    want = ["A", "C", "G", "T"]
    colmap = {c.upper(): c for c in df.columns}
    df = df[[colmap[w] for w in want]]
    arr = df.to_numpy(dtype=float).T
    colsum = arr.sum(axis=0, keepdims=True)
    colsum[colsum == 0.0] = 1.0
    arr = arr / colsum
    return arr

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
                continue
            name = os.path.basename(path)
            pwms[name] = logodds_matrix(pwm_probs, bg=bg)
    if not pwms:
        print("WARNING: no PWMs loaded.")
    else:
        print(f"pwms read in: {len(pwms)}")
    return pwms

# one-hot & ops

def one_hot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.uint8)
    s = seq.upper()
    idx = {'A':0,'C':1,'G':2,'T':3}
    for i,ch in enumerate(s):
        j = idx.get(ch, None)
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
    from collections import deque
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
    return (np.arange(n_macro)[:, None] + (np.arange(n_bins) * bin_stride)[None, :])

# engineered k-mer families


def engineered_counts_per_macro_bins(macro_ohe_500):
    L, C = macro_ohe_500.shape
    micro_L, micro_stride = 30, 10
    n_bins = ((L - micro_L) // micro_stride) + 1

    s0, s1 = macro_ohe_500.strides
    bins = np.lib.stride_tricks.as_strided(
        macro_ohe_500,
        shape=(n_bins, micro_L, C),
        strides=(micro_stride * s0, s0, s1),
        writeable=False
    )

    A = bins[:, :, 0].astype(bool)
    Cc = bins[:, :, 1].astype(bool)
    G = bins[:, :, 2].astype(bool)
    T = bins[:, :, 3].astype(bool)

    # 1) ca_ua: count of "ca" or "ta" (overlapped)
    A_next = A[:, 1:]
    CA = (Cc[:, :-1] & A_next).sum(axis=1)
    TA = (T[:, :-1]  & A_next).sum(axis=1)
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


    def count_pattern(pattern_indices):
        Lp = len(pattern_indices)
        pos_max = micro_L - Lp + 1
        mask = np.ones((n_bins, pos_max), dtype=bool)
        for off, ch in enumerate(pattern_indices):
            if ch == 0:   mask &= A[:, off:off+pos_max]
            elif ch == 1: mask &= Cc[:, off:off+pos_max]
            elif ch == 2: mask &= G[:, off:off+pos_max]
            elif ch == 3: mask &= T[:, off:off+pos_max]
        return mask.sum(axis=1)

    TATTTT = count_pattern([3,0,3,3,3,3])
    TGTTTT = count_pattern([3,2,3,3,3,3])
    TTTTTT = count_pattern([3,3,3,3,3,3])
    poly_U_kmers = TATTTT + TGTTTT + TTTTTT

    poly_U_kmer = count_pattern([3,3,3,3])

    UGUA_kmers = count_pattern([3,2,3,0])

    return np.stack([CA_UA, DSEs, poly_U_kmers, poly_U_kmer, UGUA_kmers], axis=1)

def pwm_max_affinity_per_bin_all_macros(ohe, pwm_logodds,
                                        micro_L=30, micro_stride=10,
                                        macro_L=500, macro_stride=1):
    Lp = pwm_logodds.shape[1]
    seq_len = ohe.shape[0]
    if seq_len < macro_L:
        return np.empty((0, (macro_L - micro_L)//micro_stride + 1))

    S = correlate_pwm_over_seq(ohe, pwm_logodds)

    win = micro_L - Lp + 1
    if win < 1:
        raise ValueError(f"PWM length {Lp} longer than micro window {micro_L}")
    M = sliding_max_1d(S, win)

    n_macro = (seq_len - macro_L)//macro_stride + 1
    n_bins = (macro_L - micro_L)//micro_stride + 1
    starts = bins_index_matrix(n_macro, n_bins, micro_stride)
    return M[starts]


def process_fasta(args, pwms):
    macro_L, macro_stride = 500, 1
    micro_L, micro_stride = 30, 10
    n_bins = (macro_L - micro_L)//micro_stride + 1

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    write_header = not os.path.exists(args.outfile)

    for rec in SeqIO.parse(args.fasta, "fasta"):
        rid = rec.id
        seq = str(rec.seq).upper()
        if len(seq) < macro_L:
            continue

        ohe = one_hot(seq)
        seq_len = ohe.shape[0]
        n_macro = (seq_len - macro_L)//macro_stride + 1

        pwm_frames = []
        for name, logodds in pwms.items():
            mm = pwm_max_affinity_per_bin_all_macros(
                ohe, logodds,
                micro_L=micro_L, micro_stride=micro_stride,
                macro_L=macro_L, macro_stride=macro_stride
            )
            cols = [f"{name}_{i+1}" for i in range(n_bins)]
            pwm_frames.append(pd.DataFrame(mm, columns=cols))

        s0, s1 = ohe.strides
        macros = np.lib.stride_tricks.as_strided(
            ohe,
            shape=(n_macro, macro_L, 4),
            strides=(macro_stride * s0, s0, s1),
            writeable=False
        )
        eng_rows = [engineered_counts_per_macro_bins(macros[i]) for i in range(n_macro)]
        eng = np.stack(eng_rows, axis=0).reshape(n_macro, n_bins * 5)
        eng_cols = [f"{fam}_{i+1}"
                    for fam in ["CA_UA", "DSEs", "poly_U_kmers", "poly_U_kmer", "UGUA_kmers"]
                    for i in range(n_bins)]
        eng_df = pd.DataFrame(eng, columns=eng_cols)

        total = pd.concat(pwm_frames + [eng_df], axis=1) if pwm_frames else eng_df
        expected = (len(pwms) + 5) * n_bins
        if total.shape[1] != expected:
            raise ValueError(f"Feature count mismatch: got {total.shape[1]}, expected {expected}")

        total.index = [f"{rid}_{i}" for i in range(n_macro)]
        total.to_csv(args.outfile, sep="\t", header=write_header, index=True, mode="a")
        write_header = False
        print(f"{rid}: done")

    print("done")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--fasta", required=True, help="FASTA (strand-oriented via bedtools getfasta -s)")
    p.add_argument("-pwms", "--pwms", required=True, help="Text file listing paths to PWM tables (Pos A C G U)")
    p.add_argument("-out", "--outfile", required=True, help="Output TSV")
    p.add_argument("--bg", nargs=4, type=float, default=[0.3,0.2,0.2,0.3],
                   help="Background A C G T probabilities (default uniform)")
    args = p.parse_args()

    pwms = read_pwms(args.pwms, max_len=20, bg=np.array(args.bg, dtype=float))
    process_fasta(args, pwms)
