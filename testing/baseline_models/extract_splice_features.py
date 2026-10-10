#!/usr/bin/env python3
# extract pas-machinery + splice/u1 features at candidate pas sites
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import pysam


PAS_HEXAMERS = {
    "AATAAA": 1.00,
    "ATTAAA": 0.72,
    "TATAAA": 0.30,
    "AGTAAA": 0.27,
    "AAGAAA": 0.21,
    "AATATA": 0.19,
    "AATACA": 0.17,
    "CATAAA": 0.16,
    "GATAAA": 0.13,
    "AATGAA": 0.12,
    "ACTAAA": 0.07,
    "AATAGA": 0.07,
}

U1_5P_SEQUENCE = "ACUUACCUG"

UP_FLANK   = 3000
DOWN_FLANK = 1000


ALPHA = "ACGT"

def build_pwm_from_consensus(consensus, strength=2.0):
    IUPAC = {
        "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
        "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
        "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
    }
    L = len(consensus)
    pwm = np.zeros((4, L), dtype=np.float32)
    eS = float(np.exp(strength))
    for j, c in enumerate(consensus.upper()):
        bases = IUPAC.get(c, "ACGT")
        weights = np.array([eS if b in bases else 1.0 for b in ALPHA])
        pwm[:, j] = weights / weights.sum()
    return pwm

USE_PWM = build_pwm_from_consensus("TTTTTTTT", strength=2.0)

DSE_PWM = build_pwm_from_consensus("TGTGTGTG", strength=2.0)


def scan_pwm_best(seq, pwm):
    L = pwm.shape[1]
    if len(seq) < L:
        return 0.0
    log_odds = np.log2(np.maximum(pwm, 1e-9) / 0.25)
    base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    best = -1e30
    for i in range(len(seq) - L + 1):
        score = 0.0
        valid = True
        for j in range(L):
            b = seq[i + j]
            if b not in base_idx:
                valid = False
                break
            score += log_odds[base_idx[b], j]
        if valid and score > best:
            best = score
    return float(best) if best > -1e29 else 0.0


def revcomp(seq):
    tab = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tab)[::-1]


def fetch_pas_window(fa, chrom, center, strand, up=UP_FLANK, down=DOWN_FLANK):
    if strand == "+":
        seq = fa.fetch(chrom, max(0, center - up), center + down).upper()
    else:
        seq = fa.fetch(chrom, max(0, center - down), center + up).upper()
        seq = revcomp(seq)
    return seq


def pas_hexamer_score(window_seq, window_offset_of_center=UP_FLANK,
                       search_radius=50):
    lo = max(0, window_offset_of_center - search_radius)
    hi = min(len(window_seq) - 6, window_offset_of_center + search_radius)
    best = 0.0
    for i in range(lo, hi + 1):
        hex_ = window_seq[i:i+6]
        if hex_ in PAS_HEXAMERS and PAS_HEXAMERS[hex_] > best:
            best = PAS_HEXAMERS[hex_]
    return float(best)


def use_strength(window_seq, window_offset_of_center=UP_FLANK, use_len=50):
    use = window_seq[max(0, window_offset_of_center - use_len):window_offset_of_center]
    if len(use) == 0:
        return 0.0
    return float(use.count("T")) / len(use)


def dse_strength(window_seq, window_offset_of_center=UP_FLANK, dse_len=100):
    dse = window_seq[window_offset_of_center:window_offset_of_center + dse_len]
    if len(dse) == 0:
        return 0.0
    n_gu = dse.count("G") + dse.count("T")
    return float(n_gu) / len(dse)


def ugua_count_upstream(window_seq, window_offset_of_center=UP_FLANK,
                         upstream_len=100):
    up = window_seq[max(0, window_offset_of_center - upstream_len):
                    window_offset_of_center]
    if len(up) < 4:
        return 0.0
    count = 0
    for i in range(len(up) - 3):
        if up[i:i+4] == "TGTA":
            count += 1
    return float(count)


def use_pwm_score(window_seq, window_offset_of_center=UP_FLANK,
                   upstream_len=50):
    up = window_seq[max(0, window_offset_of_center - upstream_len):
                    window_offset_of_center]
    return scan_pwm_best(up, USE_PWM)


def dse_pwm_score(window_seq, window_offset_of_center=UP_FLANK,
                   downstream_len=100):
    dn = window_seq[window_offset_of_center:
                    window_offset_of_center + downstream_len]
    return scan_pwm_best(dn, DSE_PWM)


def t_stretch_5plus(window_seq, window_offset_of_center=UP_FLANK,
                     upstream_len=100):
    up = window_seq[max(0, window_offset_of_center - upstream_len):
                    window_offset_of_center]
    if len(up) < 5:
        return 0.0
    count = 0
    run = 0
    triggered = False
    for c in up:
        if c == "T":
            run += 1
            if run == 5 and not triggered:
                count += 1
                triggered = True
        else:
            run = 0
            triggered = False
    return float(count)


def rnahybrid_min_mfe(target_seq, rnahybrid_path, query_seq=U1_5P_SEQUENCE):
    if len(target_seq) < len(query_seq):
        return 0.0

    def to_rna(s):
        return s.upper().replace("T", "U").translate(
            {ord(c): None for c in "NRYSWKMBDHV-."})
    tgt = to_rna(target_seq)
    qry = to_rna(query_seq)
    if len(tgt) < len(qry) or len(qry) == 0:
        return 0.0

    tgt_fa = None
    qry_fa = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".fa",
                                          delete=False) as f:
            f.write(f">target\n{tgt}\n")
            tgt_fa = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".fa",
                                          delete=False) as f:
            f.write(f">query\n{qry}\n")
            qry_fa = f.name
        result = subprocess.run(
            [rnahybrid_path, "-s", "3utr_human", "-c", "-n", "30",
             "-t", tgt_fa, "-q", qry_fa],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0
        fields = result.stdout.strip().split("\n")[0].split(":")
        if len(fields) < 5:
            return 0.0
        return float(fields[4])
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return 0.0
    finally:
        for p in (tgt_fa, qry_fa):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def pas_prox_u1(window_seq, rnahybrid_path,
                window_offset_of_center=UP_FLANK, search_radius=200):
    lo = max(0, window_offset_of_center - search_radius)
    hi = min(len(window_seq), window_offset_of_center + search_radius)
    region = window_seq[lo:hi]
    return rnahybrid_min_mfe(region, rnahybrid_path)


def maxentscan_best(seq_list, maxentscan_dir, kind="5"):
    if not seq_list:
        return 0.0
    script = "score5.pl" if kind == "5" else "score3.pl"
    expected_len = 9 if kind == "5" else 23
    valid_seqs = [s for s in seq_list if len(s) == expected_len
                   and set(s.upper()) <= set("ACGT")]
    if not valid_seqs:
        return 0.0
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                          delete=False) as f:
            f.write("\n".join(valid_seqs) + "\n")
            tmp_path = f.name
        result = subprocess.run(
            ["perl", os.path.join(maxentscan_dir, script), tmp_path],
            capture_output=True, text=True, timeout=60, check=False,
            cwd=maxentscan_dir,
        )
        os.remove(tmp_path)
        if result.returncode != 0:
            return 0.0
        scores = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    scores.append(float(parts[-1]))
                except ValueError:
                    pass
        return max(scores) if scores else 0.0
    except (subprocess.TimeoutExpired, OSError):
        return 0.0


def extract_5ss_candidates(seq, start, end):
    cands = []
    for i in range(start, min(end, len(seq) - 9)):
        sub = seq[i:i+9]
        if sub[3:5] == "GT":
            cands.append(sub)
    if len(cands) > 200:
        cands = cands[::len(cands) // 200][:200]
    return cands


def extract_3ss_candidates(seq, start, end):
    cands = []
    for i in range(start, min(end, len(seq) - 23)):
        sub = seq[i:i+23]
        if sub[18:20] == "AG":
            cands.append(sub)
    if len(cands) > 200:
        cands = cands[::len(cands) // 200][:200]
    return cands


def term_3ss_strength(window_seq, maxentscan_dir,
                       window_offset_of_center=UP_FLANK):
    cands = extract_3ss_candidates(window_seq, 0, window_offset_of_center)
    return maxentscan_best(cands, maxentscan_dir, kind="3")


def intervening_5ss(window_seq, maxentscan_dir,
                     window_offset_of_center=UP_FLANK,
                     search_radius=500):
    lo = max(0, window_offset_of_center - search_radius)
    hi = window_offset_of_center
    cands = extract_5ss_candidates(window_seq, lo, hi)
    return maxentscan_best(cands, maxentscan_dir, kind="5")


def downstream_5ss(window_seq, maxentscan_dir,
                    window_offset_of_center=UP_FLANK):
    cands = extract_5ss_candidates(
        window_seq, window_offset_of_center, len(window_seq))
    return maxentscan_best(cands, maxentscan_dir, kind="5")


def features_for_site(chrom, center, strand, genome_fa_path,
                       maxentscan_dir, rnahybrid_path):
    fa = pysam.FastaFile(genome_fa_path)
    seq = fetch_pas_window(fa, chrom, int(center), strand)
    fa.close()
    if len(seq) < UP_FLANK + DOWN_FLANK - 200:
        return None
    return {
        "pas_hexamer_score":    pas_hexamer_score(seq),
        "use_strength":         use_strength(seq),
        "dse_strength":         dse_strength(seq),
        "ugua_count_upstream":  ugua_count_upstream(seq),
        "use_pwm_score":        use_pwm_score(seq),
        "dse_pwm_score":        dse_pwm_score(seq),
        "t_stretch_5plus":      t_stretch_5plus(seq),
        "pas_prox_u1":          pas_prox_u1(seq, rnahybrid_path),
        "term_3ss_strength":    term_3ss_strength(seq, maxentscan_dir),
        "intervening_5ss":      intervening_5ss(seq, maxentscan_dir),
        "downstream_5ss":       downstream_5ss(seq, maxentscan_dir),
    }


def _worker(row, genome_fa_path, maxentscan_dir, rnahybrid_path):
    feats = features_for_site(
        row["chrom"], row["center"], row["strand"],
        genome_fa_path, maxentscan_dir, rnahybrid_path)
    if feats is None:
        return None
    feats.update({
        "chrom":  row["chrom"],
        "center": int(row["center"]),
        "strand": row["strand"],
        "name":   row.get("name", ""),
    })
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_bed",   required=True,
                    help="6-column BED of candidate PAS sites")
    ap.add_argument("--genome_fa",   required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--maxentscan",  required=True,
                    help="Directory containing score5.pl, score3.pl")
    ap.add_argument("--rnahybrid",   required=True,
                    help="Path to RNAhybrid binary")
    ap.add_argument("--n_workers",   type=int, default=4)
    ap.add_argument("--limit",       type=int, default=None,
                    help="Process only the first N sites (for debugging)")
    args = ap.parse_args()

    sites = []
    with open(args.sites_bed) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            chrom, start, end = p[0], int(p[1]), int(p[2])
            name   = p[3] if len(p) > 3 else ""
            strand = p[5] if len(p) > 5 else "+"
            sites.append({
                "chrom":  chrom,
                "center": (start + end) // 2,
                "strand": strand,
                "name":   name,
            })
    if args.limit:
        sites = sites[:args.limit]
    print(f"[features] processing {len(sites):,} sites with "
          f"{args.n_workers} workers", file=sys.stderr)

    rows = []
    if args.n_workers == 1:
        for i, r in enumerate(sites):
            out = _worker(r, args.genome_fa, args.maxentscan, args.rnahybrid)
            if out:
                rows.append(out)
            if (i + 1) % 100 == 0:
                print(f"    {i+1:,}/{len(sites):,}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = [
                ex.submit(_worker, r, args.genome_fa,
                          args.maxentscan, args.rnahybrid)
                for r in sites
            ]
            for i, fut in enumerate(as_completed(futures)):
                out = fut.result()
                if out:
                    rows.append(out)
                if (i + 1) % 100 == 0:
                    print(f"    {i+1:,}/{len(sites):,}", file=sys.stderr)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    df.to_parquet(args.out_parquet, index=False, compression="snappy")
    print(f"[features] wrote {len(df):,} rows -> {args.out_parquet}",
          file=sys.stderr)
    print(df.describe().to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()