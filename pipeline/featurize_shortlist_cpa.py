#!/usr/bin/env python3
# (figure 2 featurizer) distinct from featurize_shortlist.py
import argparse, os, sys, random
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

TEST_CHROMS = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS  = {"chr3","chr13"}
KEY = ["gene","chrom","center","strand"]
FP_LAB = "Model false positive"

PAS_HEXAMERS = ["AATAAA","ATTAAA","TATAAA","AGTAAA","AAGAAA","AATATA",
                "AATACA","CATAAA","GATAAA","AATGAA","TTTAAA","ACTAAA","AATAGA"]
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(s): return s.translate(COMP)[::-1]

class Fasta:
    def __init__(self, path):
        self.pysam = None
        try:
            import pysam
            self.fa = pysam.FastaFile(path); self.pysam = True
        except Exception:
            self._load_plain(path)
    def _load_plain(self, path):
        self.seqs = {}; name = None; buf = []
        with open(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    if name is not None: self.seqs[name] = "".join(buf)
                    name = line[1:].split()[0]; buf = []
                else:
                    buf.append(line.strip())
            if name is not None: self.seqs[name] = "".join(buf)
    def fetch(self, chrom, start, end, strand="+"):
        start = max(0, start)
        if self.pysam:
            s = self.fa.fetch(chrom, start, end).upper()
        else:
            s = self.seqs.get(chrom, "")[start:end].upper()
        return revcomp(s) if strand == "-" else s

def motif_positions(seq, up, down):
    n=len(seq)
    if n==0: return None
    pos={}
    hits=[]
    for h in PAS_HEXAMERS:
        st=seq.find(h)
        while st!=-1:
            hits.append(st-up); st=seq.find(h,st+1)
    pos["PAS"]=np.array(sorted(hits),dtype=float)
    hits=[]; st=seq.find("TGTA")
    while st!=-1: hits.append(st-up); st=seq.find("TGTA",st+1)
    pos["UGUA"]=np.array(hits,dtype=float)
    arr=np.frombuffer(seq.encode(),dtype=np.uint8)
    pos["U-rich"]=(np.nonzero(arr==ord("T"))[0]-up).astype(float)
    gu=[i for i in range(n-1) if seq[i:i+2] in ("GT","TG")]
    pos["GU-rich"]=(np.array(gu,dtype=float)-up)
    return pos

def count_motifs(seq, up, down):
    n = len(seq)
    if n == 0: return None
    upstream = seq[:up] if up <= n else seq
    downstream = seq[up:] if up < n else ""
    pas_zone = upstream[-50:] if len(upstream) >= 50 else upstream
    pas_hits = sum(pas_zone.count(h) for h in PAS_HEXAMERS)
    has_pas = int(any(h in pas_zone for h in PAS_HEXAMERS))
    ugua = upstream.count("TGTA")
    dse = downstream[:50]
    def maxrun(s, chars):
        best = run = 0
        for ch in s:
            run = run + 1 if ch in chars else 0
            best = max(best, run)
        return best
    u_run  = maxrun(dse, "T")
    gu_cnt = sum(dse[i:i+2] in ("GT","TG") for i in range(len(dse)-1))
    return dict(pas_hits=pas_hits, has_pas=has_pas, ugua=ugua,
                dse_urun=u_run, dse_gu=gu_cnt)

# altschul-erikson dinucleotide-preserving shuffle
def dinuc_shuffle(seq, rng):
    s = [c for c in seq if c in "ACGT"]
    if len(s) < 4: return seq
    from collections import defaultdict
    edges = defaultdict(list)
    for a, b in zip(s[:-1], s[1:]):
        edges[a].append(b)
    last = s[-1]
    for k in edges: rng.shuffle(edges[k])
    verts = list(set(s))
    def try_build():
        lastedge = {}
        for v in verts:
            if v == last: continue
            cand = [i for i,b in enumerate(edges[v])]
            if not cand: return None
            lastedge[v] = edges[v][cand[-1]]
        for v in verts:
            if v == last: continue
            seen=set(); cur=v
            while cur != last:
                if cur in seen or cur not in lastedge: return None
                seen.add(cur); cur = lastedge[cur]
        return lastedge
    lastedge = None
    for _ in range(12):
        for k in edges: rng.shuffle(edges[k])
        lastedge = try_build()
        if lastedge is not None: break
    if lastedge is None: return seq
    out = [s[0]]; used = defaultdict(int)
    order = {}
    for v in verts:
        lst = list(edges[v])
        if v != last and v in lastedge:
            lst.remove(lastedge[v]); lst.append(lastedge[v])
        order[v] = lst
    cur = s[0]
    for _ in range(len(s)-1):
        i = used[cur]
        if i >= len(order[cur]): break
        nxt = order[cur][i]; used[cur] += 1; out.append(nxt); cur = nxt
    if len(out) != len(s): return seq
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--peaks", default=None,
                    help="cpa_peaks_final.parquet. Merged on "
                         "gene/chrom/center/strand to drop non-canonical peaks "
                         "outside the analysis set. Without it the density "
                         "panels cover a different peak set from the rest of "
                         "Figure 2.")
    ap.add_argument("--no_filter", action="store_true",
                    help="keep every peak (old behaviour)")
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--up", type=int, default=100, help="upstream nt around center")
    ap.add_argument("--down", type=int, default=100, help="downstream nt around center")
    ap.add_argument("--n_shuffle", type=int, default=20)
    ap.add_argument("--split", default="test", choices=["all","test","val","train"])
    ap.add_argument("--contested_only", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="shortlist_cpa")
    a = ap.parse_args()

    df = pd.read_parquet(a.shortlist)
    if a.split == "test":  df = df[df.chrom.isin(TEST_CHROMS)]
    elif a.split == "val": df = df[df.chrom.isin(VAL_CHROMS)]
    elif a.split == "train": df = df[~df.chrom.isin(TEST_CHROMS | VAL_CHROMS)]
    if a.contested_only: df = df[df.n_internal > 0]
    df = df.reset_index(drop=True)

    # analysis-set filter
    if a.peaks and os.path.exists(a.peaks) and not a.no_filter:
        pk = pd.read_parquet(a.peaks, columns=KEY + ["in_analysis_set",
                                                     "flag_apa_annotated"])
        n0 = len(df)
        df = df.merge(pk, on=KEY, how="left")
        nmatch = int(df.in_analysis_set.notna().sum())
        print(f"[filter] peak flags merged on {'/'.join(KEY)}: "
              f"{nmatch:,}/{n0:,} matched", file=sys.stderr)
        if nmatch < n0:
            print(f"[filter] WARNING {n0-nmatch:,} rows unmatched -> treated as "
                  f"OUTSIDE the analysis set", file=sys.stderr)
        df["in_analysis_set"] = df.in_analysis_set.fillna(False).astype(bool)
        df["flag_apa_annotated"] = df.flag_apa_annotated.fillna(False).astype(bool)
        n_apa = int(((~df.is_canonical) & df.flag_apa_annotated).sum())
        keep = df.is_canonical | df.in_analysis_set
        print(f"[filter] dropping {int((~keep).sum()):,} non-canonical peaks "
              f"outside the analysis set ({n_apa:,} of them APA-annotated)",
              file=sys.stderr)
        df = df[keep].reset_index(drop=True)
    elif not a.no_filter:
        print("[filter] no --peaks given; NOT filtered. The density panels will "
              "cover a different peak set from the rest of Figure 2.",
              file=sys.stderr)

    if a.n_shards > 1:
        genes = sorted(df.gene.unique())[a.shard::a.n_shards]
        df = df[df.gene.isin(set(genes))].reset_index(drop=True)
        print(f"[shard] {a.shard}/{a.n_shards}", file=sys.stderr)
    print(f"[cpa] {len(df):,} peaks over {df.gene.nunique():,} {a.split} genes "
          f"({int(df.is_canonical.sum())} canonical, "
          f"{int((~df.is_canonical).sum())} model false positives)",
          file=sys.stderr)
    if len(df) == 0: sys.exit("[err] no peaks after filtering")

    fa = Fasta(a.reference_fasta)
    rng = random.Random(a.seed)
    up, down = a.up, a.down
    rows = []
    miss = 0
    MOTIFS = ["PAS","UGUA","U-rich","GU-rich"]
    L = up + down
    CLASSES = ("Canonical", FP_LAB)
    dens = {c: {m: np.zeros(L) for m in MOTIFS} for c in CLASSES}
    dens_null = {c: np.zeros(L) for c in CLASSES}
    npk = {c: 0 for c in CLASSES}
    for _, r in tqdm(df.iterrows(), total=len(df), desc="cpa-motifs"):
        c = int(r["center"]); strand = r["strand"]; chrom = r["chrom"]
        if strand == "+": w0, w1 = c - up, c + down
        else:             w0, w1 = c - down, c + up
        try:
            seq = fa.fetch(chrom, w0, w1, strand=strand)
        except Exception:
            seq = ""
        m = count_motifs(seq, up, down)
        if m is None:
            miss += 1
            rows.append(dict(pas_hits=np.nan, has_pas=np.nan, ugua=np.nan,
                             dse_urun=np.nan, dse_gu=np.nan,
                             has_pas_null=np.nan, pas_hits_null=np.nan))
            continue
        cls = "Canonical" if bool(r["is_canonical"]) else FP_LAB
        npk[cls] += 1
        pp = motif_positions(seq, up, down)
        if pp is not None:
            for mo in MOTIFS:
                idx = (pp[mo] + up).astype(int)
                idx = idx[(idx >= 0) & (idx < L)]
                if idx.size: np.add.at(dens[cls][mo], idx, 1.0)
        null_hits = 0; null_cnt = 0
        for _s in range(a.n_shuffle):
            sh = dinuc_shuffle(seq, rng)
            mm = count_motifs(sh, up, down)
            if mm:
                null_hits += mm["has_pas"]; null_cnt += mm["pas_hits"]
            ppn = motif_positions(sh, up, down)
            if ppn is not None:
                idx = (ppn["PAS"] + up).astype(int)
                idx = idx[(idx >= 0) & (idx < L)]
                if idx.size: np.add.at(dens_null[cls], idx, 1.0/max(a.n_shuffle,1))
        m["has_pas_null"] = null_hits / max(a.n_shuffle, 1)
        m["pas_hits_null"] = null_cnt / max(a.n_shuffle, 1)
        rows.append(m)
    F = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), F], axis=1)
    suf = f".{a.shard:03d}" if a.n_shards > 1 else ""
    out.to_parquet(f"{a.out}{suf}.parquet", index=False)

    drows = []
    xs = np.arange(-up, down)
    for cls in CLASSES:
        if npk[cls] == 0: continue
        for mo in MOTIFS:
            v = dens[cls][mo] / npk[cls]
            for x, y in zip(xs, v): drows.append(dict(cls=cls, motif=mo, pos=int(x), density=float(y)))
        v = dens_null[cls] / npk[cls]
        for x, y in zip(xs, v): drows.append(dict(cls=cls, motif="PAS (dinuc null)", pos=int(x), density=float(y)))
    pd.DataFrame(drows).to_csv(f"{a.out}{suf}_density.tsv", sep="\t", index=False)
    print(f"[cpa] wrote {a.out}{suf}_density.tsv  ({npk})", file=sys.stderr)

    pd.DataFrame([dict(cls=c, n=npk[c]) for c in CLASSES]).to_csv(
        f"{a.out}{suf}_density_counts.tsv", sep="\t", index=False)

    for lab, sub in [("canonical", out[out.is_canonical]),
                     ("false positive", out[~out.is_canonical])]:
        if len(sub):
            print(f"[{lab:<14}] n={len(sub):>6}  has_PAS(real)={sub.has_pas.mean():.3f}  "
                  f"has_PAS(null)={sub.has_pas_null.mean():.3f}  "
                  f"UGUA={sub.ugua.mean():.2f}  U-run={sub.dse_urun.mean():.1f}  GU={sub.dse_gu.mean():.1f}",
                  file=sys.stderr)
    print(f"[cpa] {miss} windows unfetchable; [out] {a.out}{suf}.parquet", file=sys.stderr)

if __name__ == "__main__":
    main()