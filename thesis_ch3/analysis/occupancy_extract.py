#!/usr/bin/env python3
# signal of the so et al
import os, sys, numpy as np, pyBigWig
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
G = os.path.join(PUB, "external/geo/GSE135140")
TRACKS = [("U1A", "xlip/GSM3989602_U1A_XLIP_cAMO"), ("U1C", "xlip/GSM3989603_U1C_XLIP_cAMO"), ("SF3B1", "xlip/GSM3989604_SF3B1_XLIP_cAMO"),
          ("CFIm25", "xlip/GSM3989606_CFIm25_XLIP_cAMO"), ("Fip1", "xlip/GSM3989608_Fip1_XLIP_cAMO"), ("CstF64", "xlip/GSM3989610_CstF64_XLIP_cAMO"),
          ("hnRNPC", "xlip/GSM3989612_hnRNPC_XLIP_cAMO"), ("input", "xlip/GSM3989614_Bio-cAMO_Input"),
          ("polyA_cAMO", "bw/GSM3989596_EU-labeled_polyA_cAMO"), ("polyA_U1AMO", "bw/GSM3989597_EU-labeled_polyA_U1AMO")]
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]; gene = D["gene"]
DECOY = len(sys.argv) > 1 and sys.argv[1] == "decoy"
if DECOY:
    import csv
    keep = set(gene); seen = set(); rows = []
    for r in csv.DictReader(open(os.path.join(PUB, "decoy_positions.tsv")), delimiter="\t"):
        k = (r["gene"], r["competitor_row"])
        if r["gene"] in keep and k not in seen: seen.add(k); rows.append(r)
    ch = np.array([r["chrom"] for r in rows]); cen = np.array([int(r["center"]) for r in rows]); st = np.array([r["strand"] for r in rows]); gene = np.array([r["gene"] for r in rows])
N = len(cen)
UP, DN, BIN = 1000, 500, 25; NB = (UP + DN) // BIN
def work(job):
    name, stem, sf = job
    bw = pyBigWig.open(os.path.join(G, f"{stem}.{sf}.bw")); sizes = bw.chroms(); out = np.zeros((N, NB), np.float32)
    for i in range(N):
        c = str(ch[i]); p = int(cen[i])
        a, b = (p - UP, p + DN) if st[i] == "+" else (p - DN + 1, p + UP + 1)
        if c not in sizes or a < 0 or b > sizes[c]: continue
        v = np.nan_to_num(np.asarray(bw.values(c, a, b), dtype=np.float32))
        if st[i] == "-": v = v[::-1]
        out[i] = v.reshape(NB, BIN).mean(1)
    print(name, sf, "done", flush=True)
    return name, sf, out
if __name__ == "__main__":
    jobs = [(n, s, sf) for n, s in TRACKS for sf in ("fwd", "rev")]
    with Pool(6) as pool: res = pool.map(work, jobs)
    prof = np.zeros((len(TRACKS), 2, N, NB), np.float32); names = [n for n, _ in TRACKS]
    for n, sf, o in res: prof[names.index(n), 0 if sf == "fwd" else 1] = o
    out = "data/occupancy_profiles_decoy.npz" if DECOY else "data/occupancy_profiles.npz"
    np.savez_compressed(os.path.join(ROOT, out), prof=prof, tracks=np.array(names), bin_start=np.arange(-UP, DN, BIN), gene=gene, chrom=ch, center=cen, strand=st)
    print("[out]", out, N)
