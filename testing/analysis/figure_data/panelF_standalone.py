#!/usr/bin/env python3
# the panel-f metagene of make_figure3_v3_2.r, rerun on the final competitor set
import sys, os, math, time
import numpy as np
from multiprocessing import Pool

SRC = "task_v32_analyses.py"
MARK = "# ---- +/-2kb donor scan for all peaks ----"
head = open(SRC).read().split(MARK)[0]
G_ = {"__name__": "__main__"}
exec(compile(head, SRC + ":header", "exec"), G_)

donor_profile = G_["donor_profile"]
gene = G_["gene"]; isc = G_["isc"]; chrom = G_["chrom"]; center = G_["center"]; strand = G_["strand"]
NP = G_["NP"]; SPAN = G_["SPAN"]; MBIN = G_["MBIN"]; DEC_SUB = G_["DEC_SUB"]; FASTA = G_["FASTA"]
ND = G_["ND"]
BEDGES = np.arange(-SPAN, SPAN + 1, MBIN); BCEN = (BEDGES[:-1] + BEDGES[1:]) / 2.0
NBIN = len(BCEN)
PARQ = "cpa_peaks_final.parquet"

def log(*a): print(*a, file=sys.stderr, flush=True)
def _init(): G_["G"].fh = open(FASTA, "rb")

def _bin(pr):
    if pr is None: return np.full(NBIN, np.nan), False
    d, h = pr
    bi = np.digitize(d, BEDGES) - 1; ok = (bi >= 0) & (bi < NBIN)
    cnt = np.bincount(bi[ok], weights=h[ok].astype(float), minlength=NBIN)
    tot = np.bincount(bi[ok], minlength=NBIN).astype(float)
    return np.where(tot > 0, cnt / np.maximum(tot, 1) * 1000.0, 0.0), True

def peak_job(i): return _bin(donor_profile(chrom[i], center[i], strand[i]))

DEC = {}
def decoy_job(j):
    di = DEC["dsub"][j]; ci = int(DEC["gcanon"][int(DEC["dgene"][di])])
    b, ok = _bin(donor_profile(chrom[ci], int(DEC["dpos"][di]), strand[ci]))
    return (b if ok else np.full(NBIN, np.nan)), ci

def metag(bins_arr, sel):
    A = bins_arr[sel]; A = A[np.isfinite(A).all(1)]
    return A.mean(0), A.std(0, ddof=1) / math.sqrt(len(A)), len(A)

def write_tsv(path, cm, cs, cn, km, ks, kn, dm, ds, dn):
    with open(path, "w") as fh:
        fh.write("offset\tgroup\tdonor_per_kb\tse\tn\n")
        for j, off in enumerate(BCEN):
            fh.write(f"{off:.0f}\tcanonical\t{cm[j]:.4f}\t{cs[j]:.4f}\t{cn}\n")
            fh.write(f"{off:.0f}\tcompetitor\t{km[j]:.4f}\t{ks[j]:.4f}\t{kn}\n")
            fh.write(f"{off:.0f}\tdecoy\t{dm[j]:.4f}\t{ds[j]:.4f}\t{dn}\n")

if __name__ == "__main__":
    NPROC = min(32, os.cpu_count() or 8)

    # analysis-set membership, joined on the peak key
    import pyarrow.parquet as pq
    T = pq.read_table(PARQ).to_pandas()
    assert len(T) == NP, f"{PARQ} {len(T)} rows vs peak table {NP}"
    key = dict(zip(zip(T.gene, T.chrom, T.center.astype(int), T.strand), T.in_analysis_set.values))
    assert len(key) == NP, "peak key is not unique in the parquet"
    IN = np.array([key[(gene[i], chrom[i], int(center[i]), strand[i])] for i in range(NP)], bool)
    log(f"analysis set: {int((IN & (isc == 1)).sum())} canonical + {int((IN & (isc == 0)).sum())} competitor "
        f"in {len(set(gene[IN]))} genes  (of {int((isc==1).sum())} + {int((isc==0).sum())})")
    assert int((IN & (isc == 0)).sum()) == 43547, "competitor count is not 43,547"

    t = time.time()
    prof_bins = np.zeros((NP, NBIN)); prof_ok = np.zeros(NP, bool)
    with Pool(NPROC, initializer=_init) as p:
        for i, (b, ok) in enumerate(p.imap(peak_job, range(NP), chunksize=64)):
            prof_bins[i] = np.where(np.isfinite(b), b, 0.0); prof_ok[i] = ok
            if (i + 1) % 20000 == 0: log(f"  peak scan {i+1}/{NP}  {time.time()-t:.0f}s")
    log(f"peak donor scan done  {time.time()-t:.0f}s  profiles ok {int(prof_ok.sum())}/{NP}")

    D = np.load("decoy_set.npz"); dpos = D["pos"]; dgene = D["gene"]
    gi = np.load("u1_window_cache.npz")["gene_idx"]
    o = np.argsort(gi, kind="stable"); gs = gi[o]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    gcanon = []
    for j in range(len(cut) - 1):
        idx = o[cut[j]:cut[j+1]]; cc = idx[isc[idx] == 1]; kk = idx[isc[idx] == 0]
        if len(cc) == 1 and len(kk) >= 1: gcanon.append(int(cc[0]))
    gcanon = np.array(gcanon)
    assert dgene.max() < len(gcanon)
    dec_in = IN[gcanon[dgene.astype(int)]]
    log(f"decoys {len(dpos)} total, {int(dec_in.sum())} in analysis-set genes")

    def scan_decoys(pool_idx, tag):
        rng = np.random.default_rng(42)
        dsub = pool_idx[rng.choice(len(pool_idx), min(DEC_SUB, len(pool_idx)), replace=False)]
        DEC.update(dsub=dsub, dpos=dpos, dgene=dgene, gcanon=gcanon)
        t = time.time(); bins = np.empty((len(dsub), NBIN)); ci_of = []
        with Pool(NPROC, initializer=_init) as p:
            for j, (b, ci) in enumerate(p.imap(decoy_job, range(len(dsub)), chunksize=64)):
                bins[j] = b; ci_of.append(ci)
        log(f"decoy scan [{tag}] {len(dsub)} sites  {time.time()-t:.0f}s")
        return bins, np.array(ci_of)

    out = {}
    for tag, sel, dpool, path in [
        ("analysis", IN,               np.flatnonzero(dec_in),      "v32_F_metagene_analysisset.tsv"),
        ("all",      np.ones(NP, bool), np.arange(len(dpos)),       "v32_F_metagene_allpeaks.tsv")]:
        dbins, dci = scan_decoys(dpool, tag)
        cm, cs, cn = metag(prof_bins, (isc == 1) & prof_ok & sel)
        km, ks, kn = metag(prof_bins, (isc == 0) & prof_ok & sel)
        dm = np.nanmean(dbins, 0); ds = np.nanstd(dbins, 0) / math.sqrt(np.isfinite(dbins[:, 0]).sum())
        write_tsv(path, cm, cs, cn, km, ks, kn, dm, ds, len(dbins))
        out[tag] = dict(path=path, cn=cn, kn=kn, dn=len(dbins), dpool=len(dpool),
                        ng=len(set(gene[sel & prof_ok])), ngd=len(set(gene[dci])),
                        pa=(cm[19], km[19], dm[19]))

    np.savez_compressed("panelF_profiles.npz", prof_bins=prof_bins.astype(np.float32),
                        prof_ok=prof_ok, in_analysis_set=IN, is_canonical=(isc == 1),
                        gene=gene, bcen=BCEN)

    print("\n=== sites used in panel F ===")
    print(f"peak table rows                : {NP}  (profiles retrieved {int(prof_ok.sum())})")
    for tag, lab in [("analysis", "ANALYSIS SET  (competitors = model FP after APA / terminal-exon / DoG filters)"),
                     ("all", "all peaks (previous version, for comparison)")]:
        o_ = out[tag]
        print(f"\n{lab}")
        print(f"  real 3' ends (canonical)     : {o_['cn']}")
        print(f"  competitors                  : {o_['kn']}")
        print(f"  intronic decoys              : {o_['dn']}  (subsampled from {o_['dpool']})")
        print(f"  total sites plotted          : {o_['cn']+o_['kn']+o_['dn']}")
        print(f"  genes                        : {o_['ng']} (peaks), {o_['ngd']} (decoy arm)")
        print(f"  sites/kb at the pA (-50 nt)  : real {o_['pa'][0]:.2f}  comp {o_['pa'][1]:.2f}  decoy {o_['pa'][2]:.2f}")
        print(f"  [out] {o_['path']}")
    print(f"\nbins: {NBIN} x {MBIN} nt over +/-{SPAN} nt; hit = MaxEnt 5'ss >= {ND}")
    print("[out] panelF_profiles.npz")

    old = "v32_F_metagene.tsv"
    if os.path.exists(old):
        print(f"all-peaks version identical to committed {old}: "
              f"{open(old).read() == open('v32_F_metagene_allpeaks.tsv').read()}")
