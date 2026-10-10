#!/usr/bin/env python3
# every 3' splice site number produced in this session, in one frame
import csv, sys
import statistics as st
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
def load(p):
    B = {}
    for r in csv.DictReader(open(p), delimiter="\t"):
        B[(r["background"], r["seq_id"])] = float(r["best_delta"])
    return B
rows = []
def synth(path, site, n, seq, what, tag):
    B = load(path); k = ("real_canonical", f"{site}_x{n}")
    if k not in B: return
    sh = [B[("real_canonical", f"{site}_x{n}_mono{j}")] for j in range(10)
          if ("real_canonical", f"{site}_x{n}_mono{j}") in B]
    rows.append((tag, seq, what, n, B[k], B[k]-st.mean(sh)))
for n in (1, 7):
    synth("gia_dose_best.tsv", "acc_ppt", n, "TTTTTTTTTTTTTCAGG", "T-rich + AG, hand-built", "403d7eb")
    synth("gia_dose_best.tsv", "acc_best", n, "TTTCCTCTCTCTCCCCGCAGGTG", "MaxEnt optimum, 43% C", "403d7eb")
for n in (1, 3, 7):
    synth("gia_dose_matched_all.tsv", "acc_noC", n, "TTTTTTTTTGTGTGTTTTAGGTT", "T-rich + AG, score3 14.38", "6ca547f")
    synth("gia_dose_matched_all.tsv", "acc_best", n, "TTTCCTCTCTCTCCCCGCAGGTG", "MaxEnt optimum, 43% C", "6ca547f")
V = {}
for r in csv.DictReader(open("gia_native_acceptor.tsv"), delimiter="\t"):
    V.setdefault((r["background"], r["class"], int(r["length"]), r["arm"]), []).append(float(r["best_delta"]))
for cls, what in (("acc_top","real acceptors, top score3"), ("acc_mid","real acceptors, median score3"),
                  ("intron_ctl","intron with NO acceptor in it")):
    a = np.array(V[("real_canonical",cls,120,"native")]); b = np.array(V[("real_canonical",cls,120,"mono")])
    rows.append(("0f44f45", f"native 120 nt, n=150", what, 1, a.mean(), (a-b).mean()))
log(f"{'commit':<9}{'unit':<26}{'what it is':<32}{'n':>2}{'RAW':>9}{'minus own comp':>16}")
for t,s,w,n,r,d in rows:
    log(f"{t:<9}{s[:25]:<26}{w:<32}{n:>2}{r:>+9.3f}{d:>+16.3f}")
with open("acceptor_all_numbers.tsv","w") as fh:
    fh.write("commit\tunit\twhat\tn_copies\traw\tminus_composition\n")
    for r in rows: fh.write("\t".join(str(x) for x in r)+"\n")
log("\n[read] the T-rich units carry the large RAW effects and are at or BELOW a shuffle of their")
log("       own bases; the C-rich MaxEnt optimum carries the only positive arrangement-specific")
log("       effect and the smallest raw one. A 120 nt piece of intron with no acceptor in it")
log("       matches the best real acceptors. Both halves were reported; neither was wrong.")
log("[out] acceptor_all_numbers.tsv")
