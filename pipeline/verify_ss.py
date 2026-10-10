#!/usr/bin/env python3
# verify exon_reference.py splice-site coordinates: acceptor 23-mers should end in ag
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from collections import defaultdict
from extract_splicing_features import GenomeFasta, init_maxentscan
o = init_maxentscan(); score5, score3 = o[0], o[1]
fa = GenomeFasta(sys.argv[2]); gtf = sys.argv[3]

tx = defaultdict(list)
for line in open(gtf):
    if line.startswith("#"): continue
    f = line.rstrip().split("\t")
    if len(f) < 9 or f[2] != "exon" or "protein_coding" not in f[8]: continue
    tid = [x for x in f[8].split(";") if "transcript_id" in x][0].split('"')[1]
    tx[tid].append((int(f[3])-1, int(f[4]), f[6], f[0]))
    if len(tx) > 3000: break

gt=ag=tot=0; a3=[]; d5=[]
for tid, exs in tx.items():
    if len(exs) < 3: continue
    strand = exs[0][2]; exs = sorted(exs, key=lambda e: e[0])
    if strand == "-": exs = exs[::-1]
    for i in range(1, len(exs)-1):
        st, en, sd, ch = exs[i]
        if strand == "+":
            accp, donp = st, en-1
            sa = fa.fetch(ch, accp-20, accp+3, strand="+")
            sdn = fa.fetch(ch, donp-2, donp+7, strand="+")
        else:
            accp, donp = en-1, st
            sa = fa.fetch(ch, accp-2, accp+21, strand="-")
            sdn = fa.fetch(ch, donp-6, donp+3, strand="-")
        if sa and len(sa) == 23:
            tot += 1
            if sa[18:20].upper() == "AG": ag += 1
            try: a3.append(float(score3(sa.upper())))
            except Exception: pass
        if sdn and len(sdn) == 9:
            if sdn[3:5].upper() == "GT": gt += 1
            try: d5.append(float(score5(sdn.upper())))
            except Exception: pass
print(f"internal exons checked: {tot}")
print(f"acceptor ends in AG: {ag}/{tot} = {ag/max(tot,1):.1%}")
print(f"donor has GT at 4-5: {gt}/{tot} = {gt/max(tot,1):.1%}")
print(f"median 3'ss = {np.median(a3):.2f}   median 5'ss = {np.median(d5):.2f}")
print("EXPECT: AG ~99%, GT ~99%, both medians ~8-10 if coordinates are correct")