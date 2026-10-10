#!/usr/bin/env python3
# tidy tables for the mechanism figure from data/mechanism.tsv (py/mechanism.py)
import os, re
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "data/mechanism.tsv"))]
num = lambda s: [float(x) for x in re.findall(r"[+-]?\d+\.\d+", s)]
with open(os.path.join(ROOT, "data/figm_knockout.tsv"), "w") as fh:
    fh.write("experiment\tsize\tbin\tn\tdiff\tlo\thi\n")
    for r in R:
        if "knockout in" not in r[0]: continue
        exp = "terminal-exon 3' splice site" if "3' splice" in r[0] else "upstream 5' splice site"
        size = re.search(r"(\d+)-mer", r[0]).group(1) + " nt"; b = r[1]
        n = int(r[2].split()[1]); v = num([x for x in r if x.startswith("difference")][0])
        fh.write(f"{exp}\t{size}\t{b}\t{n}\t{v[0]}\t{v[1]}\t{v[2]}\n")
with open(os.path.join(ROOT, "data/figm_architecture.tsv"), "w") as fh:
    fh.write("donor_offset\tarrangement\tdonor_alone\tboth\tinteraction\tlo\thi\n")
    for r in R:
        if r[0] != "architecture" or not r[1].startswith("donor at"): continue
        d = int(r[1].split()[-1]); arr = "internal exon" if "downstream" in r[2] else "intron, then terminal exon"
        v = num(r[5]); fh.write(f"{d}\t{arr}\t{num(r[3])[0]}\t{num(r[4])[0]}\t{v[0]}\t{v[1]}\t{v[2]}\n")
with open(os.path.join(ROOT, "data/figm_scan.tsv"), "w") as fh:
    fh.write("element\toffset\tdiff\tlo\thi\n")
    for r in R:
        if r[0] != "scan" or not r[2].startswith("offset"): continue
        v = num(r[3]); fh.write(f"{r[1]}\t{int(r[2].split()[1])}\t{v[0]}\t{v[1]}\t{v[2]}\n")
print("[out] figm tables")
