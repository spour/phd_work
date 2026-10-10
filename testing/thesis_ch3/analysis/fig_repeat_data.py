#!/usr/bin/env python3
# tidy tables for the repeat figure
import os, csv
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = [r for r in csv.reader(open(os.path.join(ROOT, "data/repeat_deep.tsv")), delimiter="\t")]
short = lambda g: g.replace("competitor, ", "")
with open(os.path.join(ROOT, "data/figr_pas.tsv"), "w") as fh:
    fh.write("group\tfeature\tvalue\n")
    for r in R:
        if r[0] == "elements" and r[2] in ("canonical PAS (AATAAA or ATTAAA) in -35 to -10", "TGTA in -100 to -20"):
            fh.write(f"{short(r[1])}\t{'AAUAAA or AUUAAA at -35 to -10' if 'PAS' in r[2] else 'mean UGUA copies, -100 to -20'}\t{r[3]}\n")
with open(os.path.join(ROOT, "data/figr_structure.tsv"), "w") as fh:
    fh.write("group\tpas_unpaired\tmfe_per_nt\n")
    for r in R:
        if r[0] == "structure":
            fh.write(f"{short(r[1])}\t{r[6].split()[3]}\t{r[3].split()[3]}\n")
with open(os.path.join(ROOT, "data/figr_usage.tsv"), "w") as fh:
    fh.write("site\tn\tfraction\n")
    for r in R:
        if r[0] == "usage": fh.write(f"{'in a repeat' if r[1].startswith('site in repeat') else 'not in a repeat'}\t{r[2].split()[1]}\t{r[3].split()[-1]}\n")
S = {r["group"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data/repeat_shuffle.tsv")), delimiter="\t")}
with open(os.path.join(ROOT, "data/figr_network.tsv"), "w") as fh:
    fh.write("group\tcentred_logit\tshuffle_delta\tlo\thi\n")
    for r in R:
        if r[0] == "network" and not r[1].startswith("real"):
            g = short(r[1]); v = r[2].split()[-1]
            key = {"L1 antisense": "competitor, L1 antisense", "L1 sense": "competitor, L1 sense", "Alu": "competitor, Alu", "no repeat": "competitor, no repeat",
                   "DNA transposon": "competitor, DNA transposon"}.get(g)
            s = S.get(key); fh.write(f"{g}\t{v}\t{s['delta_logit'] if s else 'NA'}\t{s['lo'] if s else 'NA'}\t{s['hi'] if s else 'NA'}\n")
    for k in ("competitor, LTR sense", "competitor, LTR antisense"):
        fh.write(f"{short(k)}\tNA\t{S[k]['delta_logit']}\t{S[k]['lo']}\t{S[k]['hi']}\n")
    fh.write(f"real 3' end\tNA\t{S[chr(114)+'eal 3' + chr(39) + ' end']['delta_logit']}\t{S[chr(114)+'eal 3' + chr(39) + ' end']['lo']}\t{S[chr(114)+'eal 3' + chr(39) + ' end']['hi']}\n")
with open(os.path.join(ROOT, "data/figr_positions.tsv"), "w") as fh:
    fh.write("element\tdecile\tsites\tbackground\tn\n")
    for r in csv.DictReader(open(os.path.join(ROOT, "data/repeat_site_positions.tsv")), delimiter="\t"):
        s = r["site_deciles"].split(); b = r["background_deciles"].split()
        for k in range(10): fh.write(f"{r['family']}, {r['orientation']}\t{k}\t{s[k]}\t{b[k]}\t{r['n']}\n")
print("[out] figr tables")
