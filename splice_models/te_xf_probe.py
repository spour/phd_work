import sys
sys.path.insert(0, "scripts")
from extract_splicing_features import GenomeFasta, init_maxentscan
import splice_extra_features as XF

UP, DN = 3000, 1000
fa = GenomeFasta("../data/hg38.fa")
_, score3, _, m3 = init_maxentscan()

hdr = open("../data/te_dataset/pos_train.fa").readline().strip()
parts = hdr[1:].split("|"); loc = parts[1].split(":")
chrom, center, strand = loc[0], int(loc[1]), loc[2]
if strand == "+": w0, w1 = center-UP, center+DN
else:             w0, w1 = center-DN, center+UP
seq = fa.fetch(chrom, w0, w1, strand=strand)
print("seq len:", None if seq is None else len(seq), "| header:", hdr)

acc = XF._find_acceptor(seq, UP, score3=score3, m3=m3)
print("_find_acceptor ->", acc)

for name, fn in [
    ("branchpoint_score",   lambda: XF.branchpoint_score(seq, UP, acc_pos=acc, score3=score3, m3=m3)),
    ("u2af65_ppt",          lambda: XF.u2af65_ppt(seq, UP, acc_pos=acc, score3=score3, m3=m3)),
    ("u2af35_ag",           lambda: XF.u2af35_ag(seq, UP, acc_pos=acc)),
    ("u1_suppression_down", lambda: XF.u1_suppression_down(seq, UP, DN)),
]:
    try:
        print(f"{name:22} -> {fn()}")
    except Exception as e:
        print(f"{name:22} -> RAISED {type(e).__name__}: {e}")
