#!/usr/bin/env python3
# merge sharded shortlist outputs into one and reprint the global summary
import glob, sys, numpy as np, pandas as pd
stem=sys.argv[1] if len(sys.argv)>1 else "shortlist"
wins=sorted(glob.glob(f"{stem}_windows.*.parquet"))
sums=sorted(glob.glob(f"{stem}_summary.*.tsv"))
if not wins: sys.exit(f"no shards matching {stem}_windows.*.parquet")
R=pd.concat([pd.read_parquet(f) for f in wins],ignore_index=True)
S=pd.concat([pd.read_csv(f,sep="\t") for f in sums],ignore_index=True)
R.to_parquet(f"{stem}_windows.parquet",index=False)
S.to_csv(f"{stem}_summary.tsv",sep="\t",index=False)
contested=S[S.contested==1]
print(f"[merged] {len(R):,} windows, {len(S):,} genes from {len(wins)} shards")
print(f"[contest] {len(contested):,}/{len(S):,} contested ({len(contested)/len(S):.1%}); "
      f"mean shortlist (contested)={contested.n_shortlist.mean():.2f} "
      f"mean competitors={contested.n_internal.mean():.2f}")
for lab,lo,hi in [("1",1,1),("2",2,2),("3-4",3,4),("5-8",5,8),("9+",9,10**9)]:
    n=int(((S.n_shortlist>=lo)&(S.n_shortlist<=hi)).sum())
    print(f"    {lab:<5} {n:>6}  {n/len(S):.3f}")
print(f"[floor] random-tie-break P@1: overall={float((1.0/S.n_shortlist).mean()):.3f}  "
      f"contested-only={float((1.0/contested.n_shortlist).mean()):.3f}")
print(f"[out] {stem}_windows.parquet, {stem}_summary.tsv")