#!/usr/bin/env python3
# step 0: turn vastdb event_info/event_metrics
import gzip, sys, re
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

INFO = "EVENT_INFO-hg38.tab.gz"
METR = "EVENT_METRICS-hg38.tab.gz"
RMIN = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
PAD  = 300
OUT  = sys.argv[2] if len(sys.argv) > 2 else "as_events.tsv"

CO = re.compile(r'^(chr[^:]+):(\d+)-(\d+)$')
def iv(s):
    m = CO.match(s.strip()) if s and s != "NA" else None
    return None if m is None else (m.group(1), int(m.group(2)) - 1, int(m.group(3)))

def etype_of(e):
    for t in ("HsaEX", "HsaINT", "HsaALTA", "HsaALTD"):
        if e.startswith(t): return t[3:]
    return None

# metrics
var, seen = set(), defaultdict(int)
grange = {}
tot_t, var_t = defaultdict(int), defaultdict(int)
with gzip.open(METR, "rt") as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) < 5: continue
        e = f[0]; t = etype_of(e)
        if t is None: continue
        try: avg, mn, mx, rg = (float(x) for x in f[1:5])
        except ValueError: continue
        grp = e.split("-")[0]
        seen[grp] += 1
        grange[grp] = max(grange.get(grp, -1.0), rg)
        tot_t[t] += 1
        if rg >= RMIN and mx >= 10.0 and mn <= 90.0:
            var.add(grp); var_t[t] += 1
log(f"[metrics] variability rule: Range >= {RMIN} PSI AND Max >= 10 AND Min <= 90")
log(f"[metrics] rows {sum(tot_t.values()):,}  groups {len(seen):,}  variable groups {len(var):,}")
for t in ("EX", "INT", "ALTA", "ALTD"):
    log(f"[metrics]   {t:5s} rows {tot_t[t]:8,}  variable rows {var_t[t]:8,}")

# info
STR = re.compile(r':([+-])\s*$')
alt_seg = defaultdict(list)
alt_meta = {}
rows = []
n_tot = defaultdict(int); n_var = defaultdict(int); n_nocoord = defaultdict(int)
with gzip.open(INFO, "rt") as fh:
    hdr = next(fh).rstrip("\n").split("\t")
    ix = {c: i for i, c in enumerate(hdr)}
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) < 11: continue
        gene, ev = f[ix["GENE"]], f[ix["EVENT"]]
        t = etype_of(ev)
        if t is None: continue
        n_tot[t] += 1
        grp = ev.split("-")[0]
        if grp not in seen: continue
        n_var[t] += (grp in var)
        m = STR.search(f[ix["REF_CO"]]) or STR.search(f[ix["FULL_CO"]])
        sd = m.group(1) if m else None
        if sd is None: n_nocoord[t] += 1; continue
        if t == "INT":
            a = iv(f[ix["COORD_o"]])
            if a: rows.append((a[0], a[1], a[2], sd, "INT", grp, gene, int(grp in var), grange.get(grp, -1.0)))
            else: n_nocoord[t] += 1
        elif t == "EX":
            a = iv(f[ix["COORD_o"]])
            if a: rows.append((a[0], a[1], a[2], sd, "EX", grp, gene, int(grp in var), grange.get(grp, -1.0)))
            c1, aa, c2 = (iv(f[ix[k]]) for k in ("CO_C1", "CO_A", "CO_C2"))
            segs = [x for x in (c1, aa, c2) if x]
            if len(segs) == 3 and len({s[0] for s in segs}) == 1:
                segs.sort(key=lambda s: s[1])
                for p, q in zip(segs, segs[1:]):
                    if q[1] > p[2]:
                        rows.append((p[0], p[2], q[1], sd, "EXFLANK", grp, gene, int(grp in var), grange.get(grp, -1.0)))
            elif a is None: n_nocoord[t] += 1
        else:
            aa = iv(f[ix["CO_A"]])
            if aa: alt_seg[grp].append(aa)
            alt_meta[grp] = (t, sd, gene)

for grp, segs in alt_seg.items():
    t, sd, gene = alt_meta[grp]
    c = segs[0][0]
    if any(s[0] != c for s in segs): continue
    lo, hi = min(s[1] for s in segs), max(s[2] for s in segs)
    v = int(grp in var); rg = grange.get(grp, -1.0)
    rows.append((c, lo, hi, sd, t, grp, gene, v, rg))
    rows.append((c, max(0, lo - PAD), hi + PAD, sd, t + "_W", grp, gene, v, rg))

log("[info] events in EVENT_INFO / of which in a variable group / unusable coords:")
for t in ("EX", "INT", "ALTA", "ALTD"):
    log(f"[info]   {t:5s} {n_tot[t]:8,} {n_var[t]:8,} {n_nocoord[t]:6,}")

by_t = defaultdict(set)
for r in rows: by_t[r[4]].add(r[5])
log("[out] intervals written, by emitted region type (unique event groups in brackets):")
cnt = defaultdict(int); cntv = defaultdict(int)
for r in rows:
    cnt[r[4]] += 1; cntv[r[4]] += r[7]
with open(OUT, "w") as o:
    o.write("chrom\tstart0\tend0\tstrand\tetype\tevent\tgene\tvariable\trange\n")
    for r in sorted(rows):
        o.write("\t".join(map(str, r)) + "\n")
for t in sorted(cnt):
    log(f"[out]   {t:9s} {cnt[t]:8,} intervals ({cntv[t]:,} variable)  [{len(by_t[t]):,} groups]")
log(f"[out] {OUT}: {len(rows):,} intervals")
