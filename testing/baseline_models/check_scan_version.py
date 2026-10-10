#!/usr/bin/env python3
# report scan progress across per-gene parquets
import argparse
import glob
import os
from collections import Counter

import pyarrow.parquet as pq


# marker spec parser

def parse_marker_spec(spec):
    if "=" in spec:
        col, val = spec.split("=", 1)
        return col.strip(), val.strip()
    return spec.strip(), None


DEFAULT_MARKERS = [
    ("cnn_scan_version",      "v2"),
    ("aparent2_scan_version", "aparent2_v1"),
]


# per-parquet inspection

def inspect_parquet(path, markers, extra_cols):
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        return {"error": str(e), "cols": set(),
                "marker_status": {}, "extras": {}}

    cols = set(pf.schema_arrow.names)
    out = {"error": None, "cols": cols,
           "marker_status": {}, "extras": {}}

    cols_to_read = [c for (c, _) in markers if c in cols]
    table = None
    if cols_to_read:
        try:
            table = pf.read_row_group(0, columns=cols_to_read)
        except Exception:
            table = None

    for col, expected in markers:
        if col not in cols:
            out["marker_status"][col] = ("missing_col", None)
            continue
        if table is None:
            out["marker_status"][col] = ("?", None)
            continue
        try:
            seen = table.column(col)[0].as_py()
            seen_s = str(seen) if seen is not None else None
        except Exception:
            seen_s = None

        if seen_s is None:
            out["marker_status"][col] = ("todo", None)
        elif expected is None:
            out["marker_status"][col] = ("done", seen_s)
        elif seen_s == expected:
            out["marker_status"][col] = ("done", seen_s)
        else:
            out["marker_status"][col] = ("wrong_version", seen_s)

    for col in extra_cols:
        out["extras"][col] = col in cols

    return out


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--markers", nargs="+", default=None,
                    help="One or more 'column=value' specs (or just 'column' "
                         "to accept any non-null). Default: "
                         "cnn_scan_version=v2  aparent2_scan_version=aparent2_v1")
    ap.add_argument("--extra_cols", nargs="+", default=None,
                    help="Additional column names to report presence of "
                         "(default: cnn_prob_v1 cnn_logit aparent2_logit "
                         "aparent2_cleavage_sum)")
    ap.add_argument("--gene", nargs="+", default=None,
                    help="Inspect only these gene names")
    ap.add_argument("--write_todo", default=None,
                    help="Write genes missing the FIRST marker to this file")
    ap.add_argument("--write_todo_marker", default=None,
                    help="If multiple markers are given, write the TODO list "
                         "for this specific marker (col name) instead of the first.")
    ap.add_argument("--show_each", action="store_true",
                    help="Print one line per parquet")
    args = ap.parse_args()

    if args.markers:
        markers = [parse_marker_spec(s) for s in args.markers]
    else:
        markers = list(DEFAULT_MARKERS)

    extra_cols = (args.extra_cols if args.extra_cols is not None else
                   ["cnn_prob_v1", "cnn_logit",
                    "aparent2_logit", "aparent2_cleavage_sum"])

    per_gene = os.path.join(args.scan_dir, "per_gene")
    parquets = sorted(glob.glob(os.path.join(per_gene, "*.parquet")))
    if args.gene:
        keep = set(args.gene)
        parquets = [p for p in parquets
                    if os.path.basename(p).replace(".parquet", "") in keep]
    if not parquets:
        raise SystemExit(f"No parquets found under {per_gene}")

    n = len(parquets)
    print(f"Inspecting {n:,} parquets in {per_gene}")
    print(f"Markers being checked:")
    for col, val in markers:
        v_disp = repr(val) if val is not None else "(any non-null)"
        print(f"  {col:<24} target={v_disp}")
    print(f"Extra columns reported:")
    print(f"  {', '.join(extra_cols)}\n")

    marker_counters = {col: Counter() for (col, _) in markers}
    extra_counters = {c: 0 for c in extra_cols}
    todo_lists = {col: [] for (col, _) in markers}
    n_errors = 0

    for i, path in enumerate(parquets):
        gene = os.path.basename(path).replace(".parquet", "")
        info = inspect_parquet(path, markers, extra_cols)
        if info["error"]:
            n_errors += 1
            if args.show_each:
                print(f"  {gene:<20} ERROR: {info['error']}")
            continue

        for col, _ in markers:
            status, _seen = info["marker_status"][col]
            marker_counters[col][status] += 1
            if status != "done":
                todo_lists[col].append(gene)

        for c in extra_cols:
            if info["extras"].get(c):
                extra_counters[c] += 1

        if args.show_each:
            parts = []
            for col, _ in markers:
                status, seen = info["marker_status"][col]
                tag = "OK  " if status == "done" else "TODO"
                seen_disp = (f"={seen}" if seen is not None
                              and status == "done" else "")
                parts.append(f"{col.split('_')[0]:>8}:{tag}{seen_disp}")
            extras = " ".join(c for c in extra_cols if info["extras"].get(c))
            print(f"  {gene:<20} " + "  ".join(parts) +
                  (f"   ({extras})" if extras else ""))

        if (i + 1) % 2000 == 0 and not args.show_each:
            print(f"  ...inspected {i+1:,}/{n:,}")

    # summary
    print(f"\n========= Summary ({n:,} parquets) =========")
    if n_errors:
        print(f"  {n_errors:,} parquets had read errors\n")

    for col, expected in markers:
        c = marker_counters[col]
        n_done    = c.get("done", 0)
        n_todo    = c.get("todo", 0)
        n_miss    = c.get("missing_col", 0)
        n_wrong   = c.get("wrong_version", 0)
        n_unknown = c.get("?", 0)
        exp_disp = expected if expected is not None else "(any non-null)"
        print(f"--- {col} (target {exp_disp}) ---")
        print(f"  done           {n_done:>6,}  ({n_done/n*100:5.1f}%)")
        print(f"  todo           "
              f"{n_todo + n_miss + n_wrong + n_unknown:>6,}  "
              f"({(n_todo+n_miss+n_wrong+n_unknown)/n*100:5.1f}%)")
        if n_miss:
            print(f"    column missing entirely: {n_miss:,}")
        if n_todo:
            print(f"    column present but null: {n_todo:,}")
        if n_wrong:
            print(f"    wrong version:           {n_wrong:,}")
        if n_unknown:
            print(f"    couldn't read:           {n_unknown:,}")
        print()

    print(f"--- Auxiliary columns ---")
    for c in extra_cols:
        v = extra_counters[c]
        print(f"  {c:<24} {v:>6,}  ({v/n*100:5.1f}%)")
    print()

    if args.write_todo:
        target_col = args.write_todo_marker or markers[0][0]
        if target_col not in todo_lists:
            print(f"ERROR: marker {target_col!r} not in checked markers")
            return
        todo = todo_lists[target_col]
        if not todo:
            print(f"All genes done for {target_col}. Nothing written.")
        else:
            with open(args.write_todo, "w") as f:
                for g in todo:
                    f.write(g + "\n")
            print(f"Wrote TODO ({len(todo):,} genes for {target_col}) "
                  f"to {args.write_todo}")


if __name__ == "__main__":
    main()