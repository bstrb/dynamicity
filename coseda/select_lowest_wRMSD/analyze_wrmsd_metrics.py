#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_wrmsd_metrics.py

Read JSONL.GZ metrics produced by the 'calc' stage and summarize:
- total keys seen (img_base,event)
- resolved keys (≥1 valid wRMSD across any stream)
- unresolved keys (never had a valid wRMSD), with breakdown by 'reason'
- optional listing/export of unresolved keys
- optional grouping by image base (how many unresolved per image)

Usage examples:
  python3 analyze_wrmsd_metrics.py --metrics-dir DIR/metrics
  python3 analyze_wrmsd_metrics.py --metrics-dir DIR/metrics --showexamples 10
  python3 analyze_wrmsd_metrics.py --metrics-dir DIR/metrics --export-csv unresolved.csv
  python3 analyze_wrmsd_metrics.py --metrics-dir DIR/metrics --group-by-image
  python3 analyze_wrmsd_metrics.py --metrics-dir DIR/metrics --filter-reason no_matches --list-unresolved
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Optional

Reason = str
Key = Tuple[str, str]  # (img_base, event)

def iter_metric_files(metrics_dir: Path) -> Iterable[Path]:
    for p in sorted(metrics_dir.iterdir()):
        # accept *.jsonl.gz (strict)
        if p.is_file() and p.name.endswith(".jsonl.gz"):
            yield p

def analyze(metrics_dir: Path):
    if not metrics_dir.is_dir():
        raise SystemExit(f"ERROR: metrics dir not found: {metrics_dir}")

    seen_keys: set[Key] = set()
    resolved: set[Key] = set()        # at least one record had numeric wrmsd
    reasons_per_key: Dict[Key, Counter] = defaultdict(Counter)

    # Optional diagnostics
    has_refl_any: Dict[Key, bool] = defaultdict(bool)
    has_matches_any: Dict[Key, bool] = defaultdict(bool)
    kept_gt0_any: Dict[Key, bool] = defaultdict(bool)

    files = list(iter_metric_files(metrics_dir))
    if not files:
        raise SystemExit(f"ERROR: no *.jsonl.gz found in {metrics_dir}")

    for mf in files:
        try:
            with gzip.open(mf, "rt", encoding="utf-8") as gz:
                for line in gz:
                    if not line.strip():
                        continue
                    rec = json.loads(line)

                    img_base = str(rec.get("img_base"))
                    event = str(rec.get("event"))
                    key: Key = (img_base, event)
                    seen_keys.add(key)

                    wrmsd = rec.get("wrmsd", None)
                    n_refl = int(rec.get("n_refl", 0))
                    n_matches = int(rec.get("n_matches", 0))
                    n_kept = int(rec.get("n_kept", 0))
                    reason = rec.get("reason", None)

                    # Diagnostics
                    if n_refl > 0:
                        has_refl_any[key] = True
                    if n_matches > 0:
                        has_matches_any[key] = True
                    if n_kept > 0:
                        kept_gt0_any[key] = True

                    if wrmsd is not None:
                        resolved.add(key)
                    else:
                        if reason is None:
                            reason = "unknown"
                        reasons_per_key[key][reason] += 1
        except Exception as e:
            print(f"[WARN] Failed reading {mf.name}: {e}")

    unresolved = seen_keys - resolved
    # For each unresolved key, pick the most common reason we observed across its attempts
    reason_of_key: Dict[Key, Reason] = {}
    for k in unresolved:
        if reasons_per_key[k]:
            reason_of_key[k] = reasons_per_key[k].most_common(1)[0][0]
        else:
            reason_of_key[k] = "unknown"

    counts_by_reason = Counter(reason_of_key.values())

    summary = {
        "metric_files": len(files),
        "keys_seen": len(seen_keys),
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "counts_by_reason": counts_by_reason,
        "diag": {
            # proportions among unresolved, for quick sanity checks
            "unresolved_with_any_reflections": sum(1 for k in unresolved if has_refl_any[k]),
            "unresolved_with_any_matches": sum(1 for k in unresolved if has_matches_any[k]),
            "unresolved_with_any_kept_after_clipping": sum(1 for k in unresolved if kept_gt0_any[k]),
        }
    }
    return summary, unresolved, reason_of_key

def main():
    ap = argparse.ArgumentParser(description="Analyze wRMSD metrics and report unresolved keys.")
    ap.add_argument("--metrics-dir", type=Path, required=True, help="Directory with *.jsonl.gz metrics files")
    ap.add_argument("--showexamples", type=int, default=0, help="Show up to N example unresolved keys per reason")
    ap.add_argument("--list-unresolved", action="store_true", help="List all unresolved keys")
    ap.add_argument("--filter-reason", type=str, default=None, help="If set, restrict listings to this reason")
    ap.add_argument("--export-csv", type=Path, default=None, help="Write unresolved keys to CSV")
    ap.add_argument("--group-by-image", action="store_true", help="Print unresolved count per img_base")
    args = ap.parse_args()

    summary, unresolved, reason_of_key = analyze(args.metrics_dir)

    # --- Print summary ---
    print("=== Summary ===")
    print(f"Metric files               : {summary['metric_files']}")
    print(f"Keys seen (img,event)      : {summary['keys_seen']}")
    print(f"Resolved (have valid wRMSD): {summary['resolved']}")
    print(f"Unresolved                 : {summary['unresolved']}")
    print("By reason:")
    for reason, cnt in summary["counts_by_reason"].most_common():
        print(f"  {reason:24s} {cnt}")

    diag = summary["diag"]
    if summary["unresolved"] > 0:
        print("\nDiagnostics among UNRESOLVED keys:")
        print(f"  with any reflections present : {diag['unresolved_with_any_reflections']}")
        print(f"  with any matches within radius: {diag['unresolved_with_any_matches']}")
        print(f"  with any kept after clipping  : {diag['unresolved_with_any_kept_after_clipping']}")

    # --- Optional: group by image base ---
    if args.group_by_image and unresolved:
        by_img = Counter(k[0] for k in unresolved)
        print("\nUnresolved counts per image (top 30):")
        for img, cnt in by_img.most_common(30):
            print(f"  {img}: {cnt}")

    # Prepare filtered list if needed
    items: List[Tuple[str, str, str]] = []
    for (img, ev) in sorted(unresolved):
        reason = reason_of_key[(img, ev)]
        if args.filter_reason and reason != args.filter_reason:
            continue
        items.append((img, ev, reason))

    # --- List examples per reason ---
    if args.showexamples > 0:
        print(f"\nExamples per reason (up to {args.showexamples} each):")
        # bucket by reason
        bucket: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
        for img, ev, reason in items:
            if len(bucket[reason]) < args.showexamples:
                bucket[reason].append((img, ev))
        for reason, pairs in bucket.items():
            print(f"  [{reason}]")
            for img, ev in pairs:
                print(f"    {img}  {ev}")

    # --- List all unresolved (filtered or not) ---
    if args.list_unresolved and items:
        print("\nAll unresolved keys:")
        for img, ev, reason in items:
            print(f"{img},{ev},{reason}")

    # --- Export CSV ---
    if args.export_csv:
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.export_csv.open("w", encoding="utf-8") as f:
            f.write("img_base,event,reason\n")
            for img, ev, reason in items:
                f.write(f"{img},{ev},{reason}\n")
        print(f"\nWrote CSV: {args.export_csv}")

if __name__ == "__main__":
    main()
