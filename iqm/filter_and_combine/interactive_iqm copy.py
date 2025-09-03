import csv
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Sequence, Optional, Any

__all__ = [
    "read_metric_csv",
    "select_best_results_by_event",
    "get_metric_ranges",
    "filter_rows",
    "create_combined_metric",
    "filter_and_combine",
    "write_filtered_csv",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Columns that are expected to be numeric and should be parsed as floats.
DEFAULT_NUMERIC_METRICS: Sequence[str] = (
    "weighted_rmsd",
    "fraction_outliers",
    "length_deviation",
    "angle_deviation",
    "peak_ratio",
    "percentage_unindexed",
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _open_csv(path: Path | str, mode: str = "r"):  # pragma: no cover
    """Tiny wrapper that always opens CSVs with UTF‑8 and newline=""."""
    return open(path, mode, encoding="utf-8", newline="")


# ---------------------------------------------------------------------------
# 1. Reading
# ---------------------------------------------------------------------------

def read_metric_csv(
    csv_path: str | Path,
    *,
    numeric_metrics: Sequence[str] | None = None,
    group_by_event: bool = True,
) -> Dict[str, List[dict]] | List[dict]:
    """Read a metrics CSV and optionally group rows by their *event_number*.

    Parameters
    ----------
    csv_path
        Source CSV file.
    numeric_metrics
        Columns to parse as ``float``.  If *None*, uses :data:`DEFAULT_NUMERIC_METRICS`.
    group_by_event
        When *True* (default) return ``{event_id: [rows...]}``.  Otherwise, return
        a flat :class:`list`.
    """
    numeric_metrics = numeric_metrics or DEFAULT_NUMERIC_METRICS
    rows: List[dict[str, Any]] = []

    with _open_csv(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            # Skip junk header‑like rows sometimes found in the middle of the file
            if raw.get("stream_file", "").startswith("Event number:"):
                continue

            event_id = raw.get("event_number")  # keep as string (e.g. "0-1")
            if event_id is None:
                # Essential column missing ⇒ skip this row
                continue

            try:
                # Build a cleaned row dict with parsed numbers
                row: dict[str, Any] = dict(raw)
                row["event_number"] = event_id
                for col in numeric_metrics:
                    row[col] = float(raw[col])
            except (ValueError, KeyError):
                # Any parsing failure ⇒ row is unusable
                continue

            rows.append(row)

    if not group_by_event:
        return rows

    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        grouped.setdefault(r["event_number"], []).append(r)
    return grouped


# ---------------------------------------------------------------------------
# 2. Processing helpers
# ---------------------------------------------------------------------------

def select_best_results_by_event(
    grouped_data: Dict[str, List[dict]],
    *,
    sort_metric: str = "weighted_rmsd",
) -> List[dict]:
    """Return the *best* row (lowest *sort_metric*) for every event."""
    best = [min(lst, key=lambda r: r[sort_metric]) for lst in grouped_data.values() if lst]
    return best


def get_metric_ranges(
    rows: Iterable[dict],
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute *min* and *max* for each requested metric."""
    metrics = metrics or DEFAULT_NUMERIC_METRICS
    ranges: Dict[str, Tuple[float, float]] = {}
    for m in metrics:
        vals = [r[m] for r in rows if m in r]
        ranges[m] = (min(vals), max(vals)) if vals else (0.0, 1.0)
    return ranges


# ---------------------------------------------------------------------------
# 3. Filtering utilities
# ---------------------------------------------------------------------------

def _row_passes(r: dict, thresholds: Dict[str, float]) -> bool:
    return all((m in r and r[m] <= thr) for m, thr in thresholds.items())


def filter_rows(rows: Iterable[dict], thresholds: Dict[str, float]) -> List[dict]:
    """Return only the rows satisfying *every* ``metric ≤ threshold`` condition."""
    return [r for r in rows if _row_passes(r, thresholds)]


# ---------------------------------------------------------------------------
# 4. Combining + *new* convenience wrapper
# ---------------------------------------------------------------------------

def create_combined_metric(
    rows: Iterable[dict],
    metrics_to_combine: Sequence[str],
    weights: Sequence[float],
    *,
    new_metric_name: str = "combined_metric",
) -> None:
    """Add a weighted‑sum metric *in‑place* to every row.

    Assumes *metrics_to_combine* and *weights* are the same length.
    """
    if len(metrics_to_combine) != len(weights):
        raise ValueError("metrics_to_combine and weights must have the same length")

    for r in rows:
        r[new_metric_name] = sum(r[m] * w for m, w in zip(metrics_to_combine, weights))


def filter_and_combine(
    rows: Iterable[dict],
    *,
    pre_filter: Optional[Dict[str, float]] = None,
    metrics_to_combine: Sequence[str],
    weights: Sequence[float],
    new_metric_name: str = "combined_metric",
) -> List[dict]:
    """Convenience one‑stop helper.

    1. *Optionally* filters *rows* according to *pre_filter* thresholds *before* any
       combination is attempted.
    2. Adds a weighted‑sum metric (**in‑place**) named *new_metric_name*.
    3. Returns **only** the rows that survived the pre‑filter.
    """
    surviving = filter_rows(rows, pre_filter) if pre_filter else list(rows)
    create_combined_metric(surviving, metrics_to_combine, weights, new_metric_name=new_metric_name)
    return surviving


# ---------------------------------------------------------------------------
# 5. Writing
# ---------------------------------------------------------------------------

def write_filtered_csv(
    rows: Sequence[dict],
    output_csv_path: str | Path,
    *,
    metrics_to_write: Optional[Sequence[str]] = None,
) -> None:
    """Write *rows* to *output_csv_path* (UTF‑8 CSV).

    If *metrics_to_write* is *None*, uses ``rows[0].keys()`` as header.
    """
    output_csv_path = Path(output_csv_path)

    if not rows:
        output_csv_path.write_text("No data\n", encoding="utf-8")
        print(f"[metric_tools] No rows to write. Created empty CSV at {output_csv_path}")
        return

    metrics_to_write = list(metrics_to_write or rows[0].keys())

    with _open_csv(output_csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_to_write)
        writer.writeheader()
        for r in rows:
            writer.writerow({m: r.get(m, "") for m in metrics_to_write})

    print(f"[metric_tools] Wrote {len(rows)} rows → {output_csv_path}")


# ---------------------------------------------------------------------------
# 6. Quick CLI demo (python metric_tools_improved.py input.csv output.csv)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse, sys

    p = argparse.ArgumentParser(description="Quick filter‑and‑combine pipeline")
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--weighted-rmsd-max", type=float, default=1.0)
    p.add_argument("--peak-ratio-max", type=float, default=0.2)
    args = p.parse_args()

    grouped = read_metric_csv(args.input_csv, group_by_event=True)
    best = select_best_results_by_event(grouped)

    # Example: pre‑filter first, then combine two metrics 70/30
    filtered = filter_and_combine(
        best,
        pre_filter={
            "weighted_rmsd": args.weighted_rmsd_max,
            "peak_ratio": args.peak_ratio_max,
        },
        metrics_to_combine=["weighted_rmsd", "peak_ratio"],
        weights=[0.7, 0.3],
    )

    write_filtered_csv(filtered, args.output_csv)
