#!/usr/bin/env python3
# interactive_iqm.py — compatible with both old flat CSV and new comment-chunk CSV

from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

__all__ = [
    "read_metric_csv",
    "select_best_results_by_event",
    "get_metric_ranges",
    "filter_rows",
    "create_combined_metric",
    "filter_and_combine",
    "write_filtered_csv",
]

# ----------------------------- Configuration -----------------------------

DEFAULT_NUMERIC_METRICS: Sequence[str] = (
    "weighted_rmsd",
    "fraction_outliers",
    "length_deviation",
    "angle_deviation",
    "peak_ratio",
    "percentage_unindexed",
)

HEADER_PREFIX = "stream_file,"  # header line in the new CSV

# Keep the original UI semantics: lower values are better for all sliders and for
# the legacy weighted-sum. (If you want peak_ratio to be “higher is better” for
# the combined metric, flip its sign in _combined_score().)
LOWER_IS_BETTER = {m: True for m in DEFAULT_NUMERIC_METRICS}


# ------------------------------- Utilities -------------------------------

def _open_csv(path: Path | str, mode: str = "r"):  # pragma: no cover
    return open(path, mode, encoding="utf-8", newline="")

def _to_float(x: str | float | None) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float):
        return x
    s = str(x).strip()
    if not s or s.upper() == "NA":
        return None
    try:
        return float(s)
    except Exception:
        return None

def _fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.6f}"

def _median(vals: List[float]) -> float:
    vals = sorted(vals)
    n = len(vals)
    m = n // 2
    return vals[m] if n % 2 else 0.5 * (vals[m - 1] + vals[m])

def _mad(vals: List[float], med: Optional[float] = None) -> float:
    if not vals:
        return 0.0
    med = _median(vals) if med is None else med
    return _median([abs(v - med) for v in vals])

def _quantiles(vals: List[float], lo=0.01, hi=0.99) -> Tuple[float, float]:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return s[0], s[0]
    def q(p):
        i = p * (n - 1)
        lo_i = int(math.floor(i))
        hi_i = int(math.ceil(i))
        if lo_i == hi_i:
            return s[lo_i]
        t = i - lo_i
        return s[lo_i] * (1 - t) + s[hi_i] * t
    return q(lo), q(hi)

def _winsorize(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# --------------------------- 1) Reading CSVs -----------------------------

def _read_flat_csv_with_event_column(
    csv_path: Path,
    numeric_metrics: Sequence[str],
    group_by_event: bool,
) -> Dict[str, List[dict]] | List[dict]:
    rows: List[dict[str, Any]] = []
    with _open_csv(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            # skip accidental “Event number:” lines
            if raw.get("stream_file", "").startswith("Event number:"):
                continue
            ev = raw.get("event_number")
            if ev is None:
                continue
            rec: dict[str, Any] = dict(raw)
            rec["event_number"] = ev
            for col in numeric_metrics:
                rec[col] = _to_float(raw.get(col))
            rows.append(rec)

    if not group_by_event:
        return rows
    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        grouped.setdefault(r["event_number"], []).append(r)
    return grouped


def _read_comment_chunk_csv(
    csv_path: Path,
    numeric_metrics: Sequence[str],
    group_by_event: bool,
) -> Dict[str, List[dict]] | List[dict]:
    """
    New format:
      stream_file,weighted_rmsd,...
      # Image filename: /abs/path
      # Event: //<event>
      <data rows...>
      # Image filename: ...
      # Event: ...
      <data rows...>
    """
    # Find header line
    header: Optional[List[str]] = None
    header_line_no: Optional[int] = None
    with csv_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith(HEADER_PREFIX):
                header = next(csv.reader([line.strip()]))
                header_line_no = i
                break
    if header is None:
        raise RuntimeError("Could not find CSV header starting with 'stream_file,'")

    name_to_idx = {name: idx for idx, name in enumerate(header)}
    if "stream_file" not in name_to_idx:
        raise RuntimeError("Header missing 'stream_file' column")

    rows: List[dict] = []
    cur_image: Optional[str] = None
    cur_event: Optional[str] = None
    seen_header_again = False

    with csv_path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            s = raw.rstrip("\n")
            if i == header_line_no:
                # skip the header row itself
                continue
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("# Image filename:"):
                    cur_image = s.split(":", 1)[1].strip()
                elif s.startswith("# Event:"):
                    cur_event = s.split("//", 1)[1].strip() if "//" in s else s.split(":", 1)[1].strip()
                continue
            # Defensive: some generators may repeat the header later
            if s.startswith(HEADER_PREFIX):
                seen_header_again = True
                continue

            data = next(csv.reader([s]))
            rec = {
                "image_path": cur_image or "",
                "event_number": cur_event or "",
                "stream_file": data[name_to_idx["stream_file"]],
            }
            for col in numeric_metrics:
                if col in name_to_idx and name_to_idx[col] < len(data):
                    rec[col] = _to_float(data[name_to_idx[col]])
                else:
                    rec[col] = None
            rows.append(rec)

    if not group_by_event:
        return rows
    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        grouped.setdefault(r["event_number"], []).append(r)
    return grouped

def read_metric_csv(
    csv_path: str | Path,
    *,
    numeric_metrics: Sequence[str] | None = None,
    group_by_event: bool = True,
) -> Dict[str, List[dict]] | List[dict]:
    """
    Load either the old flat CSV (with event_number column) or the new comment-chunk CSV.
    Also treat 'combined_metric' as numeric if present.
    """
    # include combined_metric in the numeric set
    base_numeric = list(numeric_metrics or DEFAULT_NUMERIC_METRICS)
    if "combined_metric" not in base_numeric:
        base_numeric.append("combined_metric")

    csv_path = Path(csv_path)

    # Fast sniff: does the file have a DictReader header with event_number?
    with _open_csv(csv_path, "r") as f:
        head = f.readline()
        if head and "event_number" in head and "stream_file" in head:
            return _read_flat_csv_with_event_column(csv_path, base_numeric, group_by_event)

    # Else: parse the comment-chunk format
    return _read_comment_chunk_csv(csv_path, base_numeric, group_by_event)


# ---------------------- 2) Basic helpers for the UI ----------------------

def select_best_results_by_event(
    grouped_data: Dict[str, List[dict]],
    *,
    sort_metric: str = "weighted_rmsd",
) -> List[dict]:
    """Return the best (lowest) row by *sort_metric* for every event."""
    best = []
    for rows in grouped_data.values():
        # keep only rows where the metric can be read as a number
        rows2 = []
        for r in rows:
            v = _to_float(r.get(sort_metric))
            if v is not None:
                rr = dict(r)
                rr[sort_metric] = v
                rows2.append(rr)
        if rows2:
            best.append(min(rows2, key=lambda r: r[sort_metric]))
    return best

def get_metric_ranges(
    rows: Iterable[dict],
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute min/max for each requested metric (ignoring NA)."""
    metrics = metrics or DEFAULT_NUMERIC_METRICS
    ranges: Dict[str, Tuple[float, float]] = {}
    for m in metrics:
        vals = [r[m] for r in rows if r.get(m) is not None]
        ranges[m] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return ranges


# --------------------------- 3) Filtering logic --------------------------

def _row_passes(r: dict, thresholds: Dict[str, float]) -> bool:
    # Preserve the original UI's "≤" semantics for all metrics.
    for m, thr in thresholds.items():
        v = r.get(m)
        if v is None:
            return False
        if not (v <= thr):
            return False
    return True

def filter_rows(rows: Iterable[dict], thresholds: Dict[str, float]) -> List[dict]:
    """Return only the rows satisfying every `metric ≤ threshold` condition."""
    return [r for r in rows if _row_passes(r, thresholds)]


# --------- 4) Combined metric: robust normalization + weighted sum --------

def _robust_stats(rows: List[dict], metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """
    Median/MAD + 1–99% winsor bounds for each metric (on the *filtered* rows).
    """
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = [r[m] for r in rows if r.get(m) is not None]
        if not vals:
            out[m] = {"median": 0.0, "sigma": 1.0, "lo": 0.0, "hi": 0.0}
            continue
        med = _median(vals)
        mad = _mad(vals, med)
        sigma = 1.4826 * mad if mad > 0 else (max(1e-9, (max(vals) - min(vals)) / 6.0) if len(vals) > 1 else 1.0)
        lo, hi = _quantiles(vals, 0.01, 0.99)
        out[m] = {"median": med, "sigma": sigma, "lo": lo, "hi": hi}
    return out

def _combined_score(r: dict, metrics: Sequence[str], weights: Sequence[float], stats: Dict[str, Dict[str, float]]) -> float:
    """
    Build a *badness* score (lower is better):
      - Winsorize values to 1–99% bounds
      - Robust z-score: (v - median)/sigma
      - Lower-is-better across the board (keeps UI semantics)
      - Weighted mean across provided metrics
    """
    num = 0.0
    den = 0.0
    for m, w in zip(metrics, weights):
        if w == 0.0:
            continue
        v = r.get(m)
        if v is None:
            continue
        st = stats[m]
        v_w = _winsorize(v, st["lo"], st["hi"])
        z = (v_w - st["median"]) / st["sigma"] if st["sigma"] > 0 else 0.0
        # If you want to treat some metrics as higher-better, flip sign here when LOWER_IS_BETTER[m] is False.
        if not LOWER_IS_BETTER.get(m, True):
            z = -z
        num += w * z
        den += abs(w)
    return num / den if den > 0 else 0.0


def create_combined_metric(
    rows: Iterable[dict],
    metrics_to_combine: Sequence[str],
    weights: Sequence[float],
    *,
    new_metric_name: str = "combined_metric",
) -> None:
    """
    Legacy API: adds a *badness* combined metric in-place on the given rows,
    using robust z-scores under the hood (Median/MAD + winsorization).
    """
    rows = list(rows)
    stats = _robust_stats(rows, metrics_to_combine)
    for r in rows:
        r[new_metric_name] = _combined_score(r, metrics_to_combine, weights, stats)


def filter_and_combine(
    rows: Iterable[dict],
    *,
    pre_filter: Optional[Dict[str, float]] = None,
    metrics_to_combine: Sequence[str],
    weights: Sequence[float],
    new_metric_name: str = "combined_metric",
) -> List[dict]:
    """
    1) Apply pre_filter (metric ≤ threshold) in raw units.
    2) Compute robust normalized *badness* score (lower is better) as a weighted mean.
    3) Return surviving rows with the new metric added.
    """
    surviving = filter_rows(rows, pre_filter) if pre_filter else list(rows)
    if not surviving:
        return []
    stats = _robust_stats(surviving, metrics_to_combine)
    out: List[dict] = []
    for r in surviving:
        rr = dict(r)
        rr[new_metric_name] = _combined_score(rr, metrics_to_combine, weights, stats)
        out.append(rr)
    return out


# --------------------------------- 5) Writing ---------------------------------
def write_filtered_csv(
    rows: Sequence[dict],
    output_csv_path: str | Path,
    *,
    metrics_to_write: Optional[Sequence[str]] = None,
) -> None:
    """
    Write rows to CSV (UTF-8). If metrics_to_write is None, write a sensible default order.
    Coerces numeric fields (including 'combined_metric') to float if possible; otherwise writes the value verbatim.
    """
    output_csv_path = Path(output_csv_path)
    if not rows:
        output_csv_path.write_text("No data\n", encoding="utf-8")
        print(f"[metric_tools] No rows to write. Created empty CSV at {output_csv_path}")
        return

    # default column order
    default_cols = ["image_path", "event_number", "stream_file", *DEFAULT_NUMERIC_METRICS]
    if "combined_metric" in rows[0]:
        default_cols.append("combined_metric")
    cols = list(metrics_to_write or default_cols)

    with _open_csv(output_csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            out = {}
            for c in cols:
                if c in DEFAULT_NUMERIC_METRICS or c == "combined_metric":
                    fv = _to_float(r.get(c))
                    out[c] = _fmt(fv) if fv is not None else (r.get(c, ""))
                else:
                    out[c] = r.get(c, "")
            writer.writerow(out)
    print(f"[metric_tools] Wrote {len(rows)} rows → {output_csv_path}")

    # -------- Direction map (reuse/adjust as you like) --------
# True  => lower is better
# False => higher is better (will be inverted for “badness”)
LOWER_IS_BETTER.update({
    "weighted_rmsd": True,
    "fraction_outliers": True,
    "length_deviation": True,
    "angle_deviation": True,
    "peak_ratio": False,              # higher=better
    "percentage_unindexed": True,
})

# -------- Per-chunk normalization --------
def _norm_values(vals: list[float], *, method: str, robust: bool, winsor: tuple[float,float]) -> tuple[list[float], dict]:
    """Return normalized values (mean≈0, sd≈1 for z/robust_z; in [0,1] for minmax) + stats dict."""
    if not vals:
        return [], {"method": method}
    lo_q, hi_q = winsor
    if method == "minmax":
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            return [0.5 for _ in vals], {"vmin": vmin, "vmax": vmax, "method": method}
        return [ (v - vmin) / (vmax - vmin) for v in vals ], {"vmin": vmin, "vmax": vmax, "method": method}

    if robust:
        med = _median(vals)
        mad = _mad(vals, med)
        sigma = 1.4826*mad if mad > 0 else (max(1e-9, (max(vals)-min(vals))/6.0) if len(vals) > 1 else 1.0)
        lo, hi = _quantiles(vals, lo_q, hi_q)
        w = [_winsorize(v, lo, hi) for v in vals]
        return [ (vw - med)/sigma for vw in w ], {"median": med, "sigma": sigma, "lo": lo, "hi": hi, "method": "robust_z"}
    else:
        mu = sum(vals)/len(vals)
        var = sum((v-mu)**2 for v in vals)/(len(vals)-1) if len(vals) > 1 else 0.0
        sd = math.sqrt(var) if var > 0 else 1.0
        return [ (v - mu)/sd for v in vals ], {"mean": mu, "sd": sd, "method": "zscore"}

def normalize_metrics_per_chunk(
    grouped: dict[str, list[dict]],
    metrics: list[str],
    *,
    method: str = "robust_z",   # 'robust_z' | 'zscore' | 'minmax'
    winsor: tuple[float,float] = (0.01, 0.99),
) -> dict[str, list[dict]]:
    """
    For each event group, normalize each metric within that group and attach
    a direction-aware *badness* value in r[f"{m}__norm"], where lower=better.
    """
    out: dict[str, list[dict]] = {}
    robust = (method == "robust_z")
    for ev, rows in grouped.items():
        rows2 = []
        # collect per-metric vectors (skip None)
        per_m_vals: dict[str, list[tuple[int,float]]] = {m: [] for m in metrics}
        for i, r in enumerate(rows):
            for m in metrics:
                v = _to_float(r.get(m))
                if v is not None:
                    per_m_vals[m].append((i, v))

        # compute normalized series metric-by-metric
        norm_series: dict[str, dict[int, float]] = {m: {} for m in metrics}
        for m in metrics:
            if not per_m_vals[m]:
                continue
            idxs, vals = zip(*per_m_vals[m])
            norm_vals, _stats = _norm_values(list(vals),
                                             method=("minmax" if method=="minmax" else "zscore"),
                                             robust=robust,
                                             winsor=winsor)
            # convert to badness (lower=better)
            if LOWER_IS_BETTER.get(m, True):
                bad = norm_vals[:]  # already “low=good” after z or minmax centered low
            else:
                # invert: high-good → high-badness becomes negative; for minmax use 1 - x
                bad = ([-z for z in norm_vals] if method!="minmax" else [1.0 - x for x in norm_vals])
            for i, b in zip(idxs, bad):
                norm_series[m][i] = b

        # write back copies with __norm fields
        for i, r in enumerate(rows):
            rr = dict(r)
            for m in metrics:
                rr[m + "__norm"] = norm_series[m].get(i)  # may be None if metric missing
            rows2.append(rr)
        out[ev] = rows2
    return out

# -------- Combine per chunk & select best --------
def combine_per_chunk_and_select_best(
    grouped_norm: dict[str, list[dict]],
    metrics: list[str],
    weights: list[float],
    *,
    norm_suffix: str = "__norm",
    new_metric_name: str = "combined_metric",
) -> list[dict]:
    """
    Build a weighted BADNESS score from per-chunk normalized metrics, then
    pick the single lowest score per event. Returns the best rows (flattened).
    """
    best: list[dict] = []
    for ev, rows in grouped_norm.items():
        if not rows:
            continue
        # weighted mean of available normalized metrics
        scored: list[tuple[float, dict]] = []
        for r in rows:
            num = 0.0
            den = 0.0
            for m, w in zip(metrics, weights):
                if w == 0: 
                    continue
                v = r.get(m + norm_suffix)
                if v is None:
                    continue
                num += float(w) * float(v)
                den += abs(float(w))
            if den == 0:
                continue
            score = num/den
            rr = dict(r); rr[new_metric_name] = score
            scored.append((score, rr))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])
        best.append(scored[0][1])
    return best

# -------- Global normalization & optional filtering --------
def global_normalize_metric(
    rows: list[dict],
    metric: str = "combined_metric",
    *,
    method: str = "robust_z",  # 'robust_z' | 'zscore' | 'minmax'
    winsor: tuple[float,float] = (0.01, 0.99),
    out_name: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Normalize a single metric across all rows. Returns (rows_with_norm, stats).
    The normalized column is metric + "__global" (or out_name).
    """
    vals = [ _to_float(r.get(metric)) for r in rows ]
    idx = [ i for i,v in enumerate(vals) if v is not None ]
    z  = [ vals[i] for i in idx ]
    norm, stats = _norm_values(z, method=("minmax" if method=="minmax" else "zscore"),
                               robust=(method=="robust_z"), winsor=winsor)
    name = out_name or (metric + "__global")
    out = []
    it = iter(norm)
    for i, r in enumerate(rows):
        rr = dict(r)
        rr[name] = (next(it) if i in idx else None)
        out.append(rr)
    return out, stats

def filter_by_global_metric(
    rows: list[dict],
    *,
    metric_norm_name: str,
    threshold: float,
    keep_low: bool = True,
) -> list[dict]:
    """Keep rows where normalized metric ≤ threshold (keep_low=True) or ≥ if keep_low=False."""
    out = []
    for r in rows:
        v = _to_float(r.get(metric_norm_name))
        if v is None:
            continue
        if (v <= threshold) if keep_low else (v >= threshold):
            out.append(r)
    return out
