#!/usr/bin/env python3
"""Build OriDyn v5 coarse crowding-score/filter sweeps."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
for _thread_env_name in BLAS_THREAD_ENV_VARS:
    os.environ.setdefault(_thread_env_name, "1")

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_v5_aggressive_filter_poc_streams as agmod  # noqa: E402
import build_v5_crowding_target_excitation_corner_streams as cornermod  # noqa: E402
import build_v5_highEg_excitation_matched_crowding_filter_streams as highmod  # noqa: E402
import compute_v5_nonself_local_excitation_raw_scores_20_0p3 as v5mod  # noqa: E402


KEY_COLUMNS = agmod.KEY_COLUMNS
HKL_COLUMNS = agmod.HKL_COLUMNS
EG_COLUMN = "target_excitation_Eg"
BASELINE_SCORE_COLUMN = "nonself_local_excitation_raw"
ABS_SG_TARGET_COLUMN = "abs_sg_target"
EXACT_KEY_TEXT_COLUMN = "exact_key_text"
SIGNED_HKL_ID_COLUMN = "signed_hkl_id"
BLOCK_ID_COLUMN = "excitation_block_id"
COMMON_BLOCK_COLUMN = "common_block"

BASELINE_SG0 = 0.0013180204579645218
BASELINE_SIGMA_C = 0.050
DEFAULT_SG0_MULTIPLIERS = [0.75, 1.00, 1.50]
DEFAULT_SIGMA_C_MULTIPLIERS = [0.70, 1.00, 1.40]
DEFAULT_ALPHA_VALUES = [0.0]
DEFAULT_HIGH_EG_FRACTIONS = [0.30]
DEFAULT_DROP_FRACTIONS = [0.30]
DEFAULT_EXCITATION_BLOCK_SIZES = [10]
DEFAULT_MIN_FINAL_BLOCK_SIZE = 5
DEFAULT_MIN_HIGH_EG_OBSERVATIONS = 10
DEFAULT_MIN_REMAINING_PER_BLOCK = 2
DEFAULT_BLOCK_GATE_MODE = "none"
DEFAULT_SCORE_RTOL = 1.0e-8
DEFAULT_SCORE_ATOL = 1.0e-12
DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR = agmod.DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR
DEFAULT_CACHE_NAME = "cached_multi_score_table.csv.gz"
OLD_CACHE_NAME = "cached_highEg_multi_score_table.csv.gz"
SCHEMA_VERSION = "v5_coarse_score_sweep_v2"
SCORER_SCHEMA_VERSION = "raw_nonself_gaussian_coupling_alpha_v1"
CSV_FLOAT_FORMAT = "%.12g"
CACHE_FLOAT_FORMAT = "%.17g"

BASELINE_RAW_NAME = "sg100_sc100"
BASELINE_SCORE_NAME = "sg100_sc100_a000"


@dataclass(frozen=True)
class RawKernelSpec:
    name: str
    sg_label: str
    sg_multiplier: float
    sg0: float
    sigma_label: str
    sigma_multiplier: float
    sigma_c: float
    r_cut: float
    score_column: str
    coupling_sum_column: str
    internal_control: bool = False


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    raw_name: str
    sg_label: str
    sg_multiplier: float
    sg0: float
    sigma_label: str
    sigma_multiplier: float
    sigma_c: float
    r_cut: float
    alpha: float
    alpha_label: str
    raw_score_column: str
    coupling_sum_column: str
    alpha_score_column: str
    internal_control: bool = False
    score_type: str = "alpha"
    formula_kind: str = "alpha"
    eg_power: float = 0.0
    a_power: float = 0.0
    d_power: float = 0.0
    imbalance_power: float = 0.0
    derived_config_name: str = ""
    expression_tree: Any = None
    score_family: str = ""
    formula: str = ""
    p_value: float | None = None
    lambda_value: float | None = None


@dataclass(frozen=True)
class FilterSpec:
    name: str
    high_eg_fraction: float
    drop_fraction: float
    excitation_block_size: int
    min_final_block_size: int
    min_high_eg_observations: int
    min_remaining_per_block: int
    block_gate_mode: str
    block_range_threshold: float | None = None
    block_range_quantile: float | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    score_name: str
    filter_name: str
    output_stream: str


@dataclass(frozen=True)
class SweepPlan:
    raw_specs: tuple[RawKernelSpec, ...]
    requested_score_specs: tuple[ScoreSpec, ...]
    internal_baseline_score: ScoreSpec
    filter_specs: tuple[FilterSpec, ...]
    experiments: tuple[ExperimentSpec, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--v5-scores", type=Path, required=True)
    parser.add_argument("--input-stream", type=Path, required=True)
    parser.add_argument("--baseline-removals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--derived-score-config", type=Path)
    parser.add_argument(
        "--pathway-moment-screen",
        action="store_true",
        help="Use the built-in sg175/sc100 pathway-moment expression-tree screen and write pathway_moment_scores.json.",
    )
    parser.add_argument("--sg0-multipliers", nargs="+", type=float)
    parser.add_argument("--sigma-c-multipliers", nargs="+", type=float)
    parser.add_argument("--alpha-values", nargs="+", type=float)
    parser.add_argument("--high-eg-fractions", nargs="+", type=float)
    parser.add_argument("--drop-fractions", nargs="+", type=float)
    parser.add_argument("--excitation-block-sizes", nargs="+", type=int)
    parser.add_argument("--min-final-block-size", type=int)
    parser.add_argument("--min-high-eg-observations", type=int)
    parser.add_argument("--min-remaining-per-block", type=int)
    parser.add_argument("--block-gate-mode", choices=["none", "absolute_range", "range_quantile"])
    parser.add_argument("--block-range-thresholds", nargs="+", type=float)
    parser.add_argument("--block-range-quantiles", nargs="+", type=float)
    parser.add_argument("--score-cache-dir", type=Path)
    parser.add_argument("--reference-removals", type=Path)
    parser.add_argument("--reference-sweep-dir", type=Path)
    parser.add_argument("--expected-removal-count", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recompute-cache",
        action="store_true",
        help="Explicitly recompute the expensive score cache instead of reusing a matching cache.",
    )
    parser.add_argument("--max-output-streams", type=int, default=100)
    parser.add_argument("--allow-large-grid", action="store_true")
    parser.add_argument("--chunksize", type=int, default=v5mod.DEFAULT_CHUNKSIZE)
    parser.add_argument("--target-batch-size", type=int, default=v5mod.DEFAULT_TARGET_BATCH_SIZE)
    parser.add_argument("--score-rtol", type=float, default=DEFAULT_SCORE_RTOL)
    parser.add_argument("--score-atol", type=float, default=DEFAULT_SCORE_ATOL)
    args = parser.parse_args()

    for label, path in [
        ("--manifest", args.manifest),
        ("--accepted", args.accepted),
        ("--v5-scores", args.v5_scores),
        ("--input-stream", args.input_stream),
        ("--baseline-removals", args.baseline_removals),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.config is not None and not args.config.is_file():
        raise SystemExit(f"--config not found: {args.config}")
    if args.derived_score_config is not None and not args.derived_score_config.is_file():
        raise SystemExit(f"--derived-score-config not found: {args.derived_score_config}")
    if args.score_cache_dir is not None and not args.score_cache_dir.is_dir():
        raise SystemExit(f"--score-cache-dir not found: {args.score_cache_dir}")
    if args.reference_removals is not None and not args.reference_removals.is_file():
        raise SystemExit(f"--reference-removals not found: {args.reference_removals}")
    if args.reference_sweep_dir is not None and not args.reference_sweep_dir.is_dir():
        raise SystemExit(f"--reference-sweep-dir not found: {args.reference_sweep_dir}")
    if args.out_dir.exists() and not args.out_dir.is_dir():
        raise SystemExit(f"--out-dir exists but is not a directory: {args.out_dir}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not (args.overwrite or args.skip_existing):
        raise SystemExit(f"Output directory exists and is nonempty; use --overwrite to replace files: {args.out_dir}")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    if int(args.max_output_streams) < 1:
        raise SystemExit("--max-output-streams must be >= 1")
    if args.expected_removal_count is not None and int(args.expected_removal_count) < 1:
        raise SystemExit("--expected-removal-count must be >= 1")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if int(args.target_batch_size) < 1:
        raise SystemExit("--target-batch-size must be >= 1")
    if not np.isfinite(float(args.score_rtol)) or float(args.score_rtol) < 0.0:
        raise SystemExit("--score-rtol must be finite and >= 0")
    if not np.isfinite(float(args.score_atol)) or float(args.score_atol) < 0.0:
        raise SystemExit("--score-atol must be finite and >= 0")
    return args


def log(message: str) -> None:
    agmod.log(message)


def json_default(value: Any) -> Any:
    return agmod.json_default(value)


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def git_commit_for_path(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path.resolve().parents[1]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def cache_path(out_dir: Path) -> Path:
    return out_dir / DEFAULT_CACHE_NAME


def existing_cache_path(directory: Path) -> Path | None:
    for name in [DEFAULT_CACHE_NAME, OLD_CACHE_NAME]:
        path = directory / name
        if path.is_file():
            return path
    return None


def key_text_from_frame(frame: pd.DataFrame) -> pd.Series:
    return cornermod.key_text_from_frame(frame)


def key_set(table: pd.DataFrame) -> set[tuple[str, str, int, int, int]]:
    return cornermod.key_set(table)


def frame_key(source: Any, event: Any) -> str:
    return f"{agmod.normalize_source(source)}\0{agmod.normalize_event(event)}"


def add_exact_key_text(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out[EXACT_KEY_TEXT_COLUMN] = key_text_from_frame(out)
    return out


def numeric_distribution(values: Any) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    nonfinite_count = int(series.isna().sum())
    clean = series.dropna()
    if clean.empty:
        return {
            "n": 0,
            "nonfinite_count": nonfinite_count,
            "zero_count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "mean": None,
            "std": None,
            "max": None,
        }
    return {
        "n": int(len(clean)),
        "nonfinite_count": nonfinite_count,
        "zero_count": int((clean == 0.0).sum()),
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "max": float(clean.max()),
    }


def finite_corr(x: pd.Series, y: pd.Series, method: str) -> float | None:
    a = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    b = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 2:
        return None
    avals = a.loc[mask].to_numpy(dtype=float)
    bvals = b.loc[mask].to_numpy(dtype=float)
    if float(np.std(avals)) == 0.0 or float(np.std(bvals)) == 0.0:
        return None
    if method == "pearson":
        return float(np.corrcoef(avals, bvals)[0, 1])
    if method == "spearman":
        aranks = a.loc[mask].rank(method="average").to_numpy(dtype=float)
        branks = b.loc[mask].rank(method="average").to_numpy(dtype=float)
        if float(np.std(aranks)) == 0.0 or float(np.std(branks)) == 0.0:
            return None
        return float(np.corrcoef(aranks, branks)[0, 1])
    raise ValueError(f"Unknown correlation method: {method}")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse --config JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--config must contain a JSON object")
    return payload


def load_json_payload(path: Path | None, label: str) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {label} JSON: {exc}") from exc


def div_node(numerator: Any, denominator: Any) -> dict[str, Any]:
    return {
        "op": "div",
        "numerator": numerator,
        "denominator": denominator,
        "zero_policy": "zero_if_zero_over_zero_else_fail",
    }


def pow_node(base: Any, exponent: float) -> dict[str, Any]:
    return {"op": "pow", "base": base, "exponent": float(exponent)}


def var_node(name: str) -> dict[str, str]:
    return {"var": name}


def const_node(value: float) -> dict[str, float]:
    return {"const": float(value)}


def pathway_moment_score_records() -> list[dict[str, Any]]:
    eg = var_node("Eg")
    m = var_node("M")
    d = var_node("D")
    u = var_node("U")
    n = var_node("N")
    a = var_node("A")
    q = var_node("Q")
    pc = var_node("Pc")
    pe = var_node("Pe")
    neff = var_node("Neff")
    m2 = var_node("M2")
    d2 = pow_node(d, 2.0)
    d3 = pow_node(d, 3.0)
    d4 = pow_node(d, 4.0)
    current = {"op": "mul", "args": [eg, m, d3]}
    return [
        {"name": "current_reference_EgMD3", "kind": "expression_tree", "tree": current},
        {"name": "excitation_normalized_Eg_MoverU_D3", "kind": "expression_tree", "tree": {"op": "mul", "args": [eg, div_node(m, u), d3]}},
        {"name": "excitation_mass_EgMD3_UoverN", "kind": "expression_tree", "tree": {"op": "mul", "args": [eg, m, d3, div_node(u, n)]}},
        {"name": "excitation_coupling_alignment_EgMD3Q", "kind": "expression_tree", "tree": {"op": "mul", "args": [eg, m, d3, q]}},
        {
            "name": "harmonic_coexcitation_D4",
            "kind": "expression_tree",
            "tree": {
                "op": "mul",
                "args": [
                    div_node({"op": "mul", "args": [const_node(2.0), eg, a]}, {"op": "add", "args": [eg, a]}),
                    d4,
                ],
            },
        },
        {"name": "bottleneck_coexcitation_D4", "kind": "expression_tree", "tree": {"op": "mul", "args": [{"op": "min", "args": [eg, a]}, d4]}},
        {"name": "target_squared_Eg2MD3", "kind": "expression_tree", "tree": {"op": "mul", "args": [pow_node(eg, 2.0), m, d3]}},
        {"name": "distinct_coupling_channels_EgMDPc", "kind": "expression_tree", "tree": {"op": "mul", "args": [eg, m, d, pc]}},
        {"name": "active_pathway_pairs_EgD2Pe", "kind": "expression_tree", "tree": {"op": "mul", "args": [eg, d2, pe]}},
        {"name": "effective_path_count_EgMD3sqrtNeff", "kind": "expression_tree", "tree": {"op": "mul", "args": [eg, m, d3, {"op": "sqrt", "arg": neff}]}},
        {"name": "strong_link_moment_EgD2M2", "kind": "expression_tree", "tree": {"op": "mul", "args": [eg, d2, m2]}},
    ]


def pathway_moment_config() -> dict[str, Any]:
    return {
        "score_grid": {
            "sg0_multipliers": [1.75],
            "sigma_c_multipliers": [1.0],
        },
        "filter_grid": {
            "high_eg_fractions": [0.30],
            "drop_fractions": [0.30],
            "excitation_block_sizes": [10],
            "min_final_block_size": 5,
            "min_high_eg_observations": 10,
            "min_remaining_per_block": 2,
            "block_gate_mode": "none",
        },
        "derived_scores": pathway_moment_score_records(),
    }


def write_pathway_moment_config(out_dir: Path) -> Path:
    path = out_dir / "pathway_moment_scores.json"
    path.write_text(json.dumps(pathway_moment_config(), indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    return path


def config_lookup(config: dict[str, Any], section: str, key: str) -> Any:
    aliases = [key, key.replace("_", "-")]
    section_payload = config.get(section, {})
    if section_payload is not None and not isinstance(section_payload, dict):
        raise SystemExit(f"--config section {section!r} must be an object")
    for payload in [section_payload or {}, config]:
        for alias in aliases:
            if alias in payload:
                return payload[alias]
    return None


def resolve_configured_value(args: argparse.Namespace, config: dict[str, Any], attr: str, section: str, default: Any) -> Any:
    value = getattr(args, attr)
    if value is not None:
        return value
    configured = config_lookup(config, section, attr)
    if configured is not None:
        return configured
    return default


def valid_name_token(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise SystemExit(f"{label} must be a nonempty string")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if any(char not in allowed for char in text):
        raise SystemExit(f"{label} may contain only letters, numbers, underscore, dot, and dash: {text!r}")
    return text


def finite_power(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise SystemExit(f"{label} must be finite")
    return number


ALLOWED_EXPRESSION_VARIABLES = {"Eg", "M", "D", "U", "N", "C2", "W2", "M2", "A", "Q", "Neff", "Pc", "Pe"}


def validate_expression_tree(node: Any, label: str) -> Any:
    if not isinstance(node, dict):
        raise SystemExit(f"{label} must be an expression object")
    if "var" in node:
        if set(node) != {"var"}:
            raise SystemExit(f"{label} variable node may contain only var")
        var = str(node["var"])
        if var not in ALLOWED_EXPRESSION_VARIABLES:
            raise SystemExit(f"{label} variable must be one of {sorted(ALLOWED_EXPRESSION_VARIABLES)}")
        return {"var": var}
    if "const" in node:
        if set(node) != {"const"}:
            raise SystemExit(f"{label} constant node may contain only const")
        return {"const": finite_power(node["const"], f"{label}.const")}
    op = str(node.get("op", ""))
    if op in {"mul", "add"}:
        args = node.get("args")
        if not isinstance(args, list) or len(args) < 1:
            raise SystemExit(f"{label}.{op} requires a nonempty args list")
        return {"op": op, "args": [validate_expression_tree(arg, f"{label}.{op}[{idx}]") for idx, arg in enumerate(args)]}
    if op == "sub":
        if "left" not in node or "right" not in node:
            raise SystemExit(f"{label}.sub requires left and right")
        return {
            "op": "sub",
            "left": validate_expression_tree(node["left"], f"{label}.sub.left"),
            "right": validate_expression_tree(node["right"], f"{label}.sub.right"),
        }
    if op == "div":
        if node.get("zero_policy") != "zero_if_zero_over_zero_else_fail":
            raise SystemExit(f"{label}.div requires zero_policy=zero_if_zero_over_zero_else_fail")
        if "numerator" not in node or "denominator" not in node:
            raise SystemExit(f"{label}.div requires numerator and denominator")
        return {
            "op": "div",
            "numerator": validate_expression_tree(node["numerator"], f"{label}.div.numerator"),
            "denominator": validate_expression_tree(node["denominator"], f"{label}.div.denominator"),
            "zero_policy": "zero_if_zero_over_zero_else_fail",
        }
    if op == "pow":
        if "base" not in node or "exponent" not in node:
            raise SystemExit(f"{label}.pow requires base and exponent")
        return {
            "op": "pow",
            "base": validate_expression_tree(node["base"], f"{label}.pow.base"),
            "exponent": finite_power(node["exponent"], f"{label}.pow.exponent"),
        }
    if op == "sqrt":
        if "arg" not in node:
            raise SystemExit(f"{label}.sqrt requires arg")
        return {"op": "sqrt", "arg": validate_expression_tree(node["arg"], f"{label}.sqrt.arg")}
    if op == "min":
        args = node.get("args")
        if not isinstance(args, list) or len(args) != 2:
            raise SystemExit(f"{label}.min requires exactly two args")
        return {"op": "min", "args": [validate_expression_tree(arg, f"{label}.min[{idx}]") for idx, arg in enumerate(args)]}
    raise SystemExit(f"{label} has unsupported expression op: {op!r}")


def normalize_derived_score_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SystemExit(f"derived_scores[{index}] must be an object")
    name = valid_name_token(record.get("name", ""), f"derived_scores[{index}].name")
    kind = str(record.get("kind", "")).strip()
    if kind not in {"power_product", "imbalance_absolute", "imbalance_feed", "imbalance_sink", "expression_tree"}:
        raise SystemExit(
            f"derived_scores[{index}].kind must be one of "
            "power_product, imbalance_absolute, imbalance_feed, imbalance_sink, expression_tree"
        )
    out = {
        "name": name,
        "kind": kind,
        "eg_power": finite_power(record.get("eg_power", 0.0), f"derived_scores[{index}].eg_power"),
        "a_power": 0.0,
        "d_power": finite_power(record.get("d_power", 0.0), f"derived_scores[{index}].d_power"),
        "imbalance_power": 0.0,
        "expression_tree": None,
        "score_family": valid_name_token(record.get("family", record.get("score_family", "")), f"derived_scores[{index}].family")
        if record.get("family", record.get("score_family", "")) not in {None, ""}
        else "",
        "formula": str(record.get("formula", "")).strip(),
        "p_value": None,
        "lambda_value": None,
    }
    if "p" in record and record["p"] is not None:
        out["p_value"] = finite_power(record["p"], f"derived_scores[{index}].p")
    if "lambda" in record and record["lambda"] is not None:
        out["lambda_value"] = finite_power(record["lambda"], f"derived_scores[{index}].lambda")
    if kind == "expression_tree":
        tree = record.get("tree", record.get("expression"))
        if tree is None:
            raise SystemExit(f"derived_scores[{index}] expression_tree requires tree or expression")
        out["expression_tree"] = validate_expression_tree(tree, f"derived_scores[{index}].tree")
    elif kind == "power_product":
        out["a_power"] = finite_power(record.get("a_power", 0.0), f"derived_scores[{index}].a_power")
    elif kind == "imbalance_absolute":
        if "imbalance_power" not in record:
            raise SystemExit(f"derived_scores[{index}].imbalance_power is required for imbalance_absolute")
        out["imbalance_power"] = finite_power(record["imbalance_power"], f"derived_scores[{index}].imbalance_power")
    else:
        if "a_power" in record or "imbalance_power" in record:
            raise SystemExit(f"derived_scores[{index}] kind {kind} does not use a_power or imbalance_power")
    return out


def derived_scores_from_payload(payload: Any, label: str) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        records = payload.get("derived_scores", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise SystemExit(f"{label} must be either a config object or a derived_scores list")
    if records is None:
        return []
    if not isinstance(records, list):
        raise SystemExit(f"{label} derived_scores must be a list")
    normalized = [normalize_derived_score_record(record, index) for index, record in enumerate(records)]
    collision_check([record["name"] for record in normalized], "derived score")
    return normalized


def resolve_derived_scores(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    if args.pathway_moment_screen:
        return derived_scores_from_payload(pathway_moment_config(), "--pathway-moment-screen")
    if args.derived_score_config is not None:
        return derived_scores_from_payload(load_json_payload(args.derived_score_config, "--derived-score-config"), "--derived-score-config")
    return derived_scores_from_payload(config, "--config")


def dedupe_float_values(values: Any, name: str, *, lower: float | None = None, upper: float | None = None, lower_inclusive: bool = False, upper_inclusive: bool = True) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise SystemExit(f"{name} must be a list")
    clean: list[float] = []
    for value in values:
        number = float(value)
        if not np.isfinite(number):
            raise SystemExit(f"{name} values must be finite")
        if lower is not None:
            ok = number >= lower if lower_inclusive else number > lower
            if not ok:
                comparator = ">=" if lower_inclusive else ">"
                raise SystemExit(f"{name} values must be {comparator} {lower}")
        if upper is not None:
            ok = number <= upper if upper_inclusive else number < upper
            if not ok:
                comparator = "<=" if upper_inclusive else "<"
                raise SystemExit(f"{name} values must be {comparator} {upper}")
        clean.append(number)
    if not clean:
        raise SystemExit(f"{name} must contain at least one value")
    return sorted({float(value) for value in clean})


def dedupe_int_values(values: Any, name: str, *, minimum: int) -> list[int]:
    if not isinstance(values, (list, tuple)):
        raise SystemExit(f"{name} must be a list")
    clean: list[int] = []
    for value in values:
        number = int(value)
        if number < int(minimum):
            raise SystemExit(f"{name} values must be >= {int(minimum)}")
        clean.append(number)
    if not clean:
        raise SystemExit(f"{name} must contain at least one value")
    return sorted(set(clean))


def scaled_code(value: float, *, scale: float = 100.0, min_width: int = 3) -> str:
    scaled = float(value) * float(scale)
    rounded = int(round(scaled))
    if np.isclose(scaled, rounded, rtol=0.0, atol=1.0e-9):
        return f"{rounded:0{min_width}d}"
    return safe_float_token(scaled)


def safe_float_token(value: float) -> str:
    text = f"{float(value):.8g}".lower()
    text = text.replace("+", "")
    text = text.replace("-", "n")
    text = text.replace(".", "p")
    text = text.replace("e", "e")
    return text


def alpha_label(alpha: float) -> str:
    if np.isclose(float(alpha), 0.0, rtol=0.0, atol=1.0e-12):
        return "a000"
    prefix = "ap" if float(alpha) > 0.0 else "an"
    return f"{prefix}{scaled_code(abs(float(alpha)))}"


def fraction_label(prefix: str, value: float) -> str:
    return f"{prefix}{scaled_code(float(value), scale=100.0, min_width=3)}"


def raw_kernel_spec(sg_multiplier: float, sigma_multiplier: float, *, internal_control: bool = False) -> RawKernelSpec:
    sg_label = f"sg{scaled_code(sg_multiplier)}"
    sigma_label = f"sc{scaled_code(sigma_multiplier)}"
    sigma_c = float(sigma_multiplier) * BASELINE_SIGMA_C
    sg0 = float(sg_multiplier) * BASELINE_SG0
    name = f"{sg_label}_{sigma_label}"
    return RawKernelSpec(
        name=name,
        sg_label=sg_label,
        sg_multiplier=float(sg_multiplier),
        sg0=float(sg0),
        sigma_label=sigma_label,
        sigma_multiplier=float(sigma_multiplier),
        sigma_c=float(sigma_c),
        r_cut=3.0 * float(sigma_c),
        score_column=f"score_{name}",
        coupling_sum_column=f"coupling_sum_{sigma_label}",
        internal_control=bool(internal_control),
    )


def score_spec(raw: RawKernelSpec, alpha: float, *, internal_control: bool = False) -> ScoreSpec:
    label = alpha_label(alpha)
    name = f"{raw.name}_{label}"
    return ScoreSpec(
        name=name,
        raw_name=raw.name,
        sg_label=raw.sg_label,
        sg_multiplier=raw.sg_multiplier,
        sg0=raw.sg0,
        sigma_label=raw.sigma_label,
        sigma_multiplier=raw.sigma_multiplier,
        sigma_c=raw.sigma_c,
        r_cut=raw.r_cut,
        alpha=float(alpha),
        alpha_label=label,
        raw_score_column=raw.score_column,
        coupling_sum_column=raw.coupling_sum_column,
        alpha_score_column=f"alpha_score_{name}",
        internal_control=bool(internal_control),
        score_type="alpha",
        formula_kind="alpha",
    )


def derived_score_spec(raw: RawKernelSpec, record: dict[str, Any], *, include_raw_in_name: bool) -> ScoreSpec:
    base_name = str(record["name"])
    name = f"{base_name}_{raw.name}" if include_raw_in_name else base_name
    kind = str(record["kind"])
    return ScoreSpec(
        name=name,
        raw_name=raw.name,
        sg_label=raw.sg_label,
        sg_multiplier=raw.sg_multiplier,
        sg0=raw.sg0,
        sigma_label=raw.sigma_label,
        sigma_multiplier=raw.sigma_multiplier,
        sigma_c=raw.sigma_c,
        r_cut=raw.r_cut,
        alpha=0.0,
        alpha_label="",
        raw_score_column=raw.score_column,
        coupling_sum_column=raw.coupling_sum_column,
        alpha_score_column=f"derived_score_{name}",
        internal_control=False,
        score_type="derived",
        formula_kind=kind,
        eg_power=float(record.get("eg_power", 0.0)),
        a_power=float(record.get("a_power", 0.0)),
        d_power=float(record.get("d_power", 0.0)),
        imbalance_power=float(record.get("imbalance_power", 0.0)),
        derived_config_name=base_name,
        expression_tree=record.get("expression_tree"),
        score_family=str(record.get("score_family", "")),
        formula=str(record.get("formula", "")),
        p_value=None if record.get("p_value") is None else float(record["p_value"]),
        lambda_value=None if record.get("lambda_value") is None else float(record["lambda_value"]),
    )


def filter_name_for(
    high_eg_fraction: float,
    drop_fraction: float,
    excitation_block_size: int,
    min_final_block_size: int,
    min_high_eg_observations: int,
    min_remaining_per_block: int,
    block_gate_mode: str,
    block_range_threshold: float | None,
    block_range_quantile: float | None,
) -> str:
    parts = [
        fraction_label("he", high_eg_fraction),
        f"bs{int(excitation_block_size):03d}",
        fraction_label("drop", drop_fraction),
    ]
    if int(min_final_block_size) != DEFAULT_MIN_FINAL_BLOCK_SIZE:
        parts.append(f"mfb{int(min_final_block_size):03d}")
    if int(min_high_eg_observations) != DEFAULT_MIN_HIGH_EG_OBSERVATIONS:
        parts.append(f"mhe{int(min_high_eg_observations):03d}")
    if int(min_remaining_per_block) != DEFAULT_MIN_REMAINING_PER_BLOCK:
        parts.append(f"mrp{int(min_remaining_per_block):03d}")
    if block_gate_mode == "absolute_range":
        if block_range_threshold is None:
            raise SystemExit("absolute_range gate requires a block range threshold")
        parts.append(f"rg{safe_float_token(float(block_range_threshold))}")
    elif block_gate_mode == "range_quantile":
        if block_range_quantile is None:
            raise SystemExit("range_quantile gate requires a block range quantile")
        parts.append(f"rq{scaled_code(float(block_range_quantile), scale=100.0, min_width=3)}")
    elif block_gate_mode != "none":
        raise SystemExit(f"Unknown block gate mode: {block_gate_mode}")
    return "_".join(parts)


def filter_spec(
    high_eg_fraction: float,
    drop_fraction: float,
    excitation_block_size: int,
    min_final_block_size: int,
    min_high_eg_observations: int,
    min_remaining_per_block: int,
    block_gate_mode: str,
    block_range_threshold: float | None,
    block_range_quantile: float | None,
) -> FilterSpec:
    return FilterSpec(
        name=filter_name_for(
            high_eg_fraction,
            drop_fraction,
            excitation_block_size,
            min_final_block_size,
            min_high_eg_observations,
            min_remaining_per_block,
            block_gate_mode,
            block_range_threshold,
            block_range_quantile,
        ),
        high_eg_fraction=float(high_eg_fraction),
        drop_fraction=float(drop_fraction),
        excitation_block_size=int(excitation_block_size),
        min_final_block_size=int(min_final_block_size),
        min_high_eg_observations=int(min_high_eg_observations),
        min_remaining_per_block=int(min_remaining_per_block),
        block_gate_mode=str(block_gate_mode),
        block_range_threshold=None if block_range_threshold is None else float(block_range_threshold),
        block_range_quantile=None if block_range_quantile is None else float(block_range_quantile),
    )


def collision_check(values: list[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise SystemExit(f"{label} name collision(s): {duplicates}")


def is_default_filter(spec: FilterSpec) -> bool:
    return (
        np.isclose(spec.high_eg_fraction, DEFAULT_HIGH_EG_FRACTIONS[0])
        and np.isclose(spec.drop_fraction, DEFAULT_DROP_FRACTIONS[0])
        and int(spec.excitation_block_size) == DEFAULT_EXCITATION_BLOCK_SIZES[0]
        and int(spec.min_final_block_size) == DEFAULT_MIN_FINAL_BLOCK_SIZE
        and int(spec.min_high_eg_observations) == DEFAULT_MIN_HIGH_EG_OBSERVATIONS
        and int(spec.min_remaining_per_block) == DEFAULT_MIN_REMAINING_PER_BLOCK
        and spec.block_gate_mode == DEFAULT_BLOCK_GATE_MODE
    )


def resolve_sweep_plan(args: argparse.Namespace) -> SweepPlan:
    config = pathway_moment_config() if args.pathway_moment_screen else load_config(args.config)
    derived_score_records = resolve_derived_scores(args, config)
    alpha_configured = args.alpha_values is not None or config_lookup(config, "score_grid", "alpha_values") is not None
    sg_multipliers = dedupe_float_values(
        resolve_configured_value(args, config, "sg0_multipliers", "score_grid", DEFAULT_SG0_MULTIPLIERS),
        "--sg0-multipliers",
        lower=0.0,
        lower_inclusive=False,
    )
    sigma_multipliers = dedupe_float_values(
        resolve_configured_value(args, config, "sigma_c_multipliers", "score_grid", DEFAULT_SIGMA_C_MULTIPLIERS),
        "--sigma-c-multipliers",
        lower=0.0,
        lower_inclusive=False,
    )
    if derived_score_records and not alpha_configured:
        alpha_values: list[float] = []
    else:
        alpha_values = dedupe_float_values(
            resolve_configured_value(args, config, "alpha_values", "score_grid", DEFAULT_ALPHA_VALUES),
            "--alpha-values",
        )
    high_eg_fractions = dedupe_float_values(
        resolve_configured_value(args, config, "high_eg_fractions", "filter_grid", DEFAULT_HIGH_EG_FRACTIONS),
        "--high-eg-fractions",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=True,
    )
    drop_fractions = dedupe_float_values(
        resolve_configured_value(args, config, "drop_fractions", "filter_grid", DEFAULT_DROP_FRACTIONS),
        "--drop-fractions",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=False,
    )
    block_sizes = dedupe_int_values(
        resolve_configured_value(args, config, "excitation_block_sizes", "filter_grid", DEFAULT_EXCITATION_BLOCK_SIZES),
        "--excitation-block-sizes",
        minimum=1,
    )
    min_final = int(resolve_configured_value(args, config, "min_final_block_size", "filter_grid", DEFAULT_MIN_FINAL_BLOCK_SIZE))
    min_high = int(resolve_configured_value(args, config, "min_high_eg_observations", "filter_grid", DEFAULT_MIN_HIGH_EG_OBSERVATIONS))
    min_remaining = int(resolve_configured_value(args, config, "min_remaining_per_block", "filter_grid", DEFAULT_MIN_REMAINING_PER_BLOCK))
    if min_final < 1:
        raise SystemExit("--min-final-block-size must be >= 1")
    if min_high < 1:
        raise SystemExit("--min-high-eg-observations must be >= 1")
    if min_remaining < 0:
        raise SystemExit("--min-remaining-per-block must be >= 0")

    gate_mode = str(resolve_configured_value(args, config, "block_gate_mode", "filter_grid", DEFAULT_BLOCK_GATE_MODE))
    if gate_mode not in {"none", "absolute_range", "range_quantile"}:
        raise SystemExit("--block-gate-mode must be one of none, absolute_range, range_quantile")
    gate_values: list[tuple[float | None, float | None]]
    if gate_mode == "none":
        gate_values = [(None, None)]
    elif gate_mode == "absolute_range":
        thresholds = dedupe_float_values(
            resolve_configured_value(args, config, "block_range_thresholds", "filter_grid", None),
            "--block-range-thresholds",
            lower=0.0,
            lower_inclusive=True,
        )
        gate_values = [(threshold, None) for threshold in thresholds]
    else:
        quantiles = dedupe_float_values(
            resolve_configured_value(args, config, "block_range_quantiles", "filter_grid", None),
            "--block-range-quantiles",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
            upper_inclusive=True,
        )
        gate_values = [(None, quantile) for quantile in quantiles]

    raw_by_name: dict[str, RawKernelSpec] = {}
    requested_raw_names: list[str] = []
    for sg_multiplier in sg_multipliers:
        for sigma_multiplier in sigma_multipliers:
            raw = raw_kernel_spec(sg_multiplier, sigma_multiplier)
            existing = raw_by_name.get(raw.name)
            if existing is not None and (not np.isclose(existing.sg_multiplier, raw.sg_multiplier) or not np.isclose(existing.sigma_multiplier, raw.sigma_multiplier)):
                raise SystemExit(f"Raw score name collision for {raw.name}")
            raw_by_name[raw.name] = raw
            requested_raw_names.append(raw.name)

    baseline_raw = raw_kernel_spec(1.0, 1.0, internal_control=True)
    if baseline_raw.name not in raw_by_name:
        raw_by_name[baseline_raw.name] = baseline_raw
    else:
        raw_by_name[baseline_raw.name] = raw_kernel_spec(1.0, 1.0, internal_control=False)
    raw_specs = tuple(sorted(raw_by_name.values(), key=lambda spec: (spec.sg_multiplier, spec.sigma_multiplier, spec.name)))

    requested_scores: list[ScoreSpec] = []
    for raw_name in requested_raw_names:
        raw = raw_by_name[raw_name]
        for alpha in alpha_values:
            requested_scores.append(score_spec(raw, alpha))
    include_raw_in_derived_name = bool(derived_score_records and len(set(requested_raw_names)) > 1)
    for raw_name in requested_raw_names:
        raw = raw_by_name[raw_name]
        for record in derived_score_records:
            requested_scores.append(derived_score_spec(raw, record, include_raw_in_name=include_raw_in_derived_name))
    if not requested_scores:
        raise SystemExit("No score variants requested; provide alpha_values or derived_scores")
    collision_check([spec.name for spec in requested_scores], "score variant")

    internal_baseline_score = score_spec(raw_by_name[baseline_raw.name], 0.0, internal_control=True)
    filters: list[FilterSpec] = []
    for high_eg_fraction in high_eg_fractions:
        for drop_fraction in drop_fractions:
            for block_size in block_sizes:
                for threshold, quantile in gate_values:
                    filters.append(
                        filter_spec(
                            high_eg_fraction,
                            drop_fraction,
                            block_size,
                            min_final,
                            min_high,
                            min_remaining,
                            gate_mode,
                            threshold,
                            quantile,
                        )
                    )
    collision_check([spec.name for spec in filters], "filter")

    experiments: list[ExperimentSpec] = []
    for score in requested_scores:
        for flt in filters:
            name = f"{score.name}_{flt.name}"
            experiments.append(ExperimentSpec(name=name, score_name=score.name, filter_name=flt.name, output_stream=f"{name}.stream"))
    collision_check([experiment.name for experiment in experiments], "experiment")
    if len(experiments) > int(args.max_output_streams) and not args.allow_large_grid:
        raise SystemExit(
            f"Sweep would produce {len(experiments):,} output streams, exceeding --max-output-streams="
            f"{int(args.max_output_streams):,}; use --allow-large-grid if intentional"
        )
    return SweepPlan(
        raw_specs=raw_specs,
        requested_score_specs=tuple(requested_scores),
        internal_baseline_score=internal_baseline_score,
        filter_specs=tuple(filters),
        experiments=tuple(experiments),
    )


def raw_spec_by_name(plan: SweepPlan) -> dict[str, RawKernelSpec]:
    return {spec.name: spec for spec in plan.raw_specs}


def score_spec_by_name(plan: SweepPlan) -> dict[str, ScoreSpec]:
    return {spec.name: spec for spec in plan.requested_score_specs}


def filter_spec_by_name(plan: SweepPlan) -> dict[str, FilterSpec]:
    return {spec.name: spec for spec in plan.filter_specs}


def requested_score_columns(plan: SweepPlan) -> list[str]:
    return sorted({spec.score_column for spec in plan.raw_specs})


def requested_coupling_sum_columns(plan: SweepPlan) -> list[str]:
    return sorted({spec.coupling_sum_column for spec in plan.raw_specs})


def aggregate_columns_for_raw(raw: RawKernelSpec | str) -> dict[str, str]:
    raw_name = raw.name if isinstance(raw, RawKernelSpec) else str(raw)
    return {
        "U": f"neighbor_excitation_sum_{raw_name}",
        "N": f"eligible_neighbor_count_{raw_name}",
        "C2": f"coupling_sum_sq_{raw_name}",
        "W2": f"weighted_excitation_coupling_sum_sq_{raw_name}",
        "M2": f"excitation_coupling_sq_sum_{raw_name}",
    }


def expression_tree_variables(node: Any) -> set[str]:
    if not isinstance(node, dict):
        return set()
    if "var" in node:
        return {str(node["var"])}
    variables: set[str] = set()
    for key in ["args"]:
        if isinstance(node.get(key), list):
            for item in node[key]:
                variables.update(expression_tree_variables(item))
    for key in ["left", "right", "numerator", "denominator", "base", "arg"]:
        if key in node:
            variables.update(expression_tree_variables(node[key]))
    return variables


def score_required_variables(spec: ScoreSpec) -> set[str]:
    if spec.score_type == "derived" and spec.formula_kind == "expression_tree":
        return expression_tree_variables(spec.expression_tree)
    if spec.score_type == "derived":
        variables = {"Eg", "M", "D", "A"}
        if spec.formula_kind in {"imbalance_absolute", "imbalance_feed", "imbalance_sink"}:
            variables.add("A")
        return variables
    return {"M", "D"}


def required_cache_columns(plan: SweepPlan) -> list[str]:
    columns: set[str] = set()
    raw_by = raw_spec_by_name(plan)
    for spec in [*plan.requested_score_specs, plan.internal_baseline_score]:
        columns.add(spec.raw_score_column)
        columns.add(spec.coupling_sum_column)
        variables = score_required_variables(spec)
        aggregate_columns = aggregate_columns_for_raw(raw_by[spec.raw_name])
        required_aggregates = variables & {"U", "N", "C2", "W2", "M2"}
        if "Q" in variables:
            required_aggregates.update(["U", "N"])
        if "Neff" in variables:
            required_aggregates.add("W2")
        if "Pc" in variables:
            required_aggregates.add("C2")
        if "Pe" in variables:
            required_aggregates.add("W2")
        for variable in required_aggregates:
            columns.add(aggregate_columns[variable])
    return sorted(columns)


def write_variant_parameters(out_dir: Path, plan: SweepPlan) -> pd.DataFrame:
    table = pd.DataFrame.from_records([asdict(spec) for spec in plan.requested_score_specs])
    table.to_csv(out_dir / "score_variant_parameters.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    return table


def write_experiment_plan(out_dir: Path, plan: SweepPlan) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_by = raw_spec_by_name(plan)
    score_by = {spec.name: spec for spec in plan.requested_score_specs}
    filter_by = filter_spec_by_name(plan)
    records: list[dict[str, Any]] = []
    for experiment in plan.experiments:
        score = score_by[experiment.score_name]
        flt = filter_by[experiment.filter_name]
        raw = raw_by[score.raw_name]
        records.append(
            {
                "experiment": experiment.name,
                "score_name": score.name,
                "raw_kernel_name": raw.name,
                "sg_multiplier": raw.sg_multiplier,
                "sg0": raw.sg0,
                "sigma_multiplier": raw.sigma_multiplier,
                "sigma_c": raw.sigma_c,
                "r_cut": raw.r_cut,
                "score_type": score.score_type,
                "formula_kind": score.formula_kind,
                "score_family": score.score_family,
                "formula": score.formula,
                "p": score.p_value,
                "lambda": score.lambda_value,
                "alpha": score.alpha,
                "eg_power": score.eg_power,
                "a_power": score.a_power,
                "d_power": score.d_power,
                "imbalance_power": score.imbalance_power,
                "expression_tree_json": json.dumps(score.expression_tree, sort_keys=True, default=json_default) if score.expression_tree is not None else "",
                "high_eg_fraction": flt.high_eg_fraction,
                "drop_fraction": flt.drop_fraction,
                "excitation_block_size": flt.excitation_block_size,
                "min_final_block_size": flt.min_final_block_size,
                "min_high_eg_observations": flt.min_high_eg_observations,
                "min_remaining_per_block": flt.min_remaining_per_block,
                "block_gate_mode": flt.block_gate_mode,
                "block_range_threshold": flt.block_range_threshold,
                "block_range_quantile": flt.block_range_quantile,
                "expected_output_filename": experiment.output_stream,
            }
        )
    table = pd.DataFrame.from_records(records)
    table.to_csv(out_dir / "experiment_plan.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_constants": {
            "BASELINE_SG0": BASELINE_SG0,
            "BASELINE_SIGMA_C": BASELINE_SIGMA_C,
            "internal_baseline_control": asdict(plan.internal_baseline_score),
        },
        "raw_kernel_specs": [asdict(spec) for spec in plan.raw_specs],
        "score_specs": [asdict(spec) for spec in plan.requested_score_specs],
        "filter_specs": [asdict(spec) for spec in plan.filter_specs],
        "experiments": [asdict(spec) for spec in plan.experiments],
        "output_stream_count": int(len(plan.experiments)),
    }
    (out_dir / "experiment_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    return table, payload


def cache_metadata(args: argparse.Namespace, plan: SweepPlan) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scorer_schema_version": SCORER_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": file_info(args.manifest),
            "accepted": file_info(args.accepted),
            "v5_scores": file_info(args.v5_scores),
            "input_stream": file_info(args.input_stream),
            "baseline_removals": file_info(args.baseline_removals),
        },
        "baseline_constants": {
            "BASELINE_SG0": BASELINE_SG0,
            "BASELINE_SIGMA_C": BASELINE_SIGMA_C,
            "abs_sg_target_sg0": BASELINE_SG0,
        },
        "raw_kernel_registry": [asdict(spec) for spec in plan.raw_specs],
        "coupling_sum_columns": requested_coupling_sum_columns(plan),
        "raw_score_columns": requested_score_columns(plan),
        "required_cache_columns": required_cache_columns(plan),
        "parameters": {
            "chunksize": int(args.chunksize),
            "target_batch_size": int(args.target_batch_size),
            "score_rtol": float(args.score_rtol),
            "score_atol": float(args.score_atol),
        },
    }


def metadata_matches(expected: dict[str, Any], observed: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for section in ["schema_version", "scorer_schema_version", "baseline_constants"]:
        if observed.get(section) != expected.get(section):
            mismatches.append(section)
    expected_inputs = expected.get("inputs", {})
    observed_inputs = observed.get("inputs", {})
    for key in ["manifest", "accepted", "v5_scores"]:
        if observed_inputs.get(key) != expected_inputs.get(key):
            mismatches.append(f"inputs.{key}")
    expected_params = expected.get("parameters", {})
    observed_params = observed.get("parameters", {})
    for key in ["target_batch_size"]:
        if observed_params.get(key) != expected_params.get(key):
            mismatches.append(f"parameters.{key}")
    return not mismatches, mismatches


def load_previous_cache_metadata(out_dir: Path) -> dict[str, Any] | None:
    audit_path = out_dir / "sweep_audit.json"
    if not audit_path.is_file():
        audit_path = out_dir / "coarse_score_sweep_audit.json"
    if not audit_path.is_file():
        return None
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse existing cache metadata from {audit_path}: {exc}") from exc
    metadata = payload.get("cache_metadata")
    return metadata if isinstance(metadata, dict) else None


def read_manifest(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    return cornermod.read_manifest(path)


def load_accepted_manifest_keys(args: argparse.Namespace, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    return cornermod.load_accepted_manifest_keys(args, manifest)


def load_accepted_v5_rows(args: argparse.Namespace, accepted: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    log("Validating existing v5 score table columns")
    header = pd.read_csv(args.v5_scores, nrows=0).columns.tolist()
    required = [*KEY_COLUMNS, "d_angstrom", "inv_nm", "sg_target", EG_COLUMN, BASELINE_SCORE_COLUMN]
    cornermod.require_columns_from_header(header, required, "v5 score CSV")
    selected_hkls = agmod.selected_hkl_set(accepted)
    accepted_payload = accepted.loc[:, KEY_COLUMNS].copy()
    total_rows = cornermod.count_csv_data_rows(args.v5_scores, "v5 score")
    progress = agmod.StageProgress("Joining accepted observations to existing v5 rows", total=total_rows, unit="rows")
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_after_hkl = 0
    rows_after_domain = 0
    rows_matched = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.v5_scores, usecols=required, chunksize=int(args.chunksize)), start=1):
        rows_read += int(len(chunk))
        work = agmod.actionmod.normalize_key_columns(chunk)
        work = agmod.filter_to_hkls(work, selected_hkls)
        rows_after_hkl += int(len(work))
        if not work.empty:
            for column in ["d_angstrom", "inv_nm", "sg_target", EG_COLUMN, BASELINE_SCORE_COLUMN]:
                work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            work = work.dropna(subset=[EG_COLUMN, BASELINE_SCORE_COLUMN])
            work = work.loc[(work[EG_COLUMN] > 0.0) & (work[EG_COLUMN] <= 1.0)].copy()
            rows_after_domain += int(len(work))
            if not work.empty:
                matched = work.merge(accepted_payload, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
                rows_matched += int(len(matched))
                if not matched.empty:
                    chunks.append(matched.loc[:, required].copy())
        progress.update(rows_read, force=chunk_index == 1)
    progress.finish(rows_read)
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=required)
    duplicate_mask = table.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_keys = int(table.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0])
        raise SystemExit(f"Accepted/v5 join contains duplicate exact observation keys: {duplicate_keys}")
    if not table.empty:
        table = add_exact_key_text(table)
        table[SIGNED_HKL_ID_COLUMN] = [f"{int(h)},{int(k)},{int(l)}" for h, k, l in table.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
        eg = pd.to_numeric(table[EG_COLUMN], errors="coerce").to_numpy(dtype=float)
        clipped = np.clip(eg, np.finfo(float).tiny, 1.0)
        table["Eg_clipped_for_abs_sg_target"] = clipped
        table["Eg_clip_applied_for_abs_sg_target"] = clipped != eg
        table[ABS_SG_TARGET_COLUMN] = BASELINE_SG0 * np.sqrt(-np.log(clipped))
        if not np.isfinite(table[ABS_SG_TARGET_COLUMN].to_numpy(dtype=float)).all():
            raise SystemExit("Nonfinite abs_sg_target values after Eg clipping")
    return table.reset_index(drop=True), {
        "v5_rows_read": int(rows_read),
        "v5_rows_after_manifest_hkl_restriction": int(rows_after_hkl),
        "v5_rows_after_Eg_and_baseline_domain_filter": int(rows_after_domain),
        "accepted_v5_matched_rows": int(rows_matched),
        "accepted_keys_without_valid_v5_row": int(max(0, len(accepted) - len(table))),
        "eg_clipped_for_abs_sg_target_count": int(table["Eg_clip_applied_for_abs_sg_target"].sum()) if not table.empty else 0,
    }


def split_blocks_for_filter(high_pool: pd.DataFrame, filter_cfg: FilterSpec) -> list[pd.DataFrame]:
    ordered = high_pool.sort_values([ABS_SG_TARGET_COLUMN, EXACT_KEY_TEXT_COLUMN], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    block_size = int(filter_cfg.excitation_block_size)
    blocks = [ordered.iloc[start : start + block_size].copy() for start in range(0, len(ordered), block_size)]
    if len(blocks) > 1 and len(blocks[-1]) < int(filter_cfg.min_final_block_size):
        blocks[-2] = pd.concat([blocks[-2], blocks[-1]], ignore_index=True)
        blocks = blocks[:-1]
    if len(blocks) == 1 and len(blocks[0]) < int(filter_cfg.min_final_block_size):
        return []
    return blocks


def construct_hkl_blocks_worker(task: tuple[int, tuple[int, int, int], pd.DataFrame, dict[str, Any]]) -> tuple[int, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    idx, hkl_tuple, group, filter_record = task
    filter_cfg = FilterSpec(**filter_record)
    group = group.copy()
    n_eligible = int(len(group))
    n_high = int(np.floor(float(filter_cfg.high_eg_fraction) * n_eligible))
    hkl_qc: dict[str, Any] = {
        "filter_name": filter_cfg.name,
        "h": hkl_tuple[0],
        "k": hkl_tuple[1],
        "l": hkl_tuple[2],
        "eligible_accepted_observations": n_eligible,
        "high_eg_pool_size": n_high,
        "n_excitation_blocks": 0,
        "included": False,
        "excluded_reason": "",
        "worker_pid": int(os.getpid()),
    }
    if n_high < int(filter_cfg.min_high_eg_observations):
        hkl_qc["excluded_reason"] = "high_eg_pool_below_minimum"
        return int(idx), pd.DataFrame(), hkl_qc, []
    high_pool = group.sort_values([EG_COLUMN, EXACT_KEY_TEXT_COLUMN], ascending=[False, True], kind="mergesort").head(n_high).copy()
    blocks = split_blocks_for_filter(high_pool, filter_cfg)
    if not blocks:
        hkl_qc["excluded_reason"] = "no_excitation_block_meets_minimum_size"
        return int(idx), pd.DataFrame(), hkl_qc, []
    hkl_qc["included"] = True
    hkl_qc["n_excitation_blocks"] = int(len(blocks))
    target_rows: list[pd.DataFrame] = []
    block_rows: list[dict[str, Any]] = []
    for block_id, block in enumerate(blocks, start=1):
        block = block.copy()
        block["filter_name"] = filter_cfg.name
        block[BLOCK_ID_COLUMN] = int(block_id)
        block["block_key"] = f"{hkl_tuple[0]},{hkl_tuple[1]},{hkl_tuple[2]}:{block_id}"
        block["block_size"] = int(len(block))
        target_rows.append(block)
        block_rows.append(
            {
                "filter_name": filter_cfg.name,
                "h": hkl_tuple[0],
                "k": hkl_tuple[1],
                "l": hkl_tuple[2],
                SIGNED_HKL_ID_COLUMN: f"{hkl_tuple[0]},{hkl_tuple[1]},{hkl_tuple[2]}",
                BLOCK_ID_COLUMN: int(block_id),
                "block_key": f"{hkl_tuple[0]},{hkl_tuple[1]},{hkl_tuple[2]}:{block_id}",
                "block_size": int(len(block)),
                "Eg_min": float(block[EG_COLUMN].min()),
                "Eg_median": float(block[EG_COLUMN].median()),
                "Eg_max": float(block[EG_COLUMN].max()),
                "abs_sg_target_min": float(block[ABS_SG_TARGET_COLUMN].min()),
                "abs_sg_target_median": float(block[ABS_SG_TARGET_COLUMN].median()),
                "abs_sg_target_max": float(block[ABS_SG_TARGET_COLUMN].max()),
                "worker_pid": int(os.getpid()),
            }
        )
    return int(idx), pd.concat(target_rows, ignore_index=True), hkl_qc, block_rows


def construct_high_eg_blocks(accepted_v5: pd.DataFrame, manifest: pd.DataFrame, filter_cfg: FilterSpec, workers: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = {tuple(map(int, hkl)): group.copy() for hkl, group in accepted_v5.groupby(HKL_COLUMNS, sort=False)}
    target_rows: list[pd.DataFrame] = []
    hkl_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    progress = agmod.StageProgress(f"Constructing high-Eg pools and excitation blocks for {filter_cfg.name}", total=len(manifest), unit="HKLs")
    tasks = []
    for idx, hkl in enumerate(manifest.loc[:, HKL_COLUMNS].itertuples(index=False, name=None), start=1):
        hkl_tuple = tuple(map(int, hkl))
        group = grouped.get(hkl_tuple, pd.DataFrame(columns=accepted_v5.columns)).copy()
        tasks.append((idx, hkl_tuple, group, asdict(filter_cfg)))
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=agmod.worker_initializer) as executor:
            future_to_idx = {executor.submit(construct_hkl_blocks_worker, task): int(task[0]) for task in tasks}
            for future in as_completed(future_to_idx):
                _idx, target_part, hkl_qc, block_part = future.result()
                if not target_part.empty:
                    target_rows.append(target_part)
                hkl_rows.append(hkl_qc)
                block_rows.extend(block_part)
                progress.advance()
    else:
        for task in tasks:
            _idx, target_part, hkl_qc, block_part = construct_hkl_blocks_worker(task)
            if not target_part.empty:
                target_rows.append(target_part)
            hkl_rows.append(hkl_qc)
            block_rows.extend(block_part)
            progress.advance()
    progress.finish(len(manifest))
    target_table = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
    if not target_table.empty and target_table.duplicated(KEY_COLUMNS, keep=False).any():
        raise SystemExit("High-Eg target table contains duplicate exact observation keys")
    return target_table.reset_index(drop=True), pd.DataFrame.from_records(hkl_rows), pd.DataFrame.from_records(block_rows)


def normalize_frame_group(group: pd.DataFrame) -> pd.DataFrame:
    work = group.copy()
    work["source_filename"] = work["source_filename"].map(agmod.normalize_source)
    work["event"] = work["event"].map(agmod.normalize_event)
    for column in HKL_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[~work[HKL_COLUMNS].isna().any(axis=1)].copy()
    if work.empty:
        return work
    work[HKL_COLUMNS] = work[HKL_COLUMNS].astype("int64")
    for column in ["d_angstrom", "inv_nm", "sg_target", EG_COLUMN]:
        work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = add_exact_key_text(work)
    return work


def score_frame_variants_worker(task: tuple[pd.DataFrame, list[str], int, list[dict[str, Any]]]) -> tuple[int, pd.DataFrame, dict[str, Any]]:
    group, target_keys, target_batch_size, raw_records = task
    raw_specs = [RawKernelSpec(**record) for record in raw_records]
    work = normalize_frame_group(group)
    if work.empty:
        return os.getpid(), pd.DataFrame(), {"n_observations": 0, "target_rows": 0}
    target_key_set = set(target_keys)
    target_mask = work[EXACT_KEY_TEXT_COLUMN].isin(target_key_set).to_numpy(dtype=bool)
    target_positions = np.flatnonzero(target_mask)
    if len(target_positions) == 0:
        return os.getpid(), pd.DataFrame(), {"n_observations": int(len(work)), "target_rows": 0}

    n = int(len(work))
    hkls = work.loc[:, HKL_COLUMNS].to_numpy(dtype=np.int64)
    inv_nm = pd.to_numeric(work["inv_nm"], errors="coerce").to_numpy(dtype=float)
    d_values = pd.to_numeric(work["d_angstrom"], errors="coerce").to_numpy(dtype=float)
    q_invA = np.divide(inv_nm, 10.0, out=np.full_like(inv_nm, np.nan, dtype=float), where=np.isfinite(inv_nm))
    missing_q = ~np.isfinite(q_invA)
    q_invA = np.where(missing_q & np.isfinite(d_values) & (d_values > 0.0), 1.0 / d_values, q_invA)
    metric, metric_stats = v5mod.estimate_reciprocal_metric(hkls, q_invA)
    sg = pd.to_numeric(work["sg_target"], errors="coerce").to_numpy(dtype=float)
    unique_sg = {spec.sg_label: float(spec.sg0) for spec in raw_specs}
    excitation_by_sg = {sg_label: v5mod.excitation_weight_from_sg(sg, sg0) for sg_label, sg0 in unique_sg.items()}
    sigma_specs: dict[str, RawKernelSpec] = {}
    for spec in raw_specs:
        sigma_specs.setdefault(spec.sigma_label, spec)
    raw_by_sigma: dict[str, list[RawKernelSpec]] = {}
    for spec in raw_specs:
        raw_by_sigma.setdefault(spec.sigma_label, []).append(spec)

    output = work.loc[target_positions, KEY_COLUMNS].copy().reset_index(drop=True)
    output[EXACT_KEY_TEXT_COLUMN] = work.loc[target_positions, EXACT_KEY_TEXT_COLUMN].to_numpy(dtype=object)
    for column in sorted({spec.score_column for spec in raw_specs}):
        output[column] = 0.0
    for column in sorted({spec.coupling_sum_column for spec in raw_specs}):
        output[column] = 0.0
    for column in sorted({column for spec in raw_specs for column in aggregate_columns_for_raw(spec).values()}):
        output[column] = 0.0

    target_hkls_all = hkls[target_positions]
    for local_start in range(0, len(target_positions), int(target_batch_size)):
        local_stop = min(local_start + int(target_batch_size), len(target_positions))
        target_hkl = target_hkls_all[local_start:local_stop]
        delta = hkls[None, :, :] - target_hkl[:, None, :]
        nonself = np.any(delta != 0, axis=2)
        dq = v5mod.dq_from_delta(delta, metric)
        for sigma_label, sigma_spec in sigma_specs.items():
            params = v5mod.V5Params(
                sg0=BASELINE_SG0,
                kernel="gaussian",
                sigma_c=float(sigma_spec.sigma_c),
                q0=v5mod.DEFAULT_Q0,
                r_cut=float(sigma_spec.r_cut),
                target_batch_size=int(target_batch_size),
            )
            kernel = v5mod.coupling_kernel(dq, params)
            contributing = nonself & (kernel > 0.0)
            coupling_sum = np.sum(np.where(contributing, kernel, 0.0), axis=1)
            neighbor_count = np.sum(contributing, axis=1).astype(float)
            coupling_sq_sum = np.sum(np.where(contributing, kernel * kernel, 0.0), axis=1)
            output.loc[local_start : local_stop - 1, sigma_spec.coupling_sum_column] = coupling_sum
            for raw_spec in raw_by_sigma[sigma_label]:
                source_eq = excitation_by_sg[raw_spec.sg_label]
                excitation = np.where(contributing, source_eq[None, :], 0.0)
                edge = np.where(contributing, kernel * source_eq[None, :], 0.0)
                aggregate_columns = aggregate_columns_for_raw(raw_spec)
                output.loc[local_start : local_stop - 1, raw_spec.score_column] = np.sum(edge, axis=1)
                output.loc[local_start : local_stop - 1, aggregate_columns["U"]] = np.sum(excitation, axis=1)
                output.loc[local_start : local_stop - 1, aggregate_columns["N"]] = neighbor_count
                output.loc[local_start : local_stop - 1, aggregate_columns["C2"]] = coupling_sq_sum
                output.loc[local_start : local_stop - 1, aggregate_columns["W2"]] = np.sum(edge * edge, axis=1)
                output.loc[local_start : local_stop - 1, aggregate_columns["M2"]] = np.sum(np.where(contributing, source_eq[None, :] * kernel * kernel, 0.0), axis=1)

    stats = {
        "source_filename": str(work["source_filename"].iloc[0]),
        "event": str(work["event"].iloc[0]),
        "n_observations": n,
        "target_rows": int(len(output)),
        **metric_stats,
    }
    return os.getpid(), output, stats


def drain_finished(futures: set[Any], results: list[pd.DataFrame], worker_pids: set[int], progress: agmod.StageProgress, force_all: bool = False) -> set[Any]:
    if not futures:
        return futures
    done: set[Any]
    if force_all:
        done = set(futures)
    else:
        done, _pending = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        pid, frame, _stats = future.result()
        worker_pids.add(int(pid))
        if not frame.empty:
            results.append(frame)
        progress.advance()
    return futures - done


def compute_multivariant_scores(args: argparse.Namespace, high_eg_targets: pd.DataFrame, raw_specs: list[RawKernelSpec]) -> tuple[pd.DataFrame, dict[str, Any], list[int], int]:
    if high_eg_targets.empty:
        raise SystemExit("No high-Eg target rows available for scoring")
    if not raw_specs:
        raise SystemExit("No raw score kernels requested for scoring")
    target_by_frame: dict[str, set[str]] = {}
    for row in high_eg_targets.loc[:, ["source_filename", "event", EXACT_KEY_TEXT_COLUMN]].itertuples(index=False):
        target_by_frame.setdefault(frame_key(row.source_filename, row.event), set()).add(str(row.exact_key_text))
    target_frame_total = int(len(target_by_frame))
    raw_records = [asdict(spec) for spec in raw_specs]
    raw_columns = sorted({spec.score_column for spec in raw_specs})
    coupling_columns = sorted({spec.coupling_sum_column for spec in raw_specs})
    aggregate_columns = sorted({column for spec in raw_specs for column in aggregate_columns_for_raw(spec).values()})
    log(f"Requested worker count: {int(args.workers):,}")
    log(f"Actual worker count: {int(args.workers):,}")
    progress = agmod.StageProgress(
        f"Enumerating frame neighbors and computing {len(raw_columns):,} raw score column(s)",
        total=target_frame_total,
        unit="frames",
    )
    usecols = [*KEY_COLUMNS, "d_angstrom", "inv_nm", "sg_target", EG_COLUMN]
    results: list[pd.DataFrame] = []
    worker_pids: set[int] = set()
    submitted = 0
    agmod.set_worker_numeric_threads()
    if int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=agmod.worker_initializer) as executor:
            futures: set[Any] = set()
            for group in v5mod.iter_frame_groups(args.v5_scores, usecols, int(args.chunksize), None, None):
                work = normalize_frame_group(group)
                if work.empty:
                    continue
                fkey = frame_key(work["source_filename"].iloc[0], work["event"].iloc[0])
                if fkey not in target_by_frame:
                    continue
                futures.add(executor.submit(score_frame_variants_worker, (work, sorted(target_by_frame[fkey]), int(args.target_batch_size), raw_records)))
                submitted += 1
                if len(futures) >= max(1, int(args.workers) * 2):
                    futures = drain_finished(futures, results, worker_pids, progress)
            while futures:
                futures = drain_finished(futures, results, worker_pids, progress, force_all=True)
    else:
        for group in v5mod.iter_frame_groups(args.v5_scores, usecols, int(args.chunksize), None, None):
            work = normalize_frame_group(group)
            if work.empty:
                continue
            fkey = frame_key(work["source_filename"].iloc[0], work["event"].iloc[0])
            if fkey not in target_by_frame:
                continue
            pid, frame, _stats = score_frame_variants_worker((work, sorted(target_by_frame[fkey]), int(args.target_batch_size), raw_records))
            worker_pids.add(int(pid))
            if not frame.empty:
                results.append(frame)
            submitted += 1
            progress.advance()
    progress.finish(submitted)
    if submitted != target_frame_total:
        raise SystemExit(f"Only found {submitted:,} target frames in v5 score geometry input; expected {target_frame_total:,}")
    score_table = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=[*KEY_COLUMNS, EXACT_KEY_TEXT_COLUMN, *raw_columns, *coupling_columns, *aggregate_columns])
    duplicate_mask = score_table.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise SystemExit("Recomputed score table contains duplicate exact observation keys")
    missing = set(high_eg_targets[EXACT_KEY_TEXT_COLUMN]) - set(score_table[EXACT_KEY_TEXT_COLUMN])
    if missing:
        raise SystemExit(f"Recomputed scores missing {len(missing):,} high-Eg target keys")
    for column in [*raw_columns, *coupling_columns, *aggregate_columns]:
        values = pd.to_numeric(score_table[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            raise SystemExit(f"Nonfinite recomputed values in {column}")
    return score_table, {"target_frames_scored": int(submitted), "target_rows_scored": int(len(score_table))}, sorted(worker_pids), int(args.workers)


def is_integer_power(value: float) -> bool:
    return bool(np.isclose(float(value), round(float(value)), rtol=0.0, atol=1.0e-12))


def safe_divide_values(numerator: np.ndarray, denominator: np.ndarray, label: str, spec: ScoreSpec) -> tuple[np.ndarray, dict[str, int]]:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if not np.isfinite(num).all() or not np.isfinite(den).all():
        raise SystemExit(f"Nonfinite numerator or denominator for {label} in {spec.name}")
    out = np.zeros_like(num, dtype=float)
    positive = den > 0.0
    zero_zero = (den == 0.0) & (num == 0.0)
    invalid = ~(positive | zero_zero)
    stats = {
        f"{label}_zero_denominator_count": int((den == 0.0).sum()),
        f"{label}_zero_over_zero_count": int(zero_zero.sum()),
        f"{label}_invalid_division_count": int(invalid.sum()),
    }
    if invalid.any():
        raise SystemExit(f"Invalid division for {label} in {spec.name}: {stats}")
    out[positive] = num[positive] / den[positive]
    if not np.isfinite(out).all():
        raise SystemExit(f"Nonfinite division result for {label} in {spec.name}: {stats}")
    return out, stats


def pair_mass_values(left: np.ndarray, right: np.ndarray, label: str, spec: ScoreSpec) -> tuple[np.ndarray, dict[str, int]]:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise SystemExit(f"Nonfinite pair-mass inputs for {label} in {spec.name}")
    raw = lhs - rhs
    scale = np.maximum(1.0, np.maximum(np.abs(lhs), np.abs(rhs)))
    negative = raw < 0.0
    clip = negative & (np.abs(raw) <= 1.0e-12 * scale)
    invalid = negative & ~clip
    stats = {
        f"{label}_clipped_roundoff_count": int(clip.sum()),
        f"{label}_invalid_negative_count": int(invalid.sum()),
    }
    if invalid.any():
        raise SystemExit(f"{label} contains negative values beyond roundoff tolerance in {spec.name}: {stats}")
    out = raw.copy()
    out[clip] = 0.0
    if not np.isfinite(out).all():
        raise SystemExit(f"Nonfinite pair-mass result for {label} in {spec.name}: {stats}")
    return out, stats


def score_components(table: pd.DataFrame, spec: ScoreSpec) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = pd.to_numeric(table[spec.raw_score_column], errors="coerce").to_numpy(dtype=float)
    coupling = pd.to_numeric(table[spec.coupling_sum_column], errors="coerce").to_numpy(dtype=float)
    eg = pd.to_numeric(table[EG_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise SystemExit(f"Nonfinite raw scores for {spec.name}")
    if not np.isfinite(coupling).all():
        raise SystemExit(f"Nonfinite coupling sums for {spec.name}")
    if not np.isfinite(eg).all():
        raise SystemExit(f"Nonfinite target excitation values for {spec.name}")
    if (coupling < 0.0).any():
        raise SystemExit(f"Negative coupling sums for {spec.name}")
    nonzero_raw_at_zero_coupling = (coupling <= 0.0) & (raw != 0.0)
    if nonzero_raw_at_zero_coupling.any():
        raise SystemExit(f"Raw score is nonzero where coupling sum is zero for {spec.name}")
    a_values = np.zeros_like(raw, dtype=float)
    positive = coupling > 0.0
    if positive.any():
        a_values[positive] = raw[positive] / coupling[positive]
    if not np.isfinite(a_values).all():
        raise SystemExit(f"Nonfinite A = raw / coupling_sum values for {spec.name}")
    delta = a_values - eg
    ad4 = a_values * np.power(coupling, 4.0)
    components: dict[str, np.ndarray] = {"raw": raw, "M": raw, "D": coupling, "Eg": eg, "A": a_values, "delta": delta, "AD4": ad4}
    aggregate_columns = aggregate_columns_for_raw(spec.raw_name)
    for variable, column in aggregate_columns.items():
        if column in table.columns:
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SystemExit(f"Nonfinite aggregate {variable} values for {spec.name}")
            components[variable] = values
    stats = {
        "zero_coupling_count": int((coupling == 0.0).sum()),
        "nonzero_raw_at_zero_coupling_count": int(nonzero_raw_at_zero_coupling.sum()),
        "zero_A_count": int((a_values == 0.0).sum()),
        "zero_D_count": int((coupling == 0.0).sum()),
        "zero_Eg_count": int((eg == 0.0).sum()),
    }
    if {"U", "N"}.issubset(components):
        q_num = components["N"] * raw
        q_den = components["U"] * coupling
        components["Q"], q_stats = safe_divide_values(q_num, q_den, "Q", spec)
        stats.update(q_stats)
    if "W2" in components:
        components["Neff"], neff_stats = safe_divide_values(raw * raw, components["W2"], "Neff", spec)
        stats.update(neff_stats)
    if "C2" in components:
        components["Pc"], pc_stats = pair_mass_values(coupling * coupling, components["C2"], "Pc", spec)
        stats.update(pc_stats)
    if "W2" in components:
        components["Pe"], pe_stats = pair_mass_values(raw * raw, components["W2"], "Pe", spec)
        stats.update(pe_stats)
    return components, stats


def checked_power(base: np.ndarray, power: float, label: str, spec: ScoreSpec) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(base, dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit(f"Nonfinite base values for {label} in {spec.name}")
    exponent = float(power)
    if not np.isfinite(exponent):
        raise SystemExit(f"Nonfinite exponent for {label} in {spec.name}")
    zero_base = values == 0.0
    invalid = np.zeros_like(values, dtype=bool)
    if exponent < 0.0:
        invalid |= zero_base
    if not is_integer_power(exponent):
        invalid |= values < 0.0
    stats = {
        f"{label}_zero_base_count": int(zero_base.sum()),
        f"{label}_invalid_power_count": int(invalid.sum()),
    }
    if invalid.any():
        raise SystemExit(f"Invalid power operation for {label} in {spec.name}: {stats}")
    if np.isclose(exponent, 0.0, rtol=0.0, atol=0.0):
        out = np.ones_like(values, dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = np.power(values, exponent)
    if not np.isfinite(out).all():
        raise SystemExit(f"Nonfinite power result for {label}^{exponent} in {spec.name}: {stats}")
    return out, stats


def evaluate_expression_tree(node: Any, components: dict[str, np.ndarray], spec: ScoreSpec, stats: dict[str, Any], label: str = "expr") -> np.ndarray:
    if not isinstance(node, dict):
        raise SystemExit(f"Invalid expression node for {spec.name}: {label}")
    if "var" in node:
        variable = str(node["var"])
        if variable not in components:
            raise SystemExit(f"Expression {spec.name} requires unavailable variable {variable}")
        return components[variable]
    if "const" in node:
        template = next(iter(components.values()))
        return np.full_like(template, float(node["const"]), dtype=float)
    op = str(node.get("op", ""))
    if op == "mul":
        args = [evaluate_expression_tree(arg, components, spec, stats, f"{label}.mul[{idx}]") for idx, arg in enumerate(node["args"])]
        out = np.ones_like(args[0], dtype=float)
        for arg in args:
            out = out * arg
    elif op == "add":
        args = [evaluate_expression_tree(arg, components, spec, stats, f"{label}.add[{idx}]") for idx, arg in enumerate(node["args"])]
        out = np.zeros_like(args[0], dtype=float)
        for arg in args:
            out = out + arg
    elif op == "sub":
        out = evaluate_expression_tree(node["left"], components, spec, stats, f"{label}.sub.left") - evaluate_expression_tree(node["right"], components, spec, stats, f"{label}.sub.right")
    elif op == "div":
        numerator = evaluate_expression_tree(node["numerator"], components, spec, stats, f"{label}.div.numerator")
        denominator = evaluate_expression_tree(node["denominator"], components, spec, stats, f"{label}.div.denominator")
        out, div_stats = safe_divide_values(numerator, denominator, label.replace(".", "_"), spec)
        stats.update(div_stats)
    elif op == "pow":
        base = evaluate_expression_tree(node["base"], components, spec, stats, f"{label}.pow.base")
        out, pow_stats = checked_power(base, float(node["exponent"]), label.replace(".", "_"), spec)
        stats.update(pow_stats)
    elif op == "sqrt":
        arg = evaluate_expression_tree(node["arg"], components, spec, stats, f"{label}.sqrt.arg")
        negative = arg < 0.0
        stats[f"{label.replace('.', '_')}_sqrt_negative_count"] = int(negative.sum())
        if negative.any():
            raise SystemExit(f"Negative sqrt argument for {spec.name}: {label}")
        out = np.sqrt(arg)
    elif op == "min":
        left = evaluate_expression_tree(node["args"][0], components, spec, stats, f"{label}.min.left")
        right = evaluate_expression_tree(node["args"][1], components, spec, stats, f"{label}.min.right")
        out = np.minimum(left, right)
    else:
        raise SystemExit(f"Unsupported expression operation for {spec.name}: {op}")
    if not np.isfinite(out).all():
        raise SystemExit(f"Nonfinite expression result for {spec.name}: {label}")
    return out


def derived_score_values(table: pd.DataFrame, spec: ScoreSpec, *, return_stats: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    components, stats = score_components(table, spec)
    eg_term, eg_stats = checked_power(components["Eg"], spec.eg_power, "Eg", spec)
    d_term, d_stats = checked_power(components["D"], spec.d_power, "D", spec)
    stats.update(eg_stats)
    stats.update(d_stats)
    if spec.formula_kind == "expression_tree":
        out = evaluate_expression_tree(spec.expression_tree, components, spec, stats)
    elif spec.formula_kind == "power_product":
        a_term, a_stats = checked_power(components["A"], spec.a_power, "A", spec)
        stats.update(a_stats)
        out = eg_term * a_term * d_term
    elif spec.formula_kind == "imbalance_absolute":
        imbalance_term, imbalance_stats = checked_power(np.abs(components["delta"]), spec.imbalance_power, "abs_delta", spec)
        stats.update(imbalance_stats)
        out = eg_term * imbalance_term * d_term
    elif spec.formula_kind == "imbalance_feed":
        out = eg_term * components["delta"] * d_term
    elif spec.formula_kind == "imbalance_sink":
        out = eg_term * (-components["delta"]) * d_term
    else:
        raise SystemExit(f"Unknown derived score kind for {spec.name}: {spec.formula_kind}")
    stats["score_zero_count"] = int((out == 0.0).sum())
    stats["score_positive_count"] = int((out > 0.0).sum())
    stats["score_negative_count"] = int((out < 0.0).sum())
    stats["score_nonfinite_count"] = int((~np.isfinite(out)).sum())
    if stats["score_nonfinite_count"]:
        raise SystemExit(f"Nonfinite derived score values for {spec.name}: {stats}")
    return (out, stats) if return_stats else out


def alpha_score_values(table: pd.DataFrame, spec: ScoreSpec) -> np.ndarray:
    if spec.score_type == "derived":
        return derived_score_values(table, spec)  # type: ignore[return-value]
    raw = pd.to_numeric(table[spec.raw_score_column], errors="coerce").to_numpy(dtype=float)
    coupling = pd.to_numeric(table[spec.coupling_sum_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise SystemExit(f"Nonfinite raw scores for {spec.name}")
    if not np.isfinite(coupling).all():
        raise SystemExit(f"Nonfinite coupling sums for {spec.name}")
    if (coupling < 0.0).any():
        raise SystemExit(f"Negative coupling sums for {spec.name}")
    nonzero_raw_at_zero_coupling = (coupling <= 0.0) & (raw != 0.0)
    if nonzero_raw_at_zero_coupling.any():
        raise SystemExit(f"Raw score is nonzero where coupling sum is zero for {spec.name}")
    out = np.zeros_like(raw, dtype=float)
    positive = coupling > 0.0
    if positive.any():
        out[positive] = raw[positive] / np.power(coupling[positive], float(spec.alpha))
    if not np.isfinite(out).all():
        raise SystemExit(f"Nonfinite alpha-normalized scores for {spec.name}")
    return out


def validate_baseline_score_reproduction(cache: pd.DataFrame, args: argparse.Namespace, baseline_score: ScoreSpec) -> dict[str, Any]:
    progress = agmod.StageProgress("Baseline score comparison", total=1, unit="checks")
    old = pd.to_numeric(cache[BASELINE_SCORE_COLUMN], errors="coerce").to_numpy(dtype=float)
    new = pd.to_numeric(cache[baseline_score.raw_score_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(old) & np.isfinite(new)
    if not finite.all():
        raise SystemExit("Baseline score comparison contains nonfinite values")
    diff = np.abs(new - old)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rel = diff / np.maximum(np.abs(old), np.finfo(float).tiny)
    finite_rel = rel[np.isfinite(rel)]
    close = np.isclose(new, old, rtol=float(args.score_rtol), atol=float(args.score_atol))
    stats = {
        "observations_compared": int(len(cache)),
        "max_absolute_difference": float(np.max(diff)) if len(diff) else None,
        "median_absolute_difference": float(np.median(diff)) if len(diff) else None,
        "max_relative_difference": float(np.max(finite_rel)) if len(finite_rel) else None,
        "median_relative_difference": float(np.median(finite_rel)) if len(finite_rel) else None,
        "relative_difference_had_nonfinite_values": bool(len(finite_rel) != len(rel)),
        "pearson_correlation": finite_corr(pd.Series(new), pd.Series(old), "pearson"),
        "spearman_rank_correlation": finite_corr(pd.Series(new), pd.Series(old), "spearman"),
        "fraction_passing_np_isclose": float(np.mean(close)) if len(close) else None,
        "rtol": float(args.score_rtol),
        "atol": float(args.score_atol),
    }
    progress.finish(1)
    if not bool(np.all(close)):
        raise SystemExit(
            "Recomputed sg100_sc100 score is not numerically equivalent to accepted "
            f"{BASELINE_SCORE_COLUMN}: {stats}"
        )
    return stats


def block_gate_threshold(ranges: list[float], filter_cfg: FilterSpec) -> float | None:
    if filter_cfg.block_gate_mode == "none":
        return None
    if filter_cfg.block_gate_mode == "absolute_range":
        if filter_cfg.block_range_threshold is None:
            raise SystemExit("absolute_range gate missing threshold")
        return float(filter_cfg.block_range_threshold)
    if filter_cfg.block_gate_mode == "range_quantile":
        if filter_cfg.block_range_quantile is None:
            raise SystemExit("range_quantile gate missing quantile")
        if not ranges:
            return float("inf")
        return float(np.quantile(np.asarray(ranges, dtype=float), float(filter_cfg.block_range_quantile)))
    raise SystemExit(f"Unknown block gate mode: {filter_cfg.block_gate_mode}")


def define_common_actionable_blocks(cache: pd.DataFrame, filter_cfg: FilterSpec, baseline_score: ScoreSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress = agmod.StageProgress(f"Identifying baseline-actionable blocks for {filter_cfg.name}", total=1, unit="passes")
    work = cache.copy()
    grouped = work.groupby([*HKL_COLUMNS, BLOCK_ID_COLUMN], sort=False)
    baseline_ranges: list[float] = []
    for _key, group in grouped:
        scores = pd.to_numeric(group[baseline_score.raw_score_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise SystemExit("Nonfinite baseline raw scores while defining actionable blocks")
        baseline_ranges.append(float(np.max(scores) - np.min(scores)) if len(scores) else 0.0)
    threshold = block_gate_threshold(baseline_ranges, filter_cfg)
    rows: list[dict[str, Any]] = []
    common_keys: set[tuple[int, int, int, int]] = set()
    for key, group in grouped:
        h, k, l, block_id = int(key[0]), int(key[1]), int(key[2]), int(key[3])
        scores = pd.to_numeric(group[baseline_score.raw_score_column], errors="coerce").to_numpy(dtype=float)
        score_range = float(np.max(scores) - np.min(scores)) if len(scores) else 0.0
        if filter_cfg.block_gate_mode == "none":
            actionable = bool(score_range > 0.0)
        else:
            actionable = bool(score_range >= float(threshold))
        if actionable:
            common_keys.add((h, k, l, block_id))
        rows.append(
            {
                "filter_name": filter_cfg.name,
                "h": h,
                "k": k,
                "l": l,
                SIGNED_HKL_ID_COLUMN: f"{h},{k},{l}",
                BLOCK_ID_COLUMN: block_id,
                "block_size": int(len(group)),
                "baseline_score_min": float(np.min(scores)) if len(scores) else None,
                "baseline_score_median": float(np.median(scores)) if len(scores) else None,
                "baseline_score_max": float(np.max(scores)) if len(scores) else None,
                "baseline_score_range": score_range,
                "block_gate_mode": filter_cfg.block_gate_mode,
                "block_gate_threshold": threshold,
                COMMON_BLOCK_COLUMN: actionable,
                "n_remove": highmod.removal_count_for_block(int(len(group)), filter_cfg.drop_fraction, filter_cfg.min_remaining_per_block) if actionable else 0,
            }
        )
    block_qc = pd.DataFrame.from_records(rows)
    work[COMMON_BLOCK_COLUMN] = [
        (int(row.h), int(row.k), int(row.l), int(row.excitation_block_id)) in common_keys
        for row in work.loc[:, [*HKL_COLUMNS, BLOCK_ID_COLUMN]].itertuples(index=False)
    ]
    progress.finish(1)
    bad = block_qc.loc[block_qc[COMMON_BLOCK_COLUMN].astype(bool) & (block_qc["n_remove"].astype(int) < 1)]
    if not bad.empty:
        raise SystemExit(f"At least one actionable block cannot remove one observation under filter {filter_cfg.name}")
    return work, block_qc


def write_score_cache(path: Path, cache: pd.DataFrame, plan: SweepPlan) -> None:
    progress = agmod.StageProgress("Writing score cache", total=1, unit="files")
    ordered_columns = [
        "source_filename",
        "event",
        *HKL_COLUMNS,
        EXACT_KEY_TEXT_COLUMN,
        EG_COLUMN,
        ABS_SG_TARGET_COLUMN,
        SIGNED_HKL_ID_COLUMN,
        BASELINE_SCORE_COLUMN,
        *required_cache_columns(plan),
        "block_size",
        "block_key",
        "Eg_clipped_for_abs_sg_target",
        "Eg_clip_applied_for_abs_sg_target",
    ]
    keep = [column for column in ordered_columns if column in cache.columns]
    cache.loc[:, keep].to_csv(path, index=False, compression="gzip", float_format=CACHE_FLOAT_FORMAT)
    progress.finish(1)


def read_score_cache(path: Path, plan: SweepPlan) -> pd.DataFrame:
    progress = agmod.StageProgress("Reading score cache", total=1, unit="files")
    table = pd.read_csv(path, low_memory=False, compression="gzip")
    progress.finish(1)
    required = [
        "source_filename",
        "event",
        *HKL_COLUMNS,
        EXACT_KEY_TEXT_COLUMN,
        EG_COLUMN,
        ABS_SG_TARGET_COLUMN,
        SIGNED_HKL_ID_COLUMN,
        BASELINE_SCORE_COLUMN,
    ]
    agmod.require_columns(table, required, "cached high-Eg multi-score table")
    table = agmod.actionmod.normalize_key_columns(table)
    table[EXACT_KEY_TEXT_COLUMN] = key_text_from_frame(table)
    return table


def build_or_load_score_cache(
    args: argparse.Namespace,
    plan: SweepPlan,
    score_targets: pd.DataFrame,
    expected_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[int], int, bool]:
    cpath = cache_path(args.out_dir)
    score_targets = score_targets.sort_values([*KEY_COLUMNS, EXACT_KEY_TEXT_COLUMN], kind="mergesort").drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    if score_targets.empty:
        raise SystemExit("Sweep filter construction produced no target rows for scoring")

    base_columns = [
        "source_filename",
        "event",
        *HKL_COLUMNS,
        EXACT_KEY_TEXT_COLUMN,
        EG_COLUMN,
        ABS_SG_TARGET_COLUMN,
        SIGNED_HKL_ID_COLUMN,
        BASELINE_SCORE_COLUMN,
        "Eg_clipped_for_abs_sg_target",
        "Eg_clip_applied_for_abs_sg_target",
    ]
    base = score_targets.loc[:, [column for column in base_columns if column in score_targets.columns]].copy()
    needed_columns = required_cache_columns(plan)

    source_dir = args.score_cache_dir if args.score_cache_dir is not None else args.out_dir
    source_cache_path = None if args.recompute_cache else existing_cache_path(source_dir)
    previous_cache: pd.DataFrame | None = None
    previous_metadata: dict[str, Any] | None = None
    reused_columns: list[str] = []
    if source_cache_path is not None:
        previous_metadata = load_previous_cache_metadata(source_dir)
        if previous_metadata is None:
            raise SystemExit(f"Found cache without sweep_audit.json cache metadata: {source_cache_path}")
        ok, mismatches = metadata_matches(expected_metadata, previous_metadata)
        if not ok:
            raise SystemExit(f"Cache metadata mismatch in section(s): {mismatches}. Use --recompute-cache if intentional.")
        previous_cache = read_score_cache(source_cache_path, plan)
        if previous_cache.duplicated(KEY_COLUMNS, keep=False).any():
            raise SystemExit(f"Cached score table contains duplicate exact keys: {source_cache_path}")
        available_columns = [column for column in needed_columns if column in previous_cache.columns]
        if available_columns:
            previous_payload = previous_cache.loc[:, [*KEY_COLUMNS, *available_columns]].copy()
            cache = base.merge(previous_payload, on=KEY_COLUMNS, how="left", sort=False, validate="one_to_one")
            reused_columns = [
                column
                for column in available_columns
                if column in cache.columns
                and not pd.to_numeric(cache[column], errors="coerce").replace([np.inf, -np.inf], np.nan).isna().any()
            ]
        else:
            cache = base.copy()
    else:
        cache = base.copy()

    missing_columns = []
    for column in needed_columns:
        if column not in cache.columns:
            missing_columns.append(column)
            continue
        values = pd.to_numeric(cache[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            missing_columns.append(column)
    missing_columns = sorted(set(missing_columns))
    raw_specs_to_compute = [
        spec
        for spec in plan.raw_specs
        if spec.score_column in missing_columns
        or spec.coupling_sum_column in missing_columns
        or any(column in missing_columns for column in aggregate_columns_for_raw(spec).values())
    ]

    worker_pids: list[int] = []
    actual_workers = 0
    score_stats: dict[str, Any] = {"target_frames_scored": 0, "target_rows_scored": 0, "computed_raw_kernel_count": 0}
    if raw_specs_to_compute:
        score_table, score_stats, worker_pids, actual_workers = compute_multivariant_scores(args, score_targets, raw_specs_to_compute)
        computed_columns = [
            column
            for column in score_table.columns
            if column not in [*KEY_COLUMNS, EXACT_KEY_TEXT_COLUMN] and column in missing_columns
        ]
        score_payload = score_table.loc[:, [*KEY_COLUMNS, *computed_columns]].copy()
        cache = cache.drop(columns=[column for column in computed_columns if column in cache.columns], errors="ignore")
        cache = cache.merge(score_payload, on=KEY_COLUMNS, how="left", sort=False, validate="one_to_one")
        for column in computed_columns:
            values = pd.to_numeric(cache[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.isna().any():
                raise SystemExit(f"Missing recomputed values after cache merge for {column}: {int(values.isna().sum())}")
        score_stats["computed_raw_kernel_count"] = int(len(raw_specs_to_compute))
    else:
        actual_workers = 0

    for column in needed_columns:
        if column not in cache.columns:
            raise SystemExit(f"Score cache missing required column after completion: {column}")
        values = pd.to_numeric(cache[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            raise SystemExit(f"Score cache contains nonfinite required values in {column}")

    baseline_stats = validate_baseline_score_reproduction(cache, args, plan.internal_baseline_score)
    write_score_cache(cpath, cache, plan)
    stats = {
        "cache_reused": bool(source_cache_path is not None and not raw_specs_to_compute),
        "cache_completed_from_compatible_source": bool(source_cache_path is not None),
        "source_cache_path": str(source_cache_path) if source_cache_path is not None else None,
        "cache_path": str(cpath),
        "rows": int(len(cache)),
        "scoring": score_stats,
        "reused_columns": reused_columns,
        "computed_columns": missing_columns,
    }
    return cache, stats, baseline_stats, worker_pids, actual_workers, bool(source_cache_path is not None and not raw_specs_to_compute)


def block_removal_count(n_block: int, filter_cfg: FilterSpec) -> int:
    return highmod.removal_count_for_block(int(n_block), float(filter_cfg.drop_fraction), int(filter_cfg.min_remaining_per_block))


def selection_block_worker(task: tuple[int, tuple[Any, ...], pd.DataFrame, dict[str, Any], list[dict[str, Any]]]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    index, block_key, block, filter_record, score_records = task
    filter_cfg = FilterSpec(**filter_record)
    score_specs = [ScoreSpec(**record) for record in score_records]
    h, k, l, block_id = int(block_key[0]), int(block_key[1]), int(block_key[2]), int(block_key[3])
    block = block.copy()
    n_remove = block_removal_count(int(len(block)), filter_cfg)
    if n_remove < 1:
        raise SystemExit(f"Common block {h,k,l} block {block_id} cannot remove one observation")
    selected_rows: list[dict[str, Any]] = []
    per_block_rows: list[dict[str, Any]] = []
    for spec in score_specs:
        experiment = f"{spec.name}_{filter_cfg.name}"
        scores_array = alpha_score_values(block, spec)
        scores = pd.Series(scores_array, index=block.index)
        score_range = float(scores.max() - scores.min()) if len(scores) else 0.0
        ranked = block.copy()
        ranked["candidate_score"] = scores_array
        ordered = ranked.sort_values(["candidate_score", ABS_SG_TARGET_COLUMN, EXACT_KEY_TEXT_COLUMN], ascending=[False, True, True], kind="mergesort")
        chosen = ordered.head(n_remove).copy()
        tied_rows = int(scores.duplicated(keep=False).sum())
        zero_rows = int((scores == 0.0).sum())
        per_block_rows.append(
            {
                "variant": experiment,
                "score_name": spec.name,
                "filter_name": filter_cfg.name,
                "score_type": spec.score_type,
                "formula_kind": spec.formula_kind,
                "score_family": spec.score_family,
                "formula": spec.formula,
                "p": spec.p_value,
                "lambda": spec.lambda_value,
                "eg_power": float(spec.eg_power),
                "a_power": float(spec.a_power),
                "d_power": float(spec.d_power),
                "imbalance_power": float(spec.imbalance_power),
                "h": h,
                "k": k,
                "l": l,
                SIGNED_HKL_ID_COLUMN: f"{h},{k},{l}",
                BLOCK_ID_COLUMN: block_id,
                "block_size": int(len(block)),
                "n_remove": int(n_remove),
                "actual_removal_count": int(len(chosen)),
                "score_min": float(scores.min()),
                "score_median": float(scores.median()),
                "score_max": float(scores.max()),
                "score_range": score_range,
                "zero_candidate_score_range": bool(score_range == 0.0),
                "tied_score_rows": tied_rows,
                "tied_score_fraction": float(tied_rows / max(1, len(block))),
                "zero_score_rows": zero_rows,
                "zero_score_fraction": float(zero_rows / max(1, len(block))),
            }
        )
        for rank, (_idx, row) in enumerate(chosen.iterrows(), start=1):
            selected_rows.append(
                {
                    "variant": experiment,
                    "score_name": spec.name,
                    "filter_name": filter_cfg.name,
                    "sg0": float(spec.sg0),
                    "sg_multiplier": float(spec.sg_multiplier),
                    "sigma_c": float(spec.sigma_c),
                    "sigma_multiplier": float(spec.sigma_multiplier),
                    "r_cut": float(spec.r_cut),
                    "alpha": float(spec.alpha),
                    "score_type": spec.score_type,
                    "formula_kind": spec.formula_kind,
                    "score_family": spec.score_family,
                    "formula": spec.formula,
                    "p": spec.p_value,
                    "lambda": spec.lambda_value,
                    "eg_power": float(spec.eg_power),
                    "a_power": float(spec.a_power),
                    "d_power": float(spec.d_power),
                    "imbalance_power": float(spec.imbalance_power),
                    "fraction": float(filter_cfg.drop_fraction),
                    "drop_label": fraction_label("drop", filter_cfg.drop_fraction),
                    "h": h,
                    "k": k,
                    "l": l,
                    SIGNED_HKL_ID_COLUMN: f"{h},{k},{l}",
                    BLOCK_ID_COLUMN: block_id,
                    "block_size": int(len(block)),
                    "n_remove": int(n_remove),
                    "selection_rank_in_block": int(rank),
                    "source_filename": row["source_filename"],
                    "event": row["event"],
                    EXACT_KEY_TEXT_COLUMN: row[EXACT_KEY_TEXT_COLUMN],
                    EG_COLUMN: float(row[EG_COLUMN]),
                    ABS_SG_TARGET_COLUMN: float(row[ABS_SG_TARGET_COLUMN]),
                    "candidate_score": float(row["candidate_score"]),
                    "raw_score_column": spec.raw_score_column,
                    "coupling_sum_column": spec.coupling_sum_column,
                }
            )
    return int(index), selected_rows, per_block_rows


def select_variant_removals(cache: pd.DataFrame, filter_cfg: FilterSpec, score_specs: tuple[ScoreSpec, ...], workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = cache.loc[cache[COMMON_BLOCK_COLUMN].astype(bool)].copy()
    if common.empty:
        raise SystemExit(f"No baseline-actionable blocks were found for filter {filter_cfg.name}")
    grouped_blocks = list(common.groupby([*HKL_COLUMNS, BLOCK_ID_COLUMN], sort=False))
    progress = agmod.StageProgress(
        f"Selecting {len(score_specs):,} score removal set(s) from frozen blocks for {filter_cfg.name}",
        total=len(grouped_blocks),
        unit="blocks",
    )
    selected_rows: list[dict[str, Any]] = []
    per_block_rows: list[dict[str, Any]] = []
    tasks = [
        (idx, block_key, block.copy(), asdict(filter_cfg), [asdict(spec) for spec in score_specs])
        for idx, (block_key, block) in enumerate(grouped_blocks)
    ]
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=agmod.worker_initializer) as executor:
            future_to_idx = {executor.submit(selection_block_worker, task): int(task[0]) for task in tasks}
            for future in as_completed(future_to_idx):
                _idx, selected_part, per_block_part = future.result()
                selected_rows.extend(selected_part)
                per_block_rows.extend(per_block_part)
                progress.advance()
    else:
        for task in tasks:
            _idx, selected_part, per_block_part = selection_block_worker(task)
            selected_rows.extend(selected_part)
            per_block_rows.extend(per_block_part)
            progress.advance()
    progress.finish(len(grouped_blocks))
    selected = pd.DataFrame.from_records(selected_rows)
    per_block = pd.DataFrame.from_records(per_block_rows)
    per_hkl = (
        selected.groupby(["variant", "score_name", "filter_name", *HKL_COLUMNS, SIGNED_HKL_ID_COLUMN], sort=False)
        .size()
        .reset_index(name="actual_removal_count")
    )
    return selected, per_block, per_hkl


def validate_equal_counts(selected: pd.DataFrame, per_block: pd.DataFrame, per_hkl: pd.DataFrame) -> None:
    progress = agmod.StageProgress("Validating equal block/HKL/total counts within each filter", total=3, unit="checks")
    for filter_name, block_group in per_block.groupby("filter_name", sort=False):
        block_counts = block_group.pivot_table(index=[*HKL_COLUMNS, BLOCK_ID_COLUMN], columns="variant", values="actual_removal_count", aggfunc="sum")
        if not block_counts.apply(lambda row: row.nunique(dropna=False) == 1, axis=1).all():
            raise SystemExit(f"Unequal removal counts across score variants in at least one block for filter {filter_name}")
    progress.advance()
    for filter_name, hkl_group in per_hkl.groupby("filter_name", sort=False):
        hkl_counts = hkl_group.pivot_table(index=HKL_COLUMNS, columns="variant", values="actual_removal_count", aggfunc="sum")
        if not hkl_counts.apply(lambda row: row.nunique(dropna=False) == 1, axis=1).all():
            raise SystemExit(f"Unequal removal counts across score variants for at least one signed HKL for filter {filter_name}")
    progress.advance()
    for filter_name, selected_group in selected.groupby("filter_name", sort=False):
        totals = selected_group.groupby("variant", sort=False).size()
        if int(totals.nunique(dropna=False)) != 1:
            raise SystemExit(f"Unequal total removal counts across score variants for filter {filter_name}")
    for variant, group in selected.groupby("variant", sort=False):
        if len(key_set(group)) != len(group):
            raise SystemExit(f"Duplicate exact removal keys for variant {variant}")
    progress.advance()
    progress.finish(3)


def validate_expected_removal_count(selected: pd.DataFrame, expected_count: int | None) -> dict[str, Any]:
    if expected_count is None:
        return {"status": "skipped", "reason": "no --expected-removal-count supplied"}
    progress = agmod.StageProgress("Validating expected total removal count", total=1, unit="checks")
    totals = selected.groupby("variant", sort=False).size()
    bad = totals.loc[totals.astype(int) != int(expected_count)]
    progress.finish(1)
    if not bad.empty:
        raise SystemExit(f"Removal count mismatch against expected {int(expected_count):,}: {bad.to_dict()}")
    return {
        "status": "passed",
        "expected_removal_count": int(expected_count),
        "variant_count": int(len(totals)),
    }


def load_baseline_drop30_keys(path: Path) -> tuple[set[tuple[str, str, int, int, int]], pd.DataFrame]:
    table = pd.read_csv(path, low_memory=False)
    agmod.require_columns(table, KEY_COLUMNS, "baseline selected-removal table")
    work = agmod.actionmod.normalize_key_columns(table)
    mask = pd.Series(True, index=work.index)
    if "mode" in work.columns:
        mask &= work["mode"].astype(str).str.lower().eq("crowding")
    if "drop_label" in work.columns:
        drop_mask = work["drop_label"].astype(str).str.lower().isin({"drop30", "drop030"})
    elif "fraction" in work.columns:
        drop_mask = np.isclose(pd.to_numeric(work["fraction"], errors="coerce").to_numpy(dtype=float), DEFAULT_DROP_FRACTIONS[0])
    elif "variant" in work.columns:
        drop_mask = work["variant"].astype(str).str.lower().str.contains("drop30")
    else:
        raise SystemExit("Could not identify drop30 rows in baseline selected-removal table")
    work = work.loc[mask & drop_mask].copy()
    if work.empty:
        raise SystemExit("Baseline selected-removal table contains no oriented crowding drop30 rows")
    if work.duplicated(KEY_COLUMNS, keep=False).any():
        raise SystemExit("Baseline drop30 selected-removal table has duplicate exact keys")
    return key_set(work), work


def validate_baseline_removal_reproduction(observed_table: pd.DataFrame, baseline_path: Path) -> dict[str, Any]:
    progress = agmod.StageProgress("Validating baseline drop30 removal-key reproduction", total=1, unit="checks")
    expected_keys, expected_table = load_baseline_drop30_keys(baseline_path)
    observed_keys = key_set(observed_table)
    missing = expected_keys - observed_keys
    extra = observed_keys - expected_keys
    stats = {
        "baseline_table": str(baseline_path),
        "baseline_subset_rule": "mode == crowding and drop_label/fraction == drop30/0.30",
        "expected_count": int(len(expected_keys)),
        "observed_count": int(len(observed_keys)),
        "missing_count": int(len(missing)),
        "extra_count": int(len(extra)),
        "signed_hkl_preserved": True,
    }
    progress.finish(1)
    if len(expected_keys) != len(observed_keys) or missing or extra:
        raise SystemExit(f"Baseline sg100_sc100 removal-key reproduction failed: {stats}")
    expected_hkls = {key[2:] for key in expected_keys}
    observed_hkls = {key[2:] for key in observed_keys}
    if expected_hkls != observed_hkls:
        raise SystemExit("Baseline removal reproduction changed signed HKLs")
    return stats


def load_reference_ad4_table(args: argparse.Namespace) -> tuple[pd.DataFrame | None, str | None, str | None]:
    path: Path | None = args.reference_removals
    if path is None and args.reference_sweep_dir is not None:
        path = args.reference_sweep_dir / "selected_removal_observations.csv"
        if not path.is_file():
            raise SystemExit(f"--reference-sweep-dir missing selected_removal_observations.csv: {path}")
    if path is None:
        return None, None, None
    table = pd.read_csv(path, low_memory=False)
    agmod.require_columns(table, KEY_COLUMNS, "AD4 reference removal table")
    work = agmod.actionmod.normalize_key_columns(table)
    masks: list[pd.Series] = []
    reference_labels = ["target_neighbor_density_EgAD4", "sg175_sc100_an300"]
    if "score_name" in work.columns:
        for label in reference_labels:
            masks.append(work["score_name"].astype(str).eq(label))
    if "variant" in work.columns:
        for label in reference_labels:
            masks.append(work["variant"].astype(str).str.contains(label, regex=False))
    if masks:
        mask = masks[0].copy()
        for extra in masks[1:]:
            mask |= extra
        filtered = work.loc[mask].copy()
        if filtered.empty:
            raise SystemExit(f"Reference table contains no target_neighbor_density_EgAD4 or sg175_sc100_an300 rows: {path}")
        work = filtered
    detected = "reference_selection"
    for label in reference_labels:
        if ("score_name" in work.columns and work["score_name"].astype(str).eq(label).any()) or (
            "variant" in work.columns and work["variant"].astype(str).str.contains(label, regex=False).any()
        ):
            detected = label
            break
    if BLOCK_ID_COLUMN not in work.columns:
        if "block_id" in work.columns:
            work[BLOCK_ID_COLUMN] = pd.to_numeric(work["block_id"], errors="coerce").astype("Int64")
        else:
            raise SystemExit("AD4 reference table must include excitation_block_id or block_id for per-block validation")
    if work.duplicated(KEY_COLUMNS, keep=False).any():
        raise SystemExit("AD4 reference table contains duplicate exact keys")
    return work.reset_index(drop=True), str(path), detected


def key_sets_by_columns(table: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], set[tuple[str, str, int, int, int]]]:
    out: dict[tuple[Any, ...], set[tuple[str, str, int, int, int]]] = {}
    for key, group in table.groupby(columns, sort=False):
        tuple_key = key if isinstance(key, tuple) else (key,)
        out[tuple_key] = key_set(group)
    return out


def validate_ad4_reference_reproduction(selected: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    reference, path, reference_label = load_reference_ad4_table(args)
    if reference is None:
        return {"status": "skipped", "reason": "no reference removal table or sweep directory supplied"}
    observed_score = "current_reference_EgMD3" if selected["score_name"].astype(str).eq("current_reference_EgMD3").any() else "neighbor_density_AD4"
    observed = selected.loc[selected["score_name"].astype(str).eq(observed_score)].copy()
    if observed.empty:
        raise SystemExit(f"Reference supplied, but this sweep did not request {observed_score}")
    default_observed = observed.loc[observed["filter_name"].astype(str).eq("he030_bs010_drop030")].copy()
    if not default_observed.empty:
        observed = default_observed
    if observed["filter_name"].nunique(dropna=False) != 1:
        raise SystemExit("AD4 reference validation requires a single neighbor_density_AD4 filter selection")

    expected_keys = key_set(reference)
    observed_keys = key_set(observed)
    missing = expected_keys - observed_keys
    extra = observed_keys - expected_keys
    if missing or extra:
        raise SystemExit(
            f"{observed_score} did not reproduce {reference_label} reference keys: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    expected_by_block = key_sets_by_columns(reference, [*HKL_COLUMNS, BLOCK_ID_COLUMN])
    observed_by_block = key_sets_by_columns(observed, [*HKL_COLUMNS, BLOCK_ID_COLUMN])
    if expected_by_block != observed_by_block:
        missing_blocks = set(expected_by_block) - set(observed_by_block)
        extra_blocks = set(observed_by_block) - set(expected_by_block)
        changed_blocks = [
            key
            for key in sorted(set(expected_by_block) & set(observed_by_block))
            if expected_by_block[key] != observed_by_block[key]
        ]
        raise SystemExit(
            f"{observed_score} per-block reference reproduction failed: "
            f"missing_blocks={len(missing_blocks)} extra_blocks={len(extra_blocks)} changed_blocks={len(changed_blocks)}"
        )

    expected_by_hkl = key_sets_by_columns(reference, HKL_COLUMNS)
    observed_by_hkl = key_sets_by_columns(observed, HKL_COLUMNS)
    if expected_by_hkl != observed_by_hkl:
        raise SystemExit(f"{observed_score} per-HKL reference reproduction failed")
    return {
        "status": "passed",
        "reference_table": path,
        "reference_score": reference_label,
        "observed_score": observed_score,
        "filter_name": str(observed["filter_name"].iloc[0]),
        "total_removal_count": int(len(observed)),
        "signed_hkl_count": int(observed.loc[:, HKL_COLUMNS].drop_duplicates().shape[0]),
        "block_count": int(observed.loc[:, [*HKL_COLUMNS, BLOCK_ID_COLUMN]].drop_duplicates().shape[0]),
        "missing_count": 0,
        "extra_count": 0,
    }


def pairwise_score_correlations(filter_caches: dict[str, pd.DataFrame], score_specs: tuple[ScoreSpec, ...]) -> list[dict[str, Any]]:
    progress = agmod.StageProgress("Calculating pairwise score correlations", total=1, unit="passes")
    rows: list[dict[str, Any]] = []
    for filter_name, cache in filter_caches.items():
        common = cache.loc[cache[COMMON_BLOCK_COLUMN].astype(bool)].copy()
        score_values = {spec.name: pd.Series(alpha_score_values(common, spec), index=common.index) for spec in score_specs}
        for i, left in enumerate(score_specs):
            for right in score_specs[i + 1 :]:
                rows.append(
                    {
                        "filter_name": filter_name,
                        "variant_a": f"{left.name}_{filter_name}",
                        "variant_b": f"{right.name}_{filter_name}",
                        "score_a": left.name,
                        "score_b": right.name,
                        "pearson": finite_corr(score_values[left.name], score_values[right.name], "pearson"),
                        "spearman": finite_corr(score_values[left.name], score_values[right.name], "spearman"),
                        "n": int(len(common)),
                    }
                )
    progress.finish(1)
    return rows


def pairwise_removal_overlaps(selected: pd.DataFrame) -> list[dict[str, Any]]:
    progress = agmod.StageProgress("Calculating pairwise removal-key overlaps", total=1, unit="passes")
    rows: list[dict[str, Any]] = []
    for filter_name, group in selected.groupby("filter_name", sort=False):
        sets = {variant: key_set(sub) for variant, sub in group.groupby("variant", sort=False)}
        variants = sorted(sets)
        for i, left in enumerate(variants):
            for right in variants[i + 1 :]:
                a = sets.get(left, set())
                b = sets.get(right, set())
                intersection = len(a & b)
                union = len(a | b)
                rows.append(
                    {
                        "filter_name": filter_name,
                        "variant_a": left,
                        "variant_b": right,
                        "overlap_count": int(intersection),
                        "jaccard": float(intersection / union) if union else None,
                        "count_a": int(len(a)),
                        "count_b": int(len(b)),
                    }
                )
    progress.finish(1)
    return rows


def aggregate_moment_summaries(filter_caches: dict[str, pd.DataFrame], plan: SweepPlan) -> dict[str, Any]:
    progress = agmod.StageProgress("Summarizing aggregate moment distributions", total=1, unit="passes")
    raw_by = raw_spec_by_name(plan)
    template_by_raw: dict[str, ScoreSpec] = {}
    for spec in plan.requested_score_specs:
        template_by_raw.setdefault(spec.raw_name, spec)
    distributions: list[dict[str, Any]] = []
    numeric_rule_stats: list[dict[str, Any]] = []
    for filter_name, cache in filter_caches.items():
        common = cache.loc[cache[COMMON_BLOCK_COLUMN].astype(bool)].copy()
        for raw_name in sorted(template_by_raw):
            raw = raw_by[raw_name]
            spec = template_by_raw[raw_name]
            components, stats = score_components(common, spec)
            numeric_rule_stats.append(
                {
                    "filter_name": filter_name,
                    "raw_kernel_name": raw.name,
                    "sg_multiplier": float(raw.sg_multiplier),
                    "sg0": float(raw.sg0),
                    "sigma_multiplier": float(raw.sigma_multiplier),
                    "sigma_c": float(raw.sigma_c),
                    "r_cut": float(raw.r_cut),
                    **stats,
                }
            )
            for variable in ["Eg", "M", "D", "U", "N", "C2", "W2", "M2", "A", "Q", "Neff", "Pc", "Pe"]:
                if variable not in components:
                    continue
                distributions.append(
                    {
                        "filter_name": filter_name,
                        "raw_kernel_name": raw.name,
                        "variable": variable,
                        "sg_multiplier": float(raw.sg_multiplier),
                        "sigma_multiplier": float(raw.sigma_multiplier),
                        **numeric_distribution(components[variable]),
                    }
                )
    progress.finish(1)
    return {
        "distributions": distributions,
        "numeric_rule_stats": numeric_rule_stats,
    }


def duplicated_score_count_by_block(common: pd.DataFrame, spec: ScoreSpec, score_array: np.ndarray | None = None) -> int:
    if common.empty:
        return 0
    scores = alpha_score_values(common, spec) if score_array is None else np.asarray(score_array, dtype=float)
    work = common.loc[:, [*HKL_COLUMNS, BLOCK_ID_COLUMN]].copy()
    work["_candidate_score"] = scores
    return int(work.duplicated([*HKL_COLUMNS, BLOCK_ID_COLUMN, "_candidate_score"], keep=False).sum())


def build_variant_summary(
    plan: SweepPlan,
    filter_caches: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    per_block: pd.DataFrame,
    baseline_control_keys: dict[str, set[tuple[str, str, int, int, int]]],
) -> pd.DataFrame:
    score_by = score_spec_by_name(plan)
    filter_by = filter_spec_by_name(plan)
    selected_keys_by_filter_score = {
        (str(filter_name), str(score_name)): key_set(group)
        for (filter_name, score_name), group in selected.groupby(["filter_name", "score_name"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for experiment in plan.experiments:
        spec = score_by[experiment.score_name]
        filter_cfg = filter_by[experiment.filter_name]
        cache = filter_caches[filter_cfg.name]
        common = cache.loc[cache[COMMON_BLOCK_COLUMN].astype(bool)].copy()
        components, component_stats = score_components(common, spec)
        if spec.score_type == "derived":
            score_array, formula_stats = derived_score_values(common, spec, return_stats=True)  # type: ignore[misc]
        else:
            score_array = alpha_score_values(common, spec)
            formula_stats = {
                "score_zero_count": int((score_array == 0.0).sum()),
                "score_positive_count": int((score_array > 0.0).sum()),
                "score_negative_count": int((score_array < 0.0).sum()),
                "score_nonfinite_count": int((~np.isfinite(score_array)).sum()),
            }
        formula_stats = {**component_stats, **formula_stats}
        scores = pd.Series(score_array)
        variant_selected = selected.loc[selected["variant"] == experiment.name].copy()
        variant_keys = key_set(variant_selected)
        baseline_keys = baseline_control_keys.get(filter_cfg.name, set())
        overlap = len(variant_keys & baseline_keys) if baseline_keys else None
        union = len(variant_keys | baseline_keys) if baseline_keys else None
        d_keys = selected_keys_by_filter_score.get((filter_cfg.name, "density_D"), set())
        ad4_keys = selected_keys_by_filter_score.get((filter_cfg.name, "neighbor_density_AD4"), set())
        current_reference_keys = selected_keys_by_filter_score.get((filter_cfg.name, "current_reference_EgMD3"), set())
        d_overlap = len(variant_keys & d_keys) if d_keys else None
        d_union = len(variant_keys | d_keys) if d_keys else None
        ad4_overlap = len(variant_keys & ad4_keys) if ad4_keys else None
        ad4_union = len(variant_keys | ad4_keys) if ad4_keys else None
        current_reference_overlap = len(variant_keys & current_reference_keys) if current_reference_keys else None
        current_reference_union = len(variant_keys | current_reference_keys) if current_reference_keys else None
        block_sub = per_block.loc[per_block["variant"] == experiment.name].copy()
        duplicate_score_rows = duplicated_score_count_by_block(common, spec, score_array) if not common.empty else 0
        removed_delta = None
        if not variant_selected.empty:
            selected_with_components = variant_selected.merge(
                common.loc[:, [*KEY_COLUMNS]].assign(_delta=components["delta"]),
                on=KEY_COLUMNS,
                how="left",
                sort=False,
                validate="one_to_one",
            )
            removed_delta = numeric_distribution(selected_with_components["_delta"])
        current_reference_score_corr = None
        current_reference_spec = score_by.get("current_reference_EgMD3")
        if current_reference_spec is not None:
            current_reference_score_corr = finite_corr(scores, pd.Series(alpha_score_values(common, current_reference_spec)), "spearman")
        row = {
            "variant": experiment.name,
            "score_name": spec.name,
            "filter_name": filter_cfg.name,
            "score_type": spec.score_type,
            "formula_kind": spec.formula_kind,
            "score_family": spec.score_family,
            "formula": spec.formula,
            "p": spec.p_value,
            "lambda": spec.lambda_value,
            "sg0": float(spec.sg0),
            "sg_multiplier": float(spec.sg_multiplier),
            "sigma_c": float(spec.sigma_c),
            "sigma_multiplier": float(spec.sigma_multiplier),
            "r_cut": float(spec.r_cut),
            "alpha": float(spec.alpha),
            "eg_power": float(spec.eg_power),
            "a_power": float(spec.a_power),
            "d_power": float(spec.d_power),
            "imbalance_power": float(spec.imbalance_power),
            "high_eg_fraction": float(filter_cfg.high_eg_fraction),
            "drop_fraction": float(filter_cfg.drop_fraction),
            "excitation_block_size": int(filter_cfg.excitation_block_size),
            "common_high_Eg_observation_count": int(len(common)),
            "common_excitation_block_count": int(cache.loc[:, [*HKL_COLUMNS, BLOCK_ID_COLUMN]].drop_duplicates().shape[0]),
            "common_actionable_block_count": int(common.loc[:, [*HKL_COLUMNS, BLOCK_ID_COLUMN]].drop_duplicates().shape[0]),
            "expected_removal_count": int(block_sub["n_remove"].sum()) if not block_sub.empty else 0,
            "actual_removal_count": int(len(variant_selected)),
            "block_removal_counts_equal": True,
            "hkl_removal_counts_equal": True,
            "removal_fraction_all_6732955_accepted": float(len(variant_selected) / DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR),
            "score_min": float(scores.min()) if scores.notna().any() else None,
            "score_q25": float(scores.quantile(0.25)) if scores.notna().any() else None,
            "score_median": float(scores.median()) if scores.notna().any() else None,
            "score_mean": float(scores.mean()) if scores.notna().any() else None,
            "score_q75": float(scores.quantile(0.75)) if scores.notna().any() else None,
            "score_max": float(scores.max()) if scores.notna().any() else None,
            "score_nonfinite_count": int(formula_stats.get("score_nonfinite_count", 0)),
            "score_zero_count": int(formula_stats.get("score_zero_count", 0)),
            "positive_score_fraction": float(formula_stats.get("score_positive_count", 0) / max(1, len(scores))),
            "negative_score_fraction": float(formula_stats.get("score_negative_count", 0) / max(1, len(scores))),
            "zero_score_fraction": float((scores == 0.0).sum() / max(1, len(scores))),
            "zero_D_count": int(formula_stats.get("zero_D_count", 0)),
            "zero_A_count": int(formula_stats.get("zero_A_count", 0)),
            "zero_Eg_count": int(formula_stats.get("zero_Eg_count", 0)),
            "D_invalid_power_count": int(formula_stats.get("D_invalid_power_count", 0)),
            "A_invalid_power_count": int(formula_stats.get("A_invalid_power_count", 0)),
            "Eg_invalid_power_count": int(formula_stats.get("Eg_invalid_power_count", 0)),
            "abs_delta_invalid_power_count": int(formula_stats.get("abs_delta_invalid_power_count", 0)),
            "spearman_with_D": finite_corr(scores, pd.Series(components["D"]), "spearman"),
            "spearman_with_M": finite_corr(scores, pd.Series(components["M"]), "spearman"),
            "spearman_with_U": finite_corr(scores, pd.Series(components["U"]), "spearman") if "U" in components else None,
            "spearman_with_A": finite_corr(scores, pd.Series(components["A"]), "spearman"),
            "spearman_with_Q": finite_corr(scores, pd.Series(components["Q"]), "spearman") if "Q" in components else None,
            "spearman_with_Eg": finite_corr(scores, pd.Series(components["Eg"]), "spearman"),
            "spearman_with_AD4": finite_corr(scores, pd.Series(components["AD4"]), "spearman"),
            "spearman_with_current_reference_EgMD3": current_reference_score_corr,
            "tied_score_fraction": float(duplicate_score_rows / max(1, len(common))),
            "zero_range_common_blocks": int(block_sub["zero_candidate_score_range"].sum()) if not block_sub.empty else 0,
            "block_score_range_min": float(block_sub["score_range"].min()) if not block_sub.empty else None,
            "block_score_range_median": float(block_sub["score_range"].median()) if not block_sub.empty else None,
            "block_score_range_max": float(block_sub["score_range"].max()) if not block_sub.empty else None,
            "removed_target_Eg_median": numeric_distribution(variant_selected[EG_COLUMN])["median"] if not variant_selected.empty else None,
            "removed_abs_sg_target_median": numeric_distribution(variant_selected[ABS_SG_TARGET_COLUMN])["median"] if not variant_selected.empty else None,
            "removed_delta_A_minus_Eg_min": removed_delta["min"] if removed_delta is not None else None,
            "removed_delta_A_minus_Eg_median": removed_delta["median"] if removed_delta is not None else None,
            "removed_delta_A_minus_Eg_max": removed_delta["max"] if removed_delta is not None else None,
            "overlap_count_with_density_D": int(d_overlap) if d_overlap is not None else None,
            "jaccard_with_density_D": float(d_overlap / d_union) if d_union else None,
            "overlap_count_with_neighbor_density_AD4": int(ad4_overlap) if ad4_overlap is not None else None,
            "jaccard_with_neighbor_density_AD4": float(ad4_overlap / ad4_union) if ad4_union else None,
            "overlap_count_with_current_reference_EgMD3": int(current_reference_overlap) if current_reference_overlap is not None else None,
            "jaccard_with_current_reference_EgMD3": float(current_reference_overlap / current_reference_union) if current_reference_union else None,
            "overlap_count_with_internal_baseline_sg100_sc100_a000": int(overlap) if overlap is not None else None,
            "jaccard_with_internal_baseline_sg100_sc100_a000": float(overlap / union) if union else None,
        }
        rows.append(row)
    return pd.DataFrame.from_records(rows)

def write_qc_outputs(
    args: argparse.Namespace,
    cache: pd.DataFrame,
    selected: pd.DataFrame,
    per_block: pd.DataFrame,
    per_hkl: pd.DataFrame,
    summary: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    progress = agmod.StageProgress("Writing QC outputs", total=13, unit="files")
    summary.to_csv(args.out_dir / "per_variant_filter_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    summary.to_csv(args.out_dir / "summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    comparison_columns = [
        "variant",
        "score_name",
        "filter_name",
        "score_family",
        "formula",
        "p",
        "lambda",
        "actual_removal_count",
        "jaccard_with_current_reference_EgMD3",
        "overlap_count_with_current_reference_EgMD3",
        "spearman_with_current_reference_EgMD3",
        "spearman_with_Eg",
        "spearman_with_M",
        "spearman_with_D",
        "spearman_with_U",
        "spearman_with_A",
        "spearman_with_Q",
        "score_median",
        "score_mean",
        "score_max",
    ]
    comparison = summary.loc[:, [column for column in comparison_columns if column in summary.columns]].copy()
    if "jaccard_with_current_reference_EgMD3" in comparison.columns:
        comparison = comparison.sort_values(
            ["jaccard_with_current_reference_EgMD3", "variant"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
    )
    comparison.to_csv(args.out_dir / "score_comparison_by_current_reference.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    pd.DataFrame.from_records(audit.get("pairwise_score_correlations", [])).to_csv(
        args.out_dir / "correlations.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    progress.advance()
    pd.DataFrame.from_records(audit.get("pairwise_removal_overlaps", [])).to_csv(
        args.out_dir / "removal_overlap.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    progress.advance()
    per_hkl.to_csv(args.out_dir / "per_variant_per_hkl_qc.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    per_block.to_csv(args.out_dir / "per_variant_per_block_qc.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    selected.sort_values(["variant", *HKL_COLUMNS, BLOCK_ID_COLUMN, "selection_rank_in_block", EXACT_KEY_TEXT_COLUMN], kind="mergesort").to_csv(
        args.out_dir / "selected_removal_observations.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    progress.advance()
    (args.out_dir / "sweep_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    scores_payload = {
        "created_local": datetime.now().astimezone().isoformat(),
        "scores": audit.get("score_specs", []),
        "expression_trees": audit.get("score_formula_trees", {}),
    }
    (args.out_dir / "scores.json").write_text(json.dumps(scores_payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    validation_payload = {
        "baseline_score_reproduction": audit.get("baseline_score_reproduction"),
        "baseline_removal_reproduction": audit.get("baseline_removal_reproduction"),
        "ad4_reference_reproduction": audit.get("ad4_reference_reproduction"),
        "expected_removal_count_validation": audit.get("expected_removal_count_validation"),
        "cache_stats": audit.get("cache_stats"),
        "filter_common_counts": audit.get("filter_common_counts"),
    }
    (args.out_dir / "validation.json").write_text(json.dumps(validation_payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_local": datetime.now().astimezone().isoformat(),
        "source_script": str(Path(__file__).resolve()),
        "source_script_git_commit": git_commit_for_path(Path(__file__)),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "audit": audit,
        "cache_rows": int(len(cache)),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    parameters_payload = {
        "created_local": metadata["created_local"],
        "source_script": metadata["source_script"],
        "source_script_git_commit": metadata["source_script_git_commit"],
        "args": metadata["args"],
        "input_stream": metadata["args"].get("input_stream"),
        "cache_path": audit.get("cache_stats", {}).get("cache_path"),
        "source_cache_path": audit.get("cache_stats", {}).get("source_cache_path"),
        "cache_schema": audit.get("cache_metadata", {}),
        "geometry_enumeration_performed": bool(audit.get("cache_stats", {}).get("computed_columns")),
        "kernel_parameters": audit.get("raw_kernel_specs", []),
        "filter_parameters": audit.get("filter_specs", []),
        "worker_count": audit.get("requested_worker_count"),
        "chunk_sizes": audit.get("cache_metadata", {}).get("parameters", {}),
        "score_names": [item.get("name") for item in audit.get("score_specs", [])],
        "scores": audit.get("score_specs", []),
        "expected_removal_count_validation": audit.get("expected_removal_count_validation"),
        "runtime": {
            "created_utc": audit.get("created_utc"),
            "actual_worker_count": audit.get("actual_worker_count"),
            "worker_pids": audit.get("worker_pids"),
        },
    }
    (args.out_dir / "parameters.json").write_text(json.dumps(parameters_payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    progress.finish(13)


def desired_stream_path(out_dir: Path, experiment: ExperimentSpec | str) -> Path:
    if isinstance(experiment, ExperimentSpec):
        return out_dir / experiment.output_stream
    return out_dir / f"{experiment}.stream"


def validate_output_stream_worker(task: tuple[str, list[tuple[str, str, int, int, int]], int | None]) -> dict[str, Any]:
    path_text, requested_key_list, expected_rows = task
    path = Path(path_text)
    requested_keys = set(requested_key_list)
    scan = highmod.stream_scan_absence(path, requested_keys)
    if scan["requested_removal_keys_remaining_in_output"] != 0:
        raise SystemExit(f"Requested removals remain in output stream {path}")
    if expected_rows is not None and int(scan["output_reflection_rows_verified"]) != int(expected_rows):
        raise SystemExit(
            f"Output stream removed an unrequested key or has row-count mismatch: "
            f"{path}; observed={scan['output_reflection_rows_verified']} expected={expected_rows}"
        )
    return {
        "output_stream": str(path),
        "output_reflection_rows_expected": None if expected_rows is None else int(expected_rows),
        "nonrequested_reflections_preserved_by_count": None if expected_rows is None else True,
        **scan,
    }


def write_sweep_streams(args: argparse.Namespace, plan: SweepPlan, selected: pd.DataFrame) -> pd.DataFrame:
    removals: dict[str, pd.DataFrame] = {}
    pseudo_to_variant: dict[str, str] = {}
    skipped_rows: list[dict[str, Any]] = []
    validation_requests: dict[str, tuple[list[tuple[str, str, int, int, int]], int | None]] = {}
    for experiment in plan.experiments:
        variant = experiment.name
        requested = selected.loc[selected["variant"] == variant, KEY_COLUMNS].copy()
        desired = desired_stream_path(args.out_dir, experiment)
        if args.skip_existing and desired.is_file():
            validation_requests[variant] = (list(key_set(requested)), None)
            skipped_rows.append(
                {
                    "variant": variant,
                    "pseudo_variant": None,
                    "requested_removals": int(len(requested)),
                    "removed_observations": None,
                    "kept_observations": None,
                    "total_reflection_rows_seen": None,
                    "output_stream": str(desired),
                    "skipped_existing": True,
                }
            )
            continue
        pseudo = f"aggressive_drop_{variant}"
        pseudo_to_variant[pseudo] = variant
        removals[pseudo] = requested

    stream_qc = agmod.write_stream_variants(args.input_stream, args.out_dir, removals, seed=0) if removals else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    validation_tasks: list[tuple[str, list[tuple[str, str, int, int, int]], int | None, dict[str, Any]]] = []
    for row in stream_qc.itertuples(index=False):
        pseudo = str(row.variant)
        variant = pseudo_to_variant[pseudo]
        old_path = Path(row.output_stream)
        new_path = desired_stream_path(args.out_dir, variant)
        if old_path != new_path:
            old_path.replace(new_path)
        requested = removals[pseudo]
        requested_keys = list(key_set(requested))
        expected_rows = int(row.total_reflection_rows_seen) - int(row.requested_removals)
        out = row._asdict()
        out["variant"] = variant
        out["pseudo_variant"] = pseudo
        out["output_stream"] = str(new_path)
        out["skipped_existing"] = False
        validation_tasks.append((str(new_path), requested_keys, int(expected_rows), out))
    for skipped in skipped_rows:
        variant = str(skipped["variant"])
        requested_keys, expected_rows = validation_requests[variant]
        validation_tasks.append((str(skipped["output_stream"]), requested_keys, expected_rows, skipped))

    progress = agmod.StageProgress("Final exact-key validation of output streams", total=len(validation_tasks), unit="streams")
    if int(args.workers) > 1 and len(validation_tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=agmod.worker_initializer) as executor:
            future_to_base = {
                executor.submit(validate_output_stream_worker, (path_text, requested_keys, expected_rows)): base
                for path_text, requested_keys, expected_rows, base in validation_tasks
            }
            for future in as_completed(future_to_base):
                base = future_to_base[future]
                scan = future.result()
                base.update(scan)
                rows.append(base)
                progress.advance()
    else:
        for path_text, requested_keys, expected_rows, base in validation_tasks:
            scan = validate_output_stream_worker((path_text, requested_keys, expected_rows))
            base.update(scan)
            rows.append(base)
            progress.advance()
    progress.finish(len(validation_tasks))
    return pd.DataFrame.from_records(rows)


def write_audit_only_stream_qc(out_dir: Path) -> None:
    columns = [
        "variant",
        "requested_removals",
        "removed_observations",
        "kept_observations",
        "total_reflection_rows_seen",
        "output_stream",
    ]
    pd.DataFrame(columns=columns).to_csv(out_dir / "stream_rewrite_qc.csv", index=False)


def write_readme(out_dir: Path) -> None:
    text = """# V5 Coarse Raw-Crowding Score Sweep

This run evaluates parameterized raw physical crowding score kernels:
`S(g) = sum_{q != g} exp[-(s_q / sg0)^2] C_sigma(g-q)`.

Alpha score variants use `S_alpha = S_raw / C_sum**alpha`, with zero coupling
and zero raw score mapped to zero.  JSON-derived score variants use structured
formula kinds built from `Eg`, `D`, `A = S_raw / D`, and `A - Eg`; no Python
configuration text is executed.  For each filter config, high-Eg pools,
excitation blocks, baseline-defined actionable blocks, and per-block removal
counts are frozen and then reused for every requested score variant.  No random
controls, intensity-response modelling, Partialator execution, or plots are
produced here.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def build_audit(
    args: argparse.Namespace,
    plan: SweepPlan,
    cache: pd.DataFrame,
    filter_caches: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    per_block: pd.DataFrame,
    selected: pd.DataFrame,
    cache_stats: dict[str, Any],
    cache_meta: dict[str, Any],
    baseline_score_stats: dict[str, Any],
    baseline_removal_stats: dict[str, Any],
    ad4_reference_stats: dict[str, Any],
    expected_count_stats: dict[str, Any],
    worker_pids: list[int],
    actual_workers: int,
) -> dict[str, Any]:
    common_counts = {
        name: {
            "common_high_Eg_observation_count": int(len(table.loc[table[COMMON_BLOCK_COLUMN].astype(bool)])),
            "common_excitation_block_count": int(table.loc[:, [*HKL_COLUMNS, BLOCK_ID_COLUMN]].drop_duplicates().shape[0]),
            "common_actionable_block_count": int(table.loc[table[COMMON_BLOCK_COLUMN].astype(bool), [*HKL_COLUMNS, BLOCK_ID_COLUMN]].drop_duplicates().shape[0]),
        }
        for name, table in filter_caches.items()
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "variant_count": int(len(plan.experiments)),
        "score_variant_count": int(len(plan.requested_score_specs)),
        "filter_count": int(len(plan.filter_specs)),
        "cache_metadata": cache_meta,
        "cache_stats": cache_stats,
        "baseline_score_reproduction": baseline_score_stats,
        "baseline_removal_reproduction": baseline_removal_stats,
        "ad4_reference_reproduction": ad4_reference_stats,
        "expected_removal_count_validation": expected_count_stats,
        "serial_stage_justification": (
            "Output stream rewriting performs one ordered traversal of the source stream so every variant is written "
            "from the same input order; score enumeration, block construction, selection, and final stream validation "
            "use the requested worker pool where parallel work is available."
        ),
        "requested_worker_count": int(args.workers),
        "actual_worker_count": int(actual_workers),
        "worker_pids": [int(pid) for pid in worker_pids],
        "raw_kernel_specs": [asdict(spec) for spec in plan.raw_specs],
        "score_specs": [asdict(spec) for spec in plan.requested_score_specs],
        "score_formula_trees": {
            spec.name: spec.expression_tree
            for spec in plan.requested_score_specs
            if spec.score_type == "derived" and spec.formula_kind == "expression_tree"
        },
        "filter_specs": [asdict(spec) for spec in plan.filter_specs],
        "experiments": [asdict(spec) for spec in plan.experiments],
        "filter_common_counts": common_counts,
        "common_high_Eg_observation_count": int(sum(item["common_high_Eg_observation_count"] for item in common_counts.values())),
        "common_actionable_block_count": int(sum(item["common_actionable_block_count"] for item in common_counts.values())),
        "selected_total_per_variant": {str(row.variant): int(row.actual_removal_count) for row in summary.itertuples(index=False)},
        "variant_summaries": summary.to_dict("records"),
        "aggregate_moment_summaries": aggregate_moment_summaries(filter_caches, plan),
        "pairwise_score_correlations": pairwise_score_correlations(filter_caches, plan.requested_score_specs),
        "pairwise_removal_overlaps": pairwise_removal_overlaps(selected),
        "block_score_range_distributions": {
            str(row.variant): numeric_distribution(per_block.loc[per_block["variant"] == str(row.variant), "score_range"])
            for row in summary.itertuples(index=False)
        },
        "scientific_constraints": {
            "raw_score_definition": "sum_{q != g} exp[-(s_q / sg0)^2] * C_sigma(g-q)",
            "alpha_score_definition": "S_raw / C_sum**alpha; zero coupling with zero raw score maps to zero; no epsilon",
            "derived_score_definitions": {
                "power_product": "Eg^eg_power * A^a_power * D^d_power",
                "imbalance_absolute": "Eg^eg_power * abs(A - Eg)^imbalance_power * D^d_power",
                "imbalance_feed": "Eg^eg_power * (A - Eg) * D^d_power",
                "imbalance_sink": "Eg^eg_power * (Eg - A) * D^d_power",
                "A": "raw_score / D, with A=0 when D==0 and raw_score==0",
                "D": "coupling_sum for the score sigma_c",
                "Eg": EG_COLUMN,
            },
            "target_excitation_Eg_in_score": False,
            "only_nonself_neighbors": True,
            "uses_intensities_or_response_variables_for_selection": False,
            "uses_merged_intensity_or_Fobs": False,
            "uses_prediction_errors_or_fitted_distortion": False,
            "uses_random_controls": False,
            "preserves_exact_signed_hkl": True,
            "canonicalizes_observations_to_4mmm": False,
        },
    }


def print_audit_summary(audit: dict[str, Any]) -> None:
    log(f"Variant count: {audit['variant_count']:,}")
    log(f"Cache reused: {audit['cache_stats'].get('cache_reused')}")
    log(f"Common high-Eg observations: {audit['common_high_Eg_observation_count']:,}")
    log(f"Common actionable blocks: {audit['common_actionable_block_count']:,}")
    log(f"Baseline score max abs diff: {audit['baseline_score_reproduction']['max_absolute_difference']}")
    observed = audit["baseline_removal_reproduction"].get("observed_count")
    log(f"Baseline removal count: {observed:,}" if observed is not None else "Baseline removal check: skipped")
    log(f"AD4 reference reproduction: {audit['ad4_reference_reproduction'].get('status')}")
    log("Worker PIDs: " + ", ".join(str(pid) for pid in audit["worker_pids"]))


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.pathway_moment_screen:
        config_path = write_pathway_moment_config(args.out_dir)
        log(f"Wrote built-in pathway moment score config: {config_path}")
    agmod.set_worker_numeric_threads()
    plan = resolve_sweep_plan(args)
    write_variant_parameters(args.out_dir, plan)
    write_experiment_plan(args.out_dir, plan)
    if args.plan_only:
        write_readme(args.out_dir)
        log(f"Plan-only mode wrote {len(plan.experiments):,} planned output stream(s) to: {args.out_dir}")
        return 0

    expected_cache_metadata = cache_metadata(args, plan)
    manifest, manifest_stats = read_manifest(args.manifest)
    accepted, accepted_stats = load_accepted_manifest_keys(args, manifest)
    accepted_v5, v5_stats = load_accepted_v5_rows(args, accepted)

    filter_targets: dict[str, pd.DataFrame] = {}
    filter_hkl_qc: list[pd.DataFrame] = []
    filter_block_layout_qc: list[pd.DataFrame] = []
    for filter_cfg in plan.filter_specs:
        targets, hkl_qc, block_layout_qc = construct_high_eg_blocks(accepted_v5, manifest, filter_cfg, int(args.workers))
        if targets.empty:
            raise SystemExit(f"Filter {filter_cfg.name} produced no high-Eg target rows")
        filter_targets[filter_cfg.name] = targets
        filter_hkl_qc.append(hkl_qc)
        filter_block_layout_qc.append(block_layout_qc)
    score_targets = pd.concat(filter_targets.values(), ignore_index=True)
    cache, cache_stats, baseline_score_stats, worker_pids, actual_workers, cache_reused = build_or_load_score_cache(
        args,
        plan,
        score_targets,
        expected_cache_metadata,
    )

    filter_caches: dict[str, pd.DataFrame] = {}
    actionable_block_qc_parts: list[pd.DataFrame] = []
    score_columns = required_cache_columns(plan)
    for filter_cfg in plan.filter_specs:
        target = filter_targets[filter_cfg.name].copy()
        score_payload = cache.loc[:, [*KEY_COLUMNS, *score_columns]].copy()
        filter_cache = target.drop(columns=[column for column in score_columns if column in target.columns], errors="ignore").merge(
            score_payload,
            on=KEY_COLUMNS,
            how="left",
            sort=False,
            validate="one_to_one",
        )
        filter_cache, block_qc = define_common_actionable_blocks(filter_cache, filter_cfg, plan.internal_baseline_score)
        filter_caches[filter_cfg.name] = filter_cache
        actionable_block_qc_parts.append(block_qc)

    selected_parts: list[pd.DataFrame] = []
    per_block_parts: list[pd.DataFrame] = []
    per_hkl_parts: list[pd.DataFrame] = []
    for filter_cfg in plan.filter_specs:
        selected_part, per_block_part, per_hkl_part = select_variant_removals(
            filter_caches[filter_cfg.name],
            filter_cfg,
            plan.requested_score_specs,
            int(args.workers),
        )
        selected_parts.append(selected_part)
        per_block_parts.append(per_block_part)
        per_hkl_parts.append(per_hkl_part)
    selected = pd.concat(selected_parts, ignore_index=True)
    per_block = pd.concat(per_block_parts, ignore_index=True)
    per_hkl = pd.concat(per_hkl_parts, ignore_index=True)
    validate_equal_counts(selected, per_block, per_hkl)
    expected_count_stats = validate_expected_removal_count(selected, args.expected_removal_count)

    baseline_control_keys: dict[str, set[tuple[str, str, int, int, int]]] = {}
    baseline_removal_stats: dict[str, Any] = {
        "status": "skipped",
        "reason": "default filter config not requested",
    }
    for filter_cfg in plan.filter_specs:
        control_selected, _control_per_block, _control_per_hkl = select_variant_removals(
            filter_caches[filter_cfg.name],
            filter_cfg,
            (plan.internal_baseline_score,),
            int(args.workers),
        )
        baseline_control_keys[filter_cfg.name] = key_set(control_selected)
        if is_default_filter(filter_cfg):
            baseline_removal_stats = validate_baseline_removal_reproduction(control_selected, args.baseline_removals)
    ad4_reference_stats = validate_ad4_reference_reproduction(selected, args)

    summary = build_variant_summary(plan, filter_caches, selected, per_block, baseline_control_keys)
    for column in ["block_removal_counts_equal", "hkl_removal_counts_equal"]:
        summary[column] = True
    cache_stats["manifest"] = manifest_stats
    cache_stats["accepted"] = accepted_stats
    cache_stats["accepted_v5"] = v5_stats
    cache_stats["filter_block_construction"] = {
        "hkl_qc_rows": int(sum(len(table) for table in filter_hkl_qc)),
        "block_layout_rows": int(sum(len(table) for table in filter_block_layout_qc)),
        "actionable_block_qc_rows": int(sum(len(table) for table in actionable_block_qc_parts)),
    }
    audit = build_audit(
        args,
        plan,
        cache,
        filter_caches,
        summary,
        per_block,
        selected,
        cache_stats,
        expected_cache_metadata,
        baseline_score_stats,
        baseline_removal_stats,
        ad4_reference_stats,
        expected_count_stats,
        worker_pids,
        actual_workers,
    )
    write_qc_outputs(args, cache, selected, per_block, per_hkl, summary, audit)
    print_audit_summary(audit)

    if args.audit_only:
        write_audit_only_stream_qc(args.out_dir)
    else:
        stream_qc = write_sweep_streams(args, plan, selected)
        stream_qc.to_csv(args.out_dir / "stream_rewrite_qc.csv", index=False)
    write_readme(args.out_dir)
    log(f"Outputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
