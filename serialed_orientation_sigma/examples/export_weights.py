from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export compact merge-weight tables from reflection_scores.csv")
    parser.add_argument("--input", type=str, required=True, help="Path to reflection_scores.csv")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--keep-only", action="store_true", help="Write only reflections with keep == True")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    table = pd.read_csv(args.input)
    if args.keep_only and "keep" in table.columns:
        table = table[table["keep"]].copy()
    columns = [
        "frame",
        "h",
        "k",
        "l",
        "I",
        "sigma",
        "sigma_new",
        "weight_new",
        "keep",
        "S",
        "sg",
    ]
    available = [col for col in columns if col in table.columns]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table[available].to_csv(output_path, index=False)
    print(f"Wrote {len(table)} rows to {output_path}")
