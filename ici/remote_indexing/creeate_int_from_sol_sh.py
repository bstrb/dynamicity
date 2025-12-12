#!/usr/bin/env python3
import argparse
from pathlib import Path


TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

# Auto-generated integration script. Edit as you like.

GEOM="{geom}"
LST="{lst}"
SOL="{sol}"
OUT_STREAM="{out_stream}"

indexamajig \\
  -g "$GEOM" \\
  -i "$LST" \\
  -o "$OUT_STREAM" \\
  --indexing=file \\
  --fromfile-input-file="$SOL" \\
  --integration=rings \\
  --int-radius=4,5,9 \\
  --no-refine \\
  --no-check-peaks \\
  --no-check-cell \\
  --no-revalidate \\
  --no-non-hits-in-stream \\
  --peaks=cxi \\
  --no-half-pixel-shift \\
  --fix-profile-radius=50000000
"""


def main():
    p = argparse.ArgumentParser(
        description="Write an indexamajig integration .sh script from .geom, .lst, .sol"
    )
    p.add_argument("--geom", required=True, help="Path to .geom file")
    p.add_argument("--lst", required=True, help="Path to image-files.lst")
    p.add_argument("--sol", required=True, help="Path to .sol file")
    p.add_argument(
        "--out-stream",
        default=None,
        help="Output stream filename (default: integration.stream in the same directory as the .sol)",
    )
    p.add_argument(
        "--sh-out",
        default=None,
        help="Output .sh script path (default: indexamajig_integration.sh in the same directory as the .sol)",
    )

    args = p.parse_args()

    sol = Path(args.sol).resolve()
    geom = Path(args.geom).resolve()
    lst = Path(args.lst).resolve()
    sol_dir = sol.parent

    # Default OUT_STREAM: same dir as .sol, name integration.stream
    if args.out_stream is None:
        out_stream_path = sol_dir / "integration.stream"
    else:
        out_stream_path = Path(args.out_stream).resolve()

    # Default .sh: same dir as .sol, name indexamajig_integration.sh
    if args.sh_out is None:
        sh_out = sol_dir / "indexamajig_integration.sh"
    else:
        sh_out = Path(args.sh_out).resolve()

    script_text = TEMPLATE.format(
        geom=str(geom),
        lst=str(lst),
        sol=str(sol),
        out_stream=str(out_stream_path),
    )

    sh_out.write_text(script_text)
    print(f"Wrote {sh_out}")
    print(f"Make it executable with:\n  chmod +x {sh_out}")


if __name__ == "__main__":
    main()
