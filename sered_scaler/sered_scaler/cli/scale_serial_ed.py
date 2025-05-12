# ================================================================
# file: sered_scaler/cli/scale_serial_ed.py
# ================================================================

"""Command‑line driver for *scale_serial_ed*.

Now supports **EM‑style iteration**:

```
--n-iter N      # default 1 (behaves like before)
```

For *method "bayes"* each iteration refits the mixture and then the
weighted merge. For *"zscore"* the filter is applied only in the first
iteration (extra loops just redo the weighted merge – cheap but keeps
interface symmetrical).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from ..io import stream_to_dfs
from ..scaling import zscore_filter, weighted_merge, mixture_filter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401
    p = argparse.ArgumentParser(
        prog="scale_serial_ed",
        description="Kinematic scaling / merging for Serial‑ED *.stream* files",
    )

    p.add_argument("stream", type=Path, help="Input .stream file")

    p.add_argument("--method", choices=["zscore", "bayes"], default="zscore",
                   help="Filtering strategy (default: zscore hard cut)")
    p.add_argument("--out", type=Path, default=Path("merged_F2.csv"),
                   help="Output CSV path (default: merged_F2.csv)")

    # iteration knob
    p.add_argument("--n-iter", type=int, default=1,
                   help="EM‑style iterations (default 1 = previous behaviour)")

    # z‑score
    p.add_argument("--cutoff", type=float, default=2.5,
                   help="|Z| cutoff for zscore filter")

    # Bayesian knobs
    p.add_argument("--mixture-iters", type=int, default=8000,
                   help="ADVI iterations (bayes only)")
    p.add_argument("--mixture-draws", type=int, default=300,
                   help="Posterior draws per iteration (bayes only)")
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Random fraction of rows used in mixture fit [0‑1]")

    args = p.parse_args(argv)

    # ------------------------------------------------------------------
    print("[0] parsing stream → DataFrames …", file=sys.stderr)
    _, _, refl = stream_to_dfs(args.stream)

    # ------------------------------------------------------------------
    for it in range(args.n_iter):
        tag = f"iter {it+1}/{args.n_iter}"
        if args.method == "zscore":
            if it == 0:
                print(f"[{tag}] provisional scale + z‑score filter …", file=sys.stderr)
                refl = zscore_filter(refl, cutoff=args.cutoff)
            else:
                print(f"[{tag}] re‑merge (zscore weights unchanged) …", file=sys.stderr)
        else:  # Bayesian path
            if mixture_filter is None:
                sys.exit("PyMC missing – reinstall with sered_scaler[bayes]")

            print(f"[{tag}] Bayesian mixture weighting …", file=sys.stderr)
            fit_tbl: pd.DataFrame = (
                refl.sample(frac=args.subsample, random_state=it)
                if 0 < args.subsample < 1 else refl
            )
            weight_tbl = mixture_filter(
                fit_tbl,
                max_iter=args.mixture_iters,
                draws=args.mixture_draws,
            )
            refl = refl.drop(columns=[c for c in refl.columns if c == "weight"]).merge(
                weight_tbl[["event", "h", "k", "l", "weight"]],
                on=["event", "h", "k", "l"], how="left",
            ).fillna({"weight": 0.0})

        # ----- M‑step: weighted merge ---------------------------------
        print(f"[{tag}] weighted re‑scale & merge …", file=sys.stderr)
        merged, _ = weighted_merge(refl)

        # Replace current F² predictions in *refl* so the next mixture
        # iteration (if any) gets an updated µ_ij.
        refl = (
            refl.merge(merged, on=["h", "k", "l"], suffixes=("", "_new"))
                .drop(columns=["I"], errors="ignore")
        )

        # The column is F2_new if it collided with an existing “F2”;
        # otherwise it’s just F2.  Handle both cases.
        if "F2_new" in refl.columns:
            refl["I"] = refl.pop("F2_new")
        else:
            refl["I"] = refl.pop("F2")


    # ------------------------------------------------------------------
    merged.to_csv(args.out, index=False)
    print("done ✔ merged table →", args.out, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
