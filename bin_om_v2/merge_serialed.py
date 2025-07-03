#!/usr/bin/env python
"""
merge_serialed.py
-----------------
Merge one or many SerialED CrystFEL *.stream files with careless.

Example
-------
python merge_serialed.py \
        mfm300sim_0.0_0.0.stream \
        --spacegroup "I 222" \
        --dmin 0.50 \
        --iterations 300 \
        --outdir merge/mfm300

Dependencies
------------
conda create -n careless_ed python=3.12
conda activate careless_ed
pip install careless reciprocalspaceship pandas
"""
from __future__ import annotations
import argparse, pathlib, sys
import reciprocalspaceship as rs
from careless import mono as merge            # part of the public API
                                       # (careless >= 1.4.0)
# ---------------------------------------------------------------------
def load_streams(paths: list[str]) -> rs.DataSet:
    """Read and concatenate CrystFEL stream files into one DataSet."""
    datasets = []
    for p in paths:
        ds = rs.read_crystfel(p)      # parses header + all chunks
        if "BATCH" not in ds.columns: # safety-net: create a batch tag
            ds["BATCH"] = ds["IMAGE_ID"]
        datasets.append(ds)
    return rs.concat(datasets, check_isomorphous=False)

def run_careless(ds: rs.DataSet,
                 spacegroup: str,
                 dmin: float | None,
                 iterations: int,
                 outdir: pathlib.Path) -> None:
    """Configure and execute a careless mono-merging experiment."""
    outdir.mkdir(parents=True, exist_ok=True)

    exp = merge.Experiment(
        data=ds,
        recipe="mono",
        metadata_keys=["BATCH", "s1x", "s1y"],
        spacegroup=spacegroup,
        dmin=dmin,
        iterations=iterations,
    )
    results = exp.run()                 # GPU if available, else CPU
    # Save outputs
    results.to_mtz(outdir / "merged_0.mtz")
    results.save(outdir)                # scale.pkl, structure_factor.pkl

# ---------------------------------------------------------------------
def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Merge SerialED CrystFEL streams with careless.")
    parser.add_argument("streams", nargs="+",
                        help="Input *.stream file(s)")
    parser.add_argument("--spacegroup", required=True,
                        help='Target symmetry, e.g. "I 4"')
    parser.add_argument("--dmin", type=float, default=None,
                        help="Resolution cutoff in Å")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Optimisation steps (default 30 k)")
    parser.add_argument("--outdir", default="careless_out",
                        help="Output folder (created if missing)")
    args = parser.parse_args(argv)

    ds = load_streams(args.streams)
    run_careless(ds, args.spacegroup, args.dmin,
                 args.iterations, pathlib.Path(args.outdir))

if __name__ == "__main__":
    main(sys.argv[1:])
