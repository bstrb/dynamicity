#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-frame estimates of two global parameters (e.g., x/y detector translations)
from a Millepede-II C-binary file (e.g., produced by CrystFEL/indexamajig → mille).

Default global IDs: 1 (dx), 2 (dy). Override with --globals.
Optionally scale results by a factor (e.g., --scale 17800).

Usage:
  python extract_mille_shifts.py mille-data.bin -o per_frame_dx_dy.csv --scale 17800
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def iter_records(path: Path):
    """Yield (rec_idx, glder, inder) for each event record in Millepede C-binary.

    The file is organized as records, each starting with a 4-byte length word:
      length_word = (#entries << 1) with sign indicating float size: >=0 → float32, <0 → float64
    followed by `nr` floats (derivatives & residual/sigma values) and `nr` ints (labels),
    both aligned such that glder[k] pairs with inder[k].
    """
    with path.open("rb") as f:
        rec_idx = 0
        while True:
            head = f.read(4)
            if not head or len(head) < 4:
                break
            (length_word,) = struct.unpack("<i", head)
            nr = abs(length_word >> 1)
            if nr <= 0:
                break
            is_f32 = (length_word >= 0)
            fsize = 4 if is_f32 else 8

            gbytes = f.read(fsize * nr)
            ibytes = f.read(4 * nr)
            if len(gbytes) != fsize * nr or len(ibytes) != 4 * nr:
                sys.stderr.write("Truncated record encountered; stopping.\n")
                break

            fmt_g = "<" + ("f" if is_f32 else "d") * nr
            fmt_i = "<" + "i" * nr
            try:
                glder = list(struct.unpack(fmt_g, gbytes))
                inder = list(struct.unpack(fmt_i, ibytes))
            except struct.error as e:
                sys.stderr.write(f"Struct unpack error: {e}\n")
                break

            rec_idx += 1
            yield rec_idx, glder, inder


def per_frame_two_globals(records_iter, global_ids=(1, 2)):
    """
    For each record (frame), compute frame-wise estimates of two global parameters
    using local elimination via the Schur complement.

    Returns a pandas.DataFrame with columns:
      record, dx, dy, dx_err, dy_err, n_locals, n_meas
    where dx/dy correspond to the first/second entries in `global_ids`.
    """
    gpos = {gid: i for i, gid in enumerate(global_ids)}
    G = len(global_ids)
    out_rows = []

    for rec_idx, glder, inder in records_iter:
        nr = len(inder)
        i = 0

        local_labels_all = set()
        meas_list = []  # list of (res, sig, loc_labs, dL, glob_labs, dG)

        # Parse all measurements in this record
        while i < (nr - 1):
            i += 1
            # end of local label list
            while i < nr and inder[i] != 0:
                i += 1
            ja = i
            i += 1
            # end of global label list
            while i < nr and inder[i] != 0:
                i += 1
            jb = i
            i += 1

            # Special (skip) records: when (ja+1 == jb) and glder[jb] < 0
            if (ja + 1 == jb) and (jb < len(glder)) and (glder[jb] < 0.0):
                nsp = int(-glder[jb])
                i += nsp - 1
                continue

            # move to end-of-measurement marker
            while i < nr and inder[i] != 0:
                i += 1
            i -= 1

            # Extract pieces
            if not (0 <= ja < len(glder) and 0 <= jb < len(glder)):
                continue
            res = glder[ja]
            sig = glder[jb]
            if not (math.isfinite(res) and math.isfinite(sig) and sig != 0.0):
                continue

            loc_labs = inder[ja + 1: jb]
            dL = [glder[k] for k in range(ja + 1, jb)]

            is_global = jb < i
            if is_global:
                glob_labs_full = inder[jb + 1: i + 1]
                dG_full = [glder[k] for k in range(jb + 1, i + 1)]
            else:
                glob_labs_full = []
                dG_full = []

            # Keep only target globals
            sel = [(lab, der) for (lab, der) in zip(glob_labs_full, dG_full) if lab in gpos]
            glob_labs = [lab for (lab, _) in sel]
            dG = [der for (_, der) in sel]

            local_labels_all.update(loc_labs)
            meas_list.append((res, sig, loc_labs, dL, glob_labs, dG))

        loc_list = sorted(local_labels_all)
        L = len(loc_list)

        # If this record has no measurements or did not touch our globals, return NaNs
        if len(meas_list) == 0 or all(len(m[4]) == 0 for m in meas_list):
            out_rows.append({
                "record": rec_idx,
                "dx": float("nan"),
                "dy": float("nan"),
                "dx_err": float("nan"),
                "dy_err": float("nan"),
                "n_locals": L,
                "n_meas": 0,
            })
            continue

        loc_index = {lab: j for j, lab in enumerate(loc_list)}

        # Build normal-equation blocks
        A_ll = np.zeros((L, L), dtype=float)
        A_lg = np.zeros((L, G), dtype=float)
        A_gg = np.zeros((G, G), dtype=float)
        b_l = np.zeros((L,), dtype=float)
        b_g = np.zeros((G,), dtype=float)

        for (res, sig, loc_labs, dL, glob_labs, dG) in meas_list:
            w = 1.0 / (sig * sig)

            # locals
            for a, lab_a in enumerate(loc_labs):
                ia = loc_index[lab_a]
                da = dL[a]
                b_l[ia] += w * da * res
                for b, lab_b in enumerate(loc_labs):
                    ib = loc_index[lab_b]
                    db = dL[b]
                    A_ll[ia, ib] += w * da * db

            # globals
            if len(glob_labs) > 0:
                # place derivatives in fixed order of global_ids
                dG_vec = np.zeros(G, dtype=float)
                for lab, der in zip(glob_labs, dG):
                    dG_vec[gpos[lab]] += der

                b_g += w * dG_vec * res
                A_gg += w * np.outer(dG_vec, dG_vec)

                if L > 0:
                    # cross block
                    for a, lab_l in enumerate(loc_labs):
                        ia = loc_index[lab_l]
                        da = dL[a]
                        A_lg[ia, :] += w * da * dG_vec

        # Eliminate locals: N = A_gg - A_lg^T A_ll^{-1} A_lg ; n = b_g - A_lg^T A_ll^{-1} b_l
        if L > 0:
            try:
                inv_All = np.linalg.inv(A_ll)
            except np.linalg.LinAlgError:
                inv_All = np.linalg.pinv(A_ll)
            N = A_gg - A_lg.T @ inv_All @ A_lg
            n = b_g - A_lg.T @ inv_All @ b_l
        else:
            N = A_gg
            n = b_g

        # Solve and get errors
        try:
            dp = np.linalg.solve(N, n)
        except np.linalg.LinAlgError:
            dp = np.linalg.pinv(N) @ n
        try:
            cov = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(N)
        errs = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

        # Map back to dx/dy in order of global_ids
        dx = dp[0] if len(dp) > 0 else float("nan")
        dy = dp[1] if len(dp) > 1 else float("nan")
        dx_err = errs[0] if len(errs) > 0 else float("nan")
        dy_err = errs[1] if len(errs) > 1 else float("nan")

        out_rows.append({
            "record": rec_idx,
            "dx": dx,
            "dy": dy,
            "dx_err": dx_err,
            "dy_err": dy_err,
            "n_locals": L,
            "n_meas": len(meas_list),
        })

    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser(description="Extract per-frame estimates of two global parameters (e.g., x/y shifts) from Millepede C-binary.")
    ap.add_argument("binary", type=Path, help="Path to mille-data.bin (Millepede C-binary).")
    ap.add_argument("-o", "--output", type=Path, default=None,
                help="Output CSV path (default: per_frame_dx_dy.csv in same folder as input).")
    ap.add_argument("--globals", nargs=2, type=int, default=(1, 2),
                    help="Two global parameter IDs to extract (default: 1 2).")
    ap.add_argument("--scale", type=float, default=17857.14285714286,
                    help="Optional scale factor to multiply dx,dy (e.g., res from geom file).")
    args = ap.parse_args()

    if not args.binary.exists():
        sys.stderr.write(f"File not found: {args.binary}\n")
        sys.exit(1)
    
    if args.output is None:
        output = args.binary.parent / "per_frame_dx_dy.csv"

    df = per_frame_two_globals(iter_records(args.binary), global_ids=tuple(args.globals))
    if args.scale is not None:
        df["dx_scaled"] = df["dx"] * args.scale
        df["dy_scaled"] = df["dy"] * args.scale
        df["dx_err_scaled"] = df["dx_err"] * args.scale
        df["dy_err_scaled"] = df["dy_err"] * args.scale

    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
