#!/usr/bin/env python3
"""
scale_stream_dyn.py – dynamical-aware scaler for huge .stream files
==================================================================

Command-line usage
------------------
python scale_stream_dyn.py --input raw.stream --output scaled.stream \
       --space_group I4_122 --dqe detector_dqe.csv
"""
from __future__ import annotations
import os, sys, argparse, math, warnings

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from dynlib     import (d_spacing, s_in_Ainv, parse_stream, write_stream,
                         extract_symmetry, load_dqe_table,
                         FLAG_DYN_OUTLIER)
from dynmetrics import (pattern_scale, pattern_metrics, select_good_patterns,
                        shell_means, flag_dyn_outliers)

###############################################################################
# --- CLI --------------------------------------------------------------------#
###############################################################################
def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input",  required=True,  help=".stream input")
    ap.add_argument("--output", required=True,  help="scaled .stream output")
    ap.add_argument("--space_group", default="I4(1)22", help="override space-group symbol")
    ap.add_argument("--dqe",   default="", help="CSV with 's,gain' (optional)")
    ap.add_argument("--plot",  action="store_true", help="show diagnostic plots")
    ap.add_argument("--keep_flags", action="store_true",
                    help="include ;FLAGS=xx in output stream")
    return ap.parse_args()

###############################################################################
# --- Main pipeline ----------------------------------------------------------#
###############################################################################
def main():
    args = get_args()
    header, chunks = parse_stream(args.input)
    cell_params, sg = extract_symmetry(header, args.space_group or None)
    if not cell_params:
        print(">> Unit cell / symmetry missing – abort.")
        sys.exit(1)
    sg = sg or args.space_group

    # --------------------------------------------------------------------- #
    # 1.  fast per-pattern scale k_p                                        #
    pattern_scale(chunks, cell_params)
    # 2.  per-pattern metrics                                               #
    pattern_metrics(chunks, cell_params, sg)
    select_good_patterns(chunks)
    print(f">> kept {sum(ch.good for ch in chunks)} / {len(chunks)} patterns")

    # 3.  Wilson-shell kinematic means                                      #
    s_centers, mean_I = shell_means(chunks, cell_params)

    # 4.  flag reflection-level outliers                                    #
    flag_dyn_outliers(chunks, cell_params, s_centers, mean_I)

    # 5.  optional DQE / MTF correction                                     #
    D = load_dqe_table(args.dqe)
    for ch in chunks:
        for i,r in enumerate(ch.reflections):
            # d   = dynlib.d_spacing(r.h,r.k,r.l,*cell_params)
            # s   = dynlib.s_in_Ainv(d)
            d   = d_spacing(r.h,r.k,r.l,*cell_params)
            s   = s_in_Ainv(d)
            gain= D(s)
            ch.reflections[i] = r._replace(I = r.I * ch.scale / gain,
                                           sigma = r.sigma * ch.scale / gain)
    # 6.  smooth quartic scale S(s)                                         #
    # compute log(I) vs s for *good & unflagged* reflections
    xs, ys = [], []
    for ch in chunks:
        if not ch.good: continue
        for r in ch.reflections:
            if r.flag & FLAG_DYN_OUTLIER: continue
            # d = dynlib.d_spacing(r.h,r.k,r.l,*cell_params)
            # s = dynlib.s_in_Ainv(d)
            d = d_spacing(r.h,r.k,r.l,*cell_params)
            s = s_in_Ainv(d)
            if r.I>0:
                xs.append(s);  ys.append(math.log(r.I))
    if len(xs) < 20:
        warnings.warn("too few reflections for scale fit – skipping quartic")
        poly = np.poly1d([0.0])
    else:
        poly = np.poly1d(np.polyfit(xs, ys, 4))   # log(I) ~ p(s)

    # apply quartic correction
    for ch in chunks:
        for i,r in enumerate(ch.reflections):
            # d  = dynlib.d_spacing(r.h,r.k,r.l,*cell_params)
            # s  = dynlib.s_in_Ainv(d)
            d  = d_spacing(r.h,r.k,r.l,*cell_params)
            s  = s_in_Ainv(d)
            factor = math.exp(poly(s))
            ch.reflections[i] = r._replace(I = r.I / factor,
                                           sigma = r.sigma / factor)

    # 7.  write scaled stream                                               #
    write_stream(header, chunks, args.output, include_flags=args.keep_flags)
    print(f"[✓] scaled stream  →  {args.output}")

    # 8.  per-frame CSV + quick plots                                       #
    rows = []
    for ch in chunks:
        rows.append(dict(event=ch.event,
                         k_p=ch.scale,
                         good=int(ch.good),
                         R_sysAbs=ch.R_sysAbs,
                         R_Friedel=ch.R_Friedel,
                         p90=ch.p90_log_spread))
    df = pd.DataFrame(rows)
    csv_out = os.path.splitext(args.output)[0] + "_frame_stats.csv"
    df.to_csv(csv_out, index=False); print(f"[✓] frame CSV     →  {csv_out}")

    if args.plot:
        plt.scatter(df.index, df.k_p, s=4)
        plt.xlabel("pattern #"); plt.ylabel("k_p"); plt.title("per-pattern scale")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
