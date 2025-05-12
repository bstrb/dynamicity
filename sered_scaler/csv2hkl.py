#!/usr/bin/env python3
"""
csv2hkl.py

Read a CSV with columns h,k,l,F2,sigF2 and output a SHELX‐compatible .hkl file:
  • h, k, l as 4-character wide integers
  • F2 and sigF2 as 8-character wide floats with 2 decimal places
  • terminates with the “0 0 0 0.00 0.00” line and a SHELX header comment
"""

import csv
import argparse
import sys

def csv_to_hkl(infile, outfile, remark=None):
    reader = csv.DictReader(infile)
    # validate columns
    expected = ['h', 'k', 'l', 'F2', 'sigF2']
    for col in expected:
        if col not in reader.fieldnames:
            sys.exit(f"ERROR: Input CSV missing required column '{col}'")
    # write lines
    for row in reader:
        h  = int(row['h'])
        k  = int(row['k'])
        l  = int(row['l'])
        f2 = float(row['F2'])
        s2 = float(row['sigF2'])
        # 4 spaces for h,k,l; 8 for F2,sigF2
        line = f"{h:4d}{k:4d}{l:4d}{f2:8.2f}{s2:8.2f}"
        outfile.write(line + "\n")
    # termination line
    outfile.write(f"{0:4d}{0:4d}{0:4d}{0.00:8.2f}{0.00:8.2f}\n")
    # optional SHELX remark
    if remark is None:
        remark = "_computing_structure_solution     'SHELXT 2018/2 (Sheldrick, 2018)'"
    outfile.write("\n" + remark + "\n")

def main():
    p = argparse.ArgumentParser(
        description="Convert CSV (h,k,l,F2,sigF2) to SHELX .hkl format"
    )
    p.add_argument("csv_in",  help="Input CSV file path (or '-' for stdin)")
    p.add_argument("hkl_out", help="Output .hkl file path (or '-' for stdout)")
    p.add_argument(
        "--remark", "-r",
        help="Override the SHELX remark line (default: SHELXT 2018/2)",
        default=None
    )
    args = p.parse_args()

    # open input
    if args.csv_in == "-":
        infile = sys.stdin
    else:
        infile = open(args.csv_in, newline='')
    # open output
    if args.hkl_out == "-":
        outfile = sys.stdout
    else:
        outfile = open(args.hkl_out, "w", newline='')

    try:
        csv_to_hkl(infile, outfile, remark=args.remark)
    finally:
        if infile is not sys.stdin:
            infile.close()
        if outfile is not sys.stdout:
            outfile.close()

if __name__ == "__main__":
    main()
