#!/usr/bin/env python3
"""
qc_report.py — Minimal QC for CrystFEL HKL files.

Designed to be called from a tiny shell wrapper that sets:
  DIR=</path/to/folder/with/crystfel.hkl>
  CELL=</path/to/unit.cell or .pdb>
  SYM="4/mmm"
  OUTDIR=qc_stats
  LOWRES=4
  HIGHRES=0.4
  WILSON="--wilson" (optional)

Example:
  python3 qc_report.py --dir "$DIR" --cell "$CELL" --symmetry "$SYM" \
                       --outdir "$OUTDIR" --lowres "$LOWRES" --highres "$HIGHRES" $WILSON
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# -------- helpers --------

def run_cmd(cmd, log_path) -> int:
    """Run command, tee stdout+stderr to log_path; return exit code."""
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
        proc.wait()
        return proc.returncode

def parse_shell_table(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Whitespace-or-tab separated table; ignore comments/blank lines."""
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip("\n") for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return [], []
    header = lines[0]
    body = lines[1:]
    if "\t" in header:
        cols = header.split("\t")
        rows = [ln.split("\t") for ln in body]
    else:
        splitter = re.compile(r"\s+")
        cols = splitter.split(header.strip())
        rows = [splitter.split(ln.strip()) for ln in body]
    table = []
    for r in rows:
        if len(r) < len(cols):
            r = r + [""] * (len(cols) - len(r))
        table.append({cols[i]: r[i] for i in range(len(cols))})
    return table, cols

def f2(x: Optional[str]) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def find_col(candidates: List[str], cols: List[str]) -> Optional[str]:
    lowmap = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]
    return None

def compute_overall_completeness(shell_path: str) -> Optional[float]:
    """Fallback path: compute overall completeness from the shell table."""
    table, cols = parse_shell_table(shell_path)
    if not table:
        return None
    nposs = find_col(["nposs","n_poss","npossible","n_possible","n_expected"], cols)
    nobs  = find_col(["nobs","n_obs","nmeas","n_meas","nobserved","n_reflections"], cols)
    comp  = find_col(["completeness","compl","percent_completeness","comp","complete"], cols)

    # Prefer weighted by possible reflections
    if nposs and nobs:
        tot_poss = 0.0
        tot_obs  = 0.0
        for row in table:
            p = f2(row.get(nposs)); o = f2(row.get(nobs))
            if p is not None and o is not None:
                tot_poss += p; tot_obs += o
        if tot_poss > 0:
            return 100.0 * (tot_obs / tot_poss)

    # Fallback: average of per-shell completeness
    if comp:
        vals = [f2(r.get(comp)) for r in table if f2(r.get(comp)) is not None]
        if vals:
            return sum(vals) / len(vals)

    return None

def grep_last_number(log_path: str, key: str) -> Optional[str]:
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f if key in ln]
    if not lines:
        return None
    line = lines[-1]
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?%?", line)
    return nums[-1] if nums else None

def extract_bfactor_from_log(log_path: str) -> Optional[float]:
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    m = re.search(r"(wilson\s+)?b\s*factor\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", txt, re.IGNORECASE)
    return float(m.group(2)) if m else None

def extract_overall_completeness_from_log(log_path: str) -> Optional[float]:
    """
    Preferred path: read 'Overall completeness = XX.XXXXXX %' from check_hkl_completeness.log.
    Robust to spacing/case and optional percent sign.
    """
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    m = re.search(
        r"overall\s+completeness\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        txt,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None

# -------- main --------

def main():
    ap = argparse.ArgumentParser(description="QC for crystfel.hkl [+hkl1/hkl2] in a directory.")
    ap.add_argument("--dir", help="Directory containing crystfel.hkl (and optional crystfel.hkl1/2). If omitted, use CWD.")
    ap.add_argument("--cell", "-c", required=True, help="Unit cell file (.cell or .pdb)")
    ap.add_argument("--symmetry", "-y", required=True, help='Point group symmetry, e.g. \"4/mmm\"')
    ap.add_argument("--lowres", type=float, default=4.0, help="Low resolution cutoff d (Å)")
    ap.add_argument("--highres", type=float, default=0.4, help="High resolution cutoff d (Å)")
    ap.add_argument("--outdir", "-o", default="qc_stats", help="Output directory (created under --dir/CWD)")
    ap.add_argument("--nshells", type=int, default=20, help="Number of resolution shells")
    ap.add_argument("--wilson", action="store_true", help="Also compute Wilson plot / B factor")
    args = ap.parse_args()

    # Move into the data directory if provided
    if args.dir:
        os.makedirs(args.dir, exist_ok=True)
        os.chdir(args.dir)

    data_dir = os.getcwd()
    hkl  = os.path.join(data_dir, "crystfel.hkl")
    hkl1 = os.path.join(data_dir, "crystfel.hkl1")
    hkl2 = os.path.join(data_dir, "crystfel.hkl2")

    if not os.path.exists(hkl):
        print(f"ERROR: {hkl} not found.", file=sys.stderr)
        sys.exit(2)

    # Make OUTDIR inside the data directory
    outdir = os.path.join(data_dir, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"== QC directory: {data_dir}")
    print(f"Cell: {args.cell}")
    print(f"Symmetry: {args.symmetry}")
    print(f"Range: {args.lowres}-{args.highres} Å; shells={args.nshells}")
    print(f"Output: {outdir}")

    # 1) Completeness via check_hkl (NO --wilson)
    comp_shell = os.path.join(outdir, "check_shell.tsv")
    comp_log   = os.path.join(outdir, "check_hkl_completeness.log")
    cmd_check = [
        "check_hkl", hkl, "-p", args.cell, f"--symmetry={args.symmetry}",
        f"--lowres={args.lowres}", f"--highres={args.highres}",
        f"--nshells={args.nshells}", f"--shell-file={comp_shell}"
    ]
    print("\n[1/3] check_hkl (completeness)…")
    rc = run_cmd(cmd_check, comp_log)
    if rc != 0:
        print(f"check_hkl failed (see {comp_log})", file=sys.stderr)
        sys.exit(rc)

    # Prefer reading "Overall completeness" from the log; fallback to computing from shell table.
    completeness = extract_overall_completeness_from_log(comp_log)
    if completeness is None:
        completeness = compute_overall_completeness(comp_shell)
    comp_str = f"{completeness:.2f}%" if completeness is not None else "NA"

    # 2) Optional Wilson
    b_str = None
    if args.wilson:
        wil_shell = os.path.join(outdir, "wilson_shell.tsv")
        wil_log   = os.path.join(outdir, "check_hkl_wilson.log")
        cmd_wil = [
            "check_hkl", hkl, "-p", args.cell, f"--symmetry={args.symmetry}",
            f"--lowres={args.lowres}", f"--highres={args.highres}",
            f"--nshells={args.nshells}", "--wilson", f"--shell-file={wil_shell}"
        ]
        print("[2/3] check_hkl (--wilson)…")
        _ = run_cmd(cmd_wil, wil_log)  # non-zero sometimes when atoms unknown; still log
        b = extract_bfactor_from_log(wil_log)
        b_str = f"{b:.2f} Å²" if b is not None else "NA"
    else:
        print("[2/3] (skip --wilson)")

    # 3) CC1/2 & Rsplit via compare_hkl if halves are present
    cc_str, rs_str = "NA", "NA"
    if os.path.exists(hkl1) and os.path.exists(hkl2):
        print("[3/3] compare_hkl (CC1/2 & Rsplit)…")
        cc_shell = os.path.join(outdir, "compare_cc12_shell.tsv")
        rs_shell = os.path.join(outdir, "compare_rsplit_shell.tsv")
        cc_log   = os.path.join(outdir, "compare_cc12.log")
        rs_log   = os.path.join(outdir, "compare_rsplit.log")

        cmd_cc = [
            "compare_hkl", hkl1, hkl2, "-p", args.cell, f"--symmetry={args.symmetry}",
            f"--lowres={args.lowres}", f"--highres={args.highres}",
            f"--nshells={args.nshells}", "--fom=CC", f"--shell-file={cc_shell}"
        ]
        run_cmd(cmd_cc, cc_log)
        cc_val = grep_last_number(cc_log, "CC")
        if cc_val: cc_str = cc_val

        cmd_rs = [
            "compare_hkl", hkl1, hkl2, "-p", args.cell, f"--symmetry={args.symmetry}",
            f"--lowres={args.lowres}", f"--highres={args.highres}",
            f"--nshells={args.nshells}", "--fom=Rsplit", f"--shell-file={rs_shell}"
        ]
        run_cmd(cmd_rs, rs_log)
        rs_val = grep_last_number(rs_log, "Rsplit")
        if rs_val: rs_str = rs_val
    else:
        print("[3/3] compare_hkl skipped (crystfel.hkl1/2 not found)")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Completeness: {comp_str}")
    print(f"CC1/2:       {cc_str}")
    print(f"Rsplit:      {rs_str}")
    if args.wilson:
        print(f"Wilson B:    {b_str}")
    print(f"\nShell tables & logs in: {outdir}")

if __name__ == "__main__":
    main()
