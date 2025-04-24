#!/usr/bin/env python3
"""
scale_stream.py – scaling **plus symmetry‑aware diagnostics**
============================================================
• One overall scale factor (OSF) per chunk → scaled *.stream* output
• Frame metrics: **Residual, R_dyn, CC_frame**  (per‑frame quality)
• Symmetry diagnostics with **cctbx** (if available): per‑ASU R_sym → `asu_stats.csv`
• Quick‑look PNG plots of OSF, Residual, R_dyn vs Event index

New in this version
-------------------
• Robust space‑group handling (explicit `space_group = …` preferred, otherwise `space_group_override`).
• Completed script (previous truncation fixed).

Install once:
```bash
pip install tqdm matplotlib pandas numpy cctbx‑xfel
```
"""
import re, os, csv, math, warnings
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from cctbx import crystal, miller
    from cctbx.array_family import flex
except ImportError:  # allow run without symmetry stats
    crystal = miller = flex = None
    warnings.warn("cctbx not found – symmetry analysis disabled.")

###############################################################################
# Configuration – EDIT THESE PATHS / PARAMS                                  #
###############################################################################
input_file      = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream'       # replace with your input path
output_file     = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/scaled.stream'    # replace with desired output path
space_group_override  = "I4(1)22"              # e.g. "I4_122"; leave blank to auto
min_redundancy        = 2               # reflections with redundancy < N flagged
max_cycles            = 5               # OSF refinement iterations
###############################################################################

###############################################################################
# Data structures                                                             #
###############################################################################
Reflection = namedtuple("Reflection", "h k l I sigma peak bkg fs ss panel red flag")
FLAG_LOW_REDUNDANCY = 0x01

class Chunk:
    __slots__ = ("header", "footer", "reflections", "event",
                 "osf", "resid", "rdyn", "cc")
    def __init__(self):
        self.header, self.footer = [], []
        self.reflections: list[Reflection] = []
        self.event = "unknown"
        self.osf = 1.0
        self.resid = self.rdyn = self.cc = 0.0

###############################################################################
# Regex helpers                                                               #
###############################################################################
re_begin  = re.compile(r"^----- Begin chunk -----")
re_end    = re.compile(r"^----- End chunk -----")
re_event  = re.compile(r"^\s*Event:\s*(.*)")
re_ref    = re.compile(
    r"^\s*([\-\d]+)\s+([\-\d]+)\s+([\-\d]+)\s+"
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"
    r"([A-Za-z0-9]+)")

re_float = r"([\d\.]+)"
re_cell  = {
    "a":  re.compile(fr"^a\s*=\s*{re_float}\s*A"),
    "b":  re.compile(fr"^b\s*=\s*{re_float}\s*A"),
    "c":  re.compile(fr"^c\s*=\s*{re_float}\s*A"),
    "al": re.compile(fr"^al\s*=\s*{re_float}\s*deg"),
    "be": re.compile(fr"^be\s*=\s*{re_float}\s*deg"),
    "ga": re.compile(fr"^ga\s*=\s*{re_float}\s*deg")
}
re_space_group = re.compile(r"^space_group\s*=\s*(\S+)")

###############################################################################
# 1. Parse stream                                                             #
###############################################################################

def parse_stream(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_lines = []
    i = 0
    while i < len(lines) and not re_begin.match(lines[i]):
        header_lines.append(lines[i]); i += 1
    header = "".join(header_lines)

    chunks: list[Chunk] = []
    while i < len(lines):
        if re_begin.match(lines[i]):
            ch = Chunk(); i += 1
            # chunk header
            while i < len(lines) and not re_ref.match(lines[i]):
                ch.header.append(lines[i])
                if (m := re_event.match(lines[i])):
                    ch.event = m.group(1).strip()
                i += 1
            # reflections
            while i < len(lines) and re_ref.match(lines[i]):
                m = re_ref.match(lines[i])
                h, k, l = map(int, m.group(1, 2, 3))
                I, sig  = map(float, m.group(4, 5))
                peak, bkg = map(float, m.group(6, 7))
                fs, ss = map(float, m.group(8, 9))
                panel = m.group(10)
                ch.reflections.append(Reflection(h, k, l, I, sig, peak, bkg, fs, ss, panel, 1, 0))
                i += 1
            # footer
            while i < len(lines) and not re_end.match(lines[i]):
                ch.footer.append(lines[i]); i += 1
            i += 1  # skip end marker
            chunks.append(ch)
        else:
            i += 1
    return header, chunks

###############################################################################
# 2. Extract symmetry info                                                    #
###############################################################################

def extract_symmetry(header_str):
    sg = space_group_override or (re_space_group.search(header_str).group(1)
                                  if re_space_group.search(header_str) else None)
    if not sg:
        warnings.warn("Space‑group symbol not found – symmetry analysis disabled.")
        return None, None
    vals = {k: None for k in re_cell}
    for ln in header_str.splitlines():
        for key, pat in re_cell.items():
            if (m := pat.match(ln)):
                vals[key] = float(m.group(1))
    if None in vals.values():
        warnings.warn("Incomplete unit‑cell parameters – symmetry analysis disabled.")
        return None, None
    return (vals['a'], vals['b'], vals['c'], vals['al'], vals['be'], vals['ga']), sg

###############################################################################
# 3. Scaling + per‑chunk stats                                                #
###############################################################################

def mean_intensity(chunks):
    sums, counts = defaultdict(float), defaultdict(int)
    for ch in chunks:
        for r in ch.reflections:
            key = (r.h, r.k, r.l)
            sums[key] += r.I
            counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}

def refine_osf(chunks, cycles):
    for c in range(cycles):
        target = mean_intensity(chunks)
        for ch in tqdm(chunks, desc=f"OSF cycle {c + 1}/{cycles}", leave=False):
            num = den = 0.0
            for r in ch.reflections:
                It = target[(r.h, r.k, r.l)]
                num += It * r.I
                den += r.I * r.I
            ch.osf = num / den if den else 1.0
        # global normalisation
        mean_osf = sum(ch.osf for ch in chunks) / len(chunks)
        for ch in chunks:
            ch.osf /= mean_osf
    return target

def apply_stats(chunks, target):
    for ch in tqdm(chunks, desc="Apply scaling & stats", leave=False):
        diff2 = tar_sum = 0.0; Isc = []; It = []; n = 0
        pos, neg = defaultdict(list), defaultdict(list)
        for i, r in enumerate(ch.reflections):
            I_sc = r.I * ch.osf; sig_sc = r.sigma * ch.osf
            ch.reflections[i] = r._replace(I=I_sc, sigma=sig_sc,
                                            flag=r.flag | (FLAG_LOW_REDUNDANCY if r.red < min_redundancy else 0))
            Itgt = target[(r.h, r.k, r.l)]
            diff2 += (I_sc - Itgt) ** 2; tar_sum += Itgt; n += 1
            Isc.append(I_sc); It.append(Itgt)
            pos[(r.h, r.k, r.l)].append(I_sc)
            neg[(-r.h, -r.k, -r.l)].append(I_sc)
        ch.resid = math.sqrt(diff2 / n) / (tar_sum / n) if n else 0.0
        if n:
            ms, mt = np.mean(Isc), np.mean(It)
            cov = np.sum((np.array(Isc) - ms) * (np.array(It) - mt))
            var_s = np.sum((np.array(Isc) - ms) ** 2)
            var_t = np.sum((np.array(It) - mt) ** 2)
            ch.cc = cov / math.sqrt(var_s * var_t) if var_s and var_t else 0.0
        # R_dyn
        num = den = 0.0
        for key, lst in pos.items():
            mates = neg.get(key, [])
            for a, b in zip(lst, mates):
                num += abs(a - b); den += 0.5 * (a + b)
        ch.rdyn = num / den if den else 0.0

###############################################################################
# 4. Symmetry‑group statistics                                                #
###############################################################################

def symmetry_csv(chunks, unit_cell, sg, out_dir):
    if crystal is None:
        return None
    cs = crystal.symmetry(unit_cell=unit_cell, space_group_symbol=sg)
    rows = [{"h": r.h, "k": r.k, "l": r.l, "I": r.I} for ch in chunks for r in ch.reflections]
    df = pd.DataFrame(rows)
    ms = miller.set(cs, flex.miller_index(list(zip(df.h, df.k, df.l))), False)
    df["asu"] = list(ms.map_to_asu().indices())
    stats = []
    for key, grp in df.groupby("asu"):
        I = grp.I.values; mult = len(I); Ibar = I.mean()
        R = np.sum(np.abs(I - Ibar)) / np.sum(I)
        stats.append({"asu_h": key[0], "asu_k": key[1], "asu_l": key[2],
                      "mult": mult, "I_mean": Ibar, "R_sym": R})
    out_csv = os.path.join(out_dir, "asu_stats.csv")
    pd.DataFrame(stats).to_csv(out_csv, index=False)
    return out_csv
###############################################################################
# I/O helpers                                                                 #
###############################################################################

def write_stream(header, chunks, path):
    """Write a new .stream file with scaled intensities."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for ch in chunks:
            fh.write("----- Begin chunk -----\n")
            fh.writelines(ch.header)
            for r in ch.reflections:
                fh.write(
                    f" {r.h:4d} {r.k:4d} {r.l:4d}"
                    f" {r.I:12.2f} {r.sigma:9.2f}"
                    f" {r.peak:10.2f} {r.bkg:10.2f}"
                    f" {r.fs:7.1f} {r.ss:7.1f} {r.panel}\n"
                )
            fh.writelines(ch.footer)
            fh.write("----- End chunk -----\n")


def save_frame_csv(chunks, in_path):
    """Save per-frame stats (OSF, residual, R_dyn, CC) to CSV."""
    out = os.path.join(os.path.dirname(os.path.abspath(in_path)), "frame_stats.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "osf", "residual", "R_dyn", "CC_frame"])
        for ch in chunks:
            w.writerow(
                [
                    ch.event,
                    f"{ch.osf:.6f}",
                    f"{ch.resid:.6f}",
                    f"{ch.rdyn:.6f}",
                    f"{ch.cc:.4f}",
                ]
            )
    return out


def scatter_plot(chunks, attr, fname, ylabel, in_path):
    """Quick scatter plot of a per-frame attribute vs Event index."""
    xs, ys = [], []
    for ch in chunks:
        try:
            idx = int(ch.event.split("//")[-1].split("-")[0])
        except (ValueError, IndexError):
            continue
        xs.append(idx)
        ys.append(getattr(ch, attr))
    if not xs:
        return None
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Event index")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Event index")
    out = os.path.join(os.path.dirname(os.path.abspath(in_path)), fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return out


###############################################################################
# Main                                                                        #
###############################################################################
if __name__ == "__main__":
    header, chunks = parse_stream(input_file)
    uc, sg = extract_symmetry(header)

    target = refine_osf(chunks, max_cycles)
    apply_stats(chunks, target)
    write_stream(header, chunks, output_file)

    frame_csv = save_frame_csv(chunks, input_file)
    png_osf = scatter_plot(chunks, "osf", "osf_vs_event.png", "OSF", input_file)
    png_res = scatter_plot(chunks, "resid", "residual_vs_event.png", "Residual", input_file)
    png_dyn = scatter_plot(chunks, "rdyn", "Rdyn_vs_event.png", "R_dyn", input_file)

    asu_csv = None
    if uc and sg:
        asu_csv = symmetry_csv(
            chunks, uc, sg, os.path.dirname(os.path.abspath(input_file))
        )

    # Console summary
    # print("\nFrame stats (saved to frame_stats.csv):\n")
    # print(f"{'Event':15s} {'OSF':>8s} {'Resid':>8s} {'R_dyn':>8s} {'CC':>6s}")
    # for ch in chunks:
    #     print(f"{ch.event:15s} {ch.osf:8.3f} {ch.resid:8.3f} {ch.rdyn:8.3f} {ch.cc:6.2f}")

    print(f"\n[✓] Scaled stream  → {output_file}")
    print(f"[✓] Frame CSV     → {frame_csv}")
    if asu_csv:
        print(f"[✓] ASU CSV       → {asu_csv}")
    for p in (png_osf, png_res, png_dyn):
        if p:
            print(f"[✓] Plot          → {p}")
    if not asu_csv and crystal is None:
        print("[!] Install cctbx to enable symmetry-group analysis.")
