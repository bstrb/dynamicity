#!/usr/bin/env python3
"""
scale_stream.py – chunk‑wise intensity scaling **with residuals**
================================================================
* **One OSF per chunk** (no B‑factor, no partiality)
* Keeps the global header and reflection columns unchanged except for scaled
  *I* and *σ(I)*.
* Shows **tqdm** progress bars.
* Outputs `osf_values.csv` containing **Event, OSF, residual** next to the input
  file.
* Saves two diagnostic plots in the same folder:
    * `osf_vs_event.png` – OSF vs first index of the `Event //X‑Y` tag.
    * `residual_vs_event.png` – residual vs the same index.

**Residual definition**
----------------------
For each chunk we compute
```
residual = sqrt( Σ (I_scaled − I_target)² / N ) / mean(I_target)
```
so it’s a dimensionless RMS deviation of scaled intensities from the across‑
chunks mean, normalised by that mean.

Edit the *Configuration* block below to point at your files.  Requires
`tqdm` and `matplotlib` (install with `pip install tqdm matplotlib`).
"""
import re
import os
import csv
import math
from collections import defaultdict, namedtuple

from tqdm import tqdm
import matplotlib.pyplot as plt

###############################################################################
# Configuration                                                               #
###############################################################################
input_file      = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream'       # replace with your input path
output_file     = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/scaled.stream'    # replace with desired output path
min_redundancy = 2                 # flag reflections with redundancy < N
max_cycles     = 5                 # LSQ iterations

###############################################################################
# Data containers                                                             #
###############################################################################
Reflection = namedtuple(
    "Reflection",
    "h k l I sigma peak bkg fs ss panel red flag")
FLAG_LOW_REDUNDANCY = 0x01

class Crystal:
    __slots__ = ("header", "reflections", "footer", "osf", "resid", "event")
    def __init__(self):
        self.header      = []
        self.reflections = []
        self.footer      = []
        self.osf         = 1.0
        self.resid       = 0.0  # RMS residual after scaling
        self.event       = "unknown"

###############################################################################
# Regexes                                                                     #
###############################################################################
re_begin  = re.compile(r"^----- Begin chunk -----")
re_end    = re.compile(r"^----- End chunk -----")
re_event  = re.compile(r"^\s*Event:\s*(.*)")
re_ref = re.compile(
    r"^\s*([\-\d]+)\s+([\-\d]+)\s+([\-\d]+)\s+"
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"
    r"([A-Za-z0-9]+)")

###############################################################################
# Parsing                                                                     #
###############################################################################

def parse_stream(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_lines = []
    crystals = []
    i = 0
    while i < len(lines) and not re_begin.match(lines[i]):
        header_lines.append(lines[i])
        i += 1

    while i < len(lines):
        if re_begin.match(lines[i]):
            cry = Crystal()
            i += 1
            while i < len(lines) and not re_ref.match(lines[i]):
                line = lines[i]
                cry.header.append(line)
                m_ev = re_event.match(line)
                if m_ev:
                    cry.event = m_ev.group(1).strip()
                i += 1
            while i < len(lines) and re_ref.match(lines[i]):
                m = re_ref.match(lines[i])
                h,k,l        = map(int,   m.group(1,2,3))
                I,sigma      = map(float, m.group(4,5))
                peak,bkg     = map(float, m.group(6,7))
                fs,ss        = map(float, m.group(8,9))
                panel        = m.group(10)
                cry.reflections.append(Reflection(h,k,l,I,sigma,peak,bkg,fs,ss,panel,1,0))
                i += 1
            while i < len(lines) and not re_end.match(lines[i]):
                cry.footer.append(lines[i])
                i += 1
            i += 1
            crystals.append(cry)
        else:
            i += 1
    return "".join(header_lines), crystals

###############################################################################
# Scaling helpers                                                             #
###############################################################################

def mean_intensity_map(crystals):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for cry in crystals:
        for r in cry.reflections:
            key = (r.h,r.k,r.l)
            sums[key]   += r.I
            counts[key] += 1
    return {k: sums[k]/counts[k] for k in sums}


def refine_osf(crystals, cycles):
    for cyc in range(cycles):
        target = mean_intensity_map(crystals)
        for cry in tqdm(crystals, desc=f"Cycle {cyc+1}/{cycles}", leave=False):
            num = den = 0.0
            for r in cry.reflections:
                It = target[(r.h,r.k,r.l)]
                num += It * r.I
                den += r.I * r.I
            cry.osf = num/den if den else 1.0
        mean_osf = sum(c.osf for c in crystals)/len(crystals)
        for c in crystals:
            c.osf /= mean_osf
    return target


def apply_scaling_and_residual(crystals, target_map):
    for cry in tqdm(crystals, desc="Applying scaling & residual", leave=False):
        sq_sum = 0.0
        tar_sum = 0.0
        n = 0
        for idx, r in enumerate(cry.reflections):
            scale = cry.osf
            I_scaled = r.I * scale
            sigma_scaled = r.sigma * scale
            flag = r.flag | (FLAG_LOW_REDUNDANCY if r.red < min_redundancy else 0)
            cry.reflections[idx] = r._replace(I=I_scaled, sigma=sigma_scaled, flag=flag)
            It = target_map[(r.h,r.k,r.l)]
            sq_sum += (I_scaled - It)**2
            tar_sum += It
            n += 1
        cry.resid = (math.sqrt(sq_sum / n) / (tar_sum / n)) if n else 0.0

###############################################################################
# I/O helpers                                                                 #
###############################################################################

def write_stream(header, crystals, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for cry in crystals:
            fh.write("----- Begin chunk -----\n")
            fh.writelines(cry.header)
            for r in cry.reflections:
                fh.write(f" {r.h:4d} {r.k:4d} {r.l:4d}"
                         f" {r.I:12.2f} {r.sigma:9.2f}"
                         f" {r.peak:10.2f} {r.bkg:10.2f}"
                         f" {r.fs:7.1f} {r.ss:7.1f} {r.panel}\n")
            fh.writelines(cry.footer)
            fh.write("----- End chunk -----\n")


def save_csv(crystals, in_path):
    out_csv = os.path.join(os.path.dirname(os.path.abspath(in_path)), "osf_values.csv")
    with open(out_csv, "w", newline="") as fh:
        csv.writer(fh).writerow(["event", "osf", "residual"])
        for c in crystals:
            csv.writer(fh).writerow([c.event, f"{c.osf:.6f}", f"{c.resid:.6f}"])
    return out_csv


def scatter_plot(crystals, in_path, attr, fname, ylabel):
    xs, ys = [], []
    for c in crystals:
        try:
            idx = int(c.event.split("//")[-1].split("-")[0])
        except (IndexError, ValueError):
            continue
        xs.append(idx)
        ys.append(getattr(c, attr))
    if not xs:
        return None
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("First index of Event tag (before '-')")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Event index")
    out_png = os.path.join(os.path.dirname(os.path.abspath(in_path)), fname)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png

###############################################################################
# Main                                                                        #
###############################################################################
if __name__ == "__main__":
    header, crystals = parse_stream(input_file)
    target_map = refine_osf(crystals, max_cycles)
    apply_scaling_and_residual(crystals, target_map)
    write_stream(header, crystals, output_file)

    csv_path  = save_csv(crystals, input_file)
    png_osf   = scatter_plot(crystals, input_file, "osf", "osf_vs_event.png", "OSF")
    png_resid = scatter_plot(crystals, input_file, "resid", "residual_vs_event.png", "Residual")

    # Console summary
    # print("\nChunk summary (saved to osf_values.csv):\n")
    # print(f"{'Event':15s}  {'OSF':>8s}  {'Residual':>9s}")
    # for c in crystals:
    #     print(f"{c.event:15s}  {c.osf:8.4f}  {c.resid:9.4f}")

    print(f"\n[✓] Scaled stream → {output_file}")
    print(f"[✓] CSV → {csv_path}")
    if png_osf:   print(f"[✓] Plot → {png_osf}")
    if png_resid: print(f"[✓] Plot → {png_resid}")
