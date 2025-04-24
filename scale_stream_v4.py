#!/usr/bin/env python3
"""
scale_stream.py – chunk‑wise intensity scaling with OSF report
=============================================================
*   **One OSF per chunk** (no B‑factor, no partiality)
*   Keeps full file header and reflection columns unchanged except for *I* and
    *σ(I)* after scaling.
*   Progress bars via **tqdm** while scaling.
*   Writes a CSV (`osf_values.csv`) with *Event* labels and OSFs in the input
    directory.
*   Saves a PNG (`osf_vs_event.png`) plotting OSF versus the first index of the
    `Event: //X‑Y` tag.

Edit the *Configuration* block below to point at your files.  Install
```
pip install tqdm matplotlib
```
if they’re not already available.
"""
import re
import os
import csv
import math
from collections import defaultdict, namedtuple

from tqdm import tqdm            # progress bars
import matplotlib.pyplot as plt   # plotting

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
    __slots__ = ("header", "reflections", "footer", "osf", "flag", "event")
    def __init__(self):
        self.header      = []
        self.reflections = []   # list[Reflection]
        self.footer      = []
        self.osf         = 1.0
        self.flag        = 0
        self.event       = "unknown"  # Event: //X‑Y label

###############################################################################
# Regexes                                                                     #
###############################################################################
re_begin  = re.compile(r"^----- Begin chunk -----")
re_end    = re.compile(r"^----- End chunk -----")
re_event  = re.compile(r"^\s*Event:\s*(.*)")

# Reflection columns: h k l  I  σ  peak  bkg  fs  ss  panel
re_ref = re.compile(
    r"^\s*([\-\d]+)\s+([\-\d]+)\s+([\-\d]+)\s+"   # h k l
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"            # I σ
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"            # peak bkg
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"            # fs ss
    r"([A-Za-z0-9]+)")                                     # panel

###############################################################################
# Parsing                                                                     #
###############################################################################

def parse_stream(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_lines = []
    crystals = []
    i = 0

    # File header
    while i < len(lines) and not re_begin.match(lines[i]):
        header_lines.append(lines[i])
        i += 1

    # Chunks
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
                cry.reflections.append(
                    Reflection(h,k,l,I,sigma,peak,bkg,fs,ss,panel,1,0))
                i += 1
            while i < len(lines) and not re_end.match(lines[i]):
                cry.footer.append(lines[i])
                i += 1
            i += 1  # skip End chunk
            crystals.append(cry)
        else:
            i += 1  # safety
    return "".join(header_lines), crystals

###############################################################################
# Scaling                                                                     #
###############################################################################

def lsq_intensities(crystals):
    sums   = defaultdict(float)
    counts = defaultdict(int)
    for cry in crystals:
        for r in cry.reflections:
            key = (r.h,r.k,r.l)
            sums[key]   += r.I
            counts[key] += 1
    return {k: sums[k]/counts[k] for k in sums}


def refine_scale_factors(crystals, cycles):
    for cycle in range(cycles):
        target = lsq_intensities(crystals)
        for cry in tqdm(crystals, desc=f"Cycle {cycle+1}/{cycles}", leave=False):
            num = den = 0.0
            for r in cry.reflections:
                It = target[(r.h,r.k,r.l)]
                num += It * r.I
                den += r.I * r.I
            cry.osf = num/den if den else 1.0
        # normalise mean OSF→1 each cycle (optional)
        mean_osf = sum(c.osf for c in crystals)/len(crystals)
        for c in crystals:
            c.osf /= mean_osf


def apply_scaling(crystals):
    for cry in tqdm(crystals, desc="Applying scaling", leave=False):
        for idx, r in enumerate(cry.reflections):
            flag = r.flag
            if r.red < min_redundancy:
                flag |= FLAG_LOW_REDUNDANCY
            scale = cry.osf
            cry.reflections[idx] = r._replace(I=r.I*scale,
                                               sigma=r.sigma*scale,
                                               flag=flag)

###############################################################################
# Writing                                                                     #
###############################################################################

def write_stream(file_header, crystals, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(file_header)
        for cry in crystals:
            fh.write("----- Begin chunk -----\n")
            fh.writelines(cry.header)
            for r in cry.reflections:
                fh.write(
                    f" {r.h:4d} {r.k:4d} {r.l:4d}"
                    f" {r.I:12.2f} {r.sigma:9.2f}"
                    f" {r.peak:10.2f} {r.bkg:10.2f}"
                    f" {r.fs:7.1f} {r.ss:7.1f} {r.panel}\n")
            fh.writelines(cry.footer)
            fh.write("----- End chunk -----\n")

###############################################################################
# CSV & plot                                                                  #
###############################################################################

def save_osf_csv(crystals, in_path):
    out_csv = os.path.join(os.path.dirname(os.path.abspath(in_path)),
                           "osf_values.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "osf"])
        for c in crystals:
            w.writerow([c.event, f"{c.osf:.6f}"])
    return out_csv


def plot_osf_vs_event(crystals, in_path):
    xs, ys = [], []
    for c in crystals:
        try:
            # assume Event like "//0-1": take the number before '-'
            first_num = int(c.event.split("//")[-1].split("-")[0])
        except (IndexError, ValueError):
            continue
        xs.append(first_num)
        ys.append(c.osf)
    if not xs:
        return None
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("First index of Event tag (before '-')")
    plt.ylabel("OSF")
    plt.title("OSF vs Event index")
    out_png = os.path.join(os.path.dirname(os.path.abspath(in_path)),
                           "osf_vs_event.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png

###############################################################################
# Main                                                                        #
###############################################################################
if __name__ == "__main__":
    file_header, crystals = parse_stream(input_file)
    refine_scale_factors(crystals, max_cycles)
    apply_scaling(crystals)
    write_stream(file_header, crystals, output_file)

    csv_path = save_osf_csv(crystals, input_file)
    png_path = plot_osf_vs_event(crystals, input_file)

    # Console report
    print("\nPer‑chunk scale factors (saved to", os.path.basename(csv_path), "):\n")
    for c in crystals:
        print(f"{c.event:15s}  OSF = {c.osf:8.4f}")

    print(f"\n[✓] Wrote scaled stream → {output_file}")
    print(f"[✓] CSV → {csv_path}")
    if png_path:
        print(f"[✓] Plot → {png_path}")
    else:
        print("[!] Could not create OSF plot – Event tags missing or malformed.")
