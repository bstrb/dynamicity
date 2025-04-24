#!/usr/bin/env python3
"""
scale_stream.py – chunk-wise scaling plus dynamical-scattering diagnostics
=========================================================================
**What it does**
* Refines **one overall scale factor (OSF)** per chunk (no B-factor, no partiality).
* Preserves the file-level header as-is; writes a fully scaled *.stream* file.
* Computes three per-chunk quality metrics:
  1. **Residual** – RMS mis-fit after scaling (as before).
  2. **R_dyn** – Friedel-pair disagreement (quick proxy for dynamical scattering).
  3. **CC_frame** – Pearson correlation of the frame’s scaled intensities with the across-frame mean.
* Progress bars via **tqdm**.
* Outputs `frame_stats.csv` with **event, osf, residual, R_dyn, CC_frame**.
* Generates three scatter plots vs. the first index of the `Event //X-Y` tag:
  * `osf_vs_event.png`
  * `residual_vs_event.png`
  * `Rdyn_vs_event.png`

Install extras once with:
```bash
pip install tqdm matplotlib
```
"""
import re, os, csv, math
from collections import defaultdict, namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt

###############################################################################
# Configuration                                                               #
###############################################################################
input_file      = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream'       # replace with your input path
output_file     = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/scaled.stream'    # replace with desired output path
min_redundancy = 2                # flag reflections with redundancy < N
max_cycles     = 5                # LSQ iterations

###############################################################################
# Data containers                                                             #
###############################################################################
Reflection = namedtuple(
    "Reflection",
    "h k l I sigma peak bkg fs ss panel red flag")
FLAG_LOW_REDUNDANCY = 0x01

class Crystal:
    __slots__ = (
        "header", "reflections", "footer",
        "event", "osf", "resid", "rdyn", "cc")
    def __init__(self):
        self.header = []
        self.reflections = []
        self.footer = []
        self.event = "unknown"
        # refined / diagnostic values
        self.osf = 1.0
        self.resid = 0.0
        self.rdyn = 0.0
        self.cc = 0.0

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

    header_lines, crystals = [], []
    i = 0
    while i < len(lines) and not re_begin.match(lines[i]):
        header_lines.append(lines[i]); i += 1

    while i < len(lines):
        if re_begin.match(lines[i]):
            cry = Crystal(); i += 1
            while i < len(lines) and not re_ref.match(lines[i]):
                cry.header.append(lines[i])
                m = re_event.match(lines[i])
                if m: cry.event = m.group(1).strip()
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
                cry.footer.append(lines[i]); i += 1
            i += 1  # skip End chunk
            crystals.append(cry)
        else:
            i += 1
    return "".join(header_lines), crystals

###############################################################################
# Scaling / statistics                                                        #
###############################################################################

def mean_I_map(crystals):
    s, c = defaultdict(float), defaultdict(int)
    for cry in crystals:
        for r in cry.reflections:
            key = (r.h,r.k,r.l)
            s[key] += r.I; c[key] += 1
    return {k: s[k]/c[k] for k in s}


def refine_osf(crystals, cycles):
    for cyc in range(cycles):
        target = mean_I_map(crystals)
        for cry in tqdm(crystals, desc=f"Cycle {cyc+1}/{cycles}", leave=False):
            num = den = 0.0
            for r in cry.reflections:
                It = target[(r.h,r.k,r.l)]
                num += It * r.I; den += r.I*r.I
            cry.osf = num/den if den else 1.0
        mean_osf = sum(c.osf for c in crystals)/len(crystals)
        for c in crystals: c.osf /= mean_osf
    return target


def stats_after_scaling(crystals, target_map):
    for cry in tqdm(crystals, desc="Applying scaling & computing stats", leave=False):
        # containers for metrics
        diffsq_sum = tar_sum = 0.0; n = 0
        Iscaled, Itarget = [], []
        friedel_pos, friedel_neg = defaultdict(list), defaultdict(list)

        for idx, r in enumerate(cry.reflections):
            I_sc = r.I * cry.osf; sig_sc = r.sigma * cry.osf
            cry.reflections[idx] = r._replace(I=I_sc, sigma=sig_sc,
                                               flag=r.flag | (FLAG_LOW_REDUNDANCY if r.red<min_redundancy else 0))
            It = target_map[(r.h,r.k,r.l)]
            diffsq_sum += (I_sc - It)**2; tar_sum += It; n += 1
            Iscaled.append(I_sc); Itarget.append(It)
            # collect for Friedel pairs
            friedel_pos[(r.h,r.k,r.l)].append(I_sc)
            friedel_neg[(-r.h,-r.k,-r.l)].append(I_sc)

        # residual
        cry.resid = math.sqrt(diffsq_sum/n)/(tar_sum/n) if n else 0.0
        # CC_frame
        if n:
            mean_s = sum(Iscaled)/n; mean_t = sum(Itarget)/n
            cov = sum((a-mean_s)*(b-mean_t) for a,b in zip(Iscaled,Itarget))
            var_s = sum((a-mean_s)**2 for a in Iscaled)
            var_t = sum((b-mean_t)**2 for b in Itarget)
            cry.cc = cov/math.sqrt(var_s*var_t) if var_s and var_t else 0.0
        # R_dyn (Friedel disagreement)
        num = den = 0.0
        for hkl in friedel_pos:
            mates_p = friedel_pos[hkl]
            mates_n = friedel_neg.get(hkl, [])
            for a,b in zip(mates_p, mates_n):
                num += abs(a-b)
                den += 0.5*(a+b)
        cry.rdyn = num/den if den else 0.0

###############################################################################
# Write helpers                                                               #
###############################################################################

def write_stream(header, crystals, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for cry in crystals:
            fh.write("----- Begin chunk -----\n")
            fh.writelines(cry.header)
            for r in cry.reflections:
                fh.write(f" {r.h:4d} {r.k:4d} {r.l:4d} {r.I:12.2f} {r.sigma:9.2f}"
                         f" {r.peak:10.2f} {r.bkg:10.2f} {r.fs:7.1f} {r.ss:7.1f} {r.panel}\n")
            fh.writelines(cry.footer)
            fh.write("----- End chunk -----\n")


def save_csv(crystals, in_path):
    out_csv = os.path.join(os.path.dirname(os.path.abspath(in_path)), "frame_stats.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["event", "osf", "residual", "R_dyn", "CC_frame"])
        for c in crystals:
            w.writerow([c.event, f"{c.osf:.6f}", f"{c.resid:.6f}", f"{c.rdyn:.6f}", f"{c.cc:.4f}"])
    return out_csv


def scatter(crystals, attr, fname, ylabel, in_path):
    xs, ys = [], []
    for c in crystals:
        try:
            idx = int(c.event.split("//")[-1].split("-")[0])
        except (IndexError, ValueError):
            continue
        xs.append(idx); ys.append(getattr(c, attr))
    if not xs: return None
    plt.figure(); plt.scatter(xs, ys)
    plt.xlabel("First index of Event tag (before '-')"); plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Event index");
    out_png = os.path.join(os.path.dirname(os.path.abspath(in_path)), fname)
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close();
    return out_png

###############################################################################
# Main                                                                        #
###############################################################################
if __name__ == "__main__":
    header, crystals = parse_stream(input_file)
    target_map = refine_osf(crystals, max_cycles)
    stats_after_scaling(crystals, target_map)
    write_stream(header, crystals, output_file)

    csv_path = save_csv(crystals, input_file)
    png_osf   = scatter(crystals, "osf",   "osf_vs_event.png",      "OSF",      input_file)
    png_resid = scatter(crystals, "resid", "residual_vs_event.png", "Residual", input_file)
    png_rdyn  = scatter(crystals, "rdyn",  "Rdyn_vs_event.png",     "R_dyn",    input_file)

    # Console table
    # print("\nFrame statistics (saved to frame_stats.csv):\n")
    # print(f"{'Event':15s} {'OSF':>8s} {'Residual':>9s} {'R_dyn':>8s} {'CC':>6s}")
    # for c in crystals:
    #     print(f"{c.event:15s} {c.osf:8.4f} {c.resid:9.4f} {c.rdyn:8.4f} {c.cc:6.2f}")

    print(f"\n[✓] Scaled stream  → {output_file}")
    print(f"[✓] Stats CSV     → {csv_path}")
    if png_osf:   print(f"[✓] Plot          → {png_osf}")
    if png_resid: print(f"[✓] Plot          → {png_resid}")
    if png_rdyn:  print(f"[✓] Plot          → {png_rdyn}")
