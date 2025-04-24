#!/usr/bin/env python3
"""
scale_stream.py – chunk‑wise intensity scaling only
==================================================
Hard‑wired script that takes a CrystFEL *.stream* file, refines **one global
scale factor per chunk** (no B‑factor, no partiality) and writes a new stream
with every reflection intensity and σ(I) multiplied by that factor.  All other
columns – peak height, background, detector coordinates, panel etc. – are left
untouched.

* **Input/output paths and parameters are defined directly in the code** at the
  bottom (no *argparse*).
* The global file header that precedes the first “Begin chunk” block is copied
  verbatim to the output.
* Reflections can be flagged when their redundancy is below a threshold (set
  via the `min_redundancy` constant).

Equation applied
----------------
```
I_scaled = OSF · I_raw
σ_scaled = OSF · σ_raw
```
where *OSF* is obtained by a simple least‑squares fit between each chunk and
an across‑chunks mean, solved in closed form.
"""
import re
from collections import defaultdict, namedtuple

###############################################################################
# Configuration (edit these paths / numbers to taste)                         #
###############################################################################
input_file      = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream'       # replace with your input path
output_file     = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/scaled.stream'    # replace with desired output path
min_redundancy = 2                # reflections with redundancy < N will be flagged
max_cycles     = 5                # refinement iterations (OSF converges fast)

###############################################################################
# Data containers                                                             #
###############################################################################
Reflection = namedtuple("Reflection",
                        "h k l I sigma peak bkg fs ss panel red flag")
FLAG_LOW_REDUNDANCY = 0x01

class Crystal:
    __slots__ = ("header", "reflections", "footer", "osf", "flag")
    def __init__(self):
        self.header      = []
        self.reflections = []   # list[Reflection]
        self.footer      = []
        self.osf         = 1.0  # overall scale factor
        self.flag        = 0    # for potential outlier rejection

###############################################################################
# Regexes for parsing                                                         #
###############################################################################
re_begin  = re.compile(r"^----- Begin chunk -----")
re_end    = re.compile(r"^----- End chunk -----")
re_header = re.compile(r"^----- Begin")          # start of any section inside chunk

# Reflection line – columns: h k l  I  sigma  peak  bkg  fs  ss  panel
re_ref = re.compile(
    r"^\s*([\-\d]+)\s+([\-\d]+)\s+([\-\d]+)\s+"      # h k l
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"               # I sigma
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"               # peak bkg
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"               # fs ss
    r"([A-Za-z0-9]+)"                                        # panel
)

###############################################################################
# Parsing                                                                     #
###############################################################################

def parse_stream(path):
    """Return (file_header:str, list[Crystal])."""
    with open(path, "r") as fh:
        lines = fh.readlines()

    file_header = []
    crystals = []
    i = 0

    # collect file header until first Begin chunk
    while i < len(lines) and not re_begin.match(lines[i]):
        file_header.append(lines[i])
        i += 1

    while i < len(lines):
        if re_begin.match(lines[i]):
            cry = Crystal()
            i += 1
            # everything up to first reflection line = header for this crystal
            while i < len(lines) and not re_ref.match(lines[i]):
                cry.header.append(lines[i])
                i += 1
            # reflections
            while i < len(lines) and re_ref.match(lines[i]):
                m = re_ref.match(lines[i])
                h,k,l        = map(int,   m.group(1,2,3))
                I,sigma      = map(float, m.group(4,5))
                peak,bkg     = map(float, m.group(6,7))
                fs,ss        = map(float, m.group(8,9))
                panel        = m.group(10)
                # redundancy not in line; set 1 as placeholder (adjust if known)
                cry.reflections.append(
                    Reflection(h,k,l,I,sigma,peak,bkg,fs,ss,panel,1,0))
                i += 1
            # footer until end-of-chunk marker
            while i < len(lines) and not re_end.match(lines[i]):
                cry.footer.append(lines[i])
                i += 1
            # swallow the "End chunk" line
            i += 1
            crystals.append(cry)
        else:
            i += 1  # safety
    return "".join(file_header), crystals

###############################################################################
# Scaling                                                                     #
###############################################################################

def lsq_intensities(crystals):
    """Return mean intensity per hkl across all crystals."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for cry in crystals:
        for r in cry.reflections:
            hkl = (r.h,r.k,r.l)
            sums[hkl]   += r.I
            counts[hkl] += 1
    return {hkl: sums[hkl]/counts[hkl] for hkl in sums}


def refine_scale_factors(crystals, max_cycles):
    """Iteratively update cry.osf so that scaled intensities agree across chunks."""
    for _ in range(max_cycles):
        target = lsq_intensities(crystals)
        # closed‑form LSQ for scale: minimise Σ( (osf·I - It)^2 )
        for cry in crystals:
            num = den = 0.0
            for r in cry.reflections:
                hkl = (r.h,r.k,r.l)
                if hkl not in target:  # unlikely
                    continue
                It = target[hkl]
                num += It * r.I
                den += r.I * r.I
            if den > 0:
                cry.osf = num / den
        # normalise so ⟨osf⟩ = 1 (optional, cosmetic)
        mean_osf = sum(c.osf for c in crystals) / len(crystals)
        for c in crystals:
            c.osf /= mean_osf


def apply_scaling(crystals):
    for cry in crystals:
        for i, r in enumerate(cry.reflections):
            flag = r.flag
            if r.red < min_redundancy:
                flag |= FLAG_LOW_REDUNDANCY
            scale = cry.osf
            cry.reflections[i] = r._replace(I=r.I*scale,
                                             sigma=r.sigma*scale,
                                             flag=flag)

###############################################################################
# Writing                                                                     #
###############################################################################

def write_stream(header, crystals, path):
    with open(path, "w") as fh:
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

###############################################################################
# Main                                                                        #
###############################################################################
if __name__ == "__main__":
    file_header, crystals = parse_stream(input_file)
    refine_scale_factors(crystals, max_cycles)
    apply_scaling(crystals)
    write_stream(file_header, crystals, output_file)
    print(f"[✓] Wrote scaled stream → {output_file}")
