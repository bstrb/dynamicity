#!/usr/bin/env python3
"""
scale_stream.py – chunk‑wise intensity scaling only
==================================================
Refines **one overall scale factor (OSF) per crystal chunk** in a CrystFEL
`.stream` file (no B‑factor, no partiality) and rewrites the file with every
reflection intensity and σ(I) multiplied by that OSF.  A summary of the scale
factors, labelled by the chunk’s *Event* tag (e.g. `Event: //0‑1`), is printed
at the end.

* **Hard‑wired paths and parameters**: edit the *Configuration* block below.
* The file‑level header (everything before the first `----- Begin chunk -----`)
  is preserved verbatim.
* All reflection columns – peak, background, fs/px, ss/px, panel – remain
  unchanged apart from the scaled *I* and *σ(I)*.
* Reflections whose redundancy (placeholder column) is below
  `min_redundancy` are flagged internally (flag not written).

Scaling equation
----------------
```
I_scaled   = OSF · I_raw
σ(I)_scaled = OSF · σ(I)_raw
```
```
OSF = Σ(I_target · I_raw) / Σ(I_raw²)
```
where *I_target* is the across‑chunks mean intensity for each (h,k,l).
"""
import re
import math
from collections import defaultdict, namedtuple

###############################################################################
# Configuration – adjust to your needs                                        #
###############################################################################
input_file      = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream'       # replace with your input path
output_file     = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/scaled.stream'    # replace with desired output path
min_redundancy = 2                 # flag reflections with redundancy < N
max_cycles     = 5                 # LSQ iterations (converges quickly)

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
        self.osf         = 1.0  # overall scale factor
        self.flag        = 0    # for potential outlier rejection
        self.event       = "unknown"  # Event: //X‑X label

###############################################################################
# Regexes                                                                     #
###############################################################################
re_begin  = re.compile(r"^----- Begin chunk -----")
re_end    = re.compile(r"^----- End chunk -----")
re_event  = re.compile(r"^\s*Event:\s*(.*)")

# Reflection: h k l  I  sigma  peak  bkg  fs  ss  panel
re_ref = re.compile(
    r"^\s*([\-\d]+)\s+([\-\d]+)\s+([\-\d]+)\s+"   # h k l
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"            # I sigma
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"            # peak bkg
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"            # fs ss
    r"([A-Za-z0-9]+)"                                     # panel
)

###############################################################################
# Parsing                                                                     #
###############################################################################

def parse_stream(path):
    """Return (file_header:str, list[Crystal])."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    file_header = []
    crystals = []
    i = 0

    # File header until the first chunk
    while i < len(lines) and not re_begin.match(lines[i]):
        file_header.append(lines[i])
        i += 1

    # Loop over chunks
    while i < len(lines):
        if re_begin.match(lines[i]):
            cry = Crystal()
            i += 1
            # Header within chunk
            while i < len(lines) and not re_ref.match(lines[i]):
                line = lines[i]
                cry.header.append(line)
                m_ev = re_event.match(line)
                if m_ev:
                    cry.event = m_ev.group(1).strip()
                i += 1
            # Reflection lines
            while i < len(lines) and re_ref.match(lines[i]):
                m = re_ref.match(lines[i])
                h,k,l        = map(int,   m.group(1,2,3))
                I,sigma      = map(float, m.group(4,5))
                peak,bkg     = map(float, m.group(6,7))
                fs,ss        = map(float, m.group(8,9))
                panel        = m.group(10)
                # redundancy placeholder (set 1) – adjust if real value present
                cry.reflections.append(
                    Reflection(h,k,l,I,sigma,peak,bkg,fs,ss,panel,1,0))
                i += 1
            # Footer
            while i < len(lines) and not re_end.match(lines[i]):
                cry.footer.append(lines[i])
                i += 1
            i += 1  # skip "----- End chunk -----"
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
            hkl = (r.h, r.k, r.l)
            sums[hkl]   += r.I
            counts[hkl] += 1
    return {hkl: sums[hkl] / counts[hkl] for hkl in sums}


def refine_scale_factors(crystals, cycles):
    """Iteratively refine OSF for each crystal until convergence (few cycles)."""
    for _ in range(cycles):
        target = lsq_intensities(crystals)
        for cry in crystals:
            num = den = 0.0
            for r in cry.reflections:
                hkl = (r.h, r.k, r.l)
                It  = target[hkl]
                num += It * r.I
                den += r.I * r.I
            if den > 0:
                cry.osf = num / den
        # normalise ⟨OSF⟩ = 1 (optional but helpful)
        mean_osf = sum(c.osf for c in crystals) / len(crystals)
        for c in crystals:
            c.osf /= mean_osf


def apply_scaling(crystals):
    for cry in crystals:
        for idx, r in enumerate(cry.reflections):
            flag = r.flag
            if r.red < min_redundancy:
                flag |= FLAG_LOW_REDUNDANCY
            scale = cry.osf
            cry.reflections[idx] = r._replace(I=r.I * scale,
                                               sigma=r.sigma * scale,
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
# Main                                                                        #
###############################################################################
if __name__ == "__main__":
    header, crystals = parse_stream(input_file)
    refine_scale_factors(crystals, max_cycles)
    apply_scaling(crystals)
    write_stream(header, crystals, output_file)

    # Print per‑chunk scale summary
    print("\nPer‑chunk scale factors:\n-----------------------")
    for cry in crystals:
        print(f"{cry.event:15s}  OSF = {cry.osf:8.4f}")

    print(f"\n[✓] Wrote scaled stream → {output_file}")
