#!/usr/bin/env python3
"""
scale_stream.py
================
Scale raw reflection **intensities only** (no partiality correction) on a
chunk-to-chunk basis in a CrystFEL-style `.stream` file.  For every crystal
chunk the script refines a global overall-scale factor *OSF* and temperature
factor *B*.  It then applies the transformation

    I_scaled = OSF · exp(-2·B·s²) · I_raw

(and the same multiplier to σ(I)).  Reflections with redundancy below a user
threshold can be flagged for later exclusion.

This is a completely self-contained demo implementation; replace the two
marked stubs with your own least-squares engine for production use.
"""
import re
import sys
import math
from collections import namedtuple

###############################################################################
# Data containers                                                              
###############################################################################
Reflection = namedtuple('Reflection', 'h k l I sigma s2 redundancy user_flag')
FLAG_LOW_REDUNDANCY = 1

def repl(ref, **kw):
    """Return a modified copy of *ref* (namedtuple helper)."""
    return ref._replace(**kw)

class Crystal:
    """Minimal crystal container."""
    __slots__ = ('header', 'reflections', 'footer', 'osf', 'B', 'user_flag')
    def __init__(self):
        self.header       = []   # lines before reflections
        self.reflections  = []   # list[Reflection]
        self.footer       = []   # lines after reflections
        self.osf          = 1.0  # overall scale factor
        self.B            = 0.0  # temperature factor
        self.user_flag    = 0    # for outlier rejection etc.

###############################################################################
# Stream parsing / writing                                                     
###############################################################################
re_begin = re.compile(r'^----- Begin chunk -----')
re_end   = re.compile(r'^----- End chunk -----')
# Reflection line regex – adapt if your column order differs
re_ref   = re.compile(r"^\s*([\-\d]+)\s+([\-\d]+)\s+([\-\d]+)\s+"
                      r"([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+"
                      r"[\d\.\-eE]+\s+[\d\.\-eE]+\s+"  # peak & background
                      r"([\d\.\-eE]+)\s+([\d]+)")        # s2  redundancy


def parse_stream(path):
    """Return list[Crystal] parsed from *path*."""
    crystals = []
    with open(path, 'r') as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        if re_begin.match(lines[i]):
            i += 1
            cry = Crystal()
            # Header section
            while i < len(lines) and not re_ref.match(lines[i]):
                cry.header.append(lines[i])
                i += 1
            # Reflection lines
            while i < len(lines) and re_ref.match(lines[i]):
                m = re_ref.match(lines[i])
                h, k, l = map(int, m.group(1,2,3))
                I, sig  = map(float, m.group(4,5))
                s2      = float(m.group(6))
                red     = int(m.group(7))
                cry.reflections.append(Reflection(h,k,l,I,sig,s2,red,0))
                i += 1
            # Footer until end marker
            while i < len(lines) and not re_end.match(lines[i]):
                cry.footer.append(lines[i])
                i += 1
            i += 1  # skip "End chunk"
            crystals.append(cry)
        else:
            i += 1
    return crystals


def write_stream(crystals, out_path):
    """Write scaled crystals out to a new `.stream` file."""
    with open(out_path, 'w') as fh:
        for cry in crystals:
            fh.write('----- Begin chunk -----\n')
            fh.writelines(cry.header)
            for r in cry.reflections:
                fh.write( f" {r.h:4d} {r.k:4d} {r.l:4d}"
                          f" {r.I:12.2f} {r.sigma:9.2f}   "
                          f"{r.s2:6.3f} {r.redundancy:3d}\n" )
            fh.writelines(cry.footer)
            fh.write('----- End chunk -----\n')

###############################################################################
# Very simple LSQ scaling (demo)                                               
###############################################################################

def lsq_intensities(crystals):
    """Return mean intensity per hkl across crystals (stub implementation)."""
    from collections import defaultdict
    sum_I   = defaultdict(float)
    count_I = defaultdict(int)
    for cry in crystals:
        for r in cry.reflections:
            key = (r.h, r.k, r.l)
            sum_I[key]   += r.I
            count_I[key] += 1
    return {key: sum_I[key]/count_I[key] for key in sum_I}


def iterate_scale(crystals, I_map):
    """Single least-squares update of OSF and B for every crystal."""
    for cry in crystals:
        S00 = S01 = S11 = 0.0
        b0 = b1 = 0.0
        for r in cry.reflections:
            key = (r.h, r.k, r.l)
            if key not in I_map: continue
            y = math.log(max(r.I,1e-9)) - math.log(max(I_map[key],1e-9))
            x = r.s2
            S00 += 1.0; S01 += x; S11 += x*x
            b0  += y;  b1  += y*x
        det = S00*S11 - S01*S01
        if abs(det)<1e-12: continue
        ln_osf = (b0*S11 - b1*S01)/det
        neg2B  = (S00*b1 - S01*b0)/det
        cry.osf = math.exp(ln_osf)
        cry.B   = -0.5 * neg2B

###############################################################################
# Scaling loop & application                                                   
###############################################################################

def scale_and_apply(crystals, max_cycles: int, min_redundancy: int):
    for cycle in range(max_cycles):
        I_map = lsq_intensities(crystals)
        iterate_scale(crystals, I_map)
        # normalize so ⟨OSF⟩=1, ⟨B⟩=0
        good = [c for c in crystals if c.user_flag==0]
        mean_osf = sum(c.osf for c in good)/len(good)
        mean_B   = sum(c.B   for c in good)/len(good)
        for c in crystals:
            c.osf /= mean_osf
            c.B   -= mean_B
    # apply to reflections
    for cry in crystals:
        for idx, r in enumerate(cry.reflections):
            flag = r.user_flag
            if r.redundancy < min_redundancy:
                flag |= FLAG_LOW_REDUNDANCY
            scale = cry.osf * math.exp(-2.0*cry.B*r.s2)
            cry.reflections[idx] = repl(r, I=r.I*scale, sigma=r.sigma*scale, user_flag=flag)

###############################################################################
# Main (hardcoded inputs)                                                      
###############################################################################
if __name__ == '__main__':
    # Hardcoded parameters:
    input_file      = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream'       # replace with your input path
    output_file     = '/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/scaled.stream'    # replace with desired output path
    min_redundancy  = 2                  # flag reflections < this redundancy
    max_cycles      = 15                 # number of refinement iterations

    crystals = parse_stream(input_file)
    scale_and_apply(crystals, max_cycles, min_redundancy)
    write_stream(crystals, output_file)
    print(f"[✓] Wrote scaled file → {output_file}")
