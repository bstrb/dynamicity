#!/usr/bin/env sh
# check_zones_from_stream.sh
# Parse ONE event from a CrystFEL .stream and list closest low-index [u v w].
#
# Usage:
#   sh check_zones_from_stream.sh --stream INPUT.stream --event 1200 \
#       [--max 3] [--top 20] [--beam "0 0 1"] [--keep-sign] [--print-vectors]
#
# Notes:
#   • Event can be "621" or "41-1" (the script matches the "Event: //..." line).
#   • Works in REAL SPACE: A = (G*^-1)^T with G* = [astar bstar cstar].
#   • Deduplicates primitive directions and treats ± as equivalent unless --keep-sign.
#   • Prints top-N closest directions (default N=20).

set -eu

STREAM="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/MFM300.stream"
EVENT_ID="11"
MAX=5
TOP=10
BEAM="0 0 1"
KEEPSIGN=0
PRINTVEC=1

usage() {
  sed -n '2,200p' "$0" | sed 's/^# //;t;d'
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stream) STREAM="$2"; shift 2;;
    --event)  EVENT_ID="$2"; shift 2;;    # e.g. 621 or 41-1
    --max)    MAX="$2"; shift 2;;
    --top)    TOP="$2"; shift 2;;
    --beam)   BEAM="$2"; shift 2;;
    --keep-sign) KEEPSIGN=1; shift 1;;
    --print-vectors) PRINTVEC=1; shift 1;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1" >&2; usage;;
  esac
done

[ -n "$STREAM" ] && [ -n "$EVENT_ID" ] || { echo "Error: --stream and --event are required." >&2; usage; }
[ -f "$STREAM" ] || { echo "Error: stream file not found: $STREAM" >&2; exit 2; }

python3 - <<'PY' "$STREAM" "$EVENT_ID" "$MAX" "$TOP" "$BEAM" "$KEEPSIGN" "$PRINTVEC"
import sys, math, re, numpy as np
path, event_arg, MAX, TOP, BEAM, KEEPSIGN, PRINTVEC = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], bool(int(sys.argv[6])), bool(int(sys.argv[7]))

BEGIN = b"----- Begin chunk"
END   = b"----- End chunk"
EVENT_RE = re.compile(r'^Event:\s*//\s*([^\s]+)')
AX_RE = {
  'astar': re.compile(r'^astar\s*=\s*([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s+([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s+([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)'),
  'bstar': re.compile(r'^bstar\s*=\s*([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s+([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s+([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)'),
  'cstar': re.compile(r'^cstar\s*=\s*([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s+([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s+([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)'),
}

# Read file and locate the chunk with Event: //EVENT_ID
target_ev = event_arg.strip()
with open(path, 'rb') as fh:
    data = fh.read()

# Iterate chunks
pos = 0
found = False
astar = bstar = cstar = None
while True:
    ib = data.find(BEGIN, pos)
    if ib < 0: break
    ie = data.find(END, ib)
    if ie < 0: break
    chunk = data[ib:ie]
    # decode safely
    lines = []
    for ln in chunk.splitlines():
        try: lines.append(ln.decode('utf-8'))
        except UnicodeDecodeError: lines.append(ln.decode('latin-1', errors='replace'))
    # Event?
    ev_here = None
    for ln in lines:
        m = EVENT_RE.match(ln)
        if m:
            ev_here = m.group(1).strip()
            break
    if ev_here is not None:
        # match full (supports 621 or 41-1)
        if ev_here == target_ev or ev_here.endswith(target_ev):
            # parse astar/bstar/cstar inside this chunk
            for ln in lines:
                for key, rx in AX_RE.items():
                    mm = rx.match(ln)
                    if mm:
                        vec = np.array([float(mm.group(1)), float(mm.group(2)), float(mm.group(3))], float)
                        if key=='astar': astar = vec
                        elif key=='bstar': bstar = vec
                        elif key=='cstar': cstar = vec
            found = True
            break
    pos = ie + len(END)

if not found:
    raise SystemExit(f"Event '//{target_ev}' not found in {path}")
if astar is None or bstar is None or cstar is None:
    raise SystemExit(f"Event '//{target_ev}' missing astar/bstar/cstar")

def parse_vec3(s):
    t = [float(x) for x in s.replace(",", " ").split()]
    if len(t)!=3: raise SystemExit("Beam must be 3 numbers")
    return np.array(t, float)

beam = parse_vec3(BEAM)

Gstar = np.column_stack((astar, bstar, cstar))
A = np.linalg.inv(Gstar).T   # real-space basis columns (a,b,c)

if PRINTVEC:
    print(f"# Event //{target_ev}")
    print(f"# astar = {astar.tolist()}")
    print(f"# bstar = {bstar.tolist()}")
    print(f"# cstar = {cstar.tolist()}")
    print(f"# beam  = {beam.tolist()} (lab)")

def primitive(u,v,w, keep_sign=False):
    if u==v==w==0: return None
    g = math.gcd(abs(u), math.gcd(abs(v), abs(w)))
    u//=g; v//=g; w//=g
    if not keep_sign:
        if   u!=0 and u<0: u,v,w = -u,-v,-w
        elif u==0 and v!=0 and v<0: u,v,w = -u,-v,-w
        elif u==0 and v==0 and w<0: u,v,w = -u,-v,-w
    return (u,v,w)

def angle_to_beam(u,v,w):
    r = A @ np.array([u,v,w], float)
    r /= np.linalg.norm(r)
    b = beam/np.linalg.norm(beam)
    cosang = float(np.clip(abs(np.dot(r,b)), -1.0, 1.0))  # ±equivalence
    return float(np.degrees(np.arccos(cosang)))

seen=set(); results=[]
for u in range(-MAX, MAX+1):
    for v in range(-MAX, MAX+1):
        for w in range(-MAX, MAX+1):
            p = primitive(u,v,w, keep_sign=KEEPSIGN)
            if p is None: continue
            if p in seen: continue
            seen.add(p)
            ang = angle_to_beam(*p)
            results.append((ang, p))

results.sort(key=lambda t: t[0])
print(f"# {path} :: //{target_ev} | scanned |u|,|v|,|w| <= {MAX} ; unique={len(results)} ; beam={beam.tolist()}")
for ang,(u,v,w) in results[:TOP]:
    print(f"[{u:2d} {v:2d} {w:2d}]  angle = {ang:6.2f}°")
PY
