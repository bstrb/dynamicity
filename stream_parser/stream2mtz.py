import os
import reciprocalspaceship as rs

stream_path = "/Users/xiaodong/Desktop/UOX-data/UOX1/centers_xatol_0.01_frameinterval_10_lowess_0.37_shifted_0.5_-0.3/xgandalf_iterations_max_radius_0.0_step_0.5/UOX.stream"

# ── 0. Quick file‐existence debug ─────────────────────────────
if not os.path.exists(stream_path):
    raise FileNotFoundError(f"Stream file not found: {stream_path!r}")
print(f"✔ Found stream file: {stream_path!r}, size = {os.path.getsize(stream_path)} bytes")

# ── 1. Try reading without parallelism (better tracebacks!) ────
try:
    ds = rs.read_crystfel(
    stream_path,
    spacegroup="I222",
    parallel=False,
    columns=["H","K","L","I","SigI","BATCH"]
    )

except StopIteration:
    raise RuntimeError(
        "⚠️  `read_crystfel` saw zero “chunks” in your stream file.  \n"
        "   • Verify that the file really contains “----- Begin chunk -----” blocks.  \n"
        "   • Check you pointed at the correct path (absolute vs. relative).  \n"
        "   • Try opening the file in a text editor to confirm its contents."
    )
print(f"✔ Parsed {len(ds)} reflections from the stream.")

# ── 2. Sanity checks (un-commented and printed) ───────────────
print("\nIntensity statistics:")
print(ds["I"].describe(), end="\n\n")     # mean, std, min, max…

# Will raise if any σ(I) ≤ 0
assert (ds["SigI"] > 0).all(), "Error: Some sigma(I) values are non-positive!"

# Check batch IDs
n_batches = ds["BATCH"].nunique()
print(f"Found {n_batches} unique BATCH IDs (should match # images).", end="\n\n")

# ── 3. Write out MTZ ─────────────────────────────────────────
out_mtz = os.path.join(
    os.path.dirname(stream_path),
    os.path.basename(stream_path).replace(".stream", ".mtz"))
ds.write_mtz(out_mtz)
print(f"Wrote MTZ file: {out_mtz!r}")
