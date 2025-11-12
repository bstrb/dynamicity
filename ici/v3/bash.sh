# 1) set this to your run_000 path
RUN_DIR="/home/bubl3932/files/MFM300_VIII/MP15_3x100/runs_20251112_172919/run_000"
cd "$RUN_DIR" || exit 1

echo "[scan] RUN_DIR=$RUN_DIR"

# expected = event dirs that have a .lst (i.e., scheduled)
mapfile -t EXPECTED < <(find . -maxdepth 1 -type d -name 'event_*' \
  -exec bash -lc 'shopt -s nullglob; for d; do ls "$d"/*.lst >/dev/null 2>&1 && echo "$d"; done' _ {} + | sort)

# actual (file exists): first stream_*.stream in each event dir (if any)
mapfile -t STREAMS < <(for d in "${EXPECTED[@]}"; do
  f=$(ls "$d"/stream_*.stream 2>/dev/null | sort | head -n1)
  [[ -n "$f" ]] && echo "$f"
done)

# invalid = streams that don't contain a 'Begin chunk' delimiter
mapfile -t INVALID < <(grep -L -E '^\-+\s*Begin chunk\s*\-+$' "${STREAMS[@]}" 2>/dev/null || true)

echo "[scan] expected (with .lst) : ${#EXPECTED[@]}"
echo "[scan] found any stream file: ${#STREAMS[@]}"
echo "[scan] invalid (no Begin chunk): ${#INVALID[@]}"

# list missing (scheduled but no stream file)
comm -23 <(printf "%s\n" "${EXPECTED[@]}" | sort) \
         <(printf "%s\n" "${STREAMS[@]%/*}" | sort) | sed 's#^./##' > missing_event_parts.txt

# list invalid with event id
: > invalid_event_parts.txt
for p in "${INVALID[@]}"; do
  evdir=$(dirname "$p")
  echo "${evdir#./} -> $(basename "$p") (no Begin chunk)" >> invalid_event_parts.txt
done

echo "[scan] wrote missing_event_parts.txt and invalid_event_parts.txt"
echo "[scan] If your concat wrote 299 parts, INVALID should have size 1 (or MISSING has 1)."
