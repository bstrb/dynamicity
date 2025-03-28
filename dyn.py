# Parse CrystFEL .stream file with multiple 'chunks' and extract reflection and crystal data
import re
import pandas as pd

def parse_crystfel_stream(stream_file, debug=False):
    """
    Parse a CrystFEL .stream file with multiple 'chunks'.
    For each chunk, we collect:
      - Event name (from 'Event: //...')
      - Crystal metadata (cell parameters, orientation matrix, etc.)
      - Reflection lines: h, k, l, I, sigma(I), peak, background, fs_px, ss_px, panel

    Returns:
      df_refl:  [event, h, k, l, I, sigma_I, peak, background, fs_px, ss_px, panel]
      df_cryst: [event, a, b, c, alpha, beta, gamma, astar, bstar, cstar, ...]
    """
    vec_pattern = re.compile(
        r"^(astar|bstar|cstar)\s*=\s*([\-\+\d\.Ee]+)\s+([\-\+\d\.Ee]+)\s+([\-\+\d\.Ee]+)"
    )

    chunk_id = 0
    in_chunk = False
    in_crystal = False
    in_refl_block = False

    cryst_dict = {}  # event -> {metadata...}
    refl_rows = []

    current_event = None
    crystal_lines = []

    def parse_crystal_metadata(lines, debug=False):
        """
        Parse lines from the crystal block to extract:
          - cell parameters (a, b, c, alpha, beta, gamma)
          - orientation vectors (astar, bstar, cstar)
        Return a dict of metadata.
        """
        metadata = {}

        def try_parse_cell_params(line):
            # e.g. "Cell parameters 8.72347 8.74092 8.68757 nm, 89.75536 90.25634 89.95466 deg"
            part = line.replace("Cell parameters", "").strip()
            part = part.replace("nm,", "").replace("deg", "")
            parts = part.split()
            if len(parts) >= 6:
                a, b, c = map(float, parts[:3])
                alpha, beta, gamma = map(float, parts[3:6])
                metadata["a"] = a
                metadata["b"] = b
                metadata["c"] = c
                metadata["alpha"] = alpha
                metadata["beta"] = beta
                metadata["gamma"] = gamma

        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("Cell parameters"):
                try_parse_cell_params(stripped)
            else:
                m = vec_pattern.search(stripped)
                if m:
                    vec_name = m.group(1)  # astar, bstar, or cstar
                    vx = float(m.group(2))
                    vy = float(m.group(3))
                    vz = float(m.group(4))
                    metadata[vec_name] = [vx, vy, vz]

        return metadata

    with open(stream_file, "r") as f:
        for line in f:
            stripped = line.strip()

            # Detect chunk boundaries
            if stripped.startswith("----- Begin chunk -----"):
                chunk_id += 1
                in_chunk = True
                in_crystal = False
                in_refl_block = False
                current_event = None
                crystal_lines = []
                continue

            if stripped.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_refl_block = False
                current_event = None
                crystal_lines = []
                continue

            # If inside chunk, look for event line
            if in_chunk and stripped.startswith("Event: //"):
                current_event = stripped.split("Event: //", 1)[1].strip()
                continue

            # Detect crystal block
            if in_chunk and stripped.startswith("--- Begin crystal"):
                in_crystal = True
                crystal_lines = []
                continue

            if in_chunk and in_crystal:
                if stripped.startswith("Reflections measured after indexing"):
                    # parse the lines we collected
                    metadata = parse_crystal_metadata(crystal_lines, debug=debug)

                    if not current_event:
                        current_event = f"chunk_{chunk_id}"

                    if current_event not in cryst_dict:
                        cryst_dict[current_event] = {"event": current_event}
                    cryst_dict[current_event].update(metadata)

                    in_crystal = False
                    in_refl_block = True
                    continue
                else:
                    crystal_lines.append(stripped)
                    continue

            # If we're in reflection block
            if in_chunk and in_refl_block:
                if stripped.startswith("End of reflections"):
                    in_refl_block = False
                    continue

                parts = stripped.split()
                # Reflection lines typically have >=10 fields
                if len(parts) < 10:
                    continue
                try:
                    h = int(parts[0])
                    k = int(parts[1])
                    l = int(parts[2])
                    I_val = float(parts[3])
                    sigma_I = float(parts[4])
                    peak = float(parts[5])
                    background = float(parts[6])
                    fs_px = float(parts[7])
                    ss_px = float(parts[8])
                    panel = parts[9]

                    if not current_event:
                        current_event = f"chunk_{chunk_id}"

                    refl_rows.append(
                        {
                            "event": current_event,
                            "h": h,
                            "k": k,
                            "l": l,
                            "I": I_val,
                            "sigma_I": sigma_I,
                            "peak": peak,
                            "background": background,
                            "fs_px": fs_px,
                            "ss_px": ss_px,
                            "panel": panel,
                        }
                    )
                except ValueError:
                    # It's likely a header line or something unparseable
                    continue

    df_refl = pd.DataFrame(refl_rows)
    if "event" not in df_refl.columns:
        df_refl["event"] = []

    df_cryst = pd.DataFrame.from_dict(cryst_dict, orient="index")
    df_cryst.reset_index(drop=True, inplace=True)

    # Ensure 'event' is first column if present
    if "event" in df_cryst.columns:
        cols = list(df_cryst.columns)
        cols.remove("event")
        df_cryst = df_cryst[["event"] + cols]

    return df_refl, df_cryst

# Example usage:
input_file = "/home/bubl3932/files/MIL101_ICF/xgandalf_iterations_max_radius_1.8_step_0.5/filtered_metrics/filtered_metrics.stream"
input_file = "/Users/xiaodong/Desktop/UOX-data/UOX1_sub/centers_xatol_0.01_frameinterval_5_lowess_0.53_shifted_0.5_-0.3/xgandalf_iterations_max_radius_1.8_step_0.5/UOX_0.0_0.0.stream"
df_reflections, df_crystal = parse_crystfel_stream(input_file, debug=True)
print(df_reflections.head())
print(df_crystal.head())
