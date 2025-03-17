#!/usr/bin/env python

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# cctbx imports
from cctbx.array_family import flex
from cctbx import miller, crystal, sgtbx

from collections import defaultdict

###############################################################################
# 1) Parsing the CrystFEL .stream file
###############################################################################
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


###############################################################################
# 2) Merging symmetry-equivalent reflections with cctbx, with chunking
###############################################################################
def merge_equivalents_cctbx_chunked(
    df,
    a,
    b,
    c,
    alpha,
    beta,
    gamma,
    space_group_symbol="P212121",
    anomalous_flag=False,
    debug=False,
    chunk_size=1_000_000,
):
    """
    Perform cctbx-based merging of a large DataFrame of reflections.

    Because cctbx can fail or run out of memory with tens of millions of reflections,
    we do a two-pass chunk approach:
      1) Partition df into chunks of <= chunk_size. Merge each chunk separately.
      2) Concatenate partial merges, rename I_merged -> I, then do a second merge
         to combine any overlaps between chunks.

    Returns a DataFrame with columns:
      [h, k, l, I_merged, I_sigma, multiplicity, redun]
    """
    space_group_info = sgtbx.space_group_info(space_group_symbol)
    crystal_sym = crystal.symmetry(
        unit_cell=(a, b, c, alpha, beta, gamma), space_group_info=space_group_info
    )
    if debug:
        print("Crystal symmetry:", crystal_sym)

    # 1) Break the DataFrame into chunks, merge each chunk
    n_total = len(df)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    partial_merges = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_total)
        sub = df.iloc[start:end]

        # Create Miller array
        h_list = sub["h"].to_numpy()
        k_list = sub["k"].to_numpy()
        l_list = sub["l"].to_numpy()
        I_list = sub["I"].to_numpy()

        ms = miller.set(
            crystal_symmetry=crystal_sym,
            indices=flex.miller_index(list(zip(h_list, k_list, l_list))),
            anomalous_flag=anomalous_flag,
        )
        ma = miller.array(ms, data=flex.double(I_list))

        if "sigma_I" in sub.columns:
            sigma_list = sub["sigma_I"].to_numpy()
            ma.set_sigmas(flex.double(sigma_list))
        else:
            ma.set_observation_type_xray_intensity()

        merged = ma.merge_equivalents()
        merged_ma = merged.array()

        merged_indices = merged_ma.indices()
        merged_intensities = merged_ma.data()
        merged_sigmas = merged_ma.sigmas()
        multiplicities = merged.merge_groups().sizes()

        part_data = {
            "h": [mi[0] for mi in merged_indices],
            "k": [mi[1] for mi in merged_indices],
            "l": [mi[2] for mi in merged_indices],
            "I_merged": list(merged_intensities),
            "I_sigma": list(
                merged_sigmas if merged_sigmas is not None else [np.nan] * len(merged_indices)
            ),
            "multiplicity": list(multiplicities),
        }
        partial_df = pd.DataFrame(part_data)
        partial_df["redun"] = partial_df["multiplicity"]

        partial_merges.append(partial_df)

        if debug:
            print(f"Chunk {i+1}/{n_chunks}: merged {len(sub)} -> {len(partial_df)} rows")

    # 2) Concatenate partial merges, rename I_merged->I, do second merge
    df_partials = pd.concat(partial_merges, ignore_index=True)
    df_partials.rename(columns={"I_merged": "I"}, inplace=True)

    # Now we have (h,k,l,I,I_sigma,multiplicity, redun), possibly with duplicates
    # across chunks. Merge these again.
    h_list = df_partials["h"].to_numpy()
    k_list = df_partials["k"].to_numpy()
    l_list = df_partials["l"].to_numpy()
    I_list = df_partials["I"].to_numpy()

    ms_final = miller.set(
        crystal_symmetry=crystal_sym,
        indices=flex.miller_index(list(zip(h_list, k_list, l_list))),
        anomalous_flag=anomalous_flag,
    )
    ma_final = miller.array(ms_final, data=flex.double(I_list))

    if "I_sigma" in df_partials.columns:
        # The partial merges had I_sigma but re-merging properly might need them
        # If you'd like to keep them, you'd do a more detailed weighting approach.
        # For now, we just drop them or re-assign them as if they're plain sigmas:
        if debug:
            print("Reattaching partial I_sigma for second merge.")
        # Just treat them as sigmas for the second pass
        sigmas_final = flex.double(df_partials["I_sigma"].to_numpy())
        ma_final.set_sigmas(sigmas_final)

    merged2 = ma_final.merge_equivalents()
    merged2_ma = merged2.array()

    final_indices = merged2_ma.indices()
    final_intensities = merged2_ma.data()
    final_sigmas = merged2_ma.sigmas()
    final_mult = merged2.merge_groups().sizes()

    result_data = {
        "h": [mi[0] for mi in final_indices],
        "k": [mi[1] for mi in final_indices],
        "l": [mi[2] for mi in final_indices],
        "I_merged": list(final_intensities),
        "I_sigma": list(final_sigmas if final_sigmas is not None else [np.nan] * len(final_indices)),
        "multiplicity": list(final_mult),
    }
    merged_df = pd.DataFrame(result_data)
    merged_df["redun"] = merged_df["multiplicity"]

    if debug:
        print(f"Final merged size: {len(merged_df)}")

    return merged_df


###############################################################################
# 3) Naive symmetry-equivalent grouping
###############################################################################
def symmetry_key_simple(h, k, l):
    """
    A naive 'symmetry' key: sort absolute values of (h, k, l).
    This is not real space-group symmetry but can show a rough grouping
    for simpler analyses.
    """
    return tuple(sorted([abs(h), abs(k), abs(l)]))


def analyze_symmetry_equivalents(df):
    """
    Group reflections by (event, sym_key_simple) to compute
    mean, std, and count of I within each group. Returns a DataFrame
    with [event, h, k, l, sym_key, I_mean, I_std, CV_sym, ...].
    """
    df = df.copy()
    df["sym_key"] = df.apply(lambda row: symmetry_key_simple(row["h"], row["k"], row["l"]), axis=1)

    grouped = df.groupby(["event", "sym_key"])
    stats = grouped["I"].agg(["mean", "std", "count"]).reset_index()
    stats.rename(columns={"mean": "I_mean", "std": "I_std", "count": "N"}, inplace=True)

    merged = df.merge(stats, on=["event", "sym_key"], how="left")
    merged["CV_sym"] = merged["I_std"] / merged["I_mean"].replace(0, np.nan)

    return merged


###############################################################################
# 4) Systematic row analysis example
###############################################################################
def example_forbidden_condition(h, k, l):
    """
    Example rule for an orthorhombic-like condition:
    If k=0, reflection is 'forbidden' if (h + l) is odd, else allowed if even.
    """
    if k == 0:
        return ((h + l) % 2 != 0)
    else:
        return False


def analyze_systematic_rows(df):
    """
    Filters (k=0) reflections, sees how many are 'forbidden' vs 'allowed' by the example rule.
    Returns a DataFrame: [event, forbidden_ratio].
    """
    df = df.copy()
    df["is_forbidden"] = df.apply(
        lambda row: example_forbidden_condition(row["h"], row["k"], row["l"]), axis=1
    )
    in_row = df[df["k"] == 0].copy()

    def f_ratio(sub):
        forbidden_sum = sub.loc[sub["is_forbidden"], "I"].sum()
        allowed_sum = sub.loc[~sub["is_forbidden"], "I"].sum()
        return forbidden_sum / (allowed_sum + 1e-9)

    ratio_series = in_row.groupby("event").apply(f_ratio).reset_index()
    ratio_series.columns = ["event", "forbidden_ratio"]
    return ratio_series


###############################################################################
# 5) Approximate Sayre equation analysis
###############################################################################
def build_intensity_dict(df):
    """
    Create intensity_dict[event][(h,k,l)] = I
    for quick lookups in each event.
    """
    intensity_dict = defaultdict(dict)
    for idx, row in df.iterrows():
        ev = row["event"]
        hkl = (row["h"], row["k"], row["l"])
        intensity_dict[ev][hkl] = row["I"]
    return intensity_dict


def approximate_sayre_intensity(h, k, l, intensity_map, hkl_range=3):
    """
    For a single reflection (h,k,l), approximate I_sayre by summing sqrt(I_K * I_(H-K)).
    This is a toy version of the Sayre equation.
    """
    sum_val = 0.0
    for hk1 in range(-hkl_range, hkl_range + 1):
        for hk2 in range(-hkl_range, hkl_range + 1):
            for hk3 in range(-hkl_range, hkl_range + 1):
                K = (hk1, hk2, hk3)
                h2, k2, l2 = h - hk1, k - hk2, l - hk3
                if K in intensity_map and (h2, k2, l2) in intensity_map:
                    I_k = max(intensity_map[K], 0)
                    I_hk = max(intensity_map[(h2, k2, l2)], 0)
                    sum_val += math.sqrt(I_k) * math.sqrt(I_hk)
    return sum_val ** 2


def analyze_sayre_equation(df, hkl_range=3):
    """
    Compute a 'Sayre misfit' per event as a toy example.
    Returns: [event, sayre_misfit].
    """
    intensity_dict = build_intensity_dict(df)

    results = []
    for ev in intensity_dict.keys():
        local_map = intensity_dict[ev]
        misfit_num = 0.0
        misfit_den = 0.0

        for (h, k, l), I_obs in local_map.items():
            I_sayre = approximate_sayre_intensity(h, k, l, local_map, hkl_range=hkl_range)
            misfit_num += abs(I_obs - I_sayre)
            misfit_den += abs(I_obs)

        misfit = misfit_num / (misfit_den + 1e-9)
        results.append({"event": ev, "sayre_misfit": misfit})

    sayre_df = pd.DataFrame(results)
    return sayre_df


###############################################################################
# 6) Orientation matrix + zone-axis checks
###############################################################################
def check_zone_axis(orientation_matrix, threshold=0.8, debug=False):
    """
    Check if any reciprocal lattice vector is nearly parallel to the beam [0,0,1].
    Returns (is_zone, max_dot).
    """
    beam = np.array([0, 0, 1])
    max_dot = 0.0
    for vec in orientation_matrix:
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        unit = vec / norm
        dot = abs(np.dot(unit, beam))
        if dot > max_dot:
            max_dot = dot
    is_zone = (max_dot >= threshold)
    return is_zone, max_dot


def compute_orientation_fields(df_cryst, debug=False):
    """
    For each row in df_cryst, parse astar/bstar/cstar, build orientation_matrix,
    check if any vector is ~parallel to [0,0,1]. Return updated DataFrame.
    """
    orientation_list = []
    is_zone_list = []
    zone_dot_list = []

    for idx, row in df_cryst.iterrows():
        try:
            for col in ["astar", "bstar", "cstar"]:
                if isinstance(row[col], str):
                    row[col] = eval(row[col])  # caution with untrusted input
            orientation_matrix = np.array([row["astar"], row["bstar"], row["cstar"]])
            orientation_list.append(orientation_matrix)

            is_zone, dot_val = check_zone_axis(orientation_matrix, threshold=0.95, debug=debug)
            is_zone_list.append(is_zone)
            zone_dot_list.append(dot_val if dot_val is not None else np.nan)

        except Exception as e:
            if debug:
                print(f"Error processing event {row['event']}: {e}")
            orientation_list.append(None)
            is_zone_list.append(False)
            zone_dot_list.append(np.nan)

    df_cryst["orientation_matrix"] = orientation_list
    df_cryst["is_zone_axis"] = is_zone_list
    df_cryst["zone_dot"] = zone_dot_list
    return df_cryst


###############################################################################
# Example main usage
###############################################################################
if __name__ == "__main__":

    # ---------------------------
    # User Configuration
    # ---------------------------
    stream_file = "/home/bubl3932/files/MIL101_ICF/xgandalf_iterations_max_radius_1.8_step_0.5/filtered_metrics/filtered_metrics.stream"
    # Provide the correct cell + angles + space group for your data:
    a, b, c = 87.4, 87.4, 87.4
    alpha, beta, gamma = 90, 90, 90
    space_group_sym = "Fd-3m"  # e.g. "Fd-3m", "P212121", etc.

    # ---------------------------
    # 1) Parse the .stream file
    # ---------------------------
    df_reflections, df_crystal = parse_crystfel_stream(stream_file, debug=False)
    print("\nReflections (first 5 rows):")
    print(df_reflections.head(), "\n")
    print("Crystal metadata (first 5 rows):")
    print(df_crystal.head(), "\n")

    # (Optional) confirm h, k, l are recognized as int
    print("Dtypes:\n", df_reflections[["h", "k", "l"]].dtypes)

    # ---------------------------
    # 2) Merge reflections with cctbx (chunked)
    # ---------------------------
    # merged_cctbx = merge_equivalents_cctbx_chunked(
    #     df_reflections,
    #     a,
    #     b,
    #     c,
    #     alpha,
    #     beta,
    #     gamma,
    #     space_group_symbol=space_group_sym,
    #     anomalous_flag=False,
    #     debug=True,
    #     chunk_size=1_000_000  # or set to ~1e6 or smaller if your dataset is huge
    # )
    # print("\nFinal Merged Data (cctbx) - first 5 rows:")
    # print(merged_cctbx.head())

    # ---------------------------
    # 3) Naive symmetry grouping & stats
    # ---------------------------
    # df_sym = analyze_symmetry_equivalents(df_reflections)
    # print("\nNaive symmetry stats (first 5 rows):")
    # print(df_sym[
    #     ["event", "h", "k", "l", "sym_key", "I_mean", "I_std", "CV_sym"]
    # ].head(5))

    # # ---------------------------
    # # 4) Example: forbidden reflection analysis
    # # ---------------------------
    # systematic_ratios = analyze_systematic_rows(df_sym)
    # print("\nSystematic row analysis (forbidden ratio) - first 5:")
    # print(systematic_ratios.head(5))

    # # ---------------------------
    # # 5) Approx. Sayre equation
    # # ---------------------------
    # sayre_metrics = analyze_sayre_equation(df_sym, hkl_range=1)
    # print("\nSayre misfit (first 5 events):")
    # print(sayre_metrics.head(5))

    # ---------------------------
    # 6) Orientation fields
    # ---------------------------
    df_crystal_zone = compute_orientation_fields(df_crystal, debug=False)
    df_crystal_zone.sort_values(by="zone_dot", ascending=False, inplace=True)

    print("\nOrientation analysis (first 5 events):")
    print(df_crystal_zone[["event", "is_zone_axis", "zone_dot"]].head(5))
