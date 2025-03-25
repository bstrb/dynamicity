import numpy as np
import pandas as pd
from cctbx.array_family import flex
from cctbx import miller, crystal, sgtbx

def label_asu_indices_cctbx(df_reflections, df_crystal,
                            space_group_number=71,  # e.g., 71=P4(3)21(2), 19=P212121, etc.
                            anomalous_flag=False,
                            debug=False):
    """
    For each reflection row in df_reflections, map (h,k,l) to the symmetry-equivalent
    Miller index in the chosen space group's Asymmetric Unit (ASU). We'll do it
    per-event, using that event's unit cell.

    Args:
      df_reflections (pd.DataFrame): Must have columns 'event','h','k','l','I'...
      df_crystal (pd.DataFrame): Must have columns 'event','a','b','c','alpha','beta','gamma'...
      space_group_number (int): The desired space group (by number) for the ASU mapping.
      anomalous_flag (bool): If True, keep +/- Friedel pairs separate.
      debug (bool): Print debug info.

    Returns:
      df_out (pd.DataFrame):
         Same as df_reflections, but with 3 new columns: 'h_asu','k_asu','l_asu'.
    """
    # We'll accumulate labeled rows here
    out_rows = []

    # Create a dictionary: event -> (a,b,c,alpha,beta,gamma)
    # so we don't have to re-look-up each time
    event_cell_map = {}
    for idx, row in df_crystal.iterrows():
        event_ = row["event"]
        a_     = row.get("a", np.nan)
        b_     = row.get("b", np.nan)
        c_     = row.get("c", np.nan)
        alpha_ = row.get("alpha", 90.0)
        beta_  = row.get("beta", 90.0)
        gamma_ = row.get("gamma", 90.0)
        event_cell_map[event_] = (a_, b_, c_, alpha_, beta_, gamma_)

    # Create the space group info object from the provided group number
    space_group_info = sgtbx.space_group_info(number=space_group_number)

    # Group reflections by event so we only create a cctbx crystal_symmetry once per event
    for event_id, subdf in df_reflections.groupby("event"):
        if event_id not in event_cell_map:
            # Possibly no crystal info for that event
            # We'll just keep them as-is (no mapping) or skip them
            if debug:
                print(f"[WARN] No cell data for event={event_id}, skipping ASU mapping.")
            out_rows.append(subdf)  # just keep them unmodified
            continue

        (a, b, c, alpha, beta, gamma) = event_cell_map[event_id]

        # Build the crystal symmetry object
        crystal_sym = crystal.symmetry(
            unit_cell=(a, b, c, alpha, beta, gamma),
            space_group_info=space_group_info
        )
        if debug:
            print(f"Event='{event_id}': crystal_sym={crystal_sym}")

        # Convert (h,k,l) to cctbx flex arrays
        h_list = subdf["h"].values
        k_list = subdf["k"].values
        l_list = subdf["l"].values
        miller_indices = flex.miller_index(
            [(int(h), int(k), int(l)) for (h, k, l) in zip(h_list, k_list, l_list)]
        )

        # Build a Miller set
        ms = miller.set(
            crystal_symmetry=crystal_sym,
            indices=miller_indices,
            anomalous_flag=anomalous_flag
        )

        # Map to the ASU
        ms_asu = ms.map_to_asu()

        # Extract the mapped indices
        mapped_indices = ms_asu.indices()

        # Prepare a copy of the subdf with new columns h_asu, k_asu, l_asu
        subdf_copy = subdf.copy()
        subdf_copy["h_asu"] = [idx[0] for idx in mapped_indices]
        subdf_copy["k_asu"] = [idx[1] for idx in mapped_indices]
        subdf_copy["l_asu"] = [idx[2] for idx in mapped_indices]

        out_rows.append(subdf_copy)

    # Combine all labeled data
    df_out = pd.concat(out_rows, ignore_index=True)
    return df_out
