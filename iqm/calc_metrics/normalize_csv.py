import os
import csv
import math
import statistics
from itertools import groupby

def event_sort_key(event_str):
    """
    Splits an event number string on '-' and converts each part to a float.
    Always returns a tuple, so sorting is well-defined even if conversion fails.
    """
    try:
        parts = event_str.split('-')
        return tuple(float(part) for part in parts)
    except Exception:
        # Fallback: return a 1-tuple with the original string to keep key types consistent
        return (event_str,)

def normalize_csv(folder_path,
                  input_csv_name='unnormalized_metrics.csv',
                  output_csv_name='normalized_metrics.csv',
                  normalization_method='zscore'):
    """
    Reads the unnormalized metrics CSV, performs global normalization,
    and writes a new CSV (optionally grouped by event_number).
    """
    input_csv_path = os.path.join(folder_path, input_csv_name)
    output_csv_path = os.path.join(folder_path, output_csv_name)

    # Read all rows from the unnormalized CSV
    rows = []
    with open(input_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip the header
        for row in reader:
            # row = [stream_file, event_number, weighted_rmsd, fraction_outliers,
            #        length_deviation, angle_deviation, peak_ratio, percentage_unindexed]
            rows.append(row)

    if not rows:
        print("No rows found in the unnormalized CSV.")
        return

    # Convert numeric columns to float
    # Indices of numeric columns: 2..7 (the first two columns are strings)
    for row in rows:
        for i in range(2, 8):
            row[i] = float(row[i])

    # Build global metric arrays
    metric_keys = [
        'weighted_rmsd',
        'fraction_outliers',
        'length_deviation',
        'angle_deviation',
        'peak_ratio',
        'percentage_unindexed'
    ]
    global_metrics = {key: [] for key in metric_keys}

    for row in rows:
        global_metrics['weighted_rmsd'].append(row[2])
        global_metrics['fraction_outliers'].append(row[3])
        global_metrics['length_deviation'].append(row[4])
        global_metrics['angle_deviation'].append(row[5])
        global_metrics['peak_ratio'].append(row[6])
        global_metrics['percentage_unindexed'].append(row[7])

    # Perform global normalization
    norm_values = {}
    if normalization_method == 'minmax':
        for k in metric_keys:
            values = global_metrics[k]
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                # No spread: set to 0.5 (midpoint of [0,1])
                norm_values[k] = [0.5] * len(values)
            else:
                denom = (max_val - min_val)
                norm_values[k] = [(v - min_val) / denom for v in values]

    elif normalization_method == 'zscore':
        for k in metric_keys:
            values = global_metrics[k]
            # If all values are identical, stdev == 0. In that case, define all z-scores as 0.0.
            if len(values) < 2:
                norm_values[k] = [0.0] * len(values)
                continue
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values)
            if not math.isfinite(stdev_val) or stdev_val == 0.0:
                # No variance (or weird numeric); set all to 0.0
                norm_values[k] = [0.0] * len(values)
            else:
                inv = 1.0 / stdev_val
                norm_values[k] = [(v - mean_val) * inv for v in values]
    else:
        print(f"Unknown normalization method: {normalization_method}")
        return

    # Insert normalized metrics back into rows in the same order
    for i, row in enumerate(rows):
        row[2] = norm_values['weighted_rmsd'][i]
        row[3] = norm_values['fraction_outliers'][i]
        row[4] = norm_values['length_deviation'][i]
        row[5] = norm_values['angle_deviation'][i]
        row[6] = norm_values['peak_ratio'][i]
        row[7] = norm_values['percentage_unindexed'][i]

    # Sort by (event_number, percentage_unindexed)
    rows.sort(key=lambda r: (event_sort_key(r[1]), r[7]))

    # Write out the final CSV with normalized values
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # same original header
        # Group by event number to replicate your grouping logic:
        for event_num, group in groupby(rows, key=lambda r: r[1]):
            # Write a row that indicates the event group
            writer.writerow([f"Event number: {event_num}"] + [""] * (len(header) - 1))
            for g in group:
                writer.writerow(g)

    print(f"Wrote normalized CSV → {output_csv_path}")

if __name__ == "__main__":
    folder_path = "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_1.0_step_0.1"

    # 1) The unnormalized CSV should already exist (e.g. 'unnormalized_metrics.csv')
    #    from running create_unnormalized_csv(...)

    # 2) Normalize it
    normalize_csv(
        folder_path,
        input_csv_name='unnormalized_metrics_wrmsdtol_2.0_itol_4.0.csv',
        output_csv_name='normalized_metrics.csv',
        normalization_method='zscore'
    )
