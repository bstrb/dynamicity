#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

# Import your custom modules.
from gui_util.create_scrollable_frame import create_scrollable_frame
from filter_and_combine.csv_to_stream import write_stream_from_filtered_csv
from filter_and_combine.interactive_iqm import (
    read_metric_csv,
    select_best_results_by_event,
    get_metric_ranges,
    filter_rows,
    write_filtered_csv,
    filter_and_combine,          # ← new helper
)

# Global variables to store key file paths and data.
global_csv_path = [None]            # Stores the loaded CSV file path.
global_filtered_csv_path = [None]     # Stores the path to the filtered CSV.
global_all_rows = [None]              # Stores all rows loaded from the CSV.

# The metrics to be analyzed.
metrics_in_order = [
    'weighted_rmsd',
    'fraction_outliers',
    'length_deviation',
    'angle_deviation',
    'peak_ratio',
    'percentage_unindexed'
]

def get_ui(parent):
    """
    Creates and returns a Tkinter Frame containing the interactive metrics analysis UI.
    """
    main_frame = tk.Frame(parent)
    explanation = (
    "Load a CSV file containing normalized metrics for indexing quality from previous step. Once loaded, you have two analysis options:\n\n"
    "1. Separate Metrics Filtering:\n"
    "   - Adjust individual thresholds for each separate metric using sliders.\n"
    "   - The tool filters data based on these thresholds and displays histograms for each metric.\n\n"
    "2. Combined Metric Creation & Filtering:\n"
    "   - Assign weights to each separate metric to create an overall combined metric.\n"
    "   - Adjust the combined metric threshold to filter and select the best rows per event.\n"
    "   - Optionally, convert the filtered results into a stream file for further processing.\n\n"
    )
    explanation_label = tk.Label(main_frame, text=explanation, justify=tk.LEFT, wraplength=600)
    explanation_label.pack(padx=10, pady=10)

    # --- CSV File Selection Section ---
    csv_frame = tk.LabelFrame(main_frame, text="Select CSV with Normalized Metrics", padx=10, pady=10)
    csv_frame.pack(fill="x", padx=10, pady=5)
    
    tk.Label(csv_frame, text="CSV File:").grid(row=0, column=0, sticky="w")
    csv_path_var = tk.StringVar()
    csv_entry = tk.Entry(csv_frame, textvariable=csv_path_var, width=50)
    csv_entry.grid(row=0, column=1, padx=5)
    def browse_csv():
        path = filedialog.askopenfilename(
            title="Select CSV file with normalized metrics",
            filetypes=[("CSV Files", "*.csv")],
            initialdir=os.getcwd()
        )
        if path:
            csv_path_var.set(path)
    tk.Button(csv_frame, text="Browse", command=browse_csv).grid(row=0, column=2, padx=5)
    
    load_csv_button = tk.Button(csv_frame, text="Load CSV", bg="lightblue")
    load_csv_button.grid(row=1, column=0, columnspan=3, pady=5)
    
    # Container that will later hold the rest of the UI (populated after CSV load).
    ui_container = tk.Frame(main_frame)
    ui_container.pack(fill="both", expand=True, padx=10, pady=5)
    
    def load_csv_callback():
        path = csv_path_var.get()
        if not path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        global_csv_path[0] = path
        print("Loading CSV file:", path)
        try:
            # Read CSV and group by event.
            grouped_data = read_metric_csv(path, group_by_event=True)
            all_rows = []
            for rows in grouped_data.values():
                all_rows.extend(rows)
            global_all_rows[0] = all_rows
            print(f"Loaded {len(all_rows)} rows from CSV.")
            # Set filtered CSV path in the same folder as CSV.
            filtered_csv = os.path.join(os.path.dirname(path), 'filtered_metrics.csv')
            global_filtered_csv_path[0] = filtered_csv
            # Now create the metrics analysis UI.
            create_analysis_ui(all_rows)
        except Exception as e:
            print("Error loading CSV:", e)
    
    load_csv_button.config(command=load_csv_callback)
    
    # --- Function to Create the Analysis UI (after CSV load) ---
    def create_analysis_ui(all_rows):
        # Clear any existing content.
        for widget in ui_container.winfo_children():
            widget.destroy()
        
        # SECTION 1: Separate Metrics Filtering
        separate_frame = tk.LabelFrame(ui_container, text="Separate Metrics Filtering", padx=10, pady=10)
        separate_frame.pack(fill="x", pady=5)
        
        # Get metric ranges.
        ranges_dict = get_metric_ranges(all_rows, metrics=metrics_in_order)
        metric_sliders = {}
        # Create a slider for each metric.
        row = 0
        for metric in metrics_in_order:
            mn, mx = ranges_dict[metric]
            var = tk.DoubleVar(value=mx)  # Default to max (include all values)
            metric_sliders[metric] = var
            tk.Label(separate_frame, text=f"{metric} ≤").grid(row=row, column=0, sticky="e", padx=5, pady=2)
            scale = tk.Scale(separate_frame, variable=var, from_=mn, to=mx,
                             resolution=(mx - mn)/100.0 if mx != mn else 0.01,
                             orient="horizontal", length=300)
            scale.grid(row=row, column=1, padx=5, pady=2)
            row += 1
        
        filter_separate_button = tk.Button(separate_frame, text="Apply Separate Metrics Thresholds", bg="lightblue")
        filter_separate_button.grid(row=row, column=0, columnspan=2, pady=5)
        
        def on_filter_separate_clicked():
            # Close any previous plots.
            plt.close('all')
            print("\n" + "="*50)
            print("SEPARATE METRICS FILTERING")
            thresholds = {m: metric_sliders[m].get() for m in metrics_in_order}
            filtered_separate = filter_rows(all_rows, thresholds)
            print(f"Filtering: {len(all_rows)} total rows -> {len(filtered_separate)} pass thresholds.")
            if not filtered_separate:
                print("No rows passed the thresholds.")
                return
            # Plot histograms for each metric.
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))
            axes = axes.flatten()
            for i, metric in enumerate(metrics_in_order):
                values = [r[metric] for r in filtered_separate if metric in r]
                axes[i].hist(values, bins=20)
                axes[i].set_title(f"Histogram of {metric}")
                axes[i].set_xlabel(metric)
                axes[i].set_ylabel("Count")
            plt.tight_layout()
            plt.show()
        
        filter_separate_button.config(command=on_filter_separate_clicked)
        
        # SECTION 2: Combined Metric Creation & Filtering
        combined_frame = tk.LabelFrame(ui_container, text="Combined Metric Creation & Filtering", padx=10, pady=10)
        combined_frame.pack(fill="x", pady=5)
        
        # Create weight text fields for each metric.
        weight_fields = {}
        row = 0
        for metric in metrics_in_order:
            tk.Label(combined_frame, text=f"{metric} Weight:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
            var = tk.DoubleVar(value=0.0)
            weight_fields[metric] = var
            tk.Entry(combined_frame, textvariable=var, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
        
        # Combined metric threshold slider.
        tk.Label(combined_frame, text="Combined Metric Threshold ≤").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        combined_threshold_var = tk.DoubleVar(value=0.0)
        combined_threshold_scale = tk.Scale(combined_frame, variable=combined_threshold_var,
                                            from_=0.0, to=1.0, resolution=0.01,
                                            orient="horizontal", length=300)
        combined_threshold_scale.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        
        create_combined_button = tk.Button(combined_frame, text="Create Combined Metric", bg="lightblue")
        create_combined_button.grid(row=row, column=0, columnspan=2, pady=5)
        
        def create_or_update_combined_metric():
            print("\n" + "="*50)
            print("COMBINED METRIC CREATION (via filter_and_combine)")

            weights_list     = [weight_fields[m].get() for m in metrics_in_order]
            selected_metrics = metrics_in_order     # same order
            # No pre‑filter here – we’re only building the metric:
            rows_with_metric = filter_and_combine(
                rows              = all_rows,
                pre_filter         = None,
                metrics_to_combine = selected_metrics,
                weights            = weights_list,
                new_metric_name    = "combined_metric",
            )

            combined_vals = [r["combined_metric"] for r in rows_with_metric]
            if not combined_vals:
                print("Failed to create combined metric. Check your weights.")
                return

            cmin, cmax = min(combined_vals), max(combined_vals)
            current_val = combined_threshold_var.get()
            if current_val < cmin or current_val > cmax:
                current_val = cmax
            combined_threshold_scale.config(from_=cmin, to=cmax)
            combined_threshold_var.set(current_val)

            print(f"Combined metric created successfully!\n  * Min: {cmin:.3f}\n  * Max: {cmax:.3f}")
            print("Adjust the slider and click 'Apply Combined Metric Threshold (Best Rows)' to filter.")

        
        create_combined_button.config(command=create_or_update_combined_metric)
        
        filter_combined_button = tk.Button(combined_frame, text="Apply Combined Metric Threshold (Best Rows)", bg="lightblue")
        filter_combined_button.grid(row=row+1, column=0, columnspan=2, pady=5)
        def on_filter_combined_clicked():
            # Close any previous plots.
            plt.close('all')
            print("\n" + "="*50)
            print("COMBINED METRIC FILTERING (with pre‑filter)")

            # ------------------------------------------------------------------
            # 1. Gather *per‑metric* thresholds from the sliders in Section 1
            # ------------------------------------------------------------------
            pre_thresholds = {m: metric_sliders[m].get() for m in metrics_in_order}

            # ------------------------------------------------------------------
            # 2. Gather weights from the text boxes in Section 2
            # ------------------------------------------------------------------
            weights_list      = [weight_fields[m].get() for m in metrics_in_order]
            selected_metrics  = metrics_in_order                                 # same order

            # ------------------------------------------------------------------
            # 3. Run one helper that does:
            #    • filter_rows(all_rows, pre_thresholds)
            #    • create_combined_metric(..., weights_list)
            # ------------------------------------------------------------------
            pre_filtered_rows = filter_and_combine(
                rows               = all_rows,
                pre_filter          = pre_thresholds,
                metrics_to_combine  = selected_metrics,
                weights             = weights_list,
                new_metric_name     = "combined_metric",
            )

            print(f"{len(all_rows)} total rows  →  "
                f"{len(pre_filtered_rows)} survive the separate‑metric thresholds")

            if not pre_filtered_rows:
                print("No rows left after the pre‑filter – nothing to combine.")
                return

            # ------------------------------------------------------------------
            # 4. Now apply the *combined‑metric* slider threshold
            # ------------------------------------------------------------------
            combined_thr   = combined_threshold_var.get()
            by_combined    = [r for r in pre_filtered_rows if r['combined_metric'] <= combined_thr]

            print(f"... {len(by_combined)} remain after combined_metric ≤ {combined_thr:.3f}")

            if not by_combined:
                print("No rows passed the combined‑metric threshold.")
                return

            # ------------------------------------------------------------------
            # 5. Best‑row‑per‑event → CSV + histogram (unchanged from your code)
            # ------------------------------------------------------------------
            grouped = {}
            for r in by_combined:
                grouped.setdefault(r['event_number'], []).append(r)

            best_filtered = select_best_results_by_event(grouped, sort_metric='combined_metric')
            print(f"{len(best_filtered)} best rows selected (one per event).")
            write_filtered_csv(best_filtered, global_filtered_csv_path[0])
            print(f"Wrote → {global_filtered_csv_path[0]}")

            plt.figure(figsize=(8, 6))
            plt.hist([r['combined_metric'] for r in best_filtered], bins=20)
            plt.title("Histogram of Best Rows (combined_metric)")
            plt.xlabel("combined_metric")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

        
        filter_combined_button.config(command=on_filter_combined_clicked)
        
        convert_button = tk.Button(combined_frame, text="Convert to Stream", bg="green")
        convert_button.grid(row=row+2, column=0, columnspan=2, pady=5)
        
        def on_convert_clicked():
            print("\n" + "="*50)
            print("CONVERT TO STREAM")
            print("Starting conversion...")
            output_dir = os.path.join(os.path.dirname(global_csv_path[0]), 'filtered_metrics')
            os.makedirs(output_dir, exist_ok=True)
            OUTPUT_STREAM_PATH = os.path.join(output_dir, 'filtered_metrics.stream')
            print("Step 1/5: Reading filtered CSV file...")
            filtered_grouped_data = read_metric_csv(global_filtered_csv_path[0], group_by_event=True)
            print("Step 2/5: Checking for combined metric and selecting best rows if present...")
            first_event = next(iter(filtered_grouped_data.values()))
            if "combined_metric" in first_event[0]:
                best_filtered = select_best_results_by_event(filtered_grouped_data, sort_metric="combined_metric")
                write_filtered_csv(best_filtered, global_filtered_csv_path[0])
                print("Best rows selected and CSV overwritten.")
            else:
                print("No combined_metric found, skipping best-row selection.")
            print("Step 3/5: Writing the .stream file...")
            write_stream_from_filtered_csv(
                filtered_csv_path=global_filtered_csv_path[0],
                output_stream_path=OUTPUT_STREAM_PATH,
                event_col="event_number",
                streamfile_col="stream_file"
            )
            print("Step 4/5: Conversion complete!")
            print(f"CSV has been successfully converted to:\n  {OUTPUT_STREAM_PATH}")
        
        convert_button.config(command=on_convert_clicked)
    
    return main_frame

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Interactive Metrics Analysis Tool")
    # Set a minimum size for the window
    root.minsize(800, 600)
    
    # Create a scrollable frame in the root.
    scroll_frame = create_scrollable_frame(root)
    ui = get_ui(scroll_frame)
    ui.pack(fill="both", expand=True)
    
    root.mainloop()
