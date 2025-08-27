# ssed_calc_metrics.py
#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from calc_metrics.process_indexing_metrics import process_indexing_metrics

def get_ui(parent):
    """
    Creates and returns a Frame containing the Metrics Processing GUI.
    
    This GUI lets the user select a folder containing stream files,
    set the WRMSD and Indexing tolerances, and then process the metrics.
    All feedback is printed to the terminal.
    """
    frame = tk.Frame(parent)
    
    # Explanation text.
    explanation = (
    "Processes stream files from multiple indexing iterations to evaluate key metrics on frame quality.\n\n"
    "Evaluated Metrics:\n"
    " - Weighted RMSD: Intensity-weighted overall deviation.\n"
    " - Fraction Outliers: Proportion of segments flagged as outliers in RMSD calculation.\n"
    " - Length Deviation: Variability in measured cell lengths.\n"
    " - Angle Deviation: Variability in measured cell angles.\n"
    " - Peak Ratio: Ratio of predicted to observed peaks.\n"
    " - Percentage Indexed: Percentage of peaks successfully indexed.\n\n"
    "Parameters:\n"
    " - WRMSD Tolerance (Default: 2.0): Threshold (in standard deviations) for flagging outliers. "
    "If the difference between an observed and a predicted peak exceeds this threshold, the peak is rejected.\n"
    " - Indexing Tolerance (Default: 4.0): Maximum allowable deviation (in pixels) between observed and predicted "
    "peak positions for a peak to be considered indexed."
)

    explanation_label = tk.Label(frame, text=explanation, justify=tk.LEFT, wraplength=600)
    explanation_label.pack(padx=10, pady=10)
    
    # Folder chooser for the stream files folder.
    folder_frame = tk.Frame(frame)
    folder_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(folder_frame, text="Stream File Folder:").pack(side=tk.LEFT)
    folder_var = tk.StringVar()
    folder_entry = tk.Entry(folder_frame, textvariable=folder_var, width=50)
    folder_entry.pack(side=tk.LEFT, padx=5)
    def browse_folder():
        folder = filedialog.askdirectory(
            title="Select Stream File Folder",
            initialdir=os.getcwd()
        )
        if folder:
            folder_var.set(folder)
    tk.Button(folder_frame, text="Browse", command=browse_folder).pack(side=tk.LEFT, padx=5)
    
    # Tolerance parameters.
    params_frame = tk.Frame(frame)
    params_frame.pack(padx=10, pady=5)
    
    tk.Label(params_frame, text="WRMSD Tolerance:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    wrmsd_var = tk.DoubleVar(value=2.0)
    tk.Entry(params_frame, textvariable=wrmsd_var, width=10).grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(params_frame, text="Indexing Tolerance:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
    indexing_tol_var = tk.DoubleVar(value=4.0)
    tk.Entry(params_frame, textvariable=indexing_tol_var, width=10).grid(row=0, column=3, padx=5, pady=5)
    
    # Process Metrics button.
    def on_process():
        folder = folder_var.get()
        if not folder:
            messagebox.showerror("Error", "Please select a stream file folder.")
            return
        wrmsd = wrmsd_var.get()
        indexing_tol = indexing_tol_var.get()
        print("Processing metrics for folder:", folder)
        print("WRMSD Tolerance:", wrmsd)
        print("Indexing Tolerance:", indexing_tol)
        try:
            process_indexing_metrics(folder, wrmsd_tolerance=wrmsd, indexing_tolerance=indexing_tol)
            print("Metrics processed successfully.")
        except Exception as e:
            print("Error processing metrics:", e)
    
    tk.Button(frame, text="Process Metrics", command=on_process, bg="lightblue").pack(padx=10, pady=10)
    
    return frame

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Metrics Processing GUI")
    ui = get_ui(root)
    ui.pack(fill="both", expand=True)
    root.mainloop()
