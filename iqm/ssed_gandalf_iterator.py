# ssed_gandalf_iterator.py - Gandalf Indexing GUI
#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Import the indexing function.
from gandalf_iterations.gandalf_radial_iterator import gandalf_iterator

# ==============================================================================
# Import the necessary libraries for file handling and cleanup.
import glob
import shutil
import atexit
import signal

def cleanup_temp_dirs():
    """Remove all directories in the current working directory that start with 'indexamajig'."""
    for d in glob.glob("indexamajig*"):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"Removed temporary directory: {d}")

# Register cleanup function to run at program exit.
atexit.register(cleanup_temp_dirs)

# Optionally, catch termination signals to ensure cleanup on interruptions.
def signal_handler(sig, frame):
    cleanup_temp_dirs()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# ==============================================================================

# Define default peakfinder options.
default_peakfinder_options = {
    'cxi': "--peaks=cxi",
    'peakfinder9': """--peaks=peakfinder9
--min-snr-biggest-pix=7
--min-snr-peak-pix=6
--min-snr=5
--min-sig=11
--min-peak-over-neighbour=-inf
--local-bg-radius=3""",
    'peakfinder8': """--peaks=peakfinder8
--threshold=800
--min-snr=5
--min-pix-count=2
--max-pix-count=200
--local-bg-radius=3
--min-res=0
--max-res=1200"""
}

# Fixed indexing flags.
INDEXING_FLAGS = [
    "--indexing=xgandalf",
    "--integration=rings",
]

def get_ui(parent):
    """
    Creates and returns a Frame containing the complete Gandalf Indexing GUI.
    """
    frame = tk.Frame(parent)
    description = (
    "Run the indexamajig command with optional outward center shifts in a grid.\n"
    "Select .geom and .cell files and choose the input folder with .h5 files to be processed.\n"
    "Set basic parameters such as Output Base (name of your sample), Threads (number of used CPU), Max Radius (maximum shift distance), and Step (grid spacing).\n"
    "Configure Peakfinder options, advanced indexing parameters and optionally extra flags.\n"
    "Click 'Run Indexing' to execute indexing iterations with shifted centers until the specified radius."
    )
    description_label = tk.Label(frame, text=description, justify=tk.LEFT, wraplength=600)
    description_label.pack(padx=10, pady=10)

    # ----- File Selection Section -----
    file_frame = tk.LabelFrame(frame, text="File Selection", padx=10, pady=10)
    file_frame.pack(fill="x", padx=10, pady=5)
    
    # Geometry file chooser.
    tk.Label(file_frame, text="Geometry File (.geom):").grid(row=0, column=0, sticky="w")
    geom_path_var = tk.StringVar()
    geom_entry = tk.Entry(file_frame, textvariable=geom_path_var, width=50)
    geom_entry.grid(row=0, column=1, padx=5)
    tk.Button(file_frame, text="Browse", command=lambda: geom_path_var.set(
        filedialog.askopenfilename(
            title="Select Geometry File (.geom)",
            filetypes=[("Geometry Files", "*.geom")],
            initialdir=os.getcwd()
        )
    )).grid(row=0, column=2, padx=5)
    
    # Cell file chooser.
    tk.Label(file_frame, text="Cell File (.cell):").grid(row=1, column=0, sticky="w")
    cell_path_var = tk.StringVar()
    cell_entry = tk.Entry(file_frame, textvariable=cell_path_var, width=50)
    cell_entry.grid(row=1, column=1, padx=5)
    tk.Button(file_frame, text="Browse", command=lambda: cell_path_var.set(
        filedialog.askopenfilename(
            title="Select Cell File (.cell)",
            filetypes=[("Cell Files", "*.cell")],
            initialdir=os.getcwd()
        )
    )).grid(row=1, column=2, padx=5)
    
    # Input folder chooser.
    tk.Label(file_frame, text="Input Folder:").grid(row=2, column=0, sticky="w")
    input_folder_var = tk.StringVar()
    input_folder_entry = tk.Entry(file_frame, textvariable=input_folder_var, width=50)
    input_folder_entry.grid(row=2, column=1, padx=5)
    tk.Button(file_frame, text="Browse", command=lambda: input_folder_var.set(
        filedialog.askdirectory(
            title="Select Input Folder",
            initialdir=os.getcwd()
        )
    )).grid(row=2, column=2, padx=5)
    
    # ----- Basic Parameters -----
    basic_frame = tk.LabelFrame(frame, text="Basic Parameters", padx=10, pady=10)
    basic_frame.pack(fill="x", padx=10, pady=5)
    
    tk.Label(basic_frame, text="Output Base:").grid(row=0, column=0, sticky="w")
    output_base_var = tk.StringVar(value="Xtal")
    tk.Entry(basic_frame, textvariable=output_base_var, width=40).grid(row=0, column=1, padx=5)
    
    tk.Label(basic_frame, text="Threads:").grid(row=1, column=0, sticky="w")
    threads_var = tk.StringVar(value="24")
    tk.Entry(basic_frame, textvariable=threads_var, width=10).grid(row=1, column=1, sticky="w", padx=5)
    
    tk.Label(basic_frame, text="Max Radius:").grid(row=2, column=0, sticky="w")
    max_radius_var = tk.StringVar(value="0.1")
    tk.Entry(basic_frame, textvariable=max_radius_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
    
    tk.Label(basic_frame, text="Step:").grid(row=3, column=0, sticky="w")
    step_var = tk.StringVar(value="0.1")
    tk.Entry(basic_frame, textvariable=step_var, width=10).grid(row=3, column=1, sticky="w", padx=5)
    
    # ----- Peakfinder Section -----
    peakfinder_frame = tk.LabelFrame(frame, text="Peakfinder Options", padx=10, pady=10)
    peakfinder_frame.pack(fill="x", padx=10, pady=5)
    
    tk.Label(peakfinder_frame, text="Peakfinder:").grid(row=0, column=0, sticky="w")
    peakfinder_option_var = tk.StringVar(value="cxi")
    # OptionMenu: Display labels and values are as follows.
    peakfinder_options = {"CXI": "cxi", "Peakfinder9": "peakfinder9", "Peakfinder8": "peakfinder8"}
    option_menu = tk.OptionMenu(peakfinder_frame, peakfinder_option_var, *peakfinder_options.values())
    option_menu.grid(row=0, column=1, padx=5, sticky="w")
    
    tk.Label(peakfinder_frame, text="Peakfinder Params:").grid(row=1, column=0, sticky="nw")
    # Use a Text widget for multi-line parameters.
    peakfinder_params_text = tk.Text(peakfinder_frame, width=60, height=4)
    peakfinder_params_text.grid(row=1, column=1, padx=5, pady=5)
    # Initialize with default options.
    peakfinder_params_text.insert("1.0", default_peakfinder_options[peakfinder_option_var.get()])
    
    # When the dropdown changes, update the text widget with the default options.
    def update_peakfinder_params(*args):
        method = peakfinder_option_var.get()
        peakfinder_params_text.delete("1.0", tk.END)
        peakfinder_params_text.insert("1.0", default_peakfinder_options.get(method, ""))
    peakfinder_option_var.trace_add("write", update_peakfinder_params)
    
    # ----- Advanced Indexing Parameters -----
    adv_frame = tk.LabelFrame(frame, text="Advanced Indexing Parameters", padx=10, pady=10)
    adv_frame.pack(fill="x", padx=10, pady=5)
    
    tk.Label(adv_frame, text="Min Peaks:").grid(row=0, column=0, sticky="w")
    min_peaks_var = tk.StringVar(value="15")
    tk.Entry(adv_frame, textvariable=min_peaks_var, width=10).grid(row=0, column=1, padx=5)
    
    tk.Label(adv_frame, text="Cell Tolerance:").grid(row=0, column=2, sticky="w")
    tolerance_var = tk.StringVar(value="10,10,10,5")
    tk.Entry(adv_frame, textvariable=tolerance_var, width=20).grid(row=0, column=3, padx=5)
    
    tk.Label(adv_frame, text="Sampling Pitch:").grid(row=1, column=0, sticky="w")
    sampling_pitch_var = tk.StringVar(value="5")
    tk.Entry(adv_frame, textvariable=sampling_pitch_var, width=10).grid(row=1, column=1, padx=5)
    
    tk.Label(adv_frame, text="Grad Desc Iterations:").grid(row=1, column=2, sticky="w")
    grad_desc_iter_var = tk.StringVar(value="1")
    tk.Entry(adv_frame, textvariable=grad_desc_iter_var, width=10).grid(row=1, column=3, padx=5)
    
    tk.Label(adv_frame, text="XGandalf Tolerance:").grid(row=2, column=0, sticky="w")
    xgandalf_tol_var = tk.StringVar(value="0.02")
    tk.Entry(adv_frame, textvariable=xgandalf_tol_var, width=10).grid(row=2, column=1, padx=5)
    
    tk.Label(adv_frame, text="Integration Radius:").grid(row=2, column=2, sticky="w")
    int_radius_var = tk.StringVar(value="2,4,10")
    tk.Entry(adv_frame, textvariable=int_radius_var, width=20).grid(row=2, column=3, padx=5)
    
    # ----- Other Extra Flags -----
    other_flags_frame = tk.LabelFrame(frame, text="Other Extra Flags", padx=10, pady=10)
    other_flags_frame.pack(fill="x", padx=10, pady=5)
    
    other_flags_text = tk.Text(other_flags_frame, width=60, height=4)
    other_flags_text.pack(padx=5, pady=5)
    other_flags_text.insert("1.0", """--no-revalidate
--no-half-pixel-shift
--no-refine
--no-non-hits-in-stream""")
# --no-image-data
    
    # ----- Run Button -----
    run_button = tk.Button(frame, text="Run Indexing", bg="lightblue")
    run_button.pack(padx=10, pady=10)
    
    # ----- Callback for Run Button -----
    def on_run_clicked():
        # Retrieve file selections.
        geom_file = geom_path_var.get()
        cell_file = cell_path_var.get()
        input_folder = input_folder_var.get()
        if not geom_file or not cell_file or not input_folder:
            messagebox.showerror("Error", "Please ensure Geometry, Cell, and Input Folder are selected.")
            return
        
        # Retrieve basic parameters.
        output_base = output_base_var.get()
        try:
            threads = int(threads_var.get())
        except:
            threads = 24
        try:
            max_radius = float(max_radius_var.get())
        except:
            max_radius = 1.8
        try:
            step = float(step_var.get())
        except:
            step = 0.5
        
        # Peakfinder parameters.
        peakfinder_method = peakfinder_option_var.get()
        peakfinder_params = peakfinder_params_text.get("1.0", tk.END).strip().splitlines()
        
        # Advanced parameters.
        min_peaks_flag = f"--min-peaks={min_peaks_var.get()}"
        tolerance_flag = f"--tolerance={tolerance_var.get()}"
        sampling_pitch_flag = f"--xgandalf-sampling-pitch={sampling_pitch_var.get()}"
        grad_desc_flag = f"--xgandalf-grad-desc-iterations={grad_desc_iter_var.get()}"
        xgandalf_tol_flag = f"--xgandalf-tolerance={xgandalf_tol_var.get()}"
        int_radius_flag = f"--int-radius={int_radius_var.get()}"
        advanced_flags = [min_peaks_flag, tolerance_flag, sampling_pitch_flag,
                          grad_desc_flag, xgandalf_tol_flag, int_radius_flag]
        
        # Other flags.
        other_flags = [line.strip() for line in other_flags_text.get("1.0", tk.END).splitlines() if line.strip()]
        
        # Combine all flags in the proper order.
        flags_list = advanced_flags + other_flags + peakfinder_params + INDEXING_FLAGS
        
        print("Running gandalf_iterator with the following parameters:")
        print("Geometry File:", geom_file)
        print("Cell File:", cell_file)
        print("Input Folder:", input_folder)
        print("Output Base:", output_base)
        print("Threads:", threads)
        print("Max Radius:", max_radius)
        print("Step:", step)
        print("\nPeakfinder Option:", peakfinder_method)
        print("\nAdvanced Flags:")
        for f in advanced_flags:
            print(" ", f)
        print("\nOther Flags:")
        for f in other_flags:
            print(" ", f)
        print("\nCombined Flags:", flags_list)
        
        try:
            gandalf_iterator(
                geom_file,
                cell_file,
                input_folder,
                output_base,
                threads,
                max_radius=max_radius,
                step=step,
                extra_flags=flags_list
            )
            print("Indexing completed successfully.")
        except Exception as e:
            print("Error during indexing:", e)
    
    run_button.config(command=on_run_clicked)
    
    return frame

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Gandalf Indexing GUI")
    ui = get_ui(root)
    ui.pack(fill="both", expand=True)
    root.mainloop()
