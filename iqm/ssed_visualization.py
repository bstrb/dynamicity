import tkinter as tk
from tkinter import filedialog, messagebox
import os
from visualization.indexing_histograms import plot_indexing_rate

def get_ui(parent):
    frame = tk.Frame(parent)
    
    # Description text for the user.
    description = (
    "Select a folder with indexing output stream files from Gandalf iterations with shifted centers. "
    "Clicking 'Generate Visualizations' will extract the x and y shift values from the filenames, calculate "
    "the indexing rate as the percentage of indexed frames (num_reflections / num_peaks * 100), and generate:\n\n"
    " - A 3D bar plot (bar height = indexing rate in %)\n"
    " - A 2D scatter (heatmap-style) plot (color intensity = indexing rate in %)"
    )

    
    # Display the description in a Label (or use a Text widget if you need scrollable text).
    desc_label = tk.Label(frame, text=description, justify=tk.LEFT, wraplength=500)
    desc_label.pack(pady=10, padx=10)
    
    # Folder selection section.
    folder_frame = tk.Frame(frame)
    folder_frame.pack(padx=10, pady=5, fill="x")
    
    tk.Label(folder_frame, text="Stream File Folder:").pack(side=tk.LEFT)
    folder_var = tk.StringVar()
    folder_entry = tk.Entry(folder_frame, textvariable=folder_var, width=50)
    folder_entry.pack(side=tk.LEFT, padx=5)
    
    def browse_folder():
        folder = filedialog.askdirectory(title="Select Output Folder", initialdir=os.getcwd())
        if folder:
            folder_var.set(folder)
    tk.Button(folder_frame, text="Browse", command=browse_folder).pack(side=tk.LEFT, padx=5)
    
    # Button to trigger visualization.
    def on_vis_clicked():
        output_folder = folder_var.get()
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return
        print("Generating visualizations for output folder:", output_folder)
        try:
            plot_indexing_rate(output_folder)
            print("Visualization completed successfully.")
        except Exception as e:
            print("Error during visualization:", e)
    
    tk.Button(frame, text="Generate Visualizations", command=on_vis_clicked, bg="lightblue").pack(pady=10)
    
    return frame

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Visualization GUI")
    ui = get_ui(root)
    ui.pack(fill="both", expand=True)
    root.mainloop()
