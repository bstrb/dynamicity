import tkinter as tk

def create_scrollable_frame(root):
    """
    Creates a scrollable frame within the root window.
    Returns the inner frame where widgets can be added.
    """
    container = tk.Frame(root)
    container.pack(fill="both", expand=True)
    
    canvas = tk.Canvas(container)
    canvas.pack(side="left", fill="both", expand=True)
    
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Create a frame inside the canvas.
    scrollable_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    
    # Update scrollregion after every widget is added.
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", on_configure)
    
    return scrollable_frame