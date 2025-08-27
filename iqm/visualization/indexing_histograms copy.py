import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enables 3D plotting
import matplotlib.colors as mcolors

def plot_indexing_rate(folder_path):
    os.chdir(folder_path)
    
    # Find all files with the ".stream" extension
    stream_files = glob.glob("*.stream")
    data = []
    
    for stream_file in stream_files:
        base_name = os.path.splitext(stream_file)[0]
        parts = base_name.rsplit("_", 2)
        x = float(parts[-2])
        y = float(parts[-1])
        
        # Count occurrences of "num_reflections" and "num_peaks"
        event_count = 0
        total_count = 0
        with open(stream_file, "r") as f:
            for line in f:
                if line.startswith("num_reflections"):
                    event_count += 1
                if line.startswith("num_peaks"):
                    total_count += 1
        percentage = event_count / total_count * 100 if total_count else 0
        data.append((x, y, percentage))
    
    df = pd.DataFrame(data, columns=["x", "y", "count"])
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize percentage values for the colormap
    norm = mcolors.Normalize(vmin=df["count"].min(), vmax=df["count"].max())
    cmap = plt.cm.viridis
    
    # Define bar widths
    dx = dy = 0.07
    z_base = 0  # bars start at z = 0
    
    # Plot each bar individually with a color corresponding to its percentage
    for _, row in df.iterrows():
        x_val = row["x"]
        y_val = row["y"]
        dz = row["count"]
        color = cmap(norm(dz))
        ax.bar3d(x_val, y_val, z_base, dx, dy, dz,
                 color=color, shade=True, alpha=0.95)
    
    # Add a colorbar for reference
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(df["count"])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label("Indexing Rate (%)")
    
    # Set labels and title
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Indexing Rate (%)")
    ax.set_title("3D Bar Plot of Indexing Rate at Each (x, y) Coordinate")
    
    # Adjust the viewing angle
    ax.view_init(elev=25, azim=135)
    
    # Optionally, adjust background pane fills
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    
    plt.show()

    # Create a scatter plot with color representing indexing rate (%)
    plt.figure()
    scatter = plt.scatter(
        df["x"], 
        df["y"], 
        c=df["count"],
        cmap="viridis",
        alpha=0.9,
        s=150
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Indexing Rate (%)")

    plt.title("Indexing Rate (%) at Each (x, y) File Coordinate")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    input_folder = "/home/bubl3932/files/LTA_sim/simulation-45/xgandalf_iterations_max_radius_2_step_0.5"
    plot_indexing_rate(input_folder)
