
import matplotlib.pyplot as plt

def plot_raw_intensity_hist(df_asu):
    """
    Plot a single histogram of all raw intensities, ignoring event
    but *after* we've labeled them with h_asu, k_asu, l_asu.
    """
    if df_asu.empty:
        print("[WARN] No data to plot.")
        return
    plt.figure()
    plt.hist(df_asu["I"].dropna(), bins=50, edgecolor='black')
    plt.xlabel("Raw Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Raw Intensities (All Reflections, ASU-labeled)")
    plt.show()

def plot_intensity_hist_per_event(df_asu):
    """
    Create a separate histogram for each event’s intensities.
    Useful if the number of events is small. 
    For many events, you might prefer a different approach.
    """
    all_events = df_asu["event"].unique()
    for evt in all_events:
        sub = df_asu[df_asu["event"] == evt]
        plt.figure()
        plt.hist(sub["I"].dropna(), bins=50, edgecolor='black')
        plt.xlabel("Raw Intensity")
        plt.ylabel("Frequency")
        plt.title(f"Event={evt}: histogram of intensities")
        plt.show()
