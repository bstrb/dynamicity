import numpy as np

def match_peaks_to_reflections(peaks, reflections, tolerance=1.0):
    """
    Matches detected peaks to indexed reflections within a specified tolerance.
    Args:
        peaks (list of tuples): List of (fs, ss) coordinates of detected peaks.
        reflections (list of tuples): List of (fs, ss) coordinates of indexed reflections.
        tolerance (float): Matching tolerance in pixels.
    Returns:
        int: Number of matched peaks.
    """
    matched_peaks_indices = set()
    for i, (peak_fs, peak_ss) in enumerate(peaks):
        for refl_fs, refl_ss in reflections:
            distance = np.hypot(peak_fs - refl_fs, peak_ss - refl_ss)
            if distance <= tolerance:
                matched_peaks_indices.add(i)
                break  # Stop checking reflections once a match is found
    return len(matched_peaks_indices)
