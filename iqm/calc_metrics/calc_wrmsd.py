import numpy as np

def calc_wrmsd(fs_ss, intensities, ref_fs_ss, tolerance_factor: float=2.0):
    """
    Calculate intensity-weighted RMSD (root-mean-square deviation) between
    observed peaks and reference positions, excluding outliers beyond a 
    specified tolerance factor. Also returns the fraction of outliers.
    
    Args:
        fs_ss (array-like): Array of (fs, ss) tuples for observed peaks.
        intensities (array-like): Corresponding intensities for the observed peaks.
        ref_fs_ss (array-like): Array of (fs, ss) tuples for reference positions.
        tolerance_factor (float, optional): Multiplicative factor for the standard 
            deviation to determine which peaks are considered outliers. 
            Defaults to 2.0.
            
    Returns:
        weighted_rmsd (float): Intensity-weighted RMSD using only the inlier peaks.
        fraction_outliers (float): Fraction of peaks considered outliers.
    """

    # Ensure inputs are non-empty
    if len(fs_ss) == 0 or len(ref_fs_ss) == 0:
        print("Warning: Empty input for peaks or reflections.")
        # No data to compare, return INF RMSD and 0% outliers
        return float('inf'), 0.0

    distances = []
    weights = []

    # Calculate the minimum distance for each peak to a reflection
    for (fs, ss), intensity in zip(fs_ss, intensities):
        min_distance = float('inf')
        for ref_fs, ref_ss in ref_fs_ss:
            distance = np.sqrt((fs - ref_fs) ** 2 + (ss - ref_ss) ** 2)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
        weights.append(intensity)

    # Convert to numpy arrays for easier manipulation
    distances = np.array(distances)
    weights = np.array(weights)

    # Handle cases where distances or weights are empty or invalid
    if len(distances) == 0 or len(weights) == 0:
        return float('inf'), 0.0

    # Filter out invalid weights (e.g., zero or negative values)
    valid_indices = weights > 0
    distances = distances[valid_indices]
    weights = weights[valid_indices]

    if len(distances) == 0:  # Check again after filtering
        print("Warning: No valid weights after filtering.")
        return float('inf'), 0.0

    # Calculate the mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # If there's no variation in distances, print a warning
    if std_distance == 0:
        print("Warning: Standard deviation of distances is zero.")
        # return float('inf'), 0.0

    # Identify inliers based on the tolerance factor
    inliers = distances < (mean_distance + tolerance_factor * std_distance)
    fraction_inliers = np.sum(inliers) / len(distances)
    fraction_outliers = 1.0 - fraction_inliers

    # If no inliers exist, return infinite RMSD and 100% outliers
    if not np.any(inliers):
        print("Warning: No inliers found.")
        return float('inf'), 1.0

    # Calculate intensity-weighted RMSD using only inliers
    total_rmsd = np.sum((distances[inliers] ** 2) * weights[inliers])
    total_weight = np.sum(weights[inliers])
    weighted_rmsd = np.sqrt(total_rmsd / total_weight) if total_weight > 0 else float('inf')

    # Return (weighted_rmsd, fraction_outliers)
    return weighted_rmsd, fraction_outliers
