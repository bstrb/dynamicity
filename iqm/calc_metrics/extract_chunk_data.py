import re
from .calc_wrmsd import calc_wrmsd
from .calculate_cell_deviation import calculate_cell_deviation
from .match_peaks_to_reflections import match_peaks_to_reflections

def extract_chunk_data(chunk,
                       original_cell_params,
                       wrmsd_tolerance: float = 2.0,
                       index_tolerance: float = 1.0):
    """
    Extract metrics from a single chunk of a stream file.

    Returns a tuple of:
      (event_string, weighted_rmsd, fraction_outliers,
       length_deviation, angle_deviation, peak_ratio,
       fraction_unindexed, chunk)
    or None if any required metric is missing.
    """
    # 1) Initialize all outputs to None
    event_string = None
    weighted_rmsd = None
    fraction_outliers = None
    length_deviation = None
    angle_deviation = None
    peak_ratio = None
    fraction_unindexed = None

    # 2) Parse event number
    m_evt = re.search(r'Event:\s*//\s*(\d+(?:-\d+)?)', chunk)
    if m_evt:
        event_string = m_evt.group(1)
    else:
        print("No event number found in chunk.")

    # 3) Parse peaks
    fs_ss = []
    intensities = []
    m_peaks = re.search(r'Peaks from peak search(.*?)End of peak list', chunk, re.S)
    if m_peaks:
        peaks = re.findall(
            r'\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0',
            m_peaks.group(1)
        )
        for fs, ss, d, I in peaks:
            fs_ss.append((float(fs), float(ss)))
            intensities.append(float(I))
        if not peaks:
            print("No peaks found in chunk.")
    else:
        print("No peak list found in chunk.")

    # 4) Parse reflections
    ref_fs_ss = []
    m_refl = re.search(r'Reflections measured after indexing(.*?)End of reflections', chunk, re.S)
    if m_refl:
        refls = re.findall(
            r'\s+-?\d+\s+-?\d+\s+-?\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0',
            m_refl.group(1)
        )
        for fs, ss in refls:
            ref_fs_ss.append((float(fs), float(ss)))
        if not refls:
            print("No reflections found in chunk.")
    else:
        print("No reflections section found in chunk.")

    # 5) Calculate weighted RMSD and outliers
    if fs_ss and ref_fs_ss:
        try:
            weighted_rmsd, fraction_outliers = calc_wrmsd(
                fs_ss, intensities, ref_fs_ss,
                tolerance_factor=wrmsd_tolerance
            )
        except Exception:
            print("Error computing weighted RMSD for chunk.")
    else:
        print("Unable to calculate weighted RMSD for chunk.")

    # 6) Cell params and deviation
    cell_params = None
    m_cell = re.search(
        r'Cell parameters ([\d.]+) ([\d.]+) ([\d.]+) nm, ([\d.]+) ([\d.]+) ([\d.]+) deg',
        chunk
    )
    if m_cell:
        a_nm, b_nm, c_nm, al, be, ga = m_cell.groups()
        cell_params = (
            float(a_nm) * 10, float(b_nm) * 10, float(c_nm) * 10,
            float(al), float(be), float(ga)
        )
    else:
        print("No cell parameters found in chunk.")

    if cell_params and original_cell_params:
        try:
            length_deviation, angle_deviation = calculate_cell_deviation(
                cell_params, original_cell_params
            )
        except Exception:
            print("Error computing cell deviation for chunk.")
    else:
        print("Unable to calculate cell deviation for chunk.")

    # 7) Counts and peak ratio
    m_np = re.search(r'num_peaks\s*=\s*(\d+)', chunk)
    num_peaks = int(m_np.group(1)) if m_np else len(fs_ss)
    if num_peaks == 0:
        print("No peaks count found in chunk.")

    m_nr = re.search(r'num_reflections\s*=\s*(\d+)', chunk)
    num_refl = int(m_nr.group(1)) if m_nr else len(ref_fs_ss)
    if num_refl == 0:
        print("No reflections count found in chunk.")

    peak_ratio = (num_refl / num_peaks) if num_peaks > 0 else num_refl

    # 8) Fraction indexed/unindexed
    if fs_ss and ref_fs_ss:
        try:
            matched = match_peaks_to_reflections(fs_ss, ref_fs_ss, tolerance=index_tolerance)
            frac_indexed = matched / len(fs_ss)
            fraction_unindexed = 1 - frac_indexed
        except Exception:
            print("Error computing indexed fraction for chunk.")
    else:
        print("Unable to calculate percentage of peaks indexed for chunk.")

    # 9) Bail if any required metric is missing
    required = [
        event_string, weighted_rmsd, fraction_outliers,
        length_deviation, angle_deviation, peak_ratio,
        fraction_unindexed
    ]
    if any(x is None for x in required):
        return None

    # 10) Return complete data
    return (
        event_string,
        weighted_rmsd,
        fraction_outliers,
        length_deviation,
        angle_deviation,
        peak_ratio,
        fraction_unindexed,
        chunk
    )
