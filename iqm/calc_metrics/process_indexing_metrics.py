# -------------------------
# Automate Evaluation and Integration
# -------------------------

import traceback
from .append_event_count import append_event_count
from .create_unnormalized_csv import create_unnormalized_csv
from .normalize_csv import normalize_csv

def process_indexing_metrics(stream_file_folder: str,
                        wrmsd_tolerance: float=2.0,
                        indexing_tolerance: float=1.0,
                        normalization_method: str='zscore'
                        ):

    """
    Process a stream file to extract and normalize chunk metrics.

    This function reads a stream file from the given file path, extracts data chunks
    (delimited by "----- Begin chunk -----"), and computes several metrics for each chunk.
    It filters out chunks that are unindexed (i.e., those containing "indexed_by = none").
    For each valid chunk, the function collects metrics such as weighted RMSD, length deviation,
    angle deviation, and others (including a new metric: percentage_indexed). It then normalizes
    these metrics either based on min/max values or zscore.

    Parameters:
        stream_file_path (str):
            The full path to the stream file that contains the data chunks.
        wrmsd_tolerance (float):
            The number of standard deviations away from the mean weighted RMSD for a chunk to be
            considered an outlier. Default factor is 2.0.
        indexing_tolerance (int):
            The maximum deviation in pixels between observed and predicted peak positions for
            a peak to be considered indexed. Default is 1.0 pixel.
        normalization_method (str):
            The method used to normalize the metrics. Options are 'minmax' and 'zscore'. Default
            is 'zscore'.
    """
    
    append_event_count(stream_file_folder)

    try:
        print(f"Evaluating multiple stream files")
        create_unnormalized_csv(stream_file_folder, wrmsd_tolerance=wrmsd_tolerance, index_tolerance=indexing_tolerance)
        normalize_csv(stream_file_folder, normalization_method=normalization_method)
    except Exception as e:
        print(f"An error occurred during processing the stream files: {e}")
        traceback.print_exc()
