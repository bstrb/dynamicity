# ================================================================
# file: sered_scaler/io/stream_to_dataframe.py
# ================================================================

"""Read a CrystFEL *.stream* file into three tidy *pandas* DataFrames.

Returned tuple *(header_df, peaks_df, reflections_df)* mirrors the names
used throughout the scaling pipeline.  For gigantically large streams
(>10⁷ spots) pass *return_dask=True* to get **dask.dataframe** objects
instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import pandas as pd

try:
    import dask.dataframe as dd
except ModuleNotFoundError:  # dask optional
    dd = None  # type: ignore

# local import – expecting the lightweight parser next to this file
from .parse_crystfel_stream import StreamParser  # type: ignore

__all__ = ["stream_to_dfs"]

def stream_to_dfs(
    stream_path: Union[str, Path],
    *,
    return_dask: bool = False,
) -> Tuple[
    pd.DataFrame,
    Union[pd.DataFrame, "dd.DataFrame"],
    Union[pd.DataFrame, "dd.DataFrame"],
]:
    """Parse *stream_path* and return DataFrames.

    Parameters
    ----------
    stream_path
        Path to *.stream* file.
    return_dask
        If True, convert the *peaks* and *reflections* tables to
        ``dask.dataframe`` for out-of-core scaling.
    """
    parser = StreamParser(stream_path)
    parser.parse()

    # 1) Header: single-row DataFrame from the header object
    header_df = pd.DataFrame([vars(parser.header)])

    # 2) Peaks: flatten frames → peaks into a list of dicts, then DataFrame
    peaks_records: list[dict] = []
    for frame in parser.frames:
        evt = frame.event
        for peak in frame.peaks:
            rec = {"event": evt}
            rec.update(vars(peak))
            peaks_records.append(rec)
    peaks_df = pd.DataFrame(peaks_records)

    # 3) Reflections: same approach
    refl_records: list[dict] = []
    for frame in parser.frames:
        evt = frame.event
        for refl in frame.reflections:
            rec = {"event": evt}
            rec.update(vars(refl))
            refl_records.append(rec)
    reflections_df = pd.DataFrame(refl_records)

    # 4) Optionally convert to dask
    if return_dask:
        if dd is None:
            raise ImportError("dask is not installed but return_dask=True was requested")
        peaks_df = dd.from_pandas(peaks_df, npartitions=32)    # type: ignore
        reflections_df = dd.from_pandas(reflections_df, npartitions=64)  # type: ignore

    return header_df, peaks_df, reflections_df
