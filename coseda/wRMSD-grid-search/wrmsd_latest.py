# wrmsd_latest.py
# Compute wRMSD for ALL chunks in ONE CrystFEL .stream (latest run).
from __future__ import annotations
import math, os, re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# -------- Regex (lifted & simplified from your script) --------
FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

RE_BEGIN_CHUNK   = re.compile(r"^-{5}\s+Begin chunk\s+-{5}")
RE_END_CHUNK     = re.compile(r"^-{5}\s+End(?: of)? chunk\s+-{5}")

RE_IMG_FN        = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
# accept "Event: //123" or "Event: 123"
RE_EVENT         = re.compile(r"^\s*Event:\s*(?:/+)?\s*([0-9]+)\s*$", re.IGNORECASE)
RE_IMG_SERIAL    = re.compile(r"^\s*Image\s+serial\s+number\s*:\s*([0-9]+)\s*$", re.IGNORECASE)

RE_BEGIN_PEAKS   = re.compile(r"^\s*Peaks from peak search", re.IGNORECASE)
RE_END_PEAKS     = re.compile(r"^\s*End of peak list", re.IGNORECASE)
RE_PEAK_LINE     = re.compile(rf"^\s*({FLOAT_RE})\s+({FLOAT_RE})\s+{FLOAT_RE}\s+({FLOAT_RE})\s+(\S+)\s*$")

RE_BEGIN_CRYSTAL = re.compile(r"^\s*---\s+Begin crystal", re.IGNORECASE)
RE_END_CRYSTAL   = re.compile(r"^\s*---\s+End crystal", re.IGNORECASE)

RE_BEGIN_REFL    = re.compile(r"^\s*Reflections measured after indexing", re.IGNORECASE)
RE_END_REFL      = re.compile(r"^\s*End of reflections", re.IGNORECASE)
RE_REFL_LINE     = re.compile(rf".*?\s({FLOAT_RE})\s+({FLOAT_RE})\s+(\S+)\s*$")

class _ChunkLite:
    __slots__ = ("image", "event", "peaks", "refls")
    def __init__(self):
        self.image: Optional[str] = None
        self.event: Optional[str] = None
        self.peaks: List[Tuple[float, float, float, str]] = []
        self.refls: List[Tuple[float, float, str]] = []

# ---------- wRMSD (peaks↔reflections, per panel, sigma-clipped) ----------
def _nn_dists_numpy(pfs, pss, rfs, rss) -> np.ndarray:
    if pfs.size == 0 or rfs.size == 0:
        return np.full(pfs.shape[0], np.inf, dtype=np.float32)
    df = pfs[:, None] - rfs[None, :]
    ds = pss[:, None] - rss[None, :]
    return np.sqrt((df * df + ds * ds).min(axis=1)).astype(np.float32, copy=False)

def _sigma_mask(md: np.ndarray, sigma: float) -> np.ndarray:
    if md.size == 0:
        return np.zeros((0,), dtype=bool)
    mu = float(md.mean())
    sd = float(md.std(ddof=1)) if md.size > 1 else 0.0
    if sd == 0.0:
        return np.ones_like(md, dtype=bool)
    return md <= (mu + sigma * sd)

def compute_wrmsd_details(
    peaks: List[Tuple[float, float, float, str]],
    refls: List[Tuple[float, float, str]],
    match_radius: float,
    outlier_sigma: float
) -> Tuple[Optional[float], int, int, Optional[str]]:
    # Map by panel
    pmap, rmap = {}, {}
    for fs, ss, inten, pan in peaks:
        lst = pmap.setdefault(pan, [[], [], []])
        lst[0].append(fs); lst[1].append(ss); lst[2].append(inten)
    for fs, ss, pan in refls:
        lst = rmap.setdefault(pan, [[], []])
        lst[0].append(fs); lst[1].append(ss)

    if not pmap or not rmap:
        return None, 0, 0, "too_few_peaks_or_refl"

    n_matches = 0
    md_all, w_all = [], []

    for pan, (pfs_list, pss_list, pint_list) in pmap.items():
        r = rmap.get(pan)
        if r is None: 
            continue
        rfs_list, rss_list = r
        if not pfs_list or not rfs_list:
            continue

        pfs = np.asarray(pfs_list, dtype=np.float32)
        pss = np.asarray(pss_list, dtype=np.float32)
        pint= np.asarray(pint_list, dtype=np.float32)
        rfs = np.asarray(rfs_list, dtype=np.float32)
        rss = np.asarray(rss_list, dtype=np.float32)

        d = _nn_dists_numpy(pfs, pss, rfs, rss)
        within = (d <= float(match_radius))
        cnt = int(within.sum())
        if cnt > 0:
            n_matches += cnt
            md_all.append(d[within])
            w_all.append(pint[within])

    if not md_all:
        return None, n_matches, 0, "no_matches"

    md = np.concatenate(md_all)
    w  = np.concatenate(w_all)
    keep = _sigma_mask(md, float(outlier_sigma))
    n_kept = int(keep.sum())
    if n_kept == 0:
        return None, n_matches, 0, "all_clipped"

    kept = md[keep]
    kw   = w[keep]
    wsum = float(kw.sum())
    if wsum <= 0.0:
        return None, n_matches, n_kept, "zero_weight"

    wr = math.sqrt(float((kw * (kept ** 2)).sum()) / wsum)
    if math.isnan(wr) or math.isinf(wr):
        return None, n_matches, n_kept, "nan_inf"
    return float(wr), n_matches, n_kept, None

# ------------- Iterate chunks, parse peaks/refls quickly ---------------
def iter_chunks_lite(stream_path: str) -> Iterable[_ChunkLite]:
    sp = Path(stream_path)
    def dec(b: bytes) -> str:
        return b.decode("utf-8", errors="ignore").rstrip("\r\n")

    with sp.open("rb") as fb:
        in_chunk = in_peaks = in_crystal = in_refl = False
        ch: Optional[_ChunkLite] = None

        while True:
            raw = fb.readline()
            if not raw:
                break
            line = dec(raw)

            if RE_BEGIN_CHUNK.search(line):
                in_chunk = True
                in_peaks = in_crystal = in_refl = False
                ch = _ChunkLite()
                continue
            if not in_chunk:
                continue

            if RE_END_CHUNK.search(line):
                if ch and ch.image and ch.event:
                    yield ch
                in_chunk = in_peaks = in_crystal = in_refl = False
                ch = None
                continue

            if ch is None:
                continue

            if ch.image is None:
                m = RE_IMG_FN.match(line)
                if m:
                    ch.image = m.group(1).strip()
                    continue

            if ch.event is None:
                m = RE_EVENT.match(line) or RE_IMG_SERIAL.match(line)
                if m:
                    ch.event = m.group(1).strip()
                    continue

            if RE_BEGIN_PEAKS.search(line):
                in_peaks = True;  continue
            if in_peaks:
                if RE_END_PEAKS.search(line):
                    in_peaks = False;  continue
                mpk = RE_PEAK_LINE.match(line)
                if mpk:
                    fs = float(mpk.group(1)); ss = float(mpk.group(2))
                    inten = float(mpk.group(3)); panel = mpk.group(4)
                    ch.peaks.append((fs, ss, inten, panel))
                continue

            if RE_BEGIN_CRYSTAL.search(line):
                in_crystal = True;  continue
            if in_crystal and RE_END_CRYSTAL.search(line):
                in_crystal = False; in_refl = False;  continue
            if in_crystal and RE_BEGIN_REFL.search(line):
                in_refl = True;  continue
            if in_refl:
                if RE_END_REFL.search(line):
                    in_refl = False;  continue
                mrf = RE_REFL_LINE.match(line)
                if mrf:
                    fs = float(mrf.group(1)); ss = float(mrf.group(2)); pan = mrf.group(3)
                    ch.refls.append((fs, ss, pan))
                continue

# ------------- Public: compute for all chunks in one stream ------------
def wrmsd_all_chunks_in_stream(
    stream_path: str,
    match_radius: float = 4.0,
    outlier_sigma: float = 2.0
) -> Dict[Tuple[str, str], dict]:
    """
    Return dict keyed by (overlay_basename, event_str) -> metrics dict.
    overlay_basename comes from 'Image filename:' (basename only).
    """
    out: Dict[Tuple[str, str], dict] = {}
    for ch in iter_chunks_lite(stream_path):
        img_base = os.path.basename(ch.image or "")
        ev = str(ch.event or "")
        wr, n_matches, n_kept, reason = compute_wrmsd_details(
            ch.peaks, ch.refls, match_radius, outlier_sigma
        )
        out[(img_base, ev)] = {
            "wrmsd": wr, "n_peaks": len(ch.peaks), "n_reflections": len(ch.refls),
            "n_matches": n_matches, "n_kept": n_kept, "reason": reason
        }
    return out
