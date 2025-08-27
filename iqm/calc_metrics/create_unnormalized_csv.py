import os
import csv
import re
import threading
import time
from tqdm import tqdm
from .extract_chunk_data import extract_chunk_data
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, cpu_count

_DELIM = "Begin chunk"
_HEADER_MARKER = "CrystFEL stream format"

def chunk_reader(path, counter):
    buf = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if _DELIM in line:
                if buf:
                    chunk = ''.join(buf); buf.clear()
                    if not chunk.lstrip().startswith(_HEADER_MARKER):
                        counter.value += 1
                        yield chunk
                # drop the delimiter line itself
            else:
                buf.append(line)
        if buf:
            chunk = ''.join(buf)
            if not chunk.lstrip().startswith(_HEADER_MARKER):
                counter.value += 1
                yield chunk

def process_stream_file_and_append_csv(stream_path, csv_path,
                                       lock,
                                       wrmsd_tol, idx_tol,
                                       counter):
    basename = os.path.basename(stream_path)
    # grab original cell‐params from preamble
    with open(stream_path, 'r', errors='replace') as f:
        header = f.read().split('----- Begin chunk -----', 1)[0]
    cp = re.search(
        r'a = ([\d.]+) A\nb = ([\d.]+) A\nc = ([\d.]+) A\n'
        r'al = ([\d.]+) deg\nbe = ([\d.]+) deg\nga = ([\d.]+) deg',
        header
    )
    original_cell = tuple(map(float, cp.groups())) if cp else None

    for chunk in chunk_reader(stream_path, counter):
        if "indexed_by = none" in chunk.lower():
            continue
        data = extract_chunk_data(chunk, original_cell,
                                  wrmsd_tolerance=wrmsd_tol,
                                  index_tolerance=idx_tol)
        if not data:
            continue
        evt, wrmsd, frac_out, Ldev, Adev, pr, pct_unidx, _ = data
        row = [basename, evt, wrmsd, frac_out, Ldev, Adev, pr, pct_unidx]

        # **append immediately** under the lock
        with lock:
            with open(csv_path, 'a', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(row)
                out.flush()
                os.fsync(out.fileno())

def create_unnormalized_csv(folder_path,
                            output_csv_name='unnormalized_metrics.csv',
                            wrmsd_tolerance=2.0,
                            index_tolerance=1.0):
    output_csv = os.path.join(folder_path, output_csv_name)
    if os.path.exists(output_csv):
        os.remove(output_csv)

    # 1) write the header once
    with open(output_csv, 'w', newline='') as f:
        csv.writer(f).writerow([
            'stream_file','event_number','weighted_rmsd',
            'fraction_outliers','length_deviation',
            'angle_deviation','peak_ratio',
            'percentage_unindexed'
        ])

    # 2) collect files + count chunks
    stream_files = sorted(
        os.path.join(folder_path, fn)
        for fn in os.listdir(folder_path)
        if fn.endswith('.stream')
    )
    if not stream_files:
        print("No .stream files found.")
        return

    total_chunks = 0
    for path in stream_files:
        d = sum(1 for l in open(path, 'r', errors='replace')
                if _DELIM in l)
        total_chunks += max(d - 1, 0)

    # 3) set up shared lock + counter + progress bar monitor
    mgr = Manager()
    lock    = mgr.Lock()
    counter = mgr.Value('i', 0)

    pbar = tqdm(total=total_chunks, desc="Chunks processed", unit="chunk")
    def monitor():
        last = 0
        while last < total_chunks:
            cur = counter.value
            if cur > last:
                pbar.update(cur - last)
                last = cur
            time.sleep(0.05)
        pbar.close()

    threading.Thread(target=monitor, daemon=True).start()

    # 4) fire off one process per file that writes *every* chunk
    with ProcessPoolExecutor(max_workers=cpu_count()) as exe:
        for path in stream_files:
            exe.submit(
                process_stream_file_and_append_csv,
                path, output_csv,
                lock,
                wrmsd_tolerance,
                index_tolerance,
                counter
            )

    print(f"Wrote unnormalized CSV → {output_csv}")
