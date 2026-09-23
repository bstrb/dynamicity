from __future__ import annotations

from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_v6_fullpop_directional_intensity_filters as builder  # noqa: E402


def write_score_cache(path: Path, hkl_count: int = 260) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE score_cache (
          ordinal INTEGER PRIMARY KEY,
          source_filename TEXT,
          event TEXT,
          h INTEGER,
          k INTEGER,
          l INTEGER,
          exact_key_text TEXT UNIQUE,
          source_order INTEGER,
          sg REAL,
          abs_sg REAL,
          Eg REAL,
          D REAL,
          U REAL,
          M REAL,
          M2 REAL
        )
        """
    )
    rows = []
    for ordinal in range(hkl_count):
        h = ordinal + 1
        k = ordinal % 7
        l = -(ordinal % 5)
        source = f"img_{ordinal % 3}.h5"
        event = str(ordinal % 11)
        exact_key = f"{source}|{event}|{h}|{k}|{l}"
        rows.append(
            (
                ordinal,
                source,
                event,
                h,
                k,
                l,
                exact_key,
                ordinal + 1,
                0.0,
                0.0,
                1.0 + (ordinal % 13) * 0.01,
                1.0,
                0.0,
                1.0,
                1.0 + (ordinal % 17) * 0.02,
            )
        )
    conn.executemany("INSERT INTO score_cache VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_populate_hkl_stats_parallel_path_uses_as_completed(tmp_path: Path) -> None:
    cache_db = tmp_path / "full_population_cache.sqlite"
    work_db = tmp_path / "work.sqlite"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_score_cache(cache_db)

    logger = builder.RunLogger(out_dir)
    work_conn = builder.connect_work(work_db)
    try:
        builder.init_work_db(work_conn)
        stats = builder.populate_hkl_stats(cache_db, work_conn, workers=2, logger=logger, progress_every=0.1)
    finally:
        work_conn.close()
        logger.close()

    assert stats["completed_hkls"] == 260
    assert stats["batch_count"] >= 2
    assert work_db.is_file()
    with sqlite3.connect(work_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hkl_stats").fetchone()[0] == 260
