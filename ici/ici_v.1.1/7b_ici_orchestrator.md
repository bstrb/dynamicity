# 7b. ICI Orchestrator (experimental)

> **Status:** Experimental / non-core (ICI = iterative center shift indexing loop).
>
> This module automates iterative detector-shift refinement by repeatedly:
> 1) indexing at a proposed detector shift, 2) extracting a quality metric (wRMSD),
> 3) proposing the next shift using a 2-stage policy (global hillmap search + local dx/dy refinement(only applicable when crystFEL refine is not disabled through --no-refine)),
> 4) stopping when per-event convergence criteria are met.

This workflow is implemented as a **run-loop** that creates `run_###` folders, runs per-event indexing jobs, ingests per-run metrics into a grouped log, proposes next shifts, and repeats.

---

## What it is (conceptual map)

The orchestrator manages *events* (frames) across one or more HDF5 datasets. For each event, it maintains:

- a **trial history**: `(run_n, det_shift_x_mm, det_shift_y_mm, success?, wrmsd?)`
- a **proposal history**: `(next_run_n, next_dx_mm, next_dy_mm, reason)`
- a **done state**: `next_dx_mm = done` and `next_dy_mm = done` (per event)

The “brain” of next-shift selection is `propose_next_shifts.py`, which reads the grouped `image_run_log.csv` and writes proposals back into it (and also to a JSON sidecar).

---

## Directory layout and artifacts

The run root is an output workspace. Typical structure:

```
run_root/
  run_000/
    event_000000/
      sh_000000.sh
      idx.stdout / idx.stderr
      stream_000000.stream
      per_frame_dx_dy.csv   (optional, for Step-2)
      mille-data.bin        (optional)
    event_000001/
    ...
    stream_000.stream       (concatenated)
    missing_event_parts.txt (optional)
    chunk_metrics_000.csv   (produced upstream of log ingest)
  run_001/
  ...
  image_run_log.csv
  image_run_state.json
  orchestrator.log (if enabled by wrapper)
```

Event-level parallel execution + concatenation is handled by `run_sh.py`.

---

## Inputs

### Required
- `--run-root`: output folder where `run_###` and logs are written.
- A grouped run log: `image_run_log.csv` (created/extended each iteration).

### Upstream/implicit requirements
- Per-run metrics file `chunk_metrics_###.csv` inside the latest `run_###` folder, used to ingest new trials into `image_run_log.csv`.
- For Step-2 dx/dy refinement: `per_frame_dx_dy.csv` inside the *event directory of a successful index run*.
- CrystFEL tooling (indexamajig + indexing backend) used by the event `.sh` scripts (invoked by `run_sh.py`).

---

## Outputs

### 1) `image_run_log.csv` (grouped by event)
This is the central “living ledger”.

- It is grouped into sections, one per event:

  ```
  #/abs/path/to/file.h5 event 123
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm,next_reason
  ...
  ```

- `update_image_run_log_grouped.py` ingests the newest `chunk_metrics_###.csv` into the *correct event sections*, avoiding duplicates.
- `propose_next_shifts.py` patches the **last row** of each event section to fill `next_dx_mm,next_dy_mm,next_reason` for the *next* run.

**Field meanings (as implemented):**
- `run_n`: iteration index
- `det_shift_x_mm`, `det_shift_y_mm`: the shift used for that run/event
- `indexed`: **sticky** per-event “ever indexed” flag (1 after any successful run)
- `wrmsd`: per-run wRMSD (blank / None when not indexed)
- `next_dx_mm`, `next_dy_mm`: next proposal in **mm**, or `done` when converged
- `next_reason`: a short reason code (e.g. `step1_*`, `dxdy_*`, `done_*`)

### 2) `image_run_state.json` (sidecar)
A per-event state cache:
- trials history (`trials`)
- proposal history (`proposal_history`)
- sticky `ever_indexed`
- last processed global run (`last_global_run`)

This is written by `propose_next_shifts.py` and is also used by summary tools to detect DONE more reliably.

### 3) Per-run, per-event indexing artifacts
Created by event `.sh` scripts and executed in parallel by `run_sh.py`:
- `idx.stdout`, `idx.stderr`
- per-event `stream_*.stream`
- optional `mille-data.bin` and extracted mille outputs
- `stream_<run>.stream` (concatenated run-level stream)

Index success detection uses:
- `idx.stderr` “indexable” parsing OR stream `num_reflections` fallback.

---

## Proposal algorithm (core logic)

The proposal policy is defined in `propose_next_shifts.py`.
It operates on **each event independently**, using that event’s trial history.

### Step-1 (global exploration): Hillmap weighted by wRMSD
Implemented by `step1_hillmap_wrmsd.py`.

- Samples candidate points uniformly in a disk of radius `radius_mm`
- Applies:
  - a base Gaussian centered on the first-attempt center
  - **positive “hills”** around successful centers, weighted by a Boltzmann score in wRMSD:
    - lower wRMSD ⇒ stronger hill ⇒ higher sampling probability
  - **negative “drops”** around failed centers
- Enforces a minimum spacing `min_spacing_mm` to avoid retrying near-identical shifts
- Returns a proposed `(x_mm, y_mm)` and a reason like `step1_hillmap_wrmsd_sample`

### Step-2 (local refinement): dx/dy refined center (strict)
Implemented by `step2_dxdy.py`.

- Reads `per_frame_dx_dy.csv` in the event directory (expected columns `dx`, `dy`)
- Uses the **last row** as the refined update
- **Strict contract:** if missing/invalid, it returns `(None, None, reason)` with no internal fallbacks

### Switching logic (Option A)
In `propose_event()`:
- If the *last run indexed*, attempt Step-2 dxdy:
  - If dxdy missing ⇒ fall back to Step-1 weighted proposal
  - If dxdy present ⇒ apply a damped refinement:
    - `next = last - λ * (dxdy)` (λ is `--damping-factor`)
- If the *last run did not index*:
  - If never indexed ⇒ Step-1 “pure exploration” (no success hills exist yet)
  - If indexed before ⇒ Step-1 weighted hillmap (uses success hills + failure penalties)

---

## Convergence and DONE rules (per event)

An event is marked DONE by returning `(None, None, "done_*")`, causing:
- `next_dx_mm = done`
- `next_dy_mm = done`

Key DONE triggers include:
- **Long unindexed streak after having successes**
  - `--done-on-streak-successes`
  - `--done-on-streak-length`
- **No improvement in best wRMSD**
  - `--noimprove-N`
  - `--noimprove-eps` (relative improvement threshold)
- **wRMSD stability**
  - `--stability-N`
  - `--stability-std` (relative std threshold)
- Additional recurrence/median logic used during dxdy refinement:
  - `--N-conv`
  - `--recurring-tol`
  - `--median-rel-tol`

---

## Key parameters (CLI-level)

These are the knobs exposed by `propose_next_shifts.py` (and typically surfaced through wrappers/GUI).

### Geometry / search space
- `--radius-mm` (float, default 0.05): search radius for Step-1
- `--min-spacing-mm` (float, default 0.0005): minimum spacing between candidate centers
- `--seed` (int, default 1337): base seed; event-level RNG is derived stably per (h5,event,seed)

### Step-1 hillmap knobs
- `--step1-A0` (default 2.0)
- `--step1-hill-amp-frac` (default 5.0)
- `--step1-drop-amp-frac` (default 0.1)
- `--step1-candidates` (default 8192)
- `--step1-explore-floor` (default 1e-5)
- `--step1-allow-spacing-relax` (flag)
- `--beta` (default 10.0): Boltzmann sharpness for wRMSD weighting

### Convergence knobs
- `--done-on-streak-successes` (default 2)
- `--done-on-streak-length` (default 5)
- `--noimprove-N` (default 2)
- `--noimprove-eps` (default 0.02)
- `--stability-N` (default 3)
- `--stability-std` (default 0.05)
- `--N-conv` (default 3)
- `--recurring-tol` (default 0.1)
- `--median-rel-tol` (default 0.1)

### Step-2 knobs
- `--step2-algorithm` (`dxdy` or `none`, default `dxdy`)
- `--damping-factor` (λ, default 0.8)

---

## Run execution (event parallelism)

Event-level indexing is executed by `run_sh.py`:

- Discovers `event_*` directories and runs the single `sh_*.sh` inside each
- Runs in parallel via `ProcessPoolExecutor`
- Prints `__EVENT_DONE__` after each event finishes (used by GUI progress)
- Optionally extracts mille shifts if `mille-data.bin` exists
- Concatenates per-event stream parts into `stream_###.stream`
- Writes `missing_event_parts.txt` if expected stream parts are missing

---

## Summaries / monitoring

`summarize_image_run_log.py` provides a compact per-run summary from `image_run_log.csv` + sidecar, including:
- first-run index rate and wRMSD stats
- cumulative best-per-event index rate and wRMSD mean/median (prev vs current)
- counts of proposals categorized by reason (`step1*` vs `dxdy*`)
- “done events” count/fraction and their wRMSD stats

---

## How to run (minimal example)

### Propose next shifts (typically once per iteration)
```bash
python3 propose_next_shifts.py   --run-root /path/to/run_root   --radius-mm 0.05   --min-spacing-mm 0.0005   --step2-algorithm dxdy   --damping-factor 0.8
```
This patches `image_run_log.csv` and updates `image_run_state.json`.

### Summarize current state
```bash
python3 summarize_image_run_log.py --run-root /path/to/run_root
```

---

## Notes / gotchas (implementation realities)

- **Units:** Step-2 dx/dy reads `per_frame_dx_dy.csv` and multiplies by `1e3` before returning mm. This implies the CSV values are expected in a non-mm base unit; confirm your upstream generator matches this expectation.
- **“indexed” meaning:** In the grouped log ingestion, `indexed` is “ever indexed” (sticky), while `wrmsd` is per-run. In proposals, “indexed_this_run” is determined by whether the last run has a finite wRMSD.
- **DONE is per-event:** The run loop can keep iterating even while many events are already done; those are skipped/protected by `done/done` detection.
