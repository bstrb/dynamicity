# **README.md**

# ICI SerialED Orchestrator

*A modular orchestrator for iterative detector-centering and shift refinement in continuous SerialED.*

---

## 📌 Overview

The **ICI Orchestrator** automates iterative detector-centering for **continuous Serial Electron Diffraction (SerialED)**.
It repeatedly:

1. Runs indexing at a seeded center
2. Extracts refined detector shifts from indexing output
3. Summarizes results and updates logs
4. Proposes new shifts based on convergence logic
5. Repeats until stable convergence criteria are met

A **GUI** (`ici_gui.py`) is included for interactive use, and a **CLI** (`ici_orchestrator.py`) is available for automation and scripting.

The orchestrator is designed to work both on:

* **Linux**
* **WSL2**
* **macOS**

and requires **CrystFEL** with built-in **indexamajig** + **XGandalf**.

---

## ✨ Features

* Full iterative refinement loop for detector centering
* Live progress monitoring (GUI) with event-level progress bar
* Tools for ring-based hillmap search
* Tools for dx/dy refinement
* Automatic log summarization and early-break file creation
* Flexible convergence criteria
* Free-text indexamajig flag configuration
* Multi-column parameter layout with tooltips
* Clean stop mechanism mid-run

---

# 📦 Installation

### 🐍 **Python version**

**Python 3.10+** is recommended.

---

## Python dependencies

All dependencies come directly from your script imports:

### **Required**

* PyQt6
* h5py
* numpy
* tqdm
* regex (Python `re` is used, but some modules may depend on `regex`)
* built-in Python modules (os, sys, subprocess, traceback, json, glob, time, pathlib, etc.)

If unsure, install the full set:

```bash
pip install pyqt6 h5py numpy tqdm regex
```

---

## External dependencies (required)

### **CrystFEL (latest recommended)**

Make sure these are in your PATH:

```bash
crystfel --version
```

---

# 📁 Input Requirements

To run the orchestrator, the user must provide:

### **1. A CrystFEL-compatible `.geom` file**

Defines detector geometry.

### **2. A CrystFEL-compatible `.cell` file**

Defines unit cell.

### **3. One or more `.h5` diffraction files**

Each containing:

```
/entry/data/images     (the raw diffraction frames)
```

Optional HDF5 fields:

* `/entry/data/peakX`, `/entry/data/peakY` (pre-found peaks)
* `/entry/data/det_shift` or other metadata

You **may** store everything inside the run-root for convenience,
but it is **not required**.

---

# 📂 Output Folder Structure

The `--run-root` directory is used only for **output**, and will contain:

```
run_root/
    run_000/
        stream files
        logs/
        summaries/
    run_001/
    run_002/
    ...
    orchestrator.log
```

Each run corresponds to one iteration of the refinement loop.

---

# 🖥 GUI Usage (ici_gui.py)

Start the GUI with:

```bash
python3 ici_gui.py
```

### **Steps:**

1. **Select Run Root**
   Where output will be written.

2. **Select .geom file**

3. **Select .cell file**

4. **Add HDF5 files** (one per line or using the Add… button)

5. **Adjust orchestrator parameters** (optional)

   * Proposal radius, min spacing, damping
   * Convergence tolerances
   * Stability thresholds
   * No-improvement detection
   * Streak-based exit conditions
     Hover your mouse to see tooltips.

6. **Set indexamajig flags**
   Example:

   ```
   --peaks=cxi -j 1 --indexing=xgandalf --no-half-pixel-shift —no-non-hits-in-stream —no-retry --integration=rings etc..
   ```
  ** Note that parallelizing is done locally so enter -j 1 in indexamajig is recommended. **

7. Click **Run orchestration**

8. Watch logs and progress bar in real time.

9. Click **Stop** at any time to safely terminate.

---

# 🖥 Command-line Usage (ici_orchestrator.py)

Example:

```bash
python3 ici_orchestrator.py \
    --run-root results/ \
    --geom detector.geom \
    --cell structure.cell \
    --h5 data1.h5 data2.h5 \
    --max-iters 20 \
    --jobs 24
```

All parameters available in CLI mirror what is in the GUI.

---

# 🔄 Orchestrator Workflow

For each iteration:

### **1. Create run directory**

```
run_000/
```

### **2. Prepare + execute indexing**

Using `run_sh.py`:

* Calls indexamajig
* Prints `__EVENT_DONE__` for GUI progress
* Writes CrystFEL stream files
* Extracts refined detector shifts from stream
* Updates event logs

### **3. Summaries and logs**

Scripts involved:

* `summarize_image_run_log.py`
* `update_image_run_log_grouped.py`
* `build_early_break_from_log.py`
* `build_overlays_and_list.py`

### **4. Propose next shifts**

From:

* `step1_hillmap_wrmsd.py` (ring-based hillmap search)
* `step2_dxdy.py` (dx/dy refinement)
* `propose_next_shifts.py` (combines logic)

### **5. Refined/optimized shift update**

Optional damping:

* λ = 0.0 → full damping
* λ = 1.0 → no damping
* Default: λ = 0.8

### **6. Convergence detection**

Triggers on:

| Criterion                     | Meaning                                                                            |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| **Recurring tolerance**       | Repeated similar shifts (stability)                                                |
| **Median relative tolerance** | Low median change across iterations                                                |
| **No-improve detection**      | No significant improvement (`noimprove_N`, `noimprove_eps`)                        |
| **Stability detection**       | Standard deviation of recent shifts across window (`stability_N`, `stability_std`) |
| **Streak rule**               | End if enough successes inside an unindexed streak                                 |

If none of these trigger, the loop continues.

---

# ⚙️ Parameter Overview

| Parameter                    | Description                                                            |
| ---------------------------- | ---------------------------------------------------------------------- |
| **radius_mm**                | Search radius (mm) used in hillmap ring search                         |
| **min_spacing_mm**           | Minimum spacing allowed between proposed shifts                        |
| **λ (lambda)**               | Damping factor for refined (non-optimized) shift updates               |
| **N_conv**                   | Minimum number of events required to judge convergence                 |
| **recurring_tol**            | Relative tolerance for recurring-shift detection                       |
| **median_rel_tol**           | Median relative tolerance threshold                                    |
| **noimprove_N**              | How many iterations with no improvement trigger no-improve rule        |
| **noimprove_eps**            | Minimum improvement needed to reset no-improve counter                 |
| **stability_N**              | Number of iterations considered when computing stability               |
| **stability_std**            | Std deviation threshold for stable shift pattern                       |
| **done_on_streak_successes** | Number of indexed events within an unindexed streak to consider ending |
| **done_on_streak_length**    | Length of failure streak required to apply the above                   |

---

# 🧪 Example Workflow

### 1. Prepare files:

```bash
geom = MFM300.geom
cell = MFM300.cell
h5 files = data/run01.h5 data/run02.h5
```

### 2. Run:

```bash
python3 ici_orchestrator.py \
    --run-root run_test/ \
    --geom MFM300.geom \
    --cell MFM300.cell \
    --h5 run01.h5 run02.h5
```

### 3. GUI example:

```bash
python3 ici_gui.py
```

Select files → adjust parameters → run.

---

# 🛠 Troubleshooting

### **CrystFEL errors**

Make sure:

```
crystfel —version
```

return valid output.

### **No images found**

Check `.h5` files contain:

```
/entry/data/images
```

### **Bad output folders**

Ensure `--run-root` is writable.

### **GUI does not start**

Install PyQt6:

```bash
pip install pyqt6
```

### **macOS note**

On macOS with Homebrew Python, sometimes:

```bash
python3 -m pip install PyQt6
```

works better than `pip`.

---

