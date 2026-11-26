# PaleoSTeHM Dashboard

<!-- Zenodo DOI badge: uncomment and update after publishing the first GitHub Release -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Interactive Streamlit dashboard and modular spatiotemporal Gaussian Process (STGP) framework for Holocene relative sea-level (RSL) reconstruction, with optional Glacial Isostatic Adjustment (GIA) priors, kernel decomposition, sea-level budget analysis, and sensitivity testing.

Built on the updated **PSTHM** core library (originally developed in [radical-collaboration/PaleoSTeHM](https://github.com/radical-collaboration/PaleoSTeHM)), paired with the modular `stgp_pipeline`, `gia_covariance_tools`, and `stgp_dashboard` packages (source-tree release `v0.1.0`).

---

## Quick Overview of the Workflow

1. **Install Miniforge** (recommended) or Miniconda — one-time setup.
2. **Download this repository** and **create the `paleostehm` conda environment** — one-time setup.
3. **Launch the dashboard** with `python -m stgp_dashboard` — opens automatically in your web browser.
4. **Re-run anytime later** by opening your terminal, `cd`-ing into the project folder, activating `paleostehm`, and running `python -m stgp_dashboard`.

> **GPU is optional.** Standard CPU training works smoothly on both macOS and Windows laptops for typical Holocene datasets.

---

## Step 1: Install Miniforge (Recommended) or Miniconda

We recommend **Miniforge** because it is completely free, preconfigured to use the community-maintained `conda-forge` channel, and avoids Anaconda commercial Terms of Service prompts. If you already have **Miniconda** or **Anaconda** installed, you can skip directly to [Step 2](#step-2-download-the-repository-and-create-the-environment).

### macOS (Apple Silicon M1/M2/M3/M4 or Intel)

1. Go to the official Miniforge releases page: <https://github.com/conda-forge/miniforge/releases/latest>
2. Download the installer for your Mac:
   - **Apple Silicon (M1 / M2 / M3 / M4):** `Miniforge3-MacOSX-arm64.pkg` (or `.sh`)
   - **Intel Mac:** `Miniforge3-MacOSX-x86_64.pkg` (or `.sh`)
   - *(Tip: Click the Apple icon in the top-left corner -> **About This Mac** -> check **Chip** vs **Processor**.)*
3. Double-click the `.pkg` installer and follow the on-screen prompts (install for your user account).
4. **Close and re-open** the **Terminal** app (`Applications -> Utilities -> Terminal`, or press `Cmd + Space` and type `Terminal`).
5. Verify conda works:
   ```bash
   conda --version
   ```
   You should see `(base)` at the start of your prompt and a version number like `conda 24.x.x`.

*(Alternative via Homebrew on macOS: `brew install --cask miniforge`, then run `conda init zsh` and restart Terminal.)*

### Windows (Windows 10 / 11, 64-bit)

1. Go to the official Miniforge releases page: <https://github.com/conda-forge/miniforge/releases/latest>
2. Download **`Miniforge3-Windows-x86_64.exe`**.
3. Run the installer with the default options (choose **Just Me**; you do **not** need to add Miniforge to your system `PATH`).
4. Open the **Windows Start Menu** and search for **`Miniforge Prompt`** (if using Miniconda/Anaconda instead, open **`Anaconda Prompt`**).
   > **Important for Windows users:** Always use **Miniforge Prompt** (or **Anaconda Prompt**), not regular Command Prompt / PowerShell, unless you have explicitly run `conda init`.
5. Verify conda works:
   ```bat
   conda --version
   ```

---

## Step 2: Download the Repository and Create the Environment

### 2.1 Download (Clone) the Repository

Choose a folder where you want to keep the project (for example, `Documents` or `Documents/GitHub`).

**macOS (Terminal):**

```bash
mkdir -p ~/Documents/GitHub
cd ~/Documents/GitHub
git clone https://github.com/Vicellken/PaleoSTeHM-Dashboard.git
cd PaleoSTeHM-Dashboard
```

**Windows (Miniforge Prompt):**

```bat
mkdir %USERPROFILE%\Documents\GitHub
cd %USERPROFILE%\Documents\GitHub
git clone https://github.com/Vicellken/PaleoSTeHM-Dashboard.git
cd PaleoSTeHM-Dashboard
```

> **Alternative without Git (or while repository access is restricted):**
> - If you do not have Git installed, click the green **`<> Code`** button on GitHub -> **Download ZIP**, unzip it (for example into `Documents/PaleoSTeHM-Dashboard`), and `cd` into that unzipped folder.
> - While the repository is private, `git clone` will prompt for GitHub authentication (browser login, [GitHub CLI](https://cli.github.com/), or a Personal Access Token).

### 2.2 Create the `paleostehm` Conda Environment

Make sure your terminal is inside the `PaleoSTeHM-Dashboard` folder (you should see `environment.yml` when you run `ls` on macOS or `dir` on Windows).

**Using Miniforge (fastest — uses `mamba`):**

```bash
mamba env create -f environment.yml
```

**Or using standard `conda` (Miniforge / Miniconda / Anaconda):**

```bash
conda env create -f environment.yml
```

This creates an isolated environment named **`paleostehm`** and takes a few minutes the first time.

### 2.3 Activate the Environment

```bash
conda activate paleostehm
```

Your terminal prompt should change from `(base)` to **`(paleostehm)`**.

### 2.4 Hardware Notes (macOS Apple Silicon, Intel Mac, Windows GPU)

- **macOS Apple Silicon (M1–M4):** Ready to go! Training runs on CPU (Apple MPS is intentionally disabled by the dashboard because double-precision GP algebra runs more reliably on CPU). No extra packages are needed.
- **macOS Intel (x86_64):** PyTorch stopped shipping macOS x86_64 wheels after version `2.2.2`. The committed `environment.yml` asks for `torch>=2.7.0`, so `conda env create` / `mamba env create` stops during the pip step on an Intel Mac and does not leave an activatable `paleostehm` environment. Skip Step 2.2 on Intel Macs and create the environment like this instead (leave `environment.yml` unchanged; Apple Silicon and Windows still use it):
  ```bash
  conda env remove -n paleostehm
  conda create -n paleostehm -c conda-forge --override-channels python=3.10 "numpy>=1.26,<2.3" "scipy>=1.11" "pandas>=2.0" "matplotlib>=3.8" "cartopy>=0.23" "netcdf4>=1.6.5" openpyxl pyyaml "tqdm>=4.65" pytest "pip>=24.0"
  conda activate paleostehm
  pip install "torch==2.2.2" "pyro-api==0.1.2" "pyro-ppl>=1.9.1" "streamlit>=1.28.0"
  ```
  Run `conda env remove` only when a partial `paleostehm` environment is left over from a failed create. If conda says the environment does not exist, continue with `conda create`.
- **Windows (CPU — default):** Ready to go! CPU training works out of the box.
- **Windows / Linux with an NVIDIA GPU (optional):** If you have a CUDA-capable NVIDIA GPU (including RTX 50-series / Blackwell) and want GPU-accelerated training, replace the CPU PyTorch wheel with the CUDA 12.8 build:
  ```bash
  conda activate paleostehm
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
  ```

---

## Step 3: Launch the Dashboard and Run a Quick Test

From inside the `PaleoSTeHM-Dashboard` folder with `(paleostehm)` active, run:

```bash
python -m stgp_dashboard
```

> **Always launch with `python -m stgp_dashboard`** (rather than `streamlit run stgp_dashboard/app.py`). The module launcher automatically sets `PYTHONPATH` to the project root, runs from the project root directory, and sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid OpenMP runtime conflicts on both macOS and Windows.

Your web browser will open automatically (typically at `http://localhost:8501`).

### Quick Test with the Included Open-Access Dataset

`Data/US_Atlantic_Coast_for_ESTGP.csv` is the only data file shipped in this repository. It is the open-access US Atlantic Coast dataset from upstream PaleoSTeHM. `Data/GIA/`, `Data/ANU_1D/`, and `Data/3D_GIA/` are empty placeholders for files you supply yourself. A ready-to-load sample config is at `configs/example_us_atlantic.yaml`:

1. On the **Welcome** step, verify the green banner shows **"All required packages are available."** (expand **Package check details** to inspect individual libraries).
2. Choose one of the two ways to start:
   - **Fastest test:** Under **Resume a prior run (optional)**, type or paste `configs/example_us_atlantic.yaml` into **Load existing dashboard config YAML** (or click **Browse…** to select it), click **Load config**, and step through the wizard with **Next →**.
   - **From scratch:** Click **Next →**, enter `Data/US_Atlantic_Coast_for_ESTGP.csv` in **Paleo RSL CSV path**, enter an **Analysis output directory** (e.g., `Outputs/test_run`), keep **Include GIA as prior** off on Step 3 (**GIA prior**), and step through the wizard.
3. **Two-stage execution:**
   - On **Review & run**, click **Run model** to train the GP model via stochastic variational inference (SVI) and inspect optimized hyperparameters and prior-boundary diagnostics.
   - On **Results**, click **Proceed with post-processing & visualization**. With **Include GIA as prior** left off, the run writes temporal NetCDF files (`run/rsl_run_16ka.nc` and `run/rate_run_16ka.nc`) and figures rebuilt from those files. Spatial maps, decomposition CSVs (`temporal_decomposition*.csv`, `spatial_predictions.csv`, `spatial_rates.csv`), and sea-level budget files are written only when you supply a GIA prior.

> **Note:** `configs/example_us_atlantic.yaml` is tuned as a quick smoke test (200 SVI iterations, 0.5° spatial grid, `large` prior preset). Starting a fresh run from the wizard defaults uses 500 SVI iterations, a 0.3° spatial grid, and the `regional` prior preset.

### Stopping the Dashboard

- Press **`Ctrl + C`** in the terminal window where the dashboard is running.
- Optionally deactivate the conda environment when finished:
  ```bash
  conda deactivate
  ```

---

## Step 4: How to Re-Run Later (After Closing the Terminal or Deactivating Conda)

Whenever you open a new terminal session (or after running `conda deactivate`), your prompt will show `(base)` or no environment name, and your working directory will reset to your home folder. Follow these steps to start the dashboard again:

### macOS (Terminal)

```bash
# 1. Change directory (cd) into your PaleoSTeHM-Dashboard folder
cd ~/Documents/GitHub/PaleoSTeHM-Dashboard

# 2. Activate the paleostehm conda environment
conda activate paleostehm

# 3. Launch the dashboard
python -m stgp_dashboard
```

> **macOS shortcut if you forgot the folder path:** Type `cd ` (with a trailing space) in Terminal, drag the `PaleoSTeHM-Dashboard` folder from Finder directly into the Terminal window, and press **Enter**.

### Windows (Miniforge Prompt / Anaconda Prompt)

1. Open **Miniforge Prompt** (or **Anaconda Prompt**) from the Start Menu.
2. Run:
   ```bat
   cd /d %USERPROFILE%\Documents\GitHub\PaleoSTeHM-Dashboard
   conda activate paleostehm
   python -m stgp_dashboard
   ```

> **Windows shortcut if your project is on another drive (e.g., `D:\Projects\PaleoSTeHM-Dashboard`):**
> Always include `/d` with `cd` when switching drives, e.g.:
> ```bat
> cd /d D:\Projects\PaleoSTeHM-Dashboard
> ```
> Or drag the `PaleoSTeHM-Dashboard` folder from File Explorer into Miniforge Prompt after typing `cd /d ` (with a trailing space) and press **Enter**.

---

## Step 5: Optional GIA Data, Input Format, Outputs & Programmatic Use

### Optional GIA Prior Data Setup

No private study datasets or large GIA model grids are committed in this repository. To run analyses **with a GIA prior**, place your own GIA model files into the pre-created (gitignored) folders below, or browse to any folder on your computer in Step 3 (**GIA prior**) of the dashboard:

- `Data/GIA/` — 1D ICE6G-style `.mat` GIA ensemble files (`*.mat`, plus `LatitudeGrid.mat`, `LongitudeGrid.mat`, `time.mat`)
- `Data/ANU_1D/` — 1D ANU-ICE `.txt` or `.csv` ensembles. Expected names look like `RSL_ANU-ICE_EM*_21ka.txt`, with 36 columns: longitude, latitude, then relative sea level at 34 ages.
- `Data/3D_GIA/` — Optional regional 3D GIA grids (`.csv` or `.txt`). Each file is a regular grid with columns longitude, latitude, then one column per time.

The optional five-model `.npy` loader defaults to a regional box of 115–130°E and 30–43°N at 0.5°. Override that box in the config under `gia.npy_regional_grid` (`lon_min`, `lon_max`, `lat_min`, `lat_max`, `step`).

Files placed inside these three directories and `Outputs/` are automatically ignored by Git so your local datasets and run results are never accidentally committed.

### Input Paleo RSL CSV Format

Your paleo RSL `.csv` file should follow the PaleoSTeHM / HOLSEA column convention:

- **Required columns:** `Region` (region name), `Region.1` (numeric region ID), `Latitude`, `Longitude`, `RSL`, `RSLer_up_2sd`, `RSLer_low_2sd`, `Age` (cal yr BP), `Age_up_er_2sd`, `Age_low_er_2sd`, `Limiting` (`0` = SLIP, `-1` = marine limiting, `1` = terrestrial limiting).
- **Optional column:** `Level` (e.g., `1`, `2`, `3` for HOLSEA quality-level filtering and Level-2 uncertainty inflation; if omitted, level filtering is skipped and all SLIP rows are included).
- **Included sample dataset (`Data/US_Atlantic_Coast_for_ESTGP.csv`):** 1,432 rows across 17 US Atlantic Coast regions, copied from upstream [radical-collaboration/PaleoSTeHM](https://github.com/radical-collaboration/PaleoSTeHM) for testing and demonstration. Its raw header contains two `Region` columns (pandas reads the second numeric ID column as `Region.1`), omits `Level`, includes extra `Indicator`/`CC` metadata columns, and spans `-49` to `11669` cal yr BP (including 45 post-1950 CE negative BP rows).

### Output Directory Layout & Reproducibility

Each dashboard run saves a self-contained folder under your chosen output directory (e.g., `Outputs/example_us_atlantic/`):

- `dashboard_config.yaml` — Exact configuration used for the run (reloadable on the **Welcome** screen).
- `run.log` — Captured training and post-processing log.
- `run/` — Model artifacts:
  - `trained_param_store.pt` — Saved Pyro parameter store (can be reloaded to skip SVI).
  - `optimized_params.csv`, `boundary_proximity.csv`, `optimization_track.csv` — Hyperparameter estimates, prior-boundary checks, and ELBO loss curve.
  - `rsl_run_*.nc`, `rate_run_*.nc` — Temporal RSL and rate reconstructions in NetCDF format.
  - `figures/` — Generated PNG and PDF diagnostic/decomposition figures.
  - *(GIA workflows additionally export `temporal_decomposition*.csv`, `spatial_predictions.csv`, `spatial_rates.csv`, and `sea_level_budget*` outputs.)*
- `sensitivity/` — Optional sensitivity suite outputs when run from the **Sensitivity** tab.

By default, runs enable deterministic seeding (`fixed_seed: true`, `seed: 0`) across PyTorch, NumPy, Python `random`, and Pyro before kernel initialization.

### Programmatic Execution (Python API)

The dashboard is a thin UI over `stgp_pipeline.run.run_stgp_analysis`:

```python
from pathlib import Path
import yaml
from stgp_pipeline.run import run_stgp_analysis

cfg_path = Path("configs/example_us_atlantic.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
metrics = run_stgp_analysis(
    cfg,
    output_dir=Path("Outputs/example_us_atlantic"),
    base_dir=cfg_path.parent,
    postprocess=True,
)
```

---

## Repository Layout

- `stgp_dashboard/` — Guided Streamlit dashboard UI (`python -m stgp_dashboard`)
- `stgp_pipeline/` — Modular STGP training, GIA weighting/covariance setup, prediction, export, budget, visualization, and sensitivity suite
- `PSTHM/` — Core PaleoSTeHM Gaussian Process library (kernels, kernel factory, SVI optimization, config loading, plotting)
- `gia_covariance_tools/` — 1D/3D GIA ensemble loading and WRSS-weighted covariance computation
- `Data/` — Open-access sample dataset (`US_Atlantic_Coast_for_ESTGP.csv`) and gitignored placeholders for optional user-supplied GIA grids (`GIA/`, `ANU_1D/`, `3D_GIA/`)
- `configs/` — Ready-to-run sample dashboard config (`example_us_atlantic.yaml`)
- `environment.yml` — Cross-platform `conda-forge` environment specification (`paleostehm`)
- `CITATION.cff` / `.zenodo.json` — Software citation and Zenodo release metadata

---

## Updating the Repository

To pull the latest updates from GitHub later:

```bash
cd path/to/PaleoSTeHM-Dashboard
git pull
```

If `environment.yml` has changed after an update:

```bash
conda activate paleostehm
conda env update -f environment.yml --prune
```

---

## Troubleshooting

- **`conda: command not found` (macOS) or `'conda' is not recognized` (Windows):**
  - On **macOS**, close and reopen Terminal after installing Miniforge, or run `~/miniforge3/bin/conda init zsh` and restart Terminal.
  - On **Windows**, make sure you opened **Miniforge Prompt** (or **Anaconda Prompt**) from the Start Menu rather than plain Command Prompt.
- **Port 8501 already in use:**
  Pass a custom port directly to the launcher:
  ```bash
  python -m stgp_dashboard --server.port 8502
  ```
- **OpenMP duplicate runtime error (`OMP: Error #15` on macOS `libomp.dylib` or Windows `libiomp5md.dll`):**
  Always launch via `python -m stgp_dashboard` rather than `streamlit run` — it sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically. When running custom Python scripts or `pytest` directly in a shell, prefix the command with `KMP_DUPLICATE_LIB_OK=TRUE` (macOS/Linux) or `set KMP_DUPLICATE_LIB_OK=TRUE` (Windows) if needed.
- **File browser (`Browse…`) unavailable:**
  The **Browse…** button runs Python's built-in `tkinter` file dialog in a helper subprocess. If your Python build lacks Tk support, clicking **Browse…** displays a warning — you can always type or paste file and folder paths directly into the text fields.
- **Running the test suite:**
  ```bash
  conda activate paleostehm
  pytest stgp_dashboard/tests stgp_pipeline/tests
  ```

---

## Citation, Acknowledgments & License

If you use **PaleoSTeHM Dashboard** in academic work, please cite both this software release (see [`CITATION.cff`](CITATION.cff) and the Zenodo software DOI once published) and the upstream **PaleoSTeHM** framework:

1. **PaleoSTeHM Dashboard (this software):**
   Gu, Y.-F. (2026). *PaleoSTeHM Dashboard: Interactive STGP Holocene Sea-Level Modeling Framework* (Version 0.1.0) [Computer software]. GitHub / Zenodo. <https://github.com/Vicellken/PaleoSTeHM-Dashboard>
2. **Upstream PaleoSTeHM (`PSTHM` core library):**
   Lin, Y., et al. (2025). *PaleoSTeHM: Paleo Spatio-Temporal Hierarchical Modeling*. Rutgers Earth System Science & Policy Lab. <https://github.com/radical-collaboration/PaleoSTeHM>
3. **Sample dataset (`Data/US_Atlantic_Coast_for_ESTGP.csv`):**
   Redistributed from upstream [radical-collaboration/PaleoSTeHM](https://github.com/radical-collaboration/PaleoSTeHM) for testing and demonstration purposes; please consult upstream PaleoSTeHM for original compilation citations and data terms before using it in research publications.

### License & Third-Party Notices

This repository is distributed under the **MIT License** (see [`LICENSE`](LICENSE)), retaining copyright notices for both Yi-Fei Gu (2026) and Rutgers Earth System Science & Policy Lab (2023). Select Gaussian Process kernel implementations in `PSTHM/kernels.py` are adapted from Pyro's GP module (Apache License 2.0); see [`NOTICE`](NOTICE) for third-party attribution details.
