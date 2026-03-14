# Movie Recommender: ALS Solver Comparison

This repository studies how different linear system solvers behave inside Alternating Least Squares (ALS) for collaborative filtering on MovieLens 100k.

## Reproducibility: Exact Run Instructions

### 1. Environment

Use Python 3.10+ and install required packages:

```bash
pip install numpy scipy
```

If you are using the project virtual environment in this repo:

```powershell
.\.venv\Scripts\activate
pip install numpy scipy
```

### 2. Dataset

The default run command expects:

- `dataset/ml-100k/u.data`

No extra download step is needed if the dataset folder in this repo is unchanged.

### 3. Run Experiment 1

Default full run (all configured Experiment 1 solvers, 50 epochs):

```bash
python run_experiment1.py
```

Windows explicit interpreter example:

```powershell
.\.venv\Scripts\python.exe run_experiment1.py
```

Quick smoke run example (faster):

```bash
python run_experiment1.py --epochs 1 --solvers gauss_jacobi --output-dir outputs/experiment1_smoke
```

### 4. CLI Options

`run_experiment1.py` supports:

- `--dataset` (default: `dataset/ml-100k/u.data`)
- `--output-dir` (default: `outputs/experiment1`)
- `--k` latent factors (default: `10`)
- `--lam` regularization lambda (default: `0.1`)
- `--epochs` (default: `50`)
- `--seed` (default: `42`)
- `--test-ratio` (default: `0.2`)
- `--tol` iterative solver tolerance (default: `1e-8`)
- `--max-iter` iterative solver max iterations (default: `500`)
- `--solvers` optional list from:
  - `gaussian_pivoting`
  - `matrix_inverse`
  - `lu`
  - `qr`
  - `gauss_jacobi`
  - `gauss_seidel`
  - `cholesky`

## Expected Output Files

By default, outputs are saved in `outputs/experiment1/`.

### A. Per-solver epoch log CSV

File pattern:

- `outputs/experiment1/<solver_name>_experiment1.csv`

Expected columns:

- `epoch`
- `train_rmse`
- `test_rmse`
- `loss`
- `mean_residual`
- `mean_inner_iterations`
- `mean_condition_number`
- `epoch_time_sec`
- `cumulative_time_sec`

Notes:

- `mean_inner_iterations` is `nan` for direct solvers.
- If an iterative solver diverges numerically for some subproblems, values may become `inf`/`nan`. This is logged intentionally for analysis.

### B. Experiment summary CSV

File:

- `outputs/experiment1/summary_experiment1.csv`

Expected columns:

- `solver`
- `final_train_rmse`
- `final_test_rmse`
- `final_loss`
- `final_mean_residual`
- `final_mean_inner_iterations`
- `final_mean_condition_number`
- `total_time_sec`
- `epochs`
- `k`
- `lambda`

## Project Entry Points

- Experiment runner: `run_experiment1.py`
- ALS engine and logging: `als_engine.py`
- Solver modules: `solvers/`

## HTML Dashboard With Embedded Terminal

To use the embedded terminal inside `dashboard.html`, run the local backend server:

```bash
python web_terminal.py
```

Then open:

- `http://127.0.0.1:8000/dashboard.html`

Notes:

- The browser cannot execute shell commands directly without a backend.
- The terminal panel uses `/api/terminal/exec` from `web_terminal.py`.
- Current working directory is preserved across commands, including `cd`.
