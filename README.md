# Movie Recommender System (ALS Benchmark)

This repository implements a Movie Recommender System using **Alternating Least Squares (ALS)** for collaborative filtering on the **MovieLens 100k** dataset. The primary focus of this project is to benchmark various linear system solvers implemented from scratch to compare their performance, stability, and accuracy.

## Features

- **Custom ALS Implementation**: A modular ALS engine that supports pluggable linear solvers.
- **Solvers Implemented from Scratch**:
  - Cholesky Decomposition
  - LU Decomposition (Doolittle algorithm with partial pivoting)
  - QR Decomposition (Householder reflections)
  - Matrix Inverse
  - Partial Pivoting (Gaussian elimination)
- **Comprehensive Benchmarking**: Evaluation of solvers based on:
  - **Computational Efficiency**: Runtime and relative speedup.
  - **Numerical Stability**: Residual norms and matrix condition numbers.
  - **Recommendation Quality**: Training and Testing RMSE convergence.
- **Exploratory Data Analysis (EDA)**: Detailed analysis of the MovieLens 100k dataset.

## Repository Structure

```text
├── als.py                  # Core ALS engine and logging utilities
├── dataset/                # MovieLens 100k dataset
├── solvers/                # Manual implementations of linear solvers
│   ├── cholesky_decomposition.py
│   ├── lu_decomposition.py
│   ├── qr_decomposition.py
│   ├── matrix_inverse.py
│   └── partial_pivoting.py
├── utils/                  # Helper utilities (data splitting, etc.)
├── EDA.ipynb               # Data exploration and visualization
├── test.ipynb              # Main benchmarking notebook
├── summary.md              # Summary of findings and solver comparison
└── Latex/                  # Project report and figures
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd movie-recommender
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn jupyter
   ```

## Usage

### Running the Benchmark

The main entry point for the experiment is the `test.ipynb` notebook. It loads the dataset, performs the train-test split, runs ALS with all five solvers, and generates comparison plots.

To run the notebook:
```bash
jupyter notebook test.ipynb
```

### Exploratory Data Analysis

To view the data analysis and visualizations:
```bash
jupyter notebook EDA.ipynb
```

### Using the ALS Engine Programmatically

You can import the ALS engine and solvers in your own Python scripts:

```python
import numpy as np
from als import run_als
from solvers.cholesky_decomposition import cholesky_decomposition

# Load your R_train and R_test matrices here...

log = run_als(
    R_train, 
    R_test, 
    solver_fn=cholesky_decomposition, 
    solver_name="Cholesky",
    k=10, 
    lam=10.0, 
    n_epochs=30
)
```

## Dataset

This project uses the **MovieLens 100k** dataset provided by GroupLens Research. The dataset is included in the `dataset/ml-100k/` directory.

- **Users**: 943
- **Items**: 1,682
- **Ratings**: 100,000
- **Sparsity**: 93.70%

## Results Summary

- **Recommendation Quality**: All solvers converge to approximately the same RMSE (~0.97 on test data), confirming that the choice of solver does not affect predictive accuracy when implemented correctly.
- **Efficiency**: Cholesky and LU decomposition are generally the fastest.
- **Stability**: QR decomposition and Cholesky are highly stable, while explicit matrix inversion can be risky for ill-conditioned problems.

For more details, see `summary.md`.
