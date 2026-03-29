# Movie Recommender — ALS Solver Comparison

An academic research project comparing five linear system solvers within an **Alternating Least Squares (ALS)** collaborative filtering recommender system, evaluated on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset.

## Overview

This project investigates how the choice of linear system solver affects the performance of an ALS-based matrix factorization recommender system. Five solvers are implemented from scratch (without LAPACK) and benchmarked across computational efficiency, numerical stability, and recommendation quality.

### Solvers Compared

| Solver | File |
|--------|------|
| Cholesky Decomposition | `solvers/cholesky_decomposition.py` |
| LU Decomposition (Partial Pivoting) | `solvers/lu_decomposition.py` |
| QR Decomposition (Householder) | `solvers/qr_decomposition.py` |
| Gaussian Elimination (Partial Pivoting) | `solvers/partial_pivoting.py` |
| Matrix Inversion (Gauss-Jordan) | `solvers/matrix_inverse.py` |

## Prerequisites

- Python 3.8 or higher
- [Jupyter Notebook](https://jupyter.org/install) or JupyterLab

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/LeinnarF/Movie-Recommender.git
   cd Movie-Recommender
   ```

2. **Install dependencies**

   ```bash
   pip install numpy pandas matplotlib seaborn jupyter
   ```

   The MovieLens 100K dataset is already included in the `dataset/ml-100k/` directory.

## Running the Program

### 1. Exploratory Data Analysis

Open and run `EDA.ipynb` to explore the MovieLens dataset — rating distributions, user statistics, and outlier analysis.

```bash
jupyter notebook EDA.ipynb
```

### 2. Solver Performance Comparison (Main Analysis)

Open and run `test.ipynb` to train the ALS recommender system using each of the five solvers and compare their performance metrics (runtime, RMSE, residual norms, condition numbers).

```bash
jupyter notebook test.ipynb
```

**ALS configuration used in `test.ipynb`:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `K` | 10 | Number of latent factors |
| `λ` | 10 | Regularization parameter |
| `EPOCHS` | 30 | Number of training iterations |

## Project Structure

```
Movie-Recommender/
├── EDA.ipynb                      # Exploratory Data Analysis notebook
├── test.ipynb                     # ALS solver comparison notebook
├── solvers/
│   ├── cholesky_decomposition.py  # Cholesky factorization solver
│   ├── lu_decomposition.py        # LU decomposition with partial pivoting
│   ├── matrix_inverse.py          # Gauss-Jordan matrix inversion
│   ├── partial_pivoting.py        # Gaussian elimination with partial pivoting
│   └── qr_decomposition.py        # QR decomposition (Householder reflections)
├── dataset/
│   └── ml-100k/                   # MovieLens 100K dataset
│       ├── u.data                 # 100,000 ratings (user, item, rating, timestamp)
│       ├── u.item                 # Movie metadata (title, genres)
│       ├── u.user                 # User demographics (age, gender, occupation)
│       └── ...                    # Cross-validation splits (u1–u5, ua, ub)
└── Latex/
    ├── main.tex                   # Research paper (LaTeX source)
    ├── main.pdf                   # Compiled research paper
    └── figures/                   # Charts and visualizations
```

## Dataset

The project uses the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset:

- **100,000** ratings from **943 users** on **1,682 movies**
- Ratings range from 1 to 5 (mean ≈ 3.53)
- Each user has rated at least 20 movies

## Key Findings

- All five solvers produce **identical recommendation quality** (same train/test RMSE).
- **Cholesky decomposition** is the fastest, exploiting the symmetric positive definite (SPD) structure of the ALS normal equations.
- **Matrix inversion** is the slowest and least numerically stable, producing the largest residual norms.
- Factorization-based methods (Cholesky, LU, QR) are preferred over explicit matrix inversion for both speed and stability.

## Authors

Franniel Luigi Hilario, Brian Gabriel Magbanua, Fiona Ventura, Jermaine Pasamba, Mark Oliver Medina  
*College of Science — BS Mathematics with CS Specialization, Bulacan State University*
