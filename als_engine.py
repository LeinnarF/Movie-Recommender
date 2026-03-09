import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from solvers.cholesky_decomposition import cholesky_decomposition
from solvers.gauss_jacobi import gauss_jacobi
from solvers.gauss_seidel import gauss_seidel
from solvers.lu_decomposition import lu_decomposition
from solvers.matrix_inverse import matrix_inverse
from solvers.partial_pivoting import partial_pivoting
from solvers.qr_decomposition import qr_decomposition


@dataclass
class ALSConfig:
    k: int = 10
    lam: float = 0.1
    epochs: int = 50
    seed: int = 42
    test_ratio: float = 0.2
    iterative_tol: float = 1e-8
    iterative_max_iter: int = 500


SOLVER_INFO = {
    "gaussian_pivoting": {"fn": partial_pivoting, "iterative": False},
    "matrix_inverse": {"fn": matrix_inverse, "iterative": False},
    "lu": {"fn": lu_decomposition, "iterative": False},
    "qr": {"fn": qr_decomposition, "iterative": False},
    "gauss_jacobi": {"fn": gauss_jacobi, "iterative": True},
    "gauss_seidel": {"fn": gauss_seidel, "iterative": True},
    "cholesky": {"fn": cholesky_decomposition, "iterative": False},
}


def load_movielens_100k(data_path):
    raw = np.loadtxt(data_path, delimiter="\t", dtype=np.int64)
    user_ids = raw[:, 0] - 1
    item_ids = raw[:, 1] - 1
    ratings = raw[:, 2].astype(float)

    n_users = int(user_ids.max()) + 1
    n_items = int(item_ids.max()) + 1
    R = np.zeros((n_users, n_items), dtype=float)
    R[user_ids, item_ids] = ratings
    return R


def train_test_split_sparse(R, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    observed = np.argwhere(R > 0)
    shuffled = observed.copy()
    rng.shuffle(shuffled)

    n_test = int(len(shuffled) * test_ratio)
    test_entries = shuffled[:n_test]

    R_train = R.copy()
    R_test = np.zeros_like(R)

    for u, i in test_entries:
        R_test[u, i] = R_train[u, i]
        R_train[u, i] = 0.0

    return R_train, R_test


def rmse_on_observed(R_true, R_pred):
    mask = R_true > 0
    if not np.any(mask):
        return float("nan")
    if not np.all(np.isfinite(R_pred[mask])):
        return float("nan")
    err = R_true[mask] - R_pred[mask]
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.sqrt(np.mean(err * err))
    if not np.isfinite(value):
        return float("nan")
    return float(value)


def regularized_loss(R_train, U, V, lam):
    with np.errstate(over="ignore", invalid="ignore"):
        pred = U @ V.T
    mask = R_train > 0
    if not np.all(np.isfinite(pred[mask])):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        sq_error = np.sum((R_train[mask] - pred[mask]) ** 2)
        reg = lam * (np.sum(U * U) + np.sum(V * V))
    if not np.isfinite(sq_error + reg):
        return float("nan")
    return float(sq_error + reg)


def _solve_with_solver(solver_name, A, b, iterative_tol, iterative_max_iter):
    info = SOLVER_INFO[solver_name]
    fn = info["fn"]

    if info["iterative"]:
        x, residual, iters = fn(A, b, max_iter=iterative_max_iter, tol=iterative_tol)
        return x, float(residual), int(iters)

    x, residual = fn(A, b)
    return x, float(residual), None


def run_als(R_train, R_test, solver_name, config):
    if solver_name not in SOLVER_INFO:
        raise ValueError(f"Unknown solver '{solver_name}'. Available: {sorted(SOLVER_INFO.keys())}")

    rng = np.random.default_rng(config.seed)
    n_users, n_items = R_train.shape

    U = rng.normal(0.0, 0.1, size=(n_users, config.k))
    V = rng.normal(0.0, 0.1, size=(n_items, config.k))

    user_items = [np.where(R_train[u] > 0)[0] for u in range(n_users)]
    item_users = [np.where(R_train[:, i] > 0)[0] for i in range(n_items)]

    logs = []
    cumulative_time = 0.0
    I = np.eye(config.k)

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        residuals = []
        condition_numbers = []
        inner_iterations = []

        for u in range(n_users):
            items = user_items[u]
            if items.size == 0:
                continue
            V_i = V[items]
            r_u = R_train[u, items]
            A = V_i.T @ V_i + config.lam * I
            b = V_i.T @ r_u

            x, residual, iters = _solve_with_solver(
                solver_name, A, b, config.iterative_tol, config.iterative_max_iter
            )
            U[u] = x
            residuals.append(residual)
            condition_numbers.append(float(np.linalg.cond(A)))
            if iters is not None:
                inner_iterations.append(iters)

        for i in range(n_items):
            users = item_users[i]
            if users.size == 0:
                continue
            U_u = U[users]
            r_i = R_train[users, i]
            A = U_u.T @ U_u + config.lam * I
            b = U_u.T @ r_i

            x, residual, iters = _solve_with_solver(
                solver_name, A, b, config.iterative_tol, config.iterative_max_iter
            )
            V[i] = x
            residuals.append(residual)
            condition_numbers.append(float(np.linalg.cond(A)))
            if iters is not None:
                inner_iterations.append(iters)

        epoch_time = time.perf_counter() - start
        cumulative_time += epoch_time

        with np.errstate(over="ignore", invalid="ignore"):
            pred = U @ V.T
        train_rmse = rmse_on_observed(R_train, pred)
        test_rmse = rmse_on_observed(R_test, pred)
        loss = regularized_loss(R_train, U, V, config.lam)

        epoch_row = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "loss": loss,
            "mean_residual": float(np.mean(residuals)) if residuals else float("nan"),
            "mean_inner_iterations": float(np.mean(inner_iterations)) if inner_iterations else float("nan"),
            "mean_condition_number": float(np.mean(condition_numbers)) if condition_numbers else float("nan"),
            "epoch_time_sec": epoch_time,
            "cumulative_time_sec": cumulative_time,
        }
        logs.append(epoch_row)

        print(
            f"[{solver_name}] epoch={epoch:02d} "
            f"train_rmse={train_rmse:.4f} test_rmse={test_rmse:.4f} "
            f"residual={epoch_row['mean_residual']:.2e} time={epoch_time:.2f}s"
        )

    return logs, U, V


def write_logs_to_csv(rows, csv_path):
    if not rows:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment_1(
    dataset_path,
    output_dir,
    config,
    solvers=None,
):
    if solvers is None:
        solvers = [
            "gaussian_pivoting",
            "matrix_inverse",
            "lu",
            "qr",
            "gauss_jacobi",
            "gauss_seidel",
        ]

    R = load_movielens_100k(dataset_path)
    R_train, R_test = train_test_split_sparse(R, test_ratio=config.test_ratio, seed=config.seed)

    summary = []
    for solver_name in solvers:
        print(f"\n=== Running solver: {solver_name} ===")
        logs, _, _ = run_als(R_train, R_test, solver_name, config)

        solver_csv = os.path.join(output_dir, f"{solver_name}_experiment1.csv")
        write_logs_to_csv(logs, solver_csv)

        last = logs[-1]
        summary.append(
            {
                "solver": solver_name,
                "final_train_rmse": last["train_rmse"],
                "final_test_rmse": last["test_rmse"],
                "final_loss": last["loss"],
                "final_mean_residual": last["mean_residual"],
                "final_mean_inner_iterations": last["mean_inner_iterations"],
                "final_mean_condition_number": last["mean_condition_number"],
                "total_time_sec": last["cumulative_time_sec"],
                "epochs": config.epochs,
                "k": config.k,
                "lambda": config.lam,
            }
        )

    summary_csv = os.path.join(output_dir, "summary_experiment1.csv")
    write_logs_to_csv(summary, summary_csv)
    print(f"\nSaved Experiment 1 outputs to: {output_dir}")

    return summary
