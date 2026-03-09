import numpy as np


def gauss_jacobi(A, b, max_iter=500, tol=1e-8):
    """Solve Ax=b with Gauss-Jacobi iteration.

    Returns (x, residual_norm, iterations_used).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]

    D = np.diag(A)
    if np.any(np.isclose(D, 0.0)):
        raise ValueError("Gauss-Jacobi requires non-zero diagonal entries")

    R = A - np.diag(D)
    D_inv = 1.0 / D
    x = np.zeros(n, dtype=float)

    for it in range(1, max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            x_new = D_inv * (b - R @ x)

        if not np.all(np.isfinite(x_new)):
            return x, float("inf"), it

        with np.errstate(over="ignore", invalid="ignore"):
            delta = np.max(np.abs(x_new - x))
        if not np.isfinite(delta):
            return x, float("inf"), it

        if delta < tol:
            x = x_new
            with np.errstate(over="ignore", invalid="ignore"):
                residual = float(np.linalg.norm(A @ x - b))
            if not np.isfinite(residual):
                residual = float("inf")
            return x, residual, it
        x = x_new

    with np.errstate(over="ignore", invalid="ignore"):
        residual = float(np.linalg.norm(A @ x - b))
    if not np.isfinite(residual):
        residual = float("inf")
    return x, residual, max_iter
