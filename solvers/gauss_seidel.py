import numpy as np


def gauss_seidel(A, b, max_iter=500, tol=1e-8):
    """Solve Ax=b with Gauss-Seidel iteration.

    Returns (x, residual_norm, iterations_used).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]

    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("Gauss-Seidel requires non-zero diagonal entries")

    x = np.zeros(n, dtype=float)

    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            left = np.dot(A[i, :i], x[:i])
            right = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (b[i] - left - right) / A[i, i]

        if not np.all(np.isfinite(x)):
            return x_old, float("inf"), it

        if np.linalg.norm(x - x_old) < tol:
            residual = float(np.linalg.norm(A @ x - b))
            if not np.isfinite(residual):
                residual = float("inf")
            return x, residual, it

    residual = float(np.linalg.norm(A @ x - b))
    if not np.isfinite(residual):
        residual = float("inf")
    return x, residual, max_iter
