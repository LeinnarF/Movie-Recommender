import argparse
import os

from als_engine import ALSConfig, run_experiment_1


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1 for ALS solver comparison")
    parser.add_argument(
        "--dataset",
        default=os.path.join("dataset", "ml-100k", "u.data"),
        help="Path to MovieLens u.data file",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "experiment1"),
        help="Directory for CSV outputs",
    )
    parser.add_argument("--k", type=int, default=10, help="Latent factor dimension")
    parser.add_argument("--lam", type=float, default=0.1, help="L2 regularization")
    parser.add_argument("--epochs", type=int, default=50, help="Number of ALS epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--tol", type=float, default=1e-8, help="Iterative solver tolerance")
    parser.add_argument("--max-iter", type=int, default=500, help="Iterative solver max iterations")
    parser.add_argument(
        "--solvers",
        nargs="*",
        default=None,
        help=(
            "Optional solver list. Available: gaussian_pivoting matrix_inverse lu qr "
            "gauss_jacobi gauss_seidel cholesky"
        ),
    )

    args = parser.parse_args()

    config = ALSConfig(
        k=args.k,
        lam=args.lam,
        epochs=args.epochs,
        seed=args.seed,
        test_ratio=args.test_ratio,
        iterative_tol=args.tol,
        iterative_max_iter=args.max_iter,
    )

    run_experiment_1(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        config=config,
        solvers=args.solvers,
    )


if __name__ == "__main__":
    main()
