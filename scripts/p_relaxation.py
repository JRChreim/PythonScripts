import argparse

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.relaxation.pressure_relaxation import (
    build_example_problem,
    grad_ascent_softmax,
)

DEFAULT_LEARNING_RATE = 1.0e-2
DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_TOLERANCE = 1.0e-12


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pressure-relaxation softmax optimization experiment."
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--maxit", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    pressure_term, gamma_term, total_energy, source_term = build_example_problem()
    pressure_star, alpha_star, history = grad_ascent_softmax(
        pressure_term,
        gamma_term,
        total_energy,
        source_term,
        lr=args.learning_rate,
        maxit=args.maxit,
        tol=args.tol,
    )

    print("Optimal p* =", pressure_star)
    print("Optimal alpha* =", alpha_star)
    for iteration in history:
        print(iteration)
    print(sum(alpha_star))


if __name__ == "__main__":
    main()
