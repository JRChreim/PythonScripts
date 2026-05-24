import argparse

import matplotlib.pyplot as plt

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.mesh.geometric_progression import (
    equivalent_initial_term_from_last_term,
    geometric_series_sum,
    geometric_series_terms,
)
from src.plots.publication import apply_thesis_style

DEFAULT_RATIO = 1.03
DEFAULT_COUNT = 36
DEFAULT_INITIAL_TERM = 1.0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a geometric progression and an equivalent summed progression."
    )
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--initial-term", type=float, default=DEFAULT_INITIAL_TERM)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figure without displaying the matplotlib window.",
    )
    return parser


def plot_progression(ratio: float, count: int, initial_term: float):
    apply_thesis_style()

    terms = geometric_series_terms(initial_term, ratio, count)
    equivalent_initial_term = equivalent_initial_term_from_last_term(
        terms[-1],
        ratio,
        count,
    )
    progression_sum = geometric_series_sum(equivalent_initial_term, ratio, count)

    figure, axis = plt.subplots()
    axis.plot(
        range(count),
        terms,
        marker="o",
        label=r"$\mathrm{Original\ progression}$",
    )
    axis.scatter(
        count - 1,
        progression_sum,
        color="red",
        zorder=3,
        label=r"$\mathrm{Sum\ of\ new\ progression}$",
    )
    axis.set_xlabel(r"$n$")
    axis.set_ylabel(r"$a_n$")
    axis.set_title(rf"$\mathrm{{Geometric\ Progression}}\ (q={ratio:g},\ N={count})$")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()

    print(initial_term)
    print(terms[-1])
    print(equivalent_initial_term)
    print(progression_sum)

    return figure, axis


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    plot_progression(args.ratio, args.count, args.initial_term)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
