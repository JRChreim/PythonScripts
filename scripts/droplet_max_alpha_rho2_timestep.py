"""Plot maximum alpha_rho2 versus timestep for the droplet breakup data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.plots.publication import (
    THESIS_LABEL_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    THESIS_TICK_FONT_SIZE,
    add_thesis_export_argument,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)


DEFAULT_INPUT = Path(
    "/home/user/Documents/GitHub/PythonScripts/artifacts/Data/DropletAerobreakup/"
    "max_alpha_rho2_location.csv"
)
DEFAULT_OUTPUT = None
THESIS_EXPORT_STEM = "AlphaRho2Timestep"
THESIS_FIGURE_SIZE = thesis_figure_size(0.42)
LINE_COLOR = "#1f77b4"
LINEWIDTH = 1.8


def load_max_alpha_history(filepath: Path) -> dict[str, np.ndarray]:
    """Load timestep and maximum alpha_rho2 columns from the CSV file."""

    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    rows: list[tuple[float, float, float, float]] = []

    with filepath.open(newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        required_columns = {"timestep", "max_alpha_rho2", "x", "y"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filepath}: {sorted(missing_columns)}"
            )

        for row in reader:
            if not row["timestep"] or not row["max_alpha_rho2"] or not row["x"] or not row["y"]:
                continue
            rows.append(
                (
                    float(row["timestep"]),
                    float(row["max_alpha_rho2"]),
                    float(row["x"]),
                    float(row["y"]),
                )
            )

    if not rows:
        raise ValueError(f"No usable rows were found in {filepath}")

    rows.sort(key=lambda item: item[0])
    data = np.asarray(rows, dtype=float)

    return {
        "timestep": data[:, 0],
        "max_alpha_rho2": data[:, 1],
        "x": data[:, 2],
        "y": data[:, 3],
    }


def get_global_max_alpha_timestep(data: dict[str, np.ndarray]) -> float:
    """Return the timestep where max(alpha_rho2) is largest."""

    max_index = int(np.argmax(data["max_alpha_rho2"]))
    return float(data["timestep"][max_index])


def filter_history_to_timestep_range(
    data: dict[str, np.ndarray],
    start_timestep: float,
    end_timestep: float,
) -> dict[str, np.ndarray]:
    """Keep only rows inside the requested timestep interval."""

    if start_timestep > end_timestep:
        raise ValueError(
            "The start timestep must be less than or equal to the end timestep. "
            f"Got start={start_timestep:g}, end={end_timestep:g}."
        )

    mask = (data["timestep"] >= float(start_timestep)) & (
        data["timestep"] <= float(end_timestep)
    )
    if not np.any(mask):
        raise ValueError(
            "No rows were found in the requested timestep interval. "
            f"start={start_timestep:g}, end={end_timestep:g}"
        )

    return {key: values[mask] for key, values in data.items()}


def format_timestep(value: float) -> str:
    """Format a timestep for human-readable output."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def get_shared_alpha_limits(*datasets: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return common y-axis bounds for one or more datasets."""

    min_values = [float(dataset["max_alpha_rho2"].min()) for dataset in datasets]
    max_values = [float(dataset["max_alpha_rho2"].max()) for dataset in datasets]
    vmin = min(min_values)
    vmax = max(max_values)

    if np.isclose(vmin, vmax):
        pad = 1.0e-12
    else:
        pad = 0.06 * (vmax - vmin)

    return max(0.0, vmin - pad), vmax + pad


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot max(alpha_rho2) versus timestep for the droplet breakup data."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "CSV file containing timestep, max_alpha_rho2, x, y, and z columns "
            "for the droplet breakup analysis."
        ),
    )
    parser.add_argument(
        "--start-timestep",
        type=float,
        default=0.0,
        help="Lower timestep limit for the plot. Defaults to 0.",
    )
    parser.add_argument(
        "--end-offset",
        type=float,
        default=0.0,
        help=(
            "Offset added to the timestep of the global maximum to determine the "
            "shared upper timestep limit. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--all-timesteps",
        action="store_true",
        help="Plot every row in the CSV instead of trimming to the auto cutoff.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Optional path to save the figure, for example "
            "../Caltech-Thesis---JRChreim/Figures/DropletMaxAlphaRho2Timestep.pdf."
        ),
    )
    parser.add_argument(
        "--no-show",
        "--no-plot",
        dest="no_show",
        action="store_true",
        help="Skip showing the figure window unless an output file is requested.",
    )
    return parser


def plot_max_alpha_history(
    data: dict[str, np.ndarray],
    *,
    start_timestep: float,
    end_timestep: float,
):
    """Create a max-alpha history plot."""

    figure, axis = plt.subplots(figsize=THESIS_FIGURE_SIZE, constrained_layout=True)
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    axis.plot(
        data["timestep"],
        data["max_alpha_rho2"],
        color=LINE_COLOR,
        linewidth=LINEWIDTH,
    )
    axis.grid(True, alpha=0.25)
    axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    axis.set_xlim(start_timestep, end_timestep)
    y_min, y_max = get_shared_alpha_limits(data)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel(r"$n_{\mathrm{step}}$", fontsize=THESIS_LABEL_FONT_SIZE)
    axis.set_ylabel(r"$\max(\alpha_{\rho_2})$", fontsize=THESIS_LABEL_FONT_SIZE)

    return figure, axis


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    apply_thesis_style()

    data = load_max_alpha_history(args.input)
    global_max_timestep = get_global_max_alpha_timestep(data)
    end_timestep = global_max_timestep + float(args.end_offset)
    start_timestep = float(args.start_timestep)

    if args.all_timesteps:
        plot_data = data
        print("Plotting all timesteps in the max(alpha_rho2) CSV.")
    else:
        plot_data = filter_history_to_timestep_range(
            data,
            start_timestep,
            end_timestep,
        )
        print(
            "Plotting max(alpha_rho2) from timestep "
            f"{format_timestep(start_timestep)} through {format_timestep(end_timestep)} "
            f"(global maximum occurs at timestep {format_timestep(global_max_timestep)})."
        )

    max_index = int(np.argmax(data["max_alpha_rho2"]))
    print(
        "Global max alpha_rho2 = "
        f"{data['max_alpha_rho2'][max_index]:g} at timestep "
        f"{format_timestep(data['timestep'][max_index])} "
        f"and location ({data['x'][max_index]:g}, {data['y'][max_index]:g})."
    )

    figure = None
    if not args.no_show:
        figure, _ = plot_max_alpha_history(
            plot_data,
            start_timestep=start_timestep if not args.all_timesteps else float(plot_data["timestep"].min()),
            end_timestep=end_timestep if not args.all_timesteps else float(plot_data["timestep"].max()),
        )

    if args.to_thesis:
        if figure is None:
            figure, _ = plot_max_alpha_history(
                plot_data,
                start_timestep=start_timestep if not args.all_timesteps else float(plot_data["timestep"].min()),
                end_timestep=end_timestep if not args.all_timesteps else float(plot_data["timestep"].max()),
            )
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        print(f"Figure written to {output_path}")
    elif args.output is not None:
        if figure is None:
            figure, _ = plot_max_alpha_history(
                plot_data,
                start_timestep=start_timestep if not args.all_timesteps else float(plot_data["timestep"].min()),
                end_timestep=end_timestep if not args.all_timesteps else float(plot_data["timestep"].max()),
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=600, bbox_inches="tight", pad_inches=0.02)
        print(f"Figure written to {args.output}")

    if figure is not None and args.no_show:
        plt.close(figure)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
