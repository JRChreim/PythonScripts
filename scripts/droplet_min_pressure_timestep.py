"""Plot minimum pressure versus timestep for the droplet breakup data."""

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
    "min_pressure_location_pT_pure_Fluid.csv"
)
DEFAULT_MIXTURE_INPUT = Path(
    "/home/user/Documents/GitHub/PythonScripts/artifacts/Data/DropletAerobreakup/"
    "min_pressure_location_pT_Mixture.csv"
)
THESIS_EXPORT_STEM = "MPTevol"
THESIS_FIGURE_SIZE = thesis_figure_size(0.42)
PURE_LINE_COLOR = "#1f77b4"
MIXTURE_LINE_COLOR = "#ff7f0e"
LINEWIDTH = 1.8
Y_SCALE_LINTHRESH = 0.1


def load_min_pressure_history(filepath: Path) -> dict[str, np.ndarray]:
    """Load timestep and minimum-pressure columns from a CSV file."""

    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    rows: list[tuple[float, float]] = []

    with filepath.open(newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        required_columns = {"timestep", "min_pres"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filepath}: {sorted(missing_columns)}"
            )

        for row in reader:
            if not row["timestep"] or not row["min_pres"]:
                continue
            rows.append((float(row["timestep"]), float(row["min_pres"])))

    if not rows:
        raise ValueError(f"No usable rows were found in {filepath}")

    rows.sort(key=lambda item: item[0])
    data = np.asarray(rows, dtype=float)

    return {
        "timestep": data[:, 0],
        "min_pres_pa": data[:, 1],
        "min_pres_mpa": data[:, 1] / 1.0e6,
    }


def get_global_min_pressure_timestep(data: dict[str, np.ndarray]) -> float:
    """Return the timestep where min_pres is most negative."""

    min_index = int(np.argmin(data["min_pres_pa"]))
    return float(data["timestep"][min_index])


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


def get_shared_pressure_limits(*datasets: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return common y-axis bounds for one or more datasets."""

    min_values = [float(dataset["min_pres_mpa"].min()) for dataset in datasets]
    max_values = [float(dataset["min_pres_mpa"].max()) for dataset in datasets]
    vmin = min(min_values)
    vmax = max(max_values)

    if np.isclose(vmin, vmax):
        pad = 1.0e-3
    else:
        pad = 0.06 * (vmax - vmin)

    return vmin - pad, vmax + pad


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot minimum pressure versus timestep for the droplet breakup data."
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file for the pure-fluid history containing timestep and min_pres columns.",
    )
    parser.add_argument(
        "--mixture-input",
        type=Path,
        default=DEFAULT_MIXTURE_INPUT,
        help="CSV file for the mixture history containing timestep and min_pres columns.",
    )
    parser.add_argument(
        "--start-timestep",
        type=float,
        default=0.0,
        help="Lower timestep limit shared by both panels. Defaults to 0.",
    )
    parser.add_argument(
        "--end-offset",
        type=float,
        default=0.0,
        help=(
            "Offset added to the pure-fluid minimum-pressure timestep to determine "
            "the shared upper timestep limit. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save the figure, for example "
            "../Caltech-Thesis---JRChreim/Figures/DropletMinPressureTimestep.pdf."
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


def plot_min_pressure_histories(
    pure_data: dict[str, np.ndarray],
    mixture_data: dict[str, np.ndarray],
    *,
    start_timestep: float,
    end_timestep: float,
):
    """Create a single minimum-pressure history plot with two curves."""

    figure, axis = plt.subplots(figsize=THESIS_FIGURE_SIZE, constrained_layout=True)
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    axis.plot(
        pure_data["timestep"],
        pure_data["min_pres_mpa"],
        color=PURE_LINE_COLOR,
        linewidth=LINEWIDTH,
        label=r"$\mathrm{Pure\ fluid}$",
    )
    axis.plot(
        mixture_data["timestep"],
        mixture_data["min_pres_mpa"],
        color=MIXTURE_LINE_COLOR,
        linewidth=LINEWIDTH,
        label=r"$\mathrm{Mixture}$",
    )

    y_min, y_max = get_shared_pressure_limits(pure_data, mixture_data)
    axis.set_xlim(start_timestep, end_timestep)
    axis.set_ylim(y_min, y_max)
    axis.set_yscale("symlog", linthresh=Y_SCALE_LINTHRESH, linscale=1.0)
    axis.axhline(
        0.0,
        color="0.6",
        linestyle="--",
        linewidth=1.2,
        zorder=0,
        label="_nolegend_",
    )
    axis.grid(True, alpha=0.25)
    axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    axis.set_xlabel(r"$n_{\mathrm{step}}$", fontsize=THESIS_LABEL_FONT_SIZE)
    axis.set_ylabel(r"$p_{\min}\ [\mathrm{MPa}]$", fontsize=THESIS_LABEL_FONT_SIZE)
    axis.legend(
        loc="best",
        fontsize=THESIS_TICK_FONT_SIZE,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.6",
    )

    return figure, axis


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    apply_thesis_style()

    pure_data = load_min_pressure_history(args.input)
    mixture_data = load_min_pressure_history(args.mixture_input)

    pure_min_timestep = get_global_min_pressure_timestep(pure_data)
    end_timestep = pure_min_timestep + float(args.end_offset)
    start_timestep = float(args.start_timestep)

    if start_timestep > end_timestep:
        raise ValueError(
            "The shared start timestep cannot exceed the shared end timestep. "
            f"Got start={start_timestep:g}, end={end_timestep:g}."
        )

    print(
        "Plotting both curves from timestep "
        f"{format_timestep(start_timestep)} through {format_timestep(end_timestep)} "
        f"(pure-fluid minimum occurs at timestep {format_timestep(pure_min_timestep)} "
        f"and the end offset is {format_timestep(args.end_offset)})."
    )

    needs_figure = (not args.no_show) or args.to_thesis or args.output is not None
    pure_plot_data = None
    mixture_plot_data = None
    figure = None
    if needs_figure:
        pure_plot_data = filter_history_to_timestep_range(
            pure_data,
            start_timestep,
            end_timestep,
        )
        mixture_plot_data = filter_history_to_timestep_range(
            mixture_data,
            start_timestep,
            end_timestep,
        )
        figure, _ = plot_min_pressure_histories(
            pure_plot_data,
            mixture_plot_data,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )

    if args.to_thesis:
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        print(f"Figure written to {output_path}")
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=600, bbox_inches="tight", pad_inches=0.02)
        print(f"Figure written to {args.output}")

    if figure is not None and args.no_show:
        plt.close(figure)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
