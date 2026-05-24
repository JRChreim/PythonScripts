"""Plot the locations of maximum alpha_rho2 events in the droplet breakup data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

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
THESIS_EXPORT_STEM = "AlphaRho2Map"
THESIS_FIGURE_SIZE = thesis_figure_size(0.55)
POINT_COLOR_MAP = "bone"
POINT_SIZE = 32.0


def load_max_alpha_locations(filepath: Path) -> dict[str, np.ndarray]:
    """Load timestep, max alpha_rho2, and location columns from the CSV file."""

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


def filter_locations_to_timestep_range(
    data: dict[str, np.ndarray],
    start_timestep: float,
    end_timestep: float,
) -> dict[str, np.ndarray]:
    """Keep only rows within the requested timestep interval."""

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
    """Return a common min/max range for one or more max-alpha datasets."""

    min_values = [float(dataset["max_alpha_rho2"].min()) for dataset in datasets]
    max_values = [float(dataset["max_alpha_rho2"].max()) for dataset in datasets]
    vmin = min(min_values)
    vmax = max(max_values)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0e-12
    return vmin, vmax


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the (x, y) locations where max(alpha_rho2) occurs, "
            "with max(alpha_rho2) encoded as color."
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
            "../Caltech-Thesis---JRChreim/Figures/DropletMaxAlphaRho2Map.pdf."
        ),
    )
    parser.add_argument(
        "--no-show",
        "--no-plot",
        dest="no_show",
        action="store_true",
        help="Skip showing the figure window unless an output file is requested.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=POINT_SIZE,
        help="Marker size used for the scatter plot.",
    )
    parser.add_argument(
        "--cmap",
        default=POINT_COLOR_MAP,
        help="Matplotlib colormap used for max_alpha_rho2.",
    )
    parser.add_argument(
        "--connect-points",
        action="store_true",
        help="Draw a thin line through the points in timestep order.",
    )
    parser.add_argument(
        "--highlight-global-max",
        action="store_true",
        help="Mark the single point with the largest max_alpha_rho2.",
    )
    return parser


def plot_max_alpha_locations_on_axis(
    axis,
    data: dict[str, np.ndarray],
    *,
    point_size: float = POINT_SIZE,
    cmap: str = POINT_COLOR_MAP,
    connect_points: bool = False,
    highlight_global_max: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a max-alpha trajectory on an existing axis."""

    x = data["x"]
    y = data["y"]
    max_alpha_rho2 = data["max_alpha_rho2"]

    if connect_points:
        # Optional context line for time evolution; off by default to keep the
        # plot readable when the path revisits nearby locations.
        axis.plot(x, y, color="0.65", linewidth=1.0, alpha=0.65, zorder=1)

    scatter = axis.scatter(
        x,
        y,
        c=max_alpha_rho2,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        edgecolors="0.25",
        linewidths=0.35,
        zorder=2,
    )

    if highlight_global_max:
        max_index = int(np.argmax(max_alpha_rho2))
        axis.scatter(
            [x[max_index]],
            [y[max_index]],
            s=120,
            marker="*",
            color="crimson",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_pad = 0.05 * (x_max - x_min) if not np.isclose(x_min, x_max) else 1.0e-3
    y_pad = 0.08 * (y_max - y_min) if not np.isclose(y_min, y_max) else 1.0e-3

    axis.set_xlim(x_min - x_pad, x_max + x_pad)
    axis.set_ylim(y_min - y_pad, y_max + y_pad)
    axis.set_aspect("auto")
    axis.set_xlabel(r"$x/D_0$", fontsize=THESIS_LABEL_FONT_SIZE)
    axis.set_ylabel(r"$y/D_0$", fontsize=THESIS_LABEL_FONT_SIZE)
    axis.grid(True, alpha=0.25)
    axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)

    colorbar = axis.figure.colorbar(scatter, ax=axis, pad=0.03)
    colorbar.ax.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    colorbar.set_label(
        r"$\max(\alpha_{\rho_2})$",
        fontsize=THESIS_LABEL_FONT_SIZE,
        labelpad=8,
    )
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    return scatter


def plot_max_alpha_locations(
    data: dict[str, np.ndarray],
    *,
    point_size: float = POINT_SIZE,
    cmap: str = POINT_COLOR_MAP,
    connect_points: bool = False,
    highlight_global_max: bool = False,
):
    """Create a max-alpha location plot."""

    figure, axis = plt.subplots(figsize=THESIS_FIGURE_SIZE, constrained_layout=True)
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)
    plot_max_alpha_locations_on_axis(
        axis,
        data,
        point_size=point_size,
        cmap=cmap,
        connect_points=connect_points,
        highlight_global_max=highlight_global_max,
    )
    return figure, axis


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    apply_thesis_style()

    data = load_max_alpha_locations(args.input)
    global_max_timestep = get_global_max_alpha_timestep(data)
    end_timestep = global_max_timestep + float(args.end_offset)
    start_timestep = float(args.start_timestep)

    if args.all_timesteps:
        plot_data = data
    else:
        plot_data = filter_locations_to_timestep_range(
            data,
            start_timestep,
            end_timestep,
        )

    if not args.all_timesteps:
        print(
            "Plotting max(alpha_rho2) from timestep "
            f"{format_timestep(start_timestep)} through {format_timestep(end_timestep)} "
            f"(global maximum occurs at timestep {format_timestep(global_max_timestep)})."
        )
    else:
        print("Plotting all timesteps in the max(alpha_rho2) CSV.")

    max_index = int(np.argmax(data["max_alpha_rho2"]))
    print(
        "Global max alpha_rho2 = "
        f"{data['max_alpha_rho2'][max_index]:g} at timestep "
        f"{format_timestep(data['timestep'][max_index])} "
        f"and location ({data['x'][max_index]:g}, {data['y'][max_index]:g})."
    )

    figure = None
    if not args.no_show:
        figure, _ = plot_max_alpha_locations(
            plot_data,
            point_size=args.point_size,
            cmap=args.cmap,
            connect_points=args.connect_points,
            highlight_global_max=args.highlight_global_max,
        )

    if args.to_thesis:
        if figure is None:
            figure, _ = plot_max_alpha_locations(
                plot_data,
                point_size=args.point_size,
                cmap=args.cmap,
                connect_points=args.connect_points,
                highlight_global_max=args.highlight_global_max,
            )
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        print(f"Figure written to {output_path}")
    elif args.output is not None:
        if figure is None:
            figure, _ = plot_max_alpha_locations(
                plot_data,
                point_size=args.point_size,
                cmap=args.cmap,
                connect_points=args.connect_points,
                highlight_global_max=args.highlight_global_max,
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
