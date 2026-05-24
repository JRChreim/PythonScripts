"""Plot the location of minimum pressure events in the droplet breakup data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
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
    THESIS_TITLE_FONT_SIZE,
    add_thesis_export_argument,
    apply_thesis_style,
    latex_text,
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
DEFAULT_CAUSTICS_INPUT = Path(
    "/home/user/Documents/GitHub/PythonScripts/artifacts/Data/DropletAerobreakup/"
    "Caustics.csv"
)
DEFAULT_CUSPID_INPUT = Path(
    "/home/user/Documents/GitHub/PythonScripts/artifacts/Data/DropletAerobreakup/"
    "Cuspid.csv"
)
DEFAULT_OUTPUT = None
THESIS_EXPORT_STEM = "DAMPM"
REFERENCE_QUARTER_CIRCLE_COLOR = "#4C78A8"
REFERENCE_QUARTER_CIRCLE_ALPHA = 0.28
REFERENCE_QUARTER_CIRCLE_LINEWIDTH = 1.6
REFERENCE_QUARTER_CIRCLE_RADIUS = 0.5
SOURCE_GEOMETRY_RADIUS = 1.0
SOURCE_GEOMETRY_SCALE = REFERENCE_QUARTER_CIRCLE_RADIUS / SOURCE_GEOMETRY_RADIUS
THESIS_FIGURE_SIZE = thesis_figure_size(0.55)


def load_min_pressure_locations(
    filepath: Path,
) -> dict[str, np.ndarray]:
    """Load timestep, pressure, and location columns from the CSV file."""

    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    rows: list[tuple[float, float, float, float]] = []

    with filepath.open(newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        required_columns = {"timestep", "min_pres", "x", "y"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filepath}: {sorted(missing_columns)}"
            )

        for row in reader:
            if not row["timestep"] or not row["min_pres"] or not row["x"] or not row["y"]:
                continue

            rows.append(
                (
                    float(row["timestep"]),
                    float(row["min_pres"]),
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
        "min_pres_pa": data[:, 1],
        "min_pres_mpa": data[:, 1] / 1.0e6,
        "x": data[:, 2],
        "y": data[:, 3],
    }


def load_xy_locations(
    filepath: Path,
    *,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Load x/y geometry points from a CSV file."""

    if not filepath.exists():
        raise FileNotFoundError(f"Geometry file not found: {filepath}")

    rows: list[tuple[float, float]] = []

    with filepath.open(newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        required_columns = {"x", "y"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filepath}: {sorted(missing_columns)}"
            )

        for row in reader:
            if not row["x"] or not row["y"]:
                continue
            rows.append((float(row["x"]) * scale, float(row["y"]) * scale))

    if not rows:
        raise ValueError(f"No usable geometry rows were found in {filepath}")

    data = np.asarray(rows, dtype=float)
    return {"x": data[:, 0], "y": data[:, 1]}


def get_global_min_pressure_timestep(data: dict[str, np.ndarray]) -> float:
    """Return the timestep at which the minimum pressure becomes most negative."""

    min_index = int(np.argmin(data["min_pres_pa"]))
    return float(data["timestep"][min_index])


def filter_locations_to_max_timestep(
    data: dict[str, np.ndarray],
    max_timestep: float,
) -> dict[str, np.ndarray]:
    """Keep only rows up to and including the requested timestep."""

    mask = data["timestep"] <= float(max_timestep)
    if not np.any(mask):
        raise ValueError(
            f"No rows were found at or before timestep {max_timestep:g}"
        )

    return {key: values[mask] for key, values in data.items()}


def filter_locations_to_timestep_range(
    data: dict[str, np.ndarray],
    start_timestep: float,
    max_timestep: float,
) -> dict[str, np.ndarray]:
    """Keep only rows within the requested timestep interval."""

    if start_timestep > max_timestep:
        raise ValueError(
            "The start timestep must be less than or equal to the end timestep. "
            f"Got start={start_timestep:g}, end={max_timestep:g}."
        )

    range_filtered = filter_locations_to_max_timestep(data, max_timestep)
    mask = range_filtered["timestep"] >= float(start_timestep)
    if not np.any(mask):
        raise ValueError(
            f"No rows were found at or after timestep {start_timestep:g}"
        )

    return {key: values[mask] for key, values in range_filtered.items()}


def format_timestep(value: float) -> str:
    """Format a timestep for titles and console output."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def get_shared_pressure_limits(*datasets: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return a common min/max pressure range for multiple plotted datasets."""

    min_values = [float(dataset["min_pres_mpa"].min()) for dataset in datasets]
    max_values = [float(dataset["min_pres_mpa"].max()) for dataset in datasets]
    vmin = min(min_values)
    vmax = max(max_values)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0e-12
    return vmin, vmax


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the (x, y) locations where the minimum pressure occurs, "
            "with min_pres encoded as color."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file for the pure-fluid subplot containing timestep, min_pres, x, y, z columns.",
    )
    parser.add_argument(
        "--mixture-input",
        type=Path,
        default=DEFAULT_MIXTURE_INPUT,
        help="CSV file for the mixture subplot containing timestep, min_pres, x, y, z columns.",
    )
    parser.add_argument(
        "--caustics-input",
        type=Path,
        default=DEFAULT_CAUSTICS_INPUT,
        help="CSV file containing the Caustics curve geometry.",
    )
    parser.add_argument(
        "--cuspid-input",
        type=Path,
        default=DEFAULT_CUSPID_INPUT,
        help="CSV file containing the Cuspid marker geometry.",
    )
    parser.add_argument(
        "--max-timestep",
        type=float,
        default=None,
        help=(
            "Upper timestep limit for the plot. "
            "If omitted, the script uses the timestep where min_pres is most negative."
        ),
    )
    parser.add_argument(
        "--mixture-end-offset",
        type=float,
        default=0.0,
        help=(
            "Offset added to the pure-fluid end timestep to determine the mixture "
            "panel end timestep. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--start-timestep",
        type=float,
        default=0.0,
        help="Lower timestep limit for the plot. Defaults to 0.",
    )
    parser.add_argument(
        "--timestep",
        dest="max_timestep",
        type=float,
        help=argparse.SUPPRESS,
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
            "../Caltech-Thesis---JRChreim/Figures/DropletMinPressureMap.pdf."
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
        default=32.0,
        help="Marker size used for the scatter plot.",
    )
    parser.add_argument(
        "--cmap",
        default="bone",
        help="Matplotlib colormap used for min_pres.",
    )
    parser.add_argument(
        "--connect-points",
        action="store_true",
        help="Draw a thin line through the points in timestep order.",
    )
    parser.add_argument(
        "--highlight-global-min",
        action="store_true",
        help="Mark the single point with the most negative min_pres.",
    )
    return parser


def plot_min_pressure_locations(
    data: dict[str, np.ndarray],
    *,
    title: str | None = None,
    point_size: float = 32.0,
    cmap: str = "bone",
    connect_points: bool = False,
    highlight_global_min: bool = False,
    caustics_data: dict[str, np.ndarray] | None = None,
    cuspid_data: dict[str, np.ndarray] | None = None,
    show_geometry_legend: bool = False,
):
    """Make a 2D location plot colored by minimum pressure."""

    figure, axis = plt.subplots(figsize=THESIS_FIGURE_SIZE, constrained_layout=True)
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)
    plot_min_pressure_locations_on_axis(
        axis,
        data,
        point_size=point_size,
        cmap=cmap,
        connect_points=connect_points,
        highlight_global_min=highlight_global_min,
        caustics_data=caustics_data,
        cuspid_data=cuspid_data,
        show_geometry_legend=show_geometry_legend,
        show_ylabel=True,
        add_colorbar=True,
    )
    if title:
        axis.set_title(
            _ensure_latex_title(title),
            fontsize=THESIS_TITLE_FONT_SIZE,
        )

    return figure, axis


def plot_min_pressure_locations_on_axis(
    axis,
    data: dict[str, np.ndarray],
    *,
    point_size: float = 32.0,
    cmap: str = "bone",
    connect_points: bool = False,
    highlight_global_min: bool = False,
    caustics_data: dict[str, np.ndarray] | None = None,
    cuspid_data: dict[str, np.ndarray] | None = None,
    show_geometry_legend: bool = False,
    show_ylabel: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = True,
):
    """Plot a minimum-pressure trajectory on an existing axis."""

    x = data["x"]
    y = data["y"]
    min_pres_mpa = data["min_pres_mpa"]

    quarter_circle = Wedge(
        (0.0, 0.0),
        REFERENCE_QUARTER_CIRCLE_RADIUS,
        0.0,
        90.0,
        facecolor=REFERENCE_QUARTER_CIRCLE_COLOR,
        edgecolor=REFERENCE_QUARTER_CIRCLE_COLOR,
        alpha=REFERENCE_QUARTER_CIRCLE_ALPHA,
        linewidth=REFERENCE_QUARTER_CIRCLE_LINEWIDTH,
        zorder=0,
    )
    axis.add_patch(quarter_circle)

    if connect_points:
        # Optional context line for time evolution; off by default to keep the
        # plot readable when the path revisits nearby locations.
        axis.plot(x, y, color="0.65", linewidth=1.0, alpha=0.65, zorder=1)

    scatter = axis.scatter(
        x,
        y,
        c=min_pres_mpa,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        edgecolors="0.25",
        linewidths=0.35,
        zorder=2,
    )

    if caustics_data is not None:
        axis.plot(
            caustics_data["x"],
            caustics_data["y"],
            color="red",
            linewidth=2.0,
            label=r"$\mathrm{Caustics}$" if show_geometry_legend else None,
            zorder=3,
        )

    if cuspid_data is not None:
        axis.scatter(
            cuspid_data["x"],
            cuspid_data["y"],
            marker="s",
            s=42,
            facecolors="black",
            edgecolors="black",
            linewidths=0.6,
            label=r"$\mathrm{Caustic\ Cup}$" if show_geometry_legend else None,
            zorder=4,
            clip_on=False,
        )

    if highlight_global_min:
        min_index = int(np.argmin(min_pres_mpa))
        axis.scatter(
            [x[min_index]],
            [y[min_index]],
            s=120,
            marker="*",
            color="crimson",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    axis.set_xlim(0.0, REFERENCE_QUARTER_CIRCLE_RADIUS)
    axis.set_ylim(0.0, REFERENCE_QUARTER_CIRCLE_RADIUS)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x/D_0$", fontsize=THESIS_LABEL_FONT_SIZE)
    if show_ylabel:
        axis.set_ylabel(r"$y/D_0$", fontsize=THESIS_LABEL_FONT_SIZE)
    else:
        axis.set_ylabel("")
    axis.grid(True, alpha=0.25)
    axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)

    if add_colorbar:
        colorbar = axis.figure.colorbar(scatter, ax=axis, pad=0.03)
        colorbar.ax.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
        colorbar.set_label(
            r"$p_{\min}\ [\mathrm{MPa}]$",
            fontsize=THESIS_LABEL_FONT_SIZE,
            labelpad=8,
        )
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if show_geometry_legend:
        axis.legend(
            loc="upper right",
            fontsize=THESIS_TICK_FONT_SIZE,
            frameon=True,
            framealpha=0.9,
            edgecolor="0.6",
        )

    return scatter


def plot_min_pressure_subplots(
    pure_data: dict[str, np.ndarray],
    mixture_data: dict[str, np.ndarray],
    *,
    caustics_data: dict[str, np.ndarray] | None = None,
    cuspid_data: dict[str, np.ndarray] | None = None,
    point_size: float = 32.0,
    cmap: str = "bone",
    connect_points: bool = False,
    highlight_global_min: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot the pure-fluid and mixture analyses side by side."""

    if vmin is None or vmax is None:
        vmin, vmax = get_shared_pressure_limits(pure_data, mixture_data)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=THESIS_FIGURE_SIZE,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    pure_scatter = plot_min_pressure_locations_on_axis(
        axes[0],
        pure_data,
        point_size=point_size,
        cmap=cmap,
        connect_points=connect_points,
        highlight_global_min=highlight_global_min,
        caustics_data=caustics_data,
        cuspid_data=cuspid_data,
        show_geometry_legend=True,
        show_ylabel=True,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    plot_min_pressure_locations_on_axis(
        axes[1],
        mixture_data,
        point_size=point_size,
        cmap=cmap,
        connect_points=connect_points,
        highlight_global_min=highlight_global_min,
        caustics_data=caustics_data,
        cuspid_data=cuspid_data,
        show_geometry_legend=False,
        show_ylabel=False,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    axes[0].set_title(r"$\mathrm{Pure\ fluid}$", fontsize=THESIS_TITLE_FONT_SIZE)
    axes[1].set_title(r"$\mathrm{Mixture}$", fontsize=THESIS_TITLE_FONT_SIZE)

    colorbar = figure.colorbar(pure_scatter, ax=axes, pad=0.03)
    colorbar.ax.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    colorbar.set_label(
        r"$p_{\min}\ [\mathrm{MPa}]$",
        fontsize=THESIS_LABEL_FONT_SIZE,
        labelpad=8,
    )
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    return figure, axes


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    apply_thesis_style()
    pure_data = load_min_pressure_locations(args.input)
    mixture_data = load_min_pressure_locations(args.mixture_input)
    caustics_data = load_xy_locations(
        args.caustics_input,
        scale=SOURCE_GEOMETRY_SCALE,
    )
    cuspid_data = load_xy_locations(
        args.cuspid_input,
        scale=SOURCE_GEOMETRY_SCALE,
    )
    shared_vmin, shared_vmax = get_shared_pressure_limits(pure_data, mixture_data)

    if args.all_timesteps:
        start_timestep = None
        pure_end_timestep = None
        mixture_end_timestep = None
        pure_plot_data = pure_data
        mixture_plot_data = mixture_data
    else:
        start_timestep = float(args.start_timestep)
        pure_end_timestep = (
            float(args.max_timestep)
            if args.max_timestep is not None
            else get_global_min_pressure_timestep(pure_data)
        )
        mixture_end_timestep = pure_end_timestep + float(args.mixture_end_offset)
        pure_plot_data = filter_locations_to_timestep_range(
            pure_data,
            start_timestep,
            pure_end_timestep,
        )
        mixture_plot_data = filter_locations_to_timestep_range(
            mixture_data,
            start_timestep,
            mixture_end_timestep,
        )

    if pure_end_timestep is not None:
        print(
            "Plotting pure fluid from timestep "
            f"{format_timestep(start_timestep)} through "
            f"{format_timestep(pure_end_timestep)} and mixture from timestep "
            f"{format_timestep(start_timestep)} through "
            f"{format_timestep(mixture_end_timestep)} "
            f"(offset {format_timestep(args.mixture_end_offset)})."
        )

    figure = None
    if not args.no_show:
        figure, _ = plot_min_pressure_subplots(
            pure_plot_data,
            mixture_plot_data,
            caustics_data=caustics_data,
            cuspid_data=cuspid_data,
            point_size=args.point_size,
            cmap=args.cmap,
            connect_points=args.connect_points,
            highlight_global_min=args.highlight_global_min,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )

    if args.to_thesis:
        if figure is None:
            figure, _ = plot_min_pressure_subplots(
                pure_plot_data,
                mixture_plot_data,
                caustics_data=caustics_data,
                cuspid_data=cuspid_data,
                point_size=args.point_size,
                cmap=args.cmap,
                connect_points=args.connect_points,
                highlight_global_min=args.highlight_global_min,
                vmin=shared_vmin,
                vmax=shared_vmax,
            )
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        print(f"Figure written to {output_path}")
    elif args.output is not None:
        if figure is None:
            figure, _ = plot_min_pressure_subplots(
                pure_plot_data,
                mixture_plot_data,
                caustics_data=caustics_data,
                cuspid_data=cuspid_data,
                point_size=args.point_size,
                cmap=args.cmap,
                connect_points=args.connect_points,
                highlight_global_min=args.highlight_global_min,
                vmin=shared_vmin,
                vmax=shared_vmax,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=600, bbox_inches="tight", pad_inches=0.02)
        print(f"Figure written to {args.output}")

    if figure is not None and args.no_show:
        plt.close(figure)
    if not args.no_show:
        plt.show()


def _ensure_latex_title(title: str) -> str:
    if "$" in title:
        return title
    return latex_text(title)


if __name__ == "__main__":
    main()
