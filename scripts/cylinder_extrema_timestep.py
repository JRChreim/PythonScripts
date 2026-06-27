"""Plot extrema histories versus timestep for the cylinder breakup data."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
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
    THESIS_TITLE_FONT_SIZE,
    add_show_titles_argument,
    add_thesis_export_argument,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)

DEFAULT_DATA_ROOT = Path(
    "/home/user/Documents/GitHub/PythonScripts/artifacts/Data/CylinderAerobreakup"
)
DEFAULT_CASE = "Mixture"
DEFAULT_EXTREMUM = "max"
DEFAULT_VARIABLE = "alpha_rho2"
THESIS_EXPORT_STEM = "CylinderExtremaTimestep"
THESIS_FIGURE_SIZE = thesis_figure_size(0.42)
LINEWIDTH = 1.8
LINE_COLORS = {
    "min": "#d62728",
    "max": "#1f77b4",
}
SECONDARY_LINE_COLORS = {
    "min": "#2ca02c",
    "max": "#ff7f0e",
}
GUIDE_LINE_COLOR = "0.55"
GUIDE_LINE_STYLE = "--"
GUIDE_LINE_WIDTH = 1.0
CASE_OPTIONS = ("Mixture", "PureFluid", "Subgrid")
EXTREMUM_OPTIONS = ("min", "max")
VARIABLE_LABELS = {
    "alpha_rho2": r"\alpha_{\rho_2}",
    "alpha2": r"\alpha_2",
    "pres": r"p",
    "Y2": r"Y_2",
}
VARIABLE_UNITS = {
    "alpha_rho2": "",
    "alpha2": "",
    "pres": r"\ [\mathrm{Pa}]",
    "Y2": "",
}
EXTREMUM_FUNCTIONS = {
    "min": r"\min",
    "max": r"\max",
}


@dataclass(frozen=True)
class PanelSpec:
    case: str
    extremum: str
    variable: str
    file_stem: str
    label: str


def _escape_latex_text(text: str) -> str:
    """Escape a plain-text string for use inside a LaTeX math label."""

    escaped = text
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("$", r"\$"),
    ):
        escaped = escaped.replace(source, replacement)
    return escaped


def resolve_variable_spec(variable: str, file_stem: str | None = None) -> dict[str, str]:
    """Resolve the display label and file stem for a requested variable."""

    canonical_variable = variable
    if canonical_variable in VARIABLE_LABELS:
        return {
            "variable": canonical_variable,
            "file_stem": file_stem or canonical_variable,
            "label": VARIABLE_LABELS[canonical_variable],
            "units": VARIABLE_UNITS[canonical_variable],
        }

    resolved_file_stem = file_stem or canonical_variable
    return {
        "variable": canonical_variable,
        "file_stem": resolved_file_stem,
        "label": rf"\mathrm{{{_escape_latex_text(canonical_variable)}}}",
        "units": "",
    }


def build_panel_spec(
    case: str,
    extremum: str,
    variable: str,
    file_stem: str | None = None,
) -> PanelSpec:
    """Create a panel specification from its CLI components."""

    variable_spec = resolve_variable_spec(variable, file_stem=file_stem)
    return PanelSpec(
        case=case,
        extremum=extremum,
        variable=variable_spec["variable"],
        file_stem=variable_spec["file_stem"],
        label=variable_spec["label"],
    )


def parse_panel_spec(raw_spec: str) -> PanelSpec:
    """Parse a subplot specification of the form CASE:EXTREMUM:VARIABLE[:FILE_STEM]."""

    parts = raw_spec.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "Panel specs must use CASE:EXTREMUM:VARIABLE[:FILE_STEM]. "
            f"Got {raw_spec!r}."
        )

    case, extremum, variable = parts[:3]
    file_stem = parts[3] if len(parts) == 4 else None

    if case not in CASE_OPTIONS:
        raise argparse.ArgumentTypeError(
            f"Unsupported case {case!r}. Choose from {CASE_OPTIONS}."
        )
    if extremum not in EXTREMUM_OPTIONS:
        raise argparse.ArgumentTypeError(
            f"Unsupported extremum {extremum!r}. Choose from {EXTREMUM_OPTIONS}."
        )

    return build_panel_spec(case, extremum, variable, file_stem=file_stem)


def resolve_input_path(
    data_root: Path,
    case: str,
    extremum: str,
    file_stem: str,
) -> Path:
    """Build the CSV path for a requested case, variable, and extremum."""

    return data_root / case / f"{file_stem}_{extremum}_location.csv"


def load_extrema_history(filepath: Path) -> dict[str, np.ndarray]:
    """Load timestep and extremum-value columns from a CSV file."""

    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    rows: list[tuple[float, float]] = []

    with filepath.open(newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        required_columns = {"timestep", "value"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filepath}: {sorted(missing_columns)}"
            )

        for row in reader:
            if not row["timestep"] or not row["value"]:
                continue
            rows.append((float(row["timestep"]), float(row["value"])))

    if not rows:
        raise ValueError(f"No usable rows were found in {filepath}")

    rows.sort(key=lambda item: item[0])
    data = np.asarray(rows, dtype=float)

    return {
        "timestep": data[:, 0],
        "value": data[:, 1],
    }


def format_timestep(value: float) -> str:
    """Format a timestep for console output."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


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


def get_shared_value_limits(*datasets: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return a common y-axis range for one or more history datasets."""

    min_values = [float(dataset["value"].min()) for dataset in datasets]
    max_values = [float(dataset["value"].max()) for dataset in datasets]
    vmin = min(min_values)
    vmax = max(max_values)

    if np.isclose(vmin, vmax):
        pad = 1.0e-12
    else:
        pad = 0.06 * (vmax - vmin)

    return vmin - pad, vmax + pad


def get_value_axis_label(variable_spec: dict[str, str], extremum: str) -> str:
    """Build the y-axis label for the selected variable and extremum."""

    try:
        extremum_function = EXTREMUM_FUNCTIONS[extremum]
    except KeyError as exc:
        raise ValueError(f"Unsupported extremum {extremum!r}.") from exc

    return rf"${extremum_function}({variable_spec['label']}){variable_spec['units']}$"


def panel_to_variable_spec(panel: PanelSpec) -> dict[str, str]:
    """Convert a panel specification into the variable metadata used for labels."""

    return {
        "variable": panel.variable,
        "file_stem": panel.file_stem,
        "label": panel.label,
        "units": VARIABLE_UNITS.get(panel.variable, ""),
    }


def add_reference_lines(
    axis,
    *,
    vlines: list[float] | tuple[float, ...] | None = None,
    hlines: list[float] | tuple[float, ...] | None = None,
):
    """Add shared vertical and horizontal guide lines to an axis."""

    for value in vlines or ():
        axis.axvline(
            value,
            color=GUIDE_LINE_COLOR,
            linestyle=GUIDE_LINE_STYLE,
            linewidth=GUIDE_LINE_WIDTH,
            zorder=0,
            label="_nolegend_",
        )
    for value in hlines or ():
        axis.axhline(
            value,
            color=GUIDE_LINE_COLOR,
            linestyle=GUIDE_LINE_STYLE,
            linewidth=GUIDE_LINE_WIDTH,
            zorder=0,
            label="_nolegend_",
        )


def plot_secondary_history_on_axis(
    axis,
    data: dict[str, np.ndarray],
    *,
    extremum: str,
    variable_spec: dict[str, str],
):
    """Plot the secondary history on a twinned axis."""

    try:
        line_color = SECONDARY_LINE_COLORS[extremum]
    except KeyError as exc:
        raise ValueError(f"Unsupported extremum {extremum!r}.") from exc

    axis.plot(
        data["timestep"],
        data["value"],
        color=line_color,
        linewidth=LINEWIDTH,
        linestyle="--",
    )
    axis.set_ylabel(
        get_value_axis_label(variable_spec, extremum),
        fontsize=THESIS_LABEL_FONT_SIZE,
        color=line_color,
    )
    axis.tick_params(
        axis="y",
        labelsize=THESIS_TICK_FONT_SIZE,
        colors=line_color,
    )
    axis.spines["right"].set_color(line_color)
    axis.set_ylim(get_shared_value_limits(data))
    return axis


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot min/max histories versus timestep for the cylinder breakup data."
        )
    )
    add_thesis_export_argument(parser, default_stem=None)
    add_show_titles_argument(parser)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Root folder containing the Mixture, PureFluid, and Subgrid "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--case",
        choices=CASE_OPTIONS,
        default=DEFAULT_CASE,
        help="Data family to load. Defaults to Mixture.",
    )
    parser.add_argument(
        "--extremum",
        choices=EXTREMUM_OPTIONS,
        default=DEFAULT_EXTREMUM,
        help="Choose whether to plot the minimum or maximum history.",
    )
    parser.add_argument(
        "--variable",
        default=DEFAULT_VARIABLE,
        help=(
            "Variable to plot. Known names include alpha_rho2, alpha2, pres, "
            "and Y2, but any matching variable stem is allowed."
        ),
    )
    parser.add_argument(
        "--file-stem",
        default=None,
        help=(
            "Override the CSV filename stem if it differs from the variable "
            "name. By default, the script looks for <variable>_<min|max>_location.csv."
        ),
    )
    parser.add_argument(
        "--start-timestep",
        type=float,
        default=0.0,
        help="Lower timestep limit for the plot. Defaults to 0.",
    )
    parser.add_argument(
        "--end-timestep",
        type=float,
        default=None,
        help=(
            "Upper timestep limit for the plot. If omitted, the script uses the "
            "largest timestep available in the selected CSV."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save the figure, for example "
            "../Caltech-Thesis---JRChreim/Figures/CylinderExtremaTimestep.pdf."
        ),
    )
    parser.add_argument(
        "--no-show",
        "--no-plot",
        dest="no_show",
        action="store_true",
        help="Build or save the figure without opening an interactive window.",
    )
    parser.add_argument(
        "--panel",
        action="append",
        type=parse_panel_spec,
        help=(
            "Add a subplot spec in the form CASE:EXTREMUM:VARIABLE[:FILE_STEM]. "
            "Repeat this flag to build a grid of plots. If omitted, the script "
            "uses the single-plot options above."
        ),
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=None,
        help=(
            "Optional number of subplot columns. If omitted, the script chooses "
            "a near-square layout automatically."
        ),
    )
    parser.add_argument(
        "--right-panel",
        action="append",
        type=parse_panel_spec,
        help=(
            "Optional right-axis subplot spec in the form "
            "CASE:EXTREMUM:VARIABLE[:FILE_STEM]. Repeat this flag in the same "
            "order as --panel to overlay a twinned right axis on each subplot."
        ),
    )
    parser.add_argument(
        "--vline",
        action="append",
        type=float,
        default=[],
        help=(
            "Add a vertical guide line at the given x value. Repeat the flag "
            "to add multiple guides. The guides are drawn on every panel."
        ),
    )
    parser.add_argument(
        "--hline",
        action="append",
        type=float,
        default=[],
        help=(
            "Add a horizontal guide line at the given y value. Repeat the flag "
            "to add multiple guides. The guides are drawn on every panel."
        ),
    )
    return parser


def plot_extrema_history(
    data: dict[str, np.ndarray],
    *,
    case: str,
    extremum: str,
    variable_spec: dict[str, str],
    start_timestep: float,
    end_timestep: float,
    show_title: bool = False,
    vlines: list[float] | tuple[float, ...] | None = None,
    hlines: list[float] | tuple[float, ...] | None = None,
    secondary_data: dict[str, np.ndarray] | None = None,
    secondary_extremum: str | None = None,
    secondary_variable_spec: dict[str, str] | None = None,
):
    """Create a single extrema-history plot."""

    figure, axis = plt.subplots(figsize=THESIS_FIGURE_SIZE, constrained_layout=True)
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)
    plot_extrema_history_on_axis(
        axis,
        data,
        case=case,
        extremum=extremum,
        variable_spec=variable_spec,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        show_title=show_title,
        vlines=vlines,
        hlines=hlines,
        secondary_data=secondary_data,
        secondary_extremum=secondary_extremum,
        secondary_variable_spec=secondary_variable_spec,
    )

    return figure, axis


def plot_extrema_history_on_axis(
    axis,
    data: dict[str, np.ndarray],
    *,
    case: str,
    extremum: str,
    variable_spec: dict[str, str],
    start_timestep: float,
    end_timestep: float,
    show_title: bool = False,
    show_xlabel: bool = True,
    vlines: list[float] | tuple[float, ...] | None = None,
    hlines: list[float] | tuple[float, ...] | None = None,
    secondary_data: dict[str, np.ndarray] | None = None,
    secondary_extremum: str | None = None,
    secondary_variable_spec: dict[str, str] | None = None,
):
    """Plot one extrema history on an existing axis."""

    axis.plot(
        data["timestep"],
        data["value"],
        color=LINE_COLORS[extremum],
        linewidth=LINEWIDTH,
    )
    axis.grid(True, alpha=0.25)
    axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    if show_xlabel:
        axis.set_xlabel(r"$n_{\mathrm{step}}$", fontsize=THESIS_LABEL_FONT_SIZE)
    else:
        axis.set_xlabel("")
    axis.set_ylabel(
        get_value_axis_label(variable_spec, extremum),
        fontsize=THESIS_LABEL_FONT_SIZE,
    )
    y_min, y_max = get_shared_value_limits(data)
    x_min = float(start_timestep)
    x_max = float(end_timestep)
    if vlines:
        x_min = min(x_min, min(float(value) for value in vlines))
        x_max = max(x_max, max(float(value) for value in vlines))
    if hlines:
        y_min = min(y_min, min(float(value) for value in hlines))
        y_max = max(y_max, max(float(value) for value in hlines))
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    add_reference_lines(axis, vlines=vlines, hlines=hlines)

    if show_title:
        axis.set_title(
            f"{case}: {extremum}({variable_spec['variable']})",
            fontsize=THESIS_TITLE_FONT_SIZE,
        )

    if secondary_data is not None:
        if secondary_extremum is None or secondary_variable_spec is None:
            raise ValueError(
                "Secondary plotting requires both secondary_extremum and "
                "secondary_variable_spec."
            )
        right_axis = axis.twinx()
        plot_secondary_history_on_axis(
            right_axis,
            secondary_data,
            extremum=secondary_extremum,
            variable_spec=secondary_variable_spec,
        )
        right_axis.set_xlim(start_timestep, end_timestep)
        return right_axis

    return None


def compute_subplot_grid(num_panels: int, ncols: int | None = None) -> tuple[int, int]:
    """Choose a near-square subplot grid for the requested number of panels."""

    if num_panels < 1:
        raise ValueError("At least one panel is required.")

    if ncols is None:
        ncols = int(math.ceil(math.sqrt(num_panels)))
    if ncols < 1:
        raise ValueError("The number of subplot columns must be at least 1.")

    nrows = int(math.ceil(num_panels / ncols))
    return nrows, ncols


def plot_extrema_grid(
    panel_specs: list[PanelSpec],
    *,
    right_panel_specs: list[PanelSpec] | None,
    data_root: Path,
    start_timestep: float,
    end_timestep: float,
    show_titles: bool = False,
    ncols: int | None = None,
    vlines: list[float] | tuple[float, ...] | None = None,
    hlines: list[float] | tuple[float, ...] | None = None,
):
    """Plot multiple extrema histories in a grid of subplots."""

    if right_panel_specs is not None and len(right_panel_specs) != len(panel_specs):
        raise ValueError(
            "The number of right panels must match the number of left panels."
        )

    nrows, ncols = compute_subplot_grid(len(panel_specs), ncols=ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(THESIS_FIGURE_SIZE[0], THESIS_FIGURE_SIZE[1] * nrows),
        constrained_layout=True,
        sharex=True,
    )
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)

    for index, panel in enumerate(panel_specs):
        row = index // ncols
        col = index % ncols
        axis = axes_array[row, col]
        input_path = resolve_input_path(
            data_root,
            panel.case,
            panel.extremum,
            panel.file_stem,
        )
        data = load_extrema_history(input_path)
        plot_data = filter_history_to_timestep_range(
            data,
            start_timestep,
            end_timestep,
        )

        secondary_data = None
        secondary_variable_spec = None
        secondary_extremum = None
        if right_panel_specs is not None:
            right_panel = right_panel_specs[index]
            right_input_path = resolve_input_path(
                data_root,
                right_panel.case,
                right_panel.extremum,
                right_panel.file_stem,
            )
            secondary_data = filter_history_to_timestep_range(
                load_extrema_history(right_input_path),
                start_timestep,
                end_timestep,
            )
            secondary_variable_spec = panel_to_variable_spec(right_panel)
            secondary_extremum = right_panel.extremum

        axis.set_title(
            panel.case if not show_titles else f"{panel.case}: {panel.extremum}({panel.variable})",
            fontsize=THESIS_TITLE_FONT_SIZE,
        )
        plot_extrema_history_on_axis(
            axis,
            plot_data,
            case=panel.case,
            extremum=panel.extremum,
            variable_spec=panel_to_variable_spec(panel),
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            show_title=False,
            show_xlabel=(row == nrows - 1),
            vlines=vlines,
            hlines=hlines,
            secondary_data=secondary_data,
            secondary_extremum=secondary_extremum,
            secondary_variable_spec=secondary_variable_spec,
        )

    for index in range(len(panel_specs), nrows * ncols):
        row = index // ncols
        col = index % ncols
        axes_array[row, col].axis("off")

    return figure, axes_array


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    apply_thesis_style()

    start_timestep = float(args.start_timestep)
    if args.panel:
        panel_specs = list(args.panel)
    else:
        panel_specs = [build_panel_spec(args.case, args.extremum, args.variable, args.file_stem)]

    right_panel_specs: list[PanelSpec] | None = None
    if args.right_panel:
        right_panel_specs = list(args.right_panel)
        if len(right_panel_specs) != len(panel_specs):
            raise ValueError(
                "The number of --right-panel entries must match the number of "
                "--panel entries."
            )

    if args.end_timestep is not None:
        end_timestep = float(args.end_timestep)
    else:
        first_input_path = resolve_input_path(
            args.data_root,
            panel_specs[0].case,
            panel_specs[0].extremum,
            panel_specs[0].file_stem,
        )
        first_data = load_extrema_history(first_input_path)
        end_timestep = float(first_data["timestep"].max())

    thesis_stem = args.thesis_stem or (
        f"{THESIS_EXPORT_STEM}_grid"
        if len(panel_specs) > 1
        else f"{THESIS_EXPORT_STEM}_{panel_specs[0].case}_{panel_specs[0].extremum}_{panel_specs[0].file_stem}"
    )

    print(
        f"Plotting {len(panel_specs)} panel(s) from timestep "
        f"{format_timestep(start_timestep)} through {format_timestep(end_timestep)}."
    )

    figure = None
    if not args.no_show or args.to_thesis or args.output is not None:
        if len(panel_specs) == 1:
            panel = panel_specs[0]
            input_path = resolve_input_path(
                args.data_root,
                panel.case,
                panel.extremum,
                panel.file_stem,
            )
            print(f"Loading {input_path}")
            data = load_extrema_history(input_path)
            plot_data = filter_history_to_timestep_range(
                data,
                start_timestep,
                end_timestep,
            )
            secondary_data = None
            secondary_extremum = None
            secondary_variable_spec = None
            if right_panel_specs is not None:
                right_panel = right_panel_specs[0]
                right_input_path = resolve_input_path(
                    args.data_root,
                    right_panel.case,
                    right_panel.extremum,
                    right_panel.file_stem,
                )
                print(f"Loading {right_input_path}")
                secondary_data = filter_history_to_timestep_range(
                    load_extrema_history(right_input_path),
                    start_timestep,
                    end_timestep,
                )
                secondary_extremum = right_panel.extremum
                secondary_variable_spec = panel_to_variable_spec(right_panel)
            figure, _ = plot_extrema_history(
                plot_data,
                case=panel.case,
                extremum=panel.extremum,
                variable_spec=panel_to_variable_spec(panel),
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                show_title=args.show_titles,
                vlines=args.vline,
                hlines=args.hline,
                secondary_data=secondary_data,
                secondary_extremum=secondary_extremum,
                secondary_variable_spec=secondary_variable_spec,
            )
        else:
            nrows, ncols = compute_subplot_grid(len(panel_specs), ncols=args.ncols)
            print(f"Using a {nrows} x {ncols} subplot grid.")
            figure, _ = plot_extrema_grid(
                panel_specs,
                right_panel_specs=right_panel_specs,
                data_root=args.data_root,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                show_titles=args.show_titles,
                ncols=args.ncols,
                vlines=args.vline,
                hlines=args.hline,
            )

    if args.to_thesis:
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=thesis_stem,
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
