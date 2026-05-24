from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.io.mfc_binary import (
    discover_mfc_binary_snapshot_directory,
    discover_mfc_binary_steps,
    load_mfc_binary_snapshot,
)
from src.plots.publication import (
    THESIS_TICK_FONT_SIZE,
    THESIS_TITLE_FONT_SIZE,
    add_thesis_export_argument,
    apply_thesis_style,
    latex_text,
    save_thesis_figure_from_args,
    thesis_figure_size,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIVE_EQN_FOLDER = Path(
    "/disk/simulations/Relaxation/Thesis/ExpansionTube/pT/5Eqn/binary"
)
DEFAULT_SIX_EQN_FOLDER = Path(
    "/disk/simulations/Relaxation/Thesis/ExpansionTube/pT/6Eqn/binary"
)
DEFAULT_VARIABLES = (
    "alpha_rho1",
    "alpha_rho2",
    "alpha_rho3",
    "pres",
    "vel1",
    "alpha1",
    "alpha2",
    "alpha3",
)
DEFAULT_OVERVIEW_PERCENTAGES = (0.0, 50.0, 100.0)
DEFAULT_ZOOM_VARIABLES = ("pres", "vel1", "alpha_rho1", "alpha1")
MODEL_STYLES = {
    "5Eqn": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.7},
    "6Eqn": {"color": "#333333", "linestyle": "--", "linewidth": 1.7},
}
MODEL_LABELS = {
    "5Eqn": r"$\mathrm{5\mbox{-}equation}$",
    "6Eqn": r"$\mathrm{6\mbox{-}equation}$",
}
FIELD_LABELS = {
    "alpha_rho1": r"$m_1\ [\mathrm{kg\,m^{-3}}]$",
    "alpha_rho2": r"$m_2\ [\mathrm{kg\,m^{-3}}]$",
    "alpha_rho3": r"$m_3\ [\mathrm{kg\,m^{-3}}]$",
    "alpha1": r"$\alpha_1$",
    "alpha2": r"$\alpha_2$",
    "alpha3": r"$\alpha_3$",
    "pres": r"$p\ [\mathrm{Pa}]$",
    "vel1": r"$u\ [\mathrm{m\,s^{-1}}]$",
}

OVERVIEW_FIGURE_SIZE = thesis_figure_size(1.80)
ZOOM_FIGURE_SIZE = thesis_figure_size(0.95)
SUMMARY_FIGURE_SIZE = thesis_figure_size(1.05)
ZOOM_HALF_WIDTH = 0.025
ZOOM_Y_PADDING_FRACTION = 0.01
INSET_REL_WIDTH = "42%"
INSET_REL_HEIGHT = "42%"
DEFAULT_INSET_LOCATION = "southeast"
INSET_LOCATION_MAP = {
    "southeast": "lower right",
    "southwest": "lower left",
    "northeast": "upper right",
    "northwest": "upper left",
    "east": "center right",
    "west": "center left",
}
INSET_BORDERPAD = 0.55
INSET_EDGE_COLOR = "black"
INSET_EDGE_WIDTH = 0.8
INSET_EDGE_DASH = (0, (3.0, 2.0))


@dataclass(frozen=True)
class SnapshotSelection:
    step: int
    percent: float


@dataclass(frozen=True)
class DifferenceSummary:
    max_abs: float
    step: int
    percent: float
    x: float
    value_5eqn: float
    value_6eqn: float
    field_scale: float

    @property
    def normalized_max_abs(self) -> float:
        if self.field_scale == 0.0:
            return 0.0
        return self.max_abs / self.field_scale


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 5Eqn and 6Eqn MFC binary outputs for a binary case."
        )
    )
    add_thesis_export_argument(parser, default_stem=None)
    parser.add_argument(
        "--five-eqn-folder",
        type=Path,
        default=DEFAULT_FIVE_EQN_FOLDER,
        help="Path to the 5Eqn case directory or its binary snapshot folder.",
    )
    parser.add_argument(
        "--six-eqn-folder",
        type=Path,
        default=DEFAULT_SIX_EQN_FOLDER,
        help="Path to the 6Eqn case directory or its binary snapshot folder.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help=(
            "Variables to show in the overview figure. The default set includes "
            "m_1, m_2, m_3, pres, vel1, alpha1, alpha2, "
            "and alpha3."
        ),
    )
    parser.add_argument(
        "--inset-location",
        type=_parse_inset_location,
        default=DEFAULT_INSET_LOCATION,
        help=(
            "Inset location for the zoom figure. Use southeast, southwest, "
            "northeast, northwest, east, or west."
        ),
    )
    parser.add_argument(
        "--zoom-xlimits",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help=(
            "Manually set the x-limits for all zoom boxes and insets. If "
            "omitted, the script centers each inset on the largest-difference "
            "location."
        ),
    )
    parser.add_argument(
        "--percentages",
        nargs="+",
        type=float,
        help=(
            "Snapshot percentages to plot in the overview figure. If omitted, "
            "the script uses 0, 50, and 100 percent of the simulation."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help=(
            "Explicit saved step numbers to plot in the overview figure. Use "
            "this instead of --percentages when you want exact saved outputs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save the overview figure. Defaults to a "
            "case-organized folder under artifacts/figures/mfc/<tube>/<mode>/ unless "
            "--to-thesis is used."
        ),
    )
    parser.add_argument(
        "--zoom-output",
        type=Path,
        help=(
            "Optional path to save the zoom/inset figure. Defaults to the "
            "overview output path with _zoom appended."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help=(
            "Optional path to save the summary table figure. Defaults to the "
            "overview output path with _summary appended."
        ),
    )
    parser.add_argument(
        "--case-label",
        help=(
            "Optional label used to distinguish output filenames and thesis "
            "exports. If omitted, the script infers a label from the folder "
            "path, such as ExpansionTube_pT or ShockTube_pTg."
        ),
    )
    parser.add_argument(
        "--title",
        help="Optional custom title for the overview figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figures without opening an interactive window.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)

    if args.steps is not None and args.percentages is not None:
        raise ValueError("Specify either --steps or --percentages, not both.")

    five_eqn_directory = discover_mfc_binary_snapshot_directory(args.five_eqn_folder)
    six_eqn_directory = discover_mfc_binary_snapshot_directory(args.six_eqn_folder)
    case_label = args.case_label or _infer_case_label(five_eqn_directory)
    thesis_stem = args.thesis_stem or f"MFC_{case_label}_5Eqn_vs_6Eqn"
    default_overview_output, default_zoom_output, default_summary_output = (
        _build_default_output_paths(case_label)
    )
    manual_zoom_xlim = None
    if args.zoom_xlimits is not None:
        lower, upper = (float(args.zoom_xlimits[0]), float(args.zoom_xlimits[1]))
        if upper <= lower:
            raise ValueError("--zoom-xlimits must satisfy XMIN < XMAX.")
        manual_zoom_xlim = (lower, upper)

    shared_steps = _discover_shared_steps(five_eqn_directory, six_eqn_directory)
    if not shared_steps:
        raise ValueError("The 5Eqn and 6Eqn cases do not share any saved steps.")

    comparison_summary = _build_difference_summary(
        five_eqn_directory,
        six_eqn_directory,
        shared_steps,
    )

    selected_snapshots = _select_snapshot_percentages(
        shared_steps,
        requested_steps=args.steps,
        requested_percentages=args.percentages,
    )

    selected_steps = [selection.step for selection in selected_snapshots]
    zoom_steps = [
        comparison_summary[variable].step
        for variable in DEFAULT_ZOOM_VARIABLES
        if variable in comparison_summary
    ]
    loaded_steps = sorted(dict.fromkeys(selected_steps + zoom_steps))
    loaded_cases = {
        "5Eqn": {
            step: load_mfc_binary_snapshot(five_eqn_directory / f"{step}.dat")
            for step in loaded_steps
        },
        "6Eqn": {
            step: load_mfc_binary_snapshot(six_eqn_directory / f"{step}.dat")
            for step in loaded_steps
        },
    }

    _validate_selected_steps(shared_steps, selected_steps)
    _validate_variables(loaded_cases, args.variables)

    show_titles = not args.to_thesis
    overview_figure = build_overview_figure(
        loaded_cases,
        selected_snapshots=selected_snapshots,
        variables=args.variables,
        title=args.title,
        show_titles=show_titles,
    )
    zoom_figure = build_zoom_figure(
        loaded_cases,
        comparison_summary,
        variables=DEFAULT_ZOOM_VARIABLES,
        zoom_xlim=manual_zoom_xlim,
        inset_location=args.inset_location,
        show_titles=show_titles,
    )
    summary_figure = build_summary_table_figure(
        comparison_summary,
        show_titles=show_titles,
    )

    output_path = args.output
    if output_path is None and not args.to_thesis:
        output_path = default_overview_output

    zoom_output_path = args.zoom_output
    summary_output_path = args.summary_output
    if not args.to_thesis:
        if zoom_output_path is None and output_path is not None:
            if output_path == default_overview_output:
                zoom_output_path = default_zoom_output
            else:
                zoom_output_path = _derive_sibling_output_path(output_path, "_zoom")
        elif zoom_output_path is None:
            zoom_output_path = default_zoom_output

        if summary_output_path is None and output_path is not None:
            if output_path == default_overview_output:
                summary_output_path = default_summary_output
            else:
                summary_output_path = _derive_sibling_output_path(
                    output_path,
                    "_summary",
                )
        elif summary_output_path is None:
            summary_output_path = default_summary_output

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overview_figure.savefig(output_path, dpi=200)
        print(f"Overview figure written to {output_path}")

    if zoom_output_path is not None:
        zoom_output_path.parent.mkdir(parents=True, exist_ok=True)
        zoom_figure.savefig(zoom_output_path, dpi=200)
        print(f"Zoom figure written to {zoom_output_path}")

    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_figure.savefig(summary_output_path, dpi=200)
        print(f"Summary figure written to {summary_output_path}")

    thesis_overview_path = save_thesis_figure_from_args(
        overview_figure,
        args,
        stem=thesis_stem,
        stem_suffix="_overview",
    )
    thesis_zoom_path = save_thesis_figure_from_args(
        zoom_figure,
        args,
        stem=thesis_stem,
        stem_suffix="_zoom",
    )
    thesis_summary_path = save_thesis_figure_from_args(
        summary_figure,
        args,
        stem=thesis_stem,
        stem_suffix="_summary",
    )
    if thesis_overview_path is not None:
        print(f"Thesis overview PDF written to {thesis_overview_path}")
    if thesis_zoom_path is not None:
        print(f"Thesis zoom PDF written to {thesis_zoom_path}")
    if thesis_summary_path is not None:
        print(f"Thesis summary PDF written to {thesis_summary_path}")

    _print_difference_summary(comparison_summary, args.variables)
    print(
        "Shared saved steps: "
        f"{len(shared_steps)}; plotted overview snapshots: "
        f"{', '.join(f'{selection.percent:g}%' for selection in selected_snapshots)}"
    )

    if not args.no_show:
        plt.show()
    else:
        plt.close(overview_figure)
        plt.close(zoom_figure)
        plt.close(summary_figure)


def build_overview_figure(
    loaded_cases: dict[str, dict[int, object]],
    *,
    selected_snapshots: list[SnapshotSelection] | tuple[SnapshotSelection, ...],
    variables: list[str] | tuple[str, ...],
    title: str | None,
    show_titles: bool,
):
    apply_thesis_style()

    num_rows = len(variables)
    num_columns = len(selected_snapshots)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=OVERVIEW_FIGURE_SIZE,
        squeeze=False,
        sharex="col",
        sharey="row",
    )

    for row_index, variable in enumerate(variables):
        for col_index, selection in enumerate(selected_snapshots):
            axis = axes[row_index, col_index]
            five_eqn = loaded_cases["5Eqn"][selection.step]
            six_eqn = loaded_cases["6Eqn"][selection.step]
            _plot_variable_comparison_on_axis(
                axis,
                five_eqn,
                six_eqn,
                variable,
            )
            if show_titles and row_index == 0:
                axis.set_title(
                    latex_text(f"{selection.percent:g}% of simulation"),
                    fontsize=THESIS_TITLE_FONT_SIZE,
                )
            if col_index == 0:
                axis.set_ylabel(_build_variable_label(variable))
            if row_index == num_rows - 1:
                axis.set_xlabel(r"$x\ [\mathrm{m}]$")
            axis.grid(True, alpha=0.35)

    legend_handles = _build_model_legend_handles()
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.935 if show_titles else 1.01),
    )

    if show_titles:
        default_title = latex_text("MFC 5Eqn vs 6Eqn binary comparison")
        figure.suptitle(_ensure_latex_title(title or default_title), y=0.985)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))

    return figure


def build_zoom_figure(
    loaded_cases: dict[str, dict[int, object]],
    summary: dict[str, DifferenceSummary],
    *,
    variables: list[str] | tuple[str, ...],
    zoom_xlim: tuple[float, float] | None,
    inset_location: str,
    show_titles: bool,
):
    apply_thesis_style()

    num_rows = 2
    num_columns = 2
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=ZOOM_FIGURE_SIZE,
        squeeze=False,
    )

    for panel_index, (axis, variable) in enumerate(zip(axes.ravel(), variables)):
        row_index = panel_index // num_columns
        if variable not in summary:
            raise ValueError(
                f"Variable '{variable}' is not available for zoom plotting. "
                f"Available fields: {sorted(summary)}"
            )

        stats = summary[variable]
        five_eqn = loaded_cases["5Eqn"][stats.step]
        six_eqn = loaded_cases["6Eqn"][stats.step]
        _plot_zoom_panel(
            axis,
            five_eqn,
            six_eqn,
            variable,
            focus_x=stats.x,
            zoom_xlim=zoom_xlim,
            inset_location=inset_location,
        )
        axis.set_ylabel(_build_variable_label(variable))
        if row_index == num_rows - 1:
            axis.set_xlabel(r"$x\ [\mathrm{m}]$")
        axis.grid(True, alpha=0.35)
        if show_titles:
            axis.set_title(
                latex_text(f"{stats.percent:g}% of simulation"),
                fontsize=THESIS_TITLE_FONT_SIZE,
            )

    for axis in axes.ravel()[len(variables) :]:
        axis.set_visible(False)

    figure.legend(
        handles=_build_model_legend_handles(),
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.935 if show_titles else 1.01),
    )

    if show_titles:
        figure.suptitle(latex_text("MFC 5Eqn vs 6Eqn zoom comparison"), y=0.985)
        figure.subplots_adjust(
            left=0.11,
            right=0.985,
            bottom=0.09,
            top=0.835,
            wspace=0.26,
            hspace=0.34,
        )
    else:
        figure.subplots_adjust(
            left=0.11,
            right=0.985,
            bottom=0.09,
            top=0.96,
            wspace=0.26,
            hspace=0.34,
        )

    return figure


def build_summary_table_figure(
    summary: dict[str, DifferenceSummary],
    *,
    show_titles: bool,
):
    apply_thesis_style()

    figure, axis = plt.subplots(figsize=SUMMARY_FIGURE_SIZE)
    axis.axis("off")

    table_rows = []
    for field, stats in summary.items():
        table_rows.append(
            [
                _build_variable_label(field),
                _format_scientific(stats.max_abs),
                _format_scientific(stats.normalized_max_abs),
                latex_text(f"{stats.percent:g}%"),
                _format_scientific(stats.x),
                _format_scientific(stats.value_5eqn),
                _format_scientific(stats.value_6eqn),
            ]
        )

    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "Field",
            r"max $|\Delta|$",
            "normalized",
            latex_text("sim %"),
            "x [m]",
            "5Eqn",
            "6Eqn",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.45)

    if show_titles:
        figure.suptitle(
            latex_text("MFC 5Eqn vs 6Eqn summary table"),
            y=0.985,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))

    return figure


def _plot_variable_comparison_on_axis(
    axis,
    five_eqn_snapshot,
    six_eqn_snapshot,
    variable: str,
):
    if variable not in five_eqn_snapshot.fields:
        available = ", ".join(sorted(five_eqn_snapshot.fields))
        raise ValueError(
            f"Variable '{variable}' is not available in {five_eqn_snapshot.path.name}. "
            f"Available fields: {available}"
        )
    if variable not in six_eqn_snapshot.fields:
        available = ", ".join(sorted(six_eqn_snapshot.fields))
        raise ValueError(
            f"Variable '{variable}' is not available in {six_eqn_snapshot.path.name}. "
            f"Available fields: {available}"
        )

    if not np.allclose(five_eqn_snapshot.x_centers, six_eqn_snapshot.x_centers):
        raise ValueError("The 5Eqn and 6Eqn snapshots do not share the same x-grid.")

    x_centers = five_eqn_snapshot.x_centers
    axis.plot(
        x_centers,
        five_eqn_snapshot.fields[variable],
        label=MODEL_LABELS["5Eqn"],
        **MODEL_STYLES["5Eqn"],
    )
    axis.plot(
        x_centers,
        six_eqn_snapshot.fields[variable],
        label=MODEL_LABELS["6Eqn"],
        **MODEL_STYLES["6Eqn"],
    )
    _apply_scientific_y_formatter(axis)


def _plot_zoom_panel(
    axis,
    five_eqn_snapshot,
    six_eqn_snapshot,
    variable: str,
    *,
    focus_x: float,
    zoom_xlim: tuple[float, float] | None,
    inset_location: str,
):
    if variable not in five_eqn_snapshot.fields:
        available = ", ".join(sorted(five_eqn_snapshot.fields))
        raise ValueError(
            f"Variable '{variable}' is not available in {five_eqn_snapshot.path.name}. "
            f"Available fields: {available}"
        )
    if variable not in six_eqn_snapshot.fields:
        available = ", ".join(sorted(six_eqn_snapshot.fields))
        raise ValueError(
            f"Variable '{variable}' is not available in {six_eqn_snapshot.path.name}. "
            f"Available fields: {available}"
        )

    if not np.allclose(five_eqn_snapshot.x_centers, six_eqn_snapshot.x_centers):
        raise ValueError("The 5Eqn and 6Eqn snapshots do not share the same x-grid.")

    x_centers = five_eqn_snapshot.x_centers
    values_5eqn = five_eqn_snapshot.fields[variable]
    values_6eqn = six_eqn_snapshot.fields[variable]
    axis.plot(
        x_centers,
        values_5eqn,
        label=MODEL_LABELS["5Eqn"],
        **MODEL_STYLES["5Eqn"],
    )
    axis.plot(
        x_centers,
        values_6eqn,
        label=MODEL_LABELS["6Eqn"],
        **MODEL_STYLES["6Eqn"],
    )

    zoom_xlim = _build_zoom_xlim(
        x_centers,
        focus_x,
        manual_zoom_xlim=zoom_xlim,
    )
    zoom_ylim = _build_zoom_ylim(x_centers, values_5eqn, values_6eqn, zoom_xlim)
    inset_axis = inset_axes(
        axis,
        width=INSET_REL_WIDTH,
        height=INSET_REL_HEIGHT,
        loc=_resolve_inset_location(inset_location),
        borderpad=INSET_BORDERPAD,
    )
    inset_axis.plot(x_centers, values_5eqn, **MODEL_STYLES["5Eqn"])
    inset_axis.plot(x_centers, values_6eqn, **MODEL_STYLES["6Eqn"])
    inset_axis.set_xlim(*zoom_xlim)
    inset_axis.set_ylim(*zoom_ylim)
    inset_axis.tick_params(
        labelbottom=False,
        labelleft=False,
        bottom=True,
        left=True,
        top=False,
        right=False,
        direction="in",
        length=2.5,
        pad=0.0,
    )
    inset_axis.tick_params(
        labelsize=max(THESIS_TICK_FONT_SIZE - 2, 6),
    )
    inset_axis.grid(True, alpha=0.25)
    inset_axis.set_facecolor("white")
    inset_axis.set_zorder(4)
    zoom_outline = Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        fill=False,
        edgecolor=INSET_EDGE_COLOR,
        linewidth=INSET_EDGE_WIDTH,
        linestyle=INSET_EDGE_DASH,
        zorder=7,
        clip_on=False,
    )
    axis.add_patch(zoom_outline)
    if hasattr(axis, "indicate_inset_zoom"):
        indicator = axis.indicate_inset_zoom(
            inset_axis,
            edgecolor=INSET_EDGE_COLOR,
            linewidth=INSET_EDGE_WIDTH,
            alpha=1.0,
            facecolor="none",
        )
        if isinstance(indicator, tuple) and indicator:
            inset_box = indicator[0]
            if inset_box is not None:
                inset_box.set_edgecolor(INSET_EDGE_COLOR)
                inset_box.set_linewidth(0.0)
                inset_box.set_facecolor("none")
                inset_box.set_alpha(0.0)
                inset_box.set_zorder(0)
                inset_box.set_clip_on(False)
    else:
        inset_box, connector_1, connector_2 = mark_inset(
            axis,
            inset_axis,
            loc1=3,
            loc2=4,
            fc="none",
            ec=INSET_EDGE_COLOR,
            lw=INSET_EDGE_WIDTH,
        )
        inset_box.set_linewidth(0.0)
        inset_box.set_alpha(0.0)
        inset_box.set_zorder(0)
        inset_box.set_clip_on(False)
        connector_1.set_zorder(5)
        connector_2.set_zorder(5)

    axis.set_xlim(float(x_centers[0]), float(x_centers[-1]))
    _apply_scientific_y_formatter(axis)


def _discover_shared_steps(five_eqn_directory: Path, six_eqn_directory: Path) -> list[int]:
    steps_5 = set(discover_mfc_binary_steps(five_eqn_directory))
    steps_6 = set(discover_mfc_binary_steps(six_eqn_directory))
    return sorted(steps_5 & steps_6)


def _build_difference_summary(
    five_eqn_directory: Path,
    six_eqn_directory: Path,
    shared_steps: list[int],
):
    summary = {}
    field_scales = {}
    simulation_end_step = shared_steps[-1]

    reference_snapshot = load_mfc_binary_snapshot(five_eqn_directory / f"{shared_steps[0]}.dat")
    fields = reference_snapshot.field_names

    for field in fields:
        summary[field] = DifferenceSummary(
            max_abs=0.0,
            step=shared_steps[0],
            percent=_step_to_percent(shared_steps[0], simulation_end_step),
            x=float(reference_snapshot.x_centers[0]),
            value_5eqn=float(reference_snapshot.fields[field][0]),
            value_6eqn=float(reference_snapshot.fields[field][0]),
            field_scale=0.0,
        )

    for step in shared_steps:
        five_eqn_snapshot = load_mfc_binary_snapshot(five_eqn_directory / f"{step}.dat")
        six_eqn_snapshot = load_mfc_binary_snapshot(six_eqn_directory / f"{step}.dat")

        if five_eqn_snapshot.field_names != six_eqn_snapshot.field_names:
            raise ValueError(
                f"Field mismatch at step {step}: {five_eqn_snapshot.field_names} vs "
                f"{six_eqn_snapshot.field_names}"
            )

        if not np.allclose(five_eqn_snapshot.x_faces, six_eqn_snapshot.x_faces):
            raise ValueError(f"x-grid mismatch at step {step}.")

        for field in fields:
            values_5eqn = five_eqn_snapshot.fields[field]
            values_6eqn = six_eqn_snapshot.fields[field]
            field_scales[field] = max(
                field_scales.get(field, 0.0),
                float(np.max(np.abs(values_5eqn))),
                float(np.max(np.abs(values_6eqn))),
            )

            diff = values_6eqn - values_5eqn
            max_index = int(np.argmax(np.abs(diff)))
            max_abs = float(abs(diff[max_index]))
            if max_abs > summary[field].max_abs:
                summary[field] = DifferenceSummary(
                    max_abs=max_abs,
                    step=step,
                    percent=_step_to_percent(step, simulation_end_step),
                    x=float(five_eqn_snapshot.x_centers[max_index]),
                    value_5eqn=float(values_5eqn[max_index]),
                    value_6eqn=float(values_6eqn[max_index]),
                    field_scale=field_scales[field],
                )

    refreshed_summary = {}
    for field, stats in summary.items():
        refreshed_summary[field] = DifferenceSummary(
            max_abs=stats.max_abs,
            step=stats.step,
            percent=_step_to_percent(stats.step, simulation_end_step),
            x=stats.x,
            value_5eqn=stats.value_5eqn,
            value_6eqn=stats.value_6eqn,
            field_scale=field_scales[field],
        )
    return refreshed_summary


def _infer_case_label(snapshot_directory: Path) -> str:
    resolved = Path(snapshot_directory).resolve()
    parts = resolved.parts
    for marker in ("5Eqn", "6Eqn"):
        if marker in parts:
            marker_index = parts.index(marker)
            if marker_index >= 2:
                return f"{parts[marker_index - 2]}_{parts[marker_index - 1]}"
            break
    parents = resolved.parents
    if len(parents) >= 2:
        return f"{parents[1].name}_{parents[0].name}"
    return resolved.name


def _build_default_output_paths(case_label: str) -> tuple[Path, Path, Path]:
    figures_dir = _build_case_output_directory(case_label)
    overview = figures_dir / "overview.png"
    zoom = figures_dir / "zoom.png"
    summary = figures_dir / "summary.png"
    return overview, zoom, summary


def _build_case_output_directory(case_label: str) -> Path:
    family, separator, mode = case_label.partition("_")
    family_slug = _camel_to_snake(family)
    mode_slug = mode if separator else "default"
    return REPO_ROOT / "artifacts" / "figures" / "mfc" / family_slug / mode_slug


def _camel_to_snake(text: str) -> str:
    pieces: list[str] = []
    for index, character in enumerate(text):
        if (
            index
            and character.isupper()
            and (
                not text[index - 1].isupper()
                or (index + 1 < len(text) and text[index + 1].islower())
            )
        ):
            pieces.append("_")
        pieces.append(character.lower())
    return "".join(pieces)


def _select_snapshot_percentages(
    shared_steps: list[int],
    *,
    requested_steps: list[int] | tuple[int, ...] | None,
    requested_percentages: list[float] | tuple[float, ...] | None,
) -> list[SnapshotSelection]:
    simulation_end_step = shared_steps[-1]

    if requested_steps is not None:
        selected_steps = _validate_requested_steps(shared_steps, requested_steps)
    else:
        percentages = (
            DEFAULT_OVERVIEW_PERCENTAGES
            if requested_percentages is None
            else requested_percentages
        )
        selected_steps = _select_steps_from_percentages(shared_steps, percentages)

    return [
        SnapshotSelection(
            step=step,
            percent=_step_to_percent(step, simulation_end_step),
        )
        for step in selected_steps
    ]


def _select_steps_from_percentages(
    shared_steps: list[int],
    percentages: list[float] | tuple[float, ...],
) -> list[int]:
    if not shared_steps:
        raise ValueError("No MFC snapshots were found in the selected directory.")

    simulation_end_step = shared_steps[-1]
    selected_steps = []
    for percentage in percentages:
        target_step = simulation_end_step * float(percentage) / 100.0
        selected_steps.append(_nearest_shared_step(shared_steps, target_step))
    return sorted(dict.fromkeys(selected_steps))


def _nearest_shared_step(shared_steps: list[int], target_step: float) -> int:
    step_array = np.asarray(shared_steps, dtype=float)
    nearest_index = int(np.argmin(np.abs(step_array - target_step)))
    return int(shared_steps[nearest_index])


def _validate_requested_steps(
    shared_steps: list[int],
    requested_steps: list[int] | tuple[int, ...],
) -> list[int]:
    missing_steps = [step for step in requested_steps if step not in shared_steps]
    if missing_steps:
        raise ValueError(
            "The following requested steps are not available: "
            f"{missing_steps}. Available steps include {list(shared_steps[:10])}"
        )
    return sorted(dict.fromkeys(requested_steps))


def _validate_selected_steps(shared_steps: list[int], selected_steps: list[int]):
    missing_steps = [step for step in selected_steps if step not in shared_steps]
    if missing_steps:
        raise ValueError(
            f"Selected steps are not available in both cases: {missing_steps}"
        )


def _validate_variables(loaded_cases: dict[str, dict[int, object]], variables):
    reference_snapshot = next(iter(loaded_cases["5Eqn"].values()))
    available_fields = set(reference_snapshot.field_names)
    missing = [variable for variable in variables if variable not in available_fields]
    if missing:
        raise ValueError(
            f"Requested variables are not available in the MFC snapshots: {missing}. "
            f"Available fields: {sorted(available_fields)}"
        )


def _build_variable_label(variable: str) -> str:
    return FIELD_LABELS.get(variable, latex_text(variable))


def _build_difference_label(variable: str) -> str:
    labels = {
        "alpha_rho1": r"$\Delta m_1\ [\mathrm{kg\,m^{-3}}]$",
        "alpha_rho2": r"$\Delta m_2\ [\mathrm{kg\,m^{-3}}]$",
        "alpha_rho3": r"$\Delta m_3\ [\mathrm{kg\,m^{-3}}]$",
        "alpha1": r"$\Delta \alpha_1$",
        "alpha2": r"$\Delta \alpha_2$",
        "alpha3": r"$\Delta \alpha_3$",
        "pres": r"$\Delta p\ [\mathrm{Pa}]$",
        "vel1": r"$\Delta u\ [\mathrm{m\,s^{-1}}]$",
    }
    return labels.get(variable, latex_text(f"delta {variable}"))


def _build_model_legend_handles():
    return [
        Line2D([0], [0], label=MODEL_LABELS["5Eqn"], **MODEL_STYLES["5Eqn"]),
        Line2D([0], [0], label=MODEL_LABELS["6Eqn"], **MODEL_STYLES["6Eqn"]),
    ]


def _build_zoom_xlim(
    x_centers: np.ndarray,
    focus_x: float,
    *,
    manual_zoom_xlim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    if manual_zoom_xlim is not None:
        lower = max(float(x_centers[0]), float(manual_zoom_xlim[0]))
        upper = min(float(x_centers[-1]), float(manual_zoom_xlim[1]))
        if upper <= lower:
            raise ValueError(
                "Manual zoom x-limits do not overlap the available x-domain."
            )
        return (lower, upper)

    lower = max(float(x_centers[0]), float(focus_x) - ZOOM_HALF_WIDTH)
    upper = min(float(x_centers[-1]), float(focus_x) + ZOOM_HALF_WIDTH)
    if upper <= lower:
        lower = float(x_centers[0])
        upper = float(x_centers[-1])
    return (lower, upper)


def _parse_inset_location(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "southeast": "southeast",
        "se": "southeast",
        "lower_right": "southeast",
        "lowerright": "southeast",
        "south_east": "southeast",
        "southwest": "southwest",
        "sw": "southwest",
        "lower_left": "southwest",
        "lowerleft": "southwest",
        "south_west": "southwest",
        "northeast": "northeast",
        "ne": "northeast",
        "upper_right": "northeast",
        "upperright": "northeast",
        "north_east": "northeast",
        "northwest": "northwest",
        "nw": "northwest",
        "upper_left": "northwest",
        "upperleft": "northwest",
        "north_west": "northwest",
        "east": "east",
        "e": "east",
        "center_right": "east",
        "centerright": "east",
        "west": "west",
        "w": "west",
        "center_left": "west",
        "centerleft": "west",
    }
    if normalized not in aliases:
        valid = ", ".join(
            ("southeast", "southwest", "northeast", "northwest", "east", "west")
        )
        raise argparse.ArgumentTypeError(
            f"Invalid inset location '{value}'. Choose from: {valid}."
        )
    return aliases[normalized]


def _resolve_inset_location(location: str) -> str:
    if location in INSET_LOCATION_MAP:
        return INSET_LOCATION_MAP[location]
    return location


def _build_zoom_ylim(
    x_centers: np.ndarray,
    values_5eqn: np.ndarray,
    values_6eqn: np.ndarray,
    zoom_xlim: tuple[float, float],
) -> tuple[float, float]:
    zoom_mask = (x_centers >= zoom_xlim[0]) & (x_centers <= zoom_xlim[1])
    zoom_values = np.concatenate(
        (
            np.asarray(values_5eqn)[zoom_mask],
            np.asarray(values_6eqn)[zoom_mask],
        )
    )
    if zoom_values.size == 0:
        zoom_values = np.concatenate((np.asarray(values_5eqn), np.asarray(values_6eqn)))

    y_min = float(np.min(zoom_values))
    y_max = float(np.max(zoom_values))
    if y_max == y_min:
        center = 0.5 * (y_min + y_max)
        scale = max(abs(center), 1.0)
        padding = 0.05 * scale
        return (center - padding, center + padding)

    span = y_max - y_min
    padding = ZOOM_Y_PADDING_FRACTION * span
    return (y_min - padding, y_max + padding)


def _apply_scientific_y_formatter(axis):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(True)
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.get_offset_text().set_fontsize(max(THESIS_TICK_FONT_SIZE - 2, 6))


def _format_scientific(value: float) -> str:
    return f"{float(value):.2e}"


def _step_to_percent(step: int, simulation_end_step: int) -> float:
    if simulation_end_step == 0:
        return 0.0
    return 100.0 * float(step) / float(simulation_end_step)


def _print_difference_summary(
    summary: dict[str, DifferenceSummary],
    plotted_variables: list[str] | tuple[str, ...],
):
    print("Comparison summary:")
    for field in summary:
        if field not in plotted_variables and field not in (
            "alpha_rho1",
            "alpha_rho2",
            "alpha_rho3",
        ):
            continue
        stats = summary[field]
        print(
            f"  - {field}: max |Δ| = {stats.max_abs:g} at "
            f"{stats.percent:g}% of simulation (x = {stats.x:g} m; "
            f"5Eqn = {stats.value_5eqn:g}; 6Eqn = {stats.value_6eqn:g}; "
            f"normalized = {stats.normalized_max_abs:g})"
        )


def _derive_sibling_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _ensure_latex_title(title: str) -> str:
    if "$" in title:
        return title
    return latex_text(title)


if __name__ == "__main__":
    main()
