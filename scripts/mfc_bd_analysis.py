import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "artifacts" / ".matplotlib"),
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.bubble_dynamics import (
    build_mfc_strong_collapse_case,
    build_keller_miksis_theory_histories,
    normalize_radius_history,
)
from src.io.xyz import load_time_radius_history
from src.plots.publication import (
    THESIS_LABEL_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    THESIS_TICK_FONT_SIZE,
    THESIS_TITLE_FONT_SIZE,
    add_thesis_export_argument,
    add_show_titles_argument,
    apply_publication_style,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)

DEFAULT_DATA_FOLDER = Path(
    "/disk/simulations/Relaxation/BubbleCollapse/2D/Sphere/StrongCollapse/pT/6Eqn/Axisymmetric"
)
RESOLUTION_ORDER = ("N150E1", "N160E3", "N320E3", "N640E3", "N128E4", "N256E4")
# Human-friendly labels for the refinement hierarchy shown in the figures.
REFINEMENT_LEVEL_LABELS = {
    resolution: f"L{index + 1}"
    for index, resolution in enumerate(RESOLUTION_ORDER)
}
SERIES_COLORS = {
    ("6Eqn", "p"): "#0072B2",
    ("6Eqn", "pT"): "#000000",
    ("5Eqn", "pT"): "#000000",
}
SERIES_MARKERS = {
    ("6Eqn", "p"): None,
    ("6Eqn", "pT"): "o",
    ("5Eqn", "pT"): "^",
}

BASE_LINEWIDTH = 2.0
PT_LINEWIDTH = BASE_LINEWIDTH * 1.5
KM_LINEWIDTH = 2.5
KM_ENVELOPE_COLOR = "#B0B0B0"
KM_ENVELOPE_ALPHA = 0.35
THESIS_EXPORT_STEM = "pTBD"
THESIS_FIGURE_SIZE = thesis_figure_size(0.50)
PUBLICATION_FIGURE_SIZE = (10, 5)
DEFAULT_ZOOM_X_LIMITS = (0.98, 1.02)
REFERENCE_ZOOM_X_LIMITS = (0.95, 1.05)
REFERENCE_ZOOM_Y_LIMITS = (-0.05, 0.40)
THESIS_MAIN_TITLE = r"$\mathrm{Strong\ collapse\ problem\ (MFC)}$"
PUBLICATION_MAIN_TITLE = r"$\mathrm{Radial\ evolution,\ strong\ collapse\ problem\ (MFC)}$"

STYLES = {
    "N150E1": {"linestyle": "-"},
    "N160E3": {"linestyle": "--"},
    "N320E3": {"linestyle": "-."},
    "N640E3": {"linestyle": ":"},
    "N128E4": {"linestyle": (0, (5.0, 1.5, 1.0, 1.5))},
    "N256E4": {"linestyle": (0, (1.0, 1.0, 3.0, 1.0, 1.0, 1.0))},
}
THEORY_STYLES = {
    r"$\mathrm{Isentropic\ KM}$": {"color": "#808080", "linestyle": "-", "linewidth": KM_LINEWIDTH},
    r"$\mathrm{Isothermal\ KM}$": {"color": "#808080", "linestyle": "-", "linewidth": KM_LINEWIDTH},
}
FALLBACK_STYLES = (
    {"linestyle": "-"},
    {"linestyle": "--"},
    {"linestyle": "-."},
    {"linestyle": ":"},
    {"linestyle": (0, (5.0, 1.5, 1.0, 1.5))},
    {"linestyle": (0, (1.0, 1.0, 3.0, 1.0, 1.0, 1.0))},
    {"linestyle": (0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))},
)


def infer_equation_data_folder(base_folder: Path, equation_type: str) -> Path:
    folder_parts = list(base_folder.parts)
    try:
        equation_index = folder_parts.index("6Eqn")
    except ValueError:
        return base_folder

    folder_parts[equation_index] = equation_type
    return Path(*folder_parts)


def get_series_color(case_label: str, pressure_type: str) -> str:
    return SERIES_COLORS.get((case_label, pressure_type), "#404040")


def get_series_marker(case_label: str, pressure_type: str):
    return SERIES_MARKERS.get((case_label, pressure_type))


def build_series_label(case_label: str, pressure_type: str, resolution: str) -> str:
    return rf"$\mathrm{{{case_label}\ {pressure_type}\ {get_refinement_label(resolution)}}}$"


def build_series_type_label(case_label: str, pressure_type: str) -> str:
    return rf"$\mathrm{{{case_label}\ {pressure_type}}}$"


def get_refinement_label(resolution: str) -> str:
    return REFINEMENT_LEVEL_LABELS.get(resolution, resolution)


def validate_limits(limits, *, label: str) -> tuple[float, float]:
    lower, upper = (float(limits[0]), float(limits[1]))
    if upper <= lower:
        raise ValueError(f"{label} must satisfy lower < upper.")
    return lower, upper


def build_zoom_ylim(zoom_xlim: tuple[float, float]) -> tuple[float, float]:
    zoom_xlim = validate_limits(zoom_xlim, label="zoom x-limits")
    reference_xlim = validate_limits(
        REFERENCE_ZOOM_X_LIMITS,
        label="reference zoom x-limits",
    )
    reference_ylim = validate_limits(
        REFERENCE_ZOOM_Y_LIMITS,
        label="reference zoom y-limits",
    )

    x_span = zoom_xlim[1] - zoom_xlim[0]
    reference_x_span = reference_xlim[1] - reference_xlim[0]
    reference_y_span = reference_ylim[1] - reference_ylim[0]
    y_span = x_span * (reference_y_span / reference_x_span)
    y_center = 0.5 * (reference_ylim[0] + reference_ylim[1])
    half_span = 0.5 * y_span
    return (y_center - half_span, y_center + half_span)


def resolve_zoom_limits(
    zoom_xlim: tuple[float, float] | None,
    zoom_ylim: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    resolved_zoom_xlim = validate_limits(
        DEFAULT_ZOOM_X_LIMITS if zoom_xlim is None else zoom_xlim,
        label="zoom x-limits",
    )
    if zoom_ylim is None:
        resolved_zoom_ylim = build_zoom_ylim(resolved_zoom_xlim)
    else:
        resolved_zoom_ylim = validate_limits(zoom_ylim, label="zoom y-limits")
    return resolved_zoom_xlim, resolved_zoom_ylim


def build_zoom_panel_title(zoom_xlim: tuple[float, float]) -> str:
    return rf"$\mathrm{{Zoom:}}\ {zoom_xlim[0]:.2f} \leq t/t_c \leq {zoom_xlim[1]:.2f}$"


def build_mfc_radius_history(filepath: Path):
    time_values, radius = load_time_radius_history(filepath)
    return normalize_radius_history(time_values, radius)


def get_pressure_data_folder(base_folder: Path, pressure_type: str) -> Path:
    if pressure_type == "pT":
        return base_folder

    folder_parts = list(base_folder.parts)
    try:
        pressure_index = folder_parts.index("pT")
    except ValueError:
        return base_folder

    folder_parts[pressure_index] = pressure_type
    return Path(*folder_parts)


def get_resolution_from_path(filepath: Path, pressure_type: str) -> str:
    prefix = f"BD{pressure_type}"
    if filepath.stem.startswith(prefix):
        return filepath.stem[len(prefix) :]
    return filepath.stem


def resolution_sort_key(resolution: str):
    try:
        return (0, RESOLUTION_ORDER.index(resolution))
    except ValueError:
        return (1, resolution)


def get_resolution_style(resolution: str):
    if resolution in STYLES:
        return STYLES[resolution]

    try:
        fallback_index = RESOLUTION_ORDER.index(resolution) % len(FALLBACK_STYLES)
    except ValueError:
        fallback_index = sum(ord(char) for char in resolution) % len(FALLBACK_STYLES)
    return FALLBACK_STYLES[fallback_index]


def build_refinement_legend_handles(
    radius_series,
    *,
    reference_pressure_type: str = "pT",
):
    seen_resolutions = []

    # Prefer the pT cases so the refinement ordering follows the pT sequence.
    for series in radius_series:
        if series["pressure_type"] != reference_pressure_type:
            continue
        resolution = series["resolution"]
        if resolution not in seen_resolutions:
            seen_resolutions.append(resolution)

    # Fall back to any remaining resolutions that are only present in other cases.
    for series in radius_series:
        if series["pressure_type"] == reference_pressure_type:
            continue
        resolution = series["resolution"]
        if resolution not in seen_resolutions:
            seen_resolutions.append(resolution)

    handles = []
    for resolution in seen_resolutions:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#404040",
                linestyle=get_resolution_style(resolution)["linestyle"],
                linewidth=PT_LINEWIDTH,
                label=get_refinement_label(resolution),
            )
        )
    return handles


def build_series_legend_handles(radius_series):
    seen_series = []
    for series in radius_series:
        series_key = (series["case_label"], series["pressure_type"])
        if series_key not in seen_series:
            seen_series.append(series_key)

    ordered_series = [
        series_key
        for series_key in (("6Eqn", "p"), ("6Eqn", "pT"), ("5Eqn", "pT"))
        if series_key in seen_series
    ]

    handles = []
    for case_label, pressure_type in ordered_series:
        handles.append(
            Line2D(
                [0],
                [0],
                color=get_series_color(case_label, pressure_type),
                linestyle="-",
                linewidth=PT_LINEWIDTH,
                marker=get_series_marker(case_label, pressure_type),
                markerfacecolor="none",
                markeredgecolor=get_series_color(case_label, pressure_type),
                markeredgewidth=1.5,
                markersize=8,
                label=build_series_type_label(case_label, pressure_type),
            )
        )
    return handles


def add_semantic_legends(axis, radius_series, *, thesis_mode: bool):
    legend_fontsize = max(7, THESIS_TICK_FONT_SIZE - 1) if thesis_mode else 8

    refinement_handles = build_refinement_legend_handles(
        radius_series,
        reference_pressure_type="pT",
    )
    series_handles = build_series_legend_handles(radius_series)

    if refinement_handles:
        refinement_legend = axis.legend(
            handles=refinement_handles,
            title=r"$\mathrm{Refinement\ level}$",
            loc="lower left",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            ncol=1,
            framealpha=0.95,
            handlelength=2.6,
            columnspacing=1.0,
        )
        axis.add_artist(refinement_legend)

    if series_handles:
        axis.legend(
            handles=series_handles,
            title=r"$\mathrm{Series}$",
            loc="upper right",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            ncol=1,
            framealpha=0.95,
            handlelength=2.6,
        )


def add_km_rebound_annotations(axis, theory_histories, *, thesis_mode: bool):
    def clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    x_left, x_right = axis.get_xlim()
    y_bottom, y_top = axis.get_ylim()
    x_span = x_right - x_left
    y_span = y_top - y_bottom
    wide_view = x_span > 0.2
    annotation_fontsize = max(8, THESIS_TICK_FONT_SIZE - 1) if thesis_mode else 8
    arrow_color = "#6E6E6E"
    text_color = "#202020"
    bbox_style = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)

    placements = (
        (
            r"$\mathrm{Isentropic\ KM}$",
            r"$\mathrm{KM\ isentropic}$",
            1.200,
            0.025 * x_span if not wide_view else 0.010 * x_span,
            0.080 * y_span if not wide_view else 0.405,
            "arc3,rad=0.18",
        ),
        (
            r"$\mathrm{Isothermal\ KM}$",
            r"$\mathrm{KM\ isothermal}$",
            1.200,
            -0.025 * x_span if not wide_view else 0.60,
            -0.060 * y_span if not wide_view else 0.22,
            "arc3,rad=-0.18",
        ),
    )

    for theory_label, annotation_text, anchor_x, x_shift, y_shift, connection_style in placements:
        theory = theory_histories.get(theory_label)
        if theory is None:
            continue

        target_x = clamp(anchor_x, x_left + 0.55 * x_span, x_right - 0.12 * x_span)
        target_y = float(
            np.interp(
                target_x,
                theory["normalized_time"],
                theory["normalized_radius"],
            )
        )
        if wide_view:
            if theory_label == r"$\mathrm{Isentropic\ KM}$":
                text_x = clamp(
                    target_x + x_shift,
                    x_left + 0.02 * x_span,
                    x_right - 0.03 * x_span,
                )
            else:
                text_x = clamp(x_shift, x_left + 0.02 * x_span, x_right - 0.03 * x_span)
            text_y = clamp(y_shift, y_bottom + 0.08 * y_span, y_top - 0.08 * y_span)
        else:
            text_x = clamp(target_x + x_shift, x_left + 0.02 * x_span, x_right - 0.03 * x_span)
            text_y = clamp(target_y + y_shift, y_bottom + 0.08 * y_span, y_top - 0.08 * y_span)

        axis.annotate(
            annotation_text,
            xy=(target_x, target_y),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=annotation_fontsize,
            color=text_color,
            ha="left",
            va="center",
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=1.0,
                shrinkA=0.0,
                shrinkB=0.0,
                connectionstyle=connection_style,
            ),
            bbox=bbox_style,
            zorder=5,
        )


def build_km_envelope(theory_histories, time_limit=None):
    if not theory_histories:
        return None

    normalized_times = [history["normalized_time"] for history in theory_histories.values()]
    common_start = max(time_values[0] for time_values in normalized_times)
    common_end = min(time_values[-1] for time_values in normalized_times)
    if time_limit is not None:
        common_end = min(common_end, time_limit)

    if common_end <= common_start:
        return None

    num_points = max(time_values.size for time_values in normalized_times)
    common_time = np.linspace(common_start, common_end, num_points)
    interpolated_radii = [
        np.interp(common_time, history["normalized_time"], history["normalized_radius"])
        for history in theory_histories.values()
    ]
    lower_envelope = np.minimum.reduce(interpolated_radii)
    upper_envelope = np.maximum.reduce(interpolated_radii)

    return common_time, lower_envelope, upper_envelope


def load_radius_series_for_pressure(
    data_folder: Path,
    pressure_type: str,
    case_label: str,
    selected_resolutions: list[str] | None = None,
):
    radius_series = []
    simulation_end_time = None
    pressure_data_folder = get_pressure_data_folder(data_folder, pressure_type)
    available_pressure_files = sorted(
        pressure_data_folder.glob(f"BD{pressure_type}N*.xyz"),
        key=lambda path: resolution_sort_key(get_resolution_from_path(path, pressure_type)),
    )

    if not available_pressure_files:
        print(
            f"{pressure_data_folder} has no {case_label} BD{pressure_type}N*.xyz files"
        )
        return radius_series, simulation_end_time

    selected_resolution_set = (
        set(selected_resolutions) if selected_resolutions is not None else None
    )
    available_resolutions = {
        get_resolution_from_path(filepath, pressure_type)
        for filepath in available_pressure_files
    }
    if selected_resolution_set is None:
        pressure_files = available_pressure_files
    else:
        pressure_files = [
            filepath
            for filepath in available_pressure_files
            if get_resolution_from_path(filepath, pressure_type)
            in selected_resolution_set
        ]
        missing_resolutions = selected_resolution_set - available_resolutions
        if missing_resolutions:
            missing_text = ", ".join(
                sorted(missing_resolutions, key=resolution_sort_key)
            )
            print(
                f"{pressure_data_folder} has no {case_label} BD{pressure_type} files for: {missing_text}"
            )

    for filepath in pressure_files:
        resolution = get_resolution_from_path(filepath, pressure_type)

        try:
            normalized_time, normalized_radius, _ = build_mfc_radius_history(filepath)
        except OSError:
            print(f"{filepath} not found")
            continue

        if simulation_end_time is None:
            simulation_end_time = normalized_time[-1]
        else:
            simulation_end_time = max(simulation_end_time, normalized_time[-1])

        radius_series.append(
            {
                "case_label": case_label,
                "color": get_series_color(case_label, pressure_type),
                "label": build_series_label(case_label, pressure_type, resolution),
                "marker": get_series_marker(case_label, pressure_type),
                "pressure_type": pressure_type,
                "resolution": resolution,
                "linestyle": get_resolution_style(resolution)["linestyle"],
                "normalized_radius": normalized_radius,
                "normalized_time": normalized_time,
            }
        )

    return radius_series, simulation_end_time


def build_theory_histories(simulation_end_time):
    case = build_mfc_strong_collapse_case()
    return build_keller_miksis_theory_histories(
        case,
        min_normalized_time_end=simulation_end_time,
    )


def plot_radius_histories_on_axis(
    axis,
    radius_series,
    theory_histories,
    km_envelope,
    *,
    thesis_mode: bool,
    title: str | None,
    zoom_xlim=None,
    zoom_ylim=None,
    show_legend: bool = False,
    highlight_zoom: bool = False,
):
    for series in radius_series:
        axis.plot(
            series["normalized_time"],
            series["normalized_radius"],
            color=series["color"],
            linestyle=series["linestyle"],
            marker=series["marker"],
            markerfacecolor="none",
            markeredgecolor=series["color"],
            markeredgewidth=1.5,
            markersize=10,
            markevery=60,
            linewidth=PT_LINEWIDTH,
            label=series["label"],
            zorder=3,
        )

    if highlight_zoom and zoom_xlim is not None:
        axis.axvspan(
            zoom_xlim[0],
            zoom_xlim[1],
            color="#E6E6E6",
            alpha=0.6,
            linewidth=0,
            zorder=0,
        )

    if km_envelope is not None:
        km_time, km_lower, km_upper = km_envelope
        axis.fill_between(
            km_time,
            km_lower,
            km_upper,
            color=KM_ENVELOPE_COLOR,
            alpha=KM_ENVELOPE_ALPHA,
            linewidth=0,
            zorder=1,
        )

    for label, theory in theory_histories.items():
        theory_style = dict(THEORY_STYLES[label], zorder=2)
        axis.plot(
            theory["normalized_time"],
            theory["normalized_radius"],
            label=label,
            **theory_style,
        )

    if zoom_xlim is not None:
        axis.set_xlim(*zoom_xlim)
    if zoom_ylim is not None:
        axis.set_ylim(*zoom_ylim)

    if title is not None:
        axis.set_title(
            title, fontsize=THESIS_TITLE_FONT_SIZE if thesis_mode else None
        )
    axis.grid(True)

    if thesis_mode:
        axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    if show_legend:
        add_semantic_legends(axis, radius_series, thesis_mode=thesis_mode)
    add_km_rebound_annotations(axis, theory_histories, thesis_mode=thesis_mode)


def create_radius_history_figure(
    radius_series,
    theory_histories,
    km_envelope,
    *,
    thesis_mode: bool,
    title: str | None,
    zoom_xlim=None,
    zoom_ylim=None,
    show_legend: bool = False,
    highlight_zoom: bool = False,
):
    figure_size = THESIS_FIGURE_SIZE if thesis_mode else PUBLICATION_FIGURE_SIZE
    figure, axis = plt.subplots(
        figsize=figure_size,
        constrained_layout=thesis_mode,
    )
    if thesis_mode:
        figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    plot_radius_histories_on_axis(
        axis,
        radius_series,
        theory_histories,
        km_envelope,
        thesis_mode=thesis_mode,
        title=title,
        zoom_xlim=zoom_xlim,
        zoom_ylim=zoom_ylim,
        show_legend=show_legend,
        highlight_zoom=highlight_zoom,
    )
    axis.set_xlabel(
        r"$t/t_c$",
        fontsize=THESIS_LABEL_FONT_SIZE if thesis_mode else None,
    )
    axis.set_ylabel(
        r"$R/R_0$",
        fontsize=THESIS_LABEL_FONT_SIZE if thesis_mode else None,
    )

    if not thesis_mode:
        figure.tight_layout()

    return figure, axis


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare bubble-radius histories for the MFC 6Eqn p, 6Eqn pT, and 5Eqn pT strong-collapse cases."
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    add_show_titles_argument(parser)
    parser.add_argument(
        "--data-folder",
        "--six-eqn-data-folder",
        dest="six_eqn_data_folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help="Folder containing the 6Eqn BDpTN*.xyz radius-history files; the 6Eqn p folder is inferred as a sibling.",
    )
    parser.add_argument(
        "--five-eqn-data-folder",
        type=Path,
        default=None,
        help=(
            "Folder containing the 5Eqn BDpTN*.xyz radius-history files. "
            "Defaults to the 6Eqn folder with 6Eqn replaced by 5Eqn."
        ),
    )
    parser.add_argument(
        "--p-resolutions",
        nargs="+",
        metavar="RESOLUTION",
        default=None,
        help="Load only these 6Eqn p resolutions, after sorting by the canonical resolution order. Omit to load all 6Eqn p files.",
    )
    parser.add_argument(
        "--pt-resolutions",
        "--pT-resolutions",
        dest="pt_resolutions",
        nargs="+",
        metavar="RESOLUTION",
        default=None,
        help="Load only these pT resolutions, after sorting by the canonical resolution order. The same list is used for both 6Eqn pT and 5Eqn pT; omit to load all available pT files.",
    )
    parser.add_argument(
        "--zoom-xlimits",
        dest="zoom_xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
        help=(
            "Zoom-panel x-limits. Defaults to the focused [0.98, 1.02] window. "
            "If --zoom-ylimits is omitted, the y-limits are derived to preserve "
            "the reference zoom aspect ratio."
        ),
    )
    parser.add_argument(
        "--zoom-ylimits",
        dest="zoom_ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help=(
            "Optional zoom-panel y-limits. If omitted, the script derives them "
            "from the x-limits to preserve the zoom aspect ratio."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plot without displaying the matplotlib window.",
    )
    return parser


def plot_radius_histories(
    six_eqn_data_folder: Path,
    thesis_mode: bool = False,
    show_titles: bool = False,
    p_resolutions: list[str] | None = None,
    pt_resolutions: list[str] | None = None,
    five_eqn_data_folder: Path | None = None,
    zoom_xlim: tuple[float, float] | None = None,
    zoom_ylim: tuple[float, float] | None = None,
):
    if thesis_mode:
        apply_thesis_style()
    else:
        apply_publication_style()

    if five_eqn_data_folder is None:
        five_eqn_data_folder = infer_equation_data_folder(
            six_eqn_data_folder,
            "5Eqn",
        )

    p_series, p_end_time = load_radius_series_for_pressure(
        six_eqn_data_folder,
        "p",
        "6Eqn",
        selected_resolutions=p_resolutions,
    )
    pT_series, pT_end_time = load_radius_series_for_pressure(
        six_eqn_data_folder,
        "pT",
        "6Eqn",
        selected_resolutions=pt_resolutions,
    )
    five_eqn_pT_series, five_eqn_pT_end_time = load_radius_series_for_pressure(
        five_eqn_data_folder,
        "pT",
        "5Eqn",
        selected_resolutions=pt_resolutions,
    )
    radius_series = p_series + pT_series + five_eqn_pT_series
    end_times = [
        time
        for time in (p_end_time, pT_end_time, five_eqn_pT_end_time)
        if time is not None
    ]
    simulation_end_time = max(end_times) if end_times else None
    theory_histories = build_theory_histories(simulation_end_time)

    km_envelope = build_km_envelope(theory_histories, time_limit=simulation_end_time)
    main_title = THESIS_MAIN_TITLE if thesis_mode else PUBLICATION_MAIN_TITLE
    resolved_zoom_xlim, resolved_zoom_ylim = resolve_zoom_limits(
        zoom_xlim,
        zoom_ylim,
    )
    zoom_title = build_zoom_panel_title(resolved_zoom_xlim)

    full_figure = create_radius_history_figure(
        radius_series,
        theory_histories,
        km_envelope,
        thesis_mode=thesis_mode,
        title=main_title if show_titles else None,
        zoom_xlim=None,
        zoom_ylim=None,
        show_legend=True,
        highlight_zoom=True,
    )
    zoom_figure = create_radius_history_figure(
        radius_series,
        theory_histories,
        km_envelope,
        thesis_mode=thesis_mode,
        title=zoom_title if show_titles else None,
        zoom_xlim=resolved_zoom_xlim,
        zoom_ylim=resolved_zoom_ylim,
        show_legend=True,
        highlight_zoom=False,
    )

    return full_figure, zoom_figure


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    five_eqn_data_folder = args.five_eqn_data_folder
    if five_eqn_data_folder is None:
        five_eqn_data_folder = infer_equation_data_folder(
            args.six_eqn_data_folder,
            "5Eqn",
        )
    (full_figure, _), (zoom_figure, _) = plot_radius_histories(
        args.six_eqn_data_folder,
        thesis_mode=args.to_thesis,
        show_titles=args.show_titles,
        p_resolutions=args.p_resolutions,
        pt_resolutions=args.pt_resolutions,
        five_eqn_data_folder=five_eqn_data_folder,
        zoom_xlim=args.zoom_xlim,
        zoom_ylim=args.zoom_ylim,
    )

    if args.to_thesis:
        full_output_path = save_thesis_figure_from_args(
            full_figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        zoom_output_path = save_thesis_figure_from_args(
            zoom_figure,
            args,
            stem=THESIS_EXPORT_STEM,
            stem_suffix="_zoom",
        )
        print(f"Figure written to {full_output_path}")
        print(f"Figure written to {zoom_output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(full_figure)
        plt.close(zoom_figure)


if __name__ == "__main__":
    main()
