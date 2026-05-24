import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.bubble_dynamics import (
    build_mfc_strong_collapse_case,
    normalize_radius_history,
    solve_keller_miksis,
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
PRESSURE_TYPES = ("p", "pT")
RESOLUTION_ORDER = ("N150E1", "N160E3", "N320E3", "N640E3", "N128E4", "N256E4")

# Emphasize the pressure-model data and render the KM solutions as an envelope.
COLORS = {
    "p": "#0072B2",
    "pT": "#000000",
}

BASE_LINEWIDTH = 2.0
PT_LINEWIDTH = BASE_LINEWIDTH * 1.5
KM_LINEWIDTH = 2.5
KM_ENVELOPE_COLOR = "#B0B0B0"
KM_ENVELOPE_ALPHA = 0.35
THESIS_EXPORT_STEM = "pTBD"
THESIS_FIGURE_SIZE = thesis_figure_size(0.50)
PUBLICATION_FIGURE_SIZE = (10, 5)
ZOOM_X_LIMITS = (0.95, 1.05)
ZOOM_Y_LIMITS = (-0.05, 0.40)
THESIS_MAIN_TITLE = r"$\mathrm{Strong\ collapse\ problem\ (MFC)}$"
PUBLICATION_MAIN_TITLE = r"$\mathrm{Radial\ evolution,\ strong\ collapse\ problem\ (MFC,\ p\ and\ pT)}$"
ZOOM_PANEL_TITLE = r"$\mathrm{Zoom:}\ 0.85 \leq t/t_c \leq 1.15$"

STYLES = {
    "N150E1": {"linestyle": "-", "marker": "v"},
    "N160E3": {"linestyle": "-", "marker": "o"},
    "N320E3": {"linestyle": "--", "marker": "s"},
    "N640E3": {"linestyle": "-.", "marker": "^"},
    "N128E4": {"linestyle": ":", "marker": "d"},
    "N256E4": {"linestyle": "-", "marker": "x"},
}
THEORY_STYLES = {
    r"$\mathrm{Isentropic\ KM}$": {"color": "#808080", "linestyle": "-", "linewidth": KM_LINEWIDTH},
    r"$\mathrm{Isothermal\ KM}$": {"color": "#808080", "linestyle": "--", "linewidth": KM_LINEWIDTH},
}
FALLBACK_STYLES = (
    {"linestyle": "-", "marker": "o"},
    {"linestyle": "--", "marker": "s"},
    {"linestyle": "-.", "marker": "^"},
    {"linestyle": ":", "marker": "d"},
    {"linestyle": "-", "marker": "x"},
    {"linestyle": "--", "marker": "P"},
    {"linestyle": "-.", "marker": "v"},
)


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


def load_radius_series(data_folder: Path):
    radius_series = []
    simulation_end_time = None

    for pressure_type in PRESSURE_TYPES:
        pressure_data_folder = get_pressure_data_folder(data_folder, pressure_type)
        pressure_files = sorted(
            pressure_data_folder.glob(f"BD{pressure_type}N*.xyz"),
            key=lambda path: resolution_sort_key(
                get_resolution_from_path(path, pressure_type)
            ),
        )

        if not pressure_files:
            print(f"{pressure_data_folder} has no BD{pressure_type}N*.xyz files")
            continue

        for filepath in pressure_files:
            resolution = get_resolution_from_path(filepath, pressure_type)

            try:
                normalized_time, normalized_radius, _ = build_mfc_radius_history(
                    filepath
                )
            except OSError:
                print(f"{filepath} not found")
                continue

            if simulation_end_time is None:
                simulation_end_time = normalized_time[-1]
            else:
                simulation_end_time = max(simulation_end_time, normalized_time[-1])

            radius_series.append(
                {
                    "color": COLORS[pressure_type],
                    "label": rf"$\mathrm{{{pressure_type}\ {resolution}}}$",
                    "linestyle": get_resolution_style(resolution)["linestyle"],
                    "marker": get_resolution_style(resolution)["marker"],
                    "normalized_radius": normalized_radius,
                    "normalized_time": normalized_time,
                }
            )

    return radius_series, simulation_end_time


def build_theory_histories(simulation_end_time):
    case = build_mfc_strong_collapse_case()
    theory_histories = {}
    for label, heat_transfer_coefficient in (
        (r"$\mathrm{Isentropic\ KM}$", 0.0),
        (r"$\mathrm{Isothermal\ KM}$", 20.0 * 4294967296.0e3),
    ):
        theory_histories[label] = solve_keller_miksis(
            case,
            heat_transfer_coefficient,
            min_normalized_time_end=simulation_end_time,
        )
    return theory_histories


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
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, ncol=2)


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
        description="Compare bubble-radius histories for the MFC p and pT strong-collapse cases."
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    add_show_titles_argument(parser)
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help="Folder containing the BDpTN*.xyz radius-history files; the p folder is inferred as a sibling.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plot without displaying the matplotlib window.",
    )
    return parser


def plot_radius_histories(
    data_folder: Path,
    thesis_mode: bool = False,
    show_titles: bool = False,
):
    if thesis_mode:
        apply_thesis_style()
    else:
        apply_publication_style()

    radius_series, simulation_end_time = load_radius_series(data_folder)
    theory_histories = build_theory_histories(simulation_end_time)

    km_envelope = build_km_envelope(theory_histories, time_limit=simulation_end_time)
    main_title = THESIS_MAIN_TITLE if thesis_mode else PUBLICATION_MAIN_TITLE
    zoom_title = ZOOM_PANEL_TITLE

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
        zoom_xlim=ZOOM_X_LIMITS,
        zoom_ylim=ZOOM_Y_LIMITS,
        show_legend=True,
        highlight_zoom=False,
    )

    return full_figure, zoom_figure


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    (full_figure, _), (zoom_figure, _) = plot_radius_histories(
        args.data_folder,
        thesis_mode=args.to_thesis,
        show_titles=args.show_titles,
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
