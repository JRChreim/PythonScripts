import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "artifacts" / ".matplotlib"),
)

import numpy as np
import matplotlib.pyplot as plt

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.bubble_dynamics import (
    build_ecogen_strong_collapse_case,
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
    "/disk/simulations/ECGSims/results/Bubble Collapse/Pratio1427"
)
PRESSURE_TYPES = ("P", "PT")
RESOLUTION_TYPES = ("N150E1", "N160E3", "N320E3", "N640E3", "N128E4", "N256E4")

# Color-blind safe palette (Okabe-Ito)
COLORS = {
    "P": "#0072B2",
    "PT": "#E69F00",
}

KM_LINEWIDTH = 2.5
KM_ENVELOPE_COLOR = "#B0B0B0"
KM_ENVELOPE_ALPHA = 0.35
THESIS_EXPORT_STEM = "ECOGEN_BD"
THESIS_FIGURE_SIZE = thesis_figure_size(0.62)
PUBLICATION_FIGURE_SIZE = (8, 5)

STYLES = {
    "N150E1": {"linestyle": "-", "marker": "v"},
    "N160E3": {"linestyle": ":", "marker": "o"},
    "N320E3": {"linestyle": "--", "marker": "s"},
    "N640E3": {"linestyle": "-.", "marker": "^"},
    "N128E4": {"linestyle": ":", "marker": "d"},
    "N256E4": {"linestyle": "-", "marker": "x"},
}
THEORY_STYLES = {
    r"$\mathrm{Isentropic\ KM}$": {"color": "#4D4D4D", "linestyle": "-", "linewidth": KM_LINEWIDTH},
    r"$\mathrm{Isothermal\ KM}$": {"color": "#808080", "linestyle": "--", "linewidth": KM_LINEWIDTH},
}


def build_ecogen_radius_history(filepath: Path):
    time_values, radius = load_time_radius_history(filepath)
    return normalize_radius_history(time_values, radius)


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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare bubble-radius histories across pressure models and resolutions."
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    add_show_titles_argument(parser)
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help="Folder containing *.xyz radius-history files.",
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
        figure_size = THESIS_FIGURE_SIZE
    else:
        apply_publication_style()
        figure_size = PUBLICATION_FIGURE_SIZE

    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=thesis_mode)
    if thesis_mode:
        figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)
    simulation_end_time = None

    for pressure_type in PRESSURE_TYPES:
        for resolution in RESOLUTION_TYPES:
            filepath = data_folder / f"{pressure_type}{resolution}.xyz"

            try:
                normalized_time, normalized_radius, _ = build_ecogen_radius_history(
                    filepath
                )
            except OSError:
                print(f"{filepath} not found")
                continue

            if simulation_end_time is None:
                simulation_end_time = normalized_time[-1]
            else:
                simulation_end_time = max(simulation_end_time, normalized_time[-1])

            axis.plot(
                normalized_time,
                normalized_radius,
                color=COLORS[pressure_type],
                linestyle=STYLES[resolution]["linestyle"],
                marker=STYLES[resolution]["marker"],
                markersize=14,
                markevery=50,
                linewidth=2,
                label=rf"$\mathrm{{{pressure_type}\ {resolution}}}$",
            )

    case = build_ecogen_strong_collapse_case()
    theory_histories = {}
    for label, heat_transfer_coefficient in (
        (r"$\mathrm{Isentropic\ KM}$", 0.0),
        (r"$\mathrm{Isothermal\ KM}$", 20.0 * 4294967296.0e3),
    ):
        theory = solve_keller_miksis(
            case,
            heat_transfer_coefficient,
            min_normalized_time_end=simulation_end_time,
        )
        theory_histories[label] = theory

    km_envelope = build_km_envelope(theory_histories, time_limit=simulation_end_time)
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
        axis.plot(
            theory["normalized_time"],
            theory["normalized_radius"],
            label=label,
            zorder=2,
            **THEORY_STYLES[label],
        )

    if thesis_mode:
        axis.set_xlabel(r"$t/t_c$", fontsize=THESIS_LABEL_FONT_SIZE)
        axis.set_ylabel(r"$R/R_0$", fontsize=THESIS_LABEL_FONT_SIZE)
        if show_titles:
            axis.set_title(
                r"$\mathrm{Radial\ evolution,\ strong\ collapse\ problem\ (ECOGEN)}$",
                fontsize=THESIS_TITLE_FONT_SIZE,
            )
        axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    else:
        axis.set_xlabel(r"$t/t_c$")
        axis.set_ylabel(r"$R/R_0$")
        if show_titles:
            axis.set_title(
                r"$\mathrm{Radial\ evolution,\ strong\ collapse\ problem\ (ECOGEN)}$"
            )
    axis.grid(True)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, ncol=2)
    if not thesis_mode:
        figure.tight_layout()

    return figure, axis


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.no_show:
        os.environ.setdefault("PYTHONSCRIPTS_NO_TEX", "1")
        plt.switch_backend("Agg")

    figure, _ = plot_radius_histories(
        args.data_folder,
        thesis_mode=args.to_thesis,
        show_titles=args.show_titles,
    )

    if args.to_thesis:
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        print(f"Figure written to {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
