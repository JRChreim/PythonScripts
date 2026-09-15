"""Plot Keller-Miksis bubble-surface temperature and pressure histories.

This is the generic KM analysis entrypoint used by both the MFC and ECOGEN
strong-collapse cases.
"""

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPO_ROOT / "artifacts" / ".matplotlib"),
)

import matplotlib.pyplot as plt

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.bubble_dynamics import (
    build_ecogen_strong_collapse_case,
    build_keller_miksis_theory_histories,
    build_mfc_strong_collapse_case,
)
from src.plots.publication import (
    THESIS_LABEL_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    THESIS_TICK_FONT_SIZE,
    THESIS_TITLE_FONT_SIZE,
    add_show_titles_argument,
    add_thesis_export_argument,
    apply_publication_style,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)

CASE_BUILDERS = {
    "mfc": build_mfc_strong_collapse_case,
    "ecogen": build_ecogen_strong_collapse_case,
}
CASE_DISPLAY_NAMES = {
    "mfc": "MFC",
    "ecogen": "ECOGEN",
}

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "figures" / "km"
THESIS_EXPORT_STEM = "KM_BD_surface_thermo"
THESIS_FIGURE_SIZE = thesis_figure_size(0.90)
PUBLICATION_FIGURE_SIZE = (9.0, 6.8)
SURFACE_PRESSURE_SCALE = 1.0e6
SURFACE_PRESSURE_LABEL = r"$p_s\ [\mathrm{MPa}]$"
SURFACE_TEMPERATURE_LABEL = r"$T_s\ [\mathrm{K}]$"
THEORY_LINEWIDTH = 2.5

THEORY_STYLES = {
    r"$\mathrm{Isentropic\ KM}$": {
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": THEORY_LINEWIDTH,
    },
    r"$\mathrm{Isothermal\ KM}$": {
        "color": "#D55E00",
        "linestyle": "--",
        "linewidth": THEORY_LINEWIDTH,
    },
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Keller-Miksis bubble-surface temperature and pressure "
            "histories for the MFC or ECOGEN strong-collapse case."
        )
    )
    add_thesis_export_argument(parser, default_stem=None)
    add_show_titles_argument(parser)
    parser.add_argument(
        "--case",
        choices=tuple(CASE_BUILDERS),
        default="mfc",
        help="Strong-collapse KM case to solve and plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to save the figure. Defaults to a case-specific file "
            f"under {DEFAULT_OUTPUT_DIR} unless --to-thesis is used."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figure without opening an interactive window.",
    )
    return parser


def build_case_title(case_name: str) -> str:
    case_label = CASE_DISPLAY_NAMES[case_name]
    return rf"$\mathrm{{Bubble\ surface\ thermodynamics\ ({case_label}\ strong\ collapse)}}$"


def build_default_output_path(case_name: str) -> Path:
    return DEFAULT_OUTPUT_DIR / case_name / "bubble_surface_thermo.png"


def build_thesis_stem(case_name: str) -> str:
    return f"{THESIS_EXPORT_STEM}_{case_name.upper()}"


def build_surface_histories(case_name: str):
    case = CASE_BUILDERS[case_name]()
    return build_keller_miksis_theory_histories(case)


def plot_surface_histories(
    histories,
    *,
    thesis_mode: bool,
    show_titles: bool,
    title: str | None,
):
    if thesis_mode:
        apply_thesis_style()
        figure_size = THESIS_FIGURE_SIZE
    else:
        apply_publication_style()
        figure_size = PUBLICATION_FIGURE_SIZE

    figure, axes = plt.subplots(
        2,
        1,
        figsize=figure_size,
        sharex=True,
        constrained_layout=thesis_mode,
    )
    if thesis_mode:
        figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    for label, history in histories.items():
        theory_style = dict(THEORY_STYLES[label], zorder=3)
        axes[0].plot(
            history["normalized_time"],
            history["surface_temperature"],
            label=label,
            **theory_style,
        )
        axes[1].plot(
            history["normalized_time"],
            history["surface_pressure"] / SURFACE_PRESSURE_SCALE,
            label=label,
            **theory_style,
        )

    axes[0].set_ylabel(
        SURFACE_TEMPERATURE_LABEL,
        fontsize=THESIS_LABEL_FONT_SIZE if thesis_mode else None,
    )
    axes[1].set_ylabel(
        SURFACE_PRESSURE_LABEL,
        fontsize=THESIS_LABEL_FONT_SIZE if thesis_mode else None,
    )
    axes[1].set_xlabel(
        r"$t/t_c$",
        fontsize=THESIS_LABEL_FONT_SIZE if thesis_mode else None,
    )

    axes[0].grid(True, alpha=0.35)
    axes[1].grid(True, which="both", alpha=0.35)
    axes[1].set_yscale("log")

    legend_fontsize = THESIS_TICK_FONT_SIZE if thesis_mode else 10
    axes[0].legend(
        loc="best",
        fontsize=legend_fontsize,
        framealpha=0.95,
    )

    if show_titles and title is not None:
        figure.suptitle(
            title,
            fontsize=THESIS_TITLE_FONT_SIZE if thesis_mode else None,
        )

    axes[0].tick_params(labelsize=THESIS_TICK_FONT_SIZE if thesis_mode else None)
    axes[1].tick_params(labelsize=THESIS_TICK_FONT_SIZE if thesis_mode else None)

    if not thesis_mode:
        if show_titles:
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        else:
            figure.tight_layout()

    return figure, axes


def main(argv=None):
    args = build_argument_parser().parse_args(argv)

    if args.no_show:
        os.environ.setdefault("PYTHONSCRIPTS_NO_TEX", "1")
        plt.switch_backend("Agg")

    histories = build_surface_histories(args.case)
    for label, history in histories.items():
        temperature = history["surface_temperature"]
        pressure_mpa = history["surface_pressure"] / SURFACE_PRESSURE_SCALE
        print(
            f"{args.case.upper()} {label}: "
            f"t/tc end={history['normalized_time'][-1]:.6f}, "
            f"T_s range={temperature.min():.6f}..{temperature.max():.6f} K, "
            f"p_s range={pressure_mpa.min():.6f}..{pressure_mpa.max():.6f} MPa"
        )

    figure, _ = plot_surface_histories(
        histories,
        thesis_mode=args.to_thesis,
        show_titles=args.show_titles,
        title=build_case_title(args.case),
    )

    output_path = args.output
    if output_path is None and not args.to_thesis:
        output_path = build_default_output_path(args.case)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
        print(f"Figure written to {output_path}")

    thesis_stem = args.thesis_stem or build_thesis_stem(args.case)
    thesis_path = save_thesis_figure_from_args(
        figure,
        args,
        stem=thesis_stem,
    )
    if thesis_path is not None:
        print(f"Thesis PDF written to {thesis_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
