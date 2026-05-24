from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.plots.publication import (
    THESIS_LABEL_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    THESIS_TITLE_FONT_SIZE,
    THESIS_TICK_FONT_SIZE,
    add_thesis_export_argument,
    apply_thesis_style,
    latex_text,
    save_thesis_figure_from_args,
    thesis_figure_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "artifacts" / "Data" / "SimoesMoreira"

DEFAULT_DENSITY_INPUT = DATA_DIR / "RhoVSx.csv"
DEFAULT_VELOCITY_5EQ_INPUT = DATA_DIR / "UxT_5Eq.csv"
DEFAULT_VELOCITY_6EQ_INPUT = DATA_DIR / "UxT_6Eq.csv"
DEFAULT_VELOCITY_EXP_INPUT = DATA_DIR / "UxT_Exp.csv"
THESIS_EXPORT_STEM = "NDodecane"
DEFAULT_OUTPUT = None
DEFAULT_X_MIN = 0.4
DEFAULT_X_MAX = 1.0
DEFAULT_HIGHLIGHT_START = 0.748
DEFAULT_HIGHLIGHT_END = 0.783
DEFAULT_T_MIN = 450.0
DEFAULT_T_MAX = 600.0
DEFAULT_UF_MIN = 0.0
DEFAULT_UF_MAX = 1.8
FIGURE_SIZE = (15.8, 6.1)
THESIS_FIGURE_SIZE = thesis_figure_size(0.55)
WIDTH_RATIOS = (1.0, 1.0)
FIGURE_LAYOUT_PADS = {
    "w_pad": 0.04,
    "h_pad": 0.03,
    "wspace": 0.05,
    "hspace": 0.03,
}
PLOT_FONT_SIZE = 16
LEGEND_FONT_SIZE = 13
THESIS_DENSITY_LINEWIDTH = 1.2
THESIS_EXP_MARKERSIZE = 5.5
THESIS_MODEL_MARKERSIZE = 6.5
THESIS_MODEL_LINEWIDTH = 0.8
THESIS_ANNOTATION_FONT_SIZE = 11
EXPERIMENT_LABEL = r"$\mathrm{Experimental}$"
MODEL_5EQ_LABEL = r"$\mathrm{5\mbox{-}equation}$"
MODEL_6EQ_LABEL = r"$\mathrm{6\mbox{-}equation}$"
ANNOTATION_BBOX = {
    "boxstyle": "round,pad=0.18",
    "facecolor": (1.0, 1.0, 1.0, 0.0),
    "edgecolor": "none",
}
ANNOTATION_ARROW = {
    "arrowstyle": "->",
    "color": "0.2",
    "lw": 1.0,
    "shrinkA": 0,
    "shrinkB": 0,
    "mutation_scale": 12,
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Simoes-Moreira density profile and the U_F versus T data "
            "in a 1x2 subplot layout."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DENSITY_INPUT,
        help="Two-column CSV file containing x [m] and rho [kg/m^3].",
    )
    parser.add_argument(
        "--velocity-5eq-input",
        type=Path,
        default=DEFAULT_VELOCITY_5EQ_INPUT,
        help="Two-column CSV file containing T [K] and U_F [m/s] for the 5-equation model.",
    )
    parser.add_argument(
        "--velocity-6eq-input",
        type=Path,
        default=DEFAULT_VELOCITY_6EQ_INPUT,
        help="Two-column CSV file containing T [K] and U_F [m/s] for the 6-equation model.",
    )
    parser.add_argument(
        "--velocity-exp-input",
        type=Path,
        default=DEFAULT_VELOCITY_EXP_INPUT,
        help="Two-column CSV file containing T [K] and U_F [m/s] for the experimental data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Optional path to save the figure.",
    )
    parser.add_argument(
        "--title",
        help="Optional figure title.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=DEFAULT_X_MIN,
        help="Lower x-axis limit used to trim the left side of the plot.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=DEFAULT_X_MAX,
        help="Upper x-axis limit used for the plot.",
    )
    parser.add_argument(
        "--highlight-start",
        type=float,
        default=DEFAULT_HIGHLIGHT_START,
        help="Left boundary of the gray highlight band.",
    )
    parser.add_argument(
        "--highlight-end",
        type=float,
        default=DEFAULT_HIGHLIGHT_END,
        help="Right boundary of the gray highlight band.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figure without opening an interactive window.",
    )
    return parser


def load_two_column_csv(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    data = np.loadtxt(filepath, delimiter=",")
    if data.ndim == 1:
        data = np.atleast_2d(data)

    if data.shape[1] != 2:
        raise ValueError(
            f"Expected two numeric columns in {filepath}, got shape {data.shape}."
        )

    x = data[:, 0]
    rho = data[:, 1]
    if np.any(rho <= 0.0):
        raise ValueError(
            "Semilog-y plotting requires strictly positive density values, "
            f"but {filepath} contains non-positive entries."
        )
    return x, rho


def build_analysis_figure(
    x: np.ndarray,
    rho: np.ndarray,
    velocity_series: list[dict[str, np.ndarray | str]],
    *,
    thesis_mode: bool,
    figure_size: tuple[float, float],
    density_linewidth: float,
    label_font_size: float,
    title_font_size: float,
    tick_font_size: float,
    legend_font_size: float,
    annotation_font_size: float,
    title: str | None = None,
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    highlight_start: float = DEFAULT_HIGHLIGHT_START,
    highlight_end: float = DEFAULT_HIGHLIGHT_END,
):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        constrained_layout=True,
        gridspec_kw={"width_ratios": WIDTH_RATIOS},
    )
    if thesis_mode:
        layout_pads = {
            key: max(FIGURE_LAYOUT_PADS[key], THESIS_LAYOUT_PADS[key])
            for key in FIGURE_LAYOUT_PADS
        }
    else:
        layout_pads = FIGURE_LAYOUT_PADS
    if x_max <= x_min:
        raise ValueError(
            "x_max must be greater than x_min for the density subplot. "
            f"Got x_min={x_min:g}, x_max={x_max:g}."
        )
    figure.set_constrained_layout_pads(**layout_pads)
    left_axis = axes[0]
    right_axis = axes[1]

    left_axis.axvspan(
        highlight_start,
        highlight_end,
        ymin=0.0,
        ymax=1.0,
        facecolor="0.8",
        edgecolor="0.45",
        alpha=0.45,
        linewidth=1.0,
        zorder=0.5,
    )
    left_axis.semilogy(x, rho, color="black", linewidth=density_linewidth, zorder=2.0)
    left_axis.set_xlabel(r"$x\ [\mathrm{m}]$", fontsize=label_font_size)
    left_axis.set_ylabel(r"$\rho\ [\mathrm{kg\,m^{-3}}]$", fontsize=label_font_size)
    left_axis.tick_params(axis="both", labelsize=tick_font_size)
    if title is not None:
        left_axis.set_title(_ensure_latex_title(title), fontsize=title_font_size)
    left_axis.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.32)
    left_axis.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.28)
    left_axis.set_xlim(x_min, x_max)

    _add_region_annotations(
        left_axis,
        x,
        rho,
        highlight_start,
        highlight_end,
        annotation_font_size=annotation_font_size,
    )

    _plot_velocity_panel(
        right_axis,
        velocity_series,
        label_font_size=label_font_size,
        tick_font_size=tick_font_size,
        legend_font_size=legend_font_size,
    )

    return figure, axes


def _plot_velocity_panel(
    axis,
    velocity_series: list[dict[str, np.ndarray | str]],
    *,
    label_font_size: float,
    tick_font_size: float,
    legend_font_size: float,
) -> None:
    if not velocity_series:
        raise ValueError("At least one velocity series is required for the second panel.")

    for series in velocity_series:
        axis.plot(
            series["temperature"],
            series["uf"],
            label=series["label"],
            color=series["color"],
            marker=series["marker"],
            linestyle=series["linestyle"],
            markersize=series["markersize"],
            linewidth=series["linewidth"],
            markerfacecolor=series.get("markerfacecolor", series["color"]),
            markeredgecolor=series.get("markeredgecolor", series["color"]),
            markeredgewidth=series.get("markeredgewidth", 0.8),
            zorder=series.get("zorder", 2.0),
        )

    axis.set_xlabel(r"$T\ [\mathrm{K}]$", fontsize=label_font_size)
    axis.set_ylabel(r"$U_F\ [\mathrm{m\,s^{-1}}]$", fontsize=label_font_size)
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()
    axis.tick_params(
        axis="both",
        labelsize=tick_font_size,
        direction="in",
        top=True,
        right=True,
        left=False,
        labelleft=False,
        labelright=True,
    )
    axis.xaxis.set_major_locator(MultipleLocator(50.0))
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.set_xlim(DEFAULT_T_MIN, DEFAULT_T_MAX)
    axis.set_ylim(DEFAULT_UF_MIN, DEFAULT_UF_MAX)
    axis.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.28)

    legend = axis.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.65",
        fontsize=legend_font_size,
        handlelength=1.3,
        handletextpad=0.45,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(0.8)


def _add_region_annotations(
    axis,
    x: np.ndarray,
    rho: np.ndarray,
    highlight_start: float,
    highlight_end: float,
    *,
    annotation_font_size: float,
) -> None:
    highlight_mid = 0.5 * (highlight_start + highlight_end)
    before_target_x = float(np.clip(highlight_start, float(np.min(x)), float(np.max(x))))
    after_target_x = float(np.clip(highlight_end, float(np.min(x)), float(np.max(x))))
    before_target_y = float(np.interp(before_target_x, x, rho))
    after_target_y = float(np.interp(after_target_x, x, rho))

    axis.annotate(
        r"$\mathrm{before\ (b)}$",
        xy=(before_target_x, before_target_y),
        xytext=(0.68, 100.0),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=annotation_font_size,
        fontfamily="serif",
        bbox=ANNOTATION_BBOX,
        arrowprops=ANNOTATION_ARROW,
        zorder=3.0,
    )
    axis.annotate(
        r"$\begin{array}{c}\mathrm{Evaporation}\\\mathrm{wave}\end{array}$",
        xy=(highlight_mid, 2.5),
        xytext=(0.84, 100.0),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=annotation_font_size,
        fontfamily="serif",
        bbox=ANNOTATION_BBOX,
        arrowprops=ANNOTATION_ARROW,
        zorder=3.0,
    )
    axis.annotate(
        r"$\mathrm{after\ (a)}$",
        xy=(after_target_x, after_target_y),
        xytext=(0.83, 0.01),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=annotation_font_size,
        fontfamily="serif",
        bbox=ANNOTATION_BBOX,
        arrowprops=ANNOTATION_ARROW,
        zorder=3.0,
    )


def _ensure_latex_title(title: str) -> str:
    if "$" in title:
        return title
    return latex_text(title)


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    thesis_mode = bool(args.to_thesis)
    apply_thesis_style()

    x, rho = load_two_column_csv(args.input)
    figure_size = THESIS_FIGURE_SIZE if thesis_mode else FIGURE_SIZE
    density_linewidth = THESIS_DENSITY_LINEWIDTH if thesis_mode else 1.7
    label_font_size = THESIS_LABEL_FONT_SIZE if thesis_mode else PLOT_FONT_SIZE
    title_font_size = THESIS_TITLE_FONT_SIZE if thesis_mode else PLOT_FONT_SIZE
    tick_font_size = THESIS_TICK_FONT_SIZE if thesis_mode else PLOT_FONT_SIZE
    legend_font_size = THESIS_TICK_FONT_SIZE if thesis_mode else LEGEND_FONT_SIZE
    annotation_font_size = (
        THESIS_ANNOTATION_FONT_SIZE if thesis_mode else 18
    )
    exp_markersize = THESIS_EXP_MARKERSIZE if thesis_mode else 7.5
    model_markersize = THESIS_MODEL_MARKERSIZE if thesis_mode else 9.0
    model_linewidth = THESIS_MODEL_LINEWIDTH if thesis_mode else 1.0

    velocity_series = [
        _build_velocity_series(
            *load_two_column_csv(args.velocity_exp_input),
            label=EXPERIMENT_LABEL,
            color="black",
            marker="s",
            linestyle="None",
            markersize=exp_markersize,
            linewidth=0.0,
            zorder=4.0,
        ),
        _build_velocity_series(
            *load_two_column_csv(args.velocity_5eq_input),
            label=MODEL_5EQ_LABEL,
            color="0.60",
            marker="v",
            linestyle=(0, (4.0, 3.0)),
            markersize=model_markersize,
            linewidth=model_linewidth,
            zorder=3.0,
        ),
        _build_velocity_series(
            *load_two_column_csv(args.velocity_6eq_input),
            label=MODEL_6EQ_LABEL,
            color="0.75",
            marker="v",
            linestyle=(0, (4.0, 3.0)),
            markersize=model_markersize,
            linewidth=model_linewidth,
            zorder=2.0,
        ),
    ]
    figure, _ = build_analysis_figure(
        x,
        rho,
        velocity_series,
        thesis_mode=thesis_mode,
        figure_size=figure_size,
        density_linewidth=density_linewidth,
        label_font_size=label_font_size,
        title_font_size=title_font_size,
        tick_font_size=tick_font_size,
        legend_font_size=legend_font_size,
        annotation_font_size=annotation_font_size,
        title=args.title,
        x_min=args.x_min,
        x_max=args.x_max,
        highlight_start=args.highlight_start,
        highlight_end=args.highlight_end,
    )

    if thesis_mode:
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        print(f"Figure written to {output_path}")
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=200)
        print(f"Figure written to {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


def _build_velocity_series(
    temperature: np.ndarray,
    uf: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle,
    markersize: float,
    linewidth: float,
    zorder: float,
) -> dict[str, np.ndarray | str]:
    return {
        "temperature": temperature,
        "uf": uf,
        "label": label,
        "color": color,
        "marker": marker,
        "linestyle": linestyle,
        "markersize": markersize,
        "linewidth": linewidth,
        "zorder": zorder,
    }


if __name__ == "__main__":
    main()
