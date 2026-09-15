from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    add_thesis_export_argument,
    apply_thesis_style,
    latex_text,
    save_thesis_figure_from_args,
    thesis_figure_size,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FOLDER = Path(
    "/disk/simulations/Relaxation/Thesis/ExpansionTube/pT/5Eqn/binary"
)
DEFAULT_VARIABLES = ("pres", "vel1", "alpha1", "alpha2", "alpha3")
DEFAULT_SAMPLE_COUNT = 5
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "figures" / "mfc_binary_quicklook.png"
THESIS_EXPORT_STEM = "MFC_ExpansionTube_pT_5Eqn_binary"

FIELD_LABELS = {
    "alpha_rho1": r"$\alpha_1 \rho_1\ [\mathrm{kg\,m^{-3}}]$",
    "alpha_rho2": r"$\alpha_2 \rho_2\ [\mathrm{kg\,m^{-3}}]$",
    "alpha_rho3": r"$\alpha_3 \rho_3\ [\mathrm{kg\,m^{-3}}]$",
    "alpha1": r"$\alpha_1$",
    "alpha2": r"$\alpha_2$",
    "alpha3": r"$\alpha_3$",
    "pres": r"$p\ [\mathrm{Pa}]$",
    "vel1": r"$u\ [\mathrm{m\,s^{-1}}]$",
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a few 1D profiles from the MFC binary output stored under a "
            "simulation case directory."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help="Path to the MFC case directory or directly to a snapshot directory.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help=(
            "Snapshot variables to plot, for example pres vel1 alpha1 alpha2 alpha3 "
            "or alpha_rho1 alpha_rho2 alpha_rho3."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        help="Saved step numbers to plot. Defaults to a few evenly spaced snapshots.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of evenly spaced snapshots to sample when --steps is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save the figure. Defaults to "
            f"{DEFAULT_OUTPUT} unless --to-thesis is used."
        ),
    )
    parser.add_argument(
        "--title",
        help="Optional custom figure title.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figure without opening an interactive window.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)

    snapshot_directory = discover_mfc_binary_snapshot_directory(args.data_folder)
    available_steps = discover_mfc_binary_steps(snapshot_directory)
    selected_steps = _select_steps(
        available_steps,
        requested_steps=args.steps,
        sample_count=args.sample_count,
    )

    snapshots = [
        load_mfc_binary_snapshot(snapshot_directory / f"{step}.dat")
        for step in selected_steps
    ]
    figure = build_mfc_binary_quicklook_figure(
        snapshots,
        variables=args.variables,
        title=args.title,
        thesis_mode=args.to_thesis,
        show_titles=not args.to_thesis,
    )

    output_path = args.output
    if output_path is None and not args.to_thesis:
        output_path = DEFAULT_OUTPUT

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200)
        print(f"Figure written to {output_path}")

    thesis_path = save_thesis_figure_from_args(
        figure,
        args,
        stem=THESIS_EXPORT_STEM,
    )
    if thesis_path is not None:
        print(f"Thesis PDF written to {thesis_path}")

    print(
        "Loaded MFC snapshots from "
        f"{snapshot_directory} at steps {', '.join(str(step) for step in selected_steps)}."
    )

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


def build_mfc_binary_quicklook_figure(
    snapshots,
    *,
    variables: list[str] | tuple[str, ...],
    title: str | None,
    thesis_mode: bool,
    show_titles: bool,
):
    if not snapshots:
        raise ValueError("At least one snapshot is required to build the figure.")

    if thesis_mode:
        apply_thesis_style()
        figure_size = thesis_figure_size(max(0.28 * len(variables), 0.55))
    else:
        apply_thesis_style()
        figure_size = (10.5, max(2.0 * len(variables), 4.8))

    figure, axes = plt.subplots(
        len(variables),
        1,
        figsize=figure_size,
        sharex=True,
        squeeze=False,
    )

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(snapshots)))
    reference_x = snapshots[0].x_centers

    for axis, variable in zip(axes.ravel(), variables):
        _plot_variable_on_axis(axis, snapshots, reference_x, variable, colors)

    for axis in axes.ravel():
        axis.grid(True, alpha=0.35)

    axes[-1, 0].set_xlabel(r"$x\ [\mathrm{m}]$")

    if title is None:
        title = latex_text("MFC pT 5 equation binary quicklook")

    if show_titles:
        figure.suptitle(_ensure_latex_title(title))
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    return figure


def _plot_variable_on_axis(axis, snapshots, reference_x, variable: str, colors):
    plotted_anything = False
    for color, snapshot in zip(colors, snapshots):
        if variable not in snapshot.fields:
            available = ", ".join(sorted(snapshot.fields))
            raise ValueError(
                f"Variable '{variable}' is not available in {snapshot.path.name}. "
                f"Available fields: {available}"
            )

        if not np.allclose(snapshot.x_centers, reference_x):
            raise ValueError(
                f"Snapshot {snapshot.path} does not share the same x-grid as the "
                "first loaded snapshot."
            )

        axis.plot(
            reference_x,
            snapshot.fields[variable],
            color=color,
            linewidth=2.0,
            label=latex_text(f"step {snapshot.step}"),
        )
        plotted_anything = True

    axis.set_ylabel(_build_variable_label(variable))
    axis.legend(
        ncol=2 if len(snapshots) > 3 else 1,
        fontsize=8,
        loc="upper right",
        framealpha=0.85,
    )

    if not plotted_anything:
        axis.text(
            0.5,
            0.5,
            latex_text("No data plotted"),
            ha="center",
            va="center",
            transform=axis.transAxes,
        )


def _build_variable_label(variable: str) -> str:
    return FIELD_LABELS.get(variable, latex_text(variable))


def _select_steps(
    available_steps: tuple[int, ...],
    *,
    requested_steps: list[int] | tuple[int, ...] | None,
    sample_count: int,
) -> list[int]:
    if not available_steps:
        raise ValueError("No MFC snapshots were found in the selected directory.")

    if requested_steps:
        missing_steps = [step for step in requested_steps if step not in available_steps]
        if missing_steps:
            raise ValueError(
                "The following requested steps are not available: "
                f"{missing_steps}. Available steps include {list(available_steps[:10])}"
            )
        return list(requested_steps)

    sample_count = max(1, min(sample_count, len(available_steps)))
    sample_indices = np.linspace(
        0,
        len(available_steps) - 1,
        num=sample_count,
    )
    selected_indices = sorted({int(round(index)) for index in sample_indices})
    return [available_steps[index] for index in selected_indices]


def _ensure_latex_title(title: str) -> str:
    if "$" in title:
        return title
    return latex_text(title)


if __name__ == "__main__":
    main()
