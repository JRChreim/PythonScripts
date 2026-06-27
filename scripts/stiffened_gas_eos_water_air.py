"""Dedicated two-phase water/air stiffened-gas EoS figure generator.

This script freezes the water/air parameter set used to illustrate the poorly
behaved curves and the ill-defined mixture region. Compared with the main
stiffened-gas script, it omits the metastable and mixture-path overlays and
extends the saturation sweep to a wider temperature range.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.plots.publication import add_thesis_export_argument, save_thesis_figure_from_args
from scripts.stiffened_gas_eos import (
    configure_plot_style,
    plot_3d_entropy_surfaces,
    plot_3d_gibbs_surfaces,
    plot_pv_overview_panel,
    plot_saturation_auxiliary_figure,
    plot_saturation_temperature_figure,
    print_summary,
    save_figures,
    THESIS_LAYOUT_PADS,
    THESIS_PV_FIGURE_SIZE,
    PV_FIGURE_SIZE,
)
from src.thermo.stiffened_gas import (
    PhaseParameters,
    Preset,
    build_saturation_constants,
    build_saturation_curve,
    build_three_phase_branch,
    compute_phase_fields,
    compute_reference_entropy,
)

THESIS_EXPORT_STEM = "ErroneouspvDiagram"
PRESSURE_GRID = np.linspace(1.0e1, 1.0e6, 1_000)
TEMPERATURE_GRID = np.linspace(373.15, 1373.15, 2)
SATURATION_TEMPERATURE_MIN = 373.15
SATURATION_TEMPERATURE_MAX = 1373.15
SATURATION_SAMPLE_COUNT = 100
PV_PRESSURE_TOP_GAP_FACTOR = 1.08


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the dedicated water/air stiffened-gas EoS figure with the "
            "same plotting workflow as the main analysis script."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where the generated figures will be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figures without opening an interactive matplotlib window.",
    )
    parser.add_argument(
        "--all-figures",
        action="store_true",
        help=(
            "Also generate the auxiliary entropy, Gibbs, and saturation "
            "figures from the main stiffened-gas script."
        ),
    )
    return parser


def build_preset():
    liquid = PhaseParameters(
        name="liquid",
        p_inf=1.0e9,
        q=-1.167e6,
        qp=0.0,
        cv=1816.0,
        cp=4267.0,
        color=(0.18, 0.55, 0.20),
    )
    vapor = PhaseParameters(
        name="vapor",
        p_inf=0.0,
        q=2.030e6,
        qp=-2.34e4,
        cv=1040.0,
        cp=1487.0,
        color=(0.05, 0.05, 0.05),
    )

    return Preset(
        name="water_air_two_phase",
        phases=(liquid, vapor),
        initial_volume_fractions=np.array([0.5, 0.5], dtype=float),
        reference_temperature=372.76,
        reference_pressure=1.0e5,
        reference_entropy=None,
    )


def plot_pv_diagram_figure(
    preset: Preset,
    saturation_curve,
    *,
    thesis_mode: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=THESIS_PV_FIGURE_SIZE if thesis_mode else PV_FIGURE_SIZE,
        constrained_layout=True,
    )
    if thesis_mode:
        fig.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    plot_pv_overview_panel(
        ax,
        preset,
        saturation_curve,
        branch=None,
        thesis_mode=thesis_mode,
    )
    pressure_min = float(np.min(saturation_curve.pressure))
    pressure_max = float(np.max(saturation_curve.pressure))
    ax.set_ylim(pressure_min, pressure_max * PV_PRESSURE_TOP_GAP_FACTOR)
    return fig


def build_analysis(
    preset: Preset,
    *,
    thesis_mode: bool,
    include_auxiliary_figures: bool,
):
    constants = build_saturation_constants(preset.phases[0], preset.phases[1])
    saturation_temperatures = np.linspace(
        SATURATION_TEMPERATURE_MIN,
        SATURATION_TEMPERATURE_MAX,
        SATURATION_SAMPLE_COUNT,
    )
    saturation_curve = build_saturation_curve(
        saturation_temperatures,
        preset,
        constants,
        report=print,
    )

    branch = build_three_phase_branch(
        preset,
        constants,
        saturation_curve,
        report=print,
    )
    pv_figure = plot_pv_diagram_figure(
        preset,
        saturation_curve,
        thesis_mode=thesis_mode,
    )
    figures: list[tuple[str, plt.Figure]] = [(f"{preset.name}_pv_diagram.png", pv_figure)]

    if include_auxiliary_figures:
        temperature_grid, pressure_grid = np.meshgrid(TEMPERATURE_GRID, PRESSURE_GRID)
        reference_entropy = (
            preset.reference_entropy
            if preset.reference_entropy is not None
            else compute_reference_entropy(
                preset.phases,
                preset.reference_temperature,
                preset.reference_pressure,
            )
        )
        phase_fields = [
            compute_phase_fields(
                pressure_grid,
                temperature_grid,
                phase,
                float(reference_entropy[index]),
                preset.reference_temperature,
                preset.reference_pressure,
            )
            for index, phase in enumerate(preset.phases)
        ]

        entropy_figure = plot_3d_entropy_surfaces(
            pressure_grid,
            temperature_grid,
            preset.phases,
            phase_fields,
        )
        gibbs_figure = plot_3d_gibbs_surfaces(
            pressure_grid,
            temperature_grid,
            preset.phases,
            phase_fields,
        )
        saturation_figure = plot_saturation_temperature_figure(
            preset,
            saturation_curve,
        )
        auxiliary_figure = plot_saturation_auxiliary_figure(preset, saturation_curve)

        figures = [
            (f"{preset.name}_entropy_surfaces.png", entropy_figure),
            (f"{preset.name}_gibbs_surfaces.png", gibbs_figure),
            (f"{preset.name}_saturation_vs_temperature.png", saturation_figure),
            (f"{preset.name}_saturation_auxiliary.png", auxiliary_figure),
            (f"{preset.name}_pv_diagram.png", pv_figure),
        ]

    return figures, saturation_curve, branch, pv_figure


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    preset = build_preset()
    thesis_mode = bool(args.to_thesis)

    configure_plot_style(thesis_mode)

    figures, saturation_curve, branch, pv_figure = build_analysis(
        preset,
        thesis_mode=thesis_mode,
        include_auxiliary_figures=bool(args.all_figures),
    )
    print_summary(preset, saturation_curve, branch)

    if args.output_dir is not None:
        save_figures(figures, args.output_dir)
        print(f"Saved figures to {args.output_dir}")

    if args.to_thesis:
        thesis_path = save_thesis_figure_from_args(
            pv_figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        if thesis_path is not None:
            print(f"Thesis figure written to {thesis_path}")

    if args.no_show:
        for _, figure in figures:
            plt.close(figure)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
