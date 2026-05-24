"""Translate the MATLAB stiffened-gas EoS analysis into Python.

The script reproduces the pressure-temperature field calculations, entropy
surfaces, Gibbs free-energy comparison, and the saturation-curve sweeps from
the MATLAB source. The default run focuses on the dedicated p-v figure, which
combines the schematic with the metastable liquid and metastable path curves
on a single axis for thesis export. The auxiliary MATLAB-style figures remain
available behind an opt-in flag.

The default preset matches the active two-fluid dodecane case from the MATLAB
file. A three-fluid water/air preset is also provided so the later saturation
branch can be exercised when all three constituents are available.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.plots.publication import (
    THESIS_TICK_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    add_thesis_export_argument,
    apply_publication_style,
    apply_thesis_style,
    escape_latex_text,
    latex_text,
    save_thesis_figure_from_args,
    thesis_figure_size,
)
from src.thermo.stiffened_gas import (
    PhaseFields,
    PhaseParameters,
    Preset,
    SaturationCurve,
    ThreePhaseBranch,
    build_saturation_constants,
    build_saturation_curve,
    build_three_phase_branch,
    compute_phase_fields,
    compute_reference_entropy,
    solve_saturation_temperature,
)


PRESSURE_GRID = np.linspace(1.0e1, 1.0e6, 1_000)
TEMPERATURE_GRID = np.linspace(273.15, 673.15, 2)
DEFAULT_PRESET = "dodecane"
THESIS_EXPORT_STEM = "pvDiagram"
PV_FIGURE_SIZE = (14.5, 11.0)
# Use the thesis text width for single figures and keep a 2:1 width-to-height ratio.
THESIS_PV_FIGURE_SIZE = thesis_figure_size(0.5)
PV_VOLUME_RANGE = (2.0e-4, 2.0e1)
PV_PRESSURE_MIN = 1.0e3
PV_PRESSURE_MAX = 2.0e7
SATURATION_SAMPLE_COUNT = 100
SATURATION_TEMPERATURE_MAX = 572.0 + 273.15
SATURATION_PRESSURE_MIN = 1.0e3
METASTABLE_TEMPERATURE = 398.15
METASTABLE_PRESSURE_START = 5.0e4
METASTABLE_PRESSURE_STOP = 2.0e3
METASTABLE_SAMPLE_COUNT = 300
METASTABLE_SEGMENT2_VOLUME_SCALE = 1.5


def build_presets() -> dict[str, Preset]:
    dodecane = Preset(
        name="dodecane",
        phases=(
            PhaseParameters(
                name="liquid",
                p_inf=4.0e8,
                q=-775.269e3,
                qp=0.0,
                cv=1077.7,
                cp=2.35 * 1077.7,
                color=(0.18, 0.55, 0.20),
            ),
            PhaseParameters(
                name="vapor",
                p_inf=0.0,
                q=-237.547e3,
                qp=-2.44e4,
                cv=1956.45,
                cp=1.025 * 1956.45,
                color=(0.05, 0.05, 0.05),
            ),
        ),
        initial_volume_fractions=np.array([0.48, 0.52], dtype=float),
        reference_temperature=372.76,
        reference_pressure=1.0e5,
        reference_entropy=np.array([1.302e3, 7.359e3], dtype=float),
    )

    water_air = Preset(
        name="water_air",
        phases=(
            PhaseParameters(
                name="liquid",
                p_inf=1.0e9,
                q=-1.167e6,
                qp=0.0,
                cv=1816.0,
                cp=4267.0,
                color=(0.18, 0.55, 0.20),
            ),
            PhaseParameters(
                name="vapor",
                p_inf=0.0,
                q=2.030e6,
                qp=-2.34e4,
                cv=1040.0,
                cp=1487.0,
                color=(0.05, 0.05, 0.05),
            ),
            PhaseParameters(
                name="air",
                p_inf=0.0,
                q=0.0,
                qp=0.0,
                cv=717.5,
                cp=1006.0,
                color=(0.20, 0.35, 0.75),
            ),
        ),
        initial_volume_fractions=np.array([0.48, 0.04, 0.48], dtype=float),
        reference_temperature=372.76,
        reference_pressure=1.0e5,
        reference_entropy=None,
    )

    dodecane_alt = Preset(
        name="dodecane_alt",
        phases=(
            PhaseParameters(
                name="liquid",
                p_inf=1.804e8,
                q=-9.96336e5,
                qp=0.0,
                cv=2430.0,
                cp=3056.0,
                color=(0.18, 0.55, 0.20),
            ),
            PhaseParameters(
                name="vapor",
                p_inf=0.0,
                q=-3.84592e5,
                qp=-6.280e3,
                cv=2274.0,
                cp=2322.0,
                color=(0.05, 0.05, 0.05),
            ),
        ),
        initial_volume_fractions=np.array([0.48, 0.52], dtype=float),
        reference_temperature=372.76,
        reference_pressure=1.0e5,
        reference_entropy=None,
    )

    return {
        "dodecane": dodecane,
        "water_air": water_air,
        "dodecane_alt": dodecane_alt,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the stiffened-gas MATLAB analysis in Python, with the "
            "p-v figure as the default output."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--preset",
        choices=sorted(build_presets().keys()),
        default=DEFAULT_PRESET,
        help="Thermodynamic parameter set to analyze.",
    )
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
            "Also generate the auxiliary entropy, Gibbs, saturation, and "
            "branch figures from the MATLAB script."
        ),
    )
    return parser


def configure_plot_style(thesis_mode: bool) -> None:
    if thesis_mode:
        apply_thesis_style()
    else:
        apply_publication_style()


def quantity_label(symbol_tex: str, unit_tex: str) -> str:
    return rf"${symbol_tex}\ [\mathrm{{{unit_tex}}}]$"


def symbol_label(symbol_tex: str) -> str:
    return rf"${symbol_tex}$"


def phase_quantity_label(
    quantity_tex: str,
    phase_name: str,
    unit_tex: str,
) -> str:
    return rf"${quantity_tex}_{{\mathrm{{{escape_latex_text(phase_name)}}}}}\ [\mathrm{{{unit_tex}}}]$"


def phase_sum_quantity_label(
    quantity_tex: str,
    first_phase: str,
    second_phase: str,
    unit_tex: str,
) -> str:
    return (
        rf"${quantity_tex}_{{\mathrm{{{escape_latex_text(first_phase)}}}}}"
        rf" + {quantity_tex}_{{\mathrm{{{escape_latex_text(second_phase)}}}}}"
        rf"\ [\mathrm{{{unit_tex}}}]$"
    )


def phase_legend_label(phase_name: str) -> str:
    return latex_text(phase_name)


def stacked_latex_lines(*lines: str) -> str:
    return "\n".join(latex_text(line) for line in lines)


def plot_3d_entropy_surfaces(
    pressure_grid: np.ndarray,
    temperature_grid: np.ndarray,
    phases: tuple[PhaseParameters, ...],
    fields: list[PhaseFields],
) -> plt.Figure:
    fig = plt.figure(figsize=(13.5, 9.0))
    ax = fig.add_subplot(111, projection="3d")

    for phase, phase_fields in zip(phases, fields, strict=True):
        ax.plot_surface(
            pressure_grid,
            temperature_grid,
            phase_fields.s,
            color=phase.color,
            edgecolor="none",
            linewidth=0.0,
            antialiased=True,
            alpha=0.95,
        )

    ax.set_xlabel(quantity_label("p", "Pa"))
    ax.set_ylabel(quantity_label("T", "K"))
    ax.set_zlabel(quantity_label("s", "J\\,kg^{-1}\\,K^{-1}"))
    legend_handles = [
        Patch(facecolor=phase.color, edgecolor="none", label=phase_legend_label(phase.name))
        for phase in phases
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)
    ax.view_init(elev=28.0, azim=-135.0)
    fig.tight_layout()
    return fig


def plot_3d_gibbs_surfaces(
    pressure_grid: np.ndarray,
    temperature_grid: np.ndarray,
    phases: tuple[PhaseParameters, ...],
    fields: list[PhaseFields],
) -> plt.Figure:
    fig = plt.figure(figsize=(13.5, 9.0))
    ax = fig.add_subplot(111, projection="3d")

    colors = [(0.50, 0.20, 0.30), (0.00, 0.00, 0.00)]
    for color, phase, phase_fields in zip(colors, phases[:2], fields[:2], strict=True):
        ax.plot_surface(
            pressure_grid,
            temperature_grid,
            phase_fields.g,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            antialiased=True,
            alpha=0.95,
        )

    ax.set_xlabel(quantity_label("p", "Pa"))
    ax.set_ylabel(quantity_label("T", "K"))
    ax.set_zlabel(quantity_label("g", "J\\,kg^{-1}"))
    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=phase_legend_label(phase.name))
        for color, phase in zip(colors, phases[:2], strict=True)
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)
    ax.view_init(elev=28.0, azim=-135.0)
    fig.tight_layout()
    return fig


def configure_cartesian_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    *,
    logx: bool = False,
    logy: bool = False,
) -> None:
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.35, linewidth=0.8)


def plot_pv_curves(
    ax: plt.Axes,
    pressure: np.ndarray,
    rho_series: np.ndarray,
    labels: list[str],
    *,
    linestyle: str = "-",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    colors: list[str] | tuple[str, ...] | None = None,
) -> None:
    for index, (density, label) in enumerate(zip(rho_series, labels, strict=True)):
        line_color = None
        if colors is not None and index < len(colors):
            line_color = colors[index]
        ax.loglog(
            1.0 / density,
            pressure,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            color=line_color,
            label=label,
        )


def compute_isothermal_liquid_curve(
    phase: PhaseParameters,
    temperature: float,
    pressure_start: float,
    pressure_stop: float,
    *,
    sample_count: int = METASTABLE_SAMPLE_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.logspace(
        np.log10(pressure_start),
        np.log10(pressure_stop),
        sample_count,
    )
    specific_volume = (
        phase.cv
        * (phase.gamma - 1.0)
        * temperature
        / (pressure + phase.p_inf)
    )
    return pressure, specific_volume


def interpolate_saturation_point(
    saturation_curve: SaturationCurve,
    temperature: float,
    *,
    phase_index: int = 0,
) -> tuple[float, float]:
    pressure_sat = float(np.interp(temperature, saturation_curve.temperature, saturation_curve.pressure))
    specific_volume_sat = float(
        np.interp(
            temperature,
            saturation_curve.temperature,
            1.0 / saturation_curve.rho[phase_index],
        )
    )
    return pressure_sat, specific_volume_sat


def plot_saturation_temperature_figure(
    preset: Preset,
    saturation_curve: SaturationCurve,
) -> plt.Figure:
    phase_1 = preset.phases[0].name
    temperature_label = quantity_label("T", "K")
    pressure_label = quantity_label("p", "Pa")
    energy_density_label = quantity_label("E_{\\mathrm{T}}", "J\\,m^{-3}")
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5), constrained_layout=True)
    axes_flat = axes.ravel()

    axes_flat[0].plot(saturation_curve.temperature, saturation_curve.pressure, linewidth=1.5)
    configure_cartesian_axis(axes_flat[0], temperature_label, pressure_label)

    axes_flat[1].plot(
        saturation_curve.mass_density[0],
        saturation_curve.pressure,
        linewidth=1.5,
    )
    configure_cartesian_axis(
        axes_flat[1],
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
        pressure_label,
    )

    axes_flat[2].plot(
        saturation_curve.temperature,
        saturation_curve.mass_density[0],
        linewidth=1.5,
    )
    configure_cartesian_axis(
        axes_flat[2],
        temperature_label,
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
    )

    axes_flat[3].plot(
        saturation_curve.temperature,
        saturation_curve.total_energy_density,
        linewidth=1.5,
    )
    configure_cartesian_axis(axes_flat[3], temperature_label, energy_density_label)

    axes_flat[4].plot(
        saturation_curve.mass_density[0],
        saturation_curve.total_energy_density,
        linewidth=1.5,
    )
    configure_cartesian_axis(
        axes_flat[4],
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
        energy_density_label,
    )

    axes_flat[5].plot(
        saturation_curve.pressure,
        saturation_curve.total_energy_density,
        linewidth=1.5,
    )
    configure_cartesian_axis(axes_flat[5], pressure_label, energy_density_label)

    return fig


def plot_saturation_auxiliary_figure(
    preset: Preset,
    saturation_curve: SaturationCurve,
) -> plt.Figure:
    phase_1 = preset.phases[0].name
    phase_2 = preset.phases[1].name
    temperature_label = quantity_label("T", "K")
    pressure_label = quantity_label("p", "Pa")
    energy_density_label = quantity_label("E_{\\mathrm{T}}", "J\\,m^{-3}")
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5), constrained_layout=True)
    axes_flat = axes.ravel()

    total_phase_mass = saturation_curve.mass_density[:2].sum(axis=0)

    axes_flat[0].plot(total_phase_mass, saturation_curve.total_energy_density, linewidth=1.5)
    configure_cartesian_axis(
        axes_flat[0],
        phase_sum_quantity_label("m", phase_1, phase_2, "kg\\,m^{-3}"),
        energy_density_label,
    )

    axes_flat[1].plot(
        saturation_curve.temperature,
        1.0 / saturation_curve.rho[0],
        linewidth=1.5,
    )
    configure_cartesian_axis(
        axes_flat[1],
        temperature_label,
        phase_quantity_label("v", phase_1, "m^3\\,kg^{-1}"),
    )

    axes_flat[2].plot(
        saturation_curve.temperature,
        1.0 / saturation_curve.rho[1],
        linewidth=1.5,
    )
    configure_cartesian_axis(
        axes_flat[2],
        temperature_label,
        phase_quantity_label("v", phase_2, "m^3\\,kg^{-1}"),
    )

    plot_pv_curves(
        axes_flat[3],
        saturation_curve.pressure,
        saturation_curve.rho[:2],
        [phase_legend_label(phase_1), phase_legend_label(phase_2)],
    )
    configure_cartesian_axis(
        axes_flat[3],
        quantity_label("v", "m^3\\,kg^{-1}"),
        pressure_label,
        logx=True,
        logy=True,
    )
    axes_flat[3].legend(frameon=True)

    axes_flat[4].plot(
        saturation_curve.mass_density[0],
        saturation_curve.mass_density[1],
        linewidth=1.5,
    )
    configure_cartesian_axis(
        axes_flat[4],
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
        phase_quantity_label("m", phase_2, "kg\\,m^{-3}"),
    )

    axes_flat[5].plot(
        saturation_curve.temperature,
        total_phase_mass,
        linewidth=1.5,
    )
    configure_cartesian_axis(
        axes_flat[5],
        temperature_label,
        phase_sum_quantity_label("m", phase_1, phase_2, "kg\\,m^{-3}"),
    )

    return fig


def plot_three_phase_branch_figures(
    preset: Preset,
    branch: ThreePhaseBranch,
) -> list[plt.Figure]:
    phase_1 = preset.phases[0].name
    phase_2 = preset.phases[1].name
    temperature_label = quantity_label("T", "K")
    pressure_label = quantity_label("p", "Pa")
    energy_density_label = quantity_label("E_{\\mathrm{T}}", "J\\,m^{-3}")

    figure_1, axes_1 = plt.subplots(2, 3, figsize=(16.0, 9.5), constrained_layout=True)
    axes_1_flat = axes_1.ravel()
    axes_1_flat[0].plot(branch.temperature, branch.pressure, linewidth=1.5)
    configure_cartesian_axis(axes_1_flat[0], temperature_label, pressure_label)
    axes_1_flat[1].plot(branch.mass_density[0], branch.pressure, linewidth=1.5)
    configure_cartesian_axis(
        axes_1_flat[1],
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
        pressure_label,
    )
    axes_1_flat[2].plot(branch.temperature, branch.mass_density[0], linewidth=1.5)
    configure_cartesian_axis(
        axes_1_flat[2],
        temperature_label,
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
    )
    axes_1_flat[3].plot(branch.temperature, branch.total_energy_density, linewidth=1.5)
    configure_cartesian_axis(axes_1_flat[3], temperature_label, energy_density_label)
    axes_1_flat[4].plot(branch.mass_density[0], branch.total_energy_density, linewidth=1.5)
    configure_cartesian_axis(
        axes_1_flat[4],
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
        energy_density_label,
    )
    axes_1_flat[5].plot(branch.pressure, branch.total_energy_density, linewidth=1.5)
    configure_cartesian_axis(axes_1_flat[5], pressure_label, energy_density_label)

    figure_2, axes_2 = plt.subplots(2, 3, figsize=(16.0, 9.5), constrained_layout=True)
    axes_2_flat = axes_2.ravel()
    total_phase_mass = branch.mass_density[:2].sum(axis=0)

    axes_2_flat[0].plot(total_phase_mass, branch.total_energy_density, linewidth=1.5)
    configure_cartesian_axis(
        axes_2_flat[0],
        phase_sum_quantity_label("m", phase_1, phase_2, "kg\\,m^{-3}"),
        energy_density_label,
    )
    axes_2_flat[1].plot(branch.temperature, 1.0 / branch.rho[0], linewidth=1.5)
    configure_cartesian_axis(
        axes_2_flat[1],
        temperature_label,
        phase_quantity_label("v", phase_1, "m^3\\,kg^{-1}"),
    )
    axes_2_flat[2].plot(branch.temperature, 1.0 / branch.rho[1], linewidth=1.5)
    configure_cartesian_axis(
        axes_2_flat[2],
        temperature_label,
        phase_quantity_label("v", phase_2, "m^3\\,kg^{-1}"),
    )
    plot_pv_curves(
        axes_2_flat[3],
        branch.pressure,
        branch.rho[:2],
        [
            phase_legend_label(f"{phase_1} branch"),
            phase_legend_label(f"{phase_2} branch"),
        ],
    )
    configure_cartesian_axis(
        axes_2_flat[3],
        quantity_label("v", "m^3\\,kg^{-1}"),
        quantity_label("p", "Pa"),
        logx=True,
        logy=True,
    )
    axes_2_flat[3].legend()
    axes_2_flat[4].plot(branch.mass_density[0], branch.mass_density[1], linewidth=1.5)
    configure_cartesian_axis(
        axes_2_flat[4],
        phase_quantity_label("m", phase_1, "kg\\,m^{-3}"),
        phase_quantity_label("m", phase_2, "kg\\,m^{-3}"),
    )
    axes_2_flat[5].plot(branch.temperature, total_phase_mass, linewidth=1.5)
    configure_cartesian_axis(
        axes_2_flat[5],
        temperature_label,
        phase_sum_quantity_label("m", phase_1, phase_2, "kg\\,m^{-3}"),
    )

    return [figure_1, figure_2]


def annotate_pv_schematic(
    ax: plt.Axes,
    saturation_curve: SaturationCurve,
    *,
    thesis_mode: bool,
) -> None:
    pressure = saturation_curve.pressure
    liquid_volume = 1.0 / saturation_curve.rho[0]
    vapor_volume = 1.0 / saturation_curve.rho[1]
    point_count = len(pressure)
    liquid_index = max(1, int(0.82 * (point_count - 1)))
    vapor_index = max(1, int(0.56 * (point_count - 1)))

    annotation_font_size = 10 if thesis_mode else 13
    region_font_size = 10 if thesis_mode else 13
    liquid_arrow = {
        "arrowstyle": "->",
        "color": "0.22",
        "lw": 0.9,
        "shrinkA": 0,
        "shrinkB": 0,
        "mutation_scale": 10,
    }
    vapor_arrow = {
        "arrowstyle": "->",
        "color": "0.35",
        "lw": 0.9,
        "shrinkA": 0,
        "shrinkB": 0,
        "mutation_scale": 10,
    }

    ax.annotate(
        stacked_latex_lines("Saturated", "liquid line"),
        xy=(liquid_volume[liquid_index], pressure[liquid_index]),
        xycoords="data",
        xytext=(0.07, 0.82),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=annotation_font_size,
        arrowprops=liquid_arrow,
    )
    ax.annotate(
        stacked_latex_lines("Saturated", "vapor line"),
        xy=(vapor_volume[vapor_index], pressure[vapor_index]),
        xycoords="data",
        xytext=(0.50, 0.82),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=annotation_font_size,
        arrowprops=vapor_arrow,
    )

    ax.text(
        0.08,
        0.46,
        stacked_latex_lines("Subcooled", "liquid"),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=region_font_size,
    )
    ax.text(
        0.31,
        0.46,
        stacked_latex_lines("Mixture", "region"),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=region_font_size,
    )
    ax.text(
        0.78,
        0.46,
        stacked_latex_lines("Overheated", "vapor"),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=region_font_size,
    )

    ax.margins(x=0.05, y=0.08)


def plot_pv_overview_panel(
    ax: plt.Axes,
    preset: Preset,
    saturation_curve: SaturationCurve,
    branch: ThreePhaseBranch | None,
    *,
    thesis_mode: bool,
) -> None:
    saturation_labels = ["_nolegend_" for _ in preset.phases]
    liquid_volume = 1.0 / saturation_curve.rho[0]
    vapor_volume = 1.0 / saturation_curve.rho[1]
    saturation_color = "0.18"
    branch_color = "0.55"

    ax.fill_betweenx(
        saturation_curve.pressure,
        liquid_volume,
        vapor_volume,
        color="0.90",
        alpha=0.45,
        linewidth=0.0,
        zorder=0.2,
    )
    plot_pv_curves(
        ax,
        saturation_curve.pressure,
        saturation_curve.rho,
        saturation_labels,
        linestyle="-",
        linewidth=1.8,
        alpha=0.95,
        colors=[saturation_color, saturation_color],
    )

    if branch is not None:
        branch_labels = ["_nolegend_" for _ in preset.phases]
        plot_pv_curves(
            ax,
            branch.pressure,
            branch.rho,
            branch_labels,
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            colors=[branch_color, branch_color],
        )

    configure_cartesian_axis(
        ax,
        symbol_label("v"),
        symbol_label("p"),
        logx=True,
        logy=True,
    )
    ax.tick_params(
        which="both",
        labelbottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
    )
    ax.set_xlim(*PV_VOLUME_RANGE)
    ax.set_ylim(PV_PRESSURE_MIN, PV_PRESSURE_MAX)
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.14, linewidth=0.45)
    annotate_pv_schematic(ax, saturation_curve, thesis_mode=thesis_mode)


def plot_pv_metastable_paths(
    ax: plt.Axes,
    preset: Preset,
    saturation_curve: SaturationCurve,
) -> None:
    liquid_phase = preset.phases[0]
    metastable_color = "#8a1f28"
    path_color = "#1f4e79"
    metastable_pressure, metastable_volume = compute_isothermal_liquid_curve(
        liquid_phase,
        METASTABLE_TEMPERATURE,
        METASTABLE_PRESSURE_START,
        METASTABLE_PRESSURE_STOP,
    )
    pressure_sat, volume_sat = interpolate_saturation_point(
        saturation_curve,
        METASTABLE_TEMPERATURE,
        phase_index=0,
    )

    ax.loglog(
        metastable_volume,
        metastable_pressure,
        color=metastable_color,
        linestyle="-",
        linewidth=2.0,
        alpha=0.95,
        label=phase_legend_label("Metastable state line"),
        zorder=3.0,
    )
    ax.loglog(
        np.logspace(
            np.log10(volume_sat),
            np.log10(volume_sat * METASTABLE_SEGMENT2_VOLUME_SCALE),
            METASTABLE_SAMPLE_COUNT,
        ),
        np.full(METASTABLE_SAMPLE_COUNT, pressure_sat),
        color=path_color,
        linestyle="-",
        linewidth=2.3,
        alpha=0.95,
        label=phase_legend_label("Mixture line"),
        zorder=3.2,
    )
    ax.scatter(
        [volume_sat],
        [pressure_sat],
        color="black",
        s=28,
        zorder=3.5,
    )


def plot_pv_diagram_figure(
    preset: Preset,
    saturation_curve: SaturationCurve,
    branch: ThreePhaseBranch | None = None,
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
        branch,
        thesis_mode=thesis_mode,
    )
    plot_pv_metastable_paths(
        ax,
        preset,
        saturation_curve,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=14 if not thesis_mode else THESIS_TICK_FONT_SIZE,
    )

    return fig


def build_analysis(
    preset: Preset,
    *,
    thesis_mode: bool,
    include_auxiliary_figures: bool,
) -> tuple[
    list[tuple[str, plt.Figure]],
    SaturationCurve,
    ThreePhaseBranch | None,
    plt.Figure,
]:
    constants = build_saturation_constants(preset.phases[0], preset.phases[1])
    saturation_temperature_min = solve_saturation_temperature(
        SATURATION_PRESSURE_MIN,
        preset.phases[0],
        preset.phases[1],
        constants,
    )
    saturation_temperatures = np.linspace(
        saturation_temperature_min,
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
        branch,
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

        if branch is not None:
            branch_figures = plot_three_phase_branch_figures(preset, branch)
            figures.extend(
                [
                    (f"{preset.name}_three_phase_sweep.png", branch_figures[0]),
                    (f"{preset.name}_three_phase_auxiliary.png", branch_figures[1]),
                ]
            )

    return figures, saturation_curve, branch, pv_figure


def save_figures(
    figures: list[tuple[str, plt.Figure]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, figure in figures:
        figure.savefig(output_dir / filename, dpi=250, bbox_inches="tight")


def print_summary(
    preset: Preset,
    saturation_curve: SaturationCurve,
    branch: ThreePhaseBranch | None,
) -> None:
    print(f"Preset: {preset.name}")
    print(f"Phases: {len(preset.phases)}")
    print(f"Saturation points solved: {len(saturation_curve.temperature)}")
    print(
        "Saturation curve pressure range: "
        f"{saturation_curve.pressure.min():.6e} to {saturation_curve.pressure.max():.6e} Pa"
    )
    print(
        "Saturation curve total-energy range: "
        f"{saturation_curve.total_energy_density.min():.6e} to "
        f"{saturation_curve.total_energy_density.max():.6e} J/m^3"
    )
    if len(preset.phases) < 3:
        print("Three-phase branch: not applicable because the preset has fewer than 3 phases.")
    elif branch is None:
        print("Three-phase branch: no valid continuation points were found.")
    else:
        print(f"Three-phase branch points solved: {len(branch.m1)}")
        print(
            "Three-phase branch pressure range: "
            f"{branch.pressure.min():.6e} to {branch.pressure.max():.6e} Pa"
        )
        print(
            "Three-phase branch temperature range: "
            f"{branch.temperature.min():.6e} to {branch.temperature.max():.6e} K"
        )


def main(argv: list[str] | None = None) -> int:
    presets = build_presets()
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    preset = presets[args.preset]
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
