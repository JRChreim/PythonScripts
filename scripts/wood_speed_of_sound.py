"""Compute mixture speed of sound with Wood's expression.

The phase densities and phase speeds of sound are evaluated with the
stiffened-gas EoS at a fixed pressure and temperature:

    rho_k = (p + p_inf,k) / ((gamma_k - 1) cv_k T)
    c_k^2 = gamma_k (p + p_inf,k) / rho_k

Wood's expression is then evaluated for alpha_g in [0, 1], with
alpha_l + alpha_g = 1:

    1 / (rho c^2) = alpha_l / (rho_l c_l^2)
                    + alpha_g / (rho_g c_g^2)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

if "--no-show" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.stiffened_gas_eos import build_presets, configure_cartesian_axis, quantity_label
from src.plots.publication import (
    THESIS_LAYOUT_PADS,
    add_thesis_export_argument,
    apply_publication_style,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)
from src.thermo.stiffened_gas import PhaseParameters


DEFAULT_PRESET = "water_air"
DEFAULT_TEMPERATURE = 298.15
DEFAULT_PRESSURE = 101_325.0
DEFAULT_SAMPLE_COUNT = 1_001
DEFAULT_ALPHA_SPACING = "cosine"
DEFAULT_ALPHA_MIN = 0.0
DEFAULT_ALPHA_MAX = 1.0
DEFAULT_X_SCALE = "endpoint-sine"
M_PER_S_TO_MM_PER_MICROSECOND = 1.0e-3
THESIS_EXPORT_STEM = "WoodSpeedOfSound"
PUBLICATION_FIGURE_SIZE = (8.0, 5.0)
THESIS_FIGURE_SIZE = thesis_figure_size(0.62)
ALPHA_AXIS_TICKS = np.array(
    [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
    dtype=float,
)
ALPHA_AXIS_TICK_LABELS = (
    "0",
    "0.01",
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "0.75",
    "0.9",
    "0.95",
    "0.99",
    "1",
)


def phase_density(phase: PhaseParameters, pressure: float, temperature: float) -> float:
    """Return rho from the stiffened-gas EoS."""
    return float(
        (pressure + phase.p_inf)
        / ((phase.gamma - 1.0) * phase.cv * temperature)
    )


def phase_speed_of_sound(
    phase: PhaseParameters,
    density: float,
    pressure: float,
) -> float:
    """Return c from c^2 = gamma (p + p_inf) / rho."""
    c_squared = phase.gamma * (pressure + phase.p_inf) / density
    if c_squared <= 0.0:
        raise ValueError(
            f"Non-positive c^2 for phase {phase.name!r}: {c_squared:g}."
        )
    return float(np.sqrt(c_squared))


def wood_speed_of_sound(
    alpha_g: np.ndarray,
    rho_l: float,
    rho_g: float,
    c_l: float,
    c_g: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mixture density and Wood speed of sound for gas volume fractions."""
    alpha_l = 1.0 - alpha_g
    rho = alpha_l * rho_l + alpha_g * rho_g
    wood_compressibility = (
        alpha_l / (rho_l * c_l**2)
        + alpha_g / (rho_g * c_g**2)
    )
    c = np.sqrt(1.0 / (rho * wood_compressibility))
    return rho, c


def build_alpha_g_samples(
    sample_count: int,
    spacing: str,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    base_coordinate = np.linspace(0.0, 1.0, sample_count)
    if spacing == "uniform":
        normalized_alpha = base_coordinate
    elif spacing == "cosine":
        normalized_alpha = 0.5 * (1.0 - np.cos(np.pi * base_coordinate))
    else:
        raise ValueError(f"Unknown alpha spacing: {spacing!r}.")
    return alpha_min + (alpha_max - alpha_min) * normalized_alpha


def configure_alpha_axis(axis: plt.Axes, alpha_g: np.ndarray, x_scale: str) -> np.ndarray:
    if x_scale == "endpoint-sine":
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks(alpha_axis_coordinate(ALPHA_AXIS_TICKS))
        axis.set_xticklabels(ALPHA_AXIS_TICK_LABELS)
        return alpha_axis_coordinate(alpha_g)
    if x_scale == "linear":
        axis.set_xlim(float(alpha_g.min()), float(alpha_g.max()))
        return alpha_g
    if x_scale == "log":
        positive_alpha = alpha_g[alpha_g > 0.0]
        if positive_alpha.size == 0:
            raise ValueError("A logarithmic x-axis requires at least one positive alpha_g value.")
        axis.set_xscale("log")
        axis.set_xlim(float(positive_alpha.min()), float(alpha_g.max()))
        return alpha_g
    raise ValueError(f"Unknown x-axis scale: {x_scale!r}.")


def alpha_axis_coordinate(alpha_g: np.ndarray) -> np.ndarray:
    """Map alpha_g to an endpoint-amplified sinusoidal plotting coordinate."""
    clipped_alpha = np.clip(alpha_g, 0.0, 1.0)
    return 0.5 + np.arcsin(2.0 * clipped_alpha - 1.0) / np.pi


def resolve_liquid_and_gas(
    phases: tuple[PhaseParameters, ...],
) -> tuple[PhaseParameters, PhaseParameters]:
    phases_by_name = {phase.name.lower(): phase for phase in phases}
    liquid = phases_by_name.get("liquid", phases[0])
    gas = phases_by_name.get("air")
    if gas is None:
        gas = phases_by_name.get("gas", phases[1] if len(phases) > 1 else phases[0])
    if gas is liquid:
        raise ValueError("Liquid and gas phases resolved to the same phase.")
    return liquid, gas


def build_argument_parser() -> argparse.ArgumentParser:
    presets = build_presets()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Wood's mixture speed of sound for a two-phase "
            "stiffened-gas preset."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--preset",
        choices=sorted(presets.keys()),
        default=DEFAULT_PRESET,
        help="Two-phase preset to use for liquid/gas parameters.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature T [K].",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=DEFAULT_PRESSURE,
        help="Pressure p [Pa].",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of alpha_g samples between 0 and 1.",
    )
    parser.add_argument(
        "--alpha-spacing",
        choices=("cosine", "uniform"),
        default=DEFAULT_ALPHA_SPACING,
        help=(
            "Spacing for alpha_g samples. 'cosine' clusters samples near "
            "alpha_g = 0 and alpha_g = 1."
        ),
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=DEFAULT_ALPHA_MIN,
        help="Minimum gas volume fraction alpha_g.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=DEFAULT_ALPHA_MAX,
        help="Maximum gas volume fraction alpha_g.",
    )
    parser.add_argument(
        "--x-scale",
        choices=("endpoint-sine", "linear", "log"),
        default=DEFAULT_X_SCALE,
        help=(
            "X-axis scaling for the plot. 'endpoint-sine' expands alpha_g "
            "near 0 and 1; 'log' plots positive alpha_g values on a log axis."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for alpha_g, alpha_l, rho, and c.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Optional figure path for the Wood speed-of-sound curve.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plot without opening an interactive matplotlib window.",
    )
    return parser


def write_csv(
    output_path: Path,
    alpha_g: np.ndarray,
    rho: np.ndarray,
    c: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha_g", "alpha_l", "rho_wood", "c_wood_mm_per_microsecond"])
        for alpha_g_value, rho_value, c_value in zip(alpha_g, rho, c, strict=True):
            writer.writerow(
                [
                    f"{alpha_g_value:.16e}",
                    f"{1.0 - alpha_g_value:.16e}",
                    f"{rho_value:.16e}",
                    f"{c_value * M_PER_S_TO_MM_PER_MICROSECOND:.16e}",
                ]
            )


def plot_wood_speed(
    alpha_g: np.ndarray,
    c: np.ndarray,
    *,
    thesis_mode: bool,
    x_scale: str,
) -> plt.Figure:
    figure, axis = plt.subplots(
        figsize=THESIS_FIGURE_SIZE if thesis_mode else PUBLICATION_FIGURE_SIZE,
        constrained_layout=True,
    )
    if thesis_mode:
        figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    c_mm_per_microsecond = c * M_PER_S_TO_MM_PER_MICROSECOND
    plot_alpha_g = alpha_g
    plot_c = c_mm_per_microsecond
    if x_scale == "log":
        positive_mask = alpha_g > 0.0
        plot_alpha_g = alpha_g[positive_mask]
        plot_c = c_mm_per_microsecond[positive_mask]

    plot_x = configure_alpha_axis(axis, alpha_g, x_scale)
    if x_scale == "log":
        plot_x = plot_alpha_g

    axis.plot(plot_x, plot_c, color="black", linewidth=1.8)
    configure_cartesian_axis(
        axis,
        r"$\alpha_g$",
        quantity_label("c", "mm\\,\\mu s^{-1}"),
    )
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.samples < 2:
        parser.error("--samples must be at least 2.")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive.")
    if args.alpha_min < 0.0:
        parser.error("--alpha-min must be non-negative.")
    if args.alpha_max <= args.alpha_min:
        parser.error("--alpha-max must be greater than --alpha-min.")
    if args.alpha_max > 1.0:
        parser.error("--alpha-max must not exceed 1.")
    thesis_mode = bool(args.to_thesis)
    if thesis_mode:
        apply_thesis_style()
    else:
        apply_publication_style()

    preset = build_presets()[args.preset]
    if len(preset.phases) < 2:
        parser.error(f"Preset {preset.name!r} must contain at least two phases.")

    liquid, gas = resolve_liquid_and_gas(preset.phases)
    rho_l = phase_density(liquid, args.pressure, args.temperature)
    rho_g = phase_density(gas, args.pressure, args.temperature)
    c_l = phase_speed_of_sound(liquid, rho_l, args.pressure)
    c_g = phase_speed_of_sound(gas, rho_g, args.pressure)

    alpha_g = build_alpha_g_samples(
        args.samples,
        args.alpha_spacing,
        args.alpha_min,
        args.alpha_max,
    )
    rho, c = wood_speed_of_sound(alpha_g, rho_l, rho_g, c_l, c_g)

    min_index = int(np.argmin(c))
    c_l_mm_per_microsecond = c_l * M_PER_S_TO_MM_PER_MICROSECOND
    c_g_mm_per_microsecond = c_g * M_PER_S_TO_MM_PER_MICROSECOND
    c_min_mm_per_microsecond = c[min_index] * M_PER_S_TO_MM_PER_MICROSECOND
    print(f"Preset: {preset.name}")
    print(f"Liquid phase: {liquid.name}")
    print(f"Gas phase: {gas.name}")
    print(f"T = {args.temperature:.6f} K")
    print(f"p = {args.pressure:.6f} Pa")
    print(f"rho_l = {rho_l:.12e} kg/m^3")
    print(f"rho_g = {rho_g:.12e} kg/m^3")
    print(f"c_l = {c_l_mm_per_microsecond:.12e} mm/us")
    print(f"c_g = {c_g_mm_per_microsecond:.12e} mm/us")
    print(f"alpha_g range = [{args.alpha_min:.12e}, {args.alpha_max:.12e}]")
    print(f"alpha_g spacing = {args.alpha_spacing}")
    print(f"x-axis scale = {args.x_scale}")
    if args.x_scale == "log" and args.alpha_min == 0.0:
        print("alpha_g = 0 is computed but omitted from the logarithmic x-axis plot.")
    print(
        "Minimum Wood c = "
        f"{c_min_mm_per_microsecond:.12e} mm/us at alpha_g = {alpha_g[min_index]:.12e}"
    )

    if args.output_csv is not None:
        write_csv(args.output_csv, alpha_g, rho, c)
        print(f"Wrote CSV to {args.output_csv}")

    figure = plot_wood_speed(alpha_g, c, thesis_mode=thesis_mode, x_scale=args.x_scale)
    if args.output_figure is not None:
        args.output_figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output_figure, dpi=250, bbox_inches="tight")
        print(f"Wrote figure to {args.output_figure}")

    if args.to_thesis:
        thesis_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        if thesis_path is not None:
            print(f"Thesis figure written to {thesis_path}")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
