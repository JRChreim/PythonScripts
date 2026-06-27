"""Plot the threshold equivalent radius as a function of saturation pressure.

The inequality is

    R_eq(t) < 2 sigma / (p_infty(t) - p_v(T_infty(t))) * (1 / (3 k) - 1)

The vapor pressure is computed from the Gibbs-free-energy equality between the
liquid-water and vapor phases using the stiffened-gas equation of state. The
temperature sweep is clustered around the p_v = p_infty crossing so the
samples accumulate near the discontinuity, and the same samples are reused in a
comparison subplot with p_infty = -101325 Pa.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.plots.publication import (
    THESIS_LABEL_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    THESIS_TICK_FONT_SIZE,
    add_thesis_export_argument,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "artifacts" / "figures" / "pressure_threshold"
DEFAULT_OUTPUT = FIGURE_DIR / "req_pv_radius_threshold.png"
THESIS_EXPORT_STEM = "PvRadiusThreshold"

# FLUID PROPERTIES - 0 for liquid water, 1 for vapor, 2 for air
pi = np.array([1.00e9, 0.0e0, 0.0e0], dtype=float)
qv = np.array([-1167e3, 203e4, 0.0e0], dtype=float)
qvp = np.array([0.0e0, -234e2, 0.0e0], dtype=float)
cv = np.array([1816.0, 1040.0, 717.5], dtype=float)
cp = np.array([4267.0, 1487.0, 1006.0], dtype=float)
gama = cp / cv

LIQUID_INDEX = 0
VAPOR_INDEX = 1
AIR_INDEX = 2

DEFAULT_T_START = 273.16
DEFAULT_T_STOP = 393.16
DEFAULT_T_STEP = 5.0
DEFAULT_SIGMA = 0.07275
DEFAULT_P_INFINITY = 4848.0
DEFAULT_INITIAL_PV = 101325.0
DEFAULT_GIBBS_TOLERANCE = 1.0e-9
DEFAULT_GIBBS_OMEGA = 1.0
DEFAULT_GIBBS_P_MIN = 1.0
DEFAULT_GIBBS_MAX_ITERATIONS = 100

PASCAL_TO_KILOPASCAL = 1.0e-3
METER_TO_MICROMETER = 1.0e6
THESIS_FIGURE_SIZE = thesis_figure_size(1.0)


def build_temperature_values(
    t_start: float,
    t_stop: float,
    t_step: float,
    *,
    pivot_temperature: float,
) -> np.ndarray:
    """Return an inclusive sweep clustered around a pivot temperature.

    The `t_step` input is treated as the nominal spacing used to determine the
    number of samples across the interval, but the samples are redistributed
    with a half-cosine profile on each side of the pivot so they accumulate
    near the p_v = p_infty discontinuity.
    """

    if t_start <= 0.0:
        raise ValueError(f"t_start must be positive; got {t_start:g}.")
    if t_stop <= t_start:
        raise ValueError(
            "t_stop must be greater than t_start. "
            f"Got t_start={t_start:g}, t_stop={t_stop:g}."
        )
    if t_step <= 0.0:
        raise ValueError(f"t_step must be positive; got {t_step:g}.")

    interval_count = (t_stop - t_start) / t_step
    rounded_interval_count = int(round(interval_count))
    if not np.isclose(interval_count, rounded_interval_count, atol=1.0e-10):
        raise ValueError(
            "The temperature range must be an integer multiple of t_step so "
            "the clustered sweep can be built with an integer number of "
            "intervals. "
            f"Got t_start={t_start:g}, t_stop={t_stop:g}, t_step={t_step:g}."
        )

    if pivot_temperature <= t_start or pivot_temperature >= t_stop:
        raise ValueError(
            "pivot_temperature must lie strictly inside the temperature range."
        )

    sample_count = rounded_interval_count + 1
    left_count = sample_count // 2
    right_count = sample_count - left_count

    left_phase = np.linspace(0.0, np.pi, left_count + 1, endpoint=True)[:-1]
    right_phase = np.linspace(0.0, np.pi, right_count + 1, endpoint=True)[1:]

    left_temperatures = t_start + (
        0.5
        * (pivot_temperature - t_start)
        * (1.0 - np.cos(left_phase))
    )
    right_temperatures = pivot_temperature + (
        0.5
        * (t_stop - pivot_temperature)
        * (1.0 - np.cos(right_phase))
    )

    temperature_values = np.concatenate([left_temperatures, right_temperatures])
    return np.asarray(temperature_values, dtype=float)


def solve_gibbs_equilibrium_pressure(
    t_infty: float,
    *,
    liquid_index: int = LIQUID_INDEX,
    vapor_index: int = VAPOR_INDEX,
    initial_pressure: float = DEFAULT_INITIAL_PV,
    omega: float = DEFAULT_GIBBS_OMEGA,
    p_min: float = DEFAULT_GIBBS_P_MIN,
    tolerance: float = DEFAULT_GIBBS_TOLERANCE,
    max_iterations: int = DEFAULT_GIBBS_MAX_ITERATIONS,
) -> float:
    """Solve the liquid-vapor Gibbs equality for the saturation pressure."""

    if t_infty <= 0.0:
        raise ValueError(f"t_infty must be positive; got {t_infty:g}.")
    if initial_pressure <= 0.0:
        raise ValueError(
            f"initial_pressure must be positive; got {initial_pressure:g}."
        )
    if p_min <= 0.0:
        raise ValueError(f"p_min must be positive; got {p_min:g}.")
    if omega <= 0.0:
        raise ValueError(f"omega must be positive; got {omega:g}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive; got {tolerance:g}.")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1; got {max_iterations}.")

    pv = float(initial_pressure)
    f_sat_prop = np.inf
    ns = 0

    while abs(f_sat_prop) > tolerance or ns == 0:
        if ns >= max_iterations:
            raise RuntimeError(
                "The Gibbs-equilibrium pressure solve did not converge within "
                f"{max_iterations} iterations."
            )

        ns += 1

        f_sat_prop = (
            t_infty
            * (
                (
                    cv[liquid_index] * gama[liquid_index]
                    - cv[vapor_index] * gama[vapor_index]
                )
                * (1.0 - np.log(t_infty))
                - (qvp[liquid_index] - qvp[vapor_index])
                + cv[liquid_index]
                * (gama[liquid_index] - 1.0)
                * np.log(pv + pi[liquid_index])
                - cv[vapor_index]
                * (gama[vapor_index] - 1.0)
                * np.log(pv + pi[vapor_index])
            )
            + qv[liquid_index]
            - qv[vapor_index]
        )

        dfdp = t_infty * (
            cv[liquid_index] * (gama[liquid_index] - 1.0) / (pv + pi[liquid_index])
            - cv[vapor_index] * (gama[vapor_index] - 1.0) / (pv + pi[vapor_index])
        )

        pv = max(pv - omega * f_sat_prop / dfdp, p_min)

    return pv


def solve_saturation_pressure_sweep(
    temperature_values: np.ndarray,
    *,
    initial_pressure: float = DEFAULT_INITIAL_PV,
) -> np.ndarray:
    """Solve the saturation pressure at each temperature in the sweep."""

    if temperature_values.ndim != 1:
        raise ValueError("temperature_values must be a one-dimensional array.")
    if temperature_values.size == 0:
        raise ValueError("temperature_values must contain at least one entry.")

    saturation_pressures = np.empty_like(temperature_values, dtype=float)
    guess = float(initial_pressure)

    for index, temperature in enumerate(temperature_values):
        guess = solve_gibbs_equilibrium_pressure(
            float(temperature),
            initial_pressure=guess,
        )
        saturation_pressures[index] = guess

    return saturation_pressures


def solve_temperature_for_pressure_target(
    target_pressure: float,
    t_start: float,
    t_stop: float,
    *,
    initial_pressure: float = DEFAULT_INITIAL_PV,
    tolerance: float = DEFAULT_GIBBS_TOLERANCE,
    max_iterations: int = DEFAULT_GIBBS_MAX_ITERATIONS,
) -> float:
    """Solve for the temperature where p_sat(T) equals a target pressure."""

    if target_pressure <= 0.0:
        raise ValueError(f"target_pressure must be positive; got {target_pressure:g}.")
    if t_stop <= t_start:
        raise ValueError(
            "t_stop must be greater than t_start. "
            f"Got t_start={t_start:g}, t_stop={t_stop:g}."
        )

    lower_temperature = float(t_start)
    upper_temperature = float(t_stop)

    lower_pressure = solve_gibbs_equilibrium_pressure(
        lower_temperature,
        initial_pressure=initial_pressure,
    )
    upper_pressure = solve_gibbs_equilibrium_pressure(
        upper_temperature,
        initial_pressure=lower_pressure,
    )

    lower_residual = lower_pressure - target_pressure
    upper_residual = upper_pressure - target_pressure
    if lower_residual == 0.0:
        return lower_temperature
    if upper_residual == 0.0:
        return upper_temperature
    if lower_residual * upper_residual > 0.0:
        raise ValueError(
            "The target pressure is not bracketed by the requested temperature "
            "interval."
        )

    guess = upper_pressure
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower_temperature + upper_temperature)
        midpoint_pressure = solve_gibbs_equilibrium_pressure(
            midpoint,
            initial_pressure=guess,
        )
        midpoint_residual = midpoint_pressure - target_pressure
        if abs(midpoint_residual) <= tolerance:
            return midpoint

        if lower_residual * midpoint_residual < 0.0:
            upper_temperature = midpoint
            upper_pressure = midpoint_pressure
            upper_residual = midpoint_residual
        else:
            lower_temperature = midpoint
            lower_pressure = midpoint_pressure
            lower_residual = midpoint_residual

        guess = midpoint_pressure

    return 0.5 * (lower_temperature + upper_temperature)


def compute_radius_threshold(
    pv: np.ndarray | float,
    *,
    p_infty: float,
    sigma: float,
    k: float,
) -> np.ndarray:
    """Return the threshold radius implied by the inequality.

    The sign is preserved so the plot can show the nonphysical branch when
    p_infty > p_v.
    """

    pv_array = np.asarray(pv, dtype=float)
    delta_p = p_infty - pv_array
    if np.any(np.isclose(delta_p, 0.0)):
        raise ValueError("pv must not be equal to p_infty.")
    return (2.0 * sigma / delta_p) * ((1.0 / (3.0 * k)) - 1.0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the threshold equivalent radius versus the saturation "
            "pressure p_v."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--t-start",
        type=float,
        default=DEFAULT_T_START,
        help="Lower bound of the temperature sweep [K].",
    )
    parser.add_argument(
        "--t-stop",
        type=float,
        default=DEFAULT_T_STOP,
        help="Upper bound of the temperature sweep [K].",
    )
    parser.add_argument(
        "--t-step",
        type=float,
        default=DEFAULT_T_STEP,
        help=(
            "Nominal temperature increment used to set the number of samples "
            "for the sinusoidal sweep [K]."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="Surface tension sigma [N/m].",
    )
    parser.add_argument(
        "--p-infty",
        type=float,
        default=DEFAULT_P_INFINITY,
        help=(
            "Ambient pressure p_infty [Pa] used in the radius-threshold "
            "relation."
        ),
    )
    parser.add_argument(
        "--initial-pv",
        type=float,
        default=DEFAULT_INITIAL_PV,
        help=(
            "Initial pressure guess [Pa] used to seed the first Gibbs solve in "
            "the temperature sweep."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Optional path for the saved PNG figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figure without opening an interactive window.",
    )
    return parser


def _plot_threshold_panel(
    axis,
    *,
    temperature_values: np.ndarray,
    saturation_pressures: np.ndarray,
    p_infty: float,
    k: float,
    sigma: float,
    temperature_norm,
    cmap,
    show_xlabel: bool,
):
    if p_infty == 0.0:
        raise ValueError("p_infty must be non-zero for the threshold plot.")

    pv_values_kpa = saturation_pressures * PASCAL_TO_KILOPASCAL
    p_infty_kpa = p_infty * PASCAL_TO_KILOPASCAL
    req_threshold_um = (
        compute_radius_threshold(
            saturation_pressures,
            p_infty=p_infty,
            sigma=sigma,
            k=k,
        )
        * METER_TO_MICROMETER
    )
    pv_zero_req_um = (
        compute_radius_threshold(
            0.0,
            p_infty=p_infty,
            sigma=sigma,
            k=k,
        )
        * METER_TO_MICROMETER
    )
    pv_curve_kpa = np.linspace(
        0.0,
        max(float(pv_values_kpa.max()) * 1.06, abs(p_infty_kpa) * 1.6),
        800,
    )
    req_curve_um = np.full_like(pv_curve_kpa, np.nan, dtype=float)
    asymptote_gap_pa = max(1.0, 0.005 * abs(p_infty))
    curve_mask = np.abs(pv_curve_kpa * 1.0e3 - p_infty) > asymptote_gap_pa
    req_curve_um[curve_mask] = (
        compute_radius_threshold(
            pv_curve_kpa[curve_mask] * 1.0e3,
            p_infty=p_infty,
            sigma=sigma,
            k=k,
        )
        * METER_TO_MICROMETER
    )

    threshold_line, = axis.plot(
        pv_curve_kpa,
        req_curve_um,
        color="0.08",
        linewidth=2.4,
        label=r"$R_{eq,\mathrm{th}}$",
    )

    sample_scatter = axis.scatter(
        pv_values_kpa,
        req_threshold_um,
        c=temperature_values,
        cmap=cmap,
        norm=temperature_norm,
        s=34,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    pv_zero_point = axis.scatter(
        [0.0],
        [pv_zero_req_um],
        color="crimson",
        edgecolors="white",
        linewidths=0.6,
        marker="o",
        s=60,
        label="no vapor",
        zorder=4,
    )

    axis.axhline(0.0, color="0.6", linestyle="--", linewidth=1.0)
    axis.axvline(
        p_infty_kpa,
        color="0.55",
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlim(
        0.0,
        max(float(pv_curve_kpa.max()), float(saturation_pressures.max() * PASCAL_TO_KILOPASCAL) * 1.06),
    )

    y_min = float(np.nanmin(np.concatenate([req_curve_um, [pv_zero_req_um, 0.0]])))
    y_max = float(np.nanmax(np.concatenate([req_curve_um, [pv_zero_req_um, 0.0]])))
    y_margin = max(0.05 * abs(y_min), 0.1 * max(1.0, abs(y_max - y_min)))
    axis.set_ylim(y_min - y_margin, y_max + y_margin)

    axis.grid(True, which="both", alpha=0.28)
    axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    axis.set_ylabel(
        r"$R_{eq}\ [\mu\mathrm{m}]$",
        fontsize=THESIS_LABEL_FONT_SIZE,
    )
    if show_xlabel:
        axis.set_xlabel(r"$p_v\ [\mathrm{kPa}]$", fontsize=THESIS_LABEL_FONT_SIZE)
    axis.set_title(
        rf"$p_\infty = {p_infty_kpa:+.3f}\ \mathrm{{kPa}}$",
        fontsize=THESIS_LABEL_FONT_SIZE,
    )

    axis.legend(
        handles=[threshold_line, pv_zero_point],
        loc="lower right",
        fontsize=THESIS_TICK_FONT_SIZE,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.6",
    )

    return sample_scatter


def build_threshold_plot(
    *,
    temperature_values: np.ndarray,
    saturation_pressures: np.ndarray,
    p_infty: float,
    k: float,
    sigma: float,
):
    if k <= 0.0:
        raise ValueError(f"k must be positive; got {k:g}.")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative; got {sigma:g}.")
    if p_infty <= 0.0:
        raise ValueError(f"p_infty must be positive; got {p_infty:g}.")
    if temperature_values.ndim != 1:
        raise ValueError("temperature_values must be one-dimensional.")
    if saturation_pressures.ndim != 1:
        raise ValueError("saturation_pressures must be one-dimensional.")
    if temperature_values.size != saturation_pressures.size:
        raise ValueError(
            "temperature_values and saturation_pressures must have the same length."
        )
    if np.any(np.isclose(p_infty - saturation_pressures, 0.0)):
        raise ValueError("saturation_pressures must not be equal to p_infty.")

    apply_thesis_style()

    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=THESIS_FIGURE_SIZE,
        constrained_layout=True,
    )
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    temperature_norm = colors.Normalize(
        vmin=float(temperature_values.min()),
        vmax=float(temperature_values.max()),
    )
    cmap = plt.get_cmap("viridis")
    top_scatter = _plot_threshold_panel(
        axes[0],
        temperature_values=temperature_values,
        saturation_pressures=saturation_pressures,
        p_infty=p_infty,
        k=k,
        sigma=sigma,
        temperature_norm=temperature_norm,
        cmap=cmap,
        show_xlabel=False,
    )
    _plot_threshold_panel(
        axes[1],
        temperature_values=temperature_values,
        saturation_pressures=saturation_pressures,
        p_infty=-p_infty,
        k=k,
        sigma=sigma,
        temperature_norm=temperature_norm,
        cmap=cmap,
        show_xlabel=True,
    )

    colorbar = figure.colorbar(top_scatter, ax=axes, pad=0.03)
    colorbar.ax.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
    colorbar.set_label(r"$T_\infty\ [\mathrm{K}]$", fontsize=THESIS_LABEL_FONT_SIZE)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    return figure, axes


def main(argv=None):
    args = build_argument_parser().parse_args(argv)

    k = float(gama[AIR_INDEX])
    pivot_temperature = solve_temperature_for_pressure_target(
        args.p_infty,
        args.t_start,
        args.t_stop,
        initial_pressure=args.initial_pv,
    )
    pv_zero_req_um = (
        compute_radius_threshold(
            0.0,
            p_infty=args.p_infty,
            sigma=args.sigma,
            k=k,
        )
        * METER_TO_MICROMETER
    )
    temperature_values = build_temperature_values(
        args.t_start,
        args.t_stop,
        args.t_step,
        pivot_temperature=pivot_temperature,
    )
    saturation_pressures = solve_saturation_pressure_sweep(
        temperature_values,
        initial_pressure=args.initial_pv,
    )

    print(f"Air heat-capacity ratio: k = gama[2] = {k:.8f}")
    print(f"Ambient pressure: p_infty = {args.p_infty:.6f} Pa")
    print(f"Comparison panel uses p_infty = {-args.p_infty:.6f} Pa")
    print(
        f"Pivot temperature for p_v = p_infty: T_* = {pivot_temperature:.6f} K"
    )
    print("Sinusoidally clustered temperature sweep and saturation pressures:")
    for temperature, saturation_pressure in zip(
        temperature_values,
        saturation_pressures,
    ):
        print(
            f"  T_infty = {temperature:.2f} K -> "
            f"p_sat = {saturation_pressure:.6f} Pa "
            f"({saturation_pressure * PASCAL_TO_KILOPASCAL:.6f} kPa)"
        )
    print(
        f"Reference case: no vapor -> R_eq = {pv_zero_req_um:.6f} um "
        "(evaluated in the plot)."
    )

    figure, _ = build_threshold_plot(
        temperature_values=temperature_values,
        saturation_pressures=saturation_pressures,
        p_infty=args.p_infty,
        k=k,
        sigma=args.sigma,
    )

    output_path = None
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)

    thesis_path = save_thesis_figure_from_args(
        figure,
        args,
        stem=THESIS_EXPORT_STEM,
        output_dir=FIGURE_DIR,
        dpi=600,
    )

    if output_path is not None:
        print(f"Saved figure to {output_path}")
    if thesis_path is not None:
        print(f"Saved thesis figure to {thesis_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
